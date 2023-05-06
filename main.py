import os
import copy
import numpy as np
import torch
import torch.nn.functional as F

from dgl.data.ppi import PPIDataset
import dgl
import argparse
from gat import GAT
from utils import evaluate, collate, train_epoch, train_epoch_s, FullLoss, TeacherLoss, MILoss
from utils import get_data_loader, save_checkpoint, load_checkpoint
from utils import generate_label
from auxilary_loss import gen_fit_loss, optimizing, gen_mi_loss, loss_fn_kd, gen_att_loss
from auxilary_model import collect_model
from auxilary_optimizer import block_optimizer
from plot_utils import loss_logger, parameters
import time
import matplotlib.pyplot as plt
import collections
import random
from tqdm import tqdm
from node2vec import Node2Vec
import networkx as nx
from torch.utils.data import DataLoader

torch.set_num_threads(1)

def train_student(args, auxiliary_model, data, device):
    best_score = 0
    best_loss = 1000.0

    # data split
    train_dataloader, valid_dataloader, test_dataloader = data
    
    # models loading
    t_model = auxiliary_model['t_model']['model']
    s_model = auxiliary_model['s_model']['model']
    
    # optimizer loading
    t_optimizer = auxiliary_model['t_model']['optimizer']
    s_optimizer = auxiliary_model['s_model']['optimizer']
    # loss functions
    bce_loss = torch.nn.BCEWithLogitsLoss()
    full_loss = FullLoss()
    teacher_loss = TeacherLoss()
    mi_loss = MILoss()

    if args.mode == 'full':
        loss_fcn = full_loss
    elif args.mode == 'teacher':
        loss_fcn = teacher_loss
    elif args.mode == 'mi':
        loss_fcn = mi_loss
    # for batch, batch_data in enumerate(train_dataloader):
    #     subgraph, feats, labels = batch_data
    #     feats, labels = feats.to(device), labels.to(device)
    #     for n in range(subgraph.number_of_nodes()):
    #         new_feats = torch.cat((existing_feats, deepwalk_model.wv[n]), dim=0) # additional_feats is the new features you want to concatenate
    """ new_batches = []
    for batch, batch_data in enumerate(train_dataloader):
        subgraph, feats, labels = batch_data
        # feats, labels = feats.to(device), labels.to(device)
        print(feats.shape)
        batch_feats=torch.zeros((feats.shape[0],50))
        
        for i in subgraph.nodes():
            node_feats = feats[i]
            new_feats = feats[i]
            # print(i)
            if i==0:
                print(next(iter(all_nodes)),type(next(iter(all_nodes))),type(str(i.item())))
            # print(i,all_nodes)
            if str(i.item()) in all_nodes:
                # new_feats = torch.cat((node_feats.cpu(), torch.from_numpy(deepwalk_model.wv[i.item()]).double()), dim=0)
                new_feats = torch.from_numpy(deepwalk_model.wv[i.item()]).double()
                # print("dewfweffre",len(new_feats))
            else:
                # new_feats = torch.cat((node_feats.cpu(), torch.zeros(deepwalk_model.wv[0].shape).double()), dim=0)
                new_feats = feats[i]
            #feats[i] = new_feats
            batch_feats[i]=new_feats
        print(i)
        new_batch_data = (subgraph, batch_feats, labels)
        new_batches.append(new_batch_data)
    new_dataloader = DataLoader(new_batches, batch_size=args.batch_size, collate_fn = collate, num_workers=train_dataloader.num_workers,shuffle = True) """
    
    # new_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=4, shuffle=True)


    for epoch in range(args.s_epochs):
        print(epoch)
        t0 = time.time()
        if (epoch >= args.tofull) or (args.mode == 'mi' and epoch > args.warmup_epoch):
            args.mode = 'full'
            loss_fcn = full_loss
        loss_data, lsp_loss_data = train_epoch_s(train_dataloader, s_model, t_model, loss_fcn, s_optimizer, device, args)
        print(f"Epoch {epoch:05d} | Loss: {loss_data:.4f} | Mi: {lsp_loss_data:.4f} | Time: {time.time()-t0:.4f}s")
        
        if epoch % 10 == 0:
            score, _ = evaluate(valid_dataloader, s_model, bce_loss, device)
            if score > best_score or loss_data < best_loss:
                best_score = score
                best_loss = loss_data
                test_score, _ = evaluate(test_dataloader, s_model, bce_loss, device)
    print(f"f1 score on testset: {test_score:.4f}")


def train_teacher(args, model, data, device):
    train_dataloader, valid_dataloader, test_dataloader = data
    loss_fcn = torch.nn.BCEWithLogitsLoss()    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in tqdm(range(args.t_epochs)):
        loss_data = train_epoch(train_dataloader, model, loss_fcn, optimizer, device)
        if epoch % 10 == 0:
            val_score, val_loss = evaluate(valid_dataloader, model, loss_fcn, device)
            train_score, train_loss = evaluate(train_dataloader, model, loss_fcn, device)

    score, loss = evaluate(test_dataloader, model, loss_fcn, device)
    print(f"F1-Score on testset:  {score:.4f}")
    

def load_teacher_and_stats(model_dict, args, data, device):
    t_model = model_dict['t_model']['model']

    # load or train the teacher
    if os.path.isfile("./models/t_model.pt"):
        load_checkpoint(t_model, "./models/t_model.pt", device)
    else:
        train_teacher(args, t_model, data, device)
        save_checkpoint(t_model, "./models/t_model.pt")
    
    print(f"number of parameter for teacher model: {parameters(t_model)}")
    print(f"number of parameter for student model: {parameters(model_dict['s_model']['model'])}")

    # verify the teacher model
    loss_fcn = torch.nn.BCEWithLogitsLoss()
    train_dataloader, _, test_dataloader = data
    test_score, _ = evaluate(test_dataloader, t_model, loss_fcn, device)
    train_score, _ = evaluate(train_dataloader, t_model, loss_fcn, device)
    print(f"test acc of teacher:", test_score)
    print(f"train acc of teacher:", train_score)
    print("\n\n")
    return


def main(args):
    device = torch.device("cpu") if args.gpu<0 else torch.device("cuda:" + str(args.gpu))
    data, data_info = get_data_loader(args)
    model_dict = collect_model(args, data_info)
    """ ppi_dataset = PPIDataset(mode='train')
    ppi_graph, _ = ppi_dataset[0]
    nx_graph = nx.DiGraph()
    nx_graph.add_nodes_from(range(ppi_graph.number_of_nodes()))
    src, dst = ppi_graph.edges()
    nx_graph.add_edges_from(zip(src.numpy(), dst.numpy()))
    node2vec = Node2Vec(nx_graph, dimensions=50, walk_length=100, num_walks=50, workers=4)
    deepwalk_model = node2vec.fit(window=10, min_count=1, batch_words=4)
    all_nodes = set()
    for walk in node2vec.walks:
        all_nodes.update(walk) """
    print("\n\n############ load/ train teacher and stats #############")
    load_teacher_and_stats(model_dict, args, data, device)

    print("############ train student with teacher #############")
    train_student(args, model_dict, data, device)
    #,deepwalk_model,all_nodes



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GAT')
    parser.add_argument("--gpu", type=int, default=1,
                        help="which GPU to use. Set -1 to use CPU.")
    parser.add_argument("--residual", action="store_true", default=True,
                        help="use residual connection")
    parser.add_argument("--in-drop", type=float, default=0,
                        help="input feature dropout")
    parser.add_argument("--attn-drop", type=float, default=0,
                        help="attention dropout")
    parser.add_argument('--alpha', type=float, default=0.2,
                        help="the negative slop of leaky relu")
    parser.add_argument('--batch-size', type=int, default=2,
                        help="batch size used for training, validation and test")


    parser.add_argument("--lr", type=float, default=0.005,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")

    parser.add_argument("--t-epochs", type=int, default=60,
                        help="number of training epochs")
    parser.add_argument("--t-num-heads", type=int, default=4,
                        help="number of hidden attention heads")
    parser.add_argument("--t-num-out-heads", type=int, default=6,
                        help="number of output attention heads")
    parser.add_argument("--t-num-layers", type=int, default=2,
                        help="number of hidden layers")
    parser.add_argument("--t-num-hidden", type=int, default=256,
                        help="number of hidden units")

    parser.add_argument("--s-epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--s-num-heads", type=int, default=2,
                        help="number of hidden attention heads")
    parser.add_argument("--s-num-out-heads", type=int, default=2,
                        help="number of output attention heads")
    parser.add_argument("--s-num-layers", type=int, default=4,
                        help="number of hidden layers")
    parser.add_argument("--s-num-hidden", type=int, default=68,
                        help="number of hidden units")
    parser.add_argument("--target-layers_student", type=int, default=[1,2],
                        help="the layer of student to learn")
    parser.add_argument("--target-layers-teacher", type=int, default=[0,1],
                        help="the layer of student to learn")
    
    parser.add_argument("--mode", type=str, default='mi')
    parser.add_argument("--train-mode", type=str, default='together',
                        help="training mode: together, warmup")
    parser.add_argument("--warmup-epoch", type=int, default=600,
                        help="steps to warmup")
    
    parser.add_argument('--loss-weight', type=float, default=1.0,
                        help="weight coeff of additional loss")
    parser.add_argument('--seed', type=int, default=100,
                        help="seed")
    parser.add_argument('--tofull', type=int, default=30,
                        help="change mode to full after tofull epochs")

    args = parser.parse_args()
    print(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    
    main(args)
