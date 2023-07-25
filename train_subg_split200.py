import argparse
import os
import pickle
import time

import dgl

import numpy as np
import torch
import torch.optim as optim
from dataset import LanderDataset
from models import LANDER

###########
# ArgParser
parser = argparse.ArgumentParser()

# Dataset
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--num_subsets", type=int, default=5)
parser.add_argument("--levels", type=str, default="1")
parser.add_argument("--faiss_gpu", action="store_true")
parser.add_argument("--model_filename", type=str, default="lander.pth")

# KNN
parser.add_argument("--knn_k", type=str, default="10")
parser.add_argument("--num_workers", type=int, default=0)

# Model
parser.add_argument("--hidden", type=int, default=512)
parser.add_argument("--num_conv", type=int, default=1)
parser.add_argument("--dropout", type=float, default=0.0)
parser.add_argument("--gat", action="store_true")
parser.add_argument("--gat_k", type=int, default=1)
parser.add_argument("--balance", action="store_true")
parser.add_argument("--use_cluster_feat", action="store_true")
parser.add_argument("--use_focal_loss", action="store_true")

# Training
parser.add_argument("--epochs", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--weight_decay", type=float, default=1e-5)

args = parser.parse_args()
print(args)

###########################
# Environment Configuration
if torch.cuda.is_available():
    device = torch.device("cuda")
    print("GPU mode")
else:
    device = torch.device("cpu")
    print("CPU mode, exit")
    exit()

##################
# Data Preparation
# gs_list = []
# nbrs_list = []
# ks_list = []

gs = []
nbrs = []
ks = []

for subset_id in range(1, args.num_subsets + 1):
    print("Subset_id: ", subset_id)
    subset_file = os.path.join(args.data_path,  f"voceleb_30_speakers_split{subset_id}.pkl")
    with open(subset_file, "rb") as f:
        subset_features, subset_labels = pickle.load(f)

    k_list = [int(k) for k in args.knn_k.split(",")]  # knn [10, 5, 3]
    lvl_list = [int(l) for l in args.levels.split(",")]  # levels [2, 3, 4]

    # dataset preparation includes creating ground truth graph levels for different level list with different knn
    for k, l in zip(k_list, lvl_list):
        print("k,l:", k, l)
        dataset = LanderDataset(
            features=subset_features,
            labels=subset_labels,
            k=k,
            levels=l,
            faiss_gpu=args.faiss_gpu,
        )
        gs += [g for g in dataset.gs]  # graph
        ks += [k for g in dataset.gs]
        nbrs += [nbr for nbr in dataset.nbrs]  # neighbours

    # gs_list.append(gs)
    # nbrs_list.append(nbrs)
    # ks_list.append(ks)
print("Num graphs = %d"%(len(gs)))
for idx, graph in enumerate(gs):
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    print("Graph {}:".format(idx),flush=True)
    print("Number of nodes:", num_nodes, flush=True)
    print("Number of edges:", num_edges, flush=True)
    print()

# print("Num graphs in each subset:", [len(gs) for gs in gs_list])
# for subset_id, gs in enumerate(gs_list):
#     for idx, graph in enumerate(gs):
#         num_nodes = graph.number_of_nodes()
#         num_edges = graph.number_of_edges()
#         print(f"Subset {subset_id}, Graph {idx}:")
#         print("Number of nodes:", num_nodes)
#         print("Number of edges:", num_edges)
#         print()

print("Dataset Prepared.", flush=True)


#exit()  ## for debug

def set_train_sampler_loader(g, k):
    fanouts = [k - 1 for i in range(args.num_conv + 1)]
    #print("fanouts:", fanouts)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    # fix the number of edges
    train_dataloader = dgl.dataloading.DataLoader(
        g,
        torch.arange(g.num_nodes()),
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    return train_dataloader

train_loaders = []
for gidx, g in enumerate(gs):
    train_dataloader = set_train_sampler_loader(gs[gidx], ks[gidx])
    train_loaders.append(train_dataloader)

# # Traverse through all subsets
# for subset_id in range(args.num_subsets):
#     # For each subset, set the train sampler loader
#     train_dataloader = set_train_sampler_loader(gs_list[subset_id][0], ks_list[subset_id][0])
#     train_loaders.append(train_dataloader)

# Model Definition
torch.cuda.manual_seed(940105)  # add by Allen Zhang 24/07/2023
feature_dim = gs[0].ndata["features"].shape[1]
model = LANDER(
    feature_dim=feature_dim,
    nhid=args.hidden,
    num_conv=args.num_conv,
    dropout=args.dropout,
    use_GAT=args.gat,
    K=args.gat_k,
    balance=args.balance,
    use_cluster_feat=args.use_cluster_feat,
    use_focal_loss=args.use_focal_loss,
)
model = model.to(device)
model.train()

#################
# Hyperparameters
opt = optim.SGD(
    model.parameters(),
    lr=args.lr,
    momentum=args.momentum,
    weight_decay=args.weight_decay,
)

# keep num_batch_per_loader the same for every sub_dataloader
num_batch_per_loader = len(train_loaders[0])
train_loaders = [iter(train_loader) for train_loader in train_loaders]
num_loaders = len(train_loaders)
# train_loaders = []
# for subset_id in range(args.num_subsets):
#     train_loaders.append(iter(dgl.dataloading.MultiLayerNeighborSampler([k - 1 for i in range(args.num_conv + 1)]).dataloader(gs_list[subset_id], torch.arange(gs_list[subset_id][0].number_of_nodes()), batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.num_workers)))
#
# num_loaders = len(train_loaders)

print("num_batch_per_loader, train_loaders, num_loaders: ", num_batch_per_loader, train_loaders, num_loaders)

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=args.epochs * num_batch_per_loader * num_loaders, eta_min=1e-5
)

print("Start Training.")
#exit()
###############
# Training Loop
for epoch in range(args.epochs):
    loss_den_val_total = []
    loss_conn_val_total = []
    loss_val_total = []
    for batch in range(num_batch_per_loader):
        for loader_id in range(num_loaders):
            try:
                minibatch = next(train_loaders[loader_id])
            except:
                train_loaders[loader_id] = iter(
                    set_train_sampler_loader(gs[loader_id], ks[loader_id])
                )
                minibatch = next(train_loaders[loader_id])
            input_nodes, sub_g, bipartites = minibatch
            sub_g = sub_g.to(device)
            bipartites = [b.to(device) for b in bipartites]
            # get the feature for the input_nodes
            opt.zero_grad()
            output_bipartite = model(bipartites)
            loss, loss_den_val, loss_conn_val = model.compute_loss(
                output_bipartite
            )
            loss_den_val_total.append(loss_den_val)
            loss_conn_val_total.append(loss_conn_val)
            loss_val_total.append(loss.item())
            loss.backward()
            opt.step()
            if (batch + 1) % 10 == 0:
                print(
                    "epoch: %d, batch: %d / %d, loader_id : %d / %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f"
                    % (
                        epoch,
                        batch,
                        num_batch_per_loader,
                        loader_id,
                        num_loaders,
                        loss.item(),
                        loss_den_val,
                        loss_conn_val,
                    )
                )
            scheduler.step()
    print(
        "epoch: %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f"
        % (
            epoch,
            np.array(loss_val_total).mean(),
            np.array(loss_den_val_total).mean(),
            np.array(loss_conn_val_total).mean(),
        )
    )
    torch.save(model.state_dict(), args.model_filename)

torch.save(model.state_dict(), args.model_filename)
