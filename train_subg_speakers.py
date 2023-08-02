import argparse
import os
import pickle
import pprint
import time
import random

import dgl

import numpy as np
import torch
import torch.optim as optim
from dataset import LanderDataset
from models import LANDER


###########
# ArgParser
def arguments():
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
    return args


args = arguments()
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

def prepare_dataset(m, n):
    gs_p_list = []
    nbrs_p_list = []
    ks_p_list = []
    pkl_files = [f"voceleb_speaker_{i}.pkl" for i in range(1, 151)]  # Assuming you have 5934 pkl files
    for _ in range(m):
        pkl_files_r = pkl_files.copy()
        random.shuffle(pkl_files_r)  # Shuffle the pkl_files list
        print("_-------------:", _)
        print(pkl_files_r)

        gs_p = []
        nbrs_p = []
        ks_p = []

        while len(pkl_files_r) >= 1:
            selected_files = pkl_files_r[:n]  # Select n files for the current iteration
            pkl_files_r = pkl_files_r[n:]  # Remove the selected files from the list

            merged_features = []
            merged_labels = []
            for file in selected_files:
                subset_file = os.path.join(args.data_path, file)
                with open(subset_file, "rb") as f:
                    subset_features, subset_labels = pickle.load(f)
                merged_features.append(subset_features)
                merged_labels.append(subset_labels)
                merged_features_arr = np.concatenate(merged_features, axis=0)
                merged_labels_arr = np.concatenate(merged_labels, axis=0)
            print("merged_features_arr: ")
            # pprint.pprint(merged_features_arr)
            # print(type(merged_features_arr))
            print("array shape：", merged_features_arr.shape)
            # print("array dimension：", merged_features_arr.ndim)
            # print("array type：", merged_features_arr.dtype)

            k_list = [int(k_l) for k_l in args.knn_k.split(",")]  # knn [10, 5, 3]
            lvl_list = [int(l_l) for l_l in args.levels.split(",")]  # levels [2, 3, 4]

            # Dataset preparation includes creating ground truth graph levels for different level list with different knn
            for k, l in zip(k_list, lvl_list):
                dataset = LanderDataset(
                    features=merged_features_arr,
                    labels=merged_labels_arr,
                    k=k,
                    levels=l,
                    faiss_gpu=args.faiss_gpu,
                )
                gs_p += [g_p for g_p in dataset.gs]  # graphs
                nbrs_p += [nbr for nbr in dataset.nbrs]  # neighbours
                ks_p += [k for g_p in dataset.gs]  # k

        gs_p_list.append(gs_p)
        nbrs_p_list.append(nbrs_p)
        ks_p_list.append(ks_p)
        print("1---------------------")
        print_graph_info(gs_p_list[_])
        print("1---------------------")

    return gs_p_list, ks_p_list, nbrs_p_list


def prepare_dataset_val():
    return


def print_graph_info(gs_i):
    print("Num graphs = %d" % (len(gs_i)))
    for idx, graph in enumerate(gs_i):
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        print("Graph {}:".format(idx), flush=True)
        print("Number of nodes:", num_nodes, flush=True)
        print("Number of edges:", num_edges, flush=True)


def set_train_sampler_loader(g_l, k_l):
    fanouts = [k_l - 1 for i in range(args.num_conv + 1)]#对于某个节点的k个最近邻居来讲，他自身也是最近的邻居之一，所以实际上只有k-1个邻居
    # print("fanouts:", fanouts)
    sampler = dgl.dataloading.MultiLayerNeighborSampler(fanouts)
    # fix the number of edges
    train_dataloader_l = dgl.dataloading.DataLoader(
        g_l,
        torch.arange(g_l.num_nodes()),
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
    )
    return train_dataloader_l


def train_epoch(epoch_t, gs_t, ks_t):

    for loaders in range(len(train_loaders_list[epoch])):
        for batch in range(len(train_loaders_list[epoch][loaders])):
            # train_loaders_list[epoch][loaders]
            minibatch = next(train_loaders_iter_list[epoch][loaders])
            # try:
            #     minibatch = next(train_loaders[loader_id])
            # except:
            #     train_loaders[loader_id] = iter(
            #         set_train_sampler_loader(gs_t[loader_id], ks_t[loader_id])
            #     )
            #     minibatch = next(train_loaders[loader_id])
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
            # if (batch + 1) % 10 == 0:
            #     print(
            #         "epoch: %d, batch: %d / %d, loader_id : %d / %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f"
            #         % (
            #             epoch_t,
            #             batch,
            #             # batch_per_loader,
            #             # loader_id,
            #             # num_of_loaders,
            #             loss.item(),
            #             loss_den_val,
            #             loss_conn_val,
            #         )
            #     )
            print(
                "epoch: %d, batch: %d / %d, loader_id : %d / %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f"
                % (
                    epoch_t,
                    batch,
                    len(train_loaders_list[epoch][loaders]),
                    loaders,
                    len(train_loaders_list[epoch]),
                    loss.item(),
                    loss_den_val,
                    loss_conn_val,
                )
            )
            print(
                "epoch: %d, batch: %d / %d, loader_id : %d / %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f"
                % (
                    epoch_t,
                    batch,
                    len(train_loaders_list[epoch][loaders]),
                    loaders,
                    len(train_loaders_list[epoch]),
                    np.array(loss_val_total).mean(),
                    np.array(loss_den_val_total).mean(),
                    np.array(loss_conn_val_total).mean(),
                )
            )
            scheduler.step()
    print(
        "epoch: %d, loss: %.6f, loss_den: %.6f, loss_conn: %.6f"
        % (
            epoch_t,
            np.array(loss_val_total).mean(),
            np.array(loss_den_val_total).mean(),
            np.array(loss_conn_val_total).mean(),
        )
    )
    torch.save(model.state_dict(), args.model_filename)
    torch.save(model.state_dict(), f'{args.model_filename}_epoch_{epoch}.pt')


def val(epoch_v, model_v):
    model.eval()
    loss_den_val_total = []
    loss_conn_val_total = []
    loss_val_total = []
    # ...........

    # do validation


# Model Definition
torch.cuda.manual_seed(940105)  # add by Allen Zhang 24/07/2023
# feature_dim = gs[0].ndata["features"].shape[1]
feature_dim = args.hidden
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

gs_list, ks_list, nbrs_list = prepare_dataset(args.epochs, 30)

train_loaders_list = []  #
for gs_group, ks_group in zip(gs_list, ks_list):
    train_loaders = []  # 列表 of DataLoaders for 一组 gs
    for gs, ks in zip(gs_group, ks_group):
        train_dataloader = set_train_sampler_loader(gs, ks)
        train_loaders.append(train_dataloader)
    train_loaders_list.append(train_loaders)
# print("train_loaders_list len:", len(train_loaders_list))
# print("train_loaders_list[0][0]:", len(train_loaders_list[0][0]))
# print("train_loaders_list[0][1]:", len(train_loaders_list[0][1]))
#
# print("train_loaders_list[1][0]:", len(train_loaders_list[1][0]))
# print("train_loaders_list[1][1]:", len(train_loaders_list[1][1]))

# keep num_batch_per_loader the same for every sub_dataloader
num_batch_per_loader_list = []
for nbpl in train_loaders_list:
    num_batch_per_loader = []
    for nb in nbpl:
        num_batch = len(nb)
        num_batch_per_loader.append(num_batch)
    num_batch_per_loader_list.append(num_batch_per_loader)

print("num_batch_per_loader_list len:", len(num_batch_per_loader_list))
print("num_batch_per_loader_list[0] len:", len(num_batch_per_loader_list[0]))
print("num_batch_per_loader_list[1] len:", len(num_batch_per_loader_list[1]))
# print("num_batch_per_loader_list[0][0]:", num_batch_per_loader_list[0][0])
# print("num_batch_per_loader_list[0][1]:", num_batch_per_loader_list[0][1])
# print("num_batch_per_loader_list[0][2]:", num_batch_per_loader_list[0][2])
# print("num_batch_per_loader_list[0][3]:", num_batch_per_loader_list[0][3])
# print("num_batch_per_loader_list[1][0]:", num_batch_per_loader_list[1][0])
# print("num_batch_per_loader_list[1][1]:", num_batch_per_loader_list[1][1])
# print("num_batch_per_loader_list[1][2]:", num_batch_per_loader_list[1][2])
# print("num_batch_per_loader_list[1][3]:", num_batch_per_loader_list[1][3])

# 获取数据加载的总批次数量
total_batches = 0
for tll in train_loaders_list:
    total_batches += len(tll)
print("total batches:", total_batches)

train_loaders_iter_list = []
# for i in range(len(train_loaders_list)):
#     train_loaders_iter = []
#     for j in range(len(train_loaders_list[i])):
#         train_loaders = [iter(train_loader) for train_loader in train_loaders_list[i][j]]
#         print(type(train_loaders[0]))
#         train_loaders_iter.append(train_loaders)
#     train_loaders_iter_list.append(train_loaders_iter)

for train_loaders in train_loaders_list:
    train_loaders_iter = []  # 列表 of 迭代器 for 一组 Dataloaders
    for dataloader in train_loaders:
        # 将 Dataloader 对象转换为迭代器，并添加到 train_loaders_iter 列表中
        dataloader_iter = iter(dataloader)
        train_loaders_iter.append(dataloader_iter)
    train_loaders_iter_list.append(train_loaders_iter)

# 初始化学习率调度器
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    opt, T_max=args.epochs * total_batches, eta_min=1e-5
)
# scheduler = optim.lr_scheduler.CosineAnnealingLR(
#     opt, T_max=args.epochs * num_batch_per_loader * num_loaders, eta_min=1e-5
# )

print("Start Training.")

###############
# Training Loop
for epoch in range(args.epochs):
    loss_den_val_total = []
    loss_conn_val_total = []
    loss_val_total = []

    # num_batch_per_loader = len(train_loaders[0])
    # train_loaders = [iter(train_loader) for train_loader in train_loaders]
    # num_loaders = len(train_loaders)
    # print("num_batch_per_loader, train_loaders, num_loaders: ", num_batch_per_loader, train_loaders, num_loaders)
    # for each epoch loading speakers in different sequence
    train_epoch(epoch, gs_list[epoch], ks_list[epoch])
torch.save(model.state_dict(), args.model_filename)
