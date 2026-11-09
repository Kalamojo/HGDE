"""
File: embedding_transformer.py
Author: John Ross
Description:
    This file contains the architecture of the MLP embedding transformer and the train loop.
    The loop uses Pytorch's feed forward Neural Network and Pytorch's built in DataLoader.
    The Model takes in 3 torch tensors for its data, 1 the embedding, 2 the cluster, and 3
    the user's specification.
    Then it has multiple options for the loss function and finally it outputs a saved torch
    model.pt file to the given path

Usage:
    run 'python3 embedding_transformer.py datapath.pt loss_function_name model_file.pt

    the datapath file should be formatted as follows
    data = {
        "train": N x D torch tensor where N is the number of datapoints and D is the dimensionality of the embeddings
        "labels": N x 1 torch tensor containing the cluster each point was assigned to
        "user": N x 1 torch tensor containing must-link pairs
    }
    torch.save(data, "datapath.pt")

    Options for loss_function_name
        default: the default option is an arbitrary loss function meant to test functionality of the training loop
        user_weighted_contrastive_loss: uses contrastive learning and places a large weight on user inputs

    This script will output a pt file which can be loaded in another file as follows

    from embedding_transformer import Net
    model = torch.load(model_file, weights_only=False)

import Usage:
    define repo root as the path to the root of this repo

    import torch
    sys.path.append(str(repo_root))
    from src.embedding_transformer import train_loop as train
    from src.embedding_transformer import user_weighted_contrastive_loss as loss_function
    from src.embedding_transformer import Net
    data = {
        'train': embeddings,                                 # !!!These have to be Torch Tensors!!!
        'labels': clusters,
        'user': user_constrains
    }
    model = Net(data['train'].shape[-1])                     # Net with the dimensionality of the data
    lr = 0.00005                                             # found through experimentation
    optim = torch.optim.Adam(model.parameters(), lr = lr)
    batch_size = 16                                          # found through experimentation
    weight = 2                                               # controls how much user input is weighted, must be >= 1
    train(data, epochs, loss_function, optim, batch_size, model, weight)
    final embeddings = model(data['train'])
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys

# Architecture of the model (can change)
# This sets it up to have 1 hidden layer and the input and output dimensions are the dimensions of the embeddings
# part of PyTorch's standard protocol
class Net(nn.Module):

    def __init__(self, dimensions: int):
        super(Net, self).__init__()
        self.dimensions = dimensions
        self.layer1 = nn.Linear(dimensions, 2048)
        self.dropout = nn.Dropout(p=0.2)
        self.layer2 = nn.Linear(2048, dimensions)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# pytorch's built in dataset class, this allows us to train in batches and seperate the torch tensors
# part of PyTorch's standard protocol
class ClusterDataset(Dataset):
    def __init__(self, inputs, labels, user_specified, mode = 'train'):
        self.mode = mode
        self.user_specified = user_specified
        if self.mode == 'train':
          self.train = inputs.reshape(-1, len(inputs[0])).float()
          self.train_labels = labels

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, idx):
        if self.mode == 'train':
          return {'input': self.train[idx], 'label': self.train_labels[idx], 'user_specified': self.user_specified[idx], 'idx': idx}

# This gives us the dataloader for the training loop, part of PyTorch's standard protocol
def load_data(inputs, clusters, user_specified, batch_size):
    train_set = ClusterDataset(inputs, clusters, user_specified)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    return train_dataloader

# This is the training loop for the model
def train_loop(data, epochs, loss_function, optim, batch_size, model, weight = 1, epoch_weighted = False):
    inputs, clusters, user_specified = data['train'], data['labels'], data['user'] # seperate data
    train_dataloader = load_data(inputs, clusters, user_specified, batch_size) # get a dataloader with data
    init_train_losses = [] # to keep track of losses
    m = len(inputs) * (len(inputs) - 1) # total number of distance calulations
    n = sum(1 for x in user_specified if x != 0) # number of user weighted constraints
    m = m - n # total - user weighted
    for epoch in range(1,epochs+1): # for the given epochs
        model.eval()
        with torch.no_grad():
            init_output = model(inputs) # this is the distance we will compare each element to

        #Training phase
        model.train() # Setting the model to train phase
        train_loss = [] # loss for this current epoch
        for _, batch in enumerate(train_dataloader): # get a batch from dataloader

            inputs = batch['input']
            labels = batch['label']
            user = batch['user_specified']
            idx = batch['idx']
            outputs = torch.squeeze(model(inputs.float())) # pass embeddings through the model

            loss = loss_function(outputs, labels.float(), user, idx, init_output, clusters, m, n, weight) # calculate loss
            loss.backward() # backpropogate through the model
            optim.step() # using an Adam optimizer changes the learning rate so we take an optimizer step
            train_loss.append(loss.item()) # append the loss number
            optim.zero_grad() # zero gradients between batches

        if epoch_weighted:
            weight += 1
        
        if epoch%2==0: # every 2 epochs print out current progress
            print("Epoch : {}, Train loss: {}".format(epoch, np.mean(train_loss)))
            init_train_losses.append(np.mean(train_loss)) # keep a list of losses per epoch for a graph potentially

# default arbitrary loss function
def temp_loss(outputs, labels, user_specified, idx):
    loss = 0
    for i in range(0, len(outputs) - 1):
        if labels[i] == labels[i+1]:
            loss += (outputs[i] - outputs[i+1]).sum()
        else:
            loss += (outputs[i] + outputs[i+1]).sum()
    return loss

# user weighted contrastive loss, this calculates contrastive loss with the added component of weighting
# user constraints
def user_weighted_contrasting_loss(outputs, labels, user_specified, idx, init_outputs, clusters, m, n, weighting = 1):
    # outputs is the output of the current batch of embeddings run through the model
    # labels is the given cluster that the embeddings were initially givin
    # user_specified is a 1-indexed array of positive and/or negative user specification
    # idx is the index for the current batch (used to make sure we don't weight the current vector with the initial version of current vector)
    # clusters is the initial clustering of inputs
    # m is unweighted distance caluculations
    # n is weighted calculations
    # weighting is the power that the weigths will be raised to
    margin = 2.0
    pair_losses = []
    for i in range(len(outputs) if outputs.dim() > 1 else 1): # if the batch size is 1 it len(outputs) is wrong
        for j in range(len(init_outputs)):
            # A signed, one-based value identifies a constrained dataset index.
            constraint = 0
            specified = int(user_specified[i].item())
            if abs(specified) - 1 == int(j):
                constraint = 1 if specified > 0 else -1

            same_cluster = bool(labels[i] == clusters[j])
            positive_pair = constraint == 1 or (constraint == 0 and same_cluster)
            # the weights are calculated such that is weighting = 1, the average weight of all the
            # calculations is 1 and the ratio of (total weight:total not weight) = (n:1)
            # (i.e if 2 there are 2 user constraints, they will make up for 67% of the loss
            #   and the rest will make up 33% of the loss so the ratio is 2:1)
            if idx[i] == j:
                weight = 0
            elif specified != 0: 
                weight = (m/n)+1 if constraint != 0 else (m+n)/(m+(m*n**weighting))
            else:
                weight = (m+n)/(m+(m*n**weighting))
            distance = torch.linalg.vector_norm(outputs[i] - init_outputs[j])

            # Positive pairs always shrink; negatives repel only while closer
            # than the margin. With normalized embeddings, distances are at
            # most 2, so a margin of 2 keeps nearly all negative pairs active.
            if positive_pair:
                pair_losses.append(weight * distance.square())
            else:
                pair_losses.append(weight * F.relu(margin - distance).square())

    if not pair_losses:
        return outputs.sum() * 0.0
    return torch.stack(pair_losses).mean()

def user_weighted_whole_cluster_loss(outputs, labels, user_specified, idx, init_outputs, clusters, m, n, weighting = 1):
    margin = 2.0
    pair_losses = []
    for i in range(len(outputs) if outputs.dim() > 1 else 1): # if the batch size is 1 it len(outputs) is wrong
        for j in range(len(init_outputs)):
            specified = int(user_specified[i].item())
            same_cluster = bool(labels[i] == clusters[j])
            constrained_cluster = None
            constraint = 0
            if abs(specified) - 1 == int(j):
                constraint = 1 if specified > 0 else -1
            if specified != 0:
                constrained_cluster = clusters[abs(specified) - 1]
            in_constrained_cluster = (
                specified != 0 and clusters[j] == constrained_cluster
            )
            # A user constraint applies to every item in the referenced item's cluster.
            positive_pair = (
                specified > 0 and in_constrained_cluster
            ) or (not in_constrained_cluster and same_cluster)
            if idx[i] == j:
                weight = 0
            elif specified != 0: 
                weight = (m/n)+1 if constraint else (1 if in_constrained_cluster else (m+n)/(m+(m*n**weighting)))
            else:
                weight = (m+n)/(m+(m*n**weighting))
            distance = torch.linalg.vector_norm(outputs[i] - init_outputs[j])
            
            if positive_pair:
                pair_losses.append(weight * distance.square())
            else:
                pair_losses.append(weight * F.relu(margin - distance).square())

    if not pair_losses:
        return outputs.sum() * 0.0
    return torch.stack(pair_losses).mean()

def user_weighted_goal_loss(outputs, labels, user_specified, idx, init_outputs, clusters, m, n, weighting = 1):
    margin = 2.0
    pair_losses = []
    for i in range(len(outputs) if outputs.dim() > 1 else 1): # if the batch size is 1 it len(outputs) is wrong
        for j in range(len(init_outputs)):
            # A signed, one-based value identifies a constrained dataset index.
            goal = False
            constraint = 0
            specified = int(user_specified[i].item())
            if abs(specified) - 1 == int(j):
                constraint = 1 if specified > 0 else -1

            same_cluster = bool(labels[i] == clusters[j])
            if constraint == 1 and same_cluster:
                goal == True
            elif constraint == -1 and not same_cluster:
                goal == True
            positive_pair = constraint == 1 or (constraint == 0 and same_cluster)
            if idx[i] == j:
                weight = 0
            elif specified != 0: 
                weight = (m/n)+1 if (constraint != 0 and not goal) else (m+n)/(m+(m*n**weighting))
            else:
                weight = (m+n)/(m+(m*n**weighting))
            distance = torch.linalg.vector_norm(outputs[i] - init_outputs[j])

            if positive_pair:
                pair_losses.append(weight * distance.square())
            else:
                pair_losses.append(weight * F.relu(margin - distance).square())

    if not pair_losses:
        return outputs.sum() * 0.0
    return torch.stack(pair_losses).mean()

def user_weighted_no_cluster_loss(outputs, labels, user_specified, idx, init_outputs, clusters, m, n, weighting = 1):
    margin = 2.0
    pair_losses = []
    for i in range(len(outputs) if outputs.dim() > 1 else 1): # if the batch size is 1 it len(outputs) is wrong
        for j in range(len(init_outputs)):
            # A signed, one-based value identifies a constrained dataset index.
            constraint = 0
            specified = int(user_specified[i].item())
            if specified == 0:
                continue
            if abs(specified) - 1 == int(j):
                constraint = 1 if specified > 0 else -1

            same_cluster = bool(labels[i] == clusters[j])
            positive_pair = constraint == 1 or (constraint == 0 and same_cluster)
            if idx[i] == j:
                weight = 0
            elif specified != 0: 
                weight = 1 if constraint != 0 else 0
            else:
                weight = 0
            distance = torch.linalg.vector_norm(outputs[i] - init_outputs[j])
            if weight != 0:
                if positive_pair:
                    pair_losses.append(weight * distance.square())
                else:
                    pair_losses.append(weight * F.relu(margin - distance).square())

    if not pair_losses:
        return outputs.sum() * 0.0
    return torch.stack(pair_losses).mean()

def main():
    if sys.argv[1] == "help":
        print("run the script with arguments in the following order, -> python3 embedding_transformer.py data.pt loss_name save_file_path")
        return
    else:
        if len(sys.argv) < 4:
            print("incorrect formatting of arguments, run 'python3 embedding_transformer help' to see format or look at file")
            return
    datapath = sys.argv[1]
    loss_name = sys.argv[2]
    save_path = sys.argv[3]

    data = torch.load(datapath)
    loss = None
    if loss_name == "weighted_learning":
        loss = user_weighted_contrasting_loss
    else:
        loss = temp_loss
    model = Net(len(data['train'][0]))
    batch_size = 32
    epochs = 10
    optim = torch.optim.Adam(model.parameters(), lr = 0.1)
    train_loop(data, epochs, loss, optim, batch_size, model)
    torch.save(model, save_path)
    return

if __name__ == "__main__":
    main()