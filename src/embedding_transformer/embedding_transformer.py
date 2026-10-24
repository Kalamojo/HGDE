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

    This script will output a pt file which can be loaded in another file as follows

    from embedding_transformer import Net
    model = torch.load(model_file, weights_only=False)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys

# Architecture of the model (can change)
# This sets it up to have 2 hidden layer and the input and output dimensions are the dimensions of the embeddings
# part of PyTorch's standard protocol
class Net(nn.Module):

    def __init__(self, dimensions: int):
        super(Net, self).__init__()
        self.dimensions = dimensions
        self.layer1 = nn.Linear(dimensions, 2048)
        self.dropout = nn.Dropout(p=0.5)
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
def train_loop(data, epochs, loss_function, optim, batch_size, model):
    inputs, clusters, user_specified = data['train'], data['labels'], data['user'] # seperate data
    train_dataloader = load_data(inputs, clusters, user_specified, batch_size) # get a dataloader with data
    init_train_losses = [] # to keep track of losses
    for epoch in range(1,epochs+1): # for the given epochs
        model.eval()
        with torch.no_grad():
            init_output = model(inputs)

        #Training phase
        model.train()  #Setting the model to train phase
        train_loss = [] # loss for this current epoch
        for _, batch in enumerate(train_dataloader): # get a batch from dataloader

            inputs = batch['input']
            labels = batch['label']
            user = batch['user_specified']
            idx = batch['idx']
            outputs = torch.squeeze(model(inputs.float())) # pass embeddings through the model

            loss = loss_function(outputs, labels.float(), user, idx, init_output, clusters) # calculate loss
            loss.backward() # backpropogate through the model
            optim.step() # using an Adam optimizer changes the learning rate so we take an optimizer step
            train_loss.append(loss.item()) # append the loss number
            optim.zero_grad() # zero gradients between batches
        
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

def user_weighted_contrasting_loss(outputs, labels, user_specified, idx, init_outputs, clusters):
    margin = 2.0
    pair_losses = []
    max_distance = 0
    for i in range(len(outputs) if outputs.dim() > 1 else 1):
        for j in range(len(init_outputs)):
            # A signed, one-based value identifies a constrained dataset index.
            constraint = 0
            specified = int(user_specified[i].item())
            if abs(specified) - 1 == int(j):
                constraint = 1 if specified > 0 else -1
                break

            same_cluster = bool(labels[i] == clusters[j])
            positive_pair = constraint == 1 or (constraint == 0 and same_cluster)
            if idx[i] == j:
                weight = 0
            elif specified != 0: 
                weight = (len(init_outputs))*(len(init_outputs)-1) if constraint != 0 else 1
            else:
                weight = 1
            distance = torch.linalg.vector_norm(outputs[i] - init_outputs[j])

            # Standard contrastive terms: attract positives and repel negatives
            # until they are at least margin apart.
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