import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys

# Architecture of the model (can change)
class Net(nn.Module):

    def __init__(self, dimensions: int):
        super(Net, self).__init__()
        self.dimensions = dimensions
        self.layer1 = nn.Linear(dimensions, 1024)
        self.layer2 = nn.Linear(1024, dimensions)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.relu(x)
        x = self.layer2(x)
        out = nn.functional.sigmoid(x)
        return out

class ClusterDataset(Dataset):
    def __init__(self, inputs, labels, user_specified, mode = 'train'):
        self.mode = mode
        self.user_specified = user_specified
        if self.mode == 'train':
          self.train = inputs.reshape(-1, 1024).float()
          self.train_labels = labels

    def __len__(self):
        return self.train.shape[0]

    def __getitem__(self, idx):
        if self.mode == 'train':
          return {'input': self.train[idx], 'label': self.train_labels[idx], 'user_specified': self.user_specified[idx]}

def load_data(inputs, clusters, user_specified, batch_size):
    train_set = ClusterDataset(inputs, clusters, user_specified)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    return train_dataloader

def train_loop(data, epochs, loss_function, optim, batch_size, model):
    inputs, clusters, user_specified = data
    train_dataloader = load_data(inputs, clusters, user_specified, batch_size)
    init_train_losses = []
    for epoch in range(1,epochs+1):

        #Training phase
        model.train()  #Setting the model to train phase
        train_loss = []
        train_acc = 0.

        for idx, batch in enumerate(train_dataloader):

            inputs = batch['input']
            labels = batch['label']
            user = batch['user_specified']
            outputs = torch.squeeze(model(inputs.float()))

            loss = loss_function(outputs, labels.float(), user_specified)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            optim.zero_grad()
        train_acc = train_acc / len(inputs)
        
        if epoch%10==0:
            print("Epoch : {}, Train loss: {}".format(epoch, np.mean(train_loss)))
            init_train_losses.append(np.mean(train_loss))


def temp_loss(outputs, labels, user_specified):
    loss = 0
    for i in range(0, len(outputs) - 1):
        if labels[i] == labels[i+1]:
            loss += (outputs[i] - outputs[i+1]).sum()
        else:
            loss += (outputs[i] + outputs[i+1]).sum()
    return loss


def main():
    if sys.argv[1] == "help":
        print("run the script with arguments in the following order, -> python3 embedding_transformer data.pt labels.pt user.pt loss_name save_file_path")
        return
    else:
        if len(sys.argv) < 6:
            print("incorrect formatting of arguments, run 'python3 embedding_transformer help' to see format or look at file")
            return
    datapath = sys.argv[1]
    labelpath = sys.argv[2]
    userpath = sys.argv[3]
    loss_name = sys.argv[4]
    save_path = sys.argv[5]

    train = torch.load(datapath)
    labels = torch.load(labelpath)
    user = torch.load(userpath)
    loss = None
    if loss_name == "specific function":
        x = 1 # temp
    else:
        loss = temp_loss
    model = Net(len(train[0]))
    batch_size = 32
    epochs = 100
    optim = torch.optim.Adam(model.parameters(), lr = 0.01)
    data = (train, labels, user)
    train_loop(data, epochs, loss, optim, batch_size, model)
    torch.save(model, save_path)
    return

if __name__ == "__main__":
    main()