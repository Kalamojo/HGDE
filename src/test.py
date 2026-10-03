import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import sys

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

def generate():
    rng = np.random.default_rng(seed=42)
    torch.manual_seed(10)
    train = torch.randn(100, 1024)
    labels = torch.empty(100)
    user = torch.empty(100)

    for i in range(0,100):
        labels[i] = rng.integers(0,5)
        user[i] = rng.integers(0,1, endpoint=True)
    return train, labels, user

def main():
    if len(sys.argv) < 2:
        print("invalid")
    elif sys.argv[1] == "data":
        train, labels, user = generate()
        torch.save(train, "train.pt")
        torch.save(labels, "labels.pt")
        torch.save(user, "user.pt")
        return
    elif sys.argv[1] == "test":
        if len(sys.argv) < 3:
            print("invalid")
        path = sys.argv[2]
        model = torch.load(path, weights_only=False)
        train, _, _ = generate()
        print(model(train))


if __name__ == "__main__":
    main()