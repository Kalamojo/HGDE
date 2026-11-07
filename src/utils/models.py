import torch.nn as nn
import torch.nn.functional as F


# Architecture of the model (can change)
# This sets it up to have 1 hidden layer and the input and output dimensions are the dimensions of the embeddings
# part of PyTorch's standard protocol
class JohnNet(nn.Module):
    def __init__(self, dimensions: int, hidden_size: int = 1024, dropout: float = 0.1):
        super(JohnNet, self).__init__()
        self.dimensions = dimensions
        self.layer1 = nn.Linear(dimensions, hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.layer2 = nn.Linear(hidden_size, dimensions)

    def forward(self, x):
        x = self.layer1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        x = F.normalize(x, p=2, dim=1)
        return x

# Architecture of the model (can change)
class JayNet(nn.Module):
    def __init__(self, dimensions: int, hidden_size: int = 256, scaling_factor: float=0.1):
        super(JayNet, self).__init__()
        self.dimensions = dimensions
        self.layer1 = nn.Linear(dimensions, hidden_size)
        self.layer2 = nn.Linear(hidden_size, dimensions)
        self.scaling_factor = scaling_factor

        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x):
        x_adj = self.layer1(x)
        x_adj = F.relu(x_adj)
        x_adj = self.layer2(x_adj)
        # Added a residual connection with a scaling factor (we can experiment with different scaling factors to see what works best) so that original input gets preserved
        out = x + self.scaling_factor * F.sigmoid(x_adj)
        return F.normalize(out, p=2, dim=1)
