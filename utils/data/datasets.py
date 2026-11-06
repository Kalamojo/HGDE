import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.typing import NDArray


# pytorch's built in dataset class, this allows us to train in batches and seperate the torch tensors
# part of PyTorch's standard protocol
class ContrastiveDataset(Dataset):
    def __init__(self, embeddings: NDArray[np.float32], labels, user_specified):
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.user_specified = user_specified
        self.train_labels = labels

    def __len__(self):
        return self.train.shape[0]

    def update_data(self, triplets: list[tuple[int, int, int]]):
        self.triplets += triplets

    def __getitem__(self, idx):
        return {'input': self.train[idx], 'label': self.train_labels[idx], 'user_specified': self.user_specified[idx], 'idx': idx}

# Organizes the embeddings and triplet indices into a format that PyTorch can use during training
class TripletDataset(Dataset):
    def __init__(self, embeddings: NDArray[np.float32]):
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.triplets: list[tuple[int, int, int]] = []

    def __len__(self):
        return len(self.triplets)

    def append_data(self, triplets: list[tuple[int, int, int]]):
        self.triplets += triplets

    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]
        return {'anchor': self.embeddings[a], 'positive': self.embeddings[p], 'negative': self.embeddings[n]}
