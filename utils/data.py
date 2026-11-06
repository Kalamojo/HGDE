import torch
from torch.utils.data import Dataset


# pytorch's built in dataset class, this allows us to train in batches and seperate the torch tensors
# part of PyTorch's standard protocol
class ContrastiveDataset(Dataset):
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

# Organizes the embeddings and triplet indices into a format that PyTorch can use during training
class TripletDataset(Dataset):
    def __init__(self, embeddings, triplets):
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.triplets = torch.as_tensor(triplets, dtype=torch.long)
        if self.triplets.ndim != 2 or self.triplets.shape[1] != 3 or len(self.triplets) == 0:
            raise ValueError("Expected a nonempty list of (anchor, positive, negative) triples")
        if self.triplets.min().item() < 0 or self.triplets.max().item() >= len(self.embeddings):
            raise ValueError("Triplet index outside embedding matrix")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]
        return self.embeddings[a], self.embeddings[p], self.embeddings[n]
