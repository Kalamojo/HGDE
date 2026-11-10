import torch
from torch.utils.data import Dataset
import numpy as np
from numpy.typing import NDArray
from typing import Optional


# pytorch's built in dataset class, this allows us to train in batches and seperate the torch tensors
# part of PyTorch's standard protocol
class ContrastiveDataset(Dataset):
    def __init__(self, embeddings: NDArray[np.float32]):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.total_items_tensor = torch.tensor(self.embeddings.shape[0])
        self.pairs: list[tuple[int, int, bool]] = []
        self.unconstrained: Optional[torch.Tensor] = None
        self.pred_labels: Optional[torch.Tensor] = None
        # self.user_specified = torch.tensor([0] * embeddings.shape[0])
        # self.pred_labels: Optional[torch.Tensor] = None

    def __len__(self):
        return len(self.pairs)
    
    def append_data(self, pair: tuple[int, int, bool]):
        self.pairs.append(pair)
    
    def update_stuff(self, unconstrained: NDArray[np.float64], pred_labels):
        self.unconstrained = torch.tensor(unconstrained)
        self.pred_labels =torch.tensor(pred_labels)

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        ind1, ind2, closer = self.pairs[idx]
        closer_tensor = torch.tensor(closer)
        
        return {'item1': self.embeddings[ind1], 'item2':  self.embeddings[ind2], 'closer': closer_tensor, "total_items": self.total_items_tensor, 'unconstrained': self.unconstrained, 'pred_labels': self.pred_labels}

# Organizes the embeddings and triplet indices into a format that PyTorch can use during training
class TripletDataset(Dataset):
    def __init__(self, embeddings: NDArray[np.float32]):
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.triplets: list[tuple[int, int, int]] = []

    def __len__(self):
        return len(self.triplets)

    def append_data(self, triplets: list[tuple[int, int, int]]):
        self.triplets += triplets

    def __getitem__(self, idx) -> dict[str, torch.Tensor]:
        a, p, n = self.triplets[idx]
        return {'anchor': self.embeddings[a], 'positive': self.embeddings[p], 'negative': self.embeddings[n]}
