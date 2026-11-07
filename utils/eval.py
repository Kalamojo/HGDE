import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from numpy.typing import NDArray
from typing import Any, Optional
from utils.data.datasets import ContrastiveDataset, TripletDataset
from utils.data.samplers import DataSampler


class ClusterEval:
    def __init__(
        self, batch_size: int = 16, shuffle_data: bool = True
    ):
        self.dataset: Optional[Dataset] = None
        self.loader: Optional[DataLoader] = None
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data

    def setup_dataset(self, sampler: DataSampler):
        pass;
    
    def setup_loader(self):
        self.loader = DataLoader(self.dataset, self.batch_size, self.shuffle_data)
    
    def update(self, pred_labels: NDArray[np.int_], sampler: DataSampler):
        if self.dataset is None:
            self.setup_dataset(sampler)

    def loss(self, batch: Any, model: nn.Module) -> torch.Tensor:
        pass;

# JohnEval

class ContrastiveEval(ClusterEval):
    def __init__(
        self, batch_size: int = 16, shuffle_data: bool = True,
        margin: float = 2.0
    ):
        super().__init__(batch_size, shuffle_data)
        self.margin = margin

    def setup_data(self, sampler: DataSampler):
        self.dataset: ContrastiveDataset = ContrastiveDataset(sampler.embeddings)
    
    def update(self, pred_labels: NDArray[np.int_], sampler: DataSampler):
        super().update(pred_labels, sampler)
        
        sampled_move = sampler.sample_pair(pred_labels)
        self.dataset.append_data(sampled_move)
        
        if self.loader is None:
            self.setup_loader()

    def loss(
            self, batch: dict[str, torch.Tensor], model: nn.Module
    ) -> torch.Tensor:
        item1, item2, closer, total_items = batch["item1"], batch["item2"], batch["closer"], batch["total_items"]
        item1_out, item2_out = model(item1), model(item2)

        m = item1.shape[0]
        weight = (m/total_items)+1
        distance = torch.linalg.vector_norm(item1_out - item2_out, dim=1)
        
        loss = torch.where(
            closer,
            weight * distance.square(),
            weight * F.relu(self.margin - distance).square()
        )
        return loss.mean()

# JayEval

class TripletEval(ClusterEval):
    def __init__(
        self, batch_size: int = 16, shuffle_data: bool = True,
        margin: float = 2.0, preservation_weight: float = 0.05
    ):
        super().__init__(batch_size, shuffle_data)
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=margin, p=2)
        self.preservation_weight = preservation_weight
    
    def setup_data(self, sampler: DataSampler):
        self.dataset: TripletDataset = TripletDataset(sampler.embeddings)
    
    def update(self, pred_labels: NDArray[np.int_], sampler: DataSampler):
        super().update(pred_labels, sampler)
        
        sampled_moves = sampler.sample_triplets(pred_labels)
        self.dataset.append_data(sampled_moves)
        
        if self.loader is None:
            self.setup_loader()

    def loss(
            self, batch: dict[str, torch.Tensor], model: nn.Module
    ) -> torch.Tensor:
        anchor, positive, negative = batch["anchor"], batch["positive"], batch["negative"]
        a_out, p_out, n_out = model(anchor), model(positive), model(negative)
        
        triplet_loss = self.triplet_loss_fn(a_out, p_out, n_out)
        originals = torch.cat((anchor, positive, negative), dim=0)
        transformed = torch.cat((a_out, p_out, n_out), dim=0)
        preservation_loss = F.mse_loss(transformed, F.normalize(originals, p=2, dim=1))
        
        loss = triplet_loss + self.preservation_weight * preservation_loss
        return loss
