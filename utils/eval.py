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

    def setup_data(self, embeddings: NDArray[np.float32]):
        self.loader = DataLoader(self.dataset, self.batch_size, self.shuffle_data)
    
    def update(self, sampler: DataSampler):
        pass;

    def loss(self, batch: Any, model: nn.Module) -> torch.Tensor:
        pass;

# JohnEval

class ContrastEval(ClusterEval):
    def __init__(
        self, batch_size: int = 16, shuffle_data: bool = True,
        margin: float = 2.0, weighting: float = 1.0
    ):
        super().__init__(batch_size, shuffle_data)
        self.margin = margin
        self.weighting = weighting

    def setup_data(self, embeddings: NDArray[np.float32]):
        self.dataset: ContrastiveDataset = ContrastiveDataset(embeddings)
        super().setup_data(embeddings)
    
    def update(self, pred_labels: NDArray[np.int_], sampler: DataSampler):
        if self.dataset is None or self.loader is None:
            raise ValueError("Dataset and/or loader have not been initialized with 'setup_data' yet")
        
        sampled_move = sampler.sample_pair(pred_labels)
        self.dataset.update_data(sampled_move, pred_labels)

    def loss(
            self, batch: dict[str, torch.Tensor], model: nn.Module
    ) -> torch.Tensor:
        item1, item2, closer = batch["item1"], batch["item2"], batch["closer"]
        item1_out, item2_out = model(item1), model(item2)

        m, n = item1.shape[1], item1.shape[0]
        weight = (m/n)+1
        distance = torch.linalg.vector_norm(item1_out - item2_out)
        
        loss = torch.where(
            closer,
            weight * distance.square(),
            weight * F.relu(self.margin - distance).square()
        )
        return loss

# JayEval

class JayEval(ClusterEval):
    def __init__(
        self, batch_size: int = 16, shuffle_data: bool = True,
        margin: float = 2.0, preservation_weight: float = 0.05
    ):
        super().__init__(batch_size, shuffle_data, margin)
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=0.2, p=2)
        self.preservation_weight = preservation_weight

    def setup_data(self, embeddings: NDArray[np.float32]):
        self.dataset: TripletDataset = TripletDataset(embeddings)
        super().setup_data(embeddings)

    def update(self, pred_labels: NDArray[np.int_], sampler: DataSampler):
        if self.dataset is None or self.loader is None:
            raise ValueError("Dataset and/or loader have not been initialized with 'setup_data' yet")
        
        sampled_moves = sampler.sample_triplets(pred_labels)
        self.dataset.append_data(sampled_moves)

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
