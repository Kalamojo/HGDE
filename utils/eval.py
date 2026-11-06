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
        self, batch_size: int = 16, shuffle_data: bool = True, margin: float = 2.0
    ):
        self.dataset: Optional[Dataset] = None
        self.loader: Optional[DataLoader] = None
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.margin = margin

    def setup_data(self, embeddings: NDArray[np.float32]):
        self.loader = DataLoader(self.dataset, self.batch_size, self.shuffle_data)
    
    def update(self, sampler: DataSampler):
        pass;

    def loss(self, batch: Any, model: nn.Module) -> torch.Tensor:
        pass;

# JohnEval

class JohnEvalBase(ClusterEval):
    def __init__(
        self, batch_size: int = 16, shuffle_data: bool = True,
        margin: float = 2.0, weighting: float = 1.0
    ):
        super().__init__(batch_size, shuffle_data, margin)
        self.weighting = weighting

    def setup_data(self, embeddings: NDArray[np.float32]):
        self.dataset: ContrastiveDataset = ContrastiveDataset(embeddings)
        super().setup_data(embeddings)

    # default arbitrary loss function
    def loss_method(self) -> torch.Tensor:
        loss = 0
        for i in range(0, len(outputs) - 1):
            if labels[i] == labels[i+1]:
                loss += (outputs[i] - outputs[i+1]).sum()
            else:
                loss += (outputs[i] + outputs[i+1]).sum()
        return torch.tensor(loss)

    def loss(self, batch: Any, model: nn.Module) -> torch.Tensor:
        pass;

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
            self, batch: dict[str, torch.Tensor],
            model: nn.Module
    ) -> torch.Tensor:
        anchor, positive, negative = batch["anchor"], batch["positive"], batch["negative"]
        a_out, p_out, n_out = model(anchor), model(positive), model(negative)
        
        triplet_loss = self.triplet_loss_fn(a_out, p_out, n_out)
        originals = torch.cat((anchor, positive, negative), dim=0)
        transformed = torch.cat((a_out, p_out, n_out), dim=0)
        preservation_loss = F.mse_loss(transformed, F.normalize(originals, p=2, dim=1))
        
        loss = triplet_loss + self.preservation_weight * preservation_loss
        return loss
