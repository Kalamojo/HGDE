import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from typing import Any
from utils.data import ContrastiveDataset, TripletDataset


class ClusterEval:
    def __init__(
        self, dataset: Dataset, batch_size: int = 16, 
        shuffle_data: bool = True, margin: float = 2.0
    ):
        self.dataset = dataset
        self.loader = DataLoader(self.dataset, batch_size, shuffle_data)
        self.margin = margin

    def loss(self, batch: Any, model: nn.Module) -> torch.Tensor:
        pass;

# JohnEval

class JohnEvalBase(ClusterEval):
    def __init__(
        self, batch_size: int = 16, shuffle_data: bool = True,
        margin: float = 2.0, weighting: float = 1.0
    ):
        dataset = ContrastiveDataset()
        super().__init__(dataset, batch_size, shuffle_data, margin)
        self.weighting = weighting

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
        dataset = ContrastiveDataset()
        super().__init__(dataset, batch_size, shuffle_data, margin)
        self.triplet_loss_fn = nn.TripletMarginLoss(margin=0.2, p=2)
        self.preservation_weight = preservation_weight

    def loss(
            self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
            model: nn.Module
    ) -> torch.Tensor:
        anchor, positive, negative = batch
        a_out, p_out, n_out = model(anchor), model(positive), model(negative)
        
        triplet_loss = self.triplet_loss_fn(a_out, p_out, n_out)
        originals = torch.cat((anchor, positive, negative), dim=0)
        transformed = torch.cat((a_out, p_out, n_out), dim=0)
        preservation_loss = F.mse_loss(transformed, F.normalize(originals, p=2, dim=1))
        
        loss = triplet_loss + self.preservation_weight * preservation_loss
        return loss
