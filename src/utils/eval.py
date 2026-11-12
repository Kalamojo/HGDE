import torch
from torch import nn
from torch.optim import Optimizer
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Any, Optional
from src.utils.data.datasets import ContrastiveDataset, TripletDataset
from src.utils.data.benchmarks import DataSampler


class ClusterEval:
    def __init__(
        self, batch_size: int = 16, epochs: int = 10, shuffle_data: bool = True
    ):
        self.dataset: Optional[Dataset] = None
        self.loader: Optional[DataLoader] = None
        self.epochs = epochs
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data

    def train_model(self, model: nn.Module, optimizer: Optimizer) -> list[float]:
        if self.loader is None:
            raise ValueError("Data loader has not yet been initialized. Perform eval update first")
        
        losses = []
        for epoch in range(self.epochs):
            model.train()
            batch_loss = 0.0
            batch_count = 0
            
            tqdm_loader = tqdm(
                enumerate(self.loader), total=len(self.loader),
                desc=f"Epoch {epoch+1}", unit="batch"
            )
            for _, batch in tqdm_loader:
                optimizer.zero_grad()
                loss = self.loss(batch, model)
                loss.backward()
                optimizer.step()
                loss_val = loss.item()
                batch_loss += loss_val
                batch_count += 1
                tqdm_loader.set_postfix(loss=loss_val)
            losses.append(batch_loss / batch_count)

        return losses

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
        self, batch_size: int = 16, epochs: int = 10, shuffle_data: bool = True,
        margin: float = 2.0
    ):
        super().__init__(batch_size, epochs, shuffle_data)
        self.margin = margin

    def setup_data(self, sampler: DataSampler):
        self.dataset: ContrastiveDataset = ContrastiveDataset(sampler.embeddings)
    
    def update(self, pred_labels: NDArray[np.int_], current_embeddings: NDArray[np.float64], sampler: DataSampler):
        super().update(pred_labels, sampler)
        
        sampled_move = sampler.sample_pair(pred_labels)
        self.dataset.append_data(sampled_move)
        self.dataset.update_stuff(current_embeddings, pred_labels)
        
        if self.loader is None:
            self.setup_loader()

    def loss(
            self, batch: dict[str, torch.Tensor], model: nn.Module
    ) -> torch.Tensor:
        item1, item2, label1, label2, closer, total_items, unconstrained, pred_label = batch["item1"], batch["item2"], batch["label1"], batch["label2"], batch["closer"], batch["total_items"], batch["unconstrained"], batch["pred_labels"]
        item1_out, item2_out = model(item1), model(item2)

        m = item1.shape[0]
        weight = (m/total_items)+1
        distance = torch.linalg.vector_norm(item1_out - item2_out, dim=1)
        unconstrained_loss = []
        for num in len(unconstrained):
            positive1 = (label1 == pred_label[num] and not closer) or (closer and label2 == pred_label[num])
            positive2 = (label2 == pred_label[num] and not closer) or (closer and label1 == pred_label[num])
            distance1 = torch.linalg.vector_norm(item1_out - unconstrained[num], dim=1)
            distance2 = torch.linalg.vector_norm(item2_out - unconstrained[num], dim=1)
            if positive1:
                unconstrained_loss.append(distance1.square())
            else:
                unconstrained_loss.append(F.relu(self.margin - distance1).square)
            if positive2:
                unconstrained_loss.append(distance2.square())
            else:
                unconstrained_loss.append(F.relu(self.margin - distance2).square)
        
        loss = torch.where(
            closer,
            weight * distance.square(),
            weight * F.relu(self.margin - distance).square()
        )
        return loss.mean()

# JayEval

class TripletEval(ClusterEval):
    def __init__(
        self, batch_size: int = 16, epochs: int = 10, shuffle_data: bool = True,
        margin: float = 2.0, preservation_weight: float = 0.05
    ):
        super().__init__(batch_size, epochs, shuffle_data)
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
