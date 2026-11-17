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

    def train_model(self, model: nn.Module, optimizer: Optimizer, l1_reg=0.0) -> list[float]:
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
                l1_norm = sum(p.abs().sum() for name, p in model.named_parameters() if 'bias' not in name)
                loss = self.loss(batch, model) + l1_reg * l1_norm
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

    def train_model(self, model, optimizer, l1_reg=0.0001):
        return super().train_model(model, optimizer, l1_reg)

    def loss(
            self, batch: dict[str, torch.Tensor], model: nn.Module
    ) -> torch.Tensor:
        # I want to add parameters
        # mask percent - number of parameters we pull out, 1.0 = no mask
        mask_percent = 1.0
        goal_oriented = True
        item1, item2, label1, label2, closer, total_items, unconstrained, pred_label = batch["item1"], batch["item2"], batch["label1"], batch["label2"], batch["closer"], batch["total_items"], batch["unconstrained"], batch["pred_labels"]
        item1_out, item2_out = model(item1), model(item2)

        mask = [0]*item1.shape[-1]
        for ind in range(item1.shape[0]):
            per_element_distance = torch.abs(item1[ind] - item2[ind])
            k = int(mask_percent * item1.shape[-1])
            if k < 1:
                k = 1
            indices = sorted(range(len(per_element_distance)), key=lambda i: per_element_distance[i])
            close_indices = indices[:k]
            far_indices = indices[-k:]
            for close in close_indices:
                mask[close] = 1
            for far in far_indices:
                mask[far] = 1
        mask = item1.new_tensor(mask)

        m = item1.shape[0]
        weight = 1
        uncon_weight = 1
        item1_out = F.normalize(item1_out * mask, p=2, dim=-1)
        item2_out = F.normalize(item2_out * mask, p=2, dim=-1)
        if goal_oriented:
            goal = [bool(l1 == l2) and bool(c) for l1, l2, c in zip(label1, label2, closer)]
        else:
            goal = [False] * item1.shape[0]
        non_goal_indices = [i for i, is_goal in enumerate(goal) if not is_goal]
        if non_goal_indices:
            indices = torch.tensor(non_goal_indices, device=item1_out.device)
            distance = torch.linalg.vector_norm(
                item1_out.index_select(0, indices) - item2_out.index_select(0, indices), dim=1
            )
        unconstrained_loss = []
        for uncon in range(len(unconstrained)):
            for con in range(len(item1)):
                if goal[con]:
                    continue
                positive1 = (label1[con] == pred_label[uncon][-1] and not closer[con]) or (closer[con] and label2[con] == pred_label[uncon][-1])
                positive2 = (label2[con] == pred_label[uncon][-1] and not closer[con]) or (closer[con] and label1[con] == pred_label[uncon][-1])
                masked_uncon = F.normalize(unconstrained[uncon][-1] * mask, p=2, dim=-1)
                distance1 = torch.linalg.vector_norm(item1_out[con] - masked_uncon, dim=0)
                distance2 = torch.linalg.vector_norm(item2_out[con] - masked_uncon, dim=0)
                if positive1:
                    unconstrained_loss.append(uncon_weight * distance1.square())
                else:
                    unconstrained_loss.append(uncon_weight * F.relu(self.margin - distance1).square())
                if positive2:
                    unconstrained_loss.append(uncon_weight * distance2.square())
                else:
                    unconstrained_loss.append(uncon_weight * F.relu(self.margin - distance2).square())
        if unconstrained_loss:
            unconstrained_loss_mean = torch.stack(unconstrained_loss).mean()
        else:
            unconstrained_loss_mean = 0
        
        if non_goal_indices:
            non_goal_closer = closer.index_select(0, indices)
            loss = torch.where(
                non_goal_closer,
                weight * distance.square(),
                weight * F.relu(self.margin - distance).square()
            )
        else:
            loss = item1_out.sum() * 0
        return loss.mean() + unconstrained_loss_mean

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
