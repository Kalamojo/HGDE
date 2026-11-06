import torch.nn as nn
from torch.optim import Optimizer
from tqdm import tqdm
from utils.eval import ClusterEval


class ClusterState:
    def __init__(self, eval: ClusterEval, epochs: int, model: nn.Module, optimizer: Optimizer):
        self.eval = eval
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer

    def recluster_round(self):
        # collect suggestion and prepare eval data

        # training loop
        losses = []
        for epoch in range(self.epochs):
            self.model.train()
            batch_loss = 0.0
            batch_count = 0
            
            tqdm_loader = tqdm(
                enumerate(self.eval.loader), total=len(self.eval.loader),
                desc=f"Epoch {epoch+1}", unit="batch"
            )
            for _, batch in tqdm_loader:
                loss = self.eval.loss(batch, self.model)
                loss.backward()
                self.optimizer.step()
                loss_val = loss.item()
                batch_loss += loss_val
                batch_count += 1
                tqdm_loader.set_postfix(loss=loss_val)
            losses.append(batch_loss / batch_count)

        # recluster and evaluate new clusters

        # log results


