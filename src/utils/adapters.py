import torch
import torch.nn as nn
from torch import optim
from torch.optim import Optimizer
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Optional
from src.utils.models import JohnNet, JayNet
from src.utils.clusterings import Clusterer
from src.utils.data.benchmarks import Benchmark
from src.utils.eval import ClusterEval
from src.utils.predictions import Predictions


class EmbeddingAdapter:
    def __init__(self, clusterer: Clusterer):
        self.clusterer = clusterer
        self.predictions = Predictions()
    
    def setup_predictions(self, benchmark: Benchmark):
        pass;
    
    def learn(self, eval: ClusterEval) -> Optional[list[float]]:
        pass;
    
    def predict(self, benchmark: Benchmark) -> NDArray[np.int_]:
        pass;

class NNAdapter(EmbeddingAdapter):
    def __init__(
        self, clusterer: Clusterer, epochs: int, model: nn.Module, optimizer: Optimizer, verbose: bool
    ):
        super().__init__(clusterer)
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
        self.verbose = verbose
    
    def setup_predictions(self, benchmark: Benchmark):
        initial_embeddings = benchmark.embeddings
        pred_labels = self.clusterer.run(initial_embeddings)
        self.predictions.update_labels(pred_labels)
        self.predictions.update_embeddings(initial_embeddings)   # scored by test_pair_auc; needed by ContrastiveEval

    def learn(self, eval: ClusterEval) -> list[float]:
        if eval.loader is None:
            raise ValueError("Data loader has not yet been initialized. Perform eval update first")
        
        losses = []
        for epoch in range(self.epochs):
            self.model.train()
            batch_loss = 0.0
            batch_count = 0
            
            tqdm_loader = tqdm(
                enumerate(eval.loader), total=len(eval.loader),
                desc=f"Epoch {epoch+1}", unit="batch", disable=not self.verbose
            )
            for _, batch in tqdm_loader:
                self.optimizer.zero_grad()
                loss = eval.loss(batch, self.model)
                loss.backward()
                self.optimizer.step()
                loss_val = loss.item()
                batch_loss += loss_val
                batch_count += 1
                tqdm_loader.set_postfix(loss=loss_val)
            losses.append(batch_loss / batch_count)

        return losses
    
    def predict(self, benchmark: Benchmark) -> NDArray[np.int_]:
        original_embeddings = torch.from_numpy(benchmark.embeddings)
        self.model.eval()
        new_embeddings = self.model(original_embeddings).detach().numpy()
        new_pred_labels = self.clusterer.run(new_embeddings)
        self.predictions.update_labels(new_pred_labels)
        self.predictions.update_embeddings(new_embeddings)

        return new_pred_labels

class JohnAdapter(NNAdapter):
    def __init__(self, clusterer: Clusterer, epochs: int, learning_rate: float, dimensions: int, hidden_size: int = 1024, dropout: float = 0.1, verbose: bool = True):
        model = JohnNet(dimensions, hidden_size, dropout)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        super().__init__(clusterer, epochs, model, optimizer, verbose)

class JayAdapter(NNAdapter):
    def __init__(self, clusterer: Clusterer, epochs: int, learning_rate: float, dimensions: int, hidden_size: int = 256, scaling_factor: float = 0.1, verbose: bool = True):
        model = JayNet(dimensions, hidden_size, scaling_factor)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        super().__init__(clusterer, epochs, model, optimizer, verbose)
