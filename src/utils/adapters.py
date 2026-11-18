import torch
import torch.nn as nn
from torch import optim
from torch.optim import Optimizer
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Optional
from src.utils.models import JohnNet, JayNet
from src.utils.data.benchmarks import Benchmark
from src.utils.eval import ClusterEval


class EmbeddingAdapter:
    def __init__(self):
        pass;
    
    def learn(self, eval: ClusterEval) -> Optional[list[float]]:
        pass;
    
    def predict(self, benchmark: Benchmark) -> NDArray[np.float64]:
        pass;

class NNAdapter(EmbeddingAdapter):
    def __init__(
        self, epochs: int, model: nn.Module, optimizer: Optimizer
    ):
        self.epochs = epochs
        self.model = model
        self.optimizer = optimizer
    
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
                desc=f"Epoch {epoch+1}", unit="batch"
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
    
    def predict(self, benchmark: Benchmark) -> NDArray[np.float64]:
        original_embeddings = torch.from_numpy(benchmark.embeddings)
        self.model.eval()
        new_embeddings = self.model(original_embeddings).detach().numpy()

        return new_embeddings

class JohnAdapter(NNAdapter):
    def __init__(self, epochs: int, learning_rate: float, dimensions: int, hidden_size: int = 1024, dropout: float = 0.1):
        model = JohnNet(dimensions, hidden_size, dropout)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        super().__init__(epochs, model, optimizer)

class JayAdapter(NNAdapter):
    def __init__(self, epochs: int, learning_rate: float, dimensions: int, hidden_size: int = 256, scaling_factor: float = 0.1):
        model = JayNet(dimensions, hidden_size, scaling_factor)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        super().__init__(epochs, model, optimizer)
