import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import adjusted_rand_score
from tqdm import tqdm
from utils.data.samplers import DataSampler
from utils.eval import ClusterEval
from utils.clusterings import Clusterer


class ClusterState:
    def __init__(self, sampler: DataSampler, clusterer: Clusterer, eval: ClusterEval, model: nn.Module, optimizer: Optimizer):
        self.sampler = sampler
        self.clusterer = clusterer
        self.eval = eval
        self.model = model
        self.optimizer = optimizer
    
    def full_recluster(self, rounds: int):
        self.eval.setup_data(self.sampler)
        current_embeddings = self.sampler.embeddings
        original_embeddings = torch.tensor(current_embeddings)

        for round in tqdm(range(rounds)):
            current_embeddings = self.recluster_round(current_embeddings, original_embeddings)

    def recluster_round(self, current_embeddings: NDArray[np.float64], original_embeddings: torch.Tensor) -> NDArray[np.float64]:
        # collect suggestion and prepare eval data
        pred_labels = self.clusterer.run(current_embeddings)
        self.eval.update(pred_labels, self.sampler)

        # training loop
        losses = self.eval.train_model(self.model, self.optimizer)

        # recluster and evaluate new clusters
        new_embeddings = self.model(original_embeddings).detach().numpy()
        new_pred_labels = self.clusterer.run(new_embeddings)

        # log results
        self.log_results(new_pred_labels, losses)

        return new_embeddings

    def log_results(self, pred_labels: NDArray[np.int_], losses: list[float]):
        # please feel more than welcome to add to this
        ari = adjusted_rand_score(self.sampler.true_labels, pred_labels)
        print("ARI:", ari)
