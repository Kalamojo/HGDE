import torch
import torch.nn as nn
from torch.optim import Optimizer
import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from src.utils.data.benchmarks import DataSampler
from src.utils.eval import ClusterEval
from src.utils.clusterings import Clusterer
from src.utils.metrics import all_metrics, sample_pairs


class ClusterState:
    def __init__(self, sampler: DataSampler, clusterer: Clusterer, eval: ClusterEval, model: nn.Module, optimizer: Optimizer):
        self.sampler = sampler
        self.clusterer = clusterer
        self.eval = eval
        self.model = model
        self.optimizer = optimizer
        self.history: list[dict] = []       # one row of metrics per round
        # fixed pairs of held-out items, so test_pair_auc is scored on the same pairs every round
        test_mask = getattr(sampler, "test_mask", None)
        self.auc_pairs = None
        if test_mask is not None and test_mask.any():
            self.auc_pairs = sample_pairs(np.flatnonzero(test_mask), 5000, np.random.default_rng(0))
    
    def full_recluster(self, rounds: int):
        self.eval.setup_data(self.sampler)
        current_embeddings = self.sampler.embeddings
        original_embeddings = torch.tensor(current_embeddings)

        for round in tqdm(range(rounds)):
            current_embeddings = self.recluster_round(current_embeddings, original_embeddings)

    def recluster_round(self, current_embeddings: NDArray[np.float64], original_embeddings: torch.Tensor) -> NDArray[np.float64]:
        # collect suggestion and prepare eval data
        pred_labels = self.clusterer.run(current_embeddings)
        self.eval.update(pred_labels, current_embeddings, self.sampler)

        # training loop
        losses = self.eval.train_model(self.model, self.optimizer)

        # recluster and evaluate new clusters
        new_embeddings = self.model(original_embeddings).detach().numpy()
        new_pred_labels = self.clusterer.run(new_embeddings)

        # log results
        self.log_results(new_pred_labels, losses, new_embeddings)

        return new_embeddings

    def log_results(self, pred_labels: NDArray[np.int_], losses: list[float], embeddings: NDArray[np.float64]):
        # please feel more than welcome to add to this
        row = {"round": len(self.history) + 1,
               **all_metrics(embeddings, pred_labels, self.sampler.true_labels,
                             getattr(self.sampler, "test_mask", None), self.auc_pairs),
               "loss": float(losses[-1]) if losses else float("nan")}
        self.history.append(row)
        print(f"Round {row['round']}: ARI {row['ari']:.3f} | test ARI {row['test_ari']:.3f} | "
              f"test pair AUC {row['test_pair_auc']:.3f} | pair P/R {row['pair_precision']:.3f}/{row['pair_recall']:.3f} | "
              f"homog/compl {row['homogeneity']:.3f}/{row['completeness']:.3f} | "
              f"Jay F1/P/R {row['f1']:.3f}/{row['precision']:.3f}/{row['recall']:.3f} | loss {row['loss']:.4f}")
