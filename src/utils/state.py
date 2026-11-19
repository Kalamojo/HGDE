import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Optional
from src.utils.adapters import EmbeddingAdapter
from src.utils.data.benchmarks import Benchmark
from src.utils.eval import ClusterEval
from src.utils.clusterings import Clusterer
from src.utils.metrics import all_metrics, sample_pairs


class ClusterState:
    def __init__(self, benchmark: Benchmark, clusterer: Clusterer, eval: ClusterEval, adapter: EmbeddingAdapter):
        self.benchmark = benchmark
        self.clusterer = clusterer
        self.eval = eval
        self.adapter = adapter
        self.history: list[dict] = []       # one row of metrics per round
        # fixed pairs of held-out items, so test_pair_auc is scored on the same pairs every round
        test_mask = getattr(benchmark, "test_mask", None)
        self.test_mask = test_mask
        self.auc_pairs = None
        if test_mask is not None and test_mask.any():
            self.auc_pairs = sample_pairs(np.flatnonzero(test_mask), 5000, np.random.default_rng(0))
    
    def full_recluster(self, rounds: int):
        self.eval.setup_data(self.benchmark)
        current_embeddings = self.benchmark.embeddings

        best_metrics = {'test_f1': -1,
                        'f1': -1}
        best_iteration = {'test_f1': -1,
                        'f1': -1,
                        'round': 1}

        for round in tqdm(range(rounds)):
            current_embeddings, metrics = self.recluster_round(current_embeddings)
            if self.test_mask is not None:
                if metrics['test_f1'] > best_metrics['test_f1']:
                    best_metrics = metrics
                if metrics['test_f1']**metrics['round'] > best_iteration['test_f1']**best_iteration['round']:
                    best_iteration = metrics
            else:
                if metrics['f1'] > best_metrics['f1']:
                    best_metrics = metrics
                if metrics['f1']**metrics['round'] > best_iteration['f1']**best_iteration['round']:
                    best_iteration = metrics
        return metrics, best_metrics, best_iteration

    def recluster_round(self, current_embeddings: NDArray[np.float64]) -> tuple[NDArray[np.float64], dict]:
        # collect suggestion and prepare eval data
        pred_labels = self.clusterer.run(current_embeddings)
        self.eval.update(pred_labels, current_embeddings, self.benchmark)

        # training loop
        losses = self.adapter.learn(self.eval)

        # recluster and evaluate new clusters
        new_embeddings = self.adapter.predict(self.benchmark)
        new_pred_labels = self.clusterer.run(new_embeddings)

        # log results
        metrics = self.log_results(new_pred_labels, losses, new_embeddings)

        return new_embeddings, metrics

    def log_results(self, pred_labels: NDArray[np.int_], losses: Optional[list[float]], embeddings: NDArray[np.float64]):
        # please feel more than welcome to add to this
        row = {"round": len(self.history) + 1,
               **all_metrics(embeddings, pred_labels, self.benchmark.true_labels,
                             getattr(self.benchmark, "test_mask", None), self.auc_pairs),
               "loss": float(losses[-1]) if losses else float("nan")}
        self.history.append(row)
        print(f"Round {row['round']}: ARI {row['ari']:.3f} | test ARI {row['test_ari']:.3f} | "
              f"test pair AUC {row['test_pair_auc']:.3f} | pair P/R {row['pair_precision']:.3f}/{row['pair_recall']:.3f} | "
              f"homog/compl {row['homogeneity']:.3f}/{row['completeness']:.3f} | "
              f"Jay F1/P/R {row['f1']:.3f}/{row['precision']:.3f}/{row['recall']:.3f} | loss {row['loss']:.4f}")
        return row
