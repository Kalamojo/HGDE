import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm
from typing import Optional
import csv
import json
import re
from pathlib import Path
from src.utils.adapters import EmbeddingAdapter
from src.utils.data.benchmarks import Benchmark
from src.utils.eval import ClusterEval
from src.utils.metrics import all_metrics, sample_pairs


class ClusterState:
    def __init__(self, benchmark: Benchmark, eval: ClusterEval, adapter: EmbeddingAdapter, verbose: bool):
        self.benchmark = benchmark
        self.eval = eval
        self.adapter = adapter
        self.history: list[dict] = []       # one row of metrics per round
        # fixed pairs of held-out items, so test_pair_auc is scored on the same pairs every round
        test_mask = getattr(benchmark, "test_mask", None)
        self.test_mask = test_mask
        self.auc_pairs = None
        self.verbose = verbose
        self.baseline_embeddings = None
        self.baseline_clusters = None
        if test_mask is not None and test_mask.any():
            self.auc_pairs = sample_pairs(np.flatnonzero(test_mask), 5000, np.random.default_rng(0))
    
    def full_recluster(self, rounds: int):
        self.eval.setup_data(self.benchmark)
        self.adapter.setup_predictions(self.benchmark)
        self.baseline_embeddings = self.adapter.predictions.embeddings.copy()
        self.baseline_clusters = self.adapter.predictions.labels.copy()

        best_metrics = {'test_pairwise_f1': -1,
                        'pair_f1': -1}
        best_iteration = {'test_pairwise_f1': -1,
                        'pair_f1': -1,
                        'round': 1}

        for round in tqdm(range(rounds)):
            metrics = self.recluster_round()
            if self.test_mask is not None:
                if metrics['test_pairwise_f1'] > best_metrics['test_pairwise_f1']:
                    best_metrics = metrics
                if metrics['test_pairwise_f1']**metrics['round'] > best_iteration['test_pairwise_f1']**best_iteration['round']:
                    best_iteration = metrics
            else:
                if metrics['pair_f1'] > best_metrics['pair_f1']:
                    best_metrics = metrics
                if metrics['pair_f1']**((metrics['round']+1)/2) > best_iteration['pair_f1']**((best_iteration['round']+1)/2):
                    best_iteration = metrics
        return metrics, best_metrics, best_iteration

    def save_visualization_data(self, output_dir: Path, run_settings: Optional[dict] = None) -> Path:
        """Save aligned baseline/final embeddings and labels in a PCA-friendly bundle."""
        if self.baseline_embeddings is None or self.baseline_clusters is None:
            raise RuntimeError("Run full_recluster before saving visualization data")
        final_embeddings = self.adapter.predictions.embeddings
        final_clusters = self.adapter.predictions.labels
        if final_embeddings is None or final_clusters is None:
            raise RuntimeError("The adapter has no final embedding predictions")

        def slug(value):
            return re.sub(r"[^A-Za-z0-9._-]+", "-", str(value)).strip("-") or "unknown"

        vector = self.benchmark.vector_col
        vector_name = "+".join(vector) if isinstance(vector, list) else vector
        label_values = "-".join(slug(value) for value in self.benchmark.unique_labels)
        benchmark_name = "_".join((slug(self.benchmark.table_name), slug(vector_name),
                                   slug(self.benchmark.label_col), label_values))
        output_dir = Path(output_dir) / benchmark_name
        output_dir.mkdir(parents=True, exist_ok=True)
        np.save(output_dir / "baseline_embeddings.npy", self.baseline_embeddings)
        np.save(output_dir / "transformed_embeddings.npy", final_embeddings)

        fields = ["row_index", "item_id", "true_label", "cluster_before", "cluster_after", "is_test"]
        for metadata_row in self.benchmark.metadata:
            for key in ("name", "dex_num"):
                if key in metadata_row and key not in fields:
                    fields.append(key)
        with (output_dir / "assignments.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            for i, metadata_row in enumerate(self.benchmark.metadata):
                row = {"row_index": i, "item_id": self.benchmark.ids[i],
                       "true_label": self.benchmark.true_labels[i],
                       "cluster_before": self.baseline_clusters[i],
                       "cluster_after": final_clusters[i],
                       "is_test": bool(self.benchmark.test_mask[i])}
                row.update({key: metadata_row.get(key, "") for key in fields if key in ("name", "dex_num")})
                writer.writerow({key: str(value) for key, value in row.items()})

        manifest = {
            "format": "hgde-embedding-visualization-v1",
            "dataset": self.benchmark.table_name,
            "embedding_column": vector_name,
            "label_column": self.benchmark.label_col,
            "n_items": len(self.benchmark.true_labels),
            "embedding_dim": int(self.baseline_embeddings.shape[1]),
            "run_settings": run_settings or {},
            "files": {"baseline": "baseline_embeddings.npy",
                      "transformed": "transformed_embeddings.npy",
                      "assignments": "assignments.csv"},
        }
        (output_dir / "visualization.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        return output_dir

    def recluster_round(self) -> dict:
        # collect suggestion and prepare eval data
        self.eval.update(self.adapter.predictions, self.benchmark)

        # training loop
        losses = self.adapter.learn(self.eval)

        # recluster and evaluate new clusters
        new_pred_labels = self.adapter.predict(self.benchmark)
        new_embeddings = self.adapter.predictions.embeddings

        # log results
        metrics = self.log_results(new_pred_labels, losses, new_embeddings)

        return metrics

    def log_results(self, pred_labels: NDArray[np.int_], losses: Optional[list[float]], embeddings: Optional[NDArray[np.float64]]):
        # please feel more than welcome to add to this
        row = {"round": len(self.history) + 1,
               **all_metrics(embeddings, pred_labels, self.benchmark.true_labels,
                             getattr(self.benchmark, "test_mask", None), self.auc_pairs),
               "loss": float(losses[-1]) if losses else float("nan")}
        self.history.append(row)
        if self.verbose:
          print(f"Round {row['round']}: ARI {row['ari']:.3f} | test ARI {row['test_ari']:.3f} | "
                f"test pair AUC {row['test_pair_auc']:.3f} | pair P/R {row['pair_precision']:.3f}/{row['pair_recall']:.3f} | "
                f"Jay F1/P/R {row['f1']:.3f}/{row['precision']:.3f}/{row['recall']:.3f} | loss {row['loss']:.4f}")
        return row
