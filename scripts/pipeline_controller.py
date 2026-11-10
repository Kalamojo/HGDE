"""
File: pipeline_controller.py
Author: Chris Jien Zhang
Date: 2026-10-15
Course: CMSC473
Description:
    Master orchestrator for the interactive constrained clustering pipeline.
    Executes the active learning feedback loop by routing LanceDB embeddings
    through a projection model, clustering the output, and processing
    SyntheticUser feedback. Each "move" is a batch of constraints (default: 3 Must-Links).

    Model: GravityWarp (src/gravity_warp.py), the current best, refit every move on the original
    embeddings. Clustering: Jay's KMeans (clustering.py) on the first move, then warm-started from
    the previous move's clusters.

Usage:
    Run from the terminal at the repository root:
    python scripts/pipeline_controller.py --max_moves 150 --tag warp

    Baseline (identity projection, i.e. plain clustering, for comparison):
    python scripts/pipeline_controller.py --baseline --max_moves 50 --tag baseline

    Compare runs by number of CONSTRAINTS (n_constraints in the history json), not by move,
    whenever the batch size differs.
"""

import sys
import json
import argparse
import logging
from pathlib import Path
import numpy as np
from sklearn.cluster import KMeans

current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
sys.path.append(str(repo_root))

from src.simulation_suite.simulator import SyntheticUser
from scripts.ingest_lancedb import load_lancedb_embeddings

logger = logging.getLogger(__name__)

# --- CONTROL: identity projection (kept so a run can be compared against plain clustering) ---
class StubProjectionModel:
    """Identity projection. Never changes the embeddings, so results = clustering alone."""
    def __init__(self):
        self.constraints = []

    def transform(self, embeddings):
        return embeddings

    def train(self, new_constraints, embeddings=None, labels=None):
        self.constraints.extend(new_constraints)

class WarmStartClustering:
    """KMeans that restarts each move from the previous move's clusters.

    The first call uses the base clusterer unchanged. Every later call starts KMeans at the
    centroids of the PREVIOUS clustering, recomputed in the CURRENT (re-projected) space, and runs
    it to convergence. Clusters keep their identity from move to move instead of being re-drawn
    from scratch: this removed the ~0.05-0.1 move-to-move ARI jitter and raised the warp's final
    ARI from 0.595 to 0.729 (seed 0, 300 constraints)."""
    def __init__(self, base, n_clusters, max_iter=300):
        self.base = base
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.prev = None

    def fit_predict(self, embeddings):
        X = np.asarray(embeddings, dtype=np.float64)
        if self.prev is None:
            labels = np.asarray(self.base.fit_predict(embeddings))
        else:
            # an emptied cluster is re-seeded at an arbitrary point (item c)
            init = np.vstack([X[self.prev == c].mean(axis=0) if np.any(self.prev == c) else X[c]
                              for c in range(self.n_clusters)])
            labels = KMeans(self.n_clusters, init=init, n_init=1, max_iter=self.max_iter).fit_predict(X)
        self.prev = labels
        return labels

# --- MASTER PIPELINE CLASS ---
class ActiveLearningPipeline:
    """
    Orchestrates the constrained clustering loop. Loosely coupled to allow
    any projection model or clustering algorithm to be swapped in cleanly.

    Interfaces:
        projection_model.transform(embeddings) -> embeddings
        projection_model.train(constraints, embeddings, predicted_labels)   # constraints: a list
        clustering_algo.fit_predict(embeddings) -> labels
    """
    def __init__(self, projection_model, clustering_algo, ground_truth, target_accuracy, max_moves,
                 seed=0, batch_size=3, must_link_only=True, test_fraction=0.2):
        self.projection_model = projection_model
        self.clustering_algo = clustering_algo
        self.user = SyntheticUser(ground_truth, seed=seed, batch_size=batch_size,
                                  must_link_only=must_link_only, test_fraction=test_fraction)
        self.target_accuracy = target_accuracy
        self.max_moves = max_moves

    def run(self, initial_embeddings):
        """
        Executes the primary active learning loop until convergence criteria are met.
        """
        logger.info("Starting active learning loop...")

        while self.user.move_counter <= self.max_moves:
            # 1. Project embeddings (identity before the first constraint)
            current_embeddings = self.projection_model.transform(initial_embeddings)

            # 2. Unsupervised clustering
            predicted_clusters = self.clustering_algo.fit_predict(current_embeddings)

            # 3. User evaluates clusters
            current_accuracy = self.user.evaluate_clusters(predicted_clusters)
            logger.info(f"Move {self.user.move_counter} ({self.user.constraint_counter} constraints): "
                        f"ARI = {current_accuracy:.4f}, test ARI = {self.user.history['test_ari'][-1]:.4f}")

            # 4. Check convergence criteria
            if current_accuracy >= self.target_accuracy:
                logger.info("Target accuracy reached. Halting simulation.")
                break
            if self.user.move_counter == self.max_moves:
                logger.warning("Maximum moves reached without hitting target accuracy.")
                break

            # 5. User performs a move: a batch of corrections
            new_constraints = self.user.generate_next_move(predicted_clusters)
            if not new_constraints:
                logger.info("No further errors identified. Halting simulation.")
                break

            for c in new_constraints:
                logger.info(f"User issued feedback: {c['type']} between items {c['item_1']} and {c['item_2']}")

            # 6. Train the projection model on the whole batch (one refit). It also gets the
            #    ORIGINAL embeddings and the current clustering.
            self.projection_model.train(new_constraints, initial_embeddings, predicted_clusters)
            if getattr(self.projection_model, "stats", None):
                logger.info(f"Model stats: {self.projection_model.stats}")

        return self.user.history

if __name__ == "__main__":
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Run the interactive constrained clustering pipeline.")
    parser.add_argument("--sigma", type=float, default=1.0, help="Warp kernel width (1.0 was best).")
    parser.add_argument("--strength", type=float, default=0.9, help="Warp pull toward the group centroid.")
    parser.add_argument("--batch_size", type=int, default=3, help="Constraints given per move.")
    parser.add_argument("--test_fraction", type=float, default=0.2,
                        help="Fixed random test set the user never gives feedback on; test ARI is scored on it.")
    parser.add_argument("--accuracy", type=float, default=0.95, help="Target ARI threshold.")
    parser.add_argument("--max_moves", type=int, default=100, help="Maximum number of moves (batches).")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the synthetic user and model init.")
    parser.add_argument("--baseline", action="store_true", help="Use the identity projection (no learning).")
    parser.add_argument("--tag", type=str, default="", help="Label added to output file names.")
    parser.add_argument("--vector_column", type=str, default="image_vector",
                        help="LanceDB vector column (e.g. image_vector). Must match the ground truth.")
    parser.add_argument("--ground_truth", type=str, default="ground_truth_visual.npy",
                        help="File in data/ holding the hidden ground-truth labels.")
    args = parser.parse_args()

    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(str(results_dir / "simulation.log")),
            logging.StreamHandler(sys.stdout)
        ]
    )

    logger.info("Initializing data ingestion...")
    db_path = repo_root / "data" / "pokemon_db"
    real_embeddings = load_lancedb_embeddings(db_path=str(db_path), vector_column=args.vector_column)

    ground_truth_path = repo_root / "data" / args.ground_truth

    if real_embeddings is None or not ground_truth_path.exists():
        logger.error("Data loading failed. Check LanceDB and ground truth paths.")
        sys.exit(1)

    real_embeddings = np.asarray(real_embeddings, dtype=np.float32)
    real_ground_truth = np.load(str(ground_truth_path))

    # Sanity checks: row i of the embeddings must be the same item as ground truth[i].
    if real_embeddings.ndim != 2 or len(real_embeddings) != len(real_ground_truth):
        logger.error(f"Shape mismatch: embeddings {real_embeddings.shape} vs ground truth {real_ground_truth.shape}.")
        sys.exit(1)
    if np.any(np.linalg.norm(real_embeddings, axis=1) == 0) or not np.isfinite(real_embeddings).all():
        logger.error("Embeddings contain zero-norm rows or non-finite values.")
        sys.exit(1)

    n_clusters = len(np.unique(real_ground_truth))
    logger.info(f"Embeddings {real_embeddings.shape}, ground truth '{args.ground_truth}': {n_clusters} classes, "
                f"labels: {np.unique(real_ground_truth)[:10]}{'...' if n_clusters > 10 else ''}")
    logger.info(f"User feedback: batches of {args.batch_size} Must-Links")

    # Instantiate the modules and run the pipeline
    from scripts.adapters import ClusteringAdapter

    from scripts.adapters import ClusteringAdapter

    clust_algo = WarmStartClustering(ClusteringAdapter(n_clusters=n_clusters, algorithm="kmeans", random_state=123),
                                     n_clusters)
    if args.baseline:
        logger.info("BASELINE run: identity projection (no learning).")
        proj_model = StubProjectionModel()
    else:
        from src.gravity_warp import GravityWarp
        proj_model = GravityWarp(strength=args.strength, sigma_scale=args.sigma, seed=args.seed)

    pipeline = ActiveLearningPipeline(
        projection_model=proj_model,
        clustering_algo=clust_algo,
        ground_truth=real_ground_truth,
        target_accuracy=args.accuracy,
        max_moves=args.max_moves,
        seed=args.seed,
        batch_size=args.batch_size,
        test_fraction=args.test_fraction,
    )

    final_history = pipeline.run(real_embeddings)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    tag = f"{args.tag}-" if args.tag else ""

    # Save raw history so runs (e.g. baseline vs. learned) can be compared afterwards.
    history_path = results_dir / f"history-{tag}{timestamp}.json"
    with open(history_path, "w") as f:
        json.dump({"args": vars(args),
                   "moves": [int(m) for m in final_history["moves"]],
                   "n_constraints": [int(n) for n in final_history["n_constraints"]],
                   "ari_scores": [float(a) for a in final_history["ari_scores"]],
                   "test_ari": [float(a) for a in final_history["test_ari"]]}, f, indent=2)
    logger.info(f"Saved history to {history_path}")

    try:
        from scripts.visualize_metrics import plot_convergence
        graph_output = results_dir / f"final-convergence-{tag}{timestamp}.png"
        plot_convergence(final_history, save_path=str(graph_output))
    except ImportError:
        logger.warning("Visualization module not found. Skipping graph generation.")
