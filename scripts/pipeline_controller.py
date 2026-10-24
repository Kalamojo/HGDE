"""
File: pipeline_controller.py
Author: Chris Jien Zhang
Date: 2026-10-15
Course: CMSC473
Description:
    Master orchestrator for the interactive constrained clustering pipeline.
    Executes the active learning feedback loop by routing LanceDB embeddings
    through a projection model, clustering the output, and processing
    SyntheticUser constraints.

Usage:
    Run from the terminal at the repository root:
    python scripts/pipeline_controller.py --accuracy 0.95 --max_moves 50

    Baseline (identity projection, i.e. plain clustering, for comparison):
    python scripts/pipeline_controller.py --max_moves 50 --baseline --tag baseline
"""

import sys
import json
import argparse
import logging
from pathlib import Path
import numpy as np

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

    def train(self, new_constraint, embeddings=None, labels=None):
        self.constraints.append(new_constraint)
        logger.debug(f"Baseline model recorded constraint. Total constraints: {len(self.constraints)}")

# --- MASTER PIPELINE CLASS ---
class ActiveLearningPipeline:
    """
    Orchestrates the constrained clustering loop. Loosely coupled to allow
    any projection model or clustering algorithm to be swapped in cleanly.

    Interfaces:
        projection_model.transform(embeddings) -> embeddings
        projection_model.train(constraint, embeddings, predicted_labels)
        clustering_algo.fit_predict(embeddings) -> labels
    """
    def __init__(self, projection_model, clustering_algo, ground_truth, target_accuracy, max_moves, seed=0):
        self.projection_model = projection_model
        self.clustering_algo = clustering_algo
        self.user = SyntheticUser(ground_truth, seed=seed)
        self.target_accuracy = target_accuracy
        self.max_moves = max_moves

    def run(self, initial_embeddings):
        """
        Executes the primary active learning loop until convergence criteria are met.
        """
        logger.info("Starting active learning loop...")
        current_embeddings = initial_embeddings

        while self.user.move_counter <= self.max_moves:
            # 1. Project embeddings (Identity filter on first pass)
            current_embeddings = self.projection_model.transform(initial_embeddings)

            # 2. Unsupervised clustering
            predicted_clusters = self.clustering_algo.fit_predict(current_embeddings)

            # 3. User evaluates clusters
            current_accuracy = self.user.evaluate_clusters(predicted_clusters)
            logger.info(f"Move {self.user.move_counter}: ARI = {current_accuracy:.4f}")

            # 4. Check convergence criteria
            if current_accuracy >= self.target_accuracy:
                logger.info("Target accuracy reached. Halting simulation.")
                break
            if self.user.move_counter == self.max_moves:
                logger.warning("Maximum moves reached without hitting target accuracy.")
                break

            # 5. User performs a move to correct an error
            new_constraint = self.user.generate_next_move(predicted_clusters)
            if new_constraint is None:
                logger.info("No further errors identified. Halting simulation.")
                break

            logger.info(f"User issued feedback: {new_constraint['type']} between items {new_constraint['item_1']} and {new_constraint['item_2']}")

            # 6. Train the projection model using the feedback. It also gets the ORIGINAL
            #    embeddings and the current clustering, which its loss needs.
            self.projection_model.train(new_constraint, initial_embeddings, predicted_clusters)

        return self.user.history

if __name__ == "__main__":
    from datetime import datetime

    parser = argparse.ArgumentParser(description="Run the interactive constrained clustering pipeline.")
    parser.add_argument("--accuracy", type=float, default=0.95, help="Target ARI threshold.")
    parser.add_argument("--max_moves", type=int, default=20, help="Maximum allowed constraints.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the synthetic user and model init.")
    parser.add_argument("--baseline", action="store_true", help="Use the identity projection (no learning).")
    parser.add_argument("--tag", type=str, default="", help="Label added to output file names.")
    parser.add_argument("--vector_column", type=str, default=None,
                        help="LanceDB vector column (e.g. image_vector). Must match the ground truth.")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per constraint.")
    parser.add_argument("--lr", type=float, default=0.005, help="Projection model learning rate.")
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
    load_kwargs = {"db_path": str(db_path)}
    if args.vector_column:
        load_kwargs["vector_column"] = args.vector_column
    real_embeddings = load_lancedb_embeddings(**load_kwargs)

    ground_truth_path = repo_root / "data" / "ground_truth_visual.npy"

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
    logger.info(f"Embeddings {real_embeddings.shape}, ground truth classes: {n_clusters}, "
                f"labels: {np.unique(real_ground_truth)[:10]}{'...' if n_clusters > 10 else ''}")

    # Instantiate the modules and run the pipeline
    from scripts.adapters import ClusteringAdapter

    clust_algo = ClusteringAdapter(n_clusters=n_clusters, algorithm="kmeans", random_state=123)
    if args.baseline:
        logger.info("BASELINE run: identity projection (no learning).")
        proj_model = StubProjectionModel()
    else:
        from scripts.adapters import ProjectionAdapter
        proj_model = ProjectionAdapter(dim=real_embeddings.shape[1], lr=args.lr,
                                       epochs=args.epochs, seed=args.seed)

    pipeline = ActiveLearningPipeline(
        projection_model=proj_model,
        clustering_algo=clust_algo,
        ground_truth=real_ground_truth,
        target_accuracy=args.accuracy,
        max_moves=args.max_moves,
        seed=args.seed
    )

    final_history = pipeline.run(real_embeddings)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    tag = f"{args.tag}-" if args.tag else ""

    # Save raw history so runs (e.g. baseline vs. learned) can be compared afterwards.
    history_path = results_dir / f"history-{tag}{timestamp}.json"
    with open(history_path, "w") as f:
        json.dump({"args": vars(args),
                   "moves": [int(m) for m in final_history["moves"]],
                   "ari_scores": [float(a) for a in final_history["ari_scores"]]}, f, indent=2)
    logger.info(f"Saved history to {history_path}")

    try:
        from scripts.visualize_metrics import plot_convergence
        graph_output = results_dir / f"final-convergence-{tag}{timestamp}.png"
        plot_convergence(final_history, save_path=str(graph_output))
    except ImportError:
        logger.warning("Visualization module not found. Skipping graph generation.")