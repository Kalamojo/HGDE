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
"""

import sys
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

# --- STUBS FOR TEAMMATES MODULES ---
class StubProjectionModel:
    """Placeholder for John's PyTorch model."""
    def __init__(self):
        self.constraints = []
        
    def transform(self, embeddings):
        """Passes data through the identity filter on the first run."""
        return embeddings
        
    def train(self, new_constraint):
        """Simulates training the model to satisfy the new constraint."""
        self.constraints.append(new_constraint)
        logger.debug(f"Model trained on new constraint. Total constraints: {len(self.constraints)}")

class StubClusteringAlgorithm:
    """Placeholder for Jay's K-Means clustering."""
    def fit_predict(self, embeddings):
        num_items = len(embeddings)
        dummy_clusters = np.zeros(num_items, dtype=int)
        if num_items > 1:
            dummy_clusters[1] = 1 
        return dummy_clusters

# --- MASTER PIPELINE CLASS ---
class ActiveLearningPipeline:
    """
    Orchestrates the constrained clustering loop. Loosely coupled to allow 
    any projection model or clustering algorithm to be swapped in cleanly.
    """
    def __init__(self, projection_model, clustering_algo, ground_truth, target_accuracy, max_moves):
        self.projection_model = projection_model
        self.clustering_algo = clustering_algo
        self.user = SyntheticUser(ground_truth)
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
            
            # 6. Train the projection model using the collected feedback
            self.projection_model.train(new_constraint)

        return self.user.history

if __name__ == "__main__":
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Run the interactive constrained clustering pipeline.")
    parser.add_argument("--accuracy", type=float, default=0.95, help="Target ARI threshold.")
    parser.add_argument("--max_moves", type=int, default=20, help="Maximum allowed constraints.")
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
    real_embeddings = load_lancedb_embeddings(db_path=str(db_path))
    
    ground_truth_path = repo_root / "data" / "ground_truth_visual.npy"
    
    if real_embeddings is None or not ground_truth_path.exists():
        logger.error("Data loading failed. Check LanceDB and ground truth paths.")
        sys.exit(1)
        
    real_ground_truth = np.load(str(ground_truth_path))

    # Instantiate the modules and run the pipeline
    proj_model = StubProjectionModel()
    clust_algo = StubClusteringAlgorithm()
    
    pipeline = ActiveLearningPipeline(
        projection_model=proj_model,
        clustering_algo=clust_algo,
        ground_truth=real_ground_truth,
        target_accuracy=args.accuracy,
        max_moves=args.max_moves
    )
    
    final_history = pipeline.run(real_embeddings)
    
    try:
        from scripts.visualize_metrics import plot_convergence
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        graph_output = results_dir / f"final-convergence-{timestamp}.png"
        plot_convergence(final_history, save_path=str(graph_output))
    except ImportError:
        logger.warning("Visualization module not found. Skipping graph generation.")