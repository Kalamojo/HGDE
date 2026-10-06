"""
File: pipeline_controller.py
Author: Chris Jien Zhang
Date: 2026-10-06
Course: CMSC473
Description:
    Master controller for the interactive constrained clustering pipeline.
    This script orchestrates the active learning loop by passing data between 
    the projection model, the clustering algorithm, and the synthetic user.
    Utilizes placeholder stubs for incomplete modules.
    Upgraded with a command-line interface (CLI) for hyperparameter tuning and
    formal logging to track simulation metrics across runs.

Usage:
    Run from the terminal at the repository root.
    
    Basic execution with default parameters:
    python scripts/pipeline_controller.py
    
    Execution with custom hyperparameters:
    python scripts/pipeline_controller.py --accuracy 0.99 --max_moves 50
"""

import sys
import argparse
import logging
from pathlib import Path
import numpy as np

# Dynamically append the repository root to the system path
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
sys.path.append(str(repo_root))

# Python can now successfully locate the src folder from the root
try:
    from src.simulation_suite.simulator import SyntheticUser
except ImportError:
    from simulator import SyntheticUser

# Initialize a module-level logger
logger = logging.getLogger(__name__)

# Stubs for missing team modules
def stub_projection_model(embeddings, master_constraints):
    """
    Placeholder for John's PyTorch model.
    Currently does no math and returns the original embeddings.
    """
    return embeddings

def stub_clustering_algorithm(embeddings):
    """
    Placeholder for Jay's K-Means clustering.
    Currently returns a hardcoded flawed cluster to test the loop.
    """
    # Returns an array with an intentional mistake (index 1 is wrong)
    return np.array([0, 1, 2, 2, 3, 3])

def run_active_learning_loop(initial_embeddings, ground_truth, target_accuracy=0.95, max_moves=20):
    """
    Executes the autonomous feedback loop until target accuracy is reached 
    or the maximum number of allowed moves is exceeded.

    Args:
        initial_embeddings (numpy.ndarray): The raw multimodal data features.
        ground_truth (list): The perfect cluster assignments for the user oracle.
        target_accuracy (float): The ARI threshold to stop the simulation.
        max_moves (int): Failsafe limit to prevent infinite loops during testing.

    Returns:
        dict: The simulation history containing 'moves' and 'ari_scores'.
    """
    user = SyntheticUser(ground_truth)
    master_constraints = []
    current_embeddings = initial_embeddings

    logger.info("Starting simulation loop...")

    for move_count in range(max_moves + 1):
        # Step 1 & 2: Project embeddings and generate clusters
        current_embeddings = stub_projection_model(initial_embeddings, master_constraints)
        predicted_clusters = stub_clustering_algorithm(current_embeddings)
        
        # Step 3 & 4: Evaluate and check stopping condition
        current_accuracy = user.evaluate_clusters(predicted_clusters)
        logger.info(f"Move {move_count}: ARI = {current_accuracy:.4f}")
        
        if current_accuracy >= target_accuracy:
            logger.info("Target accuracy reached! Stopping simulation.")
            break
            
        if move_count == max_moves:
            logger.warning("Max moves reached without hitting target accuracy.")
            break

        # Step 5: Synthetic user performs a feedback action
        new_constraint = user.generate_next_move(predicted_clusters)
        
        if new_constraint is None:
            logger.info("No more errors found by user. Stopping simulation.")
            break
            
        # Step 6: Update the master list for the next loop iteration
        master_constraints.append(new_constraint)
        
        # Use debug level for detailed per-move constraint tracking
        logger.debug(f"Added constraint: {new_constraint}. Total constraints: {len(master_constraints)}")

    return user.history

if __name__ == "__main__":
    from datetime import datetime
    
    # Set up the command-line interface
    parser = argparse.ArgumentParser(description="Run the interactive constrained clustering pipeline.")
    parser.add_argument("--accuracy", type=float, default=0.95, help="Target Adjusted Rand Index (ARI) to stop the simulation.")
    parser.add_argument("--max_moves", type=int, default=20, help="Maximum number of constraints the user is allowed to generate.")
    args = parser.parse_args()

    # Ensure the results directory exists before configuring the file logger
    results_dir = repo_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure formal logging to output to both the terminal and a permanent log file
    log_file_path = results_dir / "simulation.log"
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[
            logging.FileHandler(str(log_file_path)),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Import the graphing function dynamically
    try:
        from scripts.visualize_metrics import plot_convergence
    except ImportError:
        try:
            from visualize_metrics import plot_convergence
        except ImportError:
            plot_convergence = None

    # Mock data to run the file directly
    dummy_embeddings = np.array([[0.1, 0.2], [0.15, 0.25], [0.8, 0.9], [0.85, 0.95], [0.4, 0.4], [0.45, 0.45]])
    dummy_ground_truth = [0, 0, 1, 1, 2, 2]
    
    logger.info(f"Configuration: Target Accuracy = {args.accuracy}, Max Moves = {args.max_moves}")
    
    # Execute the master loop with CLI parameters
    final_history = run_active_learning_loop(
        dummy_embeddings, 
        dummy_ground_truth, 
        target_accuracy=args.accuracy, 
        max_moves=args.max_moves
    )
    logger.info(f"Final Logged History: {final_history}")
    
    # Generate a unique timestamp string formatted with hyphens
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    
    # Dynamically resolve a portable path for the final graph output
    graph_output_path = results_dir / f"final-convergence-{timestamp}.png"
    
    # Generate and save the graph if the module is available
    if plot_convergence:
        logger.info(f"Saving convergence graph to {graph_output_path}")
        plot_convergence(final_history, save_path=str(graph_output_path))
    else:
        logger.warning("Graphing module not found. Skipping visualization.")