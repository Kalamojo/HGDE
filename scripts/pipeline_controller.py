"""
File: pipeline_controller.py
Author: Chris Jien Zhang
Date: 2026-09-29
Course: CMSC473
Description:
    Master controller for the interactive constrained clustering pipeline.
    This script orchestrates the active learning loop by passing data between 
    the projection model, the clustering algorithm, and the synthetic user.
    Currently utilizes placeholder stubs for incomplete modules.

Usage:
    python scripts/pipeline_controller.py
"""

import sys
from pathlib import Path
import numpy as np
from datetime import datetime

# Dynamically append the repository root to the system path
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent
sys.path.append(str(repo_root))

# Python can now successfully locate the src folder from the root
from src.simulation_suite.simulator import SyntheticUser

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
    #item 0 is in group 0, item 1 is in group 1, item 2 is in group 2, etc.
    return np.array([0, 1, 2, 2, 3, 3])

def run_active_learning_loop(initial_embeddings, ground_truth, target_accuracy=0.95, max_moves=20):
    """
    Executes the autonomous feedback loop until target accuracy is reached 
    or the maximum number of allowed moves is exceeded.

    Args:
        initial_embeddings (numpy.ndarray): The raw multimodal data features.
        ground_truth (list): The perfect cluster assignments for the user oracle.
        target_accuracy (float): The ARI threshold to stop the simulation (e.g., 0.95).
        max_moves (int): Failsafe limit to prevent infinite loops during testing.

    Returns:
        dict: The simulation history containing 'moves' and 'ari_scores'.
    """
    user = SyntheticUser(ground_truth)
    master_constraints = []
    current_embeddings = initial_embeddings

    print("Starting simulation loop...")

    for move_count in range(max_moves + 1):
        # Step 1 & 2: Project embeddings and generate clusters
        current_embeddings = stub_projection_model(initial_embeddings, master_constraints)
        predicted_clusters = stub_clustering_algorithm(current_embeddings)
        
        # Step 3 & 4: Evaluate and check stopping condition
        current_accuracy = user.evaluate_clusters(predicted_clusters)
        print(f"Move {move_count}: ARI = {current_accuracy:.4f}")
        
        if current_accuracy >= target_accuracy:
            print(f"Target accuracy reached! Stopping simulation.")
            break
            
        if move_count == max_moves:
            print(f"Max moves reached without hitting target accuracy.")
            break

        # Step 5: Synthetic user performs a feedback action
        new_constraint = user.generate_next_move(predicted_clusters)
        
        if new_constraint is None:
            print("No more errors found by user. Stopping simulation.")
            break
            
        # Step 6: Update the master list for the next loop iteration
        master_constraints.append(new_constraint)
        print(f"Added constraint: {new_constraint}. Total constraints: {len(master_constraints)}")

    return user.history

if __name__ == "__main__":
    # Import the graphing function dynamically
    try:
        from scripts.visualize_metrics import plot_convergence
    except ImportError:
        from visualize_metrics import plot_convergence

    # Mock data to run the file directly
    dummy_embeddings = np.array([[0.1, 0.2], [0.15, 0.25], [0.8, 0.9], [0.85, 0.95], [0.4, 0.4], [0.45, 0.45]])
    dummy_ground_truth = [0, 0, 1, 1, 2, 2] #item 0 is in group 0, item 1 is in group 0, item 2 is in group 1, etc.
    #the synthetic user only cares that items that are supposed to be together are together, and is blind to actual group assignments
    #if truth is [0, 0], then [2, 2] is ok but [0, 1] isn't

    print("\nStarting learning loop...\n")
    final_history = run_active_learning_loop(dummy_embeddings, dummy_ground_truth)
    print("\nFinal Logged History:", final_history)

    # Generate a unique timestamp string formatted with hyphens
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Dynamically resolve a portable path for the final graph output
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent
    graph_output_path = repo_root / "results" / f"final-convergence-{timestamp}.png"
    
    # Generate and save the graph
    plot_convergence(final_history, save_path=str(graph_output_path))