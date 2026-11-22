"""
File: visualize_metrics.py
Author: Chris Jien Zhang
Date: 2026-09-29
Course: CMSC473
Description:
    Generates and saves convergence graphs tracking user moves versus 
    Adjusted Rand Index (ARI) accuracy. Designed to evaluate the performance 
    of the interactive constrained clustering pipeline.

Usage:
    Can be imported into the master controller to automatically generate 
    graphs at the end of a simulation run, or run independently for testing.
    
    Example:
        from scripts.visualize_metrics import plot_convergence
        plot_convergence(history_dict, save_path="results/convergence.png")
"""

import matplotlib.pyplot as plt
import os
from pathlib import Path

def plot_convergence(history, save_path=None):
    """
    Plots a line graph of ARI scores over the number of user constraints.

    Args:
        history (dict): Dictionary containing 'moves' and 'ari_scores' lists.
        save_path (str, optional): The file path where the graph should be saved. 
                                   If None, the graph is displayed interactively.

    Raises:
        ValueError: If the history dictionary is missing required keys or 
                    if the data arrays are not parallel.
    """
    # Validate data integrity before attempting to plot
    if "moves" not in history or "ari_scores" not in history:
        raise ValueError("History dictionary must contain 'moves' and 'ari_scores' keys.")
        
    moves = history["moves"]
    scores = history["ari_scores"]
    
    if len(moves) != len(scores):
        raise ValueError("The 'moves' and 'ari_scores' lists must be of equal length.")

    # Initialize the plot figure with a standard size
    plt.figure(figsize=(8, 5))
    
    # Plot the data using a line with circular markers for individual data points
    plt.plot(moves, scores, marker='o', markersize=3, linestyle='-', color='b', label='ARI (all items)')
    if history.get("test_ari"):
        plt.plot(moves, history["test_ari"], marker='o', markersize=3, linestyle='-', color='tab:orange',
                 label='Test ARI (items the user never gives feedback on)')

    # Configure graph labels and title
    plt.title('Clustering Convergence: User Effort vs Accuracy', fontsize=14)
    plt.xlabel('Move (batch of constraints)', fontsize=12)
    plt.ylabel('Adjusted Rand Index (ARI)', fontsize=12)

    # Set y-axis limits to represent standard ARI bounds (0.0 to 1.0)
    # A slight padding up to 1.05 ensures the top line is clearly visible
    plt.ylim(-0.05, 1.05)

    # Integer ticks only (partial moves do not exist); let matplotlib thin them for long runs
    plt.gca().xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()

    # Handle output routing based on provided parameters
    if save_path:
        # Resolve the directory path to ensure it exists before saving
        output_file = Path(save_path).resolve()
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        plt.savefig(output_file)
        print(f"Graph successfully saved to: {output_file}")
    else:
        plt.show()

if __name__ == "__main__":
    # Mock data to validate the graphing logic independently
    dummy_history = {
        "moves": [0, 1, 2, 3, 4],
        "ari_scores": [0.45, 0.60, 0.75, 0.90, 1.0]
    }
    
    # Dynamically resolve a portable test output path
    current_dir = Path(__file__).resolve().parent
    repo_root = current_dir.parent
    test_output_path = repo_root / "test-convergence-graph.png"
    
    plot_convergence(dummy_history, save_path=str(test_output_path))