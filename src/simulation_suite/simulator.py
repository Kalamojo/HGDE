"""
File: simulator.py
Author: Chris Jien Zhang
Date: 2026-09-22
Course: CMSC473
Description:
    This module contains the SyntheticUser class, which acts as the automated 
    simulation engine for the interactive constrained clustering pipeline. 
    It evaluates unsupervised clustering outputs against a hidden ground truth 
    and generates constraint tuples to iteratively warp the embedding space.

Usage:
    from simulation.simulator import SyntheticUser
    user = SyntheticUser(ground_truth_labels=[0, 0, 1, 1])
    ari = user.evaluate_clusters(predicted_labels=[0, 1, 1, 1])
    constraint = user.generate_next_move(predicted_labels=[0, 1, 1, 1])
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score

class SyntheticUser:
    """
    Simulates a human user providing feedback to a clustering model.
    Maintains a decoupled architecture by relying only on standard array inputs,
    making it agnostic to the specific foundation model or clustering algorithm used.
    """

    def __init__(self, ground_truth_labels):
        """
        Initializes the synthetic user with the hidden target rules.

        Args:
            ground_truth_labels (list or numpy.ndarray): The perfect cluster 
            assignments expected at the end of the simulation.
        """
        self.ground_truth_labels = np.array(ground_truth_labels)
        
        # Tracks simulation metrics for final convergence benchmark graphs
        self.history = {
            "moves": [],
            "ari_scores": []
        }
        self.move_counter = 0

    def evaluate_clusters(self, predicted_labels):
        """
        Calculates the Adjusted Rand Index (ARI) of the current clusters and logs it.

        Args:
            predicted_labels (list or numpy.ndarray): The output from the baseline clustering.

        Returns:
            float: The ARI score ranging from -1.0 to 1.0.
        """
        predicted_array = np.array(predicted_labels)
        ari_score = adjusted_rand_score(self.ground_truth_labels, predicted_array)
        
        self.history["moves"].append(self.move_counter)
        self.history["ari_scores"].append(ari_score)
        
        return ari_score

    def generate_next_move(self, predicted_labels):
        """
        Identifies two items that share a ground-truth cluster but were 
        incorrectly separated by the predictive algorithm.

        Args:
            predicted_labels (list or numpy.ndarray): The current flawed clusters.

        Returns:
            tuple: (index_1, index_2) representing a "must-link" constraint, 
            or None if the clusters perfectly match the ground truth.
        """
        predicted_array = np.array(predicted_labels)
        unique_true_classes = np.unique(self.ground_truth_labels)
        
        for true_class in unique_true_classes:
            # Locate all data points belonging to this specific target cluster
            items_in_class = np.where(self.ground_truth_labels == true_class)[0]
            
            if len(items_in_class) < 2:
                continue
                
            predicted_for_these_items = predicted_array[items_in_class]
            
            # A mismatch exists if these items were split into multiple predicted clusters
            if len(np.unique(predicted_for_these_items)) > 1:
                first_label = predicted_for_these_items[0]
                
                # Locate the index of the first item separated from the group
                mismatch_relative_idx = np.where(predicted_for_these_items != first_label)[0][0]
                
                idx_1 = items_in_class[0]
                idx_2 = items_in_class[mismatch_relative_idx]
                
                self.move_counter += 1
                
                return (int(idx_1), int(idx_2))
                
        # Returns None when no errors remain, signaling the simulation is complete
        return None