"""
File: simulator.py
Author: Chris Jien Zhang
Date: 2026-10-15
Course: CMSC473
Description:
    Stateful simulation engine for the interactive constrained clustering pipeline.
    Evaluates unsupervised clustering outputs against a hidden ground truth and 
    generates Pairwise Constraints (Must-Link and Cannot-Link) to iteratively 
    train the projection model.

Usage:
    from simulation_suite.simulator import SyntheticUser
    user = SyntheticUser(ground_truth_labels=actual_labels)
    ari = user.evaluate_clusters(predicted_labels)
    constraint = user.generate_next_move(predicted_labels)
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
import logging

logger = logging.getLogger(__name__)

class SyntheticUser:
    """
    Simulates a human user providing pairwise constraint feedback.
    Maintains memory of previously issued constraints to ensure the pipeline
    constantly explores new corrective actions.
    """

    def __init__(self, ground_truth_labels):
        """
        Initializes the synthetic user with the hidden target rules and empty state.

        Args:
            ground_truth_labels (list or numpy.ndarray): The perfect cluster assignments.
        """
        self.ground_truth = np.array(ground_truth_labels)
        self.history = {
            "moves": [],
            "ari_scores": []
        }
        self.move_counter = 0
        
        # State tracking to prevent duplicate constraints
        self.issued_constraints = set()

    def evaluate_clusters(self, predicted_labels):
        """
        Calculates the Adjusted Rand Index (ARI) of the current clusters.

        Args:
            predicted_labels (list or numpy.ndarray): The model's current cluster output.

        Returns:
            float: The ARI score ranging from -1.0 to 1.0.
        """
        ari_score = adjusted_rand_score(self.ground_truth, np.array(predicted_labels))
        self.history["moves"].append(self.move_counter)
        self.history["ari_scores"].append(ari_score)
        return ari_score

    def generate_next_move(self, predicted_labels):
        """
        Scans the predicted clusters against the ground truth to find the first 
        unissued error and generates a corrective Pairwise Constraint.

        Args:
            predicted_labels (list or numpy.ndarray): The flawed clusters.

        Returns:
            dict: A constraint dictionary containing 'type' (Must-Link or Cannot-Link), 
                  'item_1', and 'item_2', or None if perfectly clustered.
        """
        predictions = np.array(predicted_labels)
        num_items = len(self.ground_truth)

        # Iterate through every possible pair to find a relationship mismatch
        for i in range(num_items):
            for j in range(i + 1, num_items):
                
                # Check for a Must-Link error (belong together, but were separated)
                if self.ground_truth[i] == self.ground_truth[j] and predictions[i] != predictions[j]:
                    constraint = ("MUST_LINK", i, j)
                    if constraint not in self.issued_constraints:
                        self.issued_constraints.add(constraint)
                        self.move_counter += 1
                        return {"type": "MUST_LINK", "item_1": i, "item_2": j}

                # Check for a Cannot-Link error (belong apart, but were grouped together)
                elif self.ground_truth[i] != self.ground_truth[j] and predictions[i] == predictions[j]:
                    constraint = ("CANNOT_LINK", i, j)
                    if constraint not in self.issued_constraints:
                        self.issued_constraints.add(constraint)
                        self.move_counter += 1
                        return {"type": "CANNOT_LINK", "item_1": i, "item_2": j}

        # Return None if no new errors exist
        return None