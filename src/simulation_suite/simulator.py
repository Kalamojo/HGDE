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
    user = SyntheticUser(ground_truth_labels=actual_labels, seed=0)
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

    def __init__(self, ground_truth_labels, seed=0):
        """
        Initializes the synthetic user with the hidden target rules and empty state.

        Args:
            ground_truth_labels (list or numpy.ndarray): The perfect cluster assignments.
            seed (int): Seed for choosing which error to correct, so runs are reproducible
                        and a baseline run sees the same sequence of choices.
        """
        self.ground_truth = np.array(ground_truth_labels)
        self.history = {
            "moves": [],
            "ari_scores": []
        }
        self.move_counter = 0
        self.rng = np.random.default_rng(seed)

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
        Finds every pair whose predicted relationship disagrees with the ground truth,
        then picks one UNISSUED error at random (seeded) and returns it as a constraint.

        Picking at random, rather than the first error in index order, spreads feedback
        across the dataset instead of concentrating it on item 0.

        Args:
            predicted_labels (list or numpy.ndarray): The flawed clusters.

        Returns:
            dict: A constraint dictionary containing 'type' (MUST_LINK or CANNOT_LINK),
                  'item_1', and 'item_2' (0-indexed, item_1 < item_2), or None if no
                  unissued errors remain.
        """
        predictions = np.array(predicted_labels)
        gt = self.ground_truth

        same_gt = gt[:, None] == gt[None, :]
        same_pred = predictions[:, None] == predictions[None, :]
        upper = np.triu(np.ones(same_gt.shape, dtype=bool), k=1)  # each pair once, i < j

        # Must-Link error: belong together, but were separated.
        must_i, must_j = np.nonzero(same_gt & ~same_pred & upper)
        # Cannot-Link error: belong apart, but were grouped together.
        cannot_i, cannot_j = np.nonzero(~same_gt & same_pred & upper)

        n_must = len(must_i)
        total = n_must + len(cannot_i)
        if total == 0:
            return None

        # Walk the errors in random order until one has not been issued yet.
        # issued_constraints is tiny (<= max_moves), so this almost always stops immediately.
        for k in self.rng.permutation(total):
            if k < n_must:
                constraint = ("MUST_LINK", int(must_i[k]), int(must_j[k]))
            else:
                k2 = k - n_must
                constraint = ("CANNOT_LINK", int(cannot_i[k2]), int(cannot_j[k2]))
            if constraint not in self.issued_constraints:
                self.issued_constraints.add(constraint)
                self.move_counter += 1
                return {"type": constraint[0], "item_1": constraint[1], "item_2": constraint[2]}

        # Every remaining error has already been issued once.
        return None