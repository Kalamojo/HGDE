"""
File: simulator.py
Author: Chris Jien Zhang
Date: 2026-10-15
Course: CMSC473
Description:
    Stateful simulation engine for the interactive constrained clustering pipeline.
    Evaluates unsupervised clustering outputs against a hidden ground truth and
    generates Pairwise Constraints to iteratively train the projection model.

    By default the user behaves like someone grouping items: each interaction ("move") is a BATCH
    of 3 per-pair Must-Link corrections ("these two belong together"), never a Cannot-Link.

Usage:
    from simulation_suite.simulator import SyntheticUser
    user = SyntheticUser(ground_truth_labels=actual_labels, seed=0, batch_size=3)
    ari = user.evaluate_clusters(predicted_labels)
    constraints = user.generate_next_move(predicted_labels)   # list of dicts, or None
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

    def __init__(self, ground_truth_labels, seed=0, batch_size=3, must_link_only=True, distinct_items=True,
                 test_fraction=0.2):
        """
        Args:
            ground_truth_labels (list or numpy.ndarray): The perfect cluster assignments.
            seed (int): Seed for choosing which errors to correct and the test set (reproducible runs).
            batch_size (int): Number of constraints given per move (default 3).
            must_link_only (bool): If True (default) only MUST_LINK corrections are given. If
                False, MUST_LINK and CANNOT_LINK errors are both eligible.
            distinct_items (bool): If True, the constraints inside one batch do not share items.
            test_fraction (float): Fraction of items set aside as a fixed random TEST SET. The user
                never gives feedback that mentions a test item, and test ARI is scored on them only:
                an unbiased measure of whether the feedback generalised. (Scoring on "items not yet
                mentioned" is biased: the user only mentions mis-clustered items, so the unmentioned
                ones are the easy ones.) This is an evaluation protocol: the user still behaves the
                same on the other items. 0 disables it.
        """
        self.ground_truth = np.array(ground_truth_labels)
        self.batch_size = int(batch_size)
        self.must_link_only = must_link_only
        self.distinct_items = distinct_items
        self.history = {
            "moves": [],
            "ari_scores": [],
            "n_constraints": [],      # constraints given so far, at the time of each evaluation
            "test_ari": []            # ARI on the fixed test set (generalisation)
        }
        self.move_counter = 0         # number of batches (interactions)
        self.constraint_counter = 0   # total individual constraints issued
        self.rng = np.random.default_rng(seed)

        n = len(self.ground_truth)
        n_test = int(round(n * test_fraction))
        self.test_mask = np.zeros(n, dtype=bool)
        self.test_mask[np.random.default_rng([seed, 1]).permutation(n)[:n_test]] = True   # own stream

        # State tracking to prevent duplicate constraints
        self.issued_constraints = set()

    def evaluate_clusters(self, predicted_labels):
        """
        Calculates the Adjusted Rand Index (ARI) of the current clusters and logs it.

        Returns:
            float: The ARI score ranging from -1.0 to 1.0.
        """
        predicted = np.array(predicted_labels)
        ari_score = adjusted_rand_score(self.ground_truth, predicted)
        self.history["moves"].append(self.move_counter)
        self.history["ari_scores"].append(ari_score)
        self.history["n_constraints"].append(self.constraint_counter)
        self.history["test_ari"].append(self.test_ari(predicted))
        return ari_score

    def test_ari(self, predicted_labels):
        """ARI on the fixed test set only (items the user never gives feedback on)."""
        if self.test_mask.sum() < 2:
            return float("nan")
        return adjusted_rand_score(self.ground_truth[self.test_mask], np.asarray(predicted_labels)[self.test_mask])

    def generate_next_move(self, predicted_labels):
        """
        Finds every pair whose predicted relationship disagrees with the ground truth, then
        returns a random (seeded) batch of unissued corrections.

        Returns:
            list[dict] | None: up to batch_size dicts with 'type' ('MUST_LINK' or 'CANNOT_LINK'),
            'item_1' and 'item_2' (0-indexed, item_1 < item_2), or None if no unissued error
            remains. One call = one move, however many constraints it contains.
        """
        predictions = np.array(predicted_labels)
        gt = self.ground_truth

        same_gt = gt[:, None] == gt[None, :]
        same_pred = predictions[:, None] == predictions[None, :]
        upper = np.triu(np.ones(same_gt.shape, dtype=bool), k=1)  # each pair once, i < j
        upper &= ~self.test_mask[:, None] & ~self.test_mask[None, :]   # never mention a test item

        # Must-Link error: belong together, but were separated.
        must_i, must_j = np.nonzero(same_gt & ~same_pred & upper)
        n_must = len(must_i)
        if self.must_link_only:
            cannot_i = cannot_j = np.array([], dtype=int)
        else:
            # Cannot-Link error: belong apart, but were grouped together.
            cannot_i, cannot_j = np.nonzero(~same_gt & same_pred & upper)
        total = n_must + len(cannot_i)
        if total == 0:
            return None

        batch, used_items = [], set()
        for k in self.rng.permutation(total):
            if k < n_must:
                kind, i, j = "MUST_LINK", int(must_i[k]), int(must_j[k])
            else:
                k2 = k - n_must
                kind, i, j = "CANNOT_LINK", int(cannot_i[k2]), int(cannot_j[k2])
            if (kind, i, j) in self.issued_constraints:
                continue
            if self.distinct_items and (i in used_items or j in used_items):
                continue
            self.issued_constraints.add((kind, i, j))
            used_items.update((i, j))
            batch.append({"type": kind, "item_1": i, "item_2": j})
            if len(batch) == self.batch_size:
                break

        if not batch:
            return None
        self.move_counter += 1
        self.constraint_counter += len(batch)
        return batch
