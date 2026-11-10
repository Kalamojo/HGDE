"""
File: test_simulator.py
Author: Chris Jien Zhang
Date: 2026-09-22
Course: CMSC473
Description:
    Unit test suite for the SyntheticUser class. Validates cluster evaluation accuracy,
    batched constraint generation (moves are lists of constraint dicts), and metric tracking.

Usage:
    Run from the terminal at the repository root:
    python -m unittest scripts/test_simulator.py
"""

import unittest
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.simulation_suite.simulator import SyntheticUser


def User(gt, **kw):
    """Mechanics tests use tiny label lists, so they switch the test set off unless asked."""
    kw.setdefault("test_fraction", 0.0)
    return SyntheticUser(gt, **kw)


def _pairs(move):
    return {(c["item_1"], c["item_2"]) for c in move}


class TestEvaluateClusters(unittest.TestCase):
    def setUp(self):
        self.user = User([0, 0, 1, 1, 2, 2])

    def test_perfect_match(self):
        self.assertAlmostEqual(self.user.evaluate_clusters([0, 0, 1, 1, 2, 2]), 1.0, places=4)

    def test_shifted_labels(self):
        """ARI depends on the groupings, not on the label values."""
        self.assertAlmostEqual(self.user.evaluate_clusters([2, 2, 0, 0, 1, 1]), 1.0, places=4)

    def test_poor_match(self):
        self.assertLess(self.user.evaluate_clusters([0, 1, 0, 1, 0, 1]), 0.5)

    def test_history_logging(self):
        """Each evaluation logs the move index, ARI and the constraint count at that time."""
        flawed = [0, 1, 2, 2, 3, 3]
        self.user.evaluate_clusters(flawed)
        self.user.generate_next_move(flawed)
        self.user.evaluate_clusters(flawed)
        self.assertEqual(self.user.history["moves"], [0, 1])
        self.assertEqual(self.user.history["n_constraints"], [0, 1])
        self.assertEqual(len(self.user.history["ari_scores"]), 2)
        self.assertEqual(len(self.user.history["test_ari"]), 2)

class TestTestSet(unittest.TestCase):
    def setUp(self):
        self.gt = np.repeat(np.arange(5), 20)            # 100 items

    def test_default_sets_aside_a_fifth(self):
        user = SyntheticUser(self.gt, seed=0)
        self.assertEqual(user.test_mask.sum(), 20)

    def test_test_items_never_get_feedback(self):
        user = SyntheticUser(self.gt, seed=0, batch_size=5)
        singletons = np.arange(100)
        while (move := user.generate_next_move(singletons)) is not None:
            for c in move:
                self.assertFalse(user.test_mask[c["item_1"]] or user.test_mask[c["item_2"]])
        self.assertGreater(user.constraint_counter, 0)

    def test_test_ari_scores_only_test_items(self):
        user = SyntheticUser(self.gt, seed=0)
        pred = self.gt.copy()
        train = np.flatnonzero(~user.test_mask)
        pred[train] = np.roll(pred[train], 7)            # scramble the non-test items only
        ari = user.evaluate_clusters(pred)
        self.assertLess(ari, 1.0)
        self.assertAlmostEqual(user.history["test_ari"][-1], 1.0, places=4)

    def test_test_set_depends_only_on_seed(self):
        a, b, c = (SyntheticUser(self.gt, seed=s, batch_size=k).test_mask for s, k in ((3, 3), (3, 9), (4, 3)))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, c))


class TestGenerateNextMove(unittest.TestCase):
    def setUp(self):
        # 3 classes of 10 items; predicting every item as its own cluster gives many Must-Link errors
        self.gt = np.repeat([0, 1, 2], 10)
        self.singletons = np.arange(30)

    def test_finds_the_single_error(self):
        """Item 1 belongs with item 0 but was split off: the only correction is Must-Link (0, 1)."""
        user = User([0, 0, 1, 1, 2, 2])
        move = user.generate_next_move([0, 1, 2, 2, 3, 3])
        self.assertEqual(move, [{"type": "MUST_LINK", "item_1": 0, "item_2": 1}])

    def test_perfect_clusters_return_none(self):
        user = User([0, 0, 1, 1, 2, 2])
        self.assertIsNone(user.generate_next_move([0, 0, 1, 1, 2, 2]))
        self.assertEqual(user.move_counter, 0)

    def test_batch_is_valid_must_links(self):
        """Default move: 3 Must-Links, each a real same-class pair with item_1 < item_2 and no shared items."""
        user = User(self.gt, seed=0)
        move = user.generate_next_move(self.singletons)
        self.assertEqual(len(move), 3)
        items = []
        for c in move:
            self.assertEqual(c["type"], "MUST_LINK")
            self.assertLess(c["item_1"], c["item_2"])
            self.assertEqual(self.gt[c["item_1"]], self.gt[c["item_2"]])
            items += [c["item_1"], c["item_2"]]
        self.assertEqual(len(items), len(set(items)), "constraints in one batch share an item")

    def test_batch_size(self):
        user = User(self.gt, batch_size=5)
        self.assertEqual(len(user.generate_next_move(self.singletons)), 5)

    def test_counters(self):
        user = User(self.gt, batch_size=3)
        user.generate_next_move(self.singletons)
        user.generate_next_move(self.singletons)
        self.assertEqual(user.move_counter, 2)
        self.assertEqual(user.constraint_counter, 6)

    def test_no_repeated_constraints(self):
        """A constraint is never issued twice; once every error is used up, the user returns None."""
        user = User([0, 0, 0, 1], batch_size=1)
        pred = [0, 1, 2, 3]                     # errors: (0,1), (0,2), (1,2)
        seen = [user.generate_next_move(pred) for _ in range(3)]
        self.assertEqual(set().union(*map(_pairs, seen)), {(0, 1), (0, 2), (1, 2)})
        self.assertIsNone(user.generate_next_move(pred))

    def test_must_link_only_ignores_cannot_link_errors(self):
        """Everything lumped into one cluster: only Cannot-Link errors exist, and the default user gives none."""
        user = User(self.gt)
        self.assertIsNone(user.generate_next_move(np.zeros(30, dtype=int)))

    def test_allow_cannot_links(self):
        user = User(self.gt, must_link_only=False)
        move = user.generate_next_move(np.zeros(30, dtype=int))
        self.assertEqual(len(move), 3)
        for c in move:
            self.assertEqual(c["type"], "CANNOT_LINK")
            self.assertNotEqual(self.gt[c["item_1"]], self.gt[c["item_2"]])

    def test_seed_reproducible(self):
        a = User(self.gt, seed=7).generate_next_move(self.singletons)
        b = User(self.gt, seed=7).generate_next_move(self.singletons)
        self.assertEqual(a, b)


if __name__ == "__main__":
    unittest.main()
