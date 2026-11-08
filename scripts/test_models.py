"""
File: test_models.py
Description:
    Invariant tests for the GravityWarp: every refit is learned from scratch on the ORIGINAL
    embeddings, so the result depends only on the set of constraints, not on the order or batching
    in which they arrived (no morphing on top of morphed embeddings).

Usage:
    python -m unittest scripts/test_models.py
"""

import unittest
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.gravity_warp import GravityWarp, must_link_groups


def _data(seed=0, n=120, d=16):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    cons = [{"type": "MUST_LINK", "item_1": int(i), "item_2": int(i + 1)} for i in range(0, 60, 4)]
    cons += [{"type": "MUST_LINK", "item_1": 1, "item_2": 5}]          # joins two pairs into a group
    return X, cons


class TestMustLinkGroups(unittest.TestCase):
    def test_transitive_closure(self):
        cons = [{"type": "MUST_LINK", "item_1": 3, "item_2": 1}, {"type": "MUST_LINK", "item_1": 1, "item_2": 7},
                {"type": "MUST_LINK", "item_1": 9, "item_2": 8}, {"type": "CANNOT_LINK", "item_1": 3, "item_2": 9}]
        self.assertEqual(must_link_groups(cons), [[1, 3, 7], [8, 9]])


class TestOrderInvariance(unittest.TestCase):
    def _check(self, make):
        X, cons = _data()
        a, b = make(), make()
        for k in range(0, len(cons), 3):                   # batches of 3, in order
            a.train(cons[k:k + 3], X)
        b.train(list(reversed(cons)), X)                   # one batch, reversed
        np.testing.assert_allclose(a.transform(X), b.transform(X), atol=1e-5)

    def test_gravity_warp(self):
        self._check(lambda: GravityWarp())

    def test_identity_before_feedback(self):
        X, _ = _data()
        for model in (GravityWarp(),):
            self.assertIs(model.transform(X), X)


if __name__ == "__main__":
    unittest.main()
