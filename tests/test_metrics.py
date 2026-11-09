"""
Tests for src/utils/metrics.py and the held-out test items in src/utils/data/samplers.py.
Synthetic data only (no LanceDB).

Usage (from the repo root):
    python -m unittest tests/test_metrics.py
"""

import random
import sys
import unittest
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from src.utils.metrics import all_metrics, compute_metrics, pair_auc, pairwise_precision_recall, sample_pairs
from src.utils.data.samplers import DataSampler


class TestClusteringMetrics(unittest.TestCase):
    def test_perfect_clustering(self):
        true = np.array(["red"] * 4 + ["blue"] * 4)
        pred = np.array([1] * 4 + [0] * 4)
        m = all_metrics(np.zeros((8, 2)), pred, true)
        for key in ("ari", "f1", "precision", "recall", "pair_precision", "pair_recall", "homogeneity", "completeness"):
            self.assertAlmostEqual(m[key], 1.0, msg=key)

    def test_compute_metrics_keys_unchanged(self):
        self.assertEqual(set(compute_metrics([0, 1], ["a", "b"])), {"f1", "recall", "precision", "ari"})

    def test_pairwise_precision_recall_by_hand(self):
        # class A = items 0-3, class B = items 4-5; clusters {0,1,2}, {3,4,5}
        true = np.array([0, 0, 0, 0, 1, 1])
        pred = np.array([0, 0, 0, 1, 1, 1])
        # same-cluster pairs: 3 + 3 = 6, of which same-class: 3 (0-1,0-2,1-2) + 1 (4-5) = 4
        # same-class pairs: C(4,2) + C(2,2) = 7
        p, r = pairwise_precision_recall(pred, true)
        self.assertAlmostEqual(p, 4 / 6)
        self.assertAlmostEqual(r, 4 / 7)

    def test_split_lowers_completeness_merge_lowers_homogeneity(self):
        true = np.array([0] * 4 + [1] * 4)
        split = all_metrics(np.zeros((8, 2)), np.array([0, 0, 2, 2, 1, 1, 1, 1]), true)
        self.assertAlmostEqual(split["homogeneity"], 1.0)
        self.assertLess(split["completeness"], 1.0)
        merged = all_metrics(np.zeros((8, 2)), np.zeros(8, dtype=int), true)
        self.assertAlmostEqual(merged["completeness"], 1.0)
        self.assertLess(merged["homogeneity"], 1.0)

    def test_test_metrics_nan_without_test_items(self):
        m = all_metrics(np.zeros((4, 2)), [0, 0, 1, 1], ["a", "a", "b", "b"])
        self.assertTrue(np.isnan(m["test_ari"]) and np.isnan(m["test_pair_auc"]))


class TestPairAUC(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.labels = np.repeat([0, 1, 2], 50)

    def test_separated_blobs_near_one(self):
        X = self.rng.normal(scale=0.1, size=(150, 5)) + 10.0 * np.eye(5)[self.labels]
        pairs = sample_pairs(np.arange(150), 3000, self.rng)
        self.assertGreater(pair_auc(X, self.labels, pairs), 0.99)

    def test_random_points_near_half(self):
        X = self.rng.normal(size=(150, 5))
        pairs = sample_pairs(np.arange(150), 5000, self.rng)
        self.assertAlmostEqual(pair_auc(X, self.labels, pairs), 0.5, delta=0.05)

    def test_sample_pairs_distinct_and_in_index(self):
        idx = np.array([3, 7, 9, 12])
        pairs = sample_pairs(idx, 500, self.rng)
        self.assertTrue(np.all(pairs[:, 0] != pairs[:, 1]))
        self.assertTrue(np.isin(pairs, idx).all())


class TestSamplerTestItems(unittest.TestCase):
    def make_sampler(self, test_fraction):
        s = DataSampler.__new__(DataSampler)            # skip load_data (needs LanceDB)
        s.true_labels = np.array(["red", "blue", "green", "pink"] * 25, dtype=object)
        s.embeddings = np.zeros((100, 4), dtype=np.float32)
        s.set_test_items(test_fraction, seed=0)
        return s

    def test_fraction_and_default(self):
        self.assertEqual(self.make_sampler(0.2).test_mask.sum(), 20)
        self.assertEqual(self.make_sampler(0.0).test_mask.sum(), 0)

    def test_test_items_never_sampled(self):
        random.seed(0)
        s = self.make_sampler(0.2)
        pred = np.random.default_rng(1).integers(0, 4, 100)
        links = s.fetch_must_links(pred) + s.fetch_must_not_links(pred)
        self.assertTrue(links)
        self.assertFalse({i for pair in links for i in pair} & s.test_items)
        for _ in range(20):
            self.assertFalse(set(s.sample_pair(pred)[:2]) & s.test_items)
            triplets = s.sample_triplets(pred)
            self.assertFalse({i for t in triplets for i in t} & s.test_items)


if __name__ == "__main__":
    unittest.main()
