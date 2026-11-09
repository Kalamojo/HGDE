"""
File: test_pipeline_controller.py
Author: Chris Jien Zhang
Date: 2026-09-29
Course: CMSC473
Description:
    Unit tests for the master pipeline controller. Ensures the active learning loop respects
    target accuracy thresholds, max move limits and running out of errors, that the projection
    model is trained on whole batches with the ORIGINAL embeddings, and that the helper
    classes (identity baseline, per-constraint trainer, warm-start KMeans) behave as documented.

Usage:
    python -m unittest scripts/test_pipeline_controller.py
"""

import unittest
import sys
from pathlib import Path

import numpy as np

repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))

from scripts.pipeline_controller import ActiveLearningPipeline, StubProjectionModel, WarmStartClustering


class FixedClustering:
    """Returns the same labels every move and records what it was asked to cluster."""
    def __init__(self, labels):
        self.labels = np.asarray(labels)
        self.calls = []

    def fit_predict(self, embeddings):
        self.calls.append(np.asarray(embeddings))
        return self.labels


class RecordingModel(StubProjectionModel):
    """Identity projection that records every train() call."""
    def __init__(self):
        super().__init__()
        self.calls = []

    def train(self, new_constraints, embeddings=None, labels=None):
        self.calls.append((list(new_constraints), embeddings, labels))
        super().train(new_constraints, embeddings, labels)


def make_pipeline(gt, labels, max_moves, target=1.0, model=None):
    model = model or RecordingModel()
    pipe = ActiveLearningPipeline(projection_model=model, clustering_algo=FixedClustering(labels),
                                  ground_truth=gt, target_accuracy=target, max_moves=max_moves,
                                  test_fraction=0.0)
    return pipe, model


class TestActiveLearningLoop(unittest.TestCase):
    def setUp(self):
        # 3 classes of 10 items, clustered as 30 singletons: never accurate, and with plenty of
        # Must-Link errors the user never runs out within a few moves.
        self.gt = np.repeat([0, 1, 2], 10)
        self.flawed = np.arange(30)
        self.X = np.random.default_rng(0).normal(size=(30, 4))

    def test_loop_stops_at_max_moves(self):
        """max_moves=3 evaluates moves 0, 1, 2 and 3, i.e. 4 logged scores."""
        pipe, _ = make_pipeline(self.gt, self.flawed, max_moves=3)
        history = pipe.run(self.X)
        self.assertEqual(history["moves"], [0, 1, 2, 3])

    def test_zero_max_moves_only_evaluates_baseline(self):
        pipe, model = make_pipeline(self.gt, self.flawed, max_moves=0)
        history = pipe.run(self.X)
        self.assertEqual(history["moves"], [0])
        self.assertEqual(model.calls, [])

    def test_stops_early_on_target_accuracy(self):
        pipe, model = make_pipeline(self.gt, self.gt, max_moves=10)
        history = pipe.run(self.X)
        self.assertEqual(history["moves"], [0])
        self.assertAlmostEqual(history["ari_scores"][0], 1.0, places=4)
        self.assertEqual(model.calls, [])

    def test_stops_when_no_errors_remain(self):
        """One error, (0, 1): move 0 issues it; at move 1 nothing new remains, so the loop halts."""
        pipe, model = make_pipeline([0, 0, 1, 1, 2, 2], [0, 1, 2, 2, 3, 3], max_moves=10)
        history = pipe.run(np.eye(6))
        self.assertEqual(history["moves"], [0, 1])
        self.assertEqual(len(model.calls), 1)

    def test_trains_on_batches_with_original_embeddings(self):
        pipe, model = make_pipeline(self.gt, self.flawed, max_moves=2)
        pipe.run(self.X)
        self.assertEqual(len(model.calls), 2)                     # one refit per move
        for constraints, embeddings, labels in model.calls:
            self.assertEqual(len(constraints), 3)                 # the whole batch at once
            self.assertIs(embeddings, self.X)                     # never the projected embeddings
            np.testing.assert_array_equal(labels, self.flawed)    # the current clustering
        self.assertEqual(len(model.constraints), 6)

    def test_history_structure(self):
        pipe, _ = make_pipeline(self.gt, self.flawed, max_moves=2)
        history = pipe.run(self.X)
        for key in ("moves", "ari_scores", "n_constraints"):
            self.assertIn(key, history)
            self.assertEqual(len(history[key]), len(history["moves"]))
        self.assertEqual(history["n_constraints"], [0, 3, 6])


class TestHelpers(unittest.TestCase):
    def test_stub_is_identity(self):
        X = np.arange(6.0).reshape(3, 2)
        self.assertIs(StubProjectionModel().transform(X), X)

    def test_warm_start_uses_base_first_then_previous_clusters(self):
        # two well-separated blobs
        X = np.vstack([np.zeros((5, 2)), np.full((5, 2), 10.0)]) + \
            np.random.default_rng(0).normal(scale=0.1, size=(10, 2))
        base = FixedClustering([1] * 5 + [0] * 5)
        warm = WarmStartClustering(base, n_clusters=2)

        first = warm.fit_predict(X)
        self.assertEqual(len(base.calls), 1)
        np.testing.assert_array_equal(first, base.labels)

        # later calls do not use the base clusterer, and keep each cluster's label identity
        second = warm.fit_predict(X + 0.05)
        self.assertEqual(len(base.calls), 1)
        np.testing.assert_array_equal(second, first)


if __name__ == "__main__":
    unittest.main()
