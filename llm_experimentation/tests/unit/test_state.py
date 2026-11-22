"""src/utils/state.py: ClusterState's fixed AUC pairs and log_results."""

from types import SimpleNamespace

import numpy as np

import helpers  # noqa: F401
from src.utils.metrics import all_metrics
from src.utils.state import ClusterState


def make_state():
    mask = np.zeros(30, bool)
    mask[::3] = True
    benchmark = SimpleNamespace(true_labels=np.repeat(["red", "blue", "green"], 10), test_mask=mask)
    return ClusterState(benchmark, eval=None, adapter=None)


def test_auc_pairs_are_test_items_only():
    s = make_state()
    assert s.auc_pairs.shape == (5000, 2) and s.benchmark.test_mask[s.auc_pairs].all()


def test_log_results_row():
    s = make_state()
    pred, X = np.repeat([0, 1, 2], 10), np.random.default_rng(0).normal(size=(30, 4))
    s.log_results(pred, [0.9, 0.25], X)
    s.log_results(pred, [], X)
    expected = all_metrics(X, pred, s.benchmark.true_labels, s.benchmark.test_mask, s.auc_pairs)
    assert [r["round"] for r in s.history] == [1, 2] and s.history[0]["loss"] == 0.25 and np.isnan(s.history[1]["loss"])
    assert s.history[0]["test_ari"] == expected["test_ari"] and s.history[0]["test_pair_auc"] == expected["test_pair_auc"]
