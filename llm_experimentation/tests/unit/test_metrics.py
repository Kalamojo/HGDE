"""src/utils/metrics.py. The answer key in every test is a hand-made `true` label array."""

import itertools

import numpy as np
import pytest

import helpers  # noqa: F401  (puts the repo root on sys.path)
from src.utils.metrics import (_contingency, all_metrics, compute_metrics, pair_auc, pairwise_precision_recall,
                               sample_pairs)

TRUE = np.repeat([0, 1, 2], 20)


def test_contingency_by_hand():
    # rows = classes (a, b), columns = clusters (0, 1)
    assert _contingency(np.array([0, 0, 1, 1, 1]), np.array(["a", "b", "b", "b", "a"])).tolist() == [[1, 1], [1, 2]]


def test_pairwise_precision_recall_worked_example():
    # clusters {0,1,2} {3,4,5}; classes {0,1,2,3} {4,5}
    # same-cluster pairs: 6, of which same-class: 4; same-class pairs: 6 + 1 = 7
    p, r, _ = pairwise_precision_recall(np.array([0, 0, 0, 1, 1, 1]), np.array([0, 0, 0, 0, 1, 1]))
    assert p == pytest.approx(4 / 6) and r == pytest.approx(4 / 7)


def test_pairwise_precision_recall_equals_listing_every_pair():
    rng = np.random.default_rng(0)
    pred, true = rng.integers(0, 4, 40), rng.integers(0, 3, 40)
    pairs = list(itertools.combinations(range(40), 2))
    together = [(i, j) for i, j in pairs if pred[i] == pred[j]]
    same = [(i, j) for i, j in pairs if true[i] == true[j]]
    both = [(i, j) for i, j in together if true[i] == true[j]]
    assert pairwise_precision_recall(pred, true)[:2] == pytest.approx((len(both) / len(together), len(both) / len(same)))


def test_sample_pairs():
    idx = np.array([3, 7, 9, 12])
    pairs = sample_pairs(idx, 500, np.random.default_rng(0))
    assert pairs.shape == (500, 2) and np.all(pairs[:, 0] != pairs[:, 1]) and np.isin(pairs, idx).all()
    assert sample_pairs(np.array([5]), 10, np.random.default_rng(0)).shape == (0, 2)


def test_pair_auc_equals_comparing_every_pos_with_every_neg():
    rng = np.random.default_rng(0)
    X, labels = np.round(rng.normal(size=(40, 2)), 1), rng.integers(0, 3, 40)      # rounding makes ties
    pairs = sample_pairs(np.arange(40), 300, rng)
    d = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
    same = labels[pairs[:, 0]] == labels[pairs[:, 1]]
    pos, neg = d[same][:, None], d[~same][None, :]
    expected = ((pos < neg).sum() + 0.5 * (pos == neg).sum()) / (pos.size * neg.size)
    assert pair_auc(X, labels, pairs) == pytest.approx(expected)


def test_pair_auc_nan_without_both_kinds_of_pair():
    assert np.isnan(pair_auc(np.zeros((4, 2)), np.array([0, 0, 0, 0]), np.array([[0, 1], [2, 3]])))


def test_compute_metrics_split_class():
    # class 0 split over 4 clusters of 5: per-class recall 1 - (3/4)^4; the other classes are 1
    m = compute_metrics(np.r_[np.repeat([0, 3, 4, 5], 5), [1] * 20, [2] * 20], TRUE)
    assert m["recall"] == pytest.approx(1 - 0.75 ** 4) and m["precision"] == 1.0


def test_all_metrics_test_ari_uses_only_test_items():
    mask = np.zeros(60, bool)
    mask[::3] = True
    pred = TRUE.copy()
    pred[~mask] = np.random.default_rng(0).integers(0, 3, (~mask).sum())      # scramble only non-test items
    m = all_metrics(np.zeros((60, 2)), pred, TRUE, mask)
    assert m["test_ari"] == pytest.approx(1.0) and m["ari"] < 0.9
    assert np.isnan(all_metrics(np.zeros((60, 2)), pred, TRUE)["test_ari"])     # no test set
