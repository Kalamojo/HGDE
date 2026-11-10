"""Clustering and embedding-quality metrics logged by ClusterState.log_results.

Scopes:
  * all items: ari, Jay's f1/precision/recall, pairwise precision/recall, homogeneity/completeness
  * test items only (never used in any feedback pair or triplet): test_ari, test_pair_auc
"""

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score, homogeneity_completeness_v_measure


# Copied unchanged from Jay's tests/triplet_by_column.py.
# Note: precision/recall here are per-class products, not the standard pairwise precision/recall
# (those are logged separately as pair_precision / pair_recall).
def compute_metrics(clusters, classes):
    # Converting clusters and labels to integer indicies
    _, class_index = np.unique(classes, return_inverse=True)
    _, cluster_index = np.unique(clusters, return_inverse=True)

    # Creates a count matrix where each entry (i,j) represents the number of samples in class i that are assigned to cluster j
    counts = np.zeros((class_index.max() + 1, cluster_index.max() + 1), dtype=float)
    np.add.at(counts, (class_index, cluster_index), 1)

    # Compute recall and precision for each class
    recall_per_class = 1 - np.prod(1 - counts / counts.sum(axis=1, keepdims=True), axis=1)
    precision_per_class = 1 - np.prod(1 - counts / counts.sum(axis=0, keepdims=True), axis=1)

    # Calculate overall recall, precision, F1 score, and ARI
    recall = float(np.prod(recall_per_class))
    precision = float(np.prod(precision_per_class))
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    ari = float(adjusted_rand_score(classes, clusters))

    return {"f1": float(f1), "recall": recall, "precision": precision, "ari": ari}


def _contingency(clusters, classes) -> NDArray[np.float64]:
    _, class_index = np.unique(classes, return_inverse=True)
    _, cluster_index = np.unique(clusters, return_inverse=True)
    counts = np.zeros((class_index.max() + 1, cluster_index.max() + 1))
    np.add.at(counts, (class_index, cluster_index), 1)
    return counts


def pairwise_precision_recall(clusters, classes) -> tuple[float, float]:
    """Over all pairs of items:
    precision = of the pairs put in the same cluster, the fraction that share a class;
    recall    = of the pairs that share a class, the fraction put in the same cluster."""
    counts = _contingency(clusters, classes)
    pairs = lambda n: (n * (n - 1) / 2).sum()
    together_and_same = pairs(counts)
    same_cluster = pairs(counts.sum(axis=0))
    same_class = pairs(counts.sum(axis=1))
    precision = together_and_same / same_cluster if same_cluster else float("nan")
    recall = together_and_same / same_class if same_class else float("nan")
    return float(precision), float(recall)


def sample_pairs(idx: NDArray[np.int_], n_pairs: int, rng: np.random.Generator) -> NDArray[np.int_]:
    """n_pairs random distinct (i, j) pairs drawn from the items in idx (shape n_pairs x 2)."""
    idx = np.asarray(idx)
    if len(idx) < 2:
        return np.empty((0, 2), dtype=int)
    i = rng.integers(0, len(idx), n_pairs)
    j = (i + rng.integers(1, len(idx), n_pairs)) % len(idx)       # j != i
    return np.stack([idx[i], idx[j]], axis=1)


def pair_auc(embeddings, classes, pairs) -> float:
    """Probability that a random same-class pair is closer (Euclidean, in `embeddings`) than a random
    different-class pair; ties count 1/2 (Mann-Whitney U). 0.5 = chance, 1 = perfect ranking.
    Does not depend on any clustering."""
    if len(pairs) == 0:
        return float("nan")
    X = np.asarray(embeddings, dtype=np.float64)
    classes = np.asarray(classes)
    d = np.linalg.norm(X[pairs[:, 0]] - X[pairs[:, 1]], axis=1)
    same = classes[pairs[:, 0]] == classes[pairs[:, 1]]
    n_pos, n_neg = int(same.sum()), int((~same).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = rankdata(-d)                 # larger rank = smaller distance
    return float((ranks[same].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def all_metrics(embeddings, pred_labels, true_labels, test_mask=None, auc_pairs=None) -> dict:
    """Every logged metric for one round. test_mask: boolean array of held-out items (or None).
    auc_pairs: fixed pairs of test items (see sample_pairs) so the AUC is comparable across rounds."""
    pred = np.asarray(pred_labels)
    true = np.asarray(true_labels)
    row = compute_metrics(pred, true)
    row["pair_precision"], row["pair_recall"] = pairwise_precision_recall(pred, true)
    h, c, _ = homogeneity_completeness_v_measure(true, pred)
    row["homogeneity"], row["completeness"] = float(h), float(c)

    row["test_ari"] = float("nan")
    if test_mask is not None and test_mask.sum() >= 2 and len(np.unique(true[test_mask])) >= 2:
        row["test_ari"] = float(adjusted_rand_score(true[test_mask], pred[test_mask]))
    row["test_pair_auc"] = pair_auc(embeddings, true, auc_pairs) if auc_pairs is not None else float("nan")
    return row
