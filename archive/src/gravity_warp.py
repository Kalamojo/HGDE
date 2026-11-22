"""
File: gravity_warp.py
Description:
    "Gravity well" warp: a NON-LINEAR, LOCAL manipulation of the embeddings themselves.
    Current best model (run it with warm-start KMeans; see pipeline_controller.py).

    Unlike a linear model (one global matrix applied to every point), this one moves points.
    Every group of must-linked items becomes a well: each member is pulled toward the group's
    centroid, and the pull is spread smoothly to nearby points with a Gaussian kernel, so the
    neighbourhood around a constrained item is carried along. Points far from every constraint
    barely move (the warp fades to the identity).

        f(x) = normalize( x + sum_i alpha_i * k(x, x_i) ),       k = Gaussian kernel

    The anchors x_i are the constrained items, with target displacements
        delta_i = strength * (centroid(group(i)) - x_i)         (must-link: attract)
        delta_i = -push * (x_j - x_i)                           (cannot-link, optional: repel)
    and alpha solves the kernel-ridge system (K + ridge*I) alpha = delta. The field matches the
    targets at the anchors and interpolates smoothly between them (a radial-basis-function warp,
    as used in image registration). Closed form: one small linear solve per refit.

    Invariants: every refit is learned from scratch on the ORIGINAL embeddings (never on top of
    morphed ones), so the result depends only on the set of constraints, not their order or batching.
    Pulls are toward each group's own centroid, so groups tighten without drifting together.

    Tested settings (real data): sigma_scale=1.0, strength=0.9, ridge=1e-3, knn=10 are best.
    Strength, ridge and sigma sweeps were flat; strength 1.3 and smaller sigma hurt.

    Loop results (warm-start KMeans, Must-Link batches of 3, test ARI on a fixed 20% test set the
    user never gives feedback on; mean of seeds 0 and 1):
        constraints   30    90    150   210   300   360   450
        test ARI      .21   .26   .33   .38   .47   .50   .53

    Tried and removed (code not kept; results recorded here): self-training pseudo-members ("snowball"); a learned
    feature map (MLP or linear, contrastive on the must-link groups) with this warp on top, fixed or
    gated blend; a pattern-based linear projection; iterating the warp on warped embeddings.
    Spectral / hierarchical clustering instead of KMeans: spectral (warm-started) close but
    slightly behind (.51 at 450), hierarchical behind (.43); DBSCAN does not fit (no k, colour
    classes are not dense regions).

    Interface matches pipeline_controller.py:
        transform(embeddings) -> embeddings
        train(constraints, embeddings, labels=None)       # constraints: one dict or a list
"""

import logging

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


def _sqdist(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T, 0.0)


def must_link_groups(constraints):
    """Connected components (size >= 2) of the must-link graph, i.e. its transitive closure.
    Deterministic order: members sorted, groups sorted by their smallest member."""
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for c in constraints:
        if c["type"] == "MUST_LINK":
            parent[find(c["item_1"])] = find(c["item_2"])
    groups = {}
    for node in list(parent):
        groups.setdefault(find(node), []).append(node)
    return sorted((sorted(g) for g in groups.values() if len(g) >= 2), key=lambda g: g[0])


class GravityWarp:
    def __init__(self, strength=0.9, sigma_scale=1.0, knn=10, ridge=1e-3, push=0.0, seed=0):
        self.strength = strength        # fraction of the way a member is pulled to its centroid
        self.sigma_scale = sigma_scale  # kernel width, in units of the median k-th-neighbour distance
        self.knn = knn                  # which neighbour defines the "local scale"
        self.ridge = ridge
        self.push = push                # >0: also repel Cannot-Linked pairs (untested; default user gives none)
        self.seed = seed                # unused (the model is deterministic); kept for a uniform interface

        self.X = None
        self.constraints = []
        self.trained = False
        self.stats = {}

    def _setup(self, embeddings):
        self.X = normalize(np.asarray(embeddings, dtype=np.float64))
        dist, _ = NearestNeighbors(n_neighbors=min(self.knn + 1, len(self.X))).fit(self.X).kneighbors(self.X)
        self.sigma = self.sigma_scale * float(np.median(dist[:, -1]))

    def _fit(self):
        """Kernel-ridge fit of the displacement field to the target displacements of the anchors."""
        X = self.X
        groups = must_link_groups(self.constraints)
        targets = {}                                        # item -> list of target displacements
        for g in groups:
            centroid = X[g].mean(axis=0)
            for i in g:
                targets.setdefault(i, []).append(self.strength * (centroid - X[i]))
        if self.push > 0:
            for c in self.constraints:
                if c["type"] == "CANNOT_LINK":
                    i, j = c["item_1"], c["item_2"]
                    targets.setdefault(i, []).append(-self.push * (X[j] - X[i]))
                    targets.setdefault(j, []).append(-self.push * (X[i] - X[j]))
        if not targets:
            self.trained = False
            return
        anchors = np.array(sorted(targets))
        T = np.vstack([np.mean(targets[i], axis=0) for i in anchors])
        K = np.exp(-_sqdist(X[anchors], X[anchors]) / (2.0 * self.sigma ** 2))
        self._alpha = np.linalg.solve(K + self.ridge * np.eye(len(anchors)), T)
        self._anchors = X[anchors]
        self.trained = True
        sizes = [len(g) for g in groups]
        self.stats = {"groups": len(groups), "largest_group": max(sizes, default=0), "grouped_items": int(sum(sizes))}
        logger.debug(f"warp: {self.stats}")

    # ------------------------------------------------------------ public API
    def train(self, new_constraint, embeddings, labels=None):
        """labels are accepted for interface compatibility and deliberately not used."""
        if self.X is None:
            self._setup(embeddings)
        batch = new_constraint if isinstance(new_constraint, (list, tuple)) else [new_constraint]
        self.constraints.extend(dict(c) for c in batch)     # a whole batch -> one refit
        self._fit()

    def transform(self, embeddings):
        if not self.trained:
            return embeddings
        X = normalize(np.asarray(embeddings, dtype=np.float64))
        Kx = np.exp(-_sqdist(X, self._anchors) / (2.0 * self.sigma ** 2))
        return normalize(X + Kx @ self._alpha)
