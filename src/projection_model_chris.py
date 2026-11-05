"""
File: projection_model_chris.py
Description:
    Low-rank, identity-initialised projection model for constrained clustering with KMeans.

    The map is   f(x) = normalize( x + x V U^T ),   U, V in R^{d x r},   U = 0 at the start,
    so f is exactly the identity until the first constraint arrives. Because f is linear (plus a
    normalisation) it is smooth: points near a constrained point are moved in a similar way, which
    is how pulling b towards c can also carry b's neighbours (a) along.

    Every time a constraint arrives, ALL constraints so far are used to refit f (warm-started),
    so earlier constraints are never forgotten. The objective is

        L = sum_k w_k * hinge_k                       (must/cannot-link pair terms)
          + lam_id * mean_i || f(x_i) - x_i ||^2      (change the space as little as possible)
          + lam_wd * (|U|^2 + |V|^2)                  (keep the map low-capacity)

    must-link  : hinge = max(0, d - pos_margin)^2   (pull closer than a typical neighbour radius)
    cannot-link: hinge = max(0, neg_margin - d)^2   (push apart to a typical between-point distance)

    Constraint set is expanded before each fit:
      * transitive closure of must-links (small components only),
      * implied cannot-links between must-link components,
      * optional neighbour propagation (neighbor_k > 0): a constraint (i, j) also applies, with
        lower weight, to (nn(i), j) and (i, nn(j)).

    Interface matches pipeline_controller.py:
        transform(embeddings) -> embeddings
        train(new_constraint, embeddings, labels=None)
"""

import logging

import numpy as np
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class LowRankProjection:
    def __init__(self, rank=16, lam_id=1.0, lam_wd=1e-3, neighbor_k=0, neighbor_weight=0.3,
                 pos_margin=None, neg_margin=None, max_component=8, max_iter=300, seed=0):
        self.rank = rank
        self.lam_id = lam_id
        self.lam_wd = lam_wd
        self.neighbor_k = neighbor_k
        self.neighbor_weight = neighbor_weight
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.max_component = max_component
        self.max_iter = max_iter
        self.rng = np.random.default_rng(seed)

        self.X = None            # unit-normalised original embeddings
        self.U = None
        self.V = None
        self.constraints = []    # the user's original constraints, in order
        self.trained = False

    # ------------------------------------------------------------------ setup
    def _setup(self, embeddings):
        X = normalize(np.asarray(embeddings, dtype=np.float64))
        N, d = X.shape
        self.X = X
        r = min(self.rank, d)
        self.U = np.zeros((d, r))
        self.V = self.rng.normal(0.0, 1.0 / np.sqrt(d), size=(d, r))  # U=0 -> identity at start

        k = max(self.neighbor_k, 5)
        nn = NearestNeighbors(n_neighbors=min(k + 1, N)).fit(X)
        dist, idx = nn.kneighbors(X)
        self.knn_idx = idx[:, 1:]
        if self.pos_margin is None:   # "close" = within a typical 5-nearest-neighbour radius
            self.pos_margin = float(np.median(dist[:, min(5, dist.shape[1] - 1)]))
        if self.neg_margin is None:   # "apart" = a typical distance between random points
            a = self.rng.integers(0, N, 2000)
            b = self.rng.integers(0, N, 2000)
            self.neg_margin = float(np.median(np.linalg.norm(X[a] - X[b], axis=1)))
        logger.debug(f"pos_margin={self.pos_margin:.3f}, neg_margin={self.neg_margin:.3f}")

    # ------------------------------------------------------- constraint set
    def _expand(self):
        """Turn the user's constraints into weighted pairs (I, J, sign, weight)."""
        parent = {}

        def find(a):
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        musts = [(c["item_1"], c["item_2"]) for c in self.constraints if c["type"] == "MUST_LINK"]
        cants = [(c["item_1"], c["item_2"]) for c in self.constraints if c["type"] == "CANNOT_LINK"]
        for i, j in musts:
            parent[find(i)] = find(j)
        comps = {}
        for node in list(parent):
            comps.setdefault(find(node), []).append(node)

        pairs = {}  # (a, b) with a < b -> (sign, weight)

        def add(a, b, sign, w):
            if a == b:
                return
            key = (min(a, b), max(a, b))
            old = pairs.get(key)
            if old is None or (old[0] != sign and w > old[1]):
                pairs[key] = (sign, w)
            elif old[0] == sign and w > old[1]:
                pairs[key] = (sign, w)

        # user's own constraints (weight 1)
        for i, j in musts:
            add(i, j, +1, 1.0)
        for i, j in cants:
            add(i, j, -1, 1.0)
        originals = [(i, j, +1) for i, j in musts] + [(i, j, -1) for i, j in cants]

        # transitive closure of must-links (small components only)
        for members in comps.values():
            if 2 < len(members) <= self.max_component:
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        add(members[x], members[y], +1, 1.0)
        # implied cannot-links: cannot(i, j) => cannot between comp(i) and comp(j)
        for i, j in cants:
            ci = comps.get(find(i), [i])
            cj = comps.get(find(j), [j])
            if len(ci) * len(cj) <= self.max_component ** 2:
                for a in ci:
                    for b in cj:
                        add(a, b, -1, 1.0)

        # neighbour propagation (lower weight, never overrides a user constraint)
        if self.neighbor_k > 0:
            k = self.neighbor_k
            for i, j, sign in originals:
                for n in self.knn_idx[i, :k]:
                    if n != j and not (sign < 0 and find(n) == find(j)):
                        add(int(n), j, sign, self.neighbor_weight)
                for m in self.knn_idx[j, :k]:
                    if m != i and not (sign < 0 and find(m) == find(i)):
                        add(i, int(m), sign, self.neighbor_weight)

        keys = list(pairs)
        I = np.array([a for a, _ in keys], dtype=int)
        J = np.array([b for _, b in keys], dtype=int)
        S = np.array([pairs[k][0] for k in keys], dtype=float)
        W = np.array([pairs[k][1] for k in keys], dtype=float)
        return I, J, S, W

    # ------------------------------------------------------- loss + gradient
    def _loss_grad(self, theta, I, J, S, W):
        X = self.X
        N, d = X.shape
        r = self.U.shape[1]
        U = theta[: d * r].reshape(d, r)
        V = theta[d * r:].reshape(d, r)

        XV = X @ V
        Y = X + XV @ U.T
        nrm = np.linalg.norm(Y, axis=1, keepdims=True)
        Yh = Y / nrm

        # identity term
        diff = Yh - X
        loss = self.lam_id * np.mean(np.sum(diff ** 2, axis=1))
        G = (2.0 * self.lam_id / N) * diff

        # pair terms
        D = Yh[I] - Yh[J]
        dist = np.linalg.norm(D, axis=1) + 1e-12
        must = S > 0
        viol_pos = np.maximum(0.0, dist - self.pos_margin)
        viol_neg = np.maximum(0.0, self.neg_margin - dist)
        loss += np.sum(W * np.where(must, viol_pos ** 2, viol_neg ** 2))
        coef = np.where(must, 2.0 * W * viol_pos, -2.0 * W * viol_neg) / dist
        gD = coef[:, None] * D
        np.add.at(G, I, gD)
        np.add.at(G, J, -gD)

        # back through the normalisation, then through Y = X + X V U^T
        GY = (G - Yh * np.sum(G * Yh, axis=1, keepdims=True)) / nrm
        gU = GY.T @ XV
        gV = X.T @ (GY @ U)

        loss += self.lam_wd * (np.sum(U ** 2) + np.sum(V ** 2))
        gU += 2.0 * self.lam_wd * U
        gV += 2.0 * self.lam_wd * V
        return loss, np.concatenate([gU.ravel(), gV.ravel()])

    def _fit(self):
        I, J, S, W = self._expand()
        theta0 = np.concatenate([self.U.ravel(), self.V.ravel()])
        res = minimize(self._loss_grad, theta0, args=(I, J, S, W), jac=True,
                       method="L-BFGS-B", options={"maxiter": self.max_iter})
        d, r = self.U.shape
        self.U = res.x[: d * r].reshape(d, r)
        self.V = res.x[d * r:].reshape(d, r)
        self.trained = True
        logger.debug(f"fit on {len(I)} expanded pairs, loss={res.fun:.4f}, iters={res.nit}")

    # ------------------------------------------------------------ public API
    def train(self, new_constraint, embeddings, labels=None):
        """labels are accepted for interface compatibility and deliberately not used."""
        if self.X is None:
            self._setup(embeddings)
        self.constraints.append(dict(new_constraint))
        self._fit()

    def transform(self, embeddings):
        if not self.trained:
            return embeddings
        X = normalize(np.asarray(embeddings, dtype=np.float64))
        return normalize(X + (X @ self.V) @ self.U.T)

    def constraint_satisfaction(self, labels):
        """Fraction of the user's constraints that a clustering (labels) actually honours."""
        if not self.constraints:
            return 1.0
        ok = 0
        for c in self.constraints:
            same = labels[c["item_1"]] == labels[c["item_2"]]
            ok += same if c["type"] == "MUST_LINK" else (not same)
        return ok / len(self.constraints)