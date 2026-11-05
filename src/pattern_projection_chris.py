"""
File: pattern_projection_chris.py
Description:
    "Shared pattern" projection for constrained clustering with KMeans.

    Idea: an embedding mixes many patterns (colour, shape, type, ...). A must-link pair (a, b)
    tells us which patterns the user cares about: the ones in which a and b AGREE. A cannot-link
    pair tells us which patterns separate things the user wants apart. Instead of nudging points
    around with gradient descent, we estimate, from the constraints, which pattern directions are
    consistent within must-link pairs and discriminating across cannot-link pairs, keep those,
    and drop the rest. Each new constraint sharpens the estimate; directions that keep
    disagreeing inside must-link pairs are eliminated.

    Steps
      1. (once, no labels) centre the embeddings, PCA to p components, whiten every component to
         unit variance. In this space two unrelated points differ by ~2 in every direction, so
         "no information" has a known value (identity covariance).
      2. Per refit, from ALL constraints so far (with transitive closure):
            S_M = second moment of differences over must-link pairs,
            S_C = second moment of differences over cannot-link pairs,
         each shrunk toward the identity with pseudo-count tau (few pairs -> little change).
      3. Solve S_C u = nu * S_M u (generalised eigenproblem). nu > 1 means: along u, cannot-link
         pairs differ more than must-link pairs, i.e. a pattern worth clustering by. Directions
         with nu <= 1 are eliminated. Kept direction k is scaled by (nu_k - 1)^(gamma/2).
      4. transform(x) = normalize( whitened_scores(x) @ W ).

    This is a closed-form relative of Relevant Component Analysis (must-links only) and
    Discriminative Component Analysis (adds cannot-links).

    Interface matches pipeline_controller.py:
        transform(embeddings) -> embeddings
        train(new_constraint, embeddings, labels=None)
"""

import logging

import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import normalize

logger = logging.getLogger(__name__)


class PatternProjection:
    def __init__(self, p_candidates=(4, 8, 16, 32, 64), p_fixed=None, tau=3.0, gamma=1.0,
                 max_component=8, min_gain=1e-3, min_dims=4, cv_folds=5, cv_min=6, seed=0):
        self.n_components = max(p_candidates) if p_fixed is None else p_fixed
        self.p_candidates = tuple(p_candidates)   # sizes of "strongest-pattern" subspace to consider
        self.p_fixed = p_fixed                    # set to skip cross-validation and force one size
        self.tau = tau                  # prior strength: pseudo-pairs of "no information"
        self.gamma = gamma              # >1 sharpens emphasis on the strongest patterns
        self.max_component = max_component
        self.min_gain = min_gain
        self.min_dims = min_dims        # never keep fewer directions than this
        self.cv_folds = cv_folds        # constraint cross-validation used to pick the subspace size
        self.cv_min = cv_min            # fewer constraints than this: use a small default size
        self.rng = np.random.default_rng(seed)

        self.Zw = None                  # whitened PCA scores of the original embeddings
        self.W = None                   # (p x r) map from whitened scores to kept patterns
        self.gain = None
        self.p_eff = None
        self.constraints = []
        self.trained = False

    # ------------------------------------------------------------------ setup
    def _setup(self, embeddings):
        X = normalize(np.asarray(embeddings, dtype=np.float64))
        N, d = X.shape
        self.mu = X.mean(axis=0)
        Xc = X - self.mu
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        p = max(1, min(self.n_components, len(S) - 1))
        self.P = Vt[:p].T                                   # d x p
        lam = (S[:p] ** 2) / max(N - 1, 1)
        self.scale = 1.0 / np.sqrt(lam + 1e-12)
        self.Zw = (Xc @ self.P) * self.scale                # N x p, unit variance per column
        self._ref_a = self.rng.integers(0, N, 400)          # random pairs = the "typical distance"
        self._ref_b = self.rng.integers(0, N, 400)

    # ---------------------------------------------------- constraint expansion
    def _expand(self, constraints):
        parent = {}

        def find(a):
            parent.setdefault(a, a)
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        musts = [(c["item_1"], c["item_2"]) for c in constraints if c["type"] == "MUST_LINK"]
        cants = [(c["item_1"], c["item_2"]) for c in constraints if c["type"] == "CANNOT_LINK"]
        for i, j in musts:
            parent[find(i)] = find(j)
        comps = {}
        for node in list(parent):
            comps.setdefault(find(node), []).append(node)

        must_set, cant_set = set(), set()
        for i, j in musts:
            must_set.add((min(i, j), max(i, j)))
        for members in comps.values():                      # transitive closure (small groups)
            if 2 < len(members) <= self.max_component:
                for x in range(len(members)):
                    for y in range(x + 1, len(members)):
                        a, b = members[x], members[y]
                        must_set.add((min(a, b), max(a, b)))
        for i, j in cants:
            ci, cj = comps.get(find(i), [i]), comps.get(find(j), [j])
            if len(ci) * len(cj) <= self.max_component ** 2:   # implied cannot-links
                for a in ci:
                    for b in cj:
                        cant_set.add((min(a, b), max(a, b)))
            else:
                cant_set.add((min(i, j), max(i, j)))
        cant_set -= must_set
        to_arr = lambda s: (np.array([a for a, _ in s], dtype=int), np.array([b for _, b in s], dtype=int))
        return to_arr(sorted(must_set)), to_arr(sorted(cant_set)), len(musts), len(cants)

    # ------------------------------------------------------------------- fit
    def _solve(self, constraints, p):
        """Closed-form fit using the p strongest patterns. Returns (W, gain)."""
        (Im, Jm), (Ic, Jc), n_must, n_cant = self._expand(constraints)
        Zw = self.Zw[:, :p]
        eye = np.eye(p)

        def second_moment(I, J):
            if len(I) == 0:
                return eye
            D = Zw[I] - Zw[J]
            return D.T @ D / (2.0 * len(I))     # = identity for unrelated pairs

        # shrink toward the identity using the number of ORIGINAL constraints (expanded pairs are
        # correlated, so counting them would be over-confident)
        SM = (n_must * second_moment(Im, Jm) + self.tau * eye) / (n_must + self.tau)
        SC = (n_cant * second_moment(Ic, Jc) + self.tau * eye) / (n_cant + self.tau)

        nu, U = eigh(SC, SM)                    # SC u = nu SM u, with U^T SM U = I
        gain = nu - 1.0
        keep = gain > self.min_gain
        if keep.sum() < self.min_dims:          # never collapse to 1-2 dims (L2-normalising a
            keep[np.argsort(gain)[::-1][: self.min_dims]] = True   # 1-d vector leaves only +-1)
        W = U[:, keep] * (np.maximum(gain[keep], self.min_gain) ** (self.gamma / 2.0))
        return W, gain

    def _cv_score(self, p):
        """Cross-validate on the user's own constraints: fit without a fold, then check whether the
        held-out pairs land on the right side of the typical (random-pair) distance."""
        cons = self.constraints
        n = len(cons)
        folds = np.array_split(np.random.default_rng(0).permutation(n), min(self.cv_folds, n))
        Zw = self.Zw[:, :p]
        correct = 0
        for fold in folds:
            held = set(fold.tolist())
            W, _ = self._solve([c for k, c in enumerate(cons) if k not in held], p)
            Y = normalize(Zw @ W)
            thr = np.median(np.linalg.norm(Y[self._ref_a] - Y[self._ref_b], axis=1))
            for k in fold:
                c = cons[k]
                d = np.linalg.norm(Y[c["item_1"]] - Y[c["item_2"]])
                correct += (d < thr) if c["type"] == "MUST_LINK" else (d > thr)
        return correct / n

    def _fit(self):
        p_all = self.Zw.shape[1]
        cands = [c for c in self.p_candidates if c <= p_all] or [p_all]
        n = len(self.constraints)
        if self.p_fixed is not None:
            p = min(self.p_fixed, p_all)
        elif n < self.cv_min:
            p = cands[min(1, len(cands) - 1)]                  # too little feedback to validate
        else:
            p, best = cands[0], -1.0
            for c in cands:                                    # ascending: ties keep the smaller p
                score = self._cv_score(c)
                if score > best + 1e-9:
                    p, best = c, score
        self.p_eff = p
        self.W, self.gain = self._solve(self.constraints, p)
        self.trained = True
        logger.debug(f"{n} constraints -> using {p} strongest patterns, {self.W.shape[1]} directions kept")

    # ------------------------------------------------------------ public API
    def train(self, new_constraint, embeddings, labels=None):
        """labels are accepted for interface compatibility and deliberately not used."""
        if self.Zw is None:
            self._setup(embeddings)
        self.constraints.append(dict(new_constraint))
        self._fit()

    def transform(self, embeddings):
        if not self.trained or self.W is None or self.W.shape[1] == 0:
            return embeddings
        X = normalize(np.asarray(embeddings, dtype=np.float64))
        p = self.W.shape[0]
        Zw = ((X - self.mu) @ self.P[:, :p]) * self.scale[:p]
        return normalize(Zw @ self.W)

    @property
    def n_active_patterns(self):
        return 0 if self.W is None else int(self.W.shape[1])

    def constraint_satisfaction(self, labels):
        """Fraction of the user's constraints that a clustering (labels) actually honours."""
        if not self.constraints:
            return 1.0
        ok = 0
        for c in self.constraints:
            same = labels[c["item_1"]] == labels[c["item_2"]]
            ok += same if c["type"] == "MUST_LINK" else (not same)
        return ok / len(self.constraints)