"""
File: benchmark_static.py
Description:
    Learning-curve benchmark for projection models, OUTSIDE the active loop.

    For each number of constraints m, sample m random constraints from the ground truth, fit each
    model on them, cluster the projected embeddings with KMeans and report ARI. Every model sees
    the same constraints, so differences are due to the model alone, not to which errors the
    simulator happened to pick. A supervised-LDA ceiling (fit on half the labels) is printed for
    reference.

    --must_fraction controls the mix. Random pairs are mostly cannot-link (about 1 - 1/k), which
    is roughly what the error-driven simulator produces; must-links carry more information per
    pair, so compare 0.5 with 0.1.

Usage (from the repo root):
    python scripts/benchmark_static.py
    python scripts/benchmark_static.py --must_fraction 0.1 --ms 10 30 100 300 1000 --seeds 3
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


def load_models():
    """Model factories. Add your own here: anything with _setup(X) / _fit() / transform(X)
    / .constraints, i.e. the same classes the pipeline uses."""
    try:
        from src.projection_model_chris import LowRankProjection
    except ImportError:
        from scripts.projection_model import LowRankProjection
    from src.pattern_projection_chris import PatternProjection
    return {
        "lowrank(r16,nb3)": lambda s: LowRankProjection(rank=16, neighbor_k=3, seed=s),
        "pattern(p16)": lambda s: PatternProjection(p_fixed=16, seed=s),
        "pattern(p32)": lambda s: PatternProjection(p_fixed=32, seed=s),
        "pattern(cv)": lambda s: PatternProjection(seed=s),
    }


def sample_constraints(gt, m, must_fraction, rng):
    n_must = int(round(m * must_fraction))
    want = ["MUST_LINK"] * n_must + ["CANNOT_LINK"] * (m - n_must)
    seen, cons = set(), []
    for kind in want:
        while True:
            i, j = (int(v) for v in rng.integers(0, len(gt), 2))
            if i == j or (gt[i] == gt[j]) != (kind == "MUST_LINK"):
                continue
            key = (min(i, j), max(i, j))
            if key in seen:
                continue
            seen.add(key)
            cons.append({"type": kind, "item_1": key[0], "item_2": key[1]})
            break
    return cons


def cluster_ari(Z, gt, k, seed=0):
    return adjusted_rand_score(gt, KMeans(k, n_init=10, random_state=seed).fit_predict(normalize(Z)))


def lda_ceiling(X, gt, k, seed=0):
    """Supervised linear projection fit on half the items (all labels), ARI on the other half."""
    Xn = normalize(X)
    tr, te = train_test_split(np.arange(len(gt)), test_size=0.5, random_state=seed, stratify=gt)
    pca = PCA(n_components=min(64, len(tr) - 1), random_state=seed).fit(Xn[tr])
    lda = LinearDiscriminantAnalysis().fit(pca.transform(Xn[tr]), gt[tr])
    Z = lda.transform(pca.transform(Xn[te]))
    return adjusted_rand_score(gt[te], KMeans(k, n_init=10, random_state=seed).fit_predict(Z))


def run(X, gt, ms, seeds, must_fraction, models=None):
    models = models or load_models()
    k = len(np.unique(gt))
    X = np.asarray(X, dtype=np.float32)
    raw = np.mean([cluster_ari(X, gt, k, s) for s in range(seeds)])
    table = {name: np.zeros((seeds, len(ms))) for name in models}
    for s in range(seeds):
        rng = np.random.default_rng(1000 + s)
        for mi, m in enumerate(ms):
            cons = sample_constraints(gt, m, must_fraction, rng)
            for name, make in models.items():
                model = make(s)
                model.constraints = [dict(c) for c in cons]
                model._setup(X)
                model._fit()
                table[name][s, mi] = cluster_ari(model.transform(X), gt, k, s)
        print(f"  seed {s + 1}/{seeds} done", flush=True)
    return raw, table, lda_ceiling(X, gt, k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ms", type=int, nargs="*", default=[10, 30, 100, 300, 1000])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--must_fraction", type=float, default=0.5)
    parser.add_argument("--vector_column", default="image_vector")
    parser.add_argument("--ground_truth", default="ground_truth_visual.npy")
    args = parser.parse_args()

    from scripts.ingest_lancedb import load_lancedb_embeddings
    X = load_lancedb_embeddings(db_path=str(repo_root / "data" / "pokemon_db"), vector_column=args.vector_column)
    gt = np.load(str(repo_root / "data" / args.ground_truth))
    assert X is not None and len(X) == len(gt), "embeddings / ground truth mismatch"

    t0 = time.time()
    print(f"{len(gt)} items, {len(np.unique(gt))} classes, must_fraction={args.must_fraction}, seeds={args.seeds}")
    raw, table, ceiling = run(X, gt, args.ms, args.seeds, args.must_fraction)
    print(f"\nARI of KMeans on the projected embeddings (mean of {args.seeds} seeds), {time.time() - t0:.0f}s")
    print(f"{'model':18s} " + " ".join(f"m={m:<5d}" for m in args.ms))
    print(f"{'raw (no learning)':18s} " + " ".join(f"{raw:7.3f}" for _ in args.ms))
    for name, arr in table.items():
        print(f"{name:18s} " + " ".join(f"{v:7.3f}" for v in arr.mean(axis=0)))
    print(f"\nceiling: supervised LDA on half the labels, ARI on the other half = {ceiling:.3f}")


if __name__ == "__main__":
    main()