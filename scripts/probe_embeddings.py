"""
File: probe_embeddings.py
Description:
    Diagnostic: how much of each metadata attribute is actually recoverable from the image
    embeddings? For every categorical column in the LanceDB table it reports

      majority   : accuracy of always guessing the most common class (the floor)
      probe acc  : 5-fold cross-validated accuracy of a linear classifier on the embeddings,
                   using ALL labels (a ceiling for any linear projection learned from few pairs)
      lda ARI    : KMeans ARI after a supervised linear projection (PCA 64 -> LDA) fit on half
                   the items and evaluated on the other half (how well a good projection could
                   cluster unseen items)
      raw ARI    : KMeans ARI on the raw embeddings (the baseline you already measured for color)

Usage (from the repo root):
    python scripts/probe_embeddings.py
    python scripts/probe_embeddings.py --columns color shape primary_type
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")
DEFAULT_COLUMNS = ["color", "shape", "primary_type", "habitat", "generation", "stage", "egg_groups"]


def probe(X, labels, seed=0):
    """Returns (n_classes, majority_acc, probe_acc, lda_ari, raw_ari) for one labelling."""
    classes, y = np.unique(labels, return_inverse=True)
    k = len(classes)
    counts = np.bincount(y)
    majority = counts.max() / len(y)
    Xn = normalize(X)

    min_class = counts.min()
    folds = max(2, min(5, int(min_class)))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    probe_acc = cross_val_score(LogisticRegression(max_iter=500), Xn, y, cv=cv).mean()

    raw_ari = adjusted_rand_score(y, KMeans(k, n_init=10, random_state=seed).fit_predict(Xn))

    # supervised linear projection fit on half the items, clustering the other half
    try:
        tr, te = train_test_split(np.arange(len(y)), test_size=0.5, random_state=seed, stratify=y)
    except ValueError:                      # a class has a single member: no stratified split
        tr, te = train_test_split(np.arange(len(y)), test_size=0.5, random_state=seed)
    pca = PCA(n_components=min(64, len(tr) - 1), random_state=seed).fit(Xn[tr])
    lda = LinearDiscriminantAnalysis().fit(pca.transform(Xn[tr]), y[tr])
    Z = lda.transform(pca.transform(Xn[te]))
    lda_ari = adjusted_rand_score(y[te], KMeans(k, n_init=10, random_state=seed).fit_predict(Z))
    return k, majority, probe_acc, lda_ari, raw_ari


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--columns", nargs="*", default=DEFAULT_COLUMNS)
    parser.add_argument("--vector_column", default="image_vector")
    args = parser.parse_args()

    import lancedb
    root = Path(__file__).resolve().parent.parent
    db = lancedb.connect(str(root / "data" / "pokemon_db"))
    df = db.open_table(db.table_names()[0]).to_pandas()
    X = np.stack(df[args.vector_column].values).astype(np.float64)
    print(f"{len(df)} items, {X.shape[1]}-d '{args.vector_column}'\n")
    print(f"{'column':14s} {'classes':>7s} {'majority':>9s} {'probe acc':>10s} {'lda ARI':>8s} {'raw ARI':>8s}")

    for col in args.columns:
        if col not in df.columns:
            print(f"{col:14s} (not in table)")
            continue
        labels = df[col].apply(lambda v: ",".join(map(str, v)) if hasattr(v, "__iter__") and not isinstance(v, str)
                               else v).fillna("unknown").astype(str).to_numpy()
        n_classes = len(np.unique(labels))
        if n_classes < 2 or n_classes > 40:
            print(f"{col:14s} skipped ({n_classes} classes)")
            continue
        k, maj, acc, lda_ari, raw_ari = probe(X, labels)
        print(f"{col:14s} {k:7d} {maj:9.3f} {acc:10.3f} {lda_ari:8.3f} {raw_ari:8.3f}")


if __name__ == "__main__":
    main()