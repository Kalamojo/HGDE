"""
File: make_ground_truth.py
Description:
    Builds data/ground_truth_visual.npy from one column of the LanceDB table, so the labels are
    in exactly the same row order as the embeddings load_lancedb_embeddings() returns.

Usage (from the repo root):
    python scripts/make_ground_truth.py --column color
    python scripts/make_ground_truth.py --column shape
"""

import argparse
from pathlib import Path

import lancedb
import numpy as np

repo_root = Path(__file__).resolve().parent.parent

parser = argparse.ArgumentParser(description="Create the ground-truth label file from a table column.")
parser.add_argument("--column", default="color", help="Table column to use as the target clustering (e.g. color, shape).")
parser.add_argument("--out", default=str(repo_root / "data" / "ground_truth_visual.npy"), help="Output .npy path.")
args = parser.parse_args()

db = lancedb.connect(str(repo_root / "data" / "pokemon_db"))
df = db.open_table(db.table_names()[0]).to_pandas()

if args.column not in df.columns:
    raise SystemExit(f"Column '{args.column}' not found. Available: {df.columns.tolist()}")

# Missing values become their own "unknown" class instead of NaN / -1.
values = df[args.column].fillna("unknown").astype(str).to_numpy()
classes, labels = np.unique(values, return_inverse=True)
labels = labels.astype(np.int64)

np.save(args.out, labels)

print(f"Saved {len(labels)} labels to {args.out}")
print(f"{len(classes)} classes from column '{args.column}':")
for name, count in zip(classes, np.bincount(labels)):
    print(f"  {name}: {count}")