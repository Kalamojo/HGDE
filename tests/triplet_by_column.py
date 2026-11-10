"""Testing triplet learning with the pokemon metadata.

Run commands:

    python tests/triplet_by_column.py --column color --modality image
    python tests/triplet_by_column.py --column shape --modality mixed --top-labels 4
    python tests/triplet_by_column.py --column primary_type --top-labels 0 --sample-size 400

"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from clustering import cluster_embeddings
from scripts.triplet_learning_testing import (
    Net,
    load_pokemon,
    train_round,
    transform_all,
    triplet_stats,
    write_csv,
)

# Clean the metadata labels to a single string value or unknown to be used as ground truth
def scalar_label(value):
    if value is None:
        return "unknown"

    return str(value).strip()

# Load pokemon embeddings and their corresponding metadata labels
def load_embeddings_and_targets(db_path: Path, modality: str, column: str):
    import lancedb

    X, items = load_pokemon(db_path, modality)
    db = lancedb.connect(str(db_path))
    tables = db.table_names()
    table = db.open_table("pokemon" if "pokemon" in tables else tables[0])
    df = table.to_pandas()

    # Check the metadata colomn exists and is valid
    if column not in df.columns:
        raise ValueError(f"Metadata column {column!r} not found. Available columns: {list(df.columns)}")

    # Sort the metadata rows by dex_num to match the order of the embeddings
    df = (df.assign(_dex_sort=df["dex_num"].astype(int)).sort_values("_dex_sort").reset_index(drop=True))

    # Verification to check that the metadata rows are algined with the embedding rows
    if len(df) != len(X) or any(str(name) != item["name"] for name, item in zip(df["name"], items)):
        raise RuntimeError("The metadata rows are not aligned with the embedding rows.")

    # Extract the ground truth labels from the metadata column
    labels = np.array([scalar_label(v) for v in df[column].tolist()], dtype=str)

    return X, items, labels

# Evaluation for how close the predicted clusters are to the ground truth metadata groupings
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

# Simulate one round of user feedback and generates triplets to train the model. Returns the new embeddings, the model, and the optimizer
def select_feedback(predicted, targets, rng, used_anchors, triplets_per_round):
    # Store pokemon that are candidates for feedback
    candidates = []

    # Loop through each pokemon in the dataset
    for anchor in range(len(targets)):
        # Finds pokemon that have the same target metadata label
        same = targets == targets[anchor]
        # Set the anchor to False so it is not included in the same group (dont want the anchor to be its own positive example)
        same[anchor] = False
        # Finds pokemon that have a different target metadata label
        different = targets != targets[anchor]

        # Skip the anchor if it has no same or different pokemon to compare to
        if not same.any() or not different.any():
            continue

        # Check if the anchor's predicted cluster is split (same target label but different predicted clusters) or mixed (different target labels but same predicted cluster)
        split = bool(np.any(same & (predicted != predicted[anchor])))
        mixed = bool(np.any(different & (predicted == predicted[anchor])))
        # Assign priority to the anchor if it has split or mixed based on the number of error types and if it has been used before
        if split or mixed:
            priority = (int(split) + int(mixed), int(anchor not in used_anchors))
            candidates.append((priority, anchor, split, mixed))

    # No pokemon with split or mixed clusters were found
    if not candidates:
        return None

    # Finds the highest priority pokemons and randomly selects one of them to be the anchor
    best_priority = max(entry[0] for entry in candidates)
    pool = [entry for entry in candidates if entry[0] == best_priority]
    _, anchor, split, mixed = pool[int(rng.integers(len(pool)))]

    # Find positive and negative candidates for the selected anchor
    positive_candidates = np.flatnonzero((targets == targets[anchor]) & (np.arange(len(targets)) != anchor))
    negative_candidates = np.flatnonzero(targets != targets[anchor])

    # Prioritize examples that are incorrectly clustered
    preferable_pos = positive_candidates[predicted[positive_candidates] != predicted[anchor]]
    preferable_neg = negative_candidates[predicted[negative_candidates] == predicted[anchor]]
    if len(preferable_pos):
        positive_candidates = preferable_pos
    if len(preferable_neg):
        negative_candidates = preferable_neg

    # Generate training triplets 
    triplets = [(int(anchor), int(rng.choice(positive_candidates)), int(rng.choice(negative_candidates))) for _ in range(triplets_per_round)]

    # Type of correction, used for logging and debugging
    kind = "split+mixed" if split and mixed else ("split" if split else "mixed")

    return int(anchor), kind, triplets

# Graph for showing clustering performance over multiple feedback rounds
def plot_history(path: Path, history):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [row["round"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, [row["ari"] for row in history], marker="o", label="ARI")
    ax.plot(rounds, [row["f1"] for row in history], marker="s", label="F1 (cluster_by_column)")
    ax.set(xlabel="Simulated corrections", ylabel="Clustering score", title="Triplet adaptation vs. metadata-defined groups")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

# Configurations for running
def parse_args():
    p = argparse.ArgumentParser(description="Metadata-guided test of HGDE triplet learning")
    p.add_argument("--db", type=Path, default=REPO_ROOT / "data/pokemon_db")
    p.add_argument("--column", default="color", help="Ground-truth metadata field")
    p.add_argument("--modality", choices=("image", "text", "mixed"), default="image")
    p.add_argument("--top-labels", type=int, default=3, help="Restrict to N most common metadata values; 0 means all")
    p.add_argument("--sample-size", type=int, default=200, help="Randomly sample N eligible Pokémon; 0 means all")
    p.add_argument("--clusters", type=int, default=None, help="K-Means cluster count; default: number of sampled target classes")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--triplets-per-round", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20, help="Epochs per feedback round")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--scale", type=float, default=0.1)
    p.add_argument("--preservation", type=float, default=0.05)
    p.add_argument("--goal-ari", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()

# Run pipeline from loading embeddings to clustering evaluation with simulated user feedback
def main():
    args = parse_args()
    if args.top_labels < 0 or args.sample_size < 0:
        raise ValueError("top-labels and sample-size must be nonnegative")
    if min(args.rounds, args.epochs, args.triplets_per_round, args.batch_size) < 1:
        raise ValueError("rounds, epochs, triplets-per-round, batch-size must be positive")
    if args.clusters is not None and args.clusters < 2:
        raise ValueError("clusters must be >= 2")

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load embeddings and metadata labels
    X, metadata, y = load_embeddings_and_targets(args.db, args.modality, args.column)
    counts = Counter(y.tolist())
    if args.top_labels:
        top_values = sorted(counts, key=lambda value: (-counts[value], value))[:args.top_labels]
        eligible = np.flatnonzero(np.isin(y, top_values))
    else:
        eligible = np.arange(len(y))

    # Randomly sample pokemon
    if args.sample_size:
        if args.sample_size > len(eligible):
            raise ValueError(
                f"sample-size={args.sample_size} exceeds {len(eligible)} eligible rows. "
                "Lower --sample-size or use --sample-size 0 to select all."
            )
        selected = rng.choice(eligible, size=args.sample_size, replace=False)
    else:
        selected = eligible
    X = X[selected]
    y = y[selected]
    metadata = [metadata[int(i)] for i in selected]

    n_classes = len(np.unique(y))
    k = args.clusters if args.clusters is not None else n_classes
    if n_classes < 2 or not 2 <= k < len(X):
        raise ValueError("Need >=2 target classes and 2 <= clusters < sample size")
    output_dir = args.output_dir or (
        REPO_ROOT / "results" / "triplet_by_column" / f"{args.column}_{args.modality}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create initial clustering and transformation model
    baseline, _ = cluster_embeddings(X, algorithm="kmeans", n_clusters=k, random_state=args.seed)
    predicted = baseline.copy()
    model = Net(X.shape[1], scaling_factor=args.scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    current_X = X.copy()
    all_triplets = []
    used_anchors = set()
    edits = []
    history = [{"round": 0, **compute_metrics(predicted, y),
                "triplets_total": 0, "new_triplets_before": "",
                "new_triplets_after": "", "all_triplets_satisfied": "",
                "last_epoch_loss": ""}]

    print(f"Testing metadata column {args.column!r} | modality={args.modality}")
    print(f"Rows={len(X)}, dimension={X.shape[1]}, target classes={n_classes}, KMeans clusters={k}")
    print(f"Sampled class counts: {dict(Counter(y.tolist()))}")
    print(f"Round 0: F1={history[-1]['f1']:.4f}, ARI={history[-1]['ari']:.4f}")

    # Feedback loop
    for round_number in range(1, args.rounds + 1):
        if history[-1]["ari"] >= args.goal_ari:
            print("Reached target ARI; stopping.")
            break
        feedback = select_feedback(predicted, y, rng, used_anchors,
                                   args.triplets_per_round)
        if feedback is None:
            print("No remaining actionable label/cluster disagreement; stopping.")
            break
        anchor, kind, new_triplets = feedback
        used_anchors.add(anchor)
        before = triplet_stats(current_X, new_triplets, predicted)
        all_triplets.extend(new_triplets)

        # Keep feeding original embeddings into the same model at every round.
        # The model's learned weights (and accumulated triplets) are what change.
        losses = train_round(model, optimizer, X, all_triplets, epochs=args.epochs, batch_size=args.batch_size, preservation_weight=args.preservation)
        current_X = transform_all(model, X)
        predicted, _ = cluster_embeddings(current_X, algorithm="kmeans", n_clusters=k, random_state=args.seed)
        
        # Evaluation
        after = triplet_stats(current_X, new_triplets, predicted)
        overall = triplet_stats(current_X, all_triplets, predicted)
        metrics = compute_metrics(predicted, y)
        row = {"round": round_number, **metrics,
               "triplets_total": len(all_triplets),
               "new_triplets_before": before["distance_triplet_satisfaction"],
               "new_triplets_after": after["distance_triplet_satisfaction"],
               "all_triplets_satisfied": overall["distance_triplet_satisfaction"],
               "last_epoch_loss": losses[-1]}
        history.append(row)
        edits.append({"round": round_number, "anchor_index": anchor,
                      "name": metadata[anchor]["name"], "target_label": str(y[anchor]),
                      "feedback_type": kind, "triplets_added": len(new_triplets)})
        print(f"Round {round_number}: {metadata[anchor]['name']} ({kind}), "
              f"F1={metrics['f1']:.4f}, ARI={metrics['ari']:.4f}, "
              f"new triplets satisfied {before['distance_triplet_satisfaction']:.3f}"
              f" -> {after['distance_triplet_satisfaction']:.3f}")

    # Save results and trained model
    write_csv(output_dir / "history.csv", history)
    write_csv(output_dir / "assignments.csv", [
        {"row_index": i, "dex_num": metadata[i]["dex_num"],
         "name": metadata[i]["name"], "modality": metadata[i]["modality"],
         args.column: str(y[i]), "cluster_before": int(baseline[i]),
         "cluster_after": int(predicted[i])}
        for i in range(len(X))
    ])
    (output_dir / "edits.json").write_text(json.dumps(edits, indent=2), encoding="utf-8")
    np.save(output_dir / "triplets.npy", np.asarray(all_triplets, dtype=np.int64).reshape(-1, 3))
    np.save(output_dir / "baseline_embeddings.npy", X)
    np.save(output_dir / "personalized_embeddings.npy", current_X)
    torch.save({"model_state_dict": model.cpu().state_dict(), "input_dim": X.shape[1], "adjustment_scale": args.scale}, output_dir / "embedding_transform.pt")
    plot_history(output_dir / "scores.png", history)
    print(f"\nFinished: initial ARI={history[0]['ari']:.4f}, final ARI={history[-1]['ari']:.4f}")
    print(f"Initial F1={history[0]['f1']:.4f}, final F1={history[-1]['f1']:.4f}")
    print(f"Files saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
