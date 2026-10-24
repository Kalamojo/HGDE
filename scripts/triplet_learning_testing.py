"""Run HGDE triplet adjustment + clustering on the pokemon LanceDB embeddings.

Run:
    pip install numpy pandas lancedb torch scikit-learn matplotlib
    python scripts/test_pokemon_triplet_pipeline.py --modality image --clusters 8

Pipeline:
1. Load pokemon embeddings
2. Cluster
3. Human feedback loop
4. Triplet Generation
5. Neural network training
6. Embedding transformation
7. Re-cluster
8. Evaluation
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from clustering import cluster_embeddings

# This is  the neural network that will be used to learn a transformation of the original embeddings
class Net(nn.Module):
    def __init__(self, dimensions: int, hidden_size: int = 256, scaling_factor: float = 0.1):
        super().__init__()
        self.layer1 = nn.Linear(dimensions, hidden_size)
        self.layer2 = nn.Linear(hidden_size, dimensions)
        self.scaling_factor = scaling_factor
        nn.init.zeros_(self.layer1.bias)
        nn.init.zeros_(self.layer2.bias)

    def forward(self, x):
        change = self.layer2(F.relu(self.layer1(x)))
        out = x + self.scaling_factor * torch.sigmoid(change)
        return F.normalize(out, p=2, dim=1)

# Organizes the embeddings and triplet indices into a format that PyTorch can use during training
class TripletDataset(Dataset):
    def __init__(self, embeddings, triplets):
        self.embeddings = torch.as_tensor(embeddings, dtype=torch.float32)
        self.triplets = torch.as_tensor(triplets, dtype=torch.long)
        if self.triplets.ndim != 2 or self.triplets.shape[1] != 3 or len(self.triplets) == 0:
            raise ValueError("Expected a nonempty list of (anchor, positive, negative) triples")
        if self.triplets.min().item() < 0 or self.triplets.max().item() >= len(self.embeddings):
            raise ValueError("Triplet index outside embedding matrix")

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        a, p, n = self.triplets[idx]
        return self.embeddings[a], self.embeddings[p], self.embeddings[n]


def train_round(model, optimizer, original_embeddings, all_triplets, epochs=60, batch_size=16, preservation_weight=0.05):
    loader = DataLoader(TripletDataset(original_embeddings, all_triplets), batch_size=batch_size, shuffle=True)
    triplet_loss_fn = nn.TripletMarginLoss(margin=0.2, p=2)
    losses = []
    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0
        for anchor, positive, negative in loader:
            optimizer.zero_grad()
            a_out, p_out, n_out = model(anchor), model(positive), model(negative)
            triplet_loss = triplet_loss_fn(a_out, p_out, n_out)
            originals = torch.cat((anchor, positive, negative), dim=0)
            transformed = torch.cat((a_out, p_out, n_out), dim=0)
            preservation_loss = F.mse_loss(transformed, F.normalize(originals, p=2, dim=1))
            loss = triplet_loss + preservation_weight * preservation_loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        losses.append(total_loss / total_batches)
    return losses

def transform_all(model, original_embeddings, batch_size=256):
    model.eval()
    chunks = []
    with torch.no_grad():
        for chunk in torch.as_tensor(original_embeddings, dtype=torch.float32).split(batch_size):
            chunks.append(model(chunk).cpu().numpy())
    return np.concatenate(chunks).astype(np.float32)

# Loads pokemon embeddings from LanceDB
def load_pokemon(db_path: Path, modality: str):
    import lancedb

    if not db_path.exists():
        raise FileNotFoundError(f"Pokémon LanceDB missing: {db_path} (pull repo data first)")
    db = lancedb.connect(str(db_path))
    names = db.table_names()
    if not names:
        raise ValueError(f"LanceDB contains no tables: {db_path}")
    table = db.open_table("pokemon" if "pokemon" in names else names[0])
    df = table.to_pandas()
    required = {"image_vector", "text_vector", "dex_num", "name"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing Pokémon columns: {sorted(missing)}. Available: {list(df.columns)}")

    df = df.assign(_dex_sort=df["dex_num"].astype(int)).sort_values("_dex_sort").reset_index(drop=True)
    if modality == "image":
        choices = ["image_vector"] * len(df)
    elif modality == "text":
        choices = ["text_vector"] * len(df)
    else:
        choices = ["image_vector" if i % 2 == 0 else "text_vector" for i in range(len(df))]

    vectors = []
    for i, key in enumerate(choices):
        value = df.iloc[i][key]
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if not np.isfinite(arr).all():
            raise ValueError(f"Invalid vector at row {i} ({df.iloc[i]['name']})")
        vectors.append(arr)
    lengths = {len(v) for v in vectors}
    if len(lengths) != 1:
        raise ValueError(f"Inconsistent embedding dimensions: {sorted(lengths)}")
    X = np.stack(vectors)
    if np.any(np.linalg.norm(X, axis=1) == 0):
        raise ValueError("Found zero-length embedding")
    metadata = [
        {"dex_num": str(row["dex_num"]).zfill(4), "name": str(row["name"]),
         "modality": "image" if col == "image_vector" else "text"}
        for (_, row), col in zip(df.iterrows(), choices)
    ]
    return normalize(X, norm="l2").astype(np.float32), metadata


def make_triplets(labels, moved_item, destination, count, rng):
    old_cluster = int(labels[moved_item])
    if destination == old_cluster:
        raise ValueError("Destination cannot equal source cluster")
    positives = np.flatnonzero(labels == destination)
    negatives = np.flatnonzero(labels == old_cluster)
    negatives = negatives[negatives != moved_item]
    if not len(positives) or not len(negatives):
        raise ValueError("Need at least one example in destination and another in source cluster")
    return [(int(moved_item), int(rng.choice(positives)), int(rng.choice(negatives)))
            for _ in range(count)]


def triplet_stats(X, triplets, labels=None, margin=0.2):
    a, p, n = np.asarray(triplets, dtype=int).T
    positive = np.linalg.norm(X[a] - X[p], axis=1)
    negative = np.linalg.norm(X[a] - X[n], axis=1)
    violation = np.maximum(0, positive - negative + margin)
    result = {
        "mean_ap_distance": float(positive.mean()),
        "mean_an_distance": float(negative.mean()),
        "mean_violation": float(violation.mean()),
        "distance_triplet_satisfaction": float(np.mean(violation <= 1e-6)),
    }
    if labels is not None:
        result["cluster_triplet_satisfaction"] = float(np.mean((labels[a] == labels[p]) & (labels[a] != labels[n])))
    return result


def print_cluster_examples(labels, metadata, examples_per_cluster=7):
    print("\nCurrent clusters (row index : name):")
    for k in np.unique(labels):
        inds = np.flatnonzero(labels == k)
        examples = ", ".join(f"{i}:{metadata[i]['name']}" for i in inds[:examples_per_cluster])
        print(f"  cluster {k:>2} ({len(inds):>4} items): {examples}")
    print("Full assignments are saved in the output CSV after the run.")


def resolve_move(value, metadata):
    if value.strip().isdigit():
        i = int(value)
        if not 0 <= i < len(metadata):
            raise ValueError(f"Index must be 0 to {len(metadata)-1}")
        return i
    matches = [i for i, obj in enumerate(metadata) if obj["name"].casefold() == value.strip().casefold()]
    if len(matches) != 1:
        raise ValueError(f"Pokémon name {value!r} matched {len(matches)} rows. Use its numeric row index.")
    return matches[0]


def demo_move(X, labels, used_indices):
    """Find an ambiguous K-Means member; no ground truth is used."""
    centers = np.stack([X[labels == k].mean(axis=0) for k in np.unique(labels)])
    unique = np.unique(labels)
    distances = np.linalg.norm(X[:, None, :] - centers[None, :, :], axis=2)
    current_col = np.searchsorted(unique, labels)
    to_current = distances[np.arange(len(X)), current_col]
    distances[np.arange(len(X)), current_col] = np.inf
    alternative = np.argmin(distances, axis=1)
    gap = distances[np.arange(len(X)), alternative] - to_current
    for i in np.argsort(gap):
        if int(i) not in used_indices and np.sum(labels == labels[i]) >= 2:
            return int(i), int(unique[alternative[i]])
    return None

def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    ap = argparse.ArgumentParser(description="Test HGDE notebook triplet learning with real Pokémon embeddings")
    ap.add_argument("--db", type=Path, default=REPO_ROOT / "data/pokemon_db")
    ap.add_argument("--embeddings-npy", type=Path, help="Optional N x D .npy input (bypasses LanceDB)")
    ap.add_argument("--modality", choices=("image", "text", "mixed"), default="image")
    ap.add_argument("--clusters", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=1, help="Human feedback rounds")
    ap.add_argument("--move", type=str, help="Initial Pokémon name or zero-based row index")
    ap.add_argument("--destination", type=int, help="Initial destination cluster ID")
    ap.add_argument("--auto-demo", action="store_true", help="Pick an ambiguous item automatically; NOT real human feedback")
    ap.add_argument("--triplets-per-move", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=60, help="Training epochs per move")
    ap.add_argument("--lr", type=float, default=0.001)
    ap.add_argument("--scale", type=float, default=0.1, help="Notebook residual scaling_factor")
    ap.add_argument("--preservation", type=float, default=0.05)
    ap.add_argument("--ground-truth-npy", type=Path, help="Optional label array aligned with SORTED Pokédex rows")
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "results/pokemon_triplet")
    ap.add_argument("--seed", type=int, default=123)
    return ap.parse_args()

def main():
    args = parse_args()
    if args.rounds <= 0 or args.epochs <= 0 or args.triplets_per_move <= 0:
        raise ValueError("rounds, epochs, and triplets-per-move must be positive")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    if args.embeddings_npy:
        X = np.asarray(np.load(args.embeddings_npy), dtype=np.float32)
        if X.ndim != 2 or not np.isfinite(X).all() or np.any(np.linalg.norm(X, axis=1) == 0):
            raise ValueError("Expected finite, nonzero N x D .npy embeddings")
        X = normalize(X, norm="l2").astype(np.float32)
        metadata = [{"dex_num": str(i), "name": f"item_{i}", "modality": "npy"} for i in range(len(X))]
    else:
        X, metadata = load_pokemon(args.db, args.modality)
    if (args.move is None) != (args.destination is None):
        raise ValueError("Provide both --move and --destination, or neither")
    if not 2 <= args.clusters < len(X):
        raise ValueError(f"clusters must be in [2, {len(X)-1}]")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    baseline_labels, _ = cluster_embeddings(X, algorithm="kmeans", n_clusters=args.clusters, random_state=args.seed)
    write_csv(args.output_dir / "baseline_assignments.csv", [
        {"row_index": i, **metadata[i], "cluster_before": int(baseline_labels[i])}
        for i in range(len(X))
    ])
    model = Net(X.shape[1], hidden_size=256, scaling_factor=args.scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    current_X = X.copy()
    current_labels = baseline_labels.copy()
    all_triplets = []
    edit_log = []
    round_log = []
    epoch_log = []
    used_indices = set()
    true_labels = None
    if args.ground_truth_npy:
        true_labels = np.load(args.ground_truth_npy)
        if true_labels.ndim != 1 or len(true_labels) != len(X):
            raise ValueError("Ground truth must be a 1D array matching the sorted rows")
        print(f"Baseline ARI vs supplied target labels: {adjusted_rand_score(true_labels, baseline_labels):.4f}")
    print(f"Loaded {len(X)} rows, embedding dimension {X.shape[1]}, modality={args.modality}")
    print_cluster_examples(current_labels, metadata)
    print(f"Full initial assignments: {args.output_dir / 'baseline_assignments.csv'}")

    for round_idx in range(1, args.rounds + 1):
        if args.auto_demo:
            move = demo_move(current_X, current_labels, used_indices)
            if move is None:
                print("No further demo candidates; stopping.")
                break
            moved, destination = move
        elif round_idx == 1 and args.move is not None and args.destination is not None:
            moved = resolve_move(args.move, metadata)
            destination = args.destination
        else:
            print_cluster_examples(current_labels, metadata)
            try:
                moved = resolve_move(input("Move which row index / exact Pokémon name? ").strip(), metadata)
                destination = int(input("Destination cluster ID? ").strip())
            except (EOFError, KeyboardInterrupt):
                print("\nStopping; saving completed rounds.")
                break
        if destination not in set(map(int, np.unique(current_labels))):
            raise ValueError(f"Destination {destination} does not exist")

        new_triplets = make_triplets(current_labels, moved, destination, args.triplets_per_move, rng)
        before_move_stats = triplet_stats(current_X, new_triplets, current_labels)
        all_triplets.extend(new_triplets)
        used_indices.add(moved)
        source_cluster = int(current_labels[moved])
        print(f"\nRound {round_idx}: move row {moved} ({metadata[moved]['name']}) "
              f"from {source_cluster} to {destination}; {len(new_triplets)} sampled triplets")

        # Model always acts on ORIGINAL vectors; only feedback accumulates.
        losses = train_round(model, optimizer, X, all_triplets, epochs=args.epochs, preservation_weight=args.preservation)
        current_X = transform_all(model, X)
        current_labels, _ = cluster_embeddings(current_X, algorithm="kmeans", n_clusters=args.clusters, random_state=args.seed)
        after_move_stats = triplet_stats(current_X, new_triplets, current_labels)
        overall_stats = triplet_stats(current_X, all_triplets, current_labels)
        row = {"round": round_idx, "moves": round_idx,
               "loss_final": losses[-1],
               "new_triplets_before_satisfied": before_move_stats["distance_triplet_satisfaction"],
               "new_triplets_after_satisfied": after_move_stats["distance_triplet_satisfaction"],
               "all_triplets_distance_satisfied": overall_stats["distance_triplet_satisfaction"],
               "all_triplets_cluster_satisfied": overall_stats["cluster_triplet_satisfaction"],
               "reclustering_ari_vs_initial": adjusted_rand_score(baseline_labels, current_labels),
               "ground_truth_ari": adjusted_rand_score(true_labels, current_labels) if true_labels is not None else ""}
        round_log.append(row)
        edit_log.append({"round": round_idx, "row": moved, "dex_num": metadata[moved]["dex_num"], "name": metadata[moved]["name"], "source": source_cluster, "destination": destination})
        epoch_log.extend({"round": round_idx, "epoch": i+1, "loss": l} for i, l in enumerate(losses))
        print(f"  Triplet satisfaction for THIS edit: {before_move_stats['distance_triplet_satisfaction']:.3f}"
              f" -> {after_move_stats['distance_triplet_satisfaction']:.3f}")
        print(f"  All {len(all_triplets)} triplets satisfied by distance: "
              f"{overall_stats['distance_triplet_satisfaction']:.3f}")
        print(f"  All triplets satisfied by cluster membership: "
              f"{overall_stats['cluster_triplet_satisfaction']:.3f}")
        print(f"  Last epoch mean loss: {losses[-1]:.6f}")
        if true_labels is not None:
            print(f"  ARI vs desired groups: {row['ground_truth_ari']:.4f}")
        write_csv(args.output_dir / "assignments.csv", [
        {"row_index": i, **metadata[i], "cluster_before": int(baseline_labels[i]),
         "cluster_after": int(current_labels[i])} for i in range(len(X))
    ])
    
    write_csv(args.output_dir / "history.csv", round_log)
    write_csv(args.output_dir / "epochs.csv", epoch_log)
    (args.output_dir / "edits.json").write_text(json.dumps(edit_log, indent=2), encoding="utf-8")
    np.save(args.output_dir / "baseline_embeddings.npy", X)
    np.save(args.output_dir / "personalized_embeddings.npy", current_X)
    np.save(args.output_dir / "triplets.npy", np.asarray(all_triplets, dtype=np.int64).reshape(-1, 3))
    torch.save({"model_state_dict": model.cpu().state_dict(), "input_dim": X.shape[1], "hidden_size": 256, "adjustment_scale": args.scale}, args.output_dir / "embedding_transform.pt")

    if round_log:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(np.arange(1, len(epoch_log) + 1), [r["loss"] for r in epoch_log])
        axes[0].set(xlabel="Training epoch (all rounds)", ylabel="Mean loss", title="Triplet + preservation loss")
        axes[1].plot([r["round"] for r in round_log], [r["all_triplets_distance_satisfied"] for r in round_log], marker="o", label="Distance triplet satisfied")
        axes[1].plot([r["round"] for r in round_log], [r["all_triplets_cluster_satisfied"] for r in round_log], marker="s", label="Cluster triplet satisfied")
        if true_labels is not None:
            axes[1].plot([r["round"] for r in round_log], [r["ground_truth_ari"] for r in round_log], marker="^", label="Target ARI")
        axes[1].set(xlabel="Human moves", ylabel="Score", ylim=(-0.1, 1.05), title="Feedback outcomes")
        axes[1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(args.output_dir / "training_and_feedback.png", dpi=150)
        plt.close(fig)

    print(f"Saved outputs to {args.output_dir.resolve()}")

if __name__ == "__main__":
    main()
