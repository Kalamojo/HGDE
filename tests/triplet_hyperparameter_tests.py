# Hyperparameter testing for triplet_by_column.py.

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
TRIPLET_SCRIPT = REPO_ROOT / "tests" / "triplet_by_column.py"

# different hyperparameters to test
hyperparameters = {
    "lr": [0.0001, 0.0003, 0.001, 0.003, 0.01],
    "batch_size": [8, 16, 32, 64, 128],
    "scale": [0.01, 0.05, 0.1, 0.2, 0.5],
    "preservation": [0.0, 0.01, 0.05, 0.1, 0.5],
    "epochs": [5, 10, 20, 40],
    "triplets_per_round": [8, 16, 32, 64],
    "rounds": [3, 5, 10, 20, 50],
    "modality": ["text", "image", "mixed"],
}

from types import SimpleNamespace

config = SimpleNamespace(
    column="color",
    modality="text",
    top_labels=0,
    sample_size=200,
    clusters=None,
    rounds=10,
    triplets_per_round=32,
    epochs=20,
    batch_size=16,
    lr=0.001,
    scale=0.1,
    preservation=0.05,
    goal_ari=1.1,
    test_fraction=0.2,
    db=REPO_ROOT / "data" / "pokemon_db",
    output_root=REPO_ROOT / "results" / "triplet_hyperparameters",
    force=False,
    seeds=[42]
)

# metrics used for summary and plots
# Using test_air because it is the most relevant metric for evaluating the quality of the learned embedding
# Using test_pair_auc because it is a more stable metric that can be computed even when the number of clusters is small
metrics = ("test_ari", "test_pair_auc")

# reads history.csv that is made from triplet_by_column.py and converts it to a list of dicts
def read_history(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))

# converts the metric value from a string to a float or NaN if missing/invalid
def metric(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def safe_name(value):
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value))

# writes experiment results to a CSV file and creates the parent directory if it doesn't exist
def write_csv(path: Path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

# collects baseline experiment hyperparameters in 1 dictionary
def base_params(args):
    return {
        "column": args.column, "modality": args.modality,
        "top_labels": args.top_labels, "sample_size": args.sample_size,
        "clusters": args.clusters, "rounds": args.rounds,
        "triplets_per_round": args.triplets_per_round,
        "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "scale": args.scale,
        "preservation": args.preservation,
        "goal_ari": args.goal_ari, "test_fraction": args.test_fraction,
        "db": str(args.db.resolve()),
    }

# runs a single trial of the triplet_by_column.py script with the given hyperparameters, and returns a summary of the results along with history of metrics
def run_trial(args, hyperparameter, value, seed):
    # create the configuration of the experiment
    config = base_params(args)
    config[hyperparameter] = value
    config["seed"] = seed

    # create the output directory for this specific hyperparameter, value, and seed
    out_dir = args.output_root / hyperparameter / f"{hyperparameter}_{safe_name(value)}" / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.csv"
    settings_path = out_dir / "hyperparameter_settings.json"
    old_settings = None
    if settings_path.exists():
        try:
            old_settings = json.loads(settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    # check if experiment already ran
    reusable = (not args.force and history_path.exists() and old_settings == config)

    if not TRIPLET_SCRIPT.is_file():
        raise FileNotFoundError(f"Cannot find triplet script: {TRIPLET_SCRIPT}")

    elapsed = float("nan")
    if not reusable:
        # run the triplet_by_column.py script with the specified hyperparameters
        command = [sys.executable, str(TRIPLET_SCRIPT)]
        for key in ("column", "modality", "top_labels", "sample_size", "rounds",
                    "triplets_per_round", "epochs", "batch_size", "lr", "scale",
                    "preservation", "goal_ari", "test_fraction", "db"):
            command.extend(["--" + key.replace("_", "-"), str(config[key])])
        if config["clusters"] is not None:
            command.extend(["--clusters", str(config["clusters"])])
        command.extend(["--seed", str(seed), "--output-dir", str(out_dir)])
        print(f"  Running {hyperparameter}={value}, seed={seed}", flush=True)
        # measure runtime
        start = time.perf_counter()
        with (out_dir / "run.log").open("w", encoding="utf-8") as log:
            outcome = subprocess.run(command, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        elapsed = time.perf_counter() - start
        (out_dir / "run_seconds.txt").write_text(str(elapsed), encoding="utf-8")
        if outcome.returncode != 0:
            raise RuntimeError(f"Run failed (exit {outcome.returncode}). Inspect {out_dir / 'run.log'}")
        settings_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    else:
        print(f"  Reusing {hyperparameter}={value}, seed={seed}", flush=True)
        runtime_file = out_dir / "run_seconds.txt"
        if runtime_file.exists():
            elapsed = float(runtime_file.read_text(encoding="utf-8"))

    # extract the history of metrics
    history = read_history(history_path)
    if not history:
        raise RuntimeError(f"No history rows found: {history_path}")
    start_row, end_row = history[0], history[-1]
    requested_rounds = int(config["rounds"])
    test_ari_curve = padded_curve(history, "test_ari", requested_rounds)
    mean_test_ari = (
        float(np.sum((test_ari_curve[1:] + test_ari_curve[:-1]) / 2.0) / requested_rounds)
        if requested_rounds > 0 else float("nan")
    )

    # create summary of results
    summary = {
        "parameter": hyperparameter, "value": str(value), "seed": seed,
        "requested_rounds": requested_rounds, "completed_rounds": int(metric(end_row, "round")),
        "initial_ari": metric(start_row, "ari"), "final_ari": metric(end_row, "ari"),
        "initial_test_ari": metric(start_row, "test_ari"),
        "final_test_ari": metric(end_row, "test_ari"),
        "delta_test_ari": metric(end_row, "test_ari") - metric(start_row, "test_ari"),
        "initial_test_pair_auc": metric(start_row, "test_pair_auc"),
        "final_test_pair_auc": metric(end_row, "test_pair_auc"),
        "delta_test_pair_auc": metric(end_row, "test_pair_auc") - metric(start_row, "test_pair_auc"),
        "mean_test_ari_over_budget": mean_test_ari,
        "final_f1": metric(end_row, "f1"),
        "final_triplet_satisfaction": metric(end_row, "all_triplets_satisfied"),
        "rounds_to_test_ari_080": first_reached(history, "test_ari", 0.8),
        "rounds_to_test_ari_090": first_reached(history, "test_ari", 0.9),
        "runtime_seconds": elapsed,
        "output_dir": str(out_dir),
    }
    return summary, history

# helper function to figure out how quickly the model improves by finding the first round where a metric exceeds a threshold
def first_reached(history, metric_key, threshold):
    for row in history:
        score = metric(row, metric_key)
        if math.isfinite(score) and score >= threshold:
            return int(metric(row, "round"))
    return ""

# helper function to create a curve of a metric over rounds, filling in missing rounds with the last observed value in case the model ends early
def padded_curve(history, metric_key, requested_rounds):
    observed = {int(metric(row, "round")): metric(row, metric_key) for row in history}
    if not observed:
        return np.full(requested_rounds + 1, np.nan)
    last = observed[min(observed)]
    curve = []
    for round_number in range(requested_rounds + 1):
        if round_number in observed:
            last = observed[round_number]
        curve.append(last)
    return np.asarray(curve, dtype=float)

# helper function to compute the mean and standard deviation
def avg_std(values):
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0

# helper function to aggregate results across seeds for each hyperparameter value and generate plots
def aggregate_hyperparameter(hyperparameter, values, summaries, traces, root):
    aggregate_rows = []
    for value in values:
        matching = [row for row in summaries if row["value"] == str(value)]
        if not matching:
            continue
        row = {"parameter": hyperparameter, "value": str(value), "num_seeds": len(matching)}
        for key in ("final_test_ari", "delta_test_ari", "final_test_pair_auc",
                    "delta_test_pair_auc", "mean_test_ari_over_budget", "final_ari",
                    "final_f1", "completed_rounds", "runtime_seconds"):
            row[f"mean_{key}"], row[f"std_{key}"] = avg_std([r[key] for r in matching])
        aggregate_rows.append(row)
    write_csv(root / "aggregate.csv", aggregate_rows)

    # generate plots for each metric across the hyperparameter values
    for metric_key in metrics:
        fig, ax = plt.subplots(figsize=(9, 5))
        for value in values:
            trial_histories = traces.get(str(value), [])
            if not trial_histories:
                continue
            budget = int(value) if hyperparameter == "rounds" else int(summaries[0]["requested_rounds"])
            lines = np.asarray([padded_curve(h, metric_key, budget) for h in trial_histories], dtype=float)
            x = np.arange(budget + 1)
            mean = np.array([avg_std(lines[:, j])[0] for j in range(budget + 1)])
            std = np.array([avg_std(lines[:, j])[1] for j in range(budget + 1)])
            line, = ax.plot(x, mean, marker="o", markersize=3, label=f"{hyperparameter}={value}")
            if lines.shape[0] > 1:
                ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.12)
        ax.set(xlabel="Simulated user feedback rounds", ylabel=metric_key,
               title=f"{metric_key} vs. feedback rounds — {hyperparameter} hyperparameter")
        if metric_key == "test_pair_auc":
            ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.5, label="Chance AUC")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(root / f"{metric_key}_vs_round.png", dpi=160)
        plt.close(fig)

    for metric_key, ylabel in (
        ("final_test_ari", "Final held-out ARI"),
        ("final_test_pair_auc", "Final held-out pair AUC"),
        ("runtime_seconds", "Runtime (seconds)"),
    ):
        if not aggregate_rows:
            continue
        x = np.arange(len(aggregate_rows))
        y = np.array([row[f"mean_{metric_key}"] for row in aggregate_rows], dtype=float)
        err = np.array([row[f"std_{metric_key}"] for row in aggregate_rows], dtype=float)
        fig, ax = plt.subplots(figsize=(9, 4.5))
        ax.errorbar(x, y, yerr=err, fmt="o-", capsize=4)
        ax.set_xticks(x, [row["value"] for row in aggregate_rows])
        ax.set(xlabel=hyperparameter, ylabel=ylabel, title=f"{ylabel} vs. {hyperparameter} (mean ± SD across seeds)")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        fig.savefig(root / f"{metric_key}_by_value.png", dpi=160)
        plt.close(fig)
    return aggregate_rows


def main():
    args = config
    if not 0 < args.test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    
    if args.rounds < 1 or not args.seeds:
        raise ValueError("rounds must be positive and seeds cannot be empty")

    selected_hyperparameters = list(hyperparameters)
    for hyperparameter in selected_hyperparameters:
        values = hyperparameters[hyperparameter]
        hyperparameter_root = args.output_root / hyperparameter
        summaries = []
        traces = defaultdict(list)
        failures = []
        print(f"\n=== hyperparameter: {hyperparameter} | values={values} | seeds={args.seeds} ===")
        for value in values:
            for seed in args.seeds:
                try:
                    row, history = run_trial(args, hyperparameter, value, seed)
                except Exception as exc:
                    failures.append({"parameter": hyperparameter, "value": str(value), "seed": seed, "error": str(exc)})
                    print(f"  FAILED: {exc}", file=sys.stderr)
                    continue
                summaries.append(row)
                traces[str(value)].append(history)
                print(f"    final test ARI={row['final_test_ari']:.4f}, "
                      f"pair AUC={row['final_test_pair_auc']:.4f}, "
                      f"rounds={row['completed_rounds']}")
        write_csv(hyperparameter_root / "summary.csv", summaries)
        write_csv(hyperparameter_root / "failures.csv", failures)
        if summaries:
            aggregates = aggregate_hyperparameter(hyperparameter, values, summaries, traces, hyperparameter_root)
            valid = [row for row in aggregates if math.isfinite(row["mean_final_test_ari"])]
            if valid:
                best = max(valid, key=lambda row: row["mean_final_test_ari"])
                print(f"Best average final test ARI: {hyperparameter}={best['value']} "
                      f"({best['mean_final_test_ari']:.4f}; {best['num_seeds']} seeds)")
            print(f"Saved comparison CSVs and plots to {hyperparameter_root.resolve()}")
        else:
            print("No successful runs for this hyperparameter; check errors and per-run logs.")


if __name__ == "__main__":
    main()
