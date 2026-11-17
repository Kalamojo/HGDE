"""Overlay LLM runs against Jay's triplet model on the same dataset.

    python llm_experimentation/compare_results.py                     # all-item runs (color_3c_all)
    python llm_experimentation/compare_results.py --config color_3c_60 # the 60-item smoke runs
    python llm_experimentation/compare_results.py --list               # show every run it found

Runs are found by reading run_config.json in every folder under llm_experimentation/results/, so
folder names don't matter. --config picks the dataset: <column>_<N>c_<all|sample size>, the same
naming as results/triplet_by_column/. Only runs on that exact item set are drawn together, and
Jay's run of the same name (if it exists) is drawn as the reference.
Encoding: color = model family, line style = input arm, marker = strategy (circle A schema,
square B hypothesis), dark neutral line = Jay's triplet model(s).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent

MODEL_COLORS = {"sonnet": "#2a78d6", "terra": "#eb6834", "opus": "#1baf7a", "fake": "#6250d6"}  # fixed slots
ARM_STYLES = {"image": "-", "image_text": "--", "text": ":"}
ARMS = ("image_text", "image", "text")
STRATEGY_MARKERS = {"schema": "o", "hypothesis": "s"}      # A = circle, B = square
STRATEGY_NAMES = {"schema": "A schema", "hypothesis": "B hypothesis"}   # longest suffix first when parsing directory names
REFERENCE_COLOR = "#3d3d3a"
TEXT_SECONDARY = "#5f5e5a"
GRID = "#e4e3de"


def read_history(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def as_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def dataset_name(cfg: dict) -> str:
    top = f"{cfg['top_labels']}c" if cfg.get("top_labels") else "allc"
    size = cfg.get("sample_size") or "all"
    return f"{cfg['column']}_{top}_{size}"


def find_runs():
    """Every folder under results/ that has run_config.json + history.csv, described by its config."""
    runs = []
    for cfg_path in sorted((HERE / "results").glob("*/run_config.json")):
        d = cfg_path.parent
        if not (d / "history.csv").exists():
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        arm = cfg["inputs"].replace("+", "_")
        strategy = cfg.get("strategy", "schema")          # early runs predate --strategy
        model_id = cfg.get("model_id") or cfg["model"]
        short_id = model_id.split(".")[-1].removeprefix("anthropic.")
        runs.append({"dir": d.name, "dataset": dataset_name(cfg), "model": cfg["model"], "model_id": short_id,
                     "arm": arm, "strategy": strategy, "dry_run": cfg.get("dry_run", False),
                     "seed": cfg.get("seed"), "rows": read_history(d / "history.csv")})
    # label: what the run is; add the folder name only where two runs would otherwise look identical
    keys = [(r["model_id"], r["arm"], r["strategy"]) for r in runs]
    for r, k in zip(runs, keys):
        r["label"] = f"{r['model_id']} / {r['arm'].replace('_', '+')} / {STRATEGY_NAMES[r['strategy']]}"
        if keys.count(k) > 1:
            r["label"] += f" ({r['dir']})"
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="color_3c_all",
                   help="dataset to plot: <column>_<N>c_<all|sample size>, e.g. color_3c_all or color_3c_60")
    p.add_argument("--jay", nargs="*", default=None,
                   help="results/triplet_by_column/<name> runs to draw as references (default: the one named "
                        "--config; missing ones are skipped)")
    p.add_argument("--runs", nargs="*", default=None, help="only these result folder names")
    p.add_argument("--include-dry-runs", action="store_true", help="also plot fake-LLM dry runs")
    p.add_argument("--list", action="store_true", help="print every run found, grouped by dataset, and exit")
    args = p.parse_args()

    all_runs = find_runs()
    if args.list:
        for ds in sorted({r["dataset"] for r in all_runs}):
            print(ds)
            for r in all_runs:
                if r["dataset"] == ds:
                    print(f"  {r['dir']:<40} {r['label']}  rounds={len(r['rows']) - 1}" + ("  [dry run]" if r["dry_run"] else ""))
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in all_runs if r["dataset"] == args.config and (args.include_dry_runs or not r["dry_run"])
            and (args.runs is None or r["dir"] in args.runs)]
    jay_names = args.jay if args.jay is not None else [args.config]
    refs = [(name, read_history(REPO_ROOT / "results" / "triplet_by_column" / name / "history.csv"))
            for name in jay_names if (REPO_ROOT / "results" / "triplet_by_column" / name / "history.csv").exists()]
    if not runs and not refs:
        found = sorted({r["dataset"] for r in all_runs})
        raise SystemExit(f"nothing to plot for {args.config}; datasets with LLM runs: {found or 'none'} (try --list)")
    if not refs:
        print(f"note: no Jay run named {jay_names} in results/triplet_by_column/, plotting LLM runs only")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, metric, title in zip(axes, ("ari", "test_ari"), ("ARI (all items)", "Test ARI (held-out 20%)")):
        for i, (name, rows) in enumerate(refs):
            ax.plot([as_float(r["triplets_total"]) for r in rows], [as_float(r[metric]) for r in rows],
                    color=REFERENCE_COLOR, lw=2, ls="-" if i == 0 else "-.", label=f"Jay triplet model ({name})")
        for run in runs:
            ax.plot([as_float(r["triplets_total"]) for r in run["rows"]], [as_float(r[metric]) for r in run["rows"]],
                    color=MODEL_COLORS.get(run["model"], "#e87ba4"), lw=2, ls=ARM_STYLES[run["arm"]],
                    marker=STRATEGY_MARKERS[run["strategy"]], ms=5, markeredgecolor="white", markeredgewidth=0.8, label=f"LLM {run['label']}")
        ax.set_title(title, loc="left", fontsize=12, color=REFERENCE_COLOR)
        ax.set_xlabel("Feedback triplets given", color=TEXT_SECONDARY)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(GRID)
        ax.tick_params(colors=TEXT_SECONDARY)
    axes[0].set_ylabel("Adjusted Rand Index", color=TEXT_SECONDARY)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False, fontsize=9)
    fig.suptitle(f"LLM baselines vs. triplet model: {args.config}", x=0.01, ha="left",
                 fontsize=13, color=REFERENCE_COLOR)
    fig.tight_layout(rect=(0, 0.06 + 0.04 * ((len(labels) - 1) // 4), 1, 0.95))
    out_png = HERE / "results" / f"comparison_{args.config}.png"
    fig.savefig(out_png, dpi=150)

    # table view: start / final / best for every series
    summary = []
    for name, rows in [(f"jay:{n}", r) for n, r in refs] + [(f"llm:{r['label']}", r["rows"]) for r in runs]:
        ari = [as_float(r["ari"]) for r in rows]
        test = [as_float(r["test_ari"]) for r in rows]
        summary.append({"series": name, "rounds": len(rows) - 1, "triplets": rows[-1]["triplets_total"],
                        "ari_start": ari[0], "ari_final": ari[-1], "ari_best": max(ari),
                        "test_ari_start": test[0], "test_ari_final": test[-1], "test_ari_best": max(test),
                        "usd_total": rows[-1].get("usd_total", "")})
    out_csv = HERE / "results" / f"summary_{args.config}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    for row in summary:
        print(f"{row['series']:<40} ARI {row['ari_start']:.3f} -> {row['ari_final']:.3f} | "
              f"test ARI {row['test_ari_start']:.3f} -> {row['test_ari_final']:.3f}")
    print(f"Saved {out_png} and {out_csv}")


if __name__ == "__main__":
    main()
