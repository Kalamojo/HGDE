"""Plot LLM runs against Jay's triplet model on the same dataset, and write a summary table.

    python llm_experimentation/compare_results.py                      # all-item runs (color_3c_all)
    python llm_experimentation/compare_results.py --config color_3c_60  # the 60-item smoke runs
    python llm_experimentation/compare_results.py --list                # show every run found

Runs are found by their run_config.json anywhere under results/llm_experimentation/ (except
results/llm_experimentation/archive/). --config picks the dataset, named <column>_<N>c_<all|sample size> like
results/triplet_by_column/; Jay's run of that name, if it exists, is drawn as the reference.
Colour = model, line style = input arm, marker = strategy.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
RESULTS = REPO_ROOT / "results" / "llm_experimentation"   # every LLM run folder lives here

MODEL_COLORS = {"sonnet": "#2a78d6", "terra": "#eb6834", "opus": "#1baf7a", "fake": "#6250d6"}
ARM_STYLES = {"image": "-", "image+text": "--", "text": ":"}
STRATEGY_MARKERS = {"schema": "o", "open": "P", "direct": "X"}     # anything else (old runs): square


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
    return f"{cfg['column']}_{top}_{cfg.get('sample_size') or 'all'}"


def find_runs() -> list[dict]:
    """Every results folder with run_config.json + history.csv."""
    runs = []
    for cfg_path in sorted(RESULTS.rglob("run_config.json")):
        d = cfg_path.parent
        if not (d / "history.csv").exists() or "archive" in d.parts:
            continue
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        strategy = cfg.get("strategy") or "schema"           # the earliest runs predate --strategy
        model_id = (cfg.get("model_id") or cfg["model"]).split(".")[-1]
        runs.append({"dir": d.name, "dataset": dataset_name(cfg), "model": cfg["model"], "arm": cfg["inputs"],
                     "strategy": strategy, "dry_run": cfg.get("dry_run", False), "rows": read_history(d / "history.csv"),
                     "label": f"{model_id} / {cfg['inputs']} / {strategy} ({d.name})"})
    return runs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="color_3c_all", help="dataset to plot, e.g. color_3c_all or color_3c_60")
    p.add_argument("--include-dry-runs", action="store_true", help="also plot fake-LLM dry runs")
    p.add_argument("--list", action="store_true", help="print every run found and exit")
    args = p.parse_args(argv)

    all_runs = find_runs()
    if args.list:
        for r in sorted(all_runs, key=lambda r: r["dataset"]):
            print(f"{r['dataset']:<16} {r['label']}  rounds={len(r['rows']) - 1}" + ("  [dry run]" if r["dry_run"] else ""))
        return

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = [r for r in all_runs if r["dataset"] == args.config and (args.include_dry_runs or not r["dry_run"])]
    jay_path = REPO_ROOT / "results" / "triplet_by_column" / args.config / "history.csv"
    series = ([("Jay triplet model", read_history(jay_path), {"color": "#3d3d3a", "lw": 2.5})] if jay_path.exists() else [])
    series += [(f"LLM {r['label']}", r["rows"], {"color": MODEL_COLORS.get(r["model"], "#e87ba4"), "lw": 2,
                                                  "ls": ARM_STYLES.get(r["arm"], "-"),
                                                  "marker": STRATEGY_MARKERS.get(r["strategy"], "s"), "ms": 5})
               for r in runs]
    if not series:
        raise SystemExit(f"nothing to plot for {args.config} (try --list)")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, metric, title in zip(axes, ("ari", "test_ari"), ("ARI (all items)", "Test ARI (held-out 20%)")):
        for label, rows, style in series:
            ax.plot([as_float(r["triplets_total"]) for r in rows], [as_float(r[metric]) for r in rows], label=label, **style)
        ax.set(title=title, xlabel="Feedback triplets given")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Adjusted Rand Index")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle(f"LLM baselines vs. triplet model: {args.config}")
    fig.tight_layout(rect=(0, 0.05 + 0.03 * (len(labels) // 2), 1, 0.95))
    out_png = RESULTS / f"comparison_{args.config}.png"
    fig.savefig(out_png, dpi=150)

    summary = []
    for label, rows, _ in series:
        ari, test = [as_float(r["ari"]) for r in rows], [as_float(r["test_ari"]) for r in rows]
        summary.append({"series": label, "rounds": len(rows) - 1, "ari_start": ari[0], "ari_final": ari[-1],
                        "test_ari_start": test[0], "test_ari_final": test[-1], "test_ari_best": max(test),
                        "usd_total": rows[-1].get("usd_total", "")})
        print(f"{label:<60} ARI {ari[0]:.3f} -> {ari[-1]:.3f} | test ARI {test[0]:.3f} -> {test[-1]:.3f}")
    out_csv = RESULTS / f"summary_{args.config}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        w.writerows(summary)
    print(f"Saved {out_png} and {out_csv}")


if __name__ == "__main__":
    main()
