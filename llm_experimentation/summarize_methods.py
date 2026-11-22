"""What did the LLM decide to do? Round-by-round summary of an open or direct run.

    python llm_experimentation/summarize_methods.py results/llm_experimentation/<run folder>
    python llm_experimentation/summarize_methods.py <run folder> --full     # full state JSON every round

Per round: the anchor, each tested draft's share of hints satisfied (all / this round), what was
committed, test ARI, and the LLM's own description of its approach.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def pct(x):
    return f"{float(x):.0%}" if x not in ("", None) else "-"


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", type=Path)
    p.add_argument("--full", action="store_true", help="print the full method / clustering JSON every round")
    args = p.parse_args(argv)
    d = args.run_dir
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = json.loads((d / "run_config.json").read_text(encoding="utf-8"))
    if cfg.get("strategy") not in ("open", "direct"):
        raise SystemExit(f"{d} is a {cfg.get('strategy')!r} run; this summary is for open or direct runs")
    with (d / "history.csv").open(encoding="utf-8") as f:
        hist = {int(r["round"]): r for r in csv.DictReader(f)}
    edits = json.loads((d / "edits.json").read_text(encoding="utf-8"))

    def state(r):
        return json.loads((d / f"schema_round_{r:02d}.json").read_text(encoding="utf-8"))

    print(f"{cfg['model_id']} | strategy={cfg['strategy']} | inputs={cfg['inputs']} | drafts/round={cfg['drafts']}")
    print(f"\nRound 0: test ARI {float(hist[0]['test_ari']):.3f}\n  approach: {state(0)['approach']}")
    for e in edits:
        r, ch = e["round"], e["changes"]
        drafts = ", ".join(f"d{x['draft']} {pct(x['sat_all'])}/{pct(x['sat_new'])}" for x in ch.get("drafts", [])) or "none"
        before = ch.get("satisfied_before", {})
        print(f"\nRound {r}: anchor {e['name']} ({e['feedback_type']}) | test ARI {float(hist[r]['test_ari']):.3f} | "
              f"hints satisfied all/new: before {pct(before.get('all'))}/{pct(before.get('new'))}, "
              f"after {pct(hist[r]['all_triplets_satisfied'])}/{pct(hist[r]['new_triplets_after'])}")
        print(f"  drafts tested (all/new): {drafts} | committed: {ch.get('committed')}")
        print(f"  approach: {ch.get('approach')}")
        if args.full:
            print(json.dumps(state(r), indent=1))
    stop = d / "stop_reason.txt"
    if stop.exists():
        print(f"\nStopped: {stop.read_text(encoding='utf-8')}")


if __name__ == "__main__":
    main()
