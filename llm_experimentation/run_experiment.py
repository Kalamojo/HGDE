"""Run one model x inputs x strategy cell of the LLM clustering baseline.

Examples (from the repo root, PowerShell):

    # free offline test of the whole pipeline (fake LLM, random answers)
    python llm_experimentation/run_experiment.py --dry-run --rounds 3

    # real runs (Bedrock API key in $env:AWS_BEARER_TOKEN_BEDROCK)
    python llm_experimentation/run_experiment.py --model sonnet --inputs image
    python llm_experimentation/run_experiment.py --model terra --inputs image+text --strategy open
    python llm_experimentation/run_experiment.py --model opus --inputs image --strategy direct
    python llm_experimentation/run_experiment.py --model sonnet --categories red blue green brown --clusters k

The items, held-out test set, feedback, clusterer and metrics are the src/utils objects
experiments/run_benchmarks.py uses for Jay's and John's models. The defaults match its benchmark1
(color, red + blue, 20% held-out, seed 0, HDBSCAN), with 30 rounds of up to 32 triplets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
repo_root = HERE.parent
sys.path.insert(0, str(HERE))

from llm_cluster import STRATEGIES  # noqa: E402
from llm_cluster.data import select_items  # noqa: E402
from llm_cluster.harness import Harness, HarnessConfig  # noqa: E402
from llm_cluster.llm import MODELS, LLMSettings, make_client  # noqa: E402
from src.utils.clusterings import HDBSCANClusterer, KMeansClusterer  # noqa: E402  (repo root on sys.path via llm_cluster.data)

# settings that define a run: --resume refuses to continue if any of them changed
RESUME_KEYS = ("column", "categories", "triplets_per_round", "goal_ari", "seed", "test_fraction",
               "inputs", "strategy", "clusters", "drafts", "model")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LLM baseline: an LLM clusters Pokémon with src/utils triplet feedback")
    # the benchmark, as in experiments/run_benchmarks.py
    p.add_argument("--column", default="color", help="hidden ground-truth metadata field")
    p.add_argument("--categories", nargs="+", default=["red", "blue"],
                   help="values of --column to keep (as PokemonBenchmark's categories); 'all' = every value")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--triplets-per-round", type=int, default=32)
    p.add_argument("--goal-ari", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0, help="test split, feedback, k-means and the LLM's sampling")
    p.add_argument("--test-fraction", type=float, default=0.2)
    p.add_argument("--clusters", choices=("auto", "k"), default="auto",
                   help="auto = the method decides how many clusters (schema / open: HDBSCAN, as Jay's model is "
                        "run; direct: the LLM picks); k = exactly k = the number of categories (schema / open: "
                        "k-means; direct: the LLM makes k groups)")
    # LLM-specific
    p.add_argument("--inputs", choices=("image", "image+text", "text"), default="image")
    p.add_argument("--strategy", choices=tuple(STRATEGIES), default="schema",
                   help="schema = LLM-designed features + the clusterer; open = the LLM designs its own per-item "
                        "method + the clusterer; direct = the LLM clusters every item itself")
    p.add_argument("--drafts", type=int, default=3,
                   help="open / direct: steps per round (the LLM may test drafts against the hints before committing)")
    p.add_argument("--model", choices=sorted(MODELS), default="sonnet")
    p.add_argument("--region", default="us-east-2")
    p.add_argument("--max-usd", type=float, default=None, help="stop cleanly once spend exceeds this")
    p.add_argument("--workers", type=int, default=4, help="parallel scoring calls")
    p.add_argument("--dry-run", action="store_true", help="use the offline fake LLM (same as --model fake)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true", help="allow reusing a non-empty output dir")
    p.add_argument("--resume", action="store_true",
                   help="continue a stopped run from its last saved round (same settings; --rounds, --workers "
                        "and --max-usd may change)")
    return p.parse_args(argv)


def check_resumable(out: Path, args):
    """--resume only continues a run made with the same settings."""
    path = out / "run_config.json"
    if not path.exists():
        raise SystemExit(f"--resume: {out} has no run_config.json")
    old = json.loads(path.read_text(encoding="utf-8"))
    diff = [f"{k} was {old.get(k)!r}, now {vars(args)[k]!r}" for k in RESUME_KEYS if old.get(k) != vars(args)[k]]
    if diff:
        raise SystemExit("--resume: settings differ from the stopped run: " + ", ".join(diff))


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        args.model = "fake"
    spec = MODELS[args.model]
    if args.inputs != "text" and not spec.supports_images:
        raise SystemExit(f"{args.model} is text-only; use --inputs text")

    categories = None if args.categories == ["all"] else args.categories
    dataset = "-".join(categories) if categories else "all"
    suffix = ("" if args.strategy == "schema" else f"_{args.strategy}") + ("_k" if args.clusters == "k" else "")
    out = args.output_dir or repo_root / "results" / "llm_experimentation" / f"{args.column}_{dataset}_{args.inputs.replace('+', '_')}_{args.model}{suffix}"
    if args.resume:
        check_resumable(out, args)
    elif out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} already has results; pass --resume, --overwrite or another --output-dir")
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for stale in ("llm_calls.jsonl", "scores.jsonl", "vectors.jsonl"):
            (out / stale).unlink(missing_ok=True)

    sel = select_items(args.column, categories, args.seed, args.test_fraction)
    # the same src/utils clusterers run_benchmarks.py uses (the direct strategy ignores it)
    clusterer = HDBSCANClusterer() if args.clusters == "auto" else KMeansClusterer(sel.n_classes, random_state=args.seed)
    (out / "run_config.json").write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_id": spec.model_id, "n_items": len(sel.items), "n_test": int(sel.test_mask.sum())}, indent=2),
        encoding="utf-8")

    client = make_client(spec, args.region, out / "llm_calls.jsonl", args.max_usd, args.seed)
    print(f"Model {spec.name} ({spec.model_id}) | strategy={args.strategy} | clusters={args.clusters} | "
          f"column={args.column!r} | output -> {out}")
    settings = LLMSettings(inputs=args.inputs, seed=args.seed, clusters=args.clusters, drafts=args.drafts,
                           workers=args.workers)
    strategy = STRATEGIES[args.strategy](sel, client, settings, out, HERE / "cache")
    harness_cfg = HarnessConfig(rounds=args.rounds, triplets_per_round=args.triplets_per_round,
                                goal_ari=args.goal_ari, resume=args.resume)
    Harness(sel, strategy, harness_cfg, out, clusterer).run()
    print(f"Files saved to: {out}")


if __name__ == "__main__":
    main()
