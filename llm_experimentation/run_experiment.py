"""Run one model x arm of the LLM clustering baseline.

Examples (from the repo root, PowerShell):

    # free offline test of the whole pipeline (fake LLM, random answers)
    python llm_experimentation/run_experiment.py --dry-run --rounds 3

    # real runs (Bedrock API key in $env:AWS_BEARER_TOKEN_BEDROCK)
    python llm_experimentation/run_experiment.py --model sonnet --inputs image
    python llm_experimentation/run_experiment.py --model terra --inputs image+text
    python llm_experimentation/run_experiment.py --model opus --inputs text --strategy hypothesis

The defaults match results/triplet_by_column/color_3c_all (color, top 3 labels, all items,
20% held-out, 30 rounds, 32 triplets per round, seed 42).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from llm_cluster import prompts  # noqa: E402
from llm_cluster.client import make_client  # noqa: E402
from llm_cluster.items import select_items  # noqa: E402
from llm_cluster.loop import Config, Experiment  # noqa: E402
from llm_cluster.models import MODELS  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LLM baseline: LLM-designed features + k-means + Jay's triplet feedback")
    # same as tests/triplet_by_column.py
    p.add_argument("--column", default="color", help="hidden ground-truth metadata field")
    p.add_argument("--top-labels", type=int, default=3, help="restrict to N most common values; 0 = all")
    p.add_argument("--sample-size", type=int, default=0, help="randomly sample N eligible items; 0 = all")
    p.add_argument("--rounds", type=int, default=30)
    p.add_argument("--triplets-per-round", type=int, default=32)
    p.add_argument("--goal-ari", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-fraction", type=float, default=0.2)
    # LLM-specific
    p.add_argument("--inputs", choices=("image", "image+text", "text"), default="image")
    p.add_argument("--strategy", choices=("schema", "hypothesis"), default="schema",
                   help="schema = A (feature schema, reweight/add/drop); hypothesis = B (criterion + rescore)")
    p.add_argument("--model", choices=sorted(MODELS), default="sonnet")
    p.add_argument("--model-id", default=None, help="override the Bedrock model / inference-profile id")
    p.add_argument("--region", default="us-east-2")
    p.add_argument("--price-in", type=float, default=None, help="USD per 1M input tokens (cost tracking)")
    p.add_argument("--price-out", type=float, default=None, help="USD per 1M output tokens (cost tracking)")
    p.add_argument("--max-usd", type=float, default=None, help="stop cleanly once spend exceeds this")
    p.add_argument("--image-px", type=int, default=256, help="longest image side sent to the model")
    p.add_argument("--design-sample", type=int, default=40, help="items shown when designing the first schema")
    p.add_argument("--score-batch", type=int, default=20, help="items per scoring call")
    p.add_argument("--workers", type=int, default=4, help="parallel scoring calls")
    p.add_argument("--max-tokens", type=int, default=32000)
    p.add_argument("--dry-run", action="store_true", help="use the offline fake LLM (same as --model fake)")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true", help="allow reusing a non-empty output dir")
    p.add_argument("--resume", action="store_true",
                   help="continue a stopped run in its output dir from the last saved round (same settings; "
                        "--rounds, --workers and --max-usd may change; --max-usd counts the whole run)")
    return p.parse_args(argv)


# settings that define the run; everything else (rounds, workers, budget) may differ on --resume
RESUME_KEYS = ("column", "top_labels", "sample_size", "triplets_per_round", "goal_ari", "seed", "test_fraction",
               "inputs", "strategy", "model_id", "image_px", "design_sample", "score_batch", "prompt_version")


def check_resumable(out: Path, args, spec):
    config_path = out / "run_config.json"
    if not config_path.exists():
        raise SystemExit(f"--resume: {out} has no run_config.json")
    old = json.loads(config_path.read_text(encoding="utf-8"))
    new = {**vars(args), "model_id": spec.model_id, "prompt_version": prompts.PROMPT_VERSION}
    diff = {k: (old.get(k), new.get(k)) for k in RESUME_KEYS if old.get(k, new.get(k)) != new.get(k)}
    if diff:
        raise SystemExit("--resume: settings differ from the stopped run: "
                         + ", ".join(f"{k} was {a!r}, now {b!r}" for k, (a, b) in diff.items()))


def main(argv=None):
    args = parse_args(argv)
    if args.dry_run:
        args.model = "fake"
    spec = MODELS[args.model].with_overrides(args.model_id, args.price_in, args.price_out)
    if args.inputs != "text" and not spec.supports_images:
        raise SystemExit(f"{args.model} is text-only on Bedrock; use --inputs text")

    config_name = f"{args.top_labels}c" if args.top_labels else "allc"
    config_name += f"_{args.sample_size}" if args.sample_size else "_all"
    suffix = "_hyp" if args.strategy == "hypothesis" else ""
    out = args.output_dir or HERE / "results" / f"{args.column}_{config_name}_{args.inputs.replace('+', '_')}_{args.model}{suffix}"
    if args.resume and args.overwrite:
        raise SystemExit("--resume and --overwrite contradict each other")
    if args.resume:
        check_resumable(out, args, spec)
    elif out.exists() and any(out.iterdir()) and not args.overwrite:
        raise SystemExit(f"{out} already has results; pass --resume, --overwrite or another --output-dir")
    out.mkdir(parents=True, exist_ok=True)
    if not args.resume:
        for stale in ("llm_calls.jsonl", "scores.jsonl", "stop_reason.txt"):
            (out / stale).unlink(missing_ok=True)

    sel = select_items(args.column, args.top_labels, args.sample_size, args.seed, args.test_fraction)
    cfg = Config(inputs=args.inputs, strategy=args.strategy, rounds=args.rounds, triplets_per_round=args.triplets_per_round,
                 goal_ari=args.goal_ari, seed=args.seed, design_sample=args.design_sample,
                 score_batch=args.score_batch, workers=args.workers, image_px=args.image_px,
                 max_tokens=args.max_tokens, resume=args.resume)
    config_path = out / "run_config.json"
    previous = json.loads(config_path.read_text(encoding="utf-8")) if args.resume else {}
    config_path.write_text(json.dumps({
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model_id": spec.model_id, "supports_images": spec.supports_images,
        "usd_in_per_m": spec.usd_in_per_m, "usd_out_per_m": spec.usd_out_per_m,
        "prompt_version": prompts.PROMPT_VERSION, "reasoning": "model default (nothing sent)",
        "n_items": len(sel.items), "n_test": int(sel.test_mask.sum()),
        "resumes": previous.get("resumes", 0) + 1 if args.resume else 0,
    }, indent=2), encoding="utf-8")

    client = make_client(spec, args.region, out / "llm_calls.jsonl", args.max_usd, args.seed)
    print(f"Model {spec.name} ({spec.model_id}) | strategy={args.strategy} | column={args.column!r} | output -> {out}")
    Experiment(sel, client, cfg, out, HERE / "cache").run()
    print(f"Files saved to: {out}")


if __name__ == "__main__":
    main()
