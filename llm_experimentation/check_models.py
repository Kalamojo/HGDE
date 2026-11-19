"""Confirm the Bedrock model IDs work before spending money on a real run (costs about $0.01).

    $env:AWS_BEARER_TOKEN_BEDROCK = "<bedrock api key>"
    python llm_experimentation/check_models.py
    python llm_experimentation/check_models.py --try-id us.anthropic.claude-sonnet-5-5   # test extra candidates

1. Lists the Sonnet / Terra model and inference-profile IDs visible in the region (if the key may list).
2. Sends each model in llm_cluster/models.py one tiny Converse call (a 64 px Pokémon image + "name the
   main colour", or text only for text-only models), through the same client the experiment uses.
3. Prints OK / the error for each, and suggests a --model-id when a listed ID looks like the right one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from llm_experimentation.llm_cluster.client import BedrockClient, FakeClient, Image, Text  # noqa: E402
from llm_experimentation.llm_cluster.items import REPO_ROOT, load_png  # noqa: E402
from llm_experimentation.llm_cluster.models import MODELS, ModelSpec  # noqa: E402

KEYWORDS = ("sonnet", "opus", "terra")
TEST_IMAGE = REPO_ROOT / "data" / "pokemon_data" / "images" / "0025_Pikachu.png"


def list_available(region: str) -> list[str]:
    import boto3

    bedrock = boto3.client("bedrock", region_name=region)
    found = []
    for label, call, key, field in (
        ("inference profiles", bedrock.list_inference_profiles, "inferenceProfileSummaries", "inferenceProfileId"),
        ("foundation models", bedrock.list_foundation_models, "modelSummaries", "modelId"),
    ):
        try:
            ids = [s[field] for s in call()[key]]
        except Exception as e:  # the API key may not be allowed to list; that's fine
            print(f"  (could not list {label}: {type(e).__name__}: {str(e)[:150]})")
            continue
        hits = sorted(i for i in ids if any(k in i.lower() for k in KEYWORDS))
        print(f"  {label}: {len(ids)} total; matching {KEYWORDS}: {hits or 'none'}")
        found += hits
    return found


def ping(spec: ModelSpec, region: str, dry_run: bool) -> tuple[str, str, str]:
    client = FakeClient(spec) if dry_run else BedrockClient(spec, region, max_retries=2)
    if spec.supports_images:
        parts = [Image(load_png(TEST_IMAGE, 64, HERE / "cache"), label="test"),
                 Text("Reply with one word: the main colour of the creature in this image.")]
    else:
        parts = [Text("Reply with one word: the colour of a ripe banana.")]
    try:
        reply = client.complete("Answer briefly.", parts, max_tokens=2000, task="check")
    except Exception as e:
        code = getattr(e, "response", {}).get("Error", {}).get("Code", type(e).__name__)
        return "FAIL", code, str(e)[:200]
    tokens = f"{client.log.input_tokens} in / {client.log.output_tokens} out"
    return "OK", (reply.strip().splitlines() or [""])[0][:40], tokens


def main():
    p = argparse.ArgumentParser(description="Check Bedrock access and model IDs")
    p.add_argument("--region", default="us-east-2")
    p.add_argument("--try-id", nargs="*", default=[], help="extra model / inference-profile IDs to test")
    p.add_argument("--dry-run", action="store_true", help="no network: test this script's plumbing only")
    args = p.parse_args()

    if not args.dry_run and not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
        print("Note: AWS_BEARER_TOKEN_BEDROCK is not set; falling back to any normal AWS credentials.")

    print(f"Region {args.region}")
    listed = [] if args.dry_run else list_available(args.region)

    specs = [s for name, s in MODELS.items() if name != "fake"]
    specs += [ModelSpec(f"try:{i}", i, supports_images=True, usd_in_per_m=None, usd_out_per_m=None) for i in args.try_id]
    print(f"\n{'model':<12} {'id':<42} {'result':<6} detail")
    failed = []
    for spec in specs:
        status, detail, extra = ping(spec, args.region, args.dry_run)
        print(f"{spec.name:<12} {spec.model_id:<42} {status:<6} {detail}  ({extra})")
        if status != "OK":
            failed.append((spec, detail))

    for spec, code in failed:
        if code == "AccessDeniedException":
            exists = spec.model_id in listed or spec.model_id.split(".", 1)[-1] in listed
            print(f"\n{spec.name}: the ID {'exists' if exists else 'may be fine'}, but this account/key is not allowed "
                  "to invoke it. Fix it on the AWS side: Bedrock console -> Model access / model catalog -> enable or "
                  "subscribe to this model (Anthropic models may need the one-time use-case form; OpenAI models "
                  "a Marketplace subscription), and make sure the key's IAM role allows bedrock:InvokeModel* "
                  "(plus aws-marketplace:Subscribe/ViewSubscriptions for first use).")
            continue
        key = spec.name if spec.name in KEYWORDS else spec.model_id.lower()
        guesses = [i for i in listed if i != spec.model_id and any(k in i.lower() and k in key for k in KEYWORDS)]
        if guesses:
            print(f"\n{spec.name}: {code}; the ID may be wrong. Listed candidates (pass --model-id): {guesses}")
            print(f"  e.g. python llm_experimentation/check_models.py --try-id {guesses[0]}")
    if not failed:
        print("\nAll models reachable. Next: a small real run (see README).")


if __name__ == "__main__":
    main()
