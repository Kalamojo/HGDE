"""Confirm the Bedrock model ids in llm_cluster/llm.py (MODELS) work before a real run (costs about $0.01).

    $env:AWS_BEARER_TOKEN_BEDROCK = "<bedrock api key>"
    python llm_experimentation/check_models.py

Sends each model one tiny call (a 64 px Pokémon image + "name the main colour") and prints OK or the error.
AccessDeniedException usually means the model isn't enabled for this account in the Bedrock console.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from llm_cluster.data import REPO_ROOT, load_png  # noqa: E402
from llm_cluster.llm import MODELS, FakeClient, Image, Text, make_client  # noqa: E402

TEST_IMAGE = REPO_ROOT / "data" / "pokemon_data" / "images" / "0025_Pikachu.png"


def ping(spec, region: str, dry_run: bool) -> str:
    client = FakeClient(spec) if dry_run else make_client(spec, region, None, None)
    parts = [Text("Reply with one word: the main colour of the creature in this image.")]
    if spec.supports_images:
        parts.insert(0, Image(load_png(TEST_IMAGE, 64, HERE / "cache"), label="test"))
    try:
        return "OK    " + client.complete("Answer briefly.", parts, task="check").strip()[:40]
    except Exception as e:
        return f"FAIL  {type(e).__name__}: {str(e)[:200]}"


def main(argv=None):
    p = argparse.ArgumentParser(description="Check Bedrock access and model ids")
    p.add_argument("--region", default="us-east-2")
    p.add_argument("--dry-run", action="store_true", help="no network: test this script's plumbing only")
    args = p.parse_args(argv)
    for name, spec in MODELS.items():
        if name != "fake":
            print(f"{name:<8} {spec.model_id:<42} {ping(spec, args.region, args.dry_run)}")


if __name__ == "__main__":
    main()
