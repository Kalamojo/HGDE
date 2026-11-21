"""Whole dry runs through run_experiment.main (fake LLM, 60 real items, no Bedrock calls), and --resume
continuing exactly like an uninterrupted run."""

import json
import re

import numpy as np
import pytest

import helpers  # noqa: F401
import run_experiment
import summarize_methods
from llm_cluster.data import select_items


@pytest.mark.parametrize("strategy", ["schema", "open", "direct"])
def test_dry_run_and_resume(strategy, tmp_path, needs_real_data):
    common = ["--dry-run", "--strategy", strategy, "--sample-size", "60", "--workers", "1"]
    run_experiment.main(common + ["--rounds", "3", "--output-dir", str(tmp_path / "full")])
    run_experiment.main(common + ["--rounds", "2", "--output-dir", str(tmp_path / "split")])
    run_experiment.main(common + ["--rounds", "3", "--output-dir", str(tmp_path / "split"), "--resume"])

    # resuming gives the same feedback as never stopping
    full = np.load(tmp_path / "full" / "triplets.npy")
    assert len(full) == 96 and np.array_equal(full, np.load(tmp_path / "split" / "triplets.npy"))

    # held-out items never appear in feedback, and no Pokémon name ever reaches the LLM
    sel = select_items("color", 3, 60, 42, 0.2)
    assert not sel.test_mask[full].any()
    names = {it.name.lower() for it in sel.items if len(it.name) >= 4}
    for line in (tmp_path / "full" / "llm_calls.jsonl").open(encoding="utf-8"):
        words = set(re.sub(r"\W+", " ", json.loads(line)["prompt"].lower()).split())
        assert not words & names

    if strategy != "schema":
        summarize_methods.main([str(tmp_path / "full")])
