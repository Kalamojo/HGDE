"""Offline checks for the LLM baseline (no Bedrock calls).

    python -m pytest llm_experimentation/tests          # if pytest is installed
    python llm_experimentation/tests/test_pipeline.py    # plain python works too
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))

from llm_cluster.client import extract_json  # noqa: E402
from llm_cluster.items import REPO_ROOT, make_redactor, select_items  # noqa: E402
from llm_cluster.schema import Feature, Schema, ScoreStore, score_key, vectorize  # noqa: E402

JAY_RUN = REPO_ROOT / "results" / "triplet_by_column" / "color_3c_all"


def test_selection_matches_jay():
    sel = select_items("color", 3, 0, 42, 0.2)
    assert len(sel.items) == 439 and int(sel.test_mask.sum()) == 88 and sel.n_classes == 3
    if (JAY_RUN / "assignments.csv").exists():
        with (JAY_RUN / "assignments.csv").open(encoding="utf-8") as f:
            jay = list(csv.DictReader(f))
        assert [r["dex_num"] for r in jay] == [it.dex_num for it in sel.items]
        assert [r["color"] for r in jay] == [it.label for it in sel.items]
    ids = [it.anon_id for it in sel.items]
    assert len(set(ids)) == len(ids)
    assert ids != sorted(ids), "anonymous ids must not follow dex order"


def test_redaction():
    redact = make_redactor()
    out = redact("Squirtle evolves into Wartortle. squirtle is blue.", "Squirtle")
    assert "Squirtle" not in out and "Wartortle" not in out and "squirtle" not in out
    assert out.startswith("This Pokémon") and "another Pokémon" in out
    sel = select_items("color", 3, 0, 42, 0.2)
    leaks = [it.name for it in sel.items if len(it.name) >= 3 and it.name.lower() in it.text.lower()]
    assert not leaks, f"names left in text: {leaks[:5]}"


def test_extract_json():
    assert extract_json('Sure!\n```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('{"a": {"b": 2}}') == {"a": {"b": 2}}


def test_vectorize_zscores_and_weights():
    tmp = Path(tempfile.mkdtemp())
    try:
        f1 = Feature(id="a", definition="first thing", kind="scale", weight=2.0)
        f2 = Feature(id="b", definition="constant thing", kind="binary", weight=1.0)
        store = ScoreStore(tmp / "s.jsonl")
        store.put_many([(f1.key, i, v) for i, v in enumerate([0.0, 0.5, 1.0])] + [(f2.key, i, 1.0) for i in range(3)])
        V = vectorize(Schema(features=[f1, f2]), store, 3)
        assert np.allclose(V[:, 1], 0) and np.isclose(V[:, 0].std(), 2.0)
        assert len(ScoreStore(tmp / "s.jsonl").scores) == 6     # reloads from disk
    finally:
        shutil.rmtree(tmp)


def test_score_key_depends_on_criterion():
    f = Feature(id="a", definition="body colour is blue", kind="scale")
    assert score_key(f, "") == f.key                       # strategy A keys unchanged
    assert score_key(f, "group by colour") != f.key
    assert score_key(f, "group by colour") != score_key(f, "group by shape")


def test_dry_run_end_to_end():
    _dry_run("schema")


def test_dry_run_hypothesis_end_to_end():
    _dry_run("hypothesis")


def test_resume_continues_identically():
    """Stop after round 2, resume to round 4. Round 3's feedback depends only on restored state (round-2
    clusters + Jay's rng), so it must equal an uninterrupted run's; rounds 1-2 must be untouched."""
    import run_experiment

    tmp = Path(tempfile.mkdtemp())
    try:
        common = ["--dry-run", "--workers", "1"]      # 1 worker: the fake LLM's answers then don't depend on thread order
        run_experiment.main(common + ["--rounds", "4", "--output-dir", str(tmp / "full")])
        run_experiment.main(common + ["--rounds", "2", "--output-dir", str(tmp / "split")])
        first = np.load(tmp / "split" / "triplets.npy")
        run_experiment.main(common + ["--rounds", "4", "--output-dir", str(tmp / "split"), "--resume"])
        full, split = np.load(tmp / "full" / "triplets.npy"), np.load(tmp / "split" / "triplets.npy")
        assert len(split) == len(full) == 128
        assert np.array_equal(split[:64], first) and np.array_equal(split[:96], full[:96])
        with (tmp / "split" / "history.csv").open(encoding="utf-8") as f:
            hist = list(csv.DictReader(f))
        assert [h["round"] for h in hist] == ["0", "1", "2", "3", "4"]
        assert int(hist[-1]["llm_calls"]) > int(hist[2]["llm_calls"])        # usage keeps counting up
        assert len(json.loads((tmp / "split" / "edits.json").read_text(encoding="utf-8"))) == 4
    finally:
        shutil.rmtree(tmp)


def _dry_run(strategy):
    import run_experiment

    tmp = Path(tempfile.mkdtemp())
    try:
        run_experiment.main(["--dry-run", "--rounds", "2", "--output-dir", str(tmp), "--workers", "2",
                             "--strategy", strategy])
        with (tmp / "history.csv").open(encoding="utf-8") as f:
            hist = list(csv.DictReader(f))
        assert [h["round"] for h in hist] == ["0", "1", "2"]
        if (JAY_RUN / "history.csv").exists():
            jay_cols = set(open(JAY_RUN / "history.csv", encoding="utf-8").readline().strip().split(","))
            missing = jay_cols - set(hist[0]) - {"last_epoch_loss"}
            assert not missing, missing
        sel = select_items("color", 3, 0, 42, 0.2)
        triplets = np.load(tmp / "triplets.npy")
        assert len(triplets) == 64
        assert not sel.test_mask[triplets].any(), "feedback mentioned a held-out item"
        calls = [json.loads(line) for line in (tmp / "llm_calls.jsonl").open(encoding="utf-8")]
        names = {it.name.lower() for it in sel.items if len(it.name) >= 4}
        for c in calls:
            words = set(c["prompt"].lower().replace(",", " ").replace(".", " ").split())
            assert not (words & names), f"name leaked into a {c['task']} prompt: {words & names}"
        if strategy == "hypothesis":
            assert {c["task"] for c in calls} >= {"hyp_design", "hyp_revise", "score"}
            scored = [c for c in calls if c["task"] == "score"]
            assert all("being grouped by the following criterion" in c["prompt"] for c in scored)
            edits = json.loads((tmp / "edits.json").read_text(encoding="utf-8"))
            assert all(e["criterion"] for e in edits)
        else:
            assert not any("being grouped by" in c["prompt"] for c in calls)
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
