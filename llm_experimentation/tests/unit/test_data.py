"""llm_cluster/data.py: the items and the simulated user."""

import csv

import numpy as np

import helpers
from llm_cluster.data import make_redactor, next_feedback, select_items


def test_selection_matches_jay(needs_real_data):
    sel = select_items("color", 3, 0, 42, 0.2)
    assert len(sel.items) == 439 and int(sel.test_mask.sum()) == 88 and sel.n_classes == 3
    jay = helpers.REPO_ROOT / "results" / "triplet_by_column" / "color_3c_all" / "assignments.csv"
    if jay.exists():
        with jay.open(encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert [r["dex_num"] for r in rows] == [it.dex_num for it in sel.items]
        assert [r["color"] for r in rows] == [it.label for it in sel.items]
    ids = [it.anon_id for it in sel.items]
    assert len(set(ids)) == len(ids) and ids != sorted(ids), "anonymous ids must be unique and not in dex order"


def test_redaction(needs_real_data):
    redact = make_redactor()
    out = redact("Squirtle evolves into Wartortle. squirtle is blue.", "Squirtle")
    assert out == "This Pokémon evolves into another Pokémon. This Pokémon is blue."
    assert redact("Mewtwo and Mew", "Mew") == "Another Pokémon and this Pokémon"     # longest name first


def test_feedback_never_uses_test_items(toy_sel):
    pred = np.random.default_rng(0).integers(0, 3, 30)
    anchor, kind, triplets = next_feedback(pred, toy_sel.y, toy_sel.rng, set(), 8, toy_sel.test_mask)
    y = toy_sel.y
    assert all(a == anchor and y[a] == y[p] != y[n] for a, p, n in triplets)
    assert not toy_sel.test_mask[np.array(triplets)].any()
