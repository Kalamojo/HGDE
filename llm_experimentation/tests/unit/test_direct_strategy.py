"""llm_cluster/direct_strategy.py."""

import numpy as np

import helpers
from helpers import ScriptedClient
from llm_cluster.direct_strategy import DirectStrategy, Grouping, validate_grouping
from llm_cluster.harness import RoundInput

IDS = [f"item_{i:03d}" for i in range(6)]


def test_validate_grouping():
    assert validate_grouping({"a": IDS[:2], "b": IDS[2:4], "c": IDS[4:]}, IDS, 3) is None
    assert "exactly 3" in validate_grouping({"a": IDS[:3], "b": IDS[3:]}, IDS, 3)
    err = validate_grouping({"a": IDS[:3], "b": IDS[2:4], "c": [IDS[4], "item_999"]}, IDS, 3)
    assert "item_002" in err and "item_999" in err and "missing 1" in err


def by_label(sel):
    return Grouping(groups={lab: [it.anon_id for it in sel.items if it.label == lab] for lab in helpers.LABELS})


def test_groups_to_vectors_and_back(toy_sel, tmp_path):
    s = helpers.make_strategy(DirectStrategy, toy_sel, ScriptedClient(), tmp_path)
    V = s.embed(by_label(toy_sel))
    assert V.shape == (30, 3) and (V.sum(axis=1) == 1).all()
    assert s.cluster(V).tolist() == [0] * 10 + [1] * 10 + [2] * 10      # items 0-9 red, 10-19 blue, 20-29 green


def test_invalid_reply_keeps_the_current_grouping(toy_sel, tmp_path):
    bad = {"action": "commit", "groups": {"a": [toy_sel.items[0].anon_id]}}
    s = helpers.make_strategy(DirectStrategy, toy_sel, ScriptedClient([bad, bad]), tmp_path)
    g = by_label(toy_sel)
    V = s.embed(g)
    chosen, changes = s.update(g, RoundInput(1, [(0, 1, 12)], [], V, s.cluster(V), [], []))
    assert chosen is g and changes["committed"] == "unchanged"
