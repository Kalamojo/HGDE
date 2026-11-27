"""llm_cluster/direct_strategy.py."""

import numpy as np

import helpers
from helpers import ScriptedClient
from llm_cluster.direct_strategy import DirectStrategy, Grouping, validate_grouping
from llm_cluster.harness import RoundInput

IDS = [f"item_{i:03d}" for i in range(6)]


def test_validate_grouping():
    # --clusters k: exactly k groups
    assert validate_grouping({"a": IDS[:2], "b": IDS[2:4], "c": IDS[4:]}, IDS, 3) is None
    assert "exactly 3" in validate_grouping({"a": IDS[:3], "b": IDS[3:]}, IDS, 3)
    err = validate_grouping({"a": IDS[:3], "b": IDS[2:4], "c": [IDS[4], "item_999"]}, IDS, 3)
    assert "item_002" in err and "item_999" in err and "missing 1" in err
    # --clusters auto (k = None): any number of groups from 2 up, still every id exactly once
    assert validate_grouping({"a": IDS[:3], "b": IDS[3:]}, IDS, None) is None
    assert validate_grouping({c: [i] for c, i in zip("abcdef", IDS)}, IDS, None) is None
    assert "at least 2" in validate_grouping({"a": IDS}, IDS, None)
    assert "missing 1" in validate_grouping({"a": IDS[:3], "b": IDS[3:5]}, IDS, None)


def by_label(sel):
    return Grouping(groups={lab: [it.anon_id for it in sel.items if it.label == lab] for lab in helpers.LABELS})


def test_groups_to_vectors_and_back(toy_sel, tmp_path):
    s = helpers.make_strategy(DirectStrategy, toy_sel, ScriptedClient(), tmp_path)
    V = s.embed(by_label(toy_sel))
    assert V.shape == (30, 3) and (V.sum(axis=1) == 1).all()
    assert s.cluster(V, None).tolist() == [0] * 10 + [1] * 10 + [2] * 10      # items 0-9 red, 10-19 blue, 20-29 green


def test_invalid_reply_keeps_the_current_grouping(toy_sel, tmp_path):
    bad = {"action": "commit", "groups": {"a": [toy_sel.items[0].anon_id]}}
    s = helpers.make_strategy(DirectStrategy, toy_sel, ScriptedClient([bad, bad]), tmp_path)
    g = by_label(toy_sel)
    V = s.embed(g)
    chosen, changes = s.update(g, RoundInput(1, [(0, 1, 12)], [], V, s.cluster(V, None), [], []))
    assert chosen is g and changes["committed"] == "unchanged"


def test_auto_mode_lets_the_llm_pick_the_number_of_groups(toy_sel, tmp_path):
    five = {"groups": {f"g{c}": [it.anon_id for it in toy_sel.items[c::5]] for c in range(5)}}
    s = helpers.make_strategy(DirectStrategy, toy_sel, ScriptedClient([five]), tmp_path)   # clusters="auto"
    V = s.embed(s.start())
    assert s.k is None and V.shape == (30, 5) and "as many groups as you judge best" in s.client.calls[0]["prompt"]


def test_k_mode_asks_for_exactly_k_groups(toy_sel, tmp_path):
    s = helpers.make_strategy(DirectStrategy, toy_sel, ScriptedClient([{"groups": by_label(toy_sel).groups}]),
                              tmp_path, clusters="k")
    assert s.k == 3 and s.embed(s.start()).shape == (30, 3) and "exactly 3 groups" in s.client.calls[0]["prompt"]
