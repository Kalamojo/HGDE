"""llm_cluster/llm.py: parsing replies, the cache, the hint check, ask, the draft loop, the budget stop."""

import numpy as np
import pytest
from pydantic import BaseModel

import helpers
from helpers import ScriptedClient
from llm_cluster.llm import MARGIN, MODELS, BedrockClient, BudgetExceeded, JsonlCache, LLMStrategy, Text, UsageLog, \
    extract_json, score_against_hints


def test_extract_json():
    assert extract_json('Sure!\n```json\n{"a": {"b": 1}}\n```') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        extract_json("no json here")


def test_jsonl_cache_reloads_from_disk(tmp_path):
    c = JsonlCache(tmp_path / "v.jsonl", "vector")
    c.put_many([("k", 0, [1.0]), ("k", 2, [3.0])])
    again = JsonlCache(tmp_path / "v.jsonl", "vector")
    assert again.values == c.values and again.missing("k", [0, 1, 2]) == [1]


def test_hint_satisfied_exactly_at_the_margin():
    E = np.array([[0.0], [0.5], [0.5 + MARGIN], [0.5 + MARGIN - 0.01]])     # A, P, N far enough, N too close
    res = score_against_hints(E, all_triplets=[(0, 1, 2), (0, 1, 3)], new_triplets=[(0, 1, 3)])
    assert res["sat_all"] == 0.5 and res["sat_new"] == 0.0 and res["violated"] == [(0, 1, 3)]


class Reply(BaseModel):
    value: int


def strategy(toy_sel, tmp_path, replies=(), drafts=3):
    return helpers.make_strategy(LLMStrategy, toy_sel, ScriptedClient(list(replies)), tmp_path, drafts=drafts)


def test_ask_retries_once_with_the_error(toy_sel, tmp_path):
    s = strategy(toy_sel, tmp_path, ["not json", {"value": 4}])
    assert s.ask([Text("q")], Reply, "t", {}).value == 4
    assert "Your previous reply was not valid" in s.client.calls[1]["prompt"]
    s = strategy(toy_sel, tmp_path, [{"value": -1}, {"value": -2}])
    assert s.ask([Text("q")], Reply, "t", {}, check=lambda r: "negative" if r.value < 0 else None) is None


@pytest.mark.parametrize("steps, expected", [
    ([("commit", "A")], ("A", "new")),
    ([("test", "A"), ("test", "B"), ("commit_draft", 1)], ("A", "draft 1")),
    ([("test", "A"), ("test", "B"), ("test", "C")], ("C", "new")),          # the last step is applied
    ([("test", "A"), ("test", "B"), None], ("B", "draft 2")),               # LLM failed: best tested draft
    ([None], (None, "unchanged")),
])
def test_draft_loop(toy_sel, tmp_path, steps, expected):
    s = strategy(toy_sel, tmp_path)
    scores = {"A": 0.2, "B": 0.8, "C": 0.5}
    chosen, how, _ = s.draft_loop(lambda step, drafts: steps[step - 1], lambda v: {"sat_all": scores[v]})
    assert (chosen, how) == expected


def test_budget_stops_before_calling():
    c = BedrockClient.__new__(BedrockClient)            # no boto3 client: the call must not happen
    c.spec, c.max_usd, c.log = MODELS["sonnet"], 1.0, UsageLog(MODELS["sonnet"], None)
    c.log.input_tokens = 1_000_000                       # $2 spent
    with pytest.raises(BudgetExceeded):
        c.complete("sys", [Text("q")])
