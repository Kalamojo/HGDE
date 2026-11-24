"""llm_cluster/open_strategy.py."""

import helpers
from helpers import ScriptedClient
from llm_cluster.open_strategy import Method, OpenStrategy


def method(refs=()):
    return Method(approach="by colour", instructions="count the colours", outputs=["a", "b"], reference_items=list(refs))


def test_check_method(toy_sel, tmp_path):
    s = helpers.make_strategy(OpenStrategy, toy_sel, ScriptedClient(), tmp_path)
    a = toy_sel.items[0].anon_id
    assert s.check_method(method([a])) is None
    assert "unknown" in s.check_method(method(["item_999"])) and "distinct" in s.check_method(method([a, a]))


def test_bad_rows_become_zeros_only_for_that_item(toy_sel, tmp_path):
    bad = toy_sel.items[0].anon_id

    def reply(task, meta):                               # item 0 always gets 3 numbers instead of 2
        return {"rows": [{"item": a, "vector": [1.0] * (3 if a == bad else meta["d"])} for a in meta["item_ids"]]}

    s = helpers.make_strategy(OpenStrategy, toy_sel, ScriptedClient(responder=reply), tmp_path)
    M = s.method_matrix(method())
    assert M[0].tolist() == [0.0, 0.0] and (M[1:] == 1.0).all() and s.filled == 2
