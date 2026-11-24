"""llm_cluster/schema_strategy.py."""

import numpy as np

import helpers
from helpers import ScriptedClient
from llm_cluster.llm import JsonlCache
from llm_cluster.schema_strategy import Feature, ReviseReply, Schema, SchemaStrategy, vectorize


def feat(i, w=1.0):
    return Feature(id=i, definition=f"thing {i}", kind="scale", weight=w)


def test_vectorize_zscores_and_weights(tmp_path):
    a, b = feat("a", 2.0), feat("b")
    store = JsonlCache(tmp_path / "s.jsonl", "score")
    store.put_many([(a.key, i, v) for i, v in enumerate([0.0, 0.5, 1.0])] + [(b.key, i, 1.0) for i in range(3)])
    V = vectorize(Schema(features=[a, b]), store, 3)
    assert np.isclose(V[:, 0].std(), 2.0) and np.isclose(V[:, 0].mean(), 0) and np.allclose(V[:, 1], 0)


def test_apply_revision(toy_sel, tmp_path):
    s = helpers.make_strategy(SchemaStrategy, toy_sel, ScriptedClient(), tmp_path)
    schema = Schema(features=[feat("a"), feat("b"), feat("c")])
    new, _ = s.apply_revision(schema, ReviseReply(weights={"a": 99, "unknown": 1}, drop=["b"], add=[feat("d", 2.0)]))
    assert {f.id: f.weight for f in new.features} == {"a": 5.0, "c": 1.0, "d": 2.0}
    unchanged, _ = s.apply_revision(schema, ReviseReply(drop=["a", "b"]))            # would leave 1 active feature
    assert unchanged is schema


def test_missing_scores_are_retried_then_filled(toy_sel, tmp_path):
    """The LLM never scores one item: everything else keeps its real score, that item gets 0.5."""
    skipped = toy_sel.items[0].anon_id

    def reply(task, meta):
        return {"rows": [{"item": a, "scores": {f: 0.9 for f in meta["feature_ids"]}}
                         for a in meta["item_ids"] if a != skipped]}

    s = helpers.make_strategy(SchemaStrategy, toy_sel, ScriptedClient(responder=reply), tmp_path)
    schema = Schema(features=[feat("a"), feat("b")])
    s.score(schema.features)
    assert s.store.get(schema.features[0].key, 0) == 0.5 and s.store.get(schema.features[0].key, 1) == 0.9
    assert s.filled == 2 and len(s.client.calls) == 3          # 2 batches, then 1 retry for the missing item
