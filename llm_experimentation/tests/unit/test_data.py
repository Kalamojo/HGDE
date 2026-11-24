"""llm_cluster/data.py: the items and the simulated user."""

import numpy as np

import helpers
from llm_cluster.data import make_redactor, next_feedback, select_items
from src.utils.data.benchmarks import PokemonBenchmark


def test_selection_is_the_benchmark(needs_real_data):
    sel = select_items("color", ["red", "blue"], 0, 0.2)
    assert len(sel.items) == 261 and int(sel.test_mask.sum()) == 52 and sel.n_classes == 2
    # the same items, labels and test split as run_benchmarks.py's benchmark1
    b = PokemonBenchmark(helpers.REPO_ROOT / "data" / "pokemon_db", "image_vector", "color", 0.2, categories=["red", "blue"])
    assert [it.label for it in sel.items] == b.true_labels.tolist() and (sel.test_mask == b.test_mask).all()
    assert [it.dex_num for it in sel.items] == [m["dex_num"] for m in b.metadata]
    ids = [it.anon_id for it in sel.items]
    assert len(set(ids)) == len(ids) and ids != sorted(ids), "anonymous ids must be unique and not in dex order"


def test_redaction(needs_real_data):
    redact = make_redactor()
    out = redact("Squirtle evolves into Wartortle. squirtle is blue.", "Squirtle")
    assert out == "This Pokémon evolves into another Pokémon. This Pokémon is blue."
    assert redact("Mewtwo and Mew", "Mew") == "Another Pokémon and this Pokémon"     # longest name first


def test_feedback_never_uses_test_items(toy_sel):
    pred = np.random.default_rng(0).integers(0, 3, 30)
    anchor, target, triplets = next_feedback(toy_sel.benchmark, pred, 8)
    assert 0 < len(triplets) <= 8 and all(a == anchor and pred[p] == target != pred[n] for a, p, n in triplets)
    assert not toy_sel.test_mask[np.array(triplets)].any()


def test_no_feedback_once_the_clustering_is_right(toy_sel):
    perfect = np.unique(toy_sel.y, return_inverse=True)[1]
    assert next_feedback(toy_sel.benchmark, perfect, 8) is None
