"""src/utils/data/benchmarks.py: the held-out test split, that no test item is ever in the feedback,
that the same seed gives the same feedback, and that a perfect clustering gets none."""

import numpy as np

from helpers import make_toy_benchmark

LABELS = np.array(["red", "blue", "green", "pink"] * 25, dtype=object)


def test_test_split_size():
    assert make_toy_benchmark(LABELS, 0.2).test_mask.sum() == 20 and make_toy_benchmark(LABELS, 0.0).test_mask.sum() == 0


def test_test_items_never_in_feedback():
    s = make_toy_benchmark(LABELS, 0.2)
    pred = np.random.default_rng(1).integers(0, 4, 100)
    links = s.fetch_must_links(pred) + s.fetch_must_not_links(pred)
    assert links and not {i for pair in links for i in pair} & s.test_items
    for _ in range(20):
        assert not {i for t in s.sample_triplets(pred) for i in t} & s.test_items
        assert not set(s.sample_pair(pred)[:2]) & s.test_items


def test_same_seed_same_feedback():
    pred = np.random.default_rng(1).integers(0, 4, 100)
    a, b = make_toy_benchmark(LABELS, seed=3), make_toy_benchmark(LABELS, seed=3)
    assert [a.sample_triplets(pred) for _ in range(5)] == [b.sample_triplets(pred) for _ in range(5)]


def test_nothing_to_fix_gives_no_feedback():
    s = make_toy_benchmark(LABELS)
    perfect = np.unique(LABELS, return_inverse=True)[1]
    assert s.sample_triplets(perfect) == [] and s.sample_pair(perfect) is None
