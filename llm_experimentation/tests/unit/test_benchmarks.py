"""src/utils/data/benchmarks.py: the held-out test split, and that no test item is ever in the feedback."""

import random

import numpy as np

import helpers  # noqa: F401
from src.utils.data.benchmarks import Benchmark


def make_sampler(test_fraction):
    s = Benchmark.__new__(Benchmark)             # skip __init__ (it needs LanceDB)
    s.true_labels = np.array(["red", "blue", "green", "pink"] * 25, dtype=object)
    s.set_test_items(test_fraction, seed=0)
    return s


def test_test_split_size():
    assert make_sampler(0.2).test_mask.sum() == 20 and make_sampler(0.0).test_mask.sum() == 0


def test_test_items_never_in_feedback():
    random.seed(0)
    s = make_sampler(0.2)
    pred = np.random.default_rng(1).integers(0, 4, 100)
    links = s.fetch_must_links(pred) + s.fetch_must_not_links(pred)
    assert links and not {i for pair in links for i in pair} & s.test_items
    for _ in range(20):
        assert not {i for t in s.sample_triplets(pred) for i in t} & s.test_items
