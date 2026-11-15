"""Jay's simulated user, reused unchanged, plus a lossless text rendering of its triplets."""

from __future__ import annotations

import numpy as np

from .items import Item, load_jay_module


def next_feedback(predicted: np.ndarray, y: np.ndarray, rng: np.random.Generator, used_anchors: set,
                  triplets_per_round: int, test_mask: np.ndarray):
    """tests/triplet_by_column.py select_feedback, called exactly as Jay's loop calls it.
    Returns (anchor, kind, triplets) or None."""
    jay = load_jay_module()
    return jay.select_feedback(predicted, y, rng, used_anchors, triplets_per_round, test_mask)


def render_triplets(triplets, items: list[Item]) -> list[str]:
    """One line per triplet, in order (duplicates kept: Jay samples with replacement, so a repeated
    triplet carries weight in his loss too)."""
    return [f"{items[a].anon_id} should be closer to {items[p].anon_id} than to {items[n].anon_id}"
            for a, p, n in triplets]
