"""The LLM's feature schema, the on-disk score cache, and vectorization.

The LLM controls only the schema (feature definitions + weights). Item scores come from a separate
scoring call that sees one feature definition and the item's raw data, so a revision can never
hand-edit individual items: that's what makes test ARI a measure of generalization.
"""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, field_validator

MAX_WEIGHT = 5.0


class Feature(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    definition: str = Field(min_length=3, max_length=600)
    kind: Literal["scale", "binary"]
    weight: float = Field(default=1.0, ge=0.0, le=MAX_WEIGHT)

    @property
    def key(self) -> str:
        """Scores are cached per definition, not per id: rewording a feature means rescoring it."""
        return score_key(self, "")


def score_key(feature: "Feature", criterion: str = "") -> str:
    """Cache key for a feature's scores. Strategy B scores items *under a criterion*, so the criterion
    is part of the key (a new hypothesis rescores everything); with criterion "" (strategy A) the key
    is the same as before."""
    base = f"{feature.kind}|{feature.definition.strip()}"
    if criterion.strip():
        base = f"{criterion.strip()}||{base}"
    return hashlib.sha1(base.encode()).hexdigest()[:12]


class Schema(BaseModel):
    features: list[Feature]
    notes: str = ""
    criterion: str = ""     # strategy B's hypothesis of the user's criterion ("" for strategy A)

    @field_validator("features")
    @classmethod
    def _unique_ids(cls, v):
        ids = [f.id for f in v]
        if len(ids) != len(set(ids)):
            raise ValueError("feature ids must be unique")
        return v

    def by_id(self) -> dict[str, Feature]:
        return {f.id: f for f in self.features}


class ScoreStore:
    """(feature key, item idx) -> score in [0, 1], appended to a jsonl file as scores arrive."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()
        self.scores: dict[tuple[str, int], float] = {}
        if path.exists():
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                self.scores[(row["key"], row["item"])] = row["score"]

    def missing(self, feature: Feature, n_items: int, criterion: str = "") -> list[int]:
        key = score_key(feature, criterion)
        return [i for i in range(n_items) if (key, i) not in self.scores]

    def put_many(self, rows: list[tuple[str, int, float]]):
        with self.lock, self.path.open("a", encoding="utf-8") as f:
            for key, item, score in rows:
                self.scores[(key, item)] = score
                f.write(json.dumps({"key": key, "item": item, "score": score}) + "\n")

    def get(self, feature: Feature, item: int, criterion: str = "") -> float:
        return self.scores[(score_key(feature, criterion), item)]


def raw_matrix(schema: Schema, store: ScoreStore, n_items: int) -> np.ndarray:
    return np.array([[store.get(f, i, schema.criterion) for f in schema.features] for i in range(n_items)], dtype=np.float64)


def vectorize(schema: Schema, store: ScoreStore, n_items: int) -> np.ndarray:
    """z-score each feature across all items (constant features become 0), then multiply by weight."""
    M = raw_matrix(schema, store, n_items)
    std = M.std(axis=0)
    Z = np.where(std > 1e-9, (M - M.mean(axis=0)) / np.where(std > 1e-9, std, 1.0), 0.0)
    return Z * np.array([f.weight for f in schema.features])
