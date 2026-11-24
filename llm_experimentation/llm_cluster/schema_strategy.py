"""Strategy "schema": the LLM is a feature engineer; k-means does the clustering.

    round 0   the LLM sees 40 sample items and designs 15-30 features ("body is mostly blue, 0-1"),
              all at weight 1
    embed     separate scoring calls score EVERY item on every feature (0-1); each feature is z-scored
              across items and multiplied by its weight -> V; the run's clusterer (HDBSCAN or k-means) clusters V
    update    the LLM sees the hints, the share satisfied and the hinted items' scores, and may only
              change the feature set: reweight (0-5), drop, add up to 5 features. It can never set an
              item's scores, so test items are placed by the features alone (generalization).
"""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field, ValidationError, field_validator

from .data import Item, render_triplets
from .harness import RoundInput
from .llm import JsonlCache, LLMStrategy, Text, score_against_hints

MAX_WEIGHT = 5.0
MAX_NEW_FEATURES = 5


# ---------- state and reply formats ----------

class Feature(BaseModel):
    id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,47}$")
    definition: str = Field(min_length=3, max_length=600)
    kind: Literal["scale", "binary"]
    weight: float = Field(default=1.0, ge=0.0, le=MAX_WEIGHT)

    @property
    def key(self) -> str:
        """Cache key for its scores: a hash of kind + definition, so rewording a feature rescores it,
        while renaming or reweighting it costs nothing."""
        return hashlib.sha1(f"{self.kind}|{self.definition.strip()}".encode()).hexdigest()[:12]


class Schema(BaseModel):
    """The strategy's whole state (saved as schema_round_XX.json)."""
    features: list[Feature]
    notes: str = ""                 # the LLM's running notes / hypothesis

    @field_validator("features")
    @classmethod
    def _unique_ids(cls, v):
        if len({f.id for f in v}) != len(v):
            raise ValueError("feature ids must be unique")
        return v


class DesignReply(BaseModel):
    notes: str = ""
    features: list[Feature] = Field(min_length=3, max_length=40)


class ScoreRow(BaseModel):
    item: str
    scores: dict[str, float]


class ScoreReply(BaseModel):
    rows: list[ScoreRow]


class ReviseReply(BaseModel):
    notes: str = ""
    weights: dict[str, float] = {}
    drop: list[str] = []
    add: list[Feature] = Field(default=[], max_length=MAX_NEW_FEATURES)


# ---------- prompts ----------

FEATURE_RULES = (
    "Each feature must be judgeable for ANY single item on its own, from what is shown for that item, "
    "without comparing to other items. kind is \"scale\" (a score from 0 to 1, define what 0 and 1 mean) or "
    "\"binary\" (0 or 1). Feature ids are short snake_case names (lowercase letters, digits, underscores)."
)


def describe_schema(schema: Schema) -> str:
    return "\n".join(f"- {f.id} ({f.kind})  weight={f.weight:g}: {f.definition}" for f in schema.features)


def design_prompt(s: SchemaStrategy, sample: list[Item]) -> list:
    parts = [Text(
        f"Below are {len(sample)} items sampled from a collection of {s.n} similar items.\n\n"
        "Design a set of 15 to 30 features that together capture the main ways these items could naturally "
        "be grouped. The features will be scored for every item, and items will then be grouped by how "
        f"similar their feature vectors are.\n\n{FEATURE_RULES}\n\n"
        "Reply with JSON of the form:\n"
        '{"notes": "<your thoughts on how these items could be grouped>", '
        '"features": [{"id": "...", "definition": "...", "kind": "scale" | "binary", "weight": 1.0}, ...]}\n'
        "\nThe items:"
    )]
    for item in sample:
        parts += s.item_parts(item)
    return parts


def score_prompt(s: SchemaStrategy, features: list[Feature], batch: list[Item]) -> list:
    feats = "\n".join(f"- {f.id} ({f.kind}): {f.definition}" for f in features)
    parts = [Text(
        f"Score each of the items below on each of these features:\n{feats}\n\n"
        "Judge each item on its own. Scale features take a number from 0 to 1; binary features take 0 or 1.\n"
        "Reply with JSON of the form:\n"
        '{"rows": [{"item": "<item id>", "scores": {"<feature id>": <number>, ...}}, ...]}\n'
        f"with one row for every one of the {len(batch)} items and a score for every feature.\n\nThe items:"
    )]
    for item in batch:
        parts += s.item_parts(item)
    return parts


def revise_prompt(s: SchemaStrategy, schema: Schema, inp: RoundInput, involved: list[int], sat: dict) -> list:
    earlier = "\n".join(render_triplets(inp.earlier_triplets, s.items)) or "(none)"
    M = raw_matrix(schema, s.store, s.n)
    parts = [Text(
        "A user is organizing this collection into groups according to a criterion they have not told you. "
        "Items are compared using your features: each feature is standardized across all items and multiplied "
        "by its weight, and items with similar weighted vectors are grouped together.\n\n"
        "After seeing the current grouping, the user gives feedback as statements \"A should be closer to B "
        "than to C\". Your job is to revise the FEATURE SET so the user's preferences hold - not only for the "
        "items they mentioned, but for every item in the collection, including ones they never mention.\n\n"
        "You can only change the feature set. You cannot set any individual item's scores: any new feature "
        "is scored for every item by an independent scorer that sees only your definition and the item.\n"
        "Allowed changes:\n"
        f"- \"weights\": new weights (0 to {MAX_WEIGHT:g}) for existing features; 0 effectively turns a feature off\n"
        "- \"drop\": ids of features to remove\n"
        f"- \"add\": up to {MAX_NEW_FEATURES} new features (new, unused ids). {FEATURE_RULES}\n\n"
        f"Current features:\n{describe_schema(schema)}\n\n"
        f"Your notes so far: {schema.notes or '(none)'}\n\n"
        f"Feedback from earlier rounds:\n{earlier}\n\n"
        f"Feedback this round (round {inp.round}):\n" + "\n".join(render_triplets(inp.new_triplets, s.items)) + "\n"
        f"\nIn your current feature space, {sat['sat_new']:.0%} of this round's statements and "
        f"{sat['sat_all']:.0%} of all statements so far are satisfied.\n"
        "\nThe items mentioned this round, with their current raw feature scores:"
    )]
    for i in involved:
        scores = ", ".join(f"{f.id}={M[i, j]:.2f}" for j, f in enumerate(schema.features))
        parts += s.item_parts(s.items[i], extra=f"current scores: {scores}")
    parts.append(Text(
        "\n\nReply with JSON of the form:\n"
        '{"notes": "<your updated hypothesis about the user\'s criterion>", '
        '"weights": {"<feature id>": <number>, ...}, "drop": ["<feature id>", ...], '
        '"add": [{"id": "...", "definition": "...", "kind": "scale" | "binary", "weight": 1.0}, ...]}'
    ))
    return parts


# ---------- vectors ----------

def raw_matrix(schema: Schema, store: JsonlCache, n_items: int) -> np.ndarray:
    """n_items x n_features matrix of raw 0-1 scores."""
    return np.array([[store.get(f.key, i) for f in schema.features] for i in range(n_items)], dtype=float)


def vectorize(schema: Schema, store: JsonlCache, n_items: int) -> np.ndarray:
    """Each feature column z-scored across items (so every feature has the same spread; a constant
    column becomes 0), then multiplied by its weight."""
    M = raw_matrix(schema, store, n_items)
    std = M.std(axis=0)
    std[std < 1e-9] = np.inf                    # constant column: (M - mean) / inf = 0
    return (M - M.mean(axis=0)) / std * [f.weight for f in schema.features]


# ---------- the strategy ----------

class SchemaStrategy(LLMStrategy):
    state_cls = Schema
    name = "schema"

    def __init__(self, sel, client, cfg, out_dir, cache_dir):
        super().__init__(sel, client, cfg, out_dir, cache_dir)
        self.store = JsonlCache(out_dir / "scores.jsonl", "score")

    def start(self) -> Schema:
        """Round 0: design the first feature set; every weight starts at 1."""
        def check(reply):
            try:
                Schema(features=reply.features)        # e.g. duplicate ids
            except ValidationError as e:
                return str(e)

        reply = self.ask(design_prompt(self, self.design_sample()), DesignReply, "design", {}, check)
        if reply is None:
            raise RuntimeError("schema design failed twice; see llm_calls.jsonl")
        return Schema(features=[f.model_copy(update={"weight": 1.0}) for f in reply.features], notes=reply.notes)

    def score(self, features: list[Feature]):
        """Make sure every item has a score for every feature. Two passes over whatever is missing;
        a score still missing after that becomes 0.5 (counted in `filled`)."""
        for _ in range(2):
            gaps = [f for f in features if self.store.missing(f.key, range(self.n))]
            todo = sorted({i for f in gaps for i in self.store.missing(f.key, range(self.n))})
            if todo:
                self.run_batches(todo, lambda batch: self.score_batch(gaps, batch))
        rows = [(f.key, i, 0.5) for f in features for i in self.store.missing(f.key, range(self.n))]
        self.filled += len(rows)
        self.store.put_many(rows)

    def score_batch(self, features: list[Feature], batch: list[int]):
        """One scoring call; keeps every valid score in the reply."""
        items = [self.items[i] for i in batch]
        reply = self.ask(score_prompt(self, features, items), ScoreReply, "score",
                         {"item_ids": [it.anon_id for it in items], "feature_ids": [f.id for f in features]})
        if reply is None:
            return
        wanted, by_id = {it.anon_id: it.idx for it in items}, {f.id: f for f in features}
        self.store.put_many([(by_id[fid].key, wanted[row.item], float(np.clip(v, 0, 1)))
                             for row in reply.rows if row.item in wanted
                             for fid, v in row.scores.items() if fid in by_id])

    def embed(self, schema: Schema) -> np.ndarray:
        self.score(schema.features)
        return vectorize(schema, self.store, self.n)

    def update(self, schema: Schema, inp: RoundInput):
        """Show the LLM the hints and the hinted items' scores, and apply its revision."""
        new = inp.new_triplets
        involved = list(dict.fromkeys([t[0] for t in new] + [t[1] for t in new] + [t[2] for t in new]))
        sat = score_against_hints(inp.E, inp.earlier_triplets + new, new)
        reply = self.ask(revise_prompt(self, schema, inp, involved, sat), ReviseReply, "revise",
                         {"feature_ids": [f.id for f in schema.features], "round": inp.round})
        if reply is None:
            return schema, {"error": "revision failed twice; schema unchanged"}
        return self.apply_revision(schema, reply)

    def apply_revision(self, schema: Schema, reply: ReviseReply):
        """The revised schema. Changes naming unknown features are ignored; a revision that would leave
        fewer than 2 features with weight > 0 is refused (schema unchanged)."""
        feats = {f.id: f for f in schema.features}
        for fid, w in reply.weights.items():
            if fid in feats:
                feats[fid] = feats[fid].model_copy(update={"weight": float(np.clip(w, 0, MAX_WEIGHT))})
        for fid in reply.drop:
            feats.pop(fid, None)
        added = [f.id for f in reply.add if f.id not in feats]
        feats.update({f.id: f for f in reply.add if f.id in added})
        if sum(f.weight > 0 for f in feats.values()) < 2:
            return schema, {"error": "revision left fewer than 2 active features; schema unchanged"}
        new = Schema(features=list(feats.values()), notes=reply.notes or schema.notes)
        return new, {"weights": reply.weights, "drop": reply.drop, "add": added}

    def log_columns(self, schema: Schema) -> dict:
        return {"n_features": len(schema.features), "n_active_features": sum(f.weight > 0 for f in schema.features),
                **self.usage_columns()}

    def summary(self, schema: Schema, changes: dict) -> str:
        return f"+{len(changes.get('add', []))} features, {sum(f.weight > 0 for f in schema.features)} active, {self.spent()}"
