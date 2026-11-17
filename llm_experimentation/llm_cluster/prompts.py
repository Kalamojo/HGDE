"""Prompts (identical for every model) and the pydantic models their JSON replies must satisfy.

The LLM is never told the hidden criterion, the labels, the number of groups k-means will use, or
any Pokémon name. It sees anonymous ids, images and/or name-redacted descriptions, and Jay's
"A should be closer to P than to N" feedback.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .client import Image, Text
from .items import Item, load_png
from .schema import Feature, Schema

PROMPT_VERSION = "v2"   # v2: strategy B (hypothesis) prompts; schema-strategy prompts unchanged from v1

MAX_NEW_FEATURES_PER_ROUND = 5


# ---------- reply formats ----------

class DesignReply(BaseModel):
    notes: str = ""
    features: list[Feature] = Field(min_length=3, max_length=40)


class ScoreRow(BaseModel):
    item: str
    scores: dict[str, float]


class ScoreReply(BaseModel):
    rows: list[ScoreRow]


class HypothesisReply(BaseModel):
    """Strategy B: the full state, returned at design time and after every round of feedback."""
    criterion: str = Field(min_length=3, max_length=1000)
    dimensions: list[Feature] = Field(min_length=3, max_length=8)
    notes: str = ""


class ReviseReply(BaseModel):
    notes: str = ""
    weights: dict[str, float] = Field(default_factory=dict)
    drop: list[str] = Field(default_factory=list)
    add: list[Feature] = Field(default_factory=list, max_length=MAX_NEW_FEATURES_PER_ROUND)


# ---------- shared pieces ----------

def system_prompt(inputs: str) -> str:
    what = {"image": "an image", "image+text": "an image and a written description",
            "text": "a written description"}[inputs]
    return (
        "You are helping organize a collection of items into groups. Each item is identified only by an "
        f"anonymous id (like item_042) and is shown to you as {what}. Use only what is shown. "
        "Always answer with exactly one JSON object and nothing else."
    )


class ItemRenderer:
    """Turns an item into prompt parts according to the arm (image / image+text / text)."""

    def __init__(self, inputs: str, image_px: int, cache_dir):
        self.inputs, self.image_px, self.cache_dir = inputs, image_px, cache_dir

    def parts(self, item: Item, extra: str = "") -> list:
        out = [Text(f"\n--- {item.anon_id} ---\n")]
        if self.inputs in ("image", "image+text"):
            out.append(Image(load_png(item.image_path, self.image_px, self.cache_dir), label=item.anon_id))
        if self.inputs in ("image+text", "text"):
            out.append(Text(f"\nDescription: {item.text}\n"))
        if extra:
            out.append(Text("\n" + extra + "\n"))
        return out


def describe_schema(schema: Schema, with_weights: bool = True) -> str:
    lines = []
    for f in schema.features:
        w = f"  weight={f.weight:g}" if with_weights else ""
        lines.append(f"- {f.id} ({f.kind}){w}: {f.definition}")
    return "\n".join(lines)


FEATURE_RULES = (
    "Each feature must be judgeable for ANY single item on its own, from what is shown for that item, "
    "without comparing to other items. kind is \"scale\" (a score from 0 to 1, define what 0 and 1 mean) or "
    "\"binary\" (0 or 1). Feature ids are short snake_case names (lowercase letters, digits, underscores)."
)


# ---------- the three tasks ----------

def design_prompt(sample: list[Item], n_total: int, renderer: ItemRenderer) -> list:
    parts = [Text(
        f"Below are {len(sample)} items sampled from a collection of {n_total} similar items.\n\n"
        "Design a set of 15 to 30 features that together capture the main ways these items could naturally "
        "be grouped. The features will be scored for every item, and items will then be grouped by how "
        f"similar their feature vectors are.\n\n{FEATURE_RULES}\n\n"
        "Reply with JSON of the form:\n"
        '{"notes": "<your thoughts on how these items could be grouped>", '
        '"features": [{"id": "...", "definition": "...", "kind": "scale" | "binary", "weight": 1.0}, ...]}\n'
        "\nThe items:"
    )]
    for item in sample:
        parts += renderer.parts(item)
    return parts


def score_prompt(features: list[Feature], batch: list[Item], renderer: ItemRenderer, criterion: str = "") -> list:
    feats = "\n".join(f"- {f.id} ({f.kind}): {f.definition}" for f in features)
    context = (f"These items are being grouped by the following criterion: {criterion}\n"
               "Score every dimension with respect to this criterion.\n\n") if criterion.strip() else ""
    parts = [Text(
        context +
        "Score each of the items below on each of these features:\n"
        f"{feats}\n\n"
        "Judge each item on its own. Scale features take a number from 0 to 1; binary features take 0 or 1.\n"
        "Reply with JSON of the form:\n"
        '{"rows": [{"item": "<item id>", "scores": {"<feature id>": <number>, ...}}, ...]}\n'
        f"with one row for every one of the {len(batch)} items and a score for every feature.\n\nThe items:"
    )]
    for item in batch:
        parts += renderer.parts(item)
    return parts


def revise_prompt(schema: Schema, round_number: int, new_triplet_lines: list[str], earlier_triplet_lines: list[str],
                  involved: list[Item], involved_values: dict[int, str], satisfied_all: float | None,
                  satisfied_new: float | None, renderer: ItemRenderer) -> list:
    sat = ""
    if satisfied_all is not None:
        sat = (f"\nIn your current feature space, {satisfied_new:.0%} of this round's statements and "
               f"{satisfied_all:.0%} of all statements so far are satisfied.\n")
    earlier = "\n".join(earlier_triplet_lines) if earlier_triplet_lines else "(none)"
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
        f"- \"weights\": new weights (0 to 5) for existing features; 0 effectively turns a feature off\n"
        "- \"drop\": ids of features to remove\n"
        f"- \"add\": up to {MAX_NEW_FEATURES_PER_ROUND} new features (new, unused ids). {FEATURE_RULES}\n\n"
        f"Current features:\n{describe_schema(schema)}\n\n"
        f"Your notes so far: {schema.notes or '(none)'}\n\n"
        f"Feedback from earlier rounds:\n{earlier}\n\n"
        f"Feedback this round (round {round_number}):\n" + "\n".join(new_triplet_lines) + "\n" + sat +
        "\nThe items mentioned this round, with their current raw feature scores:"
    )]
    for item in involved:
        parts += renderer.parts(item, extra=f"current scores: {involved_values[item.idx]}")
    parts.append(Text(
        "\n\nReply with JSON of the form:\n"
        '{"notes": "<your updated hypothesis about the user\'s criterion>", '
        '"weights": {"<feature id>": <number>, ...}, "drop": ["<feature id>", ...], '
        '"add": [{"id": "...", "definition": "...", "kind": "scale" | "binary", "weight": 1.0}, ...]}'
    ))
    return parts


# ---------- strategy B: hypothesis + rescore ----------

HYPOTHESIS_FORMAT = (
    "Reply with JSON of the form:\n"
    '{"criterion": "<one or two sentences: how the items should be grouped>", '
    '"dimensions": [{"id": "...", "definition": "...", "kind": "scale" | "binary", "weight": 1.0}, ...], '
    '"notes": "<your reasoning>"}\n'
    "Give 3 to 8 dimensions that together measure the criterion; weights from 0 to 5."
)


def hypothesis_design_prompt(sample: list[Item], n_total: int, renderer: ItemRenderer) -> list:
    parts = [Text(
        f"Below are {len(sample)} items sampled from a collection of {n_total} similar items.\n\n"
        "Decide how these items would most naturally be grouped, and state it as a CRITERION (one or two "
        "sentences). Then give a few DIMENSIONS that measure that criterion. Every item will be scored on "
        "the dimensions with your criterion as context, and items with similar scores will be grouped "
        f"together.\n\n{FEATURE_RULES}\n\n{HYPOTHESIS_FORMAT}\n\nThe items:"
    )]
    for item in sample:
        parts += renderer.parts(item)
    return parts


def hypothesis_revise_prompt(schema: Schema, round_number: int, new_triplet_lines: list[str],
                             earlier_triplet_lines: list[str], involved: list[Item], involved_values: dict[int, str],
                             satisfied_all: float | None, satisfied_new: float | None, renderer: ItemRenderer) -> list:
    sat = ""
    if satisfied_all is not None:
        sat = (f"\nUnder your current criterion, {satisfied_new:.0%} of this round's statements and "
               f"{satisfied_all:.0%} of all statements so far are satisfied.\n")
    earlier = "\n".join(earlier_triplet_lines) if earlier_triplet_lines else "(none)"
    parts = [Text(
        "A user is organizing this collection into groups according to a criterion they have not told you. "
        "You keep a hypothesis of that criterion. Every item is scored on your dimensions with your criterion "
        "as context; scores are standardized across items, multiplied by the weights, and items with similar "
        "vectors are grouped together.\n\n"
        "After seeing the current grouping, the user gives feedback as statements \"A should be closer to B "
        "than to C\". Use it to work out what the user's criterion really is, then return a complete new "
        "criterion and dimension set (keep, reword or replace anything). It must hold for every item in the "
        "collection, including items the user never mentions. You cannot set any item's scores: every item "
        "is rescored by an independent scorer that sees only your criterion, your dimension definitions and "
        f"the item.\n{FEATURE_RULES}\n\n"
        f"Current criterion: {schema.criterion}\n"
        f"Current dimensions:\n{describe_schema(schema)}\n\n"
        f"Your notes so far: {schema.notes or '(none)'}\n\n"
        f"Feedback from earlier rounds:\n{earlier}\n\n"
        f"Feedback this round (round {round_number}):\n" + "\n".join(new_triplet_lines) + "\n" + sat +
        "\nThe items mentioned this round, with their current scores:"
    )]
    for item in involved:
        parts += renderer.parts(item, extra=f"current scores: {involved_values[item.idx]}")
    parts.append(Text("\n\n" + HYPOTHESIS_FORMAT))
    return parts
