"""What is being clustered, and the simulated user who gives feedback.

This plays the role of src/utils/data/benchmarks.py (DataSampler) in Kolade's framework, but the item
selection and the feedback are Jay's (tests/triplet_by_column.py), reproduced exactly, so every LLM
run sees the same items, the same held-out test set and the same kind of hints as Jay's model.

Nothing in this file talks to an LLM.
"""

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:          # so `src.utils`, `clustering` and `scripts` import from anywhere
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.metrics import sample_pairs  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "pokemon_data" / "manifest.jsonl"   # dex_num -> image path + description
METADATA = REPO_ROOT / "data" / "pokemon_data" / "metadata.jsonl"   # every Pokémon name (for redaction)


def load_jay_module():
    """Import tests/triplet_by_column.py by file path and return it as a module.

    A plain `import tests.triplet_by_column` would be ambiguous (llm_experimentation/tests also
    exists). Teammate code is only imported, never modified. Cached in sys.modules after the first call."""
    name = "_jay_triplet_by_column"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tests" / "triplet_by_column.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ---------- items ----------

@dataclass
class Item:
    idx: int            # row in the selected set; labels, feedback triplets and vectors all use this index
    anon_id: str        # the only name the LLM ever sees, e.g. "item_317"
    dex_num: str
    name: str           # real name: logged for us, never shown to the LLM
    label: str          # hidden ground truth (e.g. its colour), never shown to the LLM
    image_path: Path
    text: str           # its written description, with every Pokémon name redacted


@dataclass
class Selection:
    """Everything about the dataset one run uses."""
    items: list[Item]
    y: np.ndarray                # true labels, one per item
    test_mask: np.ndarray        # True = held-out item: never in any hint, only used for test metrics
    auc_pairs: np.ndarray | None # fixed pairs of test items for test_pair_auc
    rng: np.random.Generator     # Jay's rng, already advanced past selection: the feedback draws from it
    n_classes: int               # k, the number of clusters k-means is asked for


def select_items(column: str = "color", top_labels: int = 3, sample_size: int = 0, seed: int = 42,
                 test_fraction: float = 0.2, db: Path | None = None) -> Selection:
    """Pick the items exactly as Jay's script does (same rng calls in the same order), so the items,
    labels, test mask and AUC pairs match results/triplet_by_column/<run>, and `rng` ends in the
    state Jay's feedback loop starts from. Then attach each item's image, redacted text and anonymous id."""
    jay = load_jay_module()
    db = db or (REPO_ROOT / "data" / "pokemon_db")

    # --- verbatim logic from tests/triplet_by_column.py main() ---
    rng = np.random.default_rng(seed)
    X, metadata, y = jay.load_embeddings_and_targets(db, "image", column)   # X unused: only the row order matters
    counts = Counter(y.tolist())
    if top_labels:                                  # keep only the N most common label values
        top_values = sorted(counts, key=lambda value: (-counts[value], value))[:top_labels]
        eligible = np.flatnonzero(np.isin(y, top_values))
    else:
        eligible = np.arange(len(y))
    if sample_size:
        if sample_size > len(eligible):
            raise ValueError(f"sample-size={sample_size} exceeds {len(eligible)} eligible rows")
        selected = rng.choice(eligible, size=sample_size, replace=False)
    else:
        selected = eligible
    y = y[selected]
    metadata = [metadata[int(i)] for i in selected]
    test_rng = np.random.default_rng([seed, 1])     # a separate stream, so the test set doesn't move the main rng
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_rng.choice(len(y), int(round(test_fraction * len(y))), replace=False)] = True
    auc_pairs = sample_pairs(np.flatnonzero(test_mask), 5000, test_rng) if test_mask.any() else None
    # --- end verbatim ---

    manifest = {row["dex_num"]: row for row in map(json.loads, MANIFEST.open(encoding="utf-8"))}
    redact = make_redactor()
    # Anonymous ids come from a seeded random permutation. Numbering in dex order would leak
    # evolution families (neighbouring numbers are usually the same family, often the same colour).
    perm = np.random.default_rng([seed, 3]).permutation(len(y))
    items = []
    for i, meta in enumerate(metadata):
        row = manifest.get(meta["dex_num"])
        if row is None:
            raise RuntimeError(f"dex {meta['dex_num']} ({meta['name']}) has no manifest entry / image")
        image_path = REPO_ROOT / row["image_path"]
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        items.append(Item(idx=i, anon_id=f"item_{perm[i]:03d}", dex_num=meta["dex_num"], name=meta["name"],
                          label=str(y[i]), image_path=image_path, text=redact(row["text"], row["name"])))
    return Selection(items, y, test_mask, auc_pairs, rng, len(np.unique(y)))


def make_redactor():
    """Return redact(text, own_name). It replaces the item's own name with "this Pokémon" and every
    other Pokémon name with "another Pokémon", so the text gives away neither the item's identity nor
    its evolution family. Names shorter than 3 characters are skipped (they would match ordinary words)."""
    names = sorted({row["name"] for row in map(json.loads, METADATA.open(encoding="utf-8"))}, key=len, reverse=True)
    names = [n for n in names if len(n) >= 3]
    # longest names first, so "Mewtwo" is matched before "Mew"; (?<!\w)/(?!\w) = whole words only
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(n) for n in names) + r")(?!\w)", flags=re.I)

    def replace(m, own_name):
        word = "this Pokémon" if m.group(0).lower() == own_name.lower() else "another Pokémon"
        before = m.string[:m.start()].rstrip()
        # capitalize at the start of the text or of a sentence
        return word[0].upper() + word[1:] if not before or before[-1] in ".!?" else word

    def redact(text: str, own_name: str) -> str:
        return pattern.sub(lambda m: replace(m, own_name), text)

    return redact


def load_png(path: Path, px: int, cache_dir: Path) -> bytes:
    """The image at `path` shrunk so its longest side is `px` pixels, as PNG bytes. Cached on disk
    under cache_dir/img_<px>/, so each size is only computed once."""
    cached = cache_dir / f"img_{px}" / (path.stem + ".png")
    if cached.exists():
        return cached.read_bytes()
    from PIL import Image as PILImage

    with PILImage.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((px, px), PILImage.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="PNG", optimize=True)
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_bytes(buf.getvalue())
    return buf.getvalue()


# ---------- the simulated user ----------

def next_feedback(predicted: np.ndarray, y: np.ndarray, rng: np.random.Generator, used_anchors: set,
                  triplets_per_round: int, test_mask: np.ndarray):
    """One round of hints: Jay's select_feedback, called exactly as his loop calls it.

    It looks at the current clusters (`predicted`) versus the true labels, picks one wrongly placed
    training item as the anchor A (not used before), and returns (anchor, kind, triplets): up to
    `triplets_per_round` (A, P, N) index triplets meaning "A should be closer to P than to N", where P
    shares A's true label and N doesn't. Held-out test items never appear. None = nothing left to fix."""
    jay = load_jay_module()
    return jay.select_feedback(predicted, y, rng, used_anchors, triplets_per_round, test_mask)


def render_triplets(triplets, items: list[Item]) -> list[str]:
    """Each (A, P, N) triplet as the sentence the LLM reads. Order and duplicates are kept: Jay samples
    triplets with replacement, so a repeated triplet carries extra weight in his loss too."""
    return [f"{items[a].anon_id} should be closer to {items[p].anon_id} than to {items[n].anon_id}"
            for a, p, n in triplets]
