"""Item selection (identical to Jay's tests/triplet_by_column.py), raw data join, anonymization, images.

Selection reproduces Jay's script exactly (same rng calls in the same order), so the items, labels,
held-out test mask and AUC pairs match results/triplet_by_column/<run>, and the returned `rng` is in
the same state Jay's feedback loop starts from.
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.metrics import sample_pairs  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "pokemon_data" / "manifest.jsonl"
METADATA = REPO_ROOT / "data" / "pokemon_data" / "metadata.jsonl"


def load_jay_module():
    """Import tests/triplet_by_column.py by path (a plain `import tests...` would be ambiguous with
    llm_experimentation/tests). Teammate code is imported, never modified."""
    name = "_jay_triplet_by_column"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "tests" / "triplet_by_column.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Item:
    idx: int            # row in the selected set (what labels / feedback indices refer to)
    anon_id: str        # what the LLM sees, e.g. "item_317"
    dex_num: str
    name: str
    label: str          # hidden ground truth (never shown to the LLM)
    image_path: Path
    text: str           # raw description, name-redacted

@dataclass
class Selection:
    items: list[Item]
    y: np.ndarray
    test_mask: np.ndarray
    auc_pairs: np.ndarray | None
    rng: np.random.Generator   # Jay's rng, already advanced past selection: use it for feedback
    n_classes: int


def select_items(column: str = "color", top_labels: int = 3, sample_size: int = 0, seed: int = 42,
                 test_fraction: float = 0.2, db: Path | None = None) -> Selection:
    jay = load_jay_module()
    db = db or (REPO_ROOT / "data" / "pokemon_db")

    # --- verbatim logic from tests/triplet_by_column.py main() ---
    rng = np.random.default_rng(seed)
    X, metadata, y = jay.load_embeddings_and_targets(db, "image", column)   # X unused: row order only
    counts = Counter(y.tolist())
    if top_labels:
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
    test_rng = np.random.default_rng([seed, 1])
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[test_rng.choice(len(y), int(round(test_fraction * len(y))), replace=False)] = True
    auc_pairs = sample_pairs(np.flatnonzero(test_mask), 5000, test_rng) if test_mask.any() else None
    # --- end verbatim ---

    manifest = {row["dex_num"]: row for row in map(json.loads, MANIFEST.open(encoding="utf-8"))}
    redact = make_redactor()
    # anonymous ids follow a seeded random permutation: dex order would leak evolution families
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
    """Replace the item's own name with 'this Pokémon' and every other Pokémon name with 'another
    Pokémon', so neither the item's identity nor its evolution family is handed over by name."""
    names = sorted({row["name"] for row in map(json.loads, METADATA.open(encoding="utf-8"))}, key=len, reverse=True)
    names = [n for n in names if len(n) >= 3]
    pattern = re.compile(r"(?<!\w)(" + "|".join(re.escape(n) for n in names) + r")(?!\w)", flags=re.I)

    def replace(m, own_name):
        word = "this Pokémon" if m.group(0).lower() == own_name.lower() else "another Pokémon"
        before = m.string[:m.start()].rstrip()
        return word[0].upper() + word[1:] if not before or before[-1] in ".!?" else word

    def redact(text: str, own_name: str) -> str:
        return pattern.sub(lambda m: replace(m, own_name), text)

    return redact


def load_png(path: Path, px: int, cache_dir: Path) -> bytes:
    """Image downscaled so its longest side is `px`, as PNG bytes (cached on disk)."""
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
