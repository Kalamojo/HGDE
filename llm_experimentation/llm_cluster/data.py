"""What is being clustered, and the simulated user who gives feedback.

Both come from Kolade's src/utils framework, exactly as experiments/run_benchmarks.py uses them for Jay's
and John's models: the items, labels and held-out test set are a PokemonBenchmark, the feedback is
Benchmark.sample_triplets (what TripletEval calls), and the fixed AUC pairs are ClusterState's. So every
LLM run is scored on the same terms as those models.

Nothing in this file talks to an LLM.
"""

from __future__ import annotations

import io
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:          # so `src.utils` imports from anywhere
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.data.benchmarks import Benchmark, PokemonBenchmark  # noqa: E402
from src.utils.state import ClusterState  # noqa: E402

MANIFEST = REPO_ROOT / "data" / "pokemon_data" / "manifest.jsonl"   # dex_num -> image path + description
METADATA = REPO_ROOT / "data" / "pokemon_data" / "metadata.jsonl"   # every Pokémon name (for redaction)


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
    benchmark: Benchmark         # gives the feedback (sample_triplets, drawing from benchmark.rng)
    n_classes: int               # number of true classes: k for --clusters k


def select_items(column: str = "color", categories: list[str] | None = ("red", "blue"), seed: int = 0,
                 test_fraction: float = 0.2, db: Path | None = None) -> Selection:
    """The items of a PokemonBenchmark built exactly as run_benchmarks.py builds it (categories=None:
    every value of `column`), with each item's image, redacted text and anonymous id attached."""
    db = db or (REPO_ROOT / "data" / "pokemon_db")
    benchmark = PokemonBenchmark(db, "image_vector", column, test_fraction,       # the vectors are unused
                                 categories=list(categories) if categories else None, seed=seed)
    y, test_mask = benchmark.true_labels.astype(str), benchmark.test_mask
    auc_pairs = ClusterState(benchmark, eval=None, adapter=None).auc_pairs        # the pairs Jay's runs use
    metadata = benchmark.metadata

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
    return Selection(items, y, test_mask, auc_pairs, benchmark, len(benchmark.unique_labels))


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

def next_feedback(benchmark: Benchmark, predicted: np.ndarray, triplets_per_round: int):
    """One round of hints: Benchmark.sample_triplets, the call TripletEval makes for Jay's model.

    It picks one training item A that shares its true label with an item in another cluster, and returns
    (anchor, target_cluster, triplets): up to `triplets_per_round` (A, P, N) index triplets meaning "A should
    be closer to P than to N", with P from the cluster A should move to (target_cluster) and N from A's
    current cluster. Held-out test items never appear. None = nothing left to fix."""
    triplets = benchmark.sample_triplets(predicted, triplets_per_round)
    if not triplets:
        return None
    triplets = [tuple(int(i) for i in t) for t in triplets]
    anchor, positive = triplets[0][0], triplets[0][1]
    return anchor, int(predicted[positive]), triplets


def render_triplets(triplets, items: list[Item]) -> list[str]:
    """Each (A, P, N) triplet as the sentence the LLM reads."""
    return [f"{items[a].anon_id} should be closer to {items[p].anon_id} than to {items[n].anon_id}"
            for a, p, n in triplets]
