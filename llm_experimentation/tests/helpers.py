"""Shared test helpers (conftest.py puts this folder on sys.path).

    make_toy_sel     a small synthetic Selection (no database): 30 items, 3 classes, a 20% test set
    ScriptedClient   a stand-in LLM that returns the replies a test gives it
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

LLM_DIR = Path(__file__).resolve().parents[1]          # llm_experimentation/
REPO_ROOT = LLM_DIR.parent
for p in (str(LLM_DIR), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from llm_cluster.data import Item, Selection  # noqa: E402
from llm_cluster.llm import MODELS, LLMSettings, UsageLog  # noqa: E402
from src.utils.metrics import sample_pairs  # noqa: E402

LABELS = ("red", "blue", "green")


def make_toy_sel(img_dir: Path, n_per_class: int = 10) -> Selection:
    """30 items (10 red, then 10 blue, then 10 green), each with a one-colour PNG. A new Selection (and
    feedback rng) every call, so two runs never share one."""
    from PIL import Image as PILImage

    y = np.repeat(LABELS, n_per_class)
    test_mask = np.zeros(len(y), dtype=bool)
    test_mask[np.random.default_rng(1).choice(len(y), len(y) // 5, replace=False)] = True
    perm = np.random.default_rng(3).permutation(len(y))
    items = []
    for i, label in enumerate(y):
        img = img_dir / f"{i:04d}_Toymon{i}.png"
        if not img.exists():
            PILImage.new("RGB", (40, 30), {"red": "red", "blue": "blue", "green": "green"}[label]).save(img)
        items.append(Item(idx=i, anon_id=f"item_{perm[i]:03d}", dex_num=f"{i:04d}", name=f"Toymon{i}",
                          label=str(label), image_path=img, text=f"A small creature, number {i}."))
    auc_pairs = sample_pairs(np.flatnonzero(test_mask), 500, np.random.default_rng(2))
    return Selection(items, y, test_mask, auc_pairs, np.random.default_rng(42), len(LABELS))


class ScriptedClient:
    """Same interface as FakeClient, but replies come from `replies` (a list of dicts / strings, used in
    order) or from `responder(task, meta)`. Every call's task and prompt are kept in .calls."""

    def __init__(self, replies=None, responder=None):
        self.replies, self.responder = list(replies or []), responder
        self.log = UsageLog(MODELS["fake"], None)
        self.calls = []

    def complete(self, system, parts, task="", meta=None):
        self.calls.append({"task": task, "prompt": "".join(getattr(p, "text", "") for p in parts)})
        reply = self.responder(task, meta or {}) if self.responder else self.replies.pop(0)
        text = reply if isinstance(reply, str) else json.dumps(reply)
        self.log.record(task, parts, text, {"inputTokens": 10, "outputTokens": 5})
        return text


def make_strategy(cls, sel, client, out_dir: Path, **settings):
    out_dir.mkdir(parents=True, exist_ok=True)
    return cls(sel, client, LLMSettings(inputs="text", workers=1, **settings), out_dir, out_dir / "cache")
