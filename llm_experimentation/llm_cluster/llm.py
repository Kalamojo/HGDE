"""What the three LLM strategies share: the model list, the two clients, prompt helpers, and LLMStrategy.

Auth: set AWS_BEARER_TOKEN_BEDROCK (a Bedrock API key); boto3 picks it up. Normal AWS credentials also work.

No prompt ever contains the hidden criterion, the labels or any Pokémon name.
"""

from __future__ import annotations

import json
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ValidationError

from .data import Item, Selection, load_png
from .harness import RoundInput, StopRun, Strategy

MARGIN = 0.2              # Jay's triplet margin: a hint (A, P, N) holds when |A-N| - |A-P| >= MARGIN
MAX_VIOLATED_SHOWN = 40   # violated hints listed back to the LLM (open / direct)
IMAGE_PX = 256            # longest side of each item image sent to the LLM
BATCH = 20                # items per scoring / executor call
MAX_TOKENS = 32000        # reply token limit per call


# ---------- models ----------

@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str                   # Bedrock model id / inference profile; edit here to try another one
    supports_images: bool
    usd_in_per_m: float | None      # USD per million input tokens; None = cost not tracked
    usd_out_per_m: float | None


# Check the ids against your account before a real run (python llm_experimentation/check_models.py).
MODELS = {
    "sonnet": ModelSpec("sonnet", "global.anthropic.claude-sonnet-4-6", True, 2.0, 10.0),
    "opus": ModelSpec("opus", "global.anthropic.claude-opus-4-6", True, 4.0, 20.0),
    #"terra": ModelSpec("terra", "global.openai.gpt-6.1-terra", True, None, None),
    "fake": ModelSpec("fake", "fake", True, 0.0, 0.0),     # offline, used by --dry-run
}


# ---------- clients ----------

@dataclass
class Text:
    text: str


@dataclass
class Image:
    png: bytes
    label: str = ""                 # what llm_calls.jsonl shows instead of the bytes


class BudgetExceeded(StopRun):
    """Spend passed --max-usd."""


class UsageLog:
    """Counts calls, tokens and cost, and appends every call to llm_calls.jsonl (images shown by label)."""

    def __init__(self, spec: ModelSpec, log_path: Path | None):
        self.spec, self.log_path = spec, log_path
        self.lock = threading.Lock()        # calls come from several worker threads
        self.calls = self.input_tokens = self.output_tokens = 0

    @property
    def usd(self) -> float:
        """Cost so far (NaN when the model's prices are unknown)."""
        if self.spec.usd_in_per_m is None:
            return float("nan")
        return (self.input_tokens * self.spec.usd_in_per_m + self.output_tokens * self.spec.usd_out_per_m) / 1e6

    def record(self, task: str, parts: list, response: str, usage: dict):
        with self.lock:
            self.calls += 1
            self.input_tokens += usage.get("inputTokens", 0)
            self.output_tokens += usage.get("outputTokens", 0)
            if self.log_path:
                prompt = "".join(p.text if isinstance(p, Text) else f"[image {p.label}]" for p in parts)
                row = {"call": self.calls, "task": task, "usage": usage, "prompt": prompt, "response": response}
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


class BedrockClient:
    """One-turn calls through the Bedrock Converse API (the same format for Claude and OpenAI models).
    Rate limits are retried by botocore itself. Any other error ends the run; every finished round is
    saved, so --resume continues from there."""

    def __init__(self, spec: ModelSpec, region: str, log_path: Path | None = None, max_usd: float | None = None):
        import boto3  # imported here so --dry-run works without boto3
        from botocore.config import Config

        self.spec, self.max_usd = spec, max_usd
        self.log = UsageLog(spec, log_path)
        self.runtime = boto3.client("bedrock-runtime", region_name=region, config=Config(
            read_timeout=900, retries={"max_attempts": 10, "mode": "adaptive"}))

    def complete(self, system: str, parts: list, task: str = "", meta: dict | None = None) -> str:
        """Send one prompt (Text / Image parts) and return the reply text. `meta` is only for FakeClient."""
        if self.max_usd is not None and self.log.usd > self.max_usd:      # False when usd is NaN
            raise BudgetExceeded(f"spent ${self.log.usd:.2f} > --max-usd {self.max_usd}")
        content = [{"text": p.text} if isinstance(p, Text) else {"image": {"format": "png", "source": {"bytes": p.png}}}
                   for p in parts if isinstance(p, Image) or p.text]
        resp = self.runtime.converse(modelId=self.spec.model_id, system=[{"text": system}],
                                     messages=[{"role": "user", "content": content}],
                                     inferenceConfig={"maxTokens": MAX_TOKENS})
        text = "".join(b["text"] for b in resp["output"]["message"]["content"] if "text" in b)   # skips reasoning
        self.log.record(task, parts, text, resp.get("usage", {}))
        return text


class FakeClient:
    """Offline stand-in for --dry-run and the tests: a random but valid JSON reply for every task.
    `meta` tells it what the caller expects (item ids, feature ids, ...)."""

    def __init__(self, spec: ModelSpec, log_path: Path | None = None, seed: int = 0):
        self.log = UsageLog(spec, log_path)
        self.rng = random.Random(seed)

    def complete(self, system: str, parts: list, task: str = "", meta: dict | None = None) -> str:
        meta, r = meta or {}, self.rng
        if task == "design":
            out = {"notes": "fake design", "features": [
                {"id": f"feat_{i}", "definition": f"random feature {i}", "kind": "scale", "weight": 1.0} for i in range(12)]}
        elif task == "score":
            out = {"rows": [{"item": a, "scores": {f: round(r.random(), 2) for f in meta["feature_ids"]}}
                            for a in meta["item_ids"]]}
        elif task == "revise":
            ids, rnd = meta["feature_ids"], meta["round"]
            out = {"notes": f"fake revision {rnd}", "weights": {f: round(r.uniform(0.2, 2), 2) for f in r.sample(ids, 3)},
                   "add": [{"id": f"new_{rnd}", "definition": f"new random feature {rnd}", "kind": "scale"}]}
        elif task == "open_execute":
            out = {"rows": [{"item": a, "vector": [round(r.random(), 3) for _ in range(meta["d"])]} for a in meta["item_ids"]]}
        elif task in ("open_design", "open_revise"):
            method = {"approach": "fake approach", "instructions": "random numbers",
                      "outputs": [f"number {j}" for j in range(r.randint(2, 6))],
                      "reference_items": r.sample(meta["ref_pool"], min(2, len(meta["ref_pool"])))}
            out = method if task == "open_design" else self.step(meta, {"method": method})
        elif task in ("direct_design", "direct_revise"):
            ids = list(meta["item_ids"])
            r.shuffle(ids)
            if r.random() < 0.2:
                ids.pop()                         # sometimes leave an item out, to exercise the retry
            grouping = {"approach": "fake clustering", "groups": {f"g{c + 1}": ids[c::meta["k"]] for c in range(meta["k"])}}
            out = grouping if task == "direct_design" else self.step(meta, grouping)
        else:
            out = {}
        text = json.dumps(out)
        self.log.record(task, parts, text, {"inputTokens": len(str(parts)) // 4, "outputTokens": len(text) // 4})
        return text

    def step(self, meta, body):
        """A random draft-loop reply: test, commit, or commit an already-tested draft."""
        if meta["n_tested"] and self.rng.random() < 0.3:
            return {"action": "commit_draft", "draft": self.rng.randint(1, meta["n_tested"])}
        return {"action": self.rng.choice(["test", "test", "commit"]), **body}


def make_client(spec: ModelSpec, region: str, log_path: Path | None, max_usd: float | None, seed: int = 0):
    if spec.name == "fake":
        return FakeClient(spec, log_path, seed)
    return BedrockClient(spec, region, log_path, max_usd)


def extract_json(text: str) -> dict:
    """The JSON object in a reply, tolerating ```json fences and text around it."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found in reply")
    return json.loads(text[start:end + 1])


# ---------- prompt helpers ----------

def system_prompt(inputs: str) -> str:
    what = {"image": "an image", "image+text": "an image and a written description", "text": "a written description"}
    return ("You are helping organize a collection of items into groups. Each item is identified only by an "
            f"anonymous id (like item_042) and is shown to you as {what[inputs]}. Use only what is shown. "
            "Always answer with exactly one JSON object and nothing else.")


def item_parts(item: Item, inputs: str, cache_dir: Path, extra: str = "") -> list:
    """One item as prompt parts: its anonymous id, then its image and/or description, then `extra`."""
    parts = [Text(f"\n--- {item.anon_id} ---\n")]
    if inputs != "text":
        parts.append(Image(load_png(item.image_path, IMAGE_PX, cache_dir), label=item.anon_id))
    if inputs != "image":
        parts.append(Text(f"\nDescription: {item.text}\n"))
    if extra:
        parts.append(Text(f"\n{extra}\n"))
    return parts


def hint_lines(triplets, items: list[Item]) -> list[str]:
    return [f"  {items[a].anon_id} should be closer to {items[p].anon_id} than to {items[n].anon_id}" for a, p, n in triplets]


def score_against_hints(E: np.ndarray, all_triplets, new_triplets) -> dict:
    """How well vectors E fit the hints: the share satisfied among all hints and among this round's,
    and the distinct violated hints (this round's first)."""
    def violated(t):
        a, p, n = t
        return np.linalg.norm(E[a] - E[p]) - np.linalg.norm(E[a] - E[n]) + MARGIN > 1e-6

    def share(triplets):
        return float(np.mean([not violated(t) for t in triplets])) if len(triplets) else float("nan")

    distinct = dict.fromkeys(tuple(int(i) for i in t) for t in list(new_triplets) + list(all_triplets))
    return {"sat_all": share(all_triplets), "sat_new": share(new_triplets),
            "violated": [t for t in distinct if violated(t)]}


class JsonlCache:
    """(key, item index) -> value the LLM produced, kept in memory and appended to a jsonl file.
    The key is a hash of a feature's / method's definition, so unchanged ones are never re-run."""

    def __init__(self, path: Path, field: str):
        self.path, self.field = path, field
        self.lock = threading.Lock()
        self.values = {}
        if path.exists():
            for line in path.open(encoding="utf-8"):
                row = json.loads(line)
                self.values[(row["key"], row["item"])] = row[field]

    def missing(self, key: str, items) -> list[int]:
        return [i for i in items if (key, i) not in self.values]

    def put_many(self, rows: list[tuple[str, int, object]]):
        with self.lock, self.path.open("a", encoding="utf-8") as f:
            for key, item, value in rows:
                self.values[(key, item)] = value
                f.write(json.dumps({"key": key, "item": item, self.field: value}) + "\n")

    def get(self, key: str, item: int):
        return self.values[(key, item)]


# ---------- the base class of the three LLM strategies ----------

@dataclass
class LLMSettings:
    inputs: str                     # what the LLM sees per item: "image", "image+text" or "text"
    seed: int = 42
    drafts: int = 3                 # open / direct: steps per round (test drafts, then commit)
    workers: int = 4                # parallel scoring / executor calls (speed only)


class LLMStrategy(Strategy):
    def __init__(self, sel: Selection, client, cfg: LLMSettings, out_dir: Path, cache_dir: Path):
        self.client, self.cfg, self.out, self.cache_dir = client, cfg, out_dir, cache_dir
        self.items, self.n, self.k, self.seed = sel.items, len(sel.items), sel.n_classes, cfg.seed
        self.by_anon = {it.anon_id: it.idx for it in self.items}     # "item_317" -> row index
        self.system = system_prompt(cfg.inputs)
        self.filled = 0             # values the LLM never gave, filled with a default

    def item_parts(self, item: Item, extra: str = "") -> list:
        return item_parts(item, self.cfg.inputs, self.cache_dir, extra)

    def ask(self, parts: list, reply_model: type[BaseModel], task: str, meta: dict, check=None):
        """One call whose reply must parse as `reply_model` and pass `check(reply)` (an error message or
        None). If not, it is asked once more with the error added. Returns the reply, or None."""
        prompt = parts
        for _ in range(2):
            text = self.client.complete(self.system, prompt, task=task, meta=meta)
            try:
                reply = reply_model.model_validate(extract_json(text))
                error = check(reply) if check else None
                if error is None:
                    return reply
            except (ValueError, ValidationError) as e:
                error = str(e)[:1500]
            prompt = parts + [Text(f"\n\nYour previous reply was not valid: {error}\nReply again with only the corrected JSON.")]
        return None

    def run_batches(self, todo: list[int], fn):
        """fn(batch) on chunks of BATCH item indices, --workers at a time."""
        with ThreadPoolExecutor(self.cfg.workers) as pool:
            list(pool.map(fn, [todo[s:s + BATCH] for s in range(0, len(todo), BATCH)]))

    def design_sample(self, size: int = 40) -> list[Item]:
        """The fixed random items shown when designing the round-0 state."""
        idx = np.random.default_rng([self.seed, 2]).choice(self.n, min(size, self.n), replace=False)
        return [self.items[i] for i in sorted(idx)]

    def draft_loop(self, ask_step, test):
        """One round of up to --drafts steps.

        ask_step(step, drafts) asks the LLM and returns (action, x) or None if it failed:
            ("test", version)       score the version against the hints; the LLM sees the result next step
            ("commit", version)     apply it now
            ("commit_draft", n)     apply the n-th tested draft
        On the last step the version is applied whatever the action. test(version) -> dict with "sat_all".
        Returns (chosen version or None, how it was chosen, the tested drafts)."""
        drafts = []
        for step in range(1, self.cfg.drafts + 1):
            reply = ask_step(step, drafts)
            if reply is None:
                break
            action, x = reply
            if action == "commit_draft":
                return drafts[x - 1]["version"], f"draft {x}", drafts
            if action == "commit" or step == self.cfg.drafts:
                return x, "new", drafts
            drafts.append({"draft": len(drafts) + 1, "version": x, **test(x)})
        if drafts:                                          # the LLM failed: use its best tested draft
            best = max(drafts, key=lambda d: d["sat_all"])
            return best["version"], f"draft {best['draft']}", drafts
        return None, "unchanged", drafts

    @staticmethod
    def step_error(reply, n_tested: int) -> str | None:
        """The draft-loop rule every reply must follow: commit_draft must name a tested draft."""
        if reply.action == "commit_draft" and not (reply.draft and 1 <= reply.draft <= n_tested):
            return f"draft must be one of the tested drafts 1..{n_tested}" if n_tested else "nothing tested yet"
        return None

    @staticmethod
    def past_rounds(inp: RoundInput, what) -> list[str]:
        """One line per earlier round: what(changes), the share of hints satisfied after it, its approach."""
        sat = {int(h["round"]): h["all_triplets_satisfied"] for h in inp.history}
        lines = []
        for e in inp.edits:
            s = sat.get(int(e["round"]), "")
            ch = e["changes"]
            lines.append(f"round {e['round']}: {what(ch)}; afterwards {f'{float(s):.0%}' if s != '' else '?'} of all "
                         f"hints so far were satisfied. Approach: {str(ch.get('approach', ''))[:300]}")
        return lines

    # ---------- harness hooks ----------

    def usage_columns(self) -> dict:
        log = self.client.log
        return {"filled": self.filled, "llm_calls": log.calls, "input_tokens": log.input_tokens,
                "output_tokens": log.output_tokens, "usd_total": round(log.usd, 4)}

    def restore(self, last_row: dict) -> None:
        """--resume: continue the counters, so cost and --max-usd count the whole run."""
        log = self.client.log
        self.filled = int(last_row.get("filled") or 0)
        log.calls, log.input_tokens, log.output_tokens = (int(last_row.get(c) or 0)
                                                          for c in ("llm_calls", "input_tokens", "output_tokens"))

    def spent(self) -> str:
        usd = self.client.log.usd
        return f"spent ${usd:.2f}" if usd == usd else f"{self.client.log.input_tokens} input tokens"
