"""One client for every model, via the Amazon Bedrock Converse API.

Converse uses the same message format (text + image blocks) for Claude and OpenAI models, so
switching models only changes the model ID. Auth: set AWS_BEARER_TOKEN_BEDROCK (a Bedrock API key);
boto3 >= 1.39 picks it up automatically. Normal AWS credentials also work.

Content passed to `complete` is a provider-neutral list of parts:
    Text("...")                       -> {"text": ...}
    Image(png_bytes, label="item_012") -> {"image": {"format": "png", "source": {"bytes": ...}}}
"""

from __future__ import annotations

import json
import random
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path

from .models import ModelSpec


@dataclass
class Text:
    text: str


@dataclass
class Image:
    png: bytes
    label: str = ""


class BudgetExceeded(RuntimeError):
    pass


class QuotaExhausted(RuntimeError):
    """Bedrock's daily token quota for this model is used up; waiting minutes won't help."""


class _UsageLog:
    """Thread-safe token / cost accounting plus a jsonl log of every call (images replaced by labels)."""

    def __init__(self, spec: ModelSpec, log_path: Path | None):
        self.spec = spec
        self.log_path = log_path
        self.lock = threading.Lock()
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    @property
    def usd(self) -> float:
        if self.spec.usd_in_per_m is None or self.spec.usd_out_per_m is None:
            return float("nan")
        return (self.input_tokens * self.spec.usd_in_per_m + self.output_tokens * self.spec.usd_out_per_m) / 1e6

    def record(self, task: str, system: str, parts: list, response: str, usage: dict, stop: str, seconds: float):
        with self.lock:
            self.calls += 1
            self.input_tokens += int(usage.get("inputTokens", 0))
            self.output_tokens += int(usage.get("outputTokens", 0))
            if self.log_path is not None:
                prompt = "".join(p.text if isinstance(p, Text) else f"[image {p.label}]" for p in parts)
                row = {"call": self.calls, "task": task, "model": self.spec.model_id, "stop_reason": stop,
                       "usage": usage, "seconds": round(seconds, 2), "system": system, "prompt": prompt,
                       "response": response}
                with self.log_path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")


class BedrockClient:
    def __init__(self, spec: ModelSpec, region: str, log_path: Path | None = None,
                 max_usd: float | None = None, max_retries: int = 16):
        import boto3  # lazy: --dry-run works without boto3 installed
        from botocore.config import Config

        self.spec = spec
        self.log = _UsageLog(spec, log_path)
        self.max_usd = max_usd
        self.max_retries = max_retries
        self.runtime = boto3.client(
            "bedrock-runtime", region_name=region,
            config=Config(read_timeout=900, connect_timeout=30, retries={"max_attempts": 2, "mode": "standard"}),
        )

    def _blocks(self, parts: list) -> list[dict]:
        blocks = []
        for p in parts:
            if isinstance(p, Text):
                if p.text:
                    blocks.append({"text": p.text})
            elif isinstance(p, Image):
                if not self.spec.supports_images:
                    raise ValueError(f"model {self.spec.name} does not accept images")
                blocks.append({"image": {"format": "png", "source": {"bytes": p.png}}})
        return blocks

    def complete(self, system: str, parts: list, max_tokens: int = 32000, task: str = "", meta: dict | None = None) -> str:
        if self.max_usd is not None and self.log.usd == self.log.usd and self.log.usd > self.max_usd:
            raise BudgetExceeded(f"spent ${self.log.usd:.2f} > --max-usd {self.max_usd}")
        from botocore.exceptions import ClientError

        request = dict(modelId=self.spec.model_id, system=[{"text": system}],
                       messages=[{"role": "user", "content": self._blocks(parts)}],
                       inferenceConfig={"maxTokens": max_tokens})
        for attempt in range(self.max_retries):
            start = time.time()
            try:
                resp = self.runtime.converse(**request)
                break
            except ClientError as e:
                code = e.response.get("Error", {}).get("Code", "")
                message = e.response.get("Error", {}).get("Message", "")
                if code == "ThrottlingException" and "per day" in message.lower():
                    raise QuotaExhausted(f"{self.spec.model_id}: {message}") from e
                retryable = code in {"ThrottlingException", "ServiceUnavailableException", "ModelNotReadyException",
                                     "InternalServerException", "ModelErrorException", "ServiceQuotaExceededException"}
                if not retryable or attempt == self.max_retries - 1:
                    raise
                # rate limits (tokens/requests per minute) clear within a minute or two, so wait long enough
                wait = min(90.0, 5 * 2 ** attempt) * (0.75 + 0.5 * random.random())
                print(f"  [{task}] {code}, retry {attempt + 1}/{self.max_retries - 1} in {wait:.0f}s", flush=True)
                time.sleep(wait)
        # reasoning blocks ("reasoningContent") are skipped; only the answer text is used
        text = "".join(b["text"] for b in resp["output"]["message"]["content"] if "text" in b)
        self.log.record(task, system, parts, text, resp.get("usage", {}), resp.get("stopReason", ""), time.time() - start)
        return text


class FakeClient:
    """Offline stand-in for --dry-run: returns valid but random JSON, so the whole loop, logging and
    metrics can be tested for free. It needs `meta` (what the caller expects) to build its answer."""

    def __init__(self, spec: ModelSpec, log_path: Path | None = None, seed: int = 0):
        self.spec = spec
        self.log = _UsageLog(spec, log_path)
        self.rng = random.Random(seed)

    def complete(self, system: str, parts: list, max_tokens: int = 32000, task: str = "", meta: dict | None = None) -> str:
        meta = meta or {}
        r = self.rng
        if task == "design":
            feats = [{"id": f"feat_{i}", "definition": f"random feature {i}", "kind": r.choice(["scale", "binary"]),
                      "weight": 1.0} for i in range(12)]
            out = {"notes": "fake design", "features": feats}
        elif task == "score":
            out = {"rows": [{"item": item, "scores": {f: round(r.random(), 2) for f in meta["feature_ids"]}}
                            for item in meta["item_ids"]]}
        elif task in ("hyp_design", "hyp_revise"):
            n = meta.get("round", 0)
            out = {"criterion": f"fake criterion {r.randint(0, 2) if n else 0}", "notes": f"fake hypothesis {n}",
                   "dimensions": [{"id": f"dim_{i}", "definition": f"random dimension {i}", "kind": "scale",
                                   "weight": round(r.uniform(0.5, 2.0), 2)} for i in range(r.randint(3, 6))]}
        elif task == "revise":
            ids = meta["feature_ids"]
            n_new = meta.get("round", 0)
            out = {"notes": f"fake revision {n_new}",
                   "weights": {f: round(r.uniform(0.2, 2.0), 2) for f in r.sample(ids, min(3, len(ids)))},
                   "drop": [], "add": [{"id": f"new_{n_new}", "definition": f"new random feature {n_new}",
                                        "kind": "scale", "weight": 1.0}]}
        else:
            out = {}
        text = "```json\n" + json.dumps(out) + "\n```"
        approx = sum(len(p.text) // 4 if isinstance(p, Text) else 90 for p in parts)
        self.log.record(task, system, parts, text, {"inputTokens": approx, "outputTokens": len(text) // 4}, "end_turn", 0.0)
        return text


def make_client(spec: ModelSpec, region: str, log_path: Path | None, max_usd: float | None, seed: int = 0):
    if spec.name == "fake":
        return FakeClient(spec, log_path, seed)
    return BedrockClient(spec, region, log_path, max_usd)


def extract_json(text: str) -> dict:
    """Parse the single JSON object in a reply, tolerating ```json fences and stray prose."""
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    candidate = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    if not candidate:
        raise ValueError("no JSON object found in reply")
    return json.loads(candidate)
