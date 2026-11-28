import boto3
from botocore.config import Config
from dataclasses import dataclass
from pathlib import Path
import json
import threading
from typing import Optional


MAX_TOKENS = 32000 # reply token limit per call

@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str                   # Bedrock model id / inference profile; edit here to try another one
    supports_images: bool
    region: str
    usd_in_per_m: Optional[float]   # USD per million input tokens; None = cost not tracked
    usd_out_per_m: Optional[float]

@dataclass
class Text:
    text: str

@dataclass
class Image:
    png: bytes
    label: str = ""                 # what llm_calls.jsonl shows instead of the bytes

class StopRun(Exception):
    """Raised by a strategy to end the run cleanly. The round in progress is not saved."""

class BudgetExceeded(StopRun):
    """Spend passed --max-usd."""

class UsageLog:
    """Counts calls, tokens and cost, and appends every call to llm_calls.jsonl (images shown by label)."""

    def __init__(self, spec: ModelSpec, log_path: Optional[Path]):
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

    def __init__(self, spec: ModelSpec, log_path: Optional[Path] = None, max_usd: Optional[float] = None):
        self.spec, self.max_usd = spec, max_usd
        self.log = UsageLog(spec, log_path)
        self.config = Config(
            read_timeout=900, retries={"max_attempts": 10, "mode": "adaptive"}
        )
        self.runtime = boto3.client("bedrock-runtime", region_name=self.spec.region, config=self.config)

    def complete(self, system: str, parts: list, task: str = "") -> str:
        """Send one prompt (Text / Image parts) and return the reply text."""
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
