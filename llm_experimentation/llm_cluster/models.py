"""Registry of the Bedrock models used in the experiment.

The model IDs below are the expected Bedrock IDs / cross-region inference profiles. VERIFY them
against your account before the first real run (README, "Step 0"):

    aws bedrock list-inference-profiles --region <region>
    aws bedrock list-foundation-models --region <region> --by-provider anthropic|openai

and override with --model-id if yours differ. Prices are USD per million tokens and are only used
for the running cost estimate / --max-usd; None means "unknown" (cost is not tracked for that model
unless you pass --price-in/--price-out).
"""

from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_id: str
    supports_images: bool
    usd_in_per_m: float | None
    usd_out_per_m: float | None

    def with_overrides(self, model_id=None, price_in=None, price_out=None) -> "ModelSpec":
        return replace(
            self,
            model_id=model_id or self.model_id,
            usd_in_per_m=self.usd_in_per_m if price_in is None else price_in,
            usd_out_per_m=self.usd_out_per_m if price_out is None else price_out,
        )


MODELS: dict[str, ModelSpec] = {
    # Claude Sonnet 5.5 (vision)
    "sonnet": ModelSpec("sonnet", "global.anthropic.claude-sonnet-5-5", True, 2.0, 10.0),
    # Claude Opus 5.5 (vision)
    "opus": ModelSpec("opus", "global.anthropic.claude-opus-5-5", True, 4.0, 20.0),
    # OpenAI GPT-6.1 Terra (text + image input)
    "terra": ModelSpec("terra", "global.openai.gpt-6.1-terra", True, None, None),
    # offline fake used by --dry-run (no network, random answers)
    "fake": ModelSpec("fake", "fake", True, 0.0, 0.0),
}
