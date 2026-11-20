from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import anthropic
from anthropic.types.beta import BetaMessage

Effort = Literal["low", "medium", "high", "xhigh", "max"]


@dataclass
class ModelRequest:
    model: str
    effort: Effort
    max_tokens: int
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]]


class ModelClient(Protocol):
    """The one thing the agent needs from a model provider. Tests swap in a scripted client."""

    def create(self, req: ModelRequest) -> BetaMessage: ...


class AnthropicModel:
    def __init__(self) -> None:
        self._client = anthropic.Anthropic()

    def create(self, req: ModelRequest) -> BetaMessage:
        return self._client.beta.messages.create(
            model=req.model,
            max_tokens=req.max_tokens,
            system=req.system,
            messages=req.messages,
            tools=req.tools,
            output_config={"effort": req.effort},
            # On a safety decline, the API re-runs the request on a fallback model within the same call.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
        )


# List prices in USD per million tokens. Check the pricing page before relying on them.
PRICES: dict[str, dict[str, float]] = {
    "claude-opus-5-5": {"input": 4, "output": 20, "cache_read": 0.2},
    "claude-sonnet-5-5": {"input": 2, "output": 10, "cache_read": 0.1},
    "claude-haiku-5-5": {"input": 0.1, "output": 0.5, "cache_read": 0.01},
}


def cost_of(model: str, usage: Any) -> float:
    p = PRICES.get(model)
    if p is None:
        return 0.0
    cache_write = usage.cache_creation_input_tokens or 0
    cache_read = usage.cache_read_input_tokens or 0
    return (
        usage.input_tokens * p["input"]
        + usage.output_tokens * p["output"]
        + cache_read * p["cache_read"]
        + cache_write * p["input"] * 1.25
    ) / 1_000_000
