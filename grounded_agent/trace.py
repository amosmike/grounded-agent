from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from anthropic.types.beta import BetaMessage

from .model import cost_of


@dataclass(frozen=True)
class Span:
    step: str
    model: str
    turn: int
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    cost_usd: float
    ms: int
    stop_reason: str | None
    tool_calls: list[str]


class Trace:
    """Every model call in a run, with what it cost and which tools it asked for."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.spans: list[Span] = []

    def record(self, step: str, turn: int, res: BetaMessage, ms: int) -> Span:
        span = Span(
            step=step,
            model=res.model,
            turn=turn,
            input_tokens=res.usage.input_tokens,
            output_tokens=res.usage.output_tokens,
            cache_read_tokens=res.usage.cache_read_input_tokens or 0,
            cost_usd=cost_of(res.model, res.usage),
            ms=ms,
            stop_reason=res.stop_reason,
            tool_calls=[b.name for b in res.content if b.type == "tool_use"],
        )
        self.spans.append(span)
        return span

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.spans)

    def write(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "trace.jsonl"
        path.write_text("".join(json.dumps({"run_id": self.run_id, **asdict(s)}) + "\n" for s in self.spans))
        return path
