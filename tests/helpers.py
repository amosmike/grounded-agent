from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Any, Callable

from anthropic.types.beta import BetaMessage

from grounded_agent.evals import GoldenCase
from grounded_agent.model import ModelRequest
from grounded_agent.router import DEFAULT_ROUTES
from grounded_agent.world import load_world

DATA = Path(__file__).resolve().parents[1] / "data"
WORLDS = DATA / "worlds"
CASES = {c.id: c for c in (GoldenCase.from_dict(d) for d in json.loads((DATA / "golden.json").read_text()))}
ROUTES = DEFAULT_ROUTES

_ids = itertools.count(1)


def world_for(case: GoldenCase):
    return load_world(WORLDS / case.world)


def tool_use(name: str, args: dict[str, Any], id: str | None = None) -> dict[str, Any]:
    return {"type": "tool_use", "id": id or f"toolu_{next(_ids)}", "name": name, "input": args}


def text(t: str) -> dict[str, Any]:
    return {"type": "text", "text": t}


def reply(*content: dict[str, Any]) -> BetaMessage:
    return BetaMessage.model_validate(
        {
            "id": f"msg_{next(_ids)}",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": list(content),
            "stop_reason": "tool_use" if any(b["type"] == "tool_use" for b in content) else "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 2_000, "output_tokens": 300, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0},
        }
    )


def verdict(supported: list[bool], answers: str = "yes", cause_follows: bool = True) -> BetaMessage:
    claims = [{"index": i, "supported": s, "why": "quote shows it" if s else "it doesn't"} for i, s in enumerate(supported)]
    return reply(tool_use("submit_verdict", {"answers_question": answers, "cause_follows": cause_follows, "claims": claims}))


class ScriptedModel:
    """Plays back a fixed list of replies, one per model call, and keeps every request it was sent."""

    def __init__(self, script: list[BetaMessage | Callable[[ModelRequest], BetaMessage]]) -> None:
        self._script = list(script)
        self.requests: list[ModelRequest] = []

    def create(self, req: ModelRequest) -> BetaMessage:
        self.requests.append(copy.deepcopy(req))
        if not self._script:
            raise AssertionError("script ran out of replies")
        nxt = self._script.pop(0)
        return nxt(req) if callable(nxt) else nxt

    @property
    def remaining(self) -> int:
        return len(self._script)
