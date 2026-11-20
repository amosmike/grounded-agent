from __future__ import annotations

import time
from typing import Any

from .model import ModelClient, ModelRequest
from .router import Route
from .trace import Trace


def ask_for_tool(
    model: ModelClient,
    route: Route,
    step: str,
    trace: Trace,
    system: str,
    prompt: str,
    tool: dict[str, Any],
) -> dict[str, Any]:
    """One-shot call that must end in a call to `tool`, for steps that return a structured verdict.

    Forced tool choice isn't available on current models, so it asks, then asks once more if needed.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    for attempt in (1, 2):
        started = time.monotonic()
        res = model.create(ModelRequest(route.model, route.effort, route.max_tokens, system, messages, [tool]))
        trace.record(step, attempt, res, int((time.monotonic() - started) * 1000))
        if res.stop_reason == "refusal":
            raise RuntimeError(f"{step}: the model declined")
        call = next((b for b in res.content if b.type == "tool_use" and b.name == tool["name"]), None)
        if call is not None:
            return dict(call.input)
        messages += [
            {"role": "assistant", "content": res.content},
            {"role": "user", "content": f"Reply by calling {tool['name']}."},
        ]
    raise RuntimeError(f"{step}: no {tool['name']} call after two attempts")
