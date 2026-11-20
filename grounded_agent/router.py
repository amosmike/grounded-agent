from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Literal, Mapping, get_args

from .model import Effort

# Each model call in a run is one of these steps, and each step picks its own model.
Step = Literal["investigate", "verify", "judge"]


@dataclass(frozen=True)
class Route:
    model: str
    effort: Effort
    max_tokens: int


DEFAULT_ROUTES: dict[Step, Route] = {
    "investigate": Route("claude-opus-5-5", "high", 16_000),
    "verify": Route("claude-opus-5-5", "medium", 8_000),
    "judge": Route("claude-opus-5-5", "medium", 4_000),
}


def routes_from_env(env: Mapping[str, str] = os.environ) -> dict[Step, Route]:
    """Override a step's model or effort, e.g. MODEL_VERIFY=claude-sonnet-5-5 EFFORT_VERIFY=low.

    Compare routes on the eval before changing a default: a cheaper call that needs more turns isn't cheaper.
    """
    routes = dict(DEFAULT_ROUTES)
    efforts = get_args(Effort)
    for step, route in routes.items():
        model = env.get(f"MODEL_{step.upper()}", route.model)
        effort = env.get(f"EFFORT_{step.upper()}", route.effort)
        if effort not in efforts:
            raise ValueError(f"EFFORT_{step.upper()} must be one of {', '.join(efforts)}")
        routes[step] = replace(route, model=model, effort=effort)  # type: ignore[arg-type]
    return routes
