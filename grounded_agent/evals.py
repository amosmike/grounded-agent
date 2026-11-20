from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .agent import AgentDeps, Outcome, investigate
from .ask import ask_for_tool
from .trace import Trace
from .types import ProposedChange, Rating, Ticket
from .world import load_world


@dataclass(frozen=True)
class ExpectedChange:
    collection: str
    record_id: str
    field: str
    new_value: Any


@dataclass(frozen=True)
class GoldenCase:
    """A reviewed ticket: what really happened, and the snapshot of its data to replay against."""

    id: str
    world: str
    ticket: Ticket
    cause: str
    escalate: bool
    change: ExpectedChange | None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> GoldenCase:
        exp = d["expected"]
        return cls(
            id=d["id"],
            world=d["world"],
            ticket=Ticket(**d["ticket"]),
            cause=exp["cause"],
            escalate=exp["escalate"],
            change=ExpectedChange(**exp["change"]) if exp["change"] else None,
        )


@dataclass(frozen=True)
class CaseScore:
    case_id: str
    run: int
    outcome: Outcome
    rating: Rating | None
    cause: float
    escalation: float
    change: float
    cost_usd: float
    turns: int

    @property
    def right(self) -> float:
        return (self.cause + self.escalation + self.change) / 3

    @property
    def passed(self) -> bool:
        return self.cause >= 0.5 and self.escalation == 1 and self.change == 1


def score_change(expected: ExpectedChange | None, proposals: list[ProposedChange]) -> float:
    """Field by field: the exact fix, and nothing else. Expecting no fix means proposing none."""
    if expected is None:
        return 1.0 if not proposals else 0.0
    hit = any(
        p.collection == expected.collection
        and p.record_id == expected.record_id
        and p.field == expected.field
        and p.new_value == expected.new_value
        for p in proposals
    )
    if not hit:
        return 0.0
    return 1.0 if len(proposals) == 1 else 0.5


JUDGE_TOOL = {
    "name": "submit_score",
    "description": "Record whether the two causes match.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "match": {"type": "string", "enum": ["same", "partly", "different"]},
            "why": {"type": "string"},
        },
        "required": ["match", "why"],
        "additionalProperties": False,
    },
}

JUDGE_SYSTEM = """You grade a support investigation against what really happened.
"same": it names the same root cause, even in different words.
"partly": it finds part of the cause, or the right area but the wrong mechanism.
"different": it names another cause, or none."""


def judge_cause(deps: AgentDeps, trace: Trace, expected: str, actual: str) -> float:
    out = ask_for_tool(
        deps.model,
        deps.routes["judge"],
        "judge",
        trace,
        JUDGE_SYSTEM,
        f"What really happened: {expected}\n\nThe agent's cause: {actual}",
        JUDGE_TOOL,
    )
    return {"same": 1.0, "partly": 0.5}.get(out.get("match"), 0.0)


def run_eval(cases: list[GoldenCase], deps: AgentDeps, world_dir: Path, k: int = 1) -> list[CaseScore]:
    """Replay each case k times against its snapshot.

    The snapshot is the data, not a recording of the calls the original run made, so a new model or prompt
    can take a different route through it and still be scored on the outcome.
    """
    scores = []
    for case in cases:
        world = load_world(world_dir / case.world)
        for run in range(1, k + 1):
            r = investigate(case.ticket, world, deps)
            a = r.answer
            cause = judge_cause(deps, Trace(f"{r.run_id}-judge"), case.cause, a.cause) if a else 0.0
            scores.append(
                CaseScore(
                    case_id=case.id,
                    run=run,
                    outcome=r.outcome,
                    rating=r.rating,
                    cause=cause,
                    escalation=1.0 if a and a.escalate == case.escalate else 0.0,
                    change=score_change(case.change, r.proposals) if a else 0.0,
                    cost_usd=r.cost_usd,
                    turns=r.turns,
                )
            )
    return scores


@dataclass(frozen=True)
class Summary:
    cases: int
    runs_per_case: int
    mean_right: float
    pass_hat_k: float  # share of cases that passed on every one of their k runs
    mean_cost_usd: float
    mean_turns: float


def summarise(scores: list[CaseScore], k: int) -> Summary:
    by_case: dict[str, list[CaseScore]] = defaultdict(list)
    for s in scores:
        by_case[s.case_id].append(s)
    return Summary(
        cases=len(by_case),
        runs_per_case=k,
        mean_right=mean(s.right for s in scores),
        pass_hat_k=mean(1.0 if all(r.passed for r in runs) else 0.0 for runs in by_case.values()),
        mean_cost_usd=mean(s.cost_usd for s in scores),
        mean_turns=mean(s.turns for s in scores),
    )
