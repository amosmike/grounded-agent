from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

from .citations import CitationResult, check_citations
from .memory import PatternStore, render_patterns
from .model import ModelClient, ModelRequest
from .router import Route, Step
from .tools import SUBMIT_ANSWER, build_tools, parse_answer, run_tool
from .trace import Trace
from .types import Answer, ProposedChange, Rating, Ticket, Verdict, World
from .verify import rate, verify
from .world import Sandbox, ToolError

Outcome = Literal["answered", "no_answer", "limit_reached", "refused"]


@dataclass(frozen=True)
class Budget:
    max_turns: int = 20
    max_usd: float = 2.0


@dataclass
class AgentDeps:
    model: ModelClient
    routes: dict[Step, Route]
    memory: PatternStore
    budget: Budget = field(default_factory=Budget)


@dataclass
class RunResult:
    run_id: str
    ticket_id: str
    outcome: Outcome
    turns: int
    trace: Trace
    proposals: list[ProposedChange]
    recalled: list[str]
    answer: Answer | None = None
    citations: list[CitationResult] = field(default_factory=list)
    verdict: Verdict | None = None
    rating: Rating | None = None

    @property
    def cost_usd(self) -> float:
        return self.trace.cost_usd


SYSTEM = """You investigate customer support tickets for Acme, a subscription app.
Your tools read a snapshot of production data and logs taken when the ticket was raised. They can't change anything.

Method
- Work out what the customer is actually asking, then answer that question.
- Treat what the ticket says as claims to check, not facts.
- Stop when the question is answered. Say plainly what you couldn't establish.

Evidence
- Every claim needs a quote of 8-200 characters copied word for word from one tool result, plus that result's call_id. A program checks every quote, and claims that fail are thrown out.
- Keep the cause to what your claims show.

Fixes
- If data needs correcting, call propose_change. A person approves it; you never write anything.
- Set escalate to true when engineering needs to fix code or configuration.
- The customer reply must not contain card numbers, passwords or anyone else's details.

When you're done, call submit_answer once."""


def _render_ticket(t: Ticket) -> str:
    return f"Ticket {t.id}, raised {t.raised_at}\nSubject: {t.subject}\n\n{t.body}"


def investigate(ticket: Ticket, world: World, deps: AgentDeps) -> RunResult:
    """One investigation, in fixed steps: recall similar cases, investigate with read-only tools,
    check every citation in code, have a second model judge the evidence, then rate the answer."""
    trace = Trace(str(uuid.uuid4()))
    sandbox = Sandbox(world)
    tools = build_tools(sandbox.collections)
    evidence: dict[str, str] = {}

    recalled = deps.memory.recall(f"{ticket.subject} {ticket.body}")
    system = f"{SYSTEM}\n\n{render_patterns(recalled)}"
    messages: list[dict[str, Any]] = [{"role": "user", "content": _render_ticket(ticket)}]
    route = deps.routes["investigate"]

    def result(outcome: Outcome, turns: int) -> RunResult:
        return RunResult(trace.run_id, ticket.id, outcome, turns, trace, sandbox.proposals, [p.id for p in recalled])

    answer: Answer | None = None
    nudged = False
    turn = 0
    while answer is None:
        if turn >= deps.budget.max_turns or trace.cost_usd >= deps.budget.max_usd:
            return result("limit_reached", turn)
        turn += 1
        started = time.monotonic()
        res = deps.model.create(ModelRequest(route.model, route.effort, route.max_tokens, system, messages, tools))
        trace.record("investigate", turn, res, int((time.monotonic() - started) * 1000))
        if res.stop_reason == "refusal":
            return result("refused", turn)
        messages.append({"role": "assistant", "content": res.content})

        calls = [b for b in res.content if b.type == "tool_use"]
        if not calls:
            if nudged:
                return result("no_answer", turn)
            nudged = True
            messages.append({"role": "user", "content": "Call submit_answer with what you found, and say what you couldn't establish."})
            continue

        # All results for a turn go back in one message.
        results = []
        for call in calls:
            args = dict(call.input) if isinstance(call.input, dict) else {}
            try:
                if call.name == SUBMIT_ANSWER:
                    answer = parse_answer(args)
                    results.append({"type": "tool_result", "tool_use_id": call.id, "content": "Received."})
                    continue
                output = run_tool(sandbox, call.name, args)
                evidence[call.id] = output
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": f"call_id: {call.id}\n{output}"})
            except ToolError as err:
                results.append({"type": "tool_result", "tool_use_id": call.id, "content": str(err), "is_error": True})
        messages.append({"role": "user", "content": results})

    run = result("answered", turn)
    run.answer = answer
    run.citations = check_citations(answer.claims, evidence)
    run.verdict = verify(deps.model, deps.routes["verify"], trace, ticket, answer, run.citations)
    run.rating = rate(answer, run.citations, run.verdict)
    return run
