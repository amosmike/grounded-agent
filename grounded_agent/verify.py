from __future__ import annotations

from .ask import ask_for_tool
from .citations import CitationResult
from .model import ModelClient
from .router import Route
from .trace import Trace
from .types import Answer, Rating, Ticket, Verdict

VERDICT_TOOL = {
    "name": "submit_verdict",
    "description": "Record your verdict on the investigation.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "answers_question": {"type": "string", "enum": ["yes", "partly", "no"]},
            "cause_follows": {"type": "boolean"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer"},
                        "supported": {"type": "boolean"},
                        "why": {"type": "string"},
                    },
                    "required": ["index", "supported", "why"],
                    "additionalProperties": False,
                },
            },
        },
        "required": ["answers_question", "cause_follows", "claims"],
        "additionalProperties": False,
    },
}

SYSTEM = """You check another agent's support investigation. You have no tools and no other context, on purpose: judge only what is in front of you.
For each claim, decide whether its quoted evidence supports it. A claim whose quote was not found in the tool output is unsupported.
Then decide whether the stated cause follows from the supported claims, and whether the answer addresses what the customer actually asked."""


def verify(
    model: ModelClient,
    route: Route,
    trace: Trace,
    ticket: Ticket,
    answer: Answer,
    citations: list[CitationResult],
) -> Verdict:
    lines = []
    for i, claim in enumerate(answer.claims):
        cite = citations[i] if i < len(citations) else None
        status = "quote found in tool output" if cite and cite.ok else f"quote NOT verified ({cite.reason if cite else 'unchecked'})"
        lines.append(f'[{i}] {claim.text}\n    evidence: "{claim.quote}" ({status})')
    prompt = (
        f"Ticket: {ticket.subject}\n{ticket.body}\n\n"
        f"Question as restated: {answer.question_asked}\nCause: {answer.cause}\nFix: {answer.fix}\n\n"
        f"Claims:\n{chr(10).join(lines) or '(none)'}"
    )
    out = ask_for_tool(model, route, "verify", trace, SYSTEM, prompt, VERDICT_TOOL)
    answers = out.get("answers_question")
    return Verdict(
        answers_question=answers if answers in ("yes", "partly", "no") else "no",
        cause_follows=out.get("cause_follows") is True,
        claims=[
            {"index": int(c.get("index", -1)), "supported": c.get("supported") is True, "why": str(c.get("why", ""))}
            for c in out.get("claims") or []
            if isinstance(c, dict)
        ],
    )


def rate(answer: Answer, citations: list[CitationResult], verdict: Verdict) -> Rating:
    """Solid: every claim cited and supported, and it answers the question. Weak: don't act on it without checking."""
    supported_by_model = {c["index"] for c in verdict.claims if c["supported"]}
    total = len(answer.claims)
    supported = sum(1 for i in range(total) if i < len(citations) and citations[i].ok and i in supported_by_model)
    if total == 0 or verdict.answers_question == "no" or not verdict.cause_follows or supported < total / 2:
        return "weak"
    if supported == total and verdict.answers_question == "yes":
        return "solid"
    return "gaps"
