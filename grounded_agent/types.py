from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Rating = Literal["solid", "gaps", "weak"]


@dataclass(frozen=True)
class Ticket:
    id: str
    raised_at: str
    subject: str
    body: str


@dataclass(frozen=True)
class World:
    """A snapshot of the data as it stood when the ticket was raised."""

    name: str
    taken_at: str
    collections: dict[str, list[dict[str, Any]]]
    logs: list[dict[str, str]]


@dataclass(frozen=True)
class Claim:
    text: str
    quote: str
    source_call_id: str


@dataclass(frozen=True)
class Answer:
    question_asked: str
    cause: str
    claims: list[Claim]
    fix: str
    escalate: bool
    customer_reply: str
    pattern_id: str = ""  # a past case the investigator says this matches, or "" for none


@dataclass(frozen=True)
class ProposedChange:
    id: str
    collection: str
    record_id: str
    field: str
    old_value: Any
    new_value: Any
    reason: str
    status: Literal["pending_approval"] = "pending_approval"


@dataclass
class Verdict:
    answers_question: Literal["yes", "partly", "no"]
    cause_follows: bool
    claims: list[dict[str, Any]] = field(default_factory=list)  # {"index", "supported", "why"}
