from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

from .types import Answer, Rating, Ticket

Review = Literal["correct", "partly", "wrong"]
Status = Literal["candidate", "confirmed", "disputed", "rejected"]


@dataclass
class Occurrence:
    ticket_id: str
    rating: Rating
    review: Review | None = None


@dataclass
class Pattern:
    """A problem the agent has seen before. It holds the problem, never the customer."""

    id: str
    symptom: str
    cause: str
    fix: str
    status: Status = "candidate"
    occurrences: list[Occurrence] = field(default_factory=list)


_SCRUBBERS = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<email>"),
    (re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I), "<uuid>"),
    (re.compile(r"\b(cus|sub|pay|evt|dev|chg)_[A-Za-z0-9]+\b"), r"<\1_id>"),
    (re.compile(r"\b\d{6,}\b"), "<number>"),
]


def scrub(text: str) -> str:
    """Strip identifiers before anything is stored, so memory can be shared across tickets and users."""
    for pattern, sub in _SCRUBBERS:
        text = pattern.sub(sub, text)
    return text


_STOPWORDS = set("the and for was are but not you your has have had this that with from its can".split())


def _tokens(text: str) -> list[str]:
    return [t for t in re.split(r"[^a-z0-9]+", text.lower()) if len(t) > 2 and t not in _STOPWORDS]


def _bm25(query: list[str], docs: list[list[str]], k1: float = 1.2, b: float = 0.75) -> list[float]:
    """Plain BM25. Swap for embeddings once the catalogue outgrows what keyword recall can rank."""
    if not docs:
        return []
    avg_len = sum(len(d) for d in docs) / len(docs)
    df = Counter(t for d in docs for t in set(d))
    scores = []
    for d in docs:
        tf = Counter(d)
        score = 0.0
        for t in set(query):
            if not tf[t]:
                continue
            idf = math.log(1 + (len(docs) - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg_len))
        scores.append(score)
    return scores


class PatternStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path
        self._patterns: list[Pattern] = []
        if path and path.exists():
            for raw in json.loads(path.read_text()):
                occurrences = [Occurrence(**o) for o in raw.pop("occurrences")]
                self._patterns.append(Pattern(**raw, occurrences=occurrences))

    def all(self) -> list[Pattern]:
        return list(self._patterns)

    def get(self, pattern_id: str) -> Pattern | None:
        return next((p for p in self._patterns if p.id == pattern_id), None)

    def recall(self, query: str, k: int = 5) -> list[Pattern]:
        """The k past cases most like this ticket. Rejected patterns are never shown again."""
        live = [p for p in self._patterns if p.status != "rejected"]
        scores = _bm25(_tokens(query), [_tokens(f"{p.symptom} {p.cause}") for p in live])
        ranked = sorted((pair for pair in zip(scores, live) if pair[0] > 0), key=lambda pair: pair[0], reverse=True)
        return [p for _, p in ranked[:k]]

    def learn(self, ticket: Ticket, answer: Answer, rating: Rating) -> Pattern | None:
        """Learn from a finished run. Weak evidence teaches nothing; a match just adds an occurrence."""
        if rating == "weak":
            return None
        matched = self.get(answer.pattern_id) if answer.pattern_id else None
        if matched:
            matched.occurrences.append(Occurrence(ticket.id, rating))
            return matched
        pattern = Pattern(
            id=f"P-{len(self._patterns) + 1}",
            symptom=scrub(f"{ticket.subject}. {ticket.body}")[:400],
            cause=scrub(answer.cause),
            fix=scrub(answer.fix),
            occurrences=[Occurrence(ticket.id, rating)],
        )
        self._patterns.append(pattern)
        return pattern

    def apply_review(self, pattern_id: str, ticket_id: str, review: Review) -> Pattern:
        """An engineer's review of an answer is the evidence for the pattern behind it, so there's no separate
        confirm step. Correct confirms a candidate nobody has called wrong; wrong or partly flags it for a person."""
        pattern = self.get(pattern_id)
        if pattern is None:
            raise KeyError(f"No pattern {pattern_id}")
        occurrence = next((o for o in pattern.occurrences if o.ticket_id == ticket_id), None)
        if occurrence is None:
            raise KeyError(f"{pattern_id} has no occurrence for ticket {ticket_id}")
        occurrence.review = review
        if pattern.status == "rejected":
            return pattern
        if review != "correct":
            pattern.status = "disputed"
        elif pattern.status == "candidate" and not any(o.review == "wrong" for o in pattern.occurrences):
            pattern.status = "confirmed"
        return pattern

    def save(self) -> None:
        if self._path:
            self._path.write_text(json.dumps([asdict(p) for p in self._patterns], indent=2) + "\n")


def render_patterns(patterns: list[Pattern]) -> str:
    if not patterns:
        return "No similar past cases."
    blocks = [
        f"[{p.id}] ({p.status}, seen {len(p.occurrences)}x)\n  Symptom: {p.symptom}\n  Cause: {p.cause}\n  Fix: {p.fix}"
        for p in patterns
    ]
    return "Similar past cases. These are hints, not answers: check the data before relying on one.\n" + "\n".join(blocks)
