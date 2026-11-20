from __future__ import annotations

import re
from dataclasses import dataclass

from .types import Claim

MIN_QUOTE_CHARS = 8


@dataclass(frozen=True)
class CitationResult:
    index: int
    ok: bool
    reason: str = ""


def _normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def check_citations(claims: list[Claim], evidence: dict[str, str]) -> list[CitationResult]:
    """Plain code, no model: every claim's quote must appear word for word in the tool result it names.

    A model can paraphrase or invent evidence; it can't get a made-up quote past this.
    """
    results = []
    for i, claim in enumerate(claims):
        quote = _normalise(claim.quote)
        source = evidence.get(claim.source_call_id)
        if len(quote) < MIN_QUOTE_CHARS:
            results.append(CitationResult(i, False, "quote too short to check"))
        elif source is None:
            results.append(CitationResult(i, False, f"no tool call with id {claim.source_call_id}"))
        elif quote not in _normalise(source):
            results.append(CitationResult(i, False, "quote not found in that tool result"))
        else:
            results.append(CitationResult(i, True))
    return results
