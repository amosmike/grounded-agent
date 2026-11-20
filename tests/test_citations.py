from grounded_agent.citations import check_citations
from grounded_agent.types import Claim

EVIDENCE = {"call_1": '[\n  {\n    "id": "sub_1001",\n    "status": "past_due"\n  }\n]'}


def claim(quote: str, source: str = "call_1") -> Claim:
    return Claim(text="", quote=quote, source_call_id=source)


def test_accepts_a_quote_in_the_named_result_ignoring_whitespace_and_case():
    assert check_citations([claim('"STATUS":   "past_due"')], EVIDENCE)[0].ok


def test_rejects_invented_short_or_misattributed_quotes():
    invented, short, wrong_call = check_citations(
        [claim('"status": "cancelled"'), claim("past"), claim('"status": "past_due"', "call_9")], EVIDENCE
    )
    assert invented.reason == "quote not found in that tool result"
    assert short.reason == "quote too short to check"
    assert wrong_call.reason == "no tool call with id call_9"
