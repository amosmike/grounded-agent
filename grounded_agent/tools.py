from __future__ import annotations

import json
from typing import Any

from .types import Answer, Claim
from .world import Sandbox, ToolError

MAX_RESULT_CHARS = 20_000
SUBMIT_ANSWER = "submit_answer"


def build_tools(collections: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "name": "find_records",
            "description": f"Find records where a field equals a value (compared as text). Collections: {', '.join(collections)}.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "field": {"type": "string"},
                    "value": {"type": "string"},
                },
                "required": ["collection", "field", "value"],
                "additionalProperties": False,
            },
        },
        {
            "name": "search_logs",
            "description": "Search application logs for lines whose message or service contains the text. Returns up to 50 lines.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
                "additionalProperties": False,
            },
        },
        {
            "name": "propose_change",
            "description": "Propose a one-field data fix on a record you have read with find_records. "
            "It is queued for a person to approve; nothing is written.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "collection": {"type": "string"},
                    "record_id": {"type": "string"},
                    "field": {"type": "string"},
                    "new_value_json": {
                        "type": "string",
                        "description": 'The new value as a JSON literal, e.g. "active", 0, true or null.',
                    },
                    "reason": {"type": "string"},
                },
                "required": ["collection", "record_id", "field", "new_value_json", "reason"],
                "additionalProperties": False,
            },
        },
        {
            "name": SUBMIT_ANSWER,
            "description": "Submit your finished investigation. Call it once, when you're done.",
            "strict": True,
            "input_schema": {
                "type": "object",
                "properties": {
                    "question_asked": {"type": "string", "description": "What the customer is actually asking, in one sentence."},
                    "cause": {"type": "string", "description": "The cause, kept to what your claims show."},
                    "claims": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "text": {"type": "string"},
                                "quote": {"type": "string", "description": "8-200 characters copied word for word from one tool result."},
                                "source_call_id": {"type": "string", "description": "The call_id shown at the top of that tool result."},
                            },
                            "required": ["text", "quote", "source_call_id"],
                            "additionalProperties": False,
                        },
                    },
                    "fix": {"type": "string"},
                    "escalate": {"type": "boolean", "description": "True when engineering needs to fix code or config."},
                    "customer_reply": {"type": "string"},
                    "pattern_id": {"type": "string", "description": 'The past case this matches, e.g. "P-3", or "" for none.'},
                },
                "required": ["question_asked", "cause", "claims", "fix", "escalate", "customer_reply", "pattern_id"],
                "additionalProperties": False,
            },
        },
    ]


def _str(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if not isinstance(value, str):
        raise ToolError(f'"{key}" must be a string.')
    return value


def _truncate(text: str) -> str:
    if len(text) <= MAX_RESULT_CHARS:
        return text
    return f"{text[:MAX_RESULT_CHARS]}\n[cut at {MAX_RESULT_CHARS} characters; narrow the query]"


def run_tool(sandbox: Sandbox, name: str, args: dict[str, Any]) -> str:
    """Run one read or propose tool and return its result text. Raises ToolError for bad input."""
    if name == "find_records":
        rows = sandbox.find_records(_str(args, "collection"), _str(args, "field"), _str(args, "value"))
        return _truncate(json.dumps(rows, indent=2))
    if name == "search_logs":
        return _truncate(json.dumps(sandbox.search_logs(_str(args, "query")), indent=2))
    if name == "propose_change":
        try:
            value = json.loads(_str(args, "new_value_json"))
        except json.JSONDecodeError:
            raise ToolError('new_value_json must be a JSON literal, e.g. "active" with the quotes.') from None
        c = sandbox.propose(_str(args, "collection"), _str(args, "record_id"), _str(args, "field"), value, _str(args, "reason"))
        return (
            f"Proposed {c.id}: {c.collection}/{c.record_id}.{c.field} {json.dumps(c.old_value)} -> {json.dumps(c.new_value)}. "
            "Pending a person's approval; nothing was written."
        )
    raise ToolError(f'Unknown tool "{name}".')


def parse_answer(args: dict[str, Any]) -> Answer:
    claims = [
        Claim(text=str(c.get("text", "")), quote=str(c.get("quote", "")), source_call_id=str(c.get("source_call_id", "")))
        for c in args.get("claims") or []
        if isinstance(c, dict)
    ]
    return Answer(
        question_asked=_str(args, "question_asked"),
        cause=_str(args, "cause"),
        claims=claims,
        fix=_str(args, "fix"),
        escalate=args.get("escalate") is True,
        customer_reply=_str(args, "customer_reply"),
        pattern_id=args.get("pattern_id") if isinstance(args.get("pattern_id"), str) else "",
    )
