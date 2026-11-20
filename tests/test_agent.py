import json

from grounded_agent.agent import AgentDeps, Budget, investigate
from grounded_agent.memory import PatternStore

from .helpers import CASES, ROUTES, ScriptedModel, reply, text, tool_use, verdict, world_for

T1001 = CASES["T-1001"]


def answer(claims: list[dict]) -> dict:
    return tool_use(
        "submit_answer",
        {
            "question_asked": "Why is premium locked when the renewal was paid?",
            "cause": "The payment succeeded but its webhook was dropped, so the subscription was marked past_due.",
            "claims": claims,
            "fix": "Set sub_1001 back to active; engineering to fix webhook signature verification.",
            "escalate": True,
            "customer_reply": "Sorry about this. Your payment went through and we've restored premium.",
            "pattern_id": "",
        },
    )


def deps(model, **kw) -> AgentDeps:
    return AgentDeps(model=model, routes=ROUTES, memory=PatternStore(), **kw)


def test_answers_with_cited_claims_a_rating_and_a_fix_left_for_approval():
    world = world_for(T1001)
    model = ScriptedModel(
        [
            reply(tool_use("find_records", {"collection": "subscriptions", "field": "customer_id", "value": "cus_1001"}, "call_sub")),
            reply(tool_use("search_logs", {"query": "pay_5001"}, "call_logs")),
            reply(
                tool_use(
                    "propose_change",
                    {"collection": "subscriptions", "record_id": "sub_1001", "field": "status", "new_value_json": '"active"', "reason": "Renewal was paid"},
                ),
                answer(
                    [
                        {"text": "The subscription is past due.", "quote": '"status": "past_due"', "source_call_id": "call_sub"},
                        {"text": "The payment's webhook was dropped.", "quote": "signature verification failed for event evt_88123", "source_call_id": "call_logs"},
                    ]
                ),
            ),
            verdict([True, True]),
        ]
    )

    r = investigate(T1001.ticket, world, deps(model))

    assert r.outcome == "answered"
    assert all(c.ok for c in r.citations)
    assert r.rating == "solid"
    assert [(p.record_id, p.new_value, p.status) for p in r.proposals] == [("sub_1001", "active", "pending_approval")]
    # The snapshot itself is never written to.
    assert next(s for s in world.collections["subscriptions"] if s["id"] == "sub_1001")["status"] == "past_due"
    assert [s.step for s in r.trace.spans] == ["investigate", "investigate", "investigate", "verify"]
    assert r.cost_usd > 0


def test_a_made_up_quote_is_thrown_out_even_when_the_verifier_is_fooled():
    model = ScriptedModel(
        [
            reply(tool_use("find_records", {"collection": "subscriptions", "field": "customer_id", "value": "cus_1001"}, "call_a")),
            reply(
                answer(
                    [
                        {"text": "The subscription is past due.", "quote": '"status": "past_due"', "source_call_id": "call_a"},
                        {"text": "The card was declined.", "quote": "card declined by issuer", "source_call_id": "call_a"},
                    ]
                )
            ),
            verdict([True, True]),
        ]
    )

    r = investigate(T1001.ticket, world_for(T1001), deps(model))

    assert [c.ok for c in r.citations] == [True, False]
    assert r.rating == "gaps"


def test_refuses_a_fix_to_a_record_it_never_read():
    model = ScriptedModel(
        [
            reply(
                tool_use(
                    "propose_change",
                    {"collection": "subscriptions", "record_id": "sub_1001", "field": "status", "new_value_json": '"active"', "reason": "guess"},
                )
            ),
            reply(answer([])),
            verdict([]),
        ]
    )

    r = investigate(T1001.ticket, world_for(T1001), deps(model))

    assert r.proposals == []
    tool_result = json.dumps(model.requests[1].messages[-1])
    assert "before proposing a change" in tool_result
    assert r.rating == "weak"


def test_stops_at_the_turn_limit_instead_of_looping():
    model = ScriptedModel([reply(tool_use("search_logs", {"query": "billing"})) for _ in range(10)])

    r = investigate(T1001.ticket, world_for(T1001), deps(model, budget=Budget(max_turns=3, max_usd=10)))

    assert r.outcome == "limit_reached"
    assert r.turns == 3


def test_nudges_once_then_gives_up_if_the_model_never_submits():
    model = ScriptedModel([reply(text("Thinking it over.")), reply(text("Still thinking."))])

    r = investigate(T1001.ticket, world_for(T1001), deps(model))

    assert r.outcome == "no_answer"
    assert model.remaining == 0
