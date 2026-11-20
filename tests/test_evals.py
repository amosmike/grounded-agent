from grounded_agent.agent import AgentDeps
from grounded_agent.evals import CaseScore, ExpectedChange, run_eval, score_change, summarise
from grounded_agent.memory import PatternStore
from grounded_agent.types import ProposedChange

from .helpers import CASES, ROUTES, WORLDS, ScriptedModel, reply, tool_use, verdict

EXPECTED = ExpectedChange("payments", "pay_6002", "refund_status", "requested")


def change(record_id: str, value: str) -> ProposedChange:
    return ProposedChange("chg_1", "payments", record_id, "refund_status", "none", value, "")


def test_score_change_needs_the_exact_record_field_and_value():
    assert score_change(EXPECTED, [change("pay_6002", "requested")]) == 1
    assert score_change(EXPECTED, [change("pay_6001", "requested")]) == 0
    assert score_change(EXPECTED, [change("pay_6002", "refunded")]) == 0


def test_score_change_marks_down_extras_and_any_change_where_none_was_needed():
    assert score_change(EXPECTED, [change("pay_6002", "requested"), change("pay_6001", "requested")]) == 0.5
    assert score_change(None, [change("pay_6002", "requested")]) == 0
    assert score_change(None, []) == 1


def test_replays_a_case_against_its_snapshot_and_scores_the_outcome():
    model = ScriptedModel(
        [
            reply(tool_use("search_logs", {"query": "cus_3001"}, "call_logs")),
            reply(
                tool_use(
                    "submit_answer",
                    {
                        "question_asked": "Why is the account locked, and was it hacked?",
                        "cause": "Five wrong passwords from her own device locked the account for 30 minutes.",
                        "claims": [
                            {"text": "The lock followed five failed attempts.", "quote": "locked for 30 minutes after 5 failed attempts", "source_call_id": "call_logs"}
                        ],
                        "fix": "None needed; it unlocks by itself.",
                        "escalate": False,
                        "customer_reply": "Your account locked after five wrong passwords from your own device. It unlocks within 30 minutes.",
                        "pattern_id": "",
                    },
                )
            ),
            verdict([True]),
            reply(tool_use("submit_score", {"match": "same", "why": "Same cause."})),
        ]
    )

    scores = run_eval([CASES["T-1003"]], AgentDeps(model, ROUTES, PatternStore()), WORLDS, k=1)

    [s] = scores
    assert (s.cause, s.escalation, s.change, s.passed) == (1, 1, 1, True)
    sm = summarise(scores, 1)
    assert (sm.cases, sm.mean_right, sm.pass_hat_k) == (1, 1, 1)


def test_pass_hat_k_counts_a_case_only_if_every_run_passed():
    def score(case: str, cause: float) -> CaseScore:
        return CaseScore(case, 1, "answered", "solid", cause, 1, 1, 0.5, 4)

    sm = summarise([score("A", 1), score("A", 0), score("B", 1), score("B", 1)], k=2)
    assert sm.pass_hat_k == 0.5
