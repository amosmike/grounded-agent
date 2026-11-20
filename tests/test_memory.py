from grounded_agent.memory import PatternStore, scrub
from grounded_agent.types import Answer, Ticket


def ticket(id: str, subject: str, body: str = "") -> Ticket:
    return Ticket(id=id, raised_at="2026-03-01T00:00:00Z", subject=subject, body=body)


def answer(cause: str, pattern_id: str = "") -> Answer:
    return Answer(question_asked="", cause=cause, claims=[], fix="Restore the subscription", escalate=True, customer_reply="", pattern_id=pattern_id)


def test_scrub_removes_emails_record_ids_uuids_and_long_numbers():
    out = scrub("alice@example.com paid pay_5001 for sub_1001, ref 4929123456781234, id 0f8fad5b-d9cb-469f-a165-70867728950e")
    assert out == "<email> paid <pay_id> for <sub_id>, ref <number>, id <uuid>"


def test_learns_a_scrubbed_candidate_and_nothing_from_weak_evidence():
    store = PatternStore()
    assert store.learn(ticket("T-1", "Locked out"), answer("whatever"), "weak") is None

    p = store.learn(ticket("T-2", "Paid but past due", "I'm alice@example.com"), answer("Webhook for pay_5001 dropped"), "solid")
    assert p.status == "candidate"
    assert "alice@example.com" not in p.symptom
    assert p.cause == "Webhook for <pay_id> dropped"


def test_a_matched_pattern_gets_an_occurrence_not_a_duplicate():
    store = PatternStore()
    p = store.learn(ticket("T-1", "Paid but past due"), answer("Webhook dropped"), "solid")
    store.learn(ticket("T-2", "Premium locked after paying"), answer("Webhook dropped", p.id), "gaps")
    assert len(store.all()) == 1
    assert [o.ticket_id for o in p.occurrences] == ["T-1", "T-2"]


def test_recalls_the_most_relevant_case_and_nothing_unrelated():
    store = PatternStore()
    store.learn(ticket("T-1", "Charged twice", "two payments on the same day"), answer("Retry without idempotency key"), "solid")
    store.learn(ticket("T-2", "Locked out of account", "can't log in"), answer("Too many failed logins"), "solid")
    assert store.recall("I was charged twice for my plan")[0].id == "P-1"
    assert store.recall("weather in Paris") == []


def test_an_engineers_review_confirms_or_disputes_the_pattern():
    store = PatternStore()
    p = store.learn(ticket("T-1", "Charged twice"), answer("Retry without idempotency key"), "solid")
    assert store.apply_review(p.id, "T-1", "correct").status == "confirmed"
    store.learn(ticket("T-2", "Charged twice again"), answer("Retry", p.id), "gaps")
    assert store.apply_review(p.id, "T-2", "wrong").status == "disputed"


def test_round_trips_through_its_file(tmp_path):
    path = tmp_path / "patterns.json"
    store = PatternStore(path)
    p = store.learn(ticket("T-1", "Charged twice"), answer("Retry without idempotency key"), "solid")
    store.apply_review(p.id, "T-1", "correct")
    store.save()

    again = PatternStore(path).get(p.id)
    assert again.status == "confirmed"
    assert again.occurrences[0].review == "correct"
