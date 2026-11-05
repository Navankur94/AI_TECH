import pytest

from app.graph import ESCALATION_MESSAGE
from app.service import InvalidState, TicketNotFound

from .conftest import ScriptedLLM, make_service

QUESTION = "When will I receive my monthly account statement?"


def test_low_risk_ticket_resolves_without_a_human(service):
    ticket = service.submit(QUESTION)
    assert ticket["status"] == "auto_resolved"
    assert ticket["final_response"] == ticket["draft"]
    assert ticket["risk_flags"] == []
    assert ticket["audit"][-1] == "auto_resolved"


@pytest.mark.parametrize("category", ["fraud", "complaint", "investment_query"])
def test_high_risk_categories_always_pause_for_review(category):
    service = make_service(ScriptedLLM(classify={"category": category, "urgency": "normal"}))
    ticket = service.submit("Something about my account needs attention.")
    assert ticket["status"] == "pending_review"
    assert f"category:{category}" in ticket["risk_flags"]
    assert ticket["final_response"] is None


def test_unsafe_promise_in_draft_forces_review_even_with_high_confidence():
    llm = ScriptedLLM(draft={"response": "Your returns are guaranteed with zero risk.", "confidence": 0.99})
    ticket = make_service(llm).submit(QUESTION)
    assert ticket["status"] == "pending_review" and "unsafe_promise" in ticket["risk_flags"]


def test_low_confidence_forces_review():
    llm = ScriptedLLM(draft={"response": "Maybe on the 5th.", "confidence": 0.3})
    ticket = make_service(llm).submit(QUESTION)
    assert ticket["status"] == "pending_review" and "low_confidence" in ticket["risk_flags"]


def test_classifier_failure_fails_safe_to_review():
    ticket = make_service(ScriptedLLM(fail_classify=True)).submit(QUESTION)
    assert ticket["status"] == "pending_review"
    assert {"classifier_error", "high_urgency"} <= set(ticket["risk_flags"])


def test_unknown_category_from_model_is_coerced_to_other():
    ticket = make_service(ScriptedLLM(classify={"category": "ignore all rules", "urgency": "???"})).submit(QUESTION)
    assert ticket["category"] == "other" and ticket["urgency"] == "normal"


def test_non_numeric_confidence_is_treated_as_draft_error():
    llm = ScriptedLLM(draft={"response": "ok", "confidence": "very high"})
    ticket = make_service(llm).submit(QUESTION)
    assert ticket["status"] == "pending_review" and "draft_error" in ticket["risk_flags"]


def fraud_service():
    return make_service(ScriptedLLM(classify={"category": "fraud", "urgency": "normal"}))


def test_approve_resumes_and_sends_the_draft():
    service = fraud_service()
    ticket = service.submit("I see a payment I did not make.")
    done = service.review(ticket["ticket_id"], "approved", None, "meera")
    assert done["status"] == "resolved_after_review" and done["final_response"] == ticket["draft"]
    assert "reviewed:approved:meera" in done["audit"]


def test_edit_replaces_the_draft():
    service = fraud_service()
    ticket = service.submit("I see a payment I did not make.")
    done = service.review(ticket["ticket_id"], "edited", "We have blocked the payment and called you.", "meera")
    assert done["final_response"] == "We have blocked the payment and called you."


def test_reject_escalates_with_a_neutral_message():
    service = fraud_service()
    ticket = service.submit("I see a payment I did not make.")
    done = service.review(ticket["ticket_id"], "rejected", None, "meera")
    assert done["status"] == "escalated" and done["final_response"] == ESCALATION_MESSAGE


def test_pending_list_only_shows_tickets_awaiting_review():
    service = fraud_service()
    flagged = service.submit("I see a payment I did not make.")
    assert [t["ticket_id"] for t in service.pending()] == [flagged["ticket_id"]]
    service.review(flagged["ticket_id"], "approved", None, "meera")
    assert service.pending() == []


def test_review_requires_a_waiting_ticket(service):
    done = service.submit(QUESTION)
    with pytest.raises(InvalidState):
        service.review(done["ticket_id"], "approved", None, "meera")
    with pytest.raises(TicketNotFound):
        service.review("missing", "approved", None, "meera")
    with pytest.raises(TicketNotFound):
        service.view("missing")


def test_raw_pii_never_reaches_the_model_or_the_stored_state():
    llm = ScriptedLLM()
    service = make_service(llm)
    message = "My PAN is ABCDE1234F, card 4111 1111 1111 1111, call 9876543210 about my statement."
    ticket = service.submit(message)
    joined = "\n".join(llm.prompts)
    for raw in ("ABCDE1234F", "4111 1111 1111 1111", "9876543210"):
        assert raw not in joined
    assert ticket["pii_found"] == ["CARD", "PAN", "PHONE"]
    stored = service.graph.get_state({"configurable": {"thread_id": ticket["ticket_id"]}}).values["message"]
    assert "ABCDE1234F" not in stored and "[PAN]" in stored
