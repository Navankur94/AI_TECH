import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

from .conftest import ScriptedLLM, make_service

HEADERS = {"x-api-key": "test-key"}


def client_for(llm=None):
    settings = Settings(api_keys=("test-key",))
    return TestClient(create_app(make_service(llm or ScriptedLLM()), settings))


def test_health_open_and_other_routes_need_key():
    c = client_for()
    assert c.get("/health").status_code == 200
    assert c.post("/v1/tickets", json={"message": "hello there"}).status_code == 401
    assert c.get("/v1/tickets").status_code == 401


def test_server_without_keys_refuses():
    app = create_app(make_service(ScriptedLLM()), Settings(api_keys=()))
    assert TestClient(app).get("/v1/tickets", headers=HEADERS).status_code == 503


def test_low_risk_ticket_is_auto_resolved():
    r = client_for().post("/v1/tickets", headers=HEADERS, json={"message": "When is my statement emailed?"})
    assert r.status_code == 201 and r.json()["status"] == "auto_resolved"


def test_review_flow_over_http():
    c = client_for(ScriptedLLM(classify={"category": "complaint", "urgency": "normal"}))
    ticket = c.post("/v1/tickets", headers=HEADERS, json={"message": "Your service has been terrible."}).json()
    assert ticket["status"] == "pending_review"
    assert [t["ticket_id"] for t in c.get("/v1/tickets", headers=HEADERS).json()["pending"]] == [ticket["ticket_id"]]

    done = c.post(f"/v1/tickets/{ticket['ticket_id']}/review", headers=HEADERS,
                  json={"decision": "approved", "reviewer": "meera"})
    assert done.status_code == 200 and done.json()["status"] == "resolved_after_review"

    again = c.post(f"/v1/tickets/{ticket['ticket_id']}/review", headers=HEADERS,
                   json={"decision": "approved", "reviewer": "meera"})
    assert again.status_code == 409


def test_edited_decision_requires_response_text():
    c = client_for(ScriptedLLM(classify={"category": "fraud", "urgency": "normal"}))
    ticket = c.post("/v1/tickets", headers=HEADERS, json={"message": "Unknown payment on my account."}).json()
    r = c.post(f"/v1/tickets/{ticket['ticket_id']}/review", headers=HEADERS,
               json={"decision": "edited", "reviewer": "meera"})
    assert r.status_code == 422


def test_unknown_ticket_is_404():
    c = client_for()
    assert c.get("/v1/tickets/nope", headers=HEADERS).status_code == 404
    r = c.post("/v1/tickets/nope/review", headers=HEADERS, json={"decision": "approved", "reviewer": "m"})
    assert r.status_code == 404


def test_short_message_fails_validation():
    assert client_for().post("/v1/tickets", headers=HEADERS, json={"message": "hi"}).status_code == 422


@pytest.mark.parametrize("decision", ["maybe", ""])
def test_invalid_decision_is_rejected(decision):
    c = client_for(ScriptedLLM(classify={"category": "fraud", "urgency": "normal"}))
    ticket = c.post("/v1/tickets", headers=HEADERS, json={"message": "Unknown payment on my account."}).json()
    r = c.post(f"/v1/tickets/{ticket['ticket_id']}/review", headers=HEADERS,
               json={"decision": decision, "reviewer": "m"})
    assert r.status_code == 422
