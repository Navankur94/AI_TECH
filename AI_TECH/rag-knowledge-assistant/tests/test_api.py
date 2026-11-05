import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app

HEADERS = {"x-api-key": "test-key"}
DOC = b"# Leave\n\nEmployees get 24 days of annual leave. Ten unused days carry forward.\n"


@pytest.fixture
def client(service, settings):
    return TestClient(create_app(service, settings))


def test_health_needs_no_key(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_missing_or_wrong_key_is_rejected(client):
    assert client.get("/v1/documents").status_code == 401
    assert client.get("/v1/documents", headers={"x-api-key": "nope"}).status_code == 401


def test_server_without_keys_refuses_requests(service):
    app = create_app(service, Settings(api_keys=()))
    assert TestClient(app).get("/v1/documents", headers=HEADERS).status_code == 503


def test_upload_list_ask_delete_flow(client):
    up = client.post("/v1/documents", headers=HEADERS, files={"file": ("leave.md", DOC)})
    assert up.status_code == 200 and up.json()["chunks"] >= 1
    assert client.get("/v1/documents", headers=HEADERS).json()["documents"][0]["source"] == "leave.md"

    ask = client.post("/v1/ask", headers=HEADERS, json={"question": "how many days of annual leave"})
    body = ask.json()
    assert ask.status_code == 200 and body["grounded"] is True
    assert body["citations"][0]["source"] == "leave.md"

    assert client.delete("/v1/documents/leave.md", headers=HEADERS).status_code == 200
    assert client.delete("/v1/documents/leave.md", headers=HEADERS).status_code == 404


def test_unsupported_extension_is_rejected(client):
    r = client.post("/v1/documents", headers=HEADERS, files={"file": ("run.exe", b"x")})
    assert r.status_code == 415


def test_oversized_upload_is_rejected(service):
    app = create_app(service, Settings(api_keys=("test-key",), max_upload_bytes=100))
    r = TestClient(app).post("/v1/documents", headers=HEADERS, files={"file": ("big.md", b"a" * 500)})
    assert r.status_code == 413


def test_path_traversal_in_filename_is_neutralised(client):
    r = client.post("/v1/documents", headers=HEADERS, files={"file": ("../../etc/passwd.md", DOC)})
    assert r.status_code == 200 and r.json()["source"] == "passwd.md"


def test_short_question_fails_validation(client):
    assert client.post("/v1/ask", headers=HEADERS, json={"question": "a"}).status_code == 422


def test_llm_failure_becomes_502(service, settings):
    service.ingest("leave.md", DOC.decode())

    def boom(system, user):
        raise RuntimeError("provider down")

    service.llm.complete = boom
    r = TestClient(create_app(service, settings)).post(
        "/v1/ask", headers=HEADERS, json={"question": "how many days of annual leave"})
    assert r.status_code == 502
