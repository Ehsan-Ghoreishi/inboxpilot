import json

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings
from app.integrations.nylas.webhooks import compute_signature

REQUIRED_ENV = {
    "OPENAI_API_KEY": "sk-test",
    "OPENAI_MODEL": "gpt-test",
    "NYLAS_API_KEY": "nylas-test",
    "NYLAS_CLIENT_ID": "client-test",
    "NYLAS_GRANT_ID": "grant-test",
    "WEBHOOK_SECRET": "super-secret",
    "SERVER_URL": "http://localhost:8000",
    "DATABASE_URL": "postgresql://user:pass@localhost/db",
    "APP_API_KEY": "app-test",
    "PROCESSING_MODE": "manual",
}


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Isolate settings and relative `requests/events/` writes under tmp_path."""
    monkeypatch.chdir(tmp_path)
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client():
    # Imported after env isolation is active so any module-level settings
    # access (there is none currently, but this keeps the fixture safe)
    # sees the test environment.
    from app.main import app

    return TestClient(app)


SAMPLE_EVENT = {
    "specversion": "1.0",
    "type": "message.created",
    "source": "/google/emails/realtime",
    "id": "evt-123",
    "time": 1706811644,
    "data": {
        "application_id": "app-1",
        "object": {
            "id": "msg-1",
            "grant_id": "grant-test",
            "thread_id": "thread-1",
            "subject": "Interview invite",
        },
    },
}


def _signed_post(client, body: bytes, secret: str = "super-secret"):
    signature = compute_signature(body, secret)
    return client.post(
        "/events",
        content=body,
        headers={"x-nylas-signature": signature, "content-type": "application/json"},
    )


# --- GET /health -------------------------------------------------------------


def test_health_returns_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- GET /events (verification handshake) -------------------------------------


def test_get_events_echoes_challenge(client):
    response = client.get("/events", params={"challenge": "abc123"})

    assert response.status_code == 200
    assert response.text == "abc123"


def test_get_events_without_challenge_is_rejected(client):
    response = client.get("/events")

    assert response.status_code == 422


# --- POST /events (notification delivery) -------------------------------------


def test_post_events_valid_signature_stores_event_and_returns_200(client, tmp_path):
    body = json.dumps(SAMPLE_EVENT).encode("utf-8")

    response = _signed_post(client, body)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    stored_path = tmp_path / "requests" / "events" / "evt-123.json"
    assert stored_path.exists()
    assert json.loads(stored_path.read_text()) == SAMPLE_EVENT


def test_post_events_duplicate_delivery_is_idempotent(client, tmp_path):
    body = json.dumps(SAMPLE_EVENT).encode("utf-8")

    first = _signed_post(client, body)
    second = _signed_post(client, body)

    assert first.status_code == 200
    assert second.status_code == 200

    events_dir = tmp_path / "requests" / "events"
    assert [p.name for p in events_dir.iterdir()] == ["evt-123.json"]


def test_post_events_invalid_signature_returns_401_and_stores_nothing(
    client, tmp_path
):
    body = json.dumps(SAMPLE_EVENT).encode("utf-8")

    response = _signed_post(client, body, secret="wrong-secret")

    assert response.status_code == 401
    assert not (tmp_path / "requests").exists()


def test_post_events_missing_signature_header_returns_401(client):
    body = json.dumps(SAMPLE_EVENT).encode("utf-8")

    response = client.post("/events", content=body)

    assert response.status_code == 401


def test_post_events_tampered_body_returns_401(client):
    body = json.dumps(SAMPLE_EVENT).encode("utf-8")
    signature = compute_signature(body, "super-secret")
    tampered_body = json.dumps({**SAMPLE_EVENT, "id": "evt-999"}).encode("utf-8")

    response = client.post(
        "/events",
        content=tampered_body,
        headers={"x-nylas-signature": signature},
    )

    assert response.status_code == 401


def test_post_events_valid_signature_invalid_json_returns_400(client):
    body = b"not-json"

    response = _signed_post(client, body)

    assert response.status_code == 400


def test_post_events_valid_signature_missing_id_returns_400(client):
    event = {"type": "message.created", "data": {}}
    body = json.dumps(event).encode("utf-8")

    response = _signed_post(client, body)

    assert response.status_code == 400
