import json
import logging
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

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


@pytest.fixture(autouse=True)
def _enable_triage_logger(monkeypatch):
    """Alembic's `fileConfig` (run by the integration suite in the same
    process) disables already-created loggers; re-enable ours for caplog."""
    monkeypatch.setattr(logging.getLogger("app.workflows.background"), "disabled", False)


@pytest.fixture
def client():
    # Imported after env isolation is active so any module-level settings
    # access (there is none currently, but this keeps the fixture safe)
    # sees the test environment.
    from app.main import app

    return TestClient(app)


@pytest.fixture(autouse=True)
def mock_db_persistence():
    """This file is pure unit tests — no real database.

    The endpoint's actual DB write (and its idempotency) is exercised for
    real against a test Postgres in tests/integration/test_events_api_db.py.
    Here we just stub it out so these tests stay fast and DB-free, while
    still letting us assert the endpoint *tried* to persist the right
    thing.
    """
    with patch("app.api.events._persist_to_db") as mock_persist:
        yield mock_persist


@pytest.fixture(autouse=True)
def mock_triage(monkeypatch):
    """Background triage is unit-tested in tests/test_triage_pipeline.py;
    here it's stubbed so these tests only check that it gets scheduled."""
    mock = MagicMock()
    monkeypatch.setattr("app.api.events.run_triage_safely", mock)
    return mock


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


def test_post_events_valid_signature_stores_event_and_returns_200(
    client, tmp_path, mock_db_persistence
):
    body = json.dumps(SAMPLE_EVENT).encode("utf-8")

    response = _signed_post(client, body)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

    mock_db_persistence.assert_called_once_with("evt-123", SAMPLE_EVENT)

    stored_path = tmp_path / "requests" / "events" / "evt-123.json"
    assert stored_path.exists()
    assert json.loads(stored_path.read_text()) == SAMPLE_EVENT


def test_post_events_duplicate_delivery_does_not_error_and_dedupes_the_file(
    client, tmp_path
):
    # Real DB-level idempotency (same event_id twice -> one row) is
    # exercised against a real database in
    # tests/integration/test_repository.py; here we just confirm the API
    # doesn't error on a repeat delivery and the optional file dump stays
    # deduplicated by filename.
    body = json.dumps(SAMPLE_EVENT).encode("utf-8")

    first = _signed_post(client, body)
    second = _signed_post(client, body)

    assert first.status_code == 200
    assert second.status_code == 200

    events_dir = tmp_path / "requests" / "events"
    assert [p.name for p in events_dir.iterdir()] == ["evt-123.json"]


def test_post_events_invalid_signature_returns_401_and_stores_nothing(
    client, tmp_path, mock_db_persistence
):
    body = json.dumps(SAMPLE_EVENT).encode("utf-8")

    response = _signed_post(client, body, secret="wrong-secret")

    assert response.status_code == 401
    assert not (tmp_path / "requests").exists()
    mock_db_persistence.assert_not_called()


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


# --- POST /events -> background triage -----------------------------------------


SYNTHETIC_SUBJECT = "Synthetic subject that must never be logged"

FULL_EVENT = {
    **SAMPLE_EVENT,
    "id": "evt-full-1",
    "data": {
        "application_id": "app-1",
        "object": {
            "id": "msg-full-1",
            "grant_id": "grant-test",
            "thread_id": "thread-full-1",
            "subject": SYNTHETIC_SUBJECT,
            "from": [{"name": "Test Sender", "email": "sender@example.com"}],
            "to": [{"email": "candidate@example.com"}],
            "date": 1706811644,
            "body": "Synthetic body.",
        },
    },
}


def test_post_events_schedules_background_triage(client, mock_triage):
    body = json.dumps(SAMPLE_EVENT).encode("utf-8")

    response = _signed_post(client, body)

    assert response.status_code == 200
    mock_triage.assert_called_once_with("evt-123", SAMPLE_EVENT)


def test_post_events_invalid_signature_schedules_no_triage(client, mock_triage):
    body = json.dumps(SAMPLE_EVENT).encode("utf-8")

    _signed_post(client, body, secret="wrong-secret")

    mock_triage.assert_not_called()


def test_post_events_agent_exception_does_not_break_the_request(
    client, monkeypatch, caplog
):
    from app.workflows.background import run_triage_safely

    monkeypatch.setattr("app.api.events.run_triage_safely", run_triage_safely)
    monkeypatch.setattr("app.workflows.background.session_scope", nullcontext)
    monkeypatch.setattr(
        "app.workflows.triage_workflow.insert_email_if_new",
        lambda session, email: SimpleNamespace(id=1),
    )
    analyzer = MagicMock(side_effect=RuntimeError(SYNTHETIC_SUBJECT))
    monkeypatch.setattr("app.workflows.triage_workflow.analyze_with_metadata", analyzer)
    body = json.dumps(FULL_EVENT).encode("utf-8")

    with caplog.at_level(logging.INFO):
        response = _signed_post(client, body)

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    analyzer.assert_called_once()
    assert "evt-full-1" in caplog.text
    assert "RuntimeError" in caplog.text
    assert SYNTHETIC_SUBJECT not in caplog.text
