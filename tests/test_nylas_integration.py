import hashlib
import hmac
import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from app.config import get_settings
from app.integrations.nylas.mapper import nylas_payload_to_email
from app.integrations.nylas.messages import list_recent_messages
from app.integrations.nylas.webhooks import (
    compute_signature,
    store_raw_event,
    verify_webhook_signature,
)

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
    """Isolate settings from the real .env, same pattern as test_config.py."""
    monkeypatch.chdir(tmp_path)
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


SAMPLE_MESSAGE_PAYLOAD = {
    "id": "msg-1",
    "object": "message",
    "grant_id": "grant-test",
    "thread_id": "thread-1",
    "subject": "Interview invite",
    "from": [{"name": "Jane Recruiter", "email": "jane@example.com"}],
    "to": [{"name": "Candidate", "email": "candidate@example.com"}],
    "date": 1706811644,
    "body": "<p>Are you free Tuesday?</p>",
    "snippet": "Are you free Tuesday?",
}


# --- compute_signature / verify_webhook_signature ---------------------------


def test_compute_signature_matches_independently_computed_hmac():
    body = b'{"hello": "world"}'
    secret = "my-secret"

    signature = compute_signature(body, secret)

    expected = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    assert signature == expected


def test_verify_webhook_signature_accepts_valid_signature():
    body = b'{"id": "evt-1"}'
    signature = compute_signature(body, "super-secret")

    assert verify_webhook_signature(body, signature) is True


def test_verify_webhook_signature_rejects_tampered_body():
    signature = compute_signature(b'{"id": "evt-1"}', "super-secret")

    assert verify_webhook_signature(b'{"id": "evt-2"}', signature) is False


def test_verify_webhook_signature_rejects_wrong_secret():
    body = b'{"id": "evt-1"}'
    signature = compute_signature(body, "wrong-secret")

    assert verify_webhook_signature(body, signature) is False


def test_verify_webhook_signature_rejects_missing_header():
    body = b'{"id": "evt-1"}'

    assert verify_webhook_signature(body, None) is False
    assert verify_webhook_signature(body, "") is False


# --- store_raw_event ---------------------------------------------------------


def test_store_raw_event_writes_json_named_by_event_id(tmp_path):
    directory = tmp_path / "events"
    event = {"id": "evt-123", "type": "message.created", "data": {"foo": "bar"}}

    path = store_raw_event(event, directory=directory)

    assert path == directory / "evt-123.json"
    assert json.loads(path.read_text(encoding="utf-8")) == event


def test_store_raw_event_missing_id_raises(tmp_path):
    with pytest.raises(ValueError):
        store_raw_event({"type": "message.created"}, directory=tmp_path / "events")


# --- nylas_payload_to_email ---------------------------------------------------


def test_nylas_payload_to_email_maps_valid_payload():
    email = nylas_payload_to_email(SAMPLE_MESSAGE_PAYLOAD)

    assert email.id == "msg-1"
    assert email.thread_id == "thread-1"
    assert email.grant_id == "grant-test"
    assert email.sender.email == "jane@example.com"
    assert email.sender.name == "Jane Recruiter"
    assert len(email.recipients) == 1
    assert email.recipients[0].email == "candidate@example.com"
    assert email.subject == "Interview invite"
    assert email.body == "<p>Are you free Tuesday?</p>"
    assert email.received_at == datetime.fromtimestamp(1706811644, tz=timezone.utc)


def test_nylas_payload_to_email_missing_required_field_raises():
    payload = {k: v for k, v in SAMPLE_MESSAGE_PAYLOAD.items() if k != "thread_id"}

    with pytest.raises((ValidationError, KeyError)):
        nylas_payload_to_email(payload)


def test_nylas_payload_to_email_defaults_blank_subject_and_body():
    payload = {**SAMPLE_MESSAGE_PAYLOAD, "subject": None, "body": None}

    email = nylas_payload_to_email(payload)

    assert email.subject == ""
    assert email.body == ""


# --- list_recent_messages -----------------------------------------------------


def test_list_recent_messages_calls_sdk_with_configured_grant_and_limit():
    fake_messages = ["message-a", "message-b"]
    fake_response = MagicMock(data=fake_messages)
    fake_client = MagicMock()
    fake_client.messages.list.return_value = fake_response

    with patch(
        "app.integrations.nylas.messages.get_client", return_value=fake_client
    ):
        result = list_recent_messages(limit=5)

    fake_client.messages.list.assert_called_once_with(
        "grant-test", query_params={"limit": 5}
    )
    assert result == fake_messages
