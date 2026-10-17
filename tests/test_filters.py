from datetime import datetime, timezone

from app.config import Settings
from app.schemas.email import Email, EmailParticipant
from app.workflows.filters import should_process

BASE_SETTINGS_KWARGS = {
    "openai_api_key": "sk-test",
    "openai_model": "gpt-test",
    "nylas_api_key": "nylas-test",
    "nylas_client_id": "client-test",
    "nylas_grant_id": "grant-test",
    "webhook_secret": "super-secret",
    "server_url": "http://localhost:8000",
    "database_url": "postgresql://user:pass@localhost/db",
    "app_api_key": "app-test",
    "confidence_threshold": 0.7,
}


def _settings(processing_mode: str, test_email: str | None = None) -> Settings:
    return Settings(
        **BASE_SETTINGS_KWARGS,
        processing_mode=processing_mode,
        test_email=test_email,
    )


def _email(sender_email: str) -> Email:
    return Email(
        id="msg-1",
        thread_id="thread-1",
        grant_id="grant-1",
        sender=EmailParticipant(name="Sender", email=sender_email),
        recipients=[EmailParticipant(email="candidate@example.com")],
        subject="Subject",
        body="Body",
        received_at=datetime(2026, 2, 5, 9, 0, tzinfo=timezone.utc),
    )


# --- live mode -----------------------------------------------------------


def test_live_mode_processes_any_sender():
    settings = _settings(processing_mode="live")

    assert should_process(_email("anyone@example.com"), settings) is True
    assert should_process(_email("someone-else@other.example"), settings) is True


def test_unrecognized_mode_defaults_to_processing_everything():
    settings = _settings(processing_mode="production")

    assert should_process(_email("anyone@example.com"), settings) is True


# --- test mode -------------------------------------------------------------


def test_test_mode_processes_matching_sender():
    settings = _settings(processing_mode="test", test_email="candidate@example.com")

    assert should_process(_email("candidate@example.com"), settings) is True


def test_test_mode_rejects_non_matching_sender():
    settings = _settings(processing_mode="test", test_email="candidate@example.com")

    assert should_process(_email("someone-else@example.com"), settings) is False


def test_test_mode_sender_match_is_case_insensitive():
    settings = _settings(processing_mode="test", test_email="Candidate@Example.com")

    assert should_process(_email("candidate@example.com"), settings) is True


def test_test_mode_without_test_email_processes_nothing():
    settings = _settings(processing_mode="test", test_email=None)

    assert should_process(_email("candidate@example.com"), settings) is False
