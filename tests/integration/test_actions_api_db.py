"""Integration tests: /actions endpoints and the dashboard, real test DB,
Nylas mocked."""

import logging
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from app.config import get_settings
from app.database.models import ActionType
from app.database.repository import get_or_create_email, save_classification
from app.database.session import session_scope
from app.schemas.email import (
    Email,
    EmailAnalysis,
    EmailParticipant,
    SuggestedAction,
    Urgency,
)
from app.services.actions import create_pending_action

API_KEY = {"X-API-Key": "app-test"}


def _email(subject: str = "Re: opportunity") -> Email:
    return Email(
        id="msg-1",
        thread_id="thread-1",
        grant_id="grant-1",
        sender=EmailParticipant(name="Jane Recruiter", email="jane@example.com"),
        recipients=[EmailParticipant(email="candidate@example.com")],
        subject=subject,
        body="Email body",
        received_at=datetime(2026, 2, 1, 9, 0, tzinfo=timezone.utc),
    )


def _analysis() -> EmailAnalysis:
    return EmailAnalysis(
        category="other",
        urgency=Urgency.LOW,
        suggested_action=SuggestedAction.NONE,
        reasoning="Synthetic analysis for testing.",
        confidence=0.9,
    )


def _truncate_all(test_engine) -> None:
    with test_engine.begin() as conn:
        for table in (
            "audit_log",
            "pending_actions",
            "classifications",
            "emails",
            "raw_events",
        ):
            conn.execute(text(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE"))


@pytest.fixture(autouse=True)
def _clean_tables(app_db_env, test_engine):
    _truncate_all(test_engine)
    yield
    _truncate_all(test_engine)


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


def test_list_pending_actions(client):
    pending_action = create_pending_action(_email(), _analysis(), ActionType.ARCHIVE)

    response = client.get("/actions?status=pending", headers=API_KEY)

    assert response.status_code == 200
    assert [a["id"] for a in response.json()["actions"]] == [pending_action.id]


def test_approve_once_then_second_approve_is_409(client):
    pending_action = create_pending_action(_email(), _analysis(), ActionType.ARCHIVE)

    with patch("app.services.actions.archive_message") as mock_archive:
        first = client.post(f"/actions/{pending_action.id}/approve", headers=API_KEY)
        second = client.post(f"/actions/{pending_action.id}/approve", headers=API_KEY)

    assert first.status_code == 200
    assert first.json()["status"] == "executed"
    assert second.status_code == 409
    mock_archive.assert_called_once_with("grant-1", "msg-1")


def test_reject_then_approve_is_409_and_never_calls_nylas(client):
    pending_action = create_pending_action(_email(), _analysis(), ActionType.ARCHIVE)

    with patch("app.services.actions.archive_message") as mock_archive:
        rejected = client.post(f"/actions/{pending_action.id}/reject", headers=API_KEY)
        approved = client.post(f"/actions/{pending_action.id}/approve", headers=API_KEY)

    assert rejected.status_code == 200
    assert rejected.json()["status"] == "rejected"
    assert approved.status_code == 409
    mock_archive.assert_not_called()


@pytest.fixture
def dry_run(app_db_env, monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")
    get_settings.cache_clear()


@pytest.fixture
def actions_logger(monkeypatch) -> None:
    """Alembic's `fileConfig` (run by this suite's migrations) disables
    already-created loggers; re-enable ours for caplog."""
    monkeypatch.setattr(logging.getLogger("app.services.actions"), "disabled", False)


@pytest.fixture
def nylas_client(monkeypatch) -> MagicMock:
    """The Nylas client, as the action integration would obtain it."""
    mock = MagicMock()
    monkeypatch.setattr("app.integrations.nylas.actions.get_client", mock)
    return mock


def _audit_rows(test_engine, pending_action_id: int) -> list[tuple[str, bool]]:
    with test_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT action, dry_run FROM audit_log "
                "WHERE pending_action_id = :id ORDER BY id"
            ),
            {"id": pending_action_id},
        ).all()
    return [(row.action, row.dry_run) for row in rows]


@pytest.mark.parametrize("action_type", list(ActionType))
def test_dry_run_approve_executes_without_calling_nylas(
    client,
    dry_run,
    nylas_client,
    actions_logger,
    test_engine,
    caplog,
    action_type,
):
    pending_action = create_pending_action(_email(), _analysis(), action_type)

    with caplog.at_level(logging.INFO, logger="app.services.actions"):
        response = client.post(
            f"/actions/{pending_action.id}/approve", headers=API_KEY
        )

    assert response.status_code == 200
    assert response.json()["status"] == "executed"
    nylas_client.assert_not_called()
    assert _audit_rows(test_engine, pending_action.id) == [
        ("created", False),
        ("approved", True),
        ("executed", True),
    ]
    assert f"pending_action_id={pending_action.id}" in caplog.text
    assert action_type.value in caplog.text
    assert "dry-run-" in caplog.text
    for content in ("opportunity", "Jane Recruiter", "jane@example.com"):
        assert content not in caplog.text


def test_dry_run_double_approve_is_409_and_never_calls_nylas(
    client, dry_run, nylas_client
):
    pending_action = create_pending_action(_email(), _analysis(), ActionType.ARCHIVE)

    first = client.post(f"/actions/{pending_action.id}/approve", headers=API_KEY)
    second = client.post(f"/actions/{pending_action.id}/approve", headers=API_KEY)

    assert first.status_code == 200
    assert second.status_code == 409
    nylas_client.assert_not_called()


def test_approve_without_api_key_is_401_and_leaves_action_pending(
    client, dry_run, nylas_client, test_engine
):
    pending_action = create_pending_action(_email(), _analysis(), ActionType.ARCHIVE)

    response = client.post(f"/actions/{pending_action.id}/approve")

    assert response.status_code == 401
    nylas_client.assert_not_called()
    with test_engine.connect() as conn:
        status = conn.execute(
            text("SELECT status FROM pending_actions WHERE id = :id"),
            {"id": pending_action.id},
        ).scalar_one()
    assert status == "pending"


def test_unknown_action_is_404(client):
    response = client.post("/actions/999/approve", headers=API_KEY)

    assert response.status_code == 404


def test_dashboard_lists_pending_action_and_escapes_subject(client):
    email = _email(subject="<script>alert(1)</script>")
    pending_action = create_pending_action(email, _analysis(), ActionType.ARCHIVE)
    with session_scope() as session:
        record = get_or_create_email(session, email)
        save_classification(
            session, record.id, _analysis(), model="gpt-test", prompt_version="v1"
        )

    response = client.get("/")

    assert response.status_code == 200
    assert f'data-action-id="{pending_action.id}"' in response.text
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in response.text
    assert "<script>alert(1)" not in response.text
    assert "other" in response.text and "(0.90)" in response.text
    assert "app-test" not in response.text
