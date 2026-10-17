"""Integration tests: /actions endpoints and the dashboard, real test DB,
Nylas mocked."""

from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

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
