"""app/services/actions.py: real DB, Nylas integration mocked."""

from datetime import date, datetime, timezone
from unittest.mock import patch

import pytest
from sqlalchemy import text

from app.database.models import ActionType, PendingActionStatus
from app.schemas.email import (
    Email,
    EmailAnalysis,
    EmailParticipant,
    SuggestedAction,
    Urgency,
)
from app.services.actions import PendingActionError, approve, create_pending_action, reject


def _email(email_id: str = "msg-1") -> Email:
    return Email(
        id=email_id,
        thread_id="thread-1",
        grant_id="grant-1",
        sender=EmailParticipant(name="Jane Recruiter", email="jane@example.com"),
        recipients=[EmailParticipant(email="candidate@example.com")],
        subject="Re: opportunity",
        body="Email body",
        received_at=datetime(2026, 2, 1, 9, 0, tzinfo=timezone.utc),
    )


def _analysis(
    suggested_action: SuggestedAction,
    deadline: date | None = None,
) -> EmailAnalysis:
    return EmailAnalysis(
        category="interview_invite",
        company="Acme Corp",
        role="Backend Engineer",
        deadline=deadline,
        urgency=Urgency.MEDIUM,
        suggested_action=suggested_action,
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
    """Every test in this file needs the app pointed at the test DB
    (`app_db_env`) and a clean slate (these tests commit real rows,
    since `app.services.actions` manages its own sessions — unlike
    `db_session`-based tests elsewhere, nothing here rolls back on its
    own). Truncate both before *and* after: before, so a prior failed
    run's leftovers don't affect this test; after, so this file's
    committed rows (e.g. a shared "msg-1" id) never leak into tests in
    other files that happen to run later in the same session."""
    _truncate_all(test_engine)
    yield
    _truncate_all(test_engine)


def _audit_actions(test_engine, pending_action_id: int) -> list[str]:
    with test_engine.connect() as conn:
        rows = conn.execute(
            text(
                "SELECT action FROM audit_log WHERE pending_action_id = :id "
                "ORDER BY id"
            ),
            {"id": pending_action_id},
        ).all()
    return [row.action for row in rows]


# --- create_pending_action ----------------------------------------------------


def test_create_pending_action_archive_has_empty_payload(test_engine):
    pending_action = create_pending_action(
        _email(), _analysis(SuggestedAction.NONE), ActionType.ARCHIVE
    )

    assert pending_action.action_type == ActionType.ARCHIVE
    assert pending_action.status == PendingActionStatus.PENDING
    assert pending_action.payload == {}
    assert _audit_actions(test_engine, pending_action.id) == ["created"]


def test_create_pending_action_draft_reply_builds_payload(test_engine):
    pending_action = create_pending_action(
        _email(), _analysis(SuggestedAction.DRAFT_REPLY), ActionType.DRAFT_REPLY
    )

    assert pending_action.action_type == ActionType.DRAFT_REPLY
    assert pending_action.payload["to"] == [
        {"name": "Jane Recruiter", "email": "jane@example.com"}
    ]
    assert "Backend Engineer" in pending_action.payload["body"]


def test_create_pending_action_create_event_uses_deadline(test_engine):
    pending_action = create_pending_action(
        _email(),
        _analysis(SuggestedAction.CREATE_EVENT, deadline=date(2026, 2, 10)),
        ActionType.CREATE_EVENT,
    )

    assert pending_action.action_type == ActionType.CREATE_EVENT
    assert pending_action.payload["when"] == {"date": "2026-02-10", "object": "date"}


def test_create_pending_action_create_event_without_deadline_uses_placeholder(
    test_engine,
):
    pending_action = create_pending_action(
        _email(),
        _analysis(SuggestedAction.CREATE_EVENT, deadline=None),
        ActionType.CREATE_EVENT,
    )

    assert pending_action.payload["when"]["object"] == "date"
    assert "placeholder" in pending_action.payload["description"].lower()


# --- approve(): archive --------------------------------------------------------


def test_approve_archive_calls_nylas_and_marks_executed(test_engine):
    pending_action = create_pending_action(
        _email(), _analysis(SuggestedAction.NONE), ActionType.ARCHIVE
    )

    with patch("app.services.actions.archive_message") as mock_archive:
        result = approve(pending_action.id)

    mock_archive.assert_called_once_with("grant-1", "msg-1")
    assert result.status == PendingActionStatus.EXECUTED
    assert _audit_actions(test_engine, pending_action.id) == ["created", "approved"]


# --- approve(): draft_reply ----------------------------------------------------


def test_approve_draft_reply_calls_create_draft_never_send(test_engine):
    pending_action = create_pending_action(
        _email(), _analysis(SuggestedAction.DRAFT_REPLY), ActionType.DRAFT_REPLY
    )

    with patch("app.services.actions.create_draft") as mock_create_draft:
        result = approve(pending_action.id)

    mock_create_draft.assert_called_once()
    args, kwargs = mock_create_draft.call_args
    assert args[0] == "grant-1"
    assert kwargs["subject"] == "Re: Re: opportunity"
    assert result.status == PendingActionStatus.EXECUTED


# --- approve(): create_event ---------------------------------------------------


def test_approve_create_event_calls_nylas(test_engine):
    pending_action = create_pending_action(
        _email(),
        _analysis(SuggestedAction.CREATE_EVENT, deadline=date(2026, 2, 10)),
        ActionType.CREATE_EVENT,
    )

    with patch("app.services.actions.create_calendar_event") as mock_create_event:
        result = approve(pending_action.id)

    mock_create_event.assert_called_once()
    args, kwargs = mock_create_event.call_args
    assert args[0] == "grant-1"
    assert kwargs["when"] == {"date": "2026-02-10", "object": "date"}
    assert result.status == PendingActionStatus.EXECUTED


# --- idempotency / invalid transitions -----------------------------------------


def test_double_approve_raises_and_calls_nylas_only_once(test_engine):
    pending_action = create_pending_action(
        _email(), _analysis(SuggestedAction.NONE), ActionType.ARCHIVE
    )

    with patch("app.services.actions.archive_message") as mock_archive:
        approve(pending_action.id)

        with pytest.raises(PendingActionError):
            approve(pending_action.id)

    mock_archive.assert_called_once()
    assert _audit_actions(test_engine, pending_action.id) == ["created", "approved"]


def test_approve_already_rejected_action_raises_and_never_calls_nylas(test_engine):
    pending_action = create_pending_action(
        _email(), _analysis(SuggestedAction.NONE), ActionType.ARCHIVE
    )
    reject(pending_action.id)

    with patch("app.services.actions.archive_message") as mock_archive:
        with pytest.raises(PendingActionError):
            approve(pending_action.id)

    mock_archive.assert_not_called()


def test_reject_already_approved_action_raises(test_engine):
    pending_action = create_pending_action(
        _email(), _analysis(SuggestedAction.NONE), ActionType.ARCHIVE
    )
    with patch("app.services.actions.archive_message"):
        approve(pending_action.id)

    with pytest.raises(PendingActionError):
        reject(pending_action.id)


def test_double_reject_raises(test_engine):
    pending_action = create_pending_action(
        _email(), _analysis(SuggestedAction.NONE), ActionType.ARCHIVE
    )
    reject(pending_action.id)

    with pytest.raises(PendingActionError):
        reject(pending_action.id)

    assert _audit_actions(test_engine, pending_action.id) == ["created", "rejected"]


def test_reject_writes_audit_log_and_never_calls_nylas(test_engine):
    pending_action = create_pending_action(
        _email(), _analysis(SuggestedAction.NONE), ActionType.ARCHIVE
    )

    with patch("app.services.actions.archive_message") as mock_archive:
        result = reject(pending_action.id)

    mock_archive.assert_not_called()
    assert result.status == PendingActionStatus.REJECTED
    assert _audit_actions(test_engine, pending_action.id) == ["created", "rejected"]


# --- approval failure handling --------------------------------------------------


def test_approve_marks_failed_and_audits_when_nylas_call_raises(test_engine):
    pending_action = create_pending_action(
        _email(), _analysis(SuggestedAction.NONE), ActionType.ARCHIVE
    )

    with patch(
        "app.services.actions.archive_message", side_effect=RuntimeError("boom")
    ):
        with pytest.raises(RuntimeError):
            approve(pending_action.id)

    with test_engine.connect() as conn:
        status = conn.execute(
            text("SELECT status FROM pending_actions WHERE id = :id"),
            {"id": pending_action.id},
        ).scalar()
    assert status == PendingActionStatus.FAILED.value
    assert _audit_actions(test_engine, pending_action.id) == [
        "created",
        "approve_failed",
    ]


def test_failed_action_can_still_not_be_approved_again(test_engine):
    pending_action = create_pending_action(
        _email(), _analysis(SuggestedAction.NONE), ActionType.ARCHIVE
    )
    with patch(
        "app.services.actions.archive_message", side_effect=RuntimeError("boom")
    ):
        with pytest.raises(RuntimeError):
            approve(pending_action.id)

    with patch("app.services.actions.archive_message") as mock_archive:
        with pytest.raises(PendingActionError):
            approve(pending_action.id)

    mock_archive.assert_not_called()
