"""Integration tests for app/database/repository.py against real Postgres."""

from datetime import datetime, timezone

from app.database.models import ActionType, PendingActionStatus
from app.database.repository import (
    create_pending_action,
    get_or_create_email,
    save_classification,
    save_raw_event,
    update_pending_action_status,
    write_audit_log,
)
from app.schemas.email import (
    Email,
    EmailAnalysis,
    EmailParticipant,
    SuggestedAction,
    Urgency,
)

SAMPLE_EMAIL = Email(
    id="msg-1",
    thread_id="thread-1",
    grant_id="grant-1",
    sender=EmailParticipant(name="Jane Recruiter", email="jane@example.com"),
    recipients=[EmailParticipant(name="Candidate", email="candidate@example.com")],
    subject="Interview invite",
    body="Are you free Tuesday?",
    received_at=datetime(2026, 2, 5, 9, 0, tzinfo=timezone.utc),
)

SAMPLE_ANALYSIS = EmailAnalysis(
    category="interview_invite",
    company="Acme Corp",
    role="Backend Engineer",
    deadline=None,
    urgency=Urgency.HIGH,
    suggested_action=SuggestedAction.CREATE_EVENT,
    reasoning="Sender proposes a specific interview time.",
    confidence=0.92,
)


# --- raw_events --------------------------------------------------------------


def test_save_raw_event_persists_payload(db_session):
    event = {"id": "evt-1", "type": "message.created", "data": {"foo": "bar"}}

    saved = save_raw_event(db_session, "evt-1", event)

    assert saved.event_id == "evt-1"
    assert saved.payload == event
    assert saved.received_at is not None


def test_save_raw_event_is_idempotent_on_duplicate_event_id(db_session):
    event = {"id": "evt-dup", "type": "message.created"}

    first = save_raw_event(db_session, "evt-dup", event)
    second = save_raw_event(db_session, "evt-dup", {**event, "extra": "ignored"})

    assert first.id == second.id
    # First write wins; duplicate delivery doesn't overwrite or duplicate.
    assert second.payload == event


# --- emails --------------------------------------------------------------------


def test_get_or_create_email_inserts_new_record(db_session):
    record = get_or_create_email(db_session, SAMPLE_EMAIL)

    assert record.id is not None
    assert record.nylas_message_id == "msg-1"
    assert record.sender_email == "jane@example.com"
    assert record.recipients == [{"name": "Candidate", "email": "candidate@example.com"}]


def test_get_or_create_email_is_idempotent_for_same_message_id(db_session):
    first = get_or_create_email(db_session, SAMPLE_EMAIL)
    second = get_or_create_email(db_session, SAMPLE_EMAIL)

    assert first.id == second.id


# --- classifications -----------------------------------------------------------


def test_save_classification_links_to_email(db_session):
    email_record = get_or_create_email(db_session, SAMPLE_EMAIL)

    classification = save_classification(
        db_session,
        email_record.id,
        SAMPLE_ANALYSIS,
        model="gpt-test",
        prompt_version="analyze_v1",
        input_tokens=120,
        output_tokens=40,
        latency_ms=850,
    )

    assert classification.email_id == email_record.id
    assert classification.category == "interview_invite"
    assert classification.confidence == 0.92
    assert classification.model == "gpt-test"
    assert classification.prompt_version == "analyze_v1"
    assert classification.input_tokens == 120


# --- pending_actions + audit_log -----------------------------------------------


def test_create_and_update_pending_action(db_session):
    email_record = get_or_create_email(db_session, SAMPLE_EMAIL)

    pending_action = create_pending_action(
        db_session,
        email_record.id,
        ActionType.CREATE_EVENT,
        payload={"start": "2026-02-07T14:00:00-08:00"},
    )

    assert pending_action.status == PendingActionStatus.PENDING
    assert pending_action.decided_at is None

    updated = update_pending_action_status(
        db_session, pending_action.id, PendingActionStatus.APPROVED
    )

    assert updated.status == PendingActionStatus.APPROVED
    assert updated.decided_at is not None


def test_write_audit_log_links_to_pending_action(db_session):
    email_record = get_or_create_email(db_session, SAMPLE_EMAIL)
    pending_action = create_pending_action(
        db_session, email_record.id, ActionType.DRAFT_REPLY, payload={}
    )

    entry = write_audit_log(
        db_session,
        actor="human:ehsan",
        action="approved",
        pending_action_id=pending_action.id,
    )

    assert entry.pending_action_id == pending_action.id
    assert entry.actor == "human:ehsan"
    assert entry.at is not None


def test_write_audit_log_without_pending_action_is_allowed(db_session):
    entry = write_audit_log(db_session, actor="system", action="startup")

    assert entry.pending_action_id is None
