"""Repository functions: plain functions over an explicit `Session`.

Each function takes the `Session` it should use rather than opening its
own — callers compose multiple calls under one transaction (typically
via `app.database.session.session_scope()`), and tests can pass in a
session bound to a test database/transaction directly.
"""

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, selectinload

from app.database.models import (
    ActionType,
    AuditLog,
    Classification,
    EmailRecord,
    PendingAction,
    PendingActionStatus,
    RawEvent,
)
from app.schemas.email import Email, EmailAnalysis


def save_raw_event(session: Session, event_id: str, payload: dict) -> RawEvent:
    """Idempotently store a raw webhook event, keyed by its event id.

    Webhook deliveries are "at least once" (Nylas may retry/duplicate),
    so a repeat `event_id` is a no-op, not an error.
    """
    stmt = (
        pg_insert(RawEvent)
        .values(event_id=event_id, payload=payload)
        .on_conflict_do_nothing(index_elements=["event_id"])
    )
    session.execute(stmt)
    session.flush()
    return session.scalars(
        select(RawEvent).where(RawEvent.event_id == event_id)
    ).one()


def get_or_create_email(session: Session, email: Email) -> EmailRecord:
    """Find the `EmailRecord` for this canonical `Email`, inserting if new."""
    existing = session.scalars(
        select(EmailRecord).where(EmailRecord.nylas_message_id == email.id)
    ).one_or_none()
    if existing is not None:
        return existing

    record = EmailRecord(
        nylas_message_id=email.id,
        thread_id=email.thread_id,
        grant_id=email.grant_id,
        sender_name=email.sender.name,
        sender_email=email.sender.email,
        recipients=[
            {"name": r.name, "email": r.email} for r in email.recipients
        ],
        subject=email.subject,
        body=email.body,
        received_at=email.received_at,
    )
    session.add(record)
    session.flush()
    return record


def save_classification(
    session: Session,
    email_record_id: int,
    analysis: EmailAnalysis,
    *,
    model: str,
    prompt_version: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: int | None = None,
) -> Classification:
    classification = Classification(
        email_id=email_record_id,
        category=analysis.category.value,
        confidence=analysis.confidence,
        model=model,
        prompt_version=prompt_version,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
    )
    session.add(classification)
    session.flush()
    return classification


def create_pending_action(
    session: Session,
    email_record_id: int,
    action_type: ActionType,
    payload: dict,
) -> PendingAction:
    pending_action = PendingAction(
        email_id=email_record_id,
        action_type=action_type,
        payload=payload,
        status=PendingActionStatus.PENDING,
    )
    session.add(pending_action)
    session.flush()
    return pending_action


def update_pending_action_status(
    session: Session,
    pending_action_id: int,
    status: PendingActionStatus,
    decided_at: datetime | None = None,
) -> PendingAction:
    pending_action = session.get_one(PendingAction, pending_action_id)
    pending_action.status = status
    pending_action.decided_at = decided_at or datetime.now()
    session.flush()
    return pending_action


def write_audit_log(
    session: Session,
    actor: str,
    action: str,
    pending_action_id: int | None = None,
) -> AuditLog:
    entry = AuditLog(
        actor=actor, action=action, pending_action_id=pending_action_id
    )
    session.add(entry)
    session.flush()
    return entry


def list_pending_actions(
    session: Session, status: PendingActionStatus
) -> list[PendingAction]:
    return list(
        session.scalars(
            select(PendingAction)
            .where(PendingAction.status == status)
            .order_by(PendingAction.created_at.desc(), PendingAction.id.desc())
        )
    )


def list_recent_emails(session: Session, limit: int = 50) -> list[EmailRecord]:
    """Most recent emails, with classifications and pending actions eagerly
    loaded so they're usable after the session closes."""
    return list(
        session.scalars(
            select(EmailRecord)
            .options(
                selectinload(EmailRecord.classifications),
                selectinload(EmailRecord.pending_actions),
            )
            .order_by(EmailRecord.received_at.desc(), EmailRecord.id.desc())
            .limit(limit)
        )
    )
