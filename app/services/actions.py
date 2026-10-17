"""Service layer: propose, approve, and reject pending actions.

Every state transition is gated here, server-side — never left to a
client's say-so — and every transition writes an `audit_log` row.
`approve()` is the *only* place that calls Nylas and takes a real,
consequential action: nothing executes before a human approves it (per
CLAUDE.md's human-confirmation rule). Rejecting, or approving twice,
never calls Nylas. With `DRY_RUN=true`, approving doesn't call Nylas
either: it goes through the same state transitions and audit trail, but
the action is only logged and given a fake id.

Nothing in this module (or the Nylas calls it makes) permanently
deletes anything — "archive" is a folder move, and a draft is created
but never sent.
"""

import logging
from datetime import timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy.orm import Session

from app.config import get_settings
from app.database.models import (
    ActionType,
    EmailRecord,
    PendingAction,
    PendingActionStatus,
)
from app.database.repository import (
    create_pending_action as _db_create_pending_action,
    get_or_create_email,
    update_pending_action_status,
    write_audit_log,
)
from app.database.session import session_scope
from app.integrations.nylas.actions import (
    archive_message,
    create_calendar_event,
    create_draft,
)
from app.schemas.email import Email, EmailAnalysis

logger = logging.getLogger(__name__)


class PendingActionError(Exception):
    """A requested transition isn't allowed from the action's current state."""


def _build_payload(
    email: Email, analysis: EmailAnalysis, action_type: ActionType
) -> dict[str, Any]:
    """What `approve()` will need to actually perform the action later."""
    if action_type is ActionType.ARCHIVE:
        return {}

    if action_type is ActionType.DRAFT_REPLY:
        role = analysis.role or "the role"
        company_clause = f" at {analysis.company}" if analysis.company else ""
        return {
            "to": [{"name": email.sender.name, "email": email.sender.email}],
            "subject": f"Re: {email.subject}",
            "body": (
                f"Thank you for reaching out about {role}{company_clause}. "
                "I'd love to connect further."
            ),
            "reply_to_message_id": email.id,
        }

    if action_type is ActionType.CREATE_EVENT:
        # Only a date (no time) is extracted today, so this is always
        # an all-day placeholder event — refining it to a specific time
        # slot is future work, not this step's job. If no deadline was
        # extracted at all, fall back to one week out rather than
        # baking an invalid `when` into the payload (which would only
        # fail later, at approval time, far from the actual cause).
        if analysis.deadline is not None:
            event_date = analysis.deadline
            description = analysis.reasoning
        else:
            event_date = email.received_at.date() + timedelta(days=7)
            description = (
                f"{analysis.reasoning}\n\n"
                "No specific date was extracted from the email — this "
                "placeholder date needs confirming before relying on it."
            )
        return {
            "title": f"Interview: {analysis.company or email.sender.email}",
            "description": description,
            "when": {"date": event_date.isoformat(), "object": "date"},
        }

    raise ValueError(f"Unhandled action type: {action_type}")  # pragma: no cover


def create_pending_action(
    email: Email, analysis: EmailAnalysis, action_type: ActionType
) -> PendingAction:
    """Persist a proposed `action_type` action for `email`. Does not call
    Nylas — that only happens on `approve()`.

    `action_type` is the caller's (deterministic routing's) decision.
    It is deliberately *not* derived from `analysis.suggested_action`:
    that field is model output, which a malicious email can steer.
    """
    with session_scope() as session:
        email_record = get_or_create_email(session, email)
        return propose_action(session, email_record.id, email, analysis, action_type)


def propose_action(
    session: Session,
    email_record_id: int,
    email: Email,
    analysis: EmailAnalysis,
    action_type: ActionType,
) -> PendingAction:
    """`create_pending_action()` within the caller's transaction, for an
    already-persisted email."""
    payload = _build_payload(email, analysis, action_type)
    pending_action = _db_create_pending_action(
        session, email_record_id, action_type, payload
    )
    write_audit_log(
        session,
        actor="system",
        action="created",
        pending_action_id=pending_action.id,
    )
    return pending_action


def _execute(pending_action: PendingAction, email_record: EmailRecord) -> str:
    """Perform the real Nylas action; return the id of what it acted on
    or created."""
    grant_id = email_record.grant_id
    payload = pending_action.payload

    if pending_action.action_type == ActionType.ARCHIVE:
        archive_message(grant_id, email_record.nylas_message_id)
        return email_record.nylas_message_id
    if pending_action.action_type == ActionType.DRAFT_REPLY:
        return str(create_draft(grant_id, **payload).get("id"))
    if pending_action.action_type == ActionType.CREATE_EVENT:
        return str(create_calendar_event(grant_id, **payload).get("id"))
    raise ValueError(f"Unhandled action type: {pending_action.action_type}")


def _execute_dry_run(pending_action: PendingAction) -> str:
    """Stand-in for `_execute()` under DRY_RUN: never touches Nylas.
    Logs only the action type and id — never payload or email content."""
    fake_id = f"dry-run-{uuid4()}"
    logger.info(
        "DRY_RUN: skipped Nylas %s for pending_action_id=%s (fake id %s)",
        pending_action.action_type.value,
        pending_action.id,
        fake_id,
    )
    return fake_id


def approve(pending_action_id: int, actor: str = "human") -> PendingAction:
    """Approve a pending action: marks it `approved`, executes the Nylas
    action (or, under DRY_RUN, a logged no-op), then marks it `executed`.

    Only an action whose status is `pending` can be approved. Approving
    an action that's already `executed`, `rejected`, `approved`, or
    `failed` raises `PendingActionError` instead of re-running it —
    this is what makes repeat approvals idempotent (never double-send,
    double-archive, or double-book a calendar event).
    """
    # `execution_error`, not a bare `raise` inside the `with` block:
    # raising there would make `session_scope()`'s own error handling
    # roll back the whole transaction — including the FAILED status and
    # the audit row we're trying to persist. Instead we let the `with`
    # block exit normally (so it commits), and only raise afterward.
    execution_error: Exception | None = None
    dry_run = get_settings().dry_run

    with session_scope() as session:
        pending_action = session.get_one(PendingAction, pending_action_id)
        if pending_action.status != PendingActionStatus.PENDING:
            raise PendingActionError(
                f"Pending action {pending_action_id} is "
                f"'{pending_action.status.value}', not 'pending' — cannot approve."
            )

        update_pending_action_status(
            session, pending_action_id, PendingActionStatus.APPROVED
        )
        write_audit_log(
            session,
            actor=actor,
            action="approved",
            pending_action_id=pending_action_id,
            dry_run=dry_run,
        )

        try:
            if dry_run:
                external_id = _execute_dry_run(pending_action)
            else:
                external_id = _execute(pending_action, pending_action.email)
        except Exception as exc:  # noqa: BLE001 - re-raised below, after commit
            execution_error = exc
            update_pending_action_status(
                session, pending_action_id, PendingActionStatus.FAILED
            )
            write_audit_log(
                session,
                actor=actor,
                action="approve_failed",
                pending_action_id=pending_action_id,
                dry_run=dry_run,
            )
        else:
            logger.info(
                "Executed pending_action_id=%s: external id %s (dry_run=%s)",
                pending_action_id,
                external_id,
                dry_run,
            )
            update_pending_action_status(
                session, pending_action_id, PendingActionStatus.EXECUTED
            )
            write_audit_log(
                session,
                actor=actor,
                action="executed",
                pending_action_id=pending_action_id,
                dry_run=dry_run,
            )

    if execution_error is not None:
        raise execution_error
    return pending_action


def reject(pending_action_id: int, actor: str = "human") -> PendingAction:
    """Reject a pending action. Only a `pending` action can be rejected —
    rejecting an already-decided action raises `PendingActionError`."""
    with session_scope() as session:
        pending_action = session.get_one(PendingAction, pending_action_id)
        if pending_action.status != PendingActionStatus.PENDING:
            raise PendingActionError(
                f"Pending action {pending_action_id} is "
                f"'{pending_action.status.value}', not 'pending' — cannot reject."
            )

        update_pending_action_status(
            session, pending_action_id, PendingActionStatus.REJECTED
        )
        write_audit_log(
            session,
            actor=actor,
            action="rejected",
            pending_action_id=pending_action_id,
        )
    return pending_action
