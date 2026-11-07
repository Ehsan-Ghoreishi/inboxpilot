"""Run the triage flow for one raw event, never raising.

Used as the FastAPI background task after `POST /events` has already
returned 200, and by `scripts/reprocess_raw_events.py`. Failures are
logged with the event id and exception *type* only: exception messages
(e.g. a pydantic `ValidationError`'s) and tracebacks can carry email
content, so they are deliberately left out.

Once triage has committed, the saved email is embedded in a separate
transaction. An embedding failure is logged the same way and never
affects triage: the email is simply left for `scripts/embed_emails.py`.

An optional `deadline` (a `time.monotonic()` value) is checked before
triage starts and just before it commits: past it, nothing runs or the
transaction is rolled back, so the raw event stays unprocessed for `scripts/reprocess_raw_events.py`, and
embedding is skipped.
"""

import logging
import time

from app.config import get_settings
from app.database.repository import get_email_by_message_id
from app.database.session import session_scope
from app.integrations.langfuse import annotate, run_flags, traced
from app.integrations.nylas.mapper import nylas_payload_to_email
from app.services.embedding import embed_email, get_embedder
from app.workflows.triage_workflow import (
    TriageOutcome,
    is_message_event,
    process_raw_event,
)

logger = logging.getLogger(__name__)


class DeadlineExceeded(Exception):
    """Triage finished after its deadline; its transaction is rolled back."""


def _past(deadline: float | None) -> bool:
    return deadline is not None and time.monotonic() > deadline


def run_triage_safely(
    event_id: str, payload: dict, deadline: float | None = None
) -> TriageOutcome | None:
    """Process one raw event in its own transaction. Returns `None` on error.

    Each `message.*` event is recorded as one `email` trace; other event
    types are skipped without one.
    """
    process = _process_email_event if is_message_event(payload) else _process_event
    return process(event_id, payload, deadline)


@traced("email", root=True)
def _process_email_event(
    event_id: str, payload: dict, deadline: float | None
) -> TriageOutcome | None:
    outcome = _process_event(event_id, payload, deadline)
    annotate(event_id=event_id, outcome=outcome.value if outcome else "error")
    return outcome


def _process_event(
    event_id: str, payload: dict, deadline: float | None
) -> TriageOutcome | None:
    try:
        if _past(deadline):
            raise DeadlineExceeded
        settings = get_settings()
        with session_scope() as session:
            outcome = process_raw_event(session, payload, settings)
            if _past(deadline):
                raise DeadlineExceeded
    except Exception as exc:  # noqa: BLE001 - must never propagate
        logger.error(
            "Triage failed for event_id=%s: %s",
            event_id,
            type(exc).__name__,
            extra={"event_id": event_id, "outcome": "error"},
        )
        return None
    logger.info(
        "Triage finished for event_id=%s: %s",
        event_id,
        outcome.value,
        extra={"event_id": event_id, "outcome": outcome.value},
    )
    if outcome is not TriageOutcome.SKIPPED_TYPE and not _past(deadline):
        embed_event_email_safely(event_id, payload)
    return outcome


@traced("embed")
def embed_event_email_safely(event_id: str, payload: dict) -> None:
    """Embed the already-saved email of this event in its own transaction."""
    try:
        message_id = nylas_payload_to_email(payload).id
        embedder = get_embedder()
        embedded = False
        with session_scope() as session:
            email = get_email_by_message_id(session, message_id)
            if email is not None:
                embedded = embed_email(session, email, embedder)
        annotate(embedded=embedded, model=embedder.model, **run_flags())
    except Exception as exc:  # noqa: BLE001 - must never propagate
        annotate(outcome="embedding_error", error_type=type(exc).__name__)
        logger.error(
            "Embedding failed for event_id=%s: %s",
            event_id,
            type(exc).__name__,
            extra={"event_id": event_id, "outcome": "embedding_error"},
        )
