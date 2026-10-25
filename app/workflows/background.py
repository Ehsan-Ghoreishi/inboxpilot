"""Run the triage flow for one raw event, never raising.

Used as the FastAPI background task after `POST /events` has already
returned 200, and by `scripts/reprocess_raw_events.py`. Failures are
logged with the event id and exception *type* only: exception messages
(e.g. a pydantic `ValidationError`'s) and tracebacks can carry email
content, so they are deliberately left out.

Once triage has committed, the saved email is embedded in a separate
transaction. An embedding failure is logged the same way and never
affects triage: the email is simply left for `scripts/embed_emails.py`.
"""

import logging

from app.config import get_settings
from app.database.repository import get_email_by_message_id
from app.database.session import session_scope
from app.integrations.nylas.mapper import nylas_payload_to_email
from app.services.embedding import embed_email, get_embedder
from app.workflows.triage_workflow import TriageOutcome, process_raw_event

logger = logging.getLogger(__name__)


def run_triage_safely(event_id: str, payload: dict) -> TriageOutcome | None:
    """Process one raw event in its own transaction. Returns `None` on error."""
    try:
        settings = get_settings()
        with session_scope() as session:
            outcome = process_raw_event(session, payload, settings)
    except Exception as exc:  # noqa: BLE001 - must never propagate
        logger.error(
            "Triage failed for event_id=%s: %s", event_id, type(exc).__name__
        )
        return None
    logger.info("Triage finished for event_id=%s: %s", event_id, outcome.value)
    if outcome is not TriageOutcome.SKIPPED_TYPE:
        embed_event_email_safely(event_id, payload)
    return outcome


def embed_event_email_safely(event_id: str, payload: dict) -> None:
    """Embed the already-saved email of this event in its own transaction."""
    try:
        message_id = nylas_payload_to_email(payload).id
        with session_scope() as session:
            email = get_email_by_message_id(session, message_id)
            if email is not None:
                embed_email(session, email, get_embedder())
    except Exception as exc:  # noqa: BLE001 - must never propagate
        logger.error(
            "Embedding failed for event_id=%s: %s", event_id, type(exc).__name__
        )
