"""Run the triage flow for one raw event, never raising.

Used as the FastAPI background task after `POST /events` has already
returned 200, and by `scripts/reprocess_raw_events.py`. Failures are
logged with the event id and exception *type* only: exception messages
(e.g. a pydantic `ValidationError`'s) and tracebacks can carry email
content, so they are deliberately left out.
"""

import logging

from app.config import get_settings
from app.database.session import session_scope
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
    return outcome
