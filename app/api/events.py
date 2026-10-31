"""Nylas webhook endpoint: verification handshake + event delivery.

Per Nylas's docs: a `GET` request carrying a `challenge` query parameter
is the one-time (or on-URL-change) verification handshake, which must be
echoed back verbatim as 200 OK. A `POST` is an actual event notification,
signed with `X-Nylas-Signature` over the exact raw body.
"""

import asyncio
import functools
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor

from fastapi import APIRouter, BackgroundTasks, Request
from fastapi.responses import JSONResponse, PlainTextResponse
from starlette.concurrency import run_in_threadpool

from app.config import get_settings
from app.database.repository import save_raw_event
from app.database.session import session_scope
from app.integrations.nylas.webhooks import store_raw_event, verify_webhook_signature
from app.workflows.background import run_triage_safely

logger = logging.getLogger(__name__)

router = APIRouter()

# Overall budget for inline triage (PROCESS_EVENTS_INLINE), kept under
# the time Nylas waits for a webhook response.
INLINE_PROCESSING_TIMEOUT_SECONDS = 8.0

# Bounded so timed-out triage threads (which can't be killed) can't pile
# up without limit; kept below the DB pool size (5). Work queued past
# its deadline is skipped by `run_triage_safely`.
_INLINE_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="inline-triage")


@router.get("/events")
def verify_webhook(challenge: str) -> PlainTextResponse:
    """Respond to Nylas's webhook verification handshake."""
    return PlainTextResponse(challenge)


def _persist_to_db(event_id: str, event: dict) -> None:
    """Blocking DB write, run off the event loop (sync SQLAlchemy engine)."""
    with session_scope() as session:
        save_raw_event(session, event_id, event)


async def _process_inline(event_id: str, event: dict) -> str:
    """Run triage within the request, bounded by the inline timeout.

    Uses a dedicated executor future rather than `run_in_threadpool`, whose
    worker can't be cancelled: `wait_for` would otherwise block until
    triage finished, defeating the timeout. A timed-out worker thread
    keeps running, but its `deadline` makes it roll back instead of
    committing, so the raw event stays unprocessed for the reprocess job.
    Returns the outcome to log.
    """
    timeout = INLINE_PROCESSING_TIMEOUT_SECONDS
    deadline = time.monotonic() + timeout
    loop = asyncio.get_running_loop()
    work = loop.run_in_executor(
        _INLINE_EXECUTOR, functools.partial(run_triage_safely, event_id, event, deadline)
    )
    try:
        outcome = await asyncio.wait_for(work, timeout=timeout)
    except TimeoutError:
        return "timeout"
    return outcome.value if outcome is not None else "error"


@router.post("/events")
async def receive_event(
    request: Request, background_tasks: BackgroundTasks
) -> JSONResponse:
    """Receive a Nylas webhook notification.

    Verifies the signature against the raw body *before* parsing JSON,
    stores the raw event in the database (idempotently, by its id) —
    the required, authoritative persistence — and optionally also dumps
    it to `requests/events/<id>.json` (controlled by
    `settings.store_raw_event_files`; failures there are logged, not
    fatal). Triage (mapping, classification, pending action) is
    scheduled as a background task that runs after the 200 is sent, so
    its failures never affect the response (see
    `app.workflows.background.run_triage_safely`). With
    `settings.process_events_inline`, triage instead runs before the
    response, bounded by `INLINE_PROCESSING_TIMEOUT_SECONDS`; a timeout
    or failure still returns 200 and leaves the event for
    `scripts/reprocess_raw_events.py`.
    """
    raw_body = await request.body()
    signature = request.headers.get("x-nylas-signature")

    if not verify_webhook_signature(raw_body, signature):
        logger.warning(
            "Rejected webhook: invalid signature",
            extra={"outcome": "invalid_signature"},
        )
        return JSONResponse(status_code=401, content={"detail": "Invalid signature"})

    try:
        event = json.loads(raw_body)
    except json.JSONDecodeError:
        logger.warning(
            "Rejected webhook: invalid JSON", extra={"outcome": "invalid_json"}
        )
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON body"})

    event_id = event.get("id") if isinstance(event, dict) else None
    if not event_id:
        logger.warning(
            "Rejected webhook: missing id", extra={"outcome": "missing_id"}
        )
        return JSONResponse(
            status_code=400, content={"detail": "Event payload missing 'id'"}
        )

    await run_in_threadpool(_persist_to_db, event_id, event)

    settings = get_settings()
    if settings.store_raw_event_files:
        try:
            store_raw_event(event)
        except (ValueError, OSError) as exc:
            logger.warning(
                "Skipped optional raw-event file write for %s: %s",
                event_id,
                type(exc).__name__,
                extra={"event_id": event_id},
            )

    if settings.process_events_inline:
        outcome = await _process_inline(event_id, event)
    else:
        background_tasks.add_task(run_triage_safely, event_id, event)
        outcome = "scheduled"
    logger.info(
        "Webhook event %s accepted: %s",
        event_id,
        outcome,
        extra={"event_id": event_id, "outcome": outcome},
    )
    return JSONResponse(status_code=200, content={"status": "ok"})
