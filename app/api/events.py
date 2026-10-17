"""Nylas webhook endpoint: verification handshake + event delivery.

Per Nylas's docs: a `GET` request carrying a `challenge` query parameter
is the one-time (or on-URL-change) verification handshake, which must be
echoed back verbatim as 200 OK. A `POST` is an actual event notification,
signed with `X-Nylas-Signature` over the exact raw body.
"""

import json
import logging

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


@router.get("/events")
def verify_webhook(challenge: str) -> PlainTextResponse:
    """Respond to Nylas's webhook verification handshake."""
    return PlainTextResponse(challenge)


def _persist_to_db(event_id: str, event: dict) -> None:
    """Blocking DB write, run off the event loop (sync SQLAlchemy engine)."""
    with session_scope() as session:
        save_raw_event(session, event_id, event)


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
    `app.workflows.background.run_triage_safely`).
    """
    raw_body = await request.body()
    signature = request.headers.get("x-nylas-signature")

    if not verify_webhook_signature(raw_body, signature):
        return JSONResponse(status_code=401, content={"detail": "Invalid signature"})

    try:
        event = json.loads(raw_body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON body"})

    event_id = event.get("id") if isinstance(event, dict) else None
    if not event_id:
        return JSONResponse(
            status_code=400, content={"detail": "Event payload missing 'id'"}
        )

    await run_in_threadpool(_persist_to_db, event_id, event)

    if get_settings().store_raw_event_files:
        try:
            store_raw_event(event)
        except ValueError:
            logger.warning(
                "Skipped optional raw-event file write for %s: missing id",
                event_id,
            )

    background_tasks.add_task(run_triage_safely, event_id, event)
    return JSONResponse(status_code=200, content={"status": "ok"})
