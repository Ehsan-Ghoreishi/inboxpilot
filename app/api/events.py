"""Nylas webhook endpoint: verification handshake + event delivery.

Per Nylas's docs: a `GET` request carrying a `challenge` query parameter
is the one-time (or on-URL-change) verification handshake, which must be
echoed back verbatim as 200 OK. A `POST` is an actual event notification,
signed with `X-Nylas-Signature` over the exact raw body.
"""

import json

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, PlainTextResponse

from app.integrations.nylas.webhooks import store_raw_event, verify_webhook_signature

router = APIRouter()


@router.get("/events")
def verify_webhook(challenge: str) -> PlainTextResponse:
    """Respond to Nylas's webhook verification handshake."""
    return PlainTextResponse(challenge)


@router.post("/events")
async def receive_event(request: Request) -> JSONResponse:
    """Receive a Nylas webhook notification.

    Verifies the signature against the raw body *before* parsing JSON,
    stores the raw event idempotently by its id, and returns fast — no
    downstream processing happens here yet.
    """
    raw_body = await request.body()
    signature = request.headers.get("x-nylas-signature")

    if not verify_webhook_signature(raw_body, signature):
        return JSONResponse(status_code=401, content={"detail": "Invalid signature"})

    try:
        event = json.loads(raw_body)
    except json.JSONDecodeError:
        return JSONResponse(status_code=400, content={"detail": "Invalid JSON body"})

    try:
        store_raw_event(event)
    except ValueError:
        return JSONResponse(
            status_code=400, content={"detail": "Event payload missing 'id'"}
        )

    return JSONResponse(status_code=200, content={"status": "ok"})
