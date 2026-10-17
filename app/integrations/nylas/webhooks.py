"""Webhook signature verification and raw event storage.

Signature scheme (confirmed from Nylas's own docs, not guessed): every
notification includes an `x-nylas-signature` header (`X-Nylas-Signature`
in some SDKs/frameworks) containing a hex-encoded HMAC-SHA256 of the
*exact raw request body*, keyed with the webhook destination's
`webhook_secret`. It must be checked before any JSON parsing and compared
in constant time. (If `compressed_delivery` is ever enabled, the
signature covers the raw gzip-compressed bytes, before decompression —
not relevant yet since there's no endpoint wired up.)
"""

import hashlib
import hmac
import json
from pathlib import Path

from app.config import get_settings

REQUESTS_EVENTS_DIR = Path("requests/events")


def compute_signature(raw_body: bytes, secret: str) -> str:
    """Hex-encoded HMAC-SHA256 of the raw body, keyed with the webhook secret."""
    return hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()


def verify_webhook_signature(raw_body: bytes, signature_header: str | None) -> bool:
    """Verify an inbound webhook's signature against the configured secret.

    `raw_body` must be the exact, unmodified request body bytes.
    """
    if not signature_header:
        return False
    expected = compute_signature(raw_body, get_settings().webhook_secret)
    return hmac.compare_digest(expected, signature_header)


def store_raw_event(event: dict, directory: Path = REQUESTS_EVENTS_DIR) -> Path:
    """Persist a raw webhook event payload as JSON, named by its event id.

    `requests/` is git-ignored: these files can contain real email
    content, so this is local/audit storage only, never committed.
    """
    event_id = event.get("id")
    if not event_id:
        raise ValueError("Event payload is missing its top-level 'id' field.")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{event_id}.json"
    path.write_text(json.dumps(event, indent=2), encoding="utf-8")
    return path
