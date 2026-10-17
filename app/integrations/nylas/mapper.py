"""Mapping a raw Nylas webhook event to the canonical `Email` schema."""

from datetime import datetime, timezone

from app.schemas.email import Email, EmailParticipant
from app.schemas.nylas_webhook import WebhookEvent


def nylas_payload_to_email(payload: dict) -> Email:
    """Map a raw Nylas webhook event dict to the canonical `Email`.

    `payload` is a full webhook event envelope — the same shape
    `store_raw_event` writes to `requests/events/<id>.json` — validated
    through `WebhookEvent`/`EmailObject` (`app/schemas/nylas_webhook.py`,
    `app/schemas/nylas_email.py`) before mapping, so malformed events
    fail fast with a clear `pydantic.ValidationError` rather than a
    confusing `KeyError` deep in field access.

    Note: Nylas's `body` is HTML; this maps it through as-is rather than
    converting to plain text, which is out of scope here.
    """
    event = WebhookEvent.model_validate(payload)
    obj = event.data.object

    if not obj.from_:
        raise ValueError("Email object has no sender ('from') address.")
    sender = obj.from_[0]

    return Email(
        id=obj.id,
        thread_id=obj.thread_id,
        grant_id=obj.grant_id,
        sender=EmailParticipant(name=sender.name, email=sender.email),
        recipients=[
            EmailParticipant(name=recipient.name, email=recipient.email)
            for recipient in obj.to
        ],
        subject=obj.subject or "",
        body=obj.body or "",
        received_at=datetime.fromtimestamp(obj.date, tz=timezone.utc),
    )
