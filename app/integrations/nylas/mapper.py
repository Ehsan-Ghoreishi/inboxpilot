"""Mapping raw Nylas message data to the canonical `Email` schema."""

from datetime import datetime, timezone

from app.schemas.email import Email, EmailParticipant


def nylas_payload_to_email(payload: dict) -> Email:
    """Map a raw Nylas message dict to the canonical `Email`.

    `payload` is a Nylas Message-shaped dict — either a webhook
    notification's `data["object"]`, or `message.to_dict()` from an SDK
    `Message` (both use the same field names: id, grant_id, thread_id,
    subject, from, to, date, body).

    Note: Nylas's `body` is HTML; this maps it through as-is rather than
    converting to plain text, which is out of scope here.
    """
    sender_data = (payload.get("from") or [{}])[0]
    return Email(
        id=payload["id"],
        thread_id=payload["thread_id"],
        grant_id=payload["grant_id"],
        sender=EmailParticipant(
            name=sender_data.get("name"), email=sender_data["email"]
        ),
        recipients=[
            EmailParticipant(name=r.get("name"), email=r["email"])
            for r in payload.get("to", [])
        ],
        subject=payload.get("subject") or "",
        body=payload.get("body") or "",
        received_at=datetime.fromtimestamp(payload["date"], tz=timezone.utc),
    )
