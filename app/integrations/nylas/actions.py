"""Consequential Nylas actions: archive, draft, calendar event.

Confirmed against the official `nylas/nylas-python` SDK source (not
guessed): `messages.update(grant_id, message_id, request_body)`,
`drafts.create(grant_id, request_body)`, and
`events.create(grant_id, request_body, query_params)`.

Nothing here ever sends a message or permanently deletes anything —
`create_draft` only calls `drafts.create` (never `drafts.send`), and
"archive" is a folder move, not a delete.
"""

from typing import Any

from app.integrations.nylas.client import get_client

# Nylas represents "archive" as a folder change rather than a dedicated
# verb. "ARCHIVE" is Nylas's special folder name for Gmail accounts;
# other providers (Outlook/EWS/IMAP) may need a different folder
# identifier. This is a reasonable default, not verified against every
# provider — flagging as a known limitation rather than asserting it's
# universally correct.
ARCHIVE_FOLDER = "ARCHIVE"


def archive_message(grant_id: str, message_id: str) -> None:
    """Archive a message by moving it out of the inbox. Never deletes it."""
    get_client().messages.update(
        grant_id, message_id, request_body={"folders": [ARCHIVE_FOLDER]}
    )


def create_draft(
    grant_id: str,
    to: list[dict],
    subject: str,
    body: str,
    reply_to_message_id: str | None = None,
) -> dict[str, Any]:
    """Create a draft reply. Never sends it — sending is a separate,
    deliberate action (`drafts.send`) that this function never calls."""
    request_body: dict[str, Any] = {"to": to, "subject": subject, "body": body}
    if reply_to_message_id:
        request_body["reply_to_message_id"] = reply_to_message_id
    response = get_client().drafts.create(grant_id, request_body=request_body)
    return response.data.to_dict()


def create_calendar_event(
    grant_id: str,
    title: str,
    when: dict[str, Any],
    description: str = "",
    calendar_id: str = "primary",
) -> dict[str, Any]:
    """Create a calendar event (e.g. to block time for an interview)."""
    request_body: dict[str, Any] = {
        "title": title,
        "when": when,
        "description": description,
    }
    response = get_client().events.create(
        grant_id,
        request_body=request_body,
        query_params={"calendar_id": calendar_id},
    )
    return response.data.to_dict()
