"""Minimal HTML dashboard: recent emails, classifications, pending actions.

The page never contains the API key. The user pastes it into the page,
and the Approve/Reject buttons send it as `X-API-Key` to `/actions/...`.
"""

from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from app.database.models import EmailRecord, PendingActionStatus
from app.database.repository import list_recent_emails
from app.database.session import session_scope
from app.schemas.actions import ClassificationOut, EmailSummary, PendingActionOut

router = APIRouter()

templates = Jinja2Templates(directory=Path(__file__).resolve().parent.parent / "templates")


def _summarize(record: EmailRecord) -> EmailSummary:
    latest = max(record.classifications, key=lambda c: c.created_at, default=None)
    return EmailSummary(
        id=record.id,
        sender_name=record.sender_name,
        sender_email=record.sender_email,
        subject=record.subject,
        received_at=record.received_at,
        classification=ClassificationOut.model_validate(latest) if latest else None,
        pending_actions=[
            PendingActionOut.model_validate(action)
            for action in record.pending_actions
            if action.status == PendingActionStatus.PENDING
        ],
    )


@router.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    with session_scope() as session:
        emails = [_summarize(record) for record in list_recent_emails(session)]
    return templates.TemplateResponse(request, "index.html", {"emails": emails})
