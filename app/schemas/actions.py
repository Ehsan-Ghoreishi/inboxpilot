"""API response schemas for pending actions and the dashboard view."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

from app.database.models import ActionType, PendingActionStatus


class PendingActionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email_id: int
    action_type: ActionType
    status: PendingActionStatus
    payload: dict[str, Any]
    created_at: datetime
    decided_at: datetime | None


class PendingActionDetail(PendingActionOut):
    """A pending action plus the email it concerns, for reviewing a queue."""

    message_id: str
    category: str | None
    confidence: float | None


class PendingActionList(BaseModel):
    actions: list[PendingActionDetail]


class AuditLogEntryOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    actor: str
    action: str
    dry_run: bool
    at: datetime


class AuditLogList(BaseModel):
    entries: list[AuditLogEntryOut]


class ClassificationOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    category: str
    confidence: float
    created_at: datetime


class EmailSummary(BaseModel):
    """One row of the dashboard: an email, its latest classification, and
    its still-pending actions."""

    id: int
    sender_name: str | None
    sender_email: str
    subject: str
    received_at: datetime
    classification: ClassificationOut | None
    pending_actions: list[PendingActionOut]
