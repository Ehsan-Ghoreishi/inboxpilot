"""SQLAlchemy 2.0 ORM models for InboxPilot's persistence layer.

Table-to-class naming note: the ORM class for the `emails` table is
`EmailRecord`, not `Email` — `Email` is already the canonical Pydantic
schema (`app/schemas/email.py`), and code that needs both (e.g. the
repository layer) would collide on the name otherwise.
"""

import enum
from datetime import datetime

from sqlalchemy import JSON, DateTime, Enum as SAEnum, ForeignKey, false, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class RawEvent(Base):
    """A raw Nylas webhook event, exactly as received (post signature-check)."""

    __tablename__ = "raw_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    event_id: Mapped[str] = mapped_column(unique=True, index=True)
    payload: Mapped[dict] = mapped_column(JSON)
    received_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class EmailRecord(Base):
    """Canonical email data, persisted (see module docstring for naming)."""

    __tablename__ = "emails"

    id: Mapped[int] = mapped_column(primary_key=True)
    nylas_message_id: Mapped[str] = mapped_column(unique=True, index=True)
    thread_id: Mapped[str]
    grant_id: Mapped[str]
    sender_name: Mapped[str | None]
    sender_email: Mapped[str]
    recipients: Mapped[list] = mapped_column(JSON)
    subject: Mapped[str]
    body: Mapped[str]
    received_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    classifications: Mapped[list["Classification"]] = relationship(
        back_populates="email"
    )
    pending_actions: Mapped[list["PendingAction"]] = relationship(
        back_populates="email"
    )


class Classification(Base):
    """One LLM classification/extraction result for an email."""

    __tablename__ = "classifications"

    id: Mapped[int] = mapped_column(primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"), index=True)
    category: Mapped[str]
    confidence: Mapped[float]
    model: Mapped[str]
    prompt_version: Mapped[str]
    input_tokens: Mapped[int | None]
    output_tokens: Mapped[int | None]
    latency_ms: Mapped[int | None]
    # Where deterministic routing sent this email (a
    # `app.workflows.triage_workflow.TriageStatus` value, e.g.
    # "needs_review"). Nullable: rows written before routing was
    # recorded have none.
    triage_status: Mapped[str | None]
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    email: Mapped["EmailRecord"] = relationship(back_populates="classifications")


class ActionType(str, enum.Enum):
    DRAFT_REPLY = "draft_reply"
    CREATE_EVENT = "create_event"
    ARCHIVE = "archive"


class PendingActionStatus(str, enum.Enum):
    PENDING = "pending"
    APPROVED = "approved"
    EXECUTED = "executed"
    REJECTED = "rejected"
    FAILED = "failed"


class PendingAction(Base):
    """A proposed action awaiting (or past) human approval."""

    __tablename__ = "pending_actions"

    id: Mapped[int] = mapped_column(primary_key=True)
    email_id: Mapped[int] = mapped_column(ForeignKey("emails.id"), index=True)
    # native_enum=False: stored as VARCHAR + CHECK, not a Postgres native
    # ENUM type, so adding a new status/action later doesn't need an
    # `ALTER TYPE ... ADD VALUE` migration. values_callable is required:
    # without it, SQLAlchemy persists the Python Enum *member name*
    # (e.g. "DRAFT_REPLY"), not its `.value` ("draft_reply") — which
    # would silently mismatch the lowercase values the migration's
    # CHECK constraint actually allows.
    action_type: Mapped[ActionType] = mapped_column(
        SAEnum(
            ActionType,
            native_enum=False,
            validate_strings=True,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        )
    )
    payload: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[PendingActionStatus] = mapped_column(
        SAEnum(
            PendingActionStatus,
            native_enum=False,
            validate_strings=True,
            values_callable=lambda enum_cls: [member.value for member in enum_cls],
        ),
        default=PendingActionStatus.PENDING,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    decided_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    email: Mapped["EmailRecord"] = relationship(back_populates="pending_actions")


class AuditLog(Base):
    """An immutable record of who/what did something to a pending action."""

    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(primary_key=True)
    actor: Mapped[str]
    action: Mapped[str]
    pending_action_id: Mapped[int | None] = mapped_column(
        ForeignKey("pending_actions.id"), index=True, nullable=True
    )
    at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    # True when the audited step ran under DRY_RUN (no Nylas call made).
    dry_run: Mapped[bool] = mapped_column(default=False, server_default=false())
