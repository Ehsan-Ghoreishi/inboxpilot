"""Repository functions: plain functions over an explicit `Session`.

Each function takes the `Session` it should use rather than opening its
own — callers compose multiple calls under one transaction (typically
via `app.database.session.session_scope()`), and tests can pass in a
session bound to a test database/transaction directly.
"""

from datetime import datetime

from sqlalchemy import exists, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session, selectinload

from app.database.models import (
    ActionType,
    AuditLog,
    Classification,
    EmailEmbedding,
    EmailRecord,
    PendingAction,
    PendingActionStatus,
    RawEvent,
)
from app.schemas.email import Email, EmailAnalysis


def save_raw_event(session: Session, event_id: str, payload: dict) -> RawEvent:
    """Idempotently store a raw webhook event, keyed by its event id.

    Webhook deliveries are "at least once" (Nylas may retry/duplicate),
    so a repeat `event_id` is a no-op, not an error.
    """
    stmt = (
        pg_insert(RawEvent)
        .values(event_id=event_id, payload=payload)
        .on_conflict_do_nothing(index_elements=["event_id"])
    )
    session.execute(stmt)
    session.flush()
    return session.scalars(
        select(RawEvent).where(RawEvent.event_id == event_id)
    ).one()


def _email_columns(email: Email) -> dict:
    return {
        "nylas_message_id": email.id,
        "thread_id": email.thread_id,
        "grant_id": email.grant_id,
        "sender_name": email.sender.name,
        "sender_email": email.sender.email,
        "recipients": [{"name": r.name, "email": r.email} for r in email.recipients],
        "subject": email.subject,
        "body": email.body,
        "received_at": email.received_at,
    }


def get_or_create_email(session: Session, email: Email) -> EmailRecord:
    """Find the `EmailRecord` for this canonical `Email`, inserting if new."""
    existing = session.scalars(
        select(EmailRecord).where(EmailRecord.nylas_message_id == email.id)
    ).one_or_none()
    if existing is not None:
        return existing

    record = EmailRecord(**_email_columns(email))
    session.add(record)
    session.flush()
    return record


def insert_email_if_new(session: Session, email: Email) -> EmailRecord | None:
    """Insert this `Email`, or return `None` if its message id is already stored.

    A concurrent insert of the same message id blocks on the unique
    index until the other transaction finishes, so exactly one caller
    gets the record back — this is what makes processing once-per-message.
    """
    stmt = (
        pg_insert(EmailRecord)
        .values(**_email_columns(email))
        .on_conflict_do_nothing(index_elements=["nylas_message_id"])
        .returning(EmailRecord.id)
    )
    record_id = session.execute(stmt).scalar_one_or_none()
    if record_id is None:
        return None
    return session.get_one(EmailRecord, record_id)


def save_classification(
    session: Session,
    email_record_id: int,
    analysis: EmailAnalysis,
    *,
    model: str,
    prompt_version: str,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    latency_ms: int | None = None,
    triage_status: str | None = None,
) -> Classification:
    classification = Classification(
        email_id=email_record_id,
        category=analysis.category.value,
        confidence=analysis.confidence,
        model=model,
        prompt_version=prompt_version,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        triage_status=triage_status,
    )
    session.add(classification)
    session.flush()
    return classification


def create_pending_action(
    session: Session,
    email_record_id: int,
    action_type: ActionType,
    payload: dict,
) -> PendingAction:
    pending_action = PendingAction(
        email_id=email_record_id,
        action_type=action_type,
        payload=payload,
        status=PendingActionStatus.PENDING,
    )
    session.add(pending_action)
    session.flush()
    return pending_action


def update_pending_action_status(
    session: Session,
    pending_action_id: int,
    status: PendingActionStatus,
    decided_at: datetime | None = None,
) -> PendingAction:
    pending_action = session.get_one(PendingAction, pending_action_id)
    pending_action.status = status
    pending_action.decided_at = decided_at or datetime.now()
    session.flush()
    return pending_action


def write_audit_log(
    session: Session,
    actor: str,
    action: str,
    pending_action_id: int | None = None,
    dry_run: bool = False,
) -> AuditLog:
    entry = AuditLog(
        actor=actor,
        action=action,
        pending_action_id=pending_action_id,
        dry_run=dry_run,
    )
    session.add(entry)
    session.flush()
    return entry


def list_pending_actions(
    session: Session, status: PendingActionStatus
) -> list[PendingAction]:
    return list(
        session.scalars(
            select(PendingAction)
            .where(PendingAction.status == status)
            .order_by(PendingAction.created_at.desc(), PendingAction.id.desc())
        )
    )


def list_recent_emails(session: Session, limit: int = 50) -> list[EmailRecord]:
    """Most recent emails, with classifications and pending actions eagerly
    loaded so they're usable after the session closes."""
    return list(
        session.scalars(
            select(EmailRecord)
            .options(
                selectinload(EmailRecord.classifications),
                selectinload(EmailRecord.pending_actions),
            )
            .order_by(EmailRecord.received_at.desc(), EmailRecord.id.desc())
            .limit(limit)
        )
    )


def list_unprocessed_raw_events(session: Session) -> list[RawEvent]:
    """Raw events whose message (`payload.data.object.id`) has no `emails` row.

    Events without a message id (e.g. non-message triggers) are included
    too; it's up to the caller to skip what it can't process.
    """
    message_id = RawEvent.payload["data"]["object"]["id"].as_string()
    return list(
        session.scalars(
            select(RawEvent)
            .where(~exists().where(EmailRecord.nylas_message_id == message_id))
            .order_by(RawEvent.id)
        )
    )


def get_email_by_message_id(session: Session, message_id: str) -> EmailRecord | None:
    return session.scalars(
        select(EmailRecord).where(EmailRecord.nylas_message_id == message_id)
    ).one_or_none()


def get_embedding_content_hash(
    session: Session, email_id: int, model: str
) -> str | None:
    return session.scalars(
        select(EmailEmbedding.content_hash).where(
            EmailEmbedding.email_id == email_id, EmailEmbedding.model == model
        )
    ).one_or_none()


def upsert_embedding(
    session: Session,
    *,
    email_id: int,
    model: str,
    content_hash: str,
    embedding: list[float],
) -> None:
    """Insert or replace the embedding for (email_id, model).

    An existing row is only overwritten when its `content_hash` differs,
    so re-upserting unchanged content is a no-op.
    """
    stmt = pg_insert(EmailEmbedding).values(
        email_id=email_id,
        model=model,
        dim=len(embedding),
        content_hash=content_hash,
        embedding=embedding,
    )
    stmt = stmt.on_conflict_do_update(
        constraint="uq_email_embeddings_email_id_model",
        set_={
            "dim": stmt.excluded.dim,
            "content_hash": stmt.excluded.content_hash,
            "embedding": stmt.excluded.embedding,
            "created_at": func.now(),
        },
        where=EmailEmbedding.content_hash != stmt.excluded.content_hash,
    )
    session.execute(stmt)
    session.flush()


def list_email_ids_without_embedding(session: Session, model: str) -> list[int]:
    has_embedding = exists().where(
        EmailEmbedding.email_id == EmailRecord.id, EmailEmbedding.model == model
    )
    return list(
        session.scalars(
            select(EmailRecord.id).where(~has_embedding).order_by(EmailRecord.id)
        )
    )


def _escape_like(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def search_email_embeddings(
    session: Session,
    *,
    query_embedding: list[float],
    model: str,
    top_k: int,
    min_score: float,
    category: str | None = None,
    sender: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> list[tuple[EmailRecord, float]]:
    """Nearest emails to `query_embedding` by cosine distance (`<=>`),
    restricted to embeddings of `model` and the given filters.

    Returns `(email, score)` pairs, best first, where score is
    `1 - distance` and at least `min_score`. `category` matches the
    email's latest classification. Emails whose embedded content is
    identical (same `content_hash`) collapse to the best-scoring one, so
    duplicates don't use up `top_k` slots.
    """
    distance = EmailEmbedding.embedding.cosine_distance(query_embedding)
    score = 1 - distance
    stmt = (
        select(
            EmailRecord.id.label("email_id"),
            distance.label("distance"),
            score.label("score"),
            func.row_number()
            .over(
                partition_by=EmailEmbedding.content_hash,
                order_by=(distance, EmailRecord.id),
            )
            .label("rank_in_hash"),
        )
        .join(EmailEmbedding, EmailEmbedding.email_id == EmailRecord.id)
        .where(EmailEmbedding.model == model, score >= min_score)
    )
    if category is not None:
        latest_category = (
            select(Classification.category)
            .where(Classification.email_id == EmailRecord.id)
            .order_by(Classification.created_at.desc(), Classification.id.desc())
            .limit(1)
            .scalar_subquery()
        )
        stmt = stmt.where(latest_category == category)
    if sender is not None:
        pattern = f"%{_escape_like(sender)}%"
        stmt = stmt.where(
            EmailRecord.sender_email.ilike(pattern, escape="\\")
            | EmailRecord.sender_name.ilike(pattern, escape="\\")
        )
    if date_from is not None:
        stmt = stmt.where(EmailRecord.received_at >= date_from)
    if date_to is not None:
        stmt = stmt.where(EmailRecord.received_at <= date_to)

    # Filters apply before collapsing, so a duplicate that matches them
    # isn't hidden by a better-scoring copy that doesn't. The window
    # function rules out an HNSW index-ordered scan: candidates are scored
    # exactly, which is fine at a single inbox's size.
    candidates = stmt.subquery()
    collapsed = (
        select(EmailRecord, candidates.c.score)
        .join(candidates, candidates.c.email_id == EmailRecord.id)
        .where(candidates.c.rank_in_hash == 1)
        .order_by(candidates.c.distance, EmailRecord.id)
        .limit(top_k)
    )
    return [(email, row_score) for email, row_score in session.execute(collapsed)]
