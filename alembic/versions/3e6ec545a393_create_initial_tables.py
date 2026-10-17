"""create initial tables

Revision ID: 3e6ec545a393
Revises:
Create Date: 2026-10-17 12:02:56.847093

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3e6ec545a393'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "raw_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("event_id", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column(
            "received_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_raw_events_event_id", "raw_events", ["event_id"], unique=True
    )

    op.create_table(
        "emails",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("nylas_message_id", sa.String(), nullable=False),
        sa.Column("thread_id", sa.String(), nullable=False),
        sa.Column("grant_id", sa.String(), nullable=False),
        sa.Column("sender_name", sa.String(), nullable=True),
        sa.Column("sender_email", sa.String(), nullable=False),
        sa.Column("recipients", sa.JSON(), nullable=False),
        sa.Column("subject", sa.String(), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_emails_nylas_message_id", "emails", ["nylas_message_id"], unique=True
    )

    op.create_table(
        "classifications",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "email_id",
            sa.Integer(),
            sa.ForeignKey("emails.id"),
            nullable=False,
        ),
        sa.Column("category", sa.String(), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("prompt_version", sa.String(), nullable=False),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("latency_ms", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_classifications_email_id", "classifications", ["email_id"]
    )

    op.create_table(
        "pending_actions",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "email_id",
            sa.Integer(),
            sa.ForeignKey("emails.id"),
            nullable=False,
        ),
        # Plain sa.String() + an explicit CheckConstraint below, rather
        # than sa.Enum(..., native_enum=False) on its own: a bare Enum
        # type used directly in op.create_table() does not reliably
        # generate its CHECK constraint (confirmed: it silently didn't
        # here) — an explicit CheckConstraint is unambiguous.
        sa.Column("action_type", sa.String(length=20), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "action_type IN ('draft_reply', 'create_event', 'archive')",
            name="ck_pending_actions_action_type",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'approved', 'executed', 'rejected', 'failed')",
            name="ck_pending_actions_status",
        ),
    )
    op.create_index(
        "ix_pending_actions_email_id", "pending_actions", ["email_id"]
    )

    op.create_table(
        "audit_log",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("actor", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column(
            "pending_action_id",
            sa.Integer(),
            sa.ForeignKey("pending_actions.id"),
            nullable=True,
        ),
        sa.Column(
            "at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_audit_log_pending_action_id", "audit_log", ["pending_action_id"]
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("audit_log")
    op.drop_table("pending_actions")
    op.drop_table("classifications")
    op.drop_table("emails")
    op.drop_table("raw_events")
