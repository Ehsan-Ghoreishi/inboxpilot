"""at most one pending action per (thread, action_type)

Revision ID: e5b1c9d47a26
Revises: d9f3a6b2e8c1
Create Date: 2026-10-25 20:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e5b1c9d47a26'
down_revision: Union[str, Sequence[str], None] = 'd9f3a6b2e8c1'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

INDEX_NAME = "uq_pending_actions_thread_id_action_type_pending"


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("pending_actions", sa.Column("thread_id", sa.String()))
    op.execute(
        "UPDATE pending_actions AS p SET thread_id = e.thread_id "
        "FROM emails AS e WHERE e.id = p.email_id"
    )
    op.alter_column("pending_actions", "thread_id", nullable=False)

    # Deciding which existing duplicate to keep is a human call (each
    # may already be under review), so refuse rather than pick one.
    duplicates = op.get_bind().execute(
        sa.text(
            "SELECT thread_id, action_type, array_agg(id ORDER BY id) AS ids "
            "FROM pending_actions WHERE status = 'pending' "
            "GROUP BY thread_id, action_type HAVING count(*) > 1"
        )
    ).all()
    if duplicates:
        details = "; ".join(
            f"thread {row.thread_id} {row.action_type}: ids {row.ids}"
            for row in duplicates
        )
        raise RuntimeError(
            "Duplicate pending actions must be approved or rejected before "
            f"this migration can run: {details}"
        )

    op.create_index(
        INDEX_NAME,
        "pending_actions",
        ["thread_id", "action_type"],
        unique=True,
        postgresql_where=sa.text("status = 'pending'"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(INDEX_NAME, table_name="pending_actions")
    op.drop_column("pending_actions", "thread_id")
