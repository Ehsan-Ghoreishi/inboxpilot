"""add audit_log.dry_run

Revision ID: c4a8e2f71d3b
Revises: b7d2c4e1f9a0
Create Date: 2026-10-17 16:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'c4a8e2f71d3b'
down_revision: Union[str, Sequence[str], None] = 'b7d2c4e1f9a0'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "audit_log",
        sa.Column(
            "dry_run", sa.Boolean(), nullable=False, server_default=sa.false()
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("audit_log", "dry_run")
