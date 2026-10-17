"""add classifications.triage_status

Revision ID: b7d2c4e1f9a0
Revises: 3e6ec545a393
Create Date: 2026-10-17 15:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b7d2c4e1f9a0'
down_revision: Union[str, Sequence[str], None] = '3e6ec545a393'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "classifications", sa.Column("triage_status", sa.String(), nullable=True)
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("classifications", "triage_status")
