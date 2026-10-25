"""add email_embeddings (pgvector)

Revision ID: d9f3a6b2e8c1
Revises: c4a8e2f71d3b
Create Date: 2026-10-25 08:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = 'd9f3a6b2e8c1'
down_revision: Union[str, Sequence[str], None] = 'c4a8e2f71d3b'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "email_embeddings",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column(
            "email_id",
            sa.Integer(),
            sa.ForeignKey("emails.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model", sa.Text(), nullable=False),
        sa.Column("dim", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(1536), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "email_id", "model", name="uq_email_embeddings_email_id_model"
        ),
    )
    op.create_index(
        "ix_email_embeddings_embedding_hnsw",
        "email_embeddings",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index(
        "ix_email_embeddings_embedding_hnsw", table_name="email_embeddings"
    )
    op.drop_table("email_embeddings")
    # No CASCADE: if anything else has started depending on the
    # extension, fail loudly rather than silently dropping it.
    op.execute("DROP EXTENSION IF EXISTS vector")
