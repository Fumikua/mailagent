"""add samples_archive table

Revision ID: 0005_add_samples_archive_table
Revises: 0004_add_samples_and_fusion_meta
Create Date: 2026-07-22
"""
from alembic import op
import sqlalchemy as sa


revision = "0005_add_samples_archive_table"
down_revision = "0004_add_samples_and_fusion_meta"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # samples_archive mirrors the samples schema but intentionally has no HNSW
    # index — archived rows are not searched, only retained for audit/replay.
    op.create_table(
        "samples_archive",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("mail_hash", sa.String(length=64), nullable=False),
        sa.Column("subject_raw", sa.Text(), nullable=False),
        sa.Column("subject_clean", sa.Text(), nullable=False),
        sa.Column("sender", sa.Text(), nullable=False),
        sa.Column("sender_domain", sa.String(length=255), nullable=False),
        sa.Column("body", sa.Text(), nullable=False),
        sa.Column("label_l1", sa.String(length=64), nullable=False),
        sa.Column("label_l2", sa.String(length=64), nullable=False),
        sa.Column("label_l3", sa.String(length=64), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("reviewed", sa.Boolean(), nullable=False),
        sa.Column("thread_parsed", sa.Boolean(), nullable=False),
        sa.Column("embedding_thread", sa.JSON(), nullable=True),
        sa.Column("embedding_segment_0", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("batch_confirmed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("idx_samples_archive_label", "samples_archive", ["label_l3"])
    op.create_index("idx_samples_archive_created_at", "samples_archive", ["created_at"])


def downgrade() -> None:
    op.drop_index("idx_samples_archive_created_at", table_name="samples_archive")
    op.drop_index("idx_samples_archive_label", table_name="samples_archive")
    op.drop_table("samples_archive")
