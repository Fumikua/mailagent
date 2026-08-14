"""add samples table and fusion_meta column

Revision ID: 0004_add_samples_and_fusion_meta
Revises: 0002_add_classification
Create Date: 2026-07-22
"""
from alembic import op
from pgvector.sqlalchemy import Vector
import sqlalchemy as sa


revision = "0004_add_samples_and_fusion_meta"
down_revision = "0002_add_classification"
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    is_postgresql = bind.dialect.name == "postgresql"
    if is_postgresql:
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    embedding_type = Vector(4096) if is_postgresql else sa.JSON()

    # Add fusion_meta JSON column to processing_runs (nullable — present when
    # three-path fusion produces audit metadata; NULL for legacy / non-fusion runs)
    op.add_column("processing_runs", sa.Column("fusion_meta", sa.JSON(), nullable=True))

    # Create samples table for Path B vector similarity search.
    # SQLite stores embeddings as JSON lists; PostgreSQL stores native pgvector
    # values so cosine HNSW indexes can be created in this migration.
    op.create_table(
        "samples",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("mail_hash", sa.String(length=64), nullable=False, unique=True),
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
        sa.Column("embedding_thread", embedding_type, nullable=True),
        sa.Column("embedding_segment_0", embedding_type, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("batch_confirmed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Auxiliary indexes for label / domain / source / time queries.
    op.create_index("idx_samples_label", "samples", ["label_l3"])
    op.create_index("idx_samples_domain", "samples", ["sender_domain"])
    op.create_index("idx_samples_source", "samples", ["source"])
    op.create_index("idx_samples_created_at", "samples", ["created_at"])
    op.create_index("idx_samples_mail_hash", "samples", ["mail_hash"])
    op.create_index("idx_samples_label_l1", "samples", ["label_l1"])

    # PostgreSQL: create pgvector HNSW indexes for coarse and fine retrieval.
    if is_postgresql:
        op.execute(
            "CREATE INDEX idx_samples_embedding_thread_hnsw "
            "ON samples USING hnsw (embedding_thread vector_cosine_ops)"
        )
        op.execute(
            "CREATE INDEX idx_samples_embedding_segment_0_hnsw "
            "ON samples USING hnsw (embedding_segment_0 vector_cosine_ops)"
        )


def downgrade() -> None:
    bind = op.get_bind()
    if bind.dialect.name == "postgresql":
        op.execute("DROP INDEX IF EXISTS idx_samples_embedding_segment_0_hnsw")
        op.execute("DROP INDEX IF EXISTS idx_samples_embedding_thread_hnsw")
    op.drop_index("idx_samples_label_l1", table_name="samples")
    op.drop_index("idx_samples_mail_hash", table_name="samples")
    op.drop_index("idx_samples_created_at", table_name="samples")
    op.drop_index("idx_samples_source", table_name="samples")
    op.drop_index("idx_samples_domain", table_name="samples")
    op.drop_index("idx_samples_label", table_name="samples")
    op.drop_table("samples")
    op.drop_column("processing_runs", "fusion_meta")
