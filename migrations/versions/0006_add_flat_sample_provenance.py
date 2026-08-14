"""add flat sample provenance and quality columns

Revision ID: 0006_add_flat_sample_provenance
Revises: 0005_add_samples_archive_table
Create Date: 2026-07-26
"""
from alembic import op
import sqlalchemy as sa


revision = "0006_add_flat_sample_provenance"
down_revision = "0005_add_samples_archive_table"
branch_labels = None
depends_on = None


def _upgrade_table(table_name: str) -> None:
    with op.batch_alter_table(table_name) as batch:
        batch.alter_column("label_l2", existing_type=sa.String(length=64), nullable=True)
        batch.alter_column("label_l3", existing_type=sa.String(length=64), nullable=True)
        batch.add_column(sa.Column("taxonomy_schema_version", sa.String(length=32), nullable=True))
        batch.add_column(sa.Column("retrieval_document", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("retrieval_fingerprint", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("retrieval_policy_version", sa.String(length=64), nullable=True))
        batch.add_column(sa.Column("quality_disposition", sa.String(length=16), nullable=True))
        batch.add_column(sa.Column("quality_reasons", sa.JSON(), nullable=True))
        batch.add_column(sa.Column("review_override_reason", sa.Text(), nullable=True))


def upgrade() -> None:
    _upgrade_table("samples")
    _upgrade_table("samples_archive")
    op.create_index("idx_samples_retrieval_fingerprint", "samples", ["retrieval_fingerprint"])
    op.create_index("idx_samples_quality_disposition", "samples", ["quality_disposition"])


def downgrade() -> None:
    op.drop_index("idx_samples_quality_disposition", table_name="samples")
    op.drop_index("idx_samples_retrieval_fingerprint", table_name="samples")
    for table_name in ("samples_archive", "samples"):
        with op.batch_alter_table(table_name) as batch:
            batch.drop_column("review_override_reason")
            batch.drop_column("quality_reasons")
            batch.drop_column("quality_disposition")
            batch.drop_column("retrieval_policy_version")
            batch.drop_column("retrieval_fingerprint")
            batch.drop_column("retrieval_document")
            batch.drop_column("taxonomy_schema_version")
            batch.alter_column("label_l3", existing_type=sa.String(length=64), nullable=False)
            batch.alter_column("label_l2", existing_type=sa.String(length=64), nullable=False)
