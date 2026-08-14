"""add mail gateway cursor and ingest ledger

Revision ID: 0007_add_mail_gateway_ingestion
Revises: 0006_add_flat_sample_provenance
"""
from alembic import op
import sqlalchemy as sa

revision = "0007_add_mail_gateway_ingestion"
down_revision = "0006_add_flat_sample_provenance"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table("mail_gateway_cursor", sa.Column("mailbox_id", sa.String(255), primary_key=True), sa.Column("uidvalidity", sa.Integer(), nullable=False), sa.Column("high_water_uid", sa.Integer(), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False))
    op.create_table("mail_gateway_ingest_ledger", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("mailbox_id", sa.String(255), nullable=False), sa.Column("dedup_key", sa.String(255), nullable=False), sa.Column("uidvalidity", sa.Integer(), nullable=False), sa.Column("uid", sa.Integer(), nullable=False), sa.Column("run_id", sa.String(36), nullable=True), sa.Column("status", sa.String(32), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False), sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False), sa.UniqueConstraint("mailbox_id", "dedup_key", name="uq_gateway_ledger_dedup"))
    op.create_index("ix_gateway_ledger_uid", "mail_gateway_ingest_ledger", ["mailbox_id", "uidvalidity", "uid"])
    op.create_table("mail_gateway_backfill_audit", sa.Column("id", sa.Integer(), primary_key=True), sa.Column("mailbox_id", sa.String(255), nullable=False), sa.Column("since_days", sa.Integer(), nullable=False), sa.Column("max_messages", sa.Integer(), nullable=False), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False))


def downgrade() -> None:
    op.drop_table("mail_gateway_backfill_audit")
    op.drop_index("ix_gateway_ledger_uid", table_name="mail_gateway_ingest_ledger")
    op.drop_table("mail_gateway_ingest_ledger")
    op.drop_table("mail_gateway_cursor")
