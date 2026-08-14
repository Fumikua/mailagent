"""add protocol column to mail_gateway tables

Revision ID: 0008_add_mail_gateway_protocol
Revises: 0007_add_mail_gateway_ingestion
"""
from alembic import op
import sqlalchemy as sa

revision = "0008_add_mail_gateway_protocol"
down_revision = "0007_add_mail_gateway_ingestion"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # SQLite needs batch mode to alter column nullability; PostgreSQL supports
    # plain ALTER COLUMN. Batch mode works for both.
    with op.batch_alter_table("mail_gateway_cursor") as batch_op:
        batch_op.add_column(sa.Column("protocol", sa.String(16), nullable=False, server_default="imap"))
        batch_op.alter_column("uidvalidity", existing_type=sa.Integer(), nullable=True)

    with op.batch_alter_table("mail_gateway_ingest_ledger") as batch_op:
        batch_op.add_column(sa.Column("protocol", sa.String(16), nullable=False, server_default="imap"))
        batch_op.alter_column("uidvalidity", existing_type=sa.Integer(), nullable=True)


def downgrade() -> None:
    with op.batch_alter_table("mail_gateway_ingest_ledger") as batch_op:
        batch_op.alter_column("uidvalidity", existing_type=sa.Integer(), nullable=False)
        batch_op.drop_column("protocol")

    with op.batch_alter_table("mail_gateway_cursor") as batch_op:
        batch_op.alter_column("uidvalidity", existing_type=sa.Integer(), nullable=False)
        batch_op.drop_column("protocol")
