"""add stable POP3 server identity to the ingestion ledger

Revision ID: 0009_add_pop3_server_id
Revises: 0008_add_mail_gateway_protocol
"""
from alembic import op
import sqlalchemy as sa

revision = "0009_add_pop3_server_id"
down_revision = "0008_add_mail_gateway_protocol"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("mail_gateway_ingest_ledger") as batch_op:
        batch_op.add_column(
            sa.Column("server_id", sa.String(255), nullable=True)
        )
        batch_op.create_index(
            "ix_gateway_ledger_server_id",
            ["mailbox_id", "protocol", "server_id"],
            unique=False,
        )


def downgrade() -> None:
    with op.batch_alter_table("mail_gateway_ingest_ledger") as batch_op:
        batch_op.drop_index("ix_gateway_ledger_server_id")
        batch_op.drop_column("server_id")
