"""add durable background-job outbox

Revision ID: 0011_add_job_outbox
Revises: 0010_add_classification_feedback
"""

from alembic import op
import sqlalchemy as sa

revision = "0011_add_job_outbox"
down_revision = "0010_add_classification_feedback"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "job_outbox",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("job_name", sa.String(128), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("dispatched_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text(), nullable=True),
    )
    op.create_index("ix_job_outbox_job_name", "job_outbox", ["job_name"])
    op.create_index("ix_job_outbox_created_at", "job_outbox", ["created_at"])
    op.create_index("ix_job_outbox_dispatched_at", "job_outbox", ["dispatched_at"])


def downgrade() -> None:
    op.drop_index("ix_job_outbox_dispatched_at", table_name="job_outbox")
    op.drop_index("ix_job_outbox_created_at", table_name="job_outbox")
    op.drop_index("ix_job_outbox_job_name", table_name="job_outbox")
    op.drop_table("job_outbox")
