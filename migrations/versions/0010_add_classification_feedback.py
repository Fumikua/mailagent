"""add immutable classification feedback

Revision ID: 0010_add_classification_feedback
Revises: 0009_add_pop3_server_id
"""
from alembic import op
import sqlalchemy as sa

revision = "0010_add_classification_feedback"
down_revision = "0009_add_pop3_server_id"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "classification_feedback",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "run_id",
            sa.String(36),
            sa.ForeignKey("processing_runs.id"),
            nullable=False,
        ),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("predicted_labels", sa.JSON(), nullable=False),
        sa.Column("final_labels", sa.JSON(), nullable=False),
        sa.Column("error_reasons", sa.JSON(), nullable=False),
        sa.Column("reviewer_id", sa.String(255), nullable=False),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("versions", sa.JSON(), nullable=True),
        sa.Column(
            "eligible_for_sample_proposal",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.UniqueConstraint(
            "run_id",
            "revision",
            name="uq_feedback_run_revision",
        ),
    )
    op.create_index(
        "ix_classification_feedback_run_id",
        "classification_feedback",
        ["run_id"],
    )
    op.create_index(
        "ix_classification_feedback_reviewed_at",
        "classification_feedback",
        ["reviewed_at"],
    )
    op.create_index(
        "ix_classification_feedback_eligible_for_sample_proposal",
        "classification_feedback",
        ["eligible_for_sample_proposal"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_classification_feedback_eligible_for_sample_proposal",
        table_name="classification_feedback",
    )
    op.drop_index(
        "ix_classification_feedback_reviewed_at",
        table_name="classification_feedback",
    )
    op.drop_index(
        "ix_classification_feedback_run_id",
        table_name="classification_feedback",
    )
    op.drop_table("classification_feedback")
