"""add classification columns to processing_runs

Revision ID: 0002_add_classification
Revises: 0001_initial_schema
Create Date: 2026-07-20
"""
from alembic import op
import sqlalchemy as sa


revision = "0002_add_classification"
down_revision = "0001_initial_schema"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # status 列已存在，无需重建；PENDING/PROCESSING 通过 enum 扩展在应用层处理
    # 新增 classification + calibration_log JSON 列（NULL 允许 PENDING 时无值）
    op.add_column("processing_runs", sa.Column("classification", sa.Text(), nullable=True))
    op.add_column("processing_runs", sa.Column("calibration_log", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("processing_runs", "calibration_log")
    op.drop_column("processing_runs", "classification")
