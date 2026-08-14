"""Dialect-specific tests for the Path B samples migration."""
from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import sqlalchemy as sa


MIGRATION_PATH = (
    Path(__file__).resolve().parents[3]
    / "migrations"
    / "versions"
    / "0004_add_samples_and_fusion_meta.py"
)


class RecordingOp:
    def __init__(self, dialect: str) -> None:
        self._bind = SimpleNamespace(dialect=SimpleNamespace(name=dialect))
        self.tables: list[tuple[str, tuple[sa.Column[Any], ...]]] = []
        self.indexes: list[str] = []
        self.executed_sql: list[str] = []

    def get_bind(self) -> Any:
        return self._bind

    def add_column(self, *args: Any, **kwargs: Any) -> None:
        return None

    def create_table(self, name: str, *columns: sa.Column[Any]) -> None:
        self.tables.append((name, columns))

    def create_index(self, name: str, *args: Any, **kwargs: Any) -> None:
        self.indexes.append(name)

    def execute(self, statement: str) -> None:
        self.executed_sql.append(statement)


def _load_migration() -> Any:
    spec = importlib.util.spec_from_file_location("vspb_migration", MIGRATION_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_columns(op: RecordingOp) -> dict[str, sa.Column[Any]]:
    name, columns = next(table for table in op.tables if table[0] == "samples")
    assert name == "samples"
    return {column.name: column for column in columns}


def test_postgres_upgrade_uses_vector_columns_and_two_hnsw_indexes() -> None:
    migration = _load_migration()
    op = RecordingOp("postgresql")
    migration.op = op

    migration.upgrade()

    columns = _sample_columns(op)
    assert str(columns["embedding_thread"].type) == "VECTOR(4096)"
    assert str(columns["embedding_segment_0"].type) == "VECTOR(4096)"
    assert columns["mail_hash"].unique is True
    assert "CREATE EXTENSION IF NOT EXISTS vector" in op.executed_sql
    assert any("idx_samples_embedding_thread_hnsw" in sql for sql in op.executed_sql)
    assert any("idx_samples_embedding_segment_0_hnsw" in sql for sql in op.executed_sql)


def test_sqlite_upgrade_uses_json_without_pgvector_sql() -> None:
    migration = _load_migration()
    op = RecordingOp("sqlite")
    migration.op = op

    migration.upgrade()

    columns = _sample_columns(op)
    assert isinstance(columns["embedding_thread"].type, sa.JSON)
    assert isinstance(columns["embedding_segment_0"].type, sa.JSON)
    assert op.executed_sql == []
