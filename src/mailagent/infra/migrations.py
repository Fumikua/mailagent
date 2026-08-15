"""数据库 schema 升级入口。

应用进程在连接数据库前调用此模块，确保已有数据库也会执行 Alembic
migration，而不只是为全新数据库创建缺失的表。
"""
from __future__ import annotations

import asyncio
from importlib import resources
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import create_async_engine


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _migration_root() -> Path:
    """Locate Alembic assets in an installed wheel or a source checkout."""

    packaged = resources.files("mailagent").joinpath("_alembic")
    if packaged.is_dir():
        return Path(str(packaged))
    return PROJECT_ROOT


def _make_config(database_url: str) -> Config:
    migration_root = _migration_root()
    config = Config(str(migration_root / "alembic.ini"))
    config.set_main_option("script_location", str(migration_root / "migrations"))
    config.attributes["database_url"] = database_url
    return config


def _upgrade_database(database_url: str, legacy_revision: str | None) -> None:
    config = _make_config(database_url)
    if legacy_revision:
        command.stamp(config, legacy_revision)
    command.upgrade(config, "head")


async def _detect_unversioned_legacy_revision(database_url: str) -> str | None:
    """识别早期 create_tables() 创建、但没有 alembic_version 的数据库。"""

    engine = create_async_engine(database_url)
    try:
        async with engine.connect() as connection:
            def inspect_schema(sync_connection) -> str | None:
                inspector = inspect(sync_connection)
                tables = set(inspector.get_table_names())
                if "processing_runs" not in tables:
                    return None

                if "alembic_version" in tables:
                    recorded_revision = sync_connection.execute(
                        text("SELECT version_num FROM alembic_version LIMIT 1")
                    ).scalar_one_or_none()
                    if recorded_revision:
                        return None

                columns = {column["name"] for column in inspector.get_columns("processing_runs")}
                if "samples_archive" in tables:
                    return "0005_add_samples_archive_table"
                if "samples" in tables or "fusion_meta" in columns:
                    return "0004_add_samples_and_fusion_meta"
                if "classification" in columns:
                    return "0002_add_classification"
                return "0001_initial_schema"

            return await connection.run_sync(inspect_schema)
    finally:
        await engine.dispose()


async def upgrade_database(database_url: str) -> None:
    """将指定数据库升级到当前最新迁移版本。"""

    parsed = make_url(database_url)
    sqlite_database = parsed.database
    if parsed.get_backend_name() == "sqlite" and sqlite_database not in {None, ":memory:"}:
        assert sqlite_database is not None
        Path(sqlite_database).expanduser().parent.mkdir(parents=True, exist_ok=True)
    legacy_revision = await _detect_unversioned_legacy_revision(database_url)
    await asyncio.to_thread(_upgrade_database, database_url, legacy_revision)
