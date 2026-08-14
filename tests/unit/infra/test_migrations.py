from __future__ import annotations

from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import create_async_engine

from mailagent.infra.migrations import upgrade_database


async def test_upgrade_database_adds_columns_to_a_version_1_database(tmp_path: Path) -> None:
    """已存在的 0001 数据库启动时会升级到当前 schema。"""

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'legacy.db'}"
    engine = create_async_engine(database_url)
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE TABLE processing_runs ("
                "id VARCHAR(36) PRIMARY KEY, status VARCHAR(32) NOT NULL, "
                "payload JSON NOT NULL, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)"
            )
        )
        await connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))
        await connection.execute(
            text("INSERT INTO alembic_version (version_num) VALUES ('0001_initial_schema')")
        )

    await upgrade_database(database_url)

    async with engine.connect() as connection:
        columns = (await connection.execute(text("PRAGMA table_info(processing_runs)"))).all()
        ledger_columns = (
            await connection.execute(
                text("PRAGMA table_info(mail_gateway_ingest_ledger)")
            )
        ).all()
        feedback_columns = (
            await connection.execute(text("PRAGMA table_info(classification_feedback)"))
        ).all()
        feedback_indexes = (
            await connection.execute(text("PRAGMA index_list(classification_feedback)"))
        ).all()
        feedback_constraints = await connection.run_sync(
            lambda sync_connection: inspect(sync_connection).get_unique_constraints(
                "classification_feedback"
            )
        )
    await engine.dispose()

    assert {column[1] for column in columns} >= {
        "classification",
        "calibration_log",
    }
    assert "server_id" in {column[1] for column in ledger_columns}
    assert {column[1] for column in feedback_columns} >= {
        "run_id",
        "revision",
        "reviewed_at",
        "eligible_for_sample_proposal",
    }
    index_names = {index[1] for index in feedback_indexes}
    assert index_names >= {
        "ix_classification_feedback_run_id",
        "ix_classification_feedback_reviewed_at",
        "ix_classification_feedback_eligible_for_sample_proposal",
    }
    assert {
        (constraint["name"], tuple(constraint["column_names"]))
        for constraint in feedback_constraints
    } >= {("uq_feedback_run_revision", ("run_id", "revision"))}


async def test_upgrade_database_bootstraps_a_legacy_database_without_alembic_history(
    tmp_path: Path,
) -> None:
    """旧版 create_tables() 建出的库没有版本记录时也能安全升级。"""

    database_url = f"sqlite+aiosqlite:///{tmp_path / 'unversioned.db'}"
    engine = create_async_engine(database_url)
    async with engine.begin() as connection:
        await connection.execute(
            text(
                "CREATE TABLE processing_runs ("
                "id VARCHAR(36) PRIMARY KEY, status VARCHAR(32) NOT NULL, "
                "payload JSON NOT NULL, created_at DATETIME NOT NULL, updated_at DATETIME NOT NULL)"
            )
        )
        await connection.execute(
            text(
                "CREATE TABLE skill_versions ("
                "id VARCHAR(36) PRIMARY KEY, skill_id VARCHAR(36) NOT NULL, "
                "version INTEGER NOT NULL, published BOOLEAN NOT NULL, payload JSON NOT NULL)"
            )
        )
        # 旧进程曾失败启动时，Alembic 可能已创建版本表但尚未来得及写入版本号。
        await connection.execute(text("CREATE TABLE alembic_version (version_num VARCHAR(32) NOT NULL)"))

    await upgrade_database(database_url)

    async with engine.connect() as connection:
        columns = (await connection.execute(text("PRAGMA table_info(processing_runs)"))).all()
    await engine.dispose()

    assert {column[1] for column in columns} >= {
        "classification",
        "calibration_log",
    }
