"""Shared pytest configuration for the vertical-neutral framework suite."""
import os
import logging
from collections.abc import Generator
from pathlib import Path

import pytest

# 项目根目录（d:\mailagent\）
PROJECT_ROOT = Path(__file__).parent.parent

# Core has no business default. Tests explicitly select the bundled example
# plugin before application modules construct Settings at import time.
os.environ.setdefault("MAILAGENT_VERTICAL__ID", "example-triage")
os.environ.setdefault("MAILAGENT_VERTICAL__VERTICALS_PATH", str(PROJECT_ROOT / "verticals"))


@pytest.fixture(autouse=True)
def _reenable_mailagent_loggers() -> Generator[None, None, None]:
    """Re-enable mailagent loggers if Alembic's fileConfig disabled them.

    When TestClient starts the API lifespan, ``upgrade_database`` triggers
    Alembic's ``fileConfig`` which sets ``disable_existing_loggers=True`` and
    silences loggers not listed in ``alembic.ini``'s ``[loggers]`` section.
    This fixture restores logger state before each test so ``caplog`` works
    regardless of test execution order.
    """
    snapshot: list[tuple[logging.Logger, bool, bool]] = []
    for name, lg in list(logging.Logger.manager.loggerDict.items()):
        if name.startswith("mailagent") and isinstance(lg, logging.Logger):
            snapshot.append((lg, lg.disabled, lg.propagate))
            lg.disabled = False
            lg.propagate = True
    yield
    for lg, prev_disabled, prev_propagate in snapshot:
        lg.disabled = prev_disabled
        lg.propagate = prev_propagate
