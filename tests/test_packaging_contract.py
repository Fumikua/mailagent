"""Release-artifact checks that do not require installing the wheel."""
from __future__ import annotations

from pathlib import Path
import tomllib


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_wheel_bundles_runtime_migration_assets() -> None:
    project = tomllib.loads(
        (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    force_include = project["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert force_include == {
        "alembic.ini": "mailagent/_alembic/alembic.ini",
        "migrations": "mailagent/_alembic/migrations",
    }


def test_every_docker_copy_source_exists() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    for line in dockerfile.splitlines():
        if not line.startswith("COPY "):
            continue
        source = line.split()[1]
        assert (PROJECT_ROOT / source).exists(), f"missing Docker COPY source: {source}"
