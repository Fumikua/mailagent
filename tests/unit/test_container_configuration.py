from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).parents[2]


def test_compose_mounts_runtime_configuration_for_api_and_worker() -> None:
    """两个运行进程必须使用同一份配置、vertical 与业务数据目录。"""

    compose = yaml.safe_load((PROJECT_ROOT / "compose.yaml").read_text())
    for service_name in ("api", "worker"):
        volumes = compose["services"][service_name]["volumes"]
        assert "./config.yml:/app/config.yml:ro" in volumes
        assert "./taxonomy.yaml:/app/taxonomy.yaml:ro" in volumes
        assert "./verticals:/app/verticals:ro" in volumes
        assert "./data:/app/data:ro" in volumes


def test_docker_image_contains_migration_and_default_taxonomy_assets() -> None:
    """镜像自身必须有迁移与默认 vertical 资源，不能依赖开发机偶然存在的文件。"""

    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text()
    assert "COPY alembic.ini ./" in dockerfile
    assert "COPY migrations ./migrations" in dockerfile
    assert "COPY taxonomy.yaml ./taxonomy.yaml" in dockerfile
    assert "COPY verticals ./verticals" in dockerfile


def test_dev_dependencies_include_ruff() -> None:
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert '"ruff>=' in pyproject
