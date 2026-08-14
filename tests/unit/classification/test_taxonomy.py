"""Taxonomy 加载/序列化/热重载/降级测试（扁平结构）。"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from mailagent.classification.taxonomy import (
    TaxonomyLoader,
    TaxonomyTree,
    load_taxonomy,
    serialize_for_prompt,
)


@pytest.fixture
def valid_taxonomy_yaml(tmp_path: Path) -> Path:
    path = tmp_path / "taxonomy.yaml"
    path.write_text(
        """
nodes:
  - code: entity_report
    label: 实体报告
    description: 抵港/中午报告
    keywords: [pre-event, status report]
    selection_guidance:
      - "  Primary when the configured report intent is explicit  "
  - code: noise
    label: 噪声
    description: 不处理
    exclusive: true
""",
        encoding="utf-8",
    )
    return path


def test_load_taxonomy_returns_tree(valid_taxonomy_yaml: Path) -> None:
    tree = load_taxonomy(valid_taxonomy_yaml)
    assert isinstance(tree, TaxonomyTree)
    assert len(tree.nodes) == 2  # entity_report + noise
    assert tree.node_count() == 2  # 扁平结构 = 顶层节点数
    assert tree.find_l1("entity_report").selection_guidance == (
        "Primary when the configured report intent is explicit",
    )
    assert tree.find_l1("noise").exclusive is True


def test_load_taxonomy_all_codes(valid_taxonomy_yaml: Path) -> None:
    tree = load_taxonomy(valid_taxonomy_yaml)
    codes = tree.all_codes()
    assert "entity_report" in codes
    assert "noise" in codes
    # 扁平结构不产生路径前缀 code
    assert "entity_report.schedule" not in codes


def test_load_taxonomy_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_taxonomy(tmp_path / "nonexistent.yaml")


def test_load_taxonomy_duplicate_code_raises(tmp_path: Path) -> None:
    path = tmp_path / "dup.yaml"
    path.write_text(
        """
nodes:
  - code: entity_report
    label: a
  - code: entity_report
    label: dup
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate taxonomy code"):
        load_taxonomy(path)


def test_load_taxonomy_children_ignored_with_warning(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """旧三级 YAML 的 children 字段应被忽略并 warning（向后兼容）"""
    path = tmp_path / "legacy.yaml"
    path.write_text(
        """
nodes:
  - code: entity_report
    label: 实体报告
    description: 报告
    children:
      - code: arrival_report
        label: 抵港报
""",
        encoding="utf-8",
    )
    with caplog.at_level("WARNING"):
        tree = load_taxonomy(path)
    # children 被忽略，只保留顶层节点
    assert tree.node_count() == 1
    assert "arrival_report" not in tree.all_codes()
    assert any("children" in rec.message for rec in caplog.records)


def test_load_taxonomy_invalid_yaml_structure(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("not_a_dict: true", encoding="utf-8")
    with pytest.raises(ValueError, match="must have top-level 'nodes'"):
        load_taxonomy(path)


def test_serialize_for_prompt_flat_format(valid_taxonomy_yaml: Path) -> None:
    tree = load_taxonomy(valid_taxonomy_yaml)
    text = serialize_for_prompt(tree)
    assert "[entity_report]" in text
    assert "[noise]" in text
    assert "实体报告" in text
    assert "selection_guidance: Primary when the configured report intent is explicit" in text
    assert "exclusive: yes" in text
    # 扁平格式无路径前缀
    assert "entity_report." not in text


def test_serialize_for_prompt_include_l3_backward_compat(valid_taxonomy_yaml: Path) -> None:
    """include_l3 参数保留但扁平结构下无实际作用"""
    tree = load_taxonomy(valid_taxonomy_yaml)
    text_with = serialize_for_prompt(tree, include_l3=True)
    text_without = serialize_for_prompt(tree, include_l3=False)
    assert text_with == text_without


def test_taxonomy_loader_hot_reload(valid_taxonomy_yaml: Path) -> None:
    loader = TaxonomyLoader(valid_taxonomy_yaml, poll_interval=0.05)
    initial_codes = loader.get_tree().all_codes()
    assert "entity_report" in initial_codes

    # 修改文件
    time.sleep(0.1)
    valid_taxonomy_yaml.write_text(
        """
nodes:
  - code: new_category
    label: new
    description: new node
""",
        encoding="utf-8",
    )
    # 等待轮询
    time.sleep(0.1)
    reloaded = loader.get_tree()
    assert "new_category" in reloaded.all_codes()
    assert "entity_report" not in reloaded.all_codes()


def test_taxonomy_successful_reload_replaces_tree_and_version_atomically(
    valid_taxonomy_yaml: Path,
) -> None:
    loader = TaxonomyLoader(valid_taxonomy_yaml, poll_interval=0)
    before = loader.get_snapshot()
    time.sleep(0.01)
    valid_taxonomy_yaml.write_text(
        "nodes:\n  - code: new_category\n    label: new\n",
        encoding="utf-8",
    )

    after = loader.get_snapshot()

    assert after.version != before.version
    assert after.value.all_codes() == {"new_category"}


def test_taxonomy_throttle_preserves_tree_and_version_together(
    valid_taxonomy_yaml: Path,
) -> None:
    loader = TaxonomyLoader(valid_taxonomy_yaml, poll_interval=60)
    before = loader.get_snapshot()
    time.sleep(0.01)
    valid_taxonomy_yaml.write_text(
        "nodes:\n  - code: new_category\n    label: new\n",
        encoding="utf-8",
    )

    after = loader.get_snapshot()

    assert after is before
    assert "entity_report" in after.value.all_codes()


def test_taxonomy_loader_invalid_yaml_keeps_previous(valid_taxonomy_yaml: Path) -> None:
    loader = TaxonomyLoader(valid_taxonomy_yaml, poll_interval=0.05)
    original_count = loader.get_tree().node_count()

    time.sleep(0.1)
    valid_taxonomy_yaml.write_text("invalid: : :", encoding="utf-8")
    time.sleep(0.1)
    # 无效 YAML 时保留旧版本
    assert loader.get_tree().node_count() == original_count


def test_taxonomy_invalid_reload_preserves_tree_and_version_together(
    valid_taxonomy_yaml: Path,
) -> None:
    loader = TaxonomyLoader(valid_taxonomy_yaml, poll_interval=0)
    before = loader.get_snapshot()
    time.sleep(0.01)
    valid_taxonomy_yaml.write_text("invalid: : :", encoding="utf-8")

    after = loader.get_snapshot()

    assert after is before
    assert "entity_report" in after.value.all_codes()


def test_taxonomy_loader_missing_file_uses_empty_fallback(tmp_path: Path) -> None:
    nonexistent = tmp_path / "missing.yaml"
    loader = TaxonomyLoader(nonexistent, poll_interval=0.05)
    tree = loader.get_tree()
    assert tree.node_count() == 0
    assert tree.all_codes() == set()


@pytest.mark.parametrize(
    "invalid_node",
    [
        "selection_guidance: ['   ']",
        "selection_guidance: [same, same]",
        "unknown_business_knob: true",
    ],
)
def test_taxonomy_rejects_ineffective_or_unknown_business_config(
    tmp_path: Path,
    invalid_node: str,
) -> None:
    path = tmp_path / "invalid-config.yaml"
    path.write_text(
        f"nodes:\n  - code: category_a\n    label: A\n    {invalid_node}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError):
        load_taxonomy(path)


def test_taxonomy_loader_serialize_above_200_nodes(tmp_path: Path) -> None:
    """扁平结构 >200 节点时 warning 但仍序列化全部（无 L3 可降级）"""
    nodes = [{"code": f"cat_{i}", "label": f"Cat-{i}"} for i in range(201)]
    import yaml as _yaml
    yaml_content = _yaml.dump({"nodes": nodes})
    path = tmp_path / "big.yaml"
    path.write_text(yaml_content, encoding="utf-8")

    loader = TaxonomyLoader(path, poll_interval=0.05)
    assert loader.get_tree().node_count() > 200
    serialized = loader.serialize_for_prompt()
    assert "[cat_0]" in serialized
    assert "[cat_200]" in serialized  # 全部序列化，不降级
