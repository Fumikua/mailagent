"""业务 taxonomy YAML 加载、验证、序列化、热重载。

扁平单层结构：每个节点是独立的顶层类别，无 children 嵌套。
taxonomy 仅用于 audit 统计，不驱动路由（路由由 SignalExtractor 驱动）。
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mailagent.domain.versioning import (
    ValidatedAssetSnapshot,
    digest_named_assets,
)

logger = logging.getLogger(__name__)


class TaxonomyNode(BaseModel):
    """单节点：通用分类字段及由业务配置提供的选择指引。

    children 字段保留用于向后兼容（旧 YAML 可能含 children），但扁平结构下
    加载时会忽略非空 children 并 warning。
    """

    code: str
    label: str
    description: str = ""
    keywords: tuple[str, ...] = ()
    selection_guidance: tuple[str, ...] = Field(default=(), max_length=20)
    exclusive: bool = False
    children: tuple["TaxonomyNode", ...] = ()

    model_config = ConfigDict(frozen=True, extra="forbid")

    @field_validator("selection_guidance")
    @classmethod
    def validate_selection_guidance(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("selection_guidance entries must not be blank")
        if any(len(value) > 500 for value in normalized):
            raise ValueError("selection_guidance entries must not exceed 500 characters")
        if len(set(normalized)) != len(normalized):
            raise ValueError("selection_guidance entries must be unique")
        return normalized


class TaxonomyTree(BaseModel):
    """扁平 taxonomy（单层类别列表）"""

    version: str | None = None
    nodes: tuple[TaxonomyNode, ...] = ()

    model_config = ConfigDict(frozen=True, extra="forbid")

    def all_codes(self) -> set[str]:
        """返回所有类别的 code 集合（扁平，无路径前缀）"""

        return {n.code for n in self.nodes}

    def node_count(self) -> int:
        """总节点数（扁平结构 = 顶层节点数）"""

        return len(self.nodes)

    def find_l1(self, code: str) -> TaxonomyNode | None:
        return next((n for n in self.nodes if n.code == code), None)


def load_taxonomy(path: str | Path) -> TaxonomyTree:
    """从 YAML 加载扁平 taxonomy，校验 code 唯一性。

    节点的 children 字段若非空会被忽略并 warning（向后兼容旧三级 YAML）。

    Args:
        path: YAML 文件路径

    Returns:
        TaxonomyTree: 加载后的扁平 taxonomy

    Raises:
            FileNotFoundError: 文件不存在
            ValueError: YAML 格式错误或 code 重复
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"taxonomy file not found: {file_path}")

    raw = yaml.safe_load(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or "nodes" not in raw:
        raise ValueError("taxonomy YAML must have top-level 'nodes' list")

    tree = TaxonomyTree.model_validate(raw)

    # 扁平结构校验：忽略 children 并 warning，code 唯一性检查
    seen: set[str] = set()
    for n in tree.nodes:
        if n.code in seen:
            raise ValueError(f"duplicate taxonomy code: {n.code}")
        seen.add(n.code)
        if n.children:
            logger.warning(
                "taxonomy node '%s' has children but flat structure ignores them; "
                "remove children or they will be silently dropped",
                n.code,
            )

    logger.info("loaded flat taxonomy: %d categories", len(tree.nodes))
    return tree


def _minimal_fallback() -> TaxonomyTree:
    """文件缺失时使用空 taxonomy，不凭空创造任何业务类别。"""

    return TaxonomyTree()


def _taxonomy_snapshot(tree: TaxonomyTree) -> ValidatedAssetSnapshot[TaxonomyTree]:
    return ValidatedAssetSnapshot(
        value=tree,
        version=digest_named_assets(
            [("taxonomy:tree", tree.model_dump_json().encode("utf-8"))]
        ),
    )


def serialize_for_prompt(tree: TaxonomyTree, include_l3: bool = True) -> str:
    """将扁平 taxonomy 序列化为 prompt 文本块。

    格式：[code] label — description  keywords: a, b, c
    每个类别一行，无缩进层级。include_l3 参数保留用于向后兼容（扁平结构无 L3）。

    Args:
        tree: taxonomy 树
        include_l3: 向后兼容参数，扁平结构下无实际作用

    Returns:
        str: 序列化文本
    """
    lines: list[str] = []
    for node in tree.nodes:
        keywords = ", ".join(node.keywords) if node.keywords else ""
        guidance = " | ".join(node.selection_guidance)
        exclusive = "yes" if node.exclusive else "no"
        lines.append(
            f"[{node.code}] {node.label} — {node.description}  keywords: {keywords}  "
            f"selection_guidance: {guidance}  exclusive: {exclusive}"
        )
    return "\n".join(lines)


class TaxonomyLoader:
    """带热重载的 taxonomy 加载器（mtime 轮询，5s 间隔）"""

    def __init__(self, path: str | Path, poll_interval: float = 5.0) -> None:
        self.path = Path(path)
        self.poll_interval = poll_interval
        self._last_mtime: float = 0.0
        self._last_check: float = 0.0
        self._snapshot = _taxonomy_snapshot(_minimal_fallback())
        self._has_loaded_file = False
        self._load()  # 启动时立即加载

    def _load(self) -> None:
        try:
            tree = load_taxonomy(self.path)
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            if self._has_loaded_file:
                logger.error(
                    "taxonomy file not found: %s, keeping previous version",
                    self.path,
                )
            else:
                logger.error("taxonomy file not found: %s, using fallback", self.path)
        except Exception as exc:
            logger.error("failed to load taxonomy: %s, keeping previous version", exc)
        else:
            self._snapshot = _taxonomy_snapshot(tree)
            self._last_mtime = mtime
            self._has_loaded_file = True
            logger.info("taxonomy loaded: %d nodes", tree.node_count())

    def _maybe_reload(self) -> None:
        now = time.monotonic()
        if now - self._last_check < self.poll_interval:
            return
        self._last_check = now
        try:
            mtime = self.path.stat().st_mtime if self.path.exists() else 0.0
        except OSError:
            return
        if mtime != self._last_mtime:
            logger.info("taxonomy file changed, reloading: %s", self.path)
            self._load()

    def get_tree(self) -> TaxonomyTree:
        """获取当前 taxonomy 树（触发热重载检查）"""

        return self.get_snapshot().value

    def get_snapshot(self) -> ValidatedAssetSnapshot[TaxonomyTree]:
        """Return the exact validated taxonomy state active after reload checks."""

        self._maybe_reload()
        return self._snapshot

    def serialize_for_prompt(self, tree: TaxonomyTree | None = None) -> str:
        """序列化当前扁平 taxonomy 为 prompt 文本（每类别一行）。"""

        active_tree = tree or self.get_snapshot().value
        node_count = active_tree.node_count()
        if node_count > 200:
            logger.warning("taxonomy has %d categories (>200), prompt may be large", node_count)
        return serialize_for_prompt(active_tree)
