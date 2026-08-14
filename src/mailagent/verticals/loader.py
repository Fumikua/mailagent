"""Load and validate a single vertical's file-based manifest."""
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class VerticalConfigurationError(ValueError):
    """Raised when a vertical cannot be safely selected at startup."""


class VerticalLLMSettings(BaseModel):
    """Vertical-level overrides for LLM-backed classification behavior.

    These knobs tune how a vertical's emails are prepared for the LLM call.
    Different business verticals have different mail shapes, so a single
    global threshold would misfit at least one of them.
    """

    body_max_chars: int = Field(default=16_000, ge=512, le=1_000_000)
    """Hard cap on the mail body length fed to the LLM classifier.

    Bodies longer than this are head/tail truncated so the LLM still sees
    both the opening context and the latest segment. Default mirrors the
    historical hard-coded value; verticals with long reply chains should
    lower it.
    """


class VerticalManifest(BaseModel):
    id: str = Field(min_length=1, pattern=r"^[a-z0-9][a-z0-9-]*$")
    namespace: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    data_schema_version: str = Field(min_length=1)
    taxonomy: str = Field(min_length=1)
    data_schema: str = Field(min_length=1)
    # Deprecated compatibility field. Executable code is selected from the
    # installed VerticalPlugin registry, never from an external profile.
    runtime_factory: str | None = Field(default=None, min_length=3)
    rules: "VerticalAssetDeclaration | None" = None
    rag: "VerticalAssetDeclaration | None" = None
    target_profiles: "VerticalAssetDeclaration | None" = None
    signals: "VerticalAssetDeclaration | None" = None
    preprocessing: "VerticalAssetDeclaration | None" = None
    retrieval_cleaning: "VerticalAssetDeclaration | None" = None
    llm: "VerticalLLMSettings | None" = None
    enrichers: list[str] = Field(default_factory=list)


class VerticalAssetDeclaration(BaseModel):
    path: str = Field(min_length=1)
    version: str = Field(min_length=1)


class VerticalAsset(BaseModel):
    path: Path
    version: str

    model_config = {"arbitrary_types_allowed": True}


class LoadedVertical(BaseModel):
    manifest: VerticalManifest
    taxonomy_path: Path
    data_schema_path: Path
    data_schema: dict[str, object]
    rules: VerticalAsset | None = None
    rag: VerticalAsset | None = None
    target_profiles: VerticalAsset | None = None
    signals: VerticalAsset | None = None
    preprocessing: VerticalAsset | None = None
    retrieval_cleaning: VerticalAsset | None = None

    model_config = {"arbitrary_types_allowed": True}


def _load_asset(
    manifest_path: Path,
    declaration: VerticalAssetDeclaration | None,
    asset_type: str,
) -> VerticalAsset | None:
    if declaration is None:
        return None
    root = manifest_path.parent.resolve()
    asset_path = (root / declaration.path).resolve()
    try:
        asset_path.relative_to(root)
    except ValueError as exc:
        raise VerticalConfigurationError(
            f"vertical {asset_type} asset escapes manifest directory: {asset_path}"
        ) from exc
    # Accept both directories (rules/, rag/) and single files (target_profiles.yaml).
    if not (asset_path.is_dir() or asset_path.is_file()):
        raise VerticalConfigurationError(
            f"vertical {asset_type} asset path does not exist: {asset_path}"
        )
    return VerticalAsset(path=asset_path, version=declaration.version)


def load_vertical(manifest_path: str | Path) -> LoadedVertical:
    """Read manifest and verify every declared local asset before processing mail."""

    path = Path(manifest_path)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise VerticalConfigurationError(f"cannot read vertical manifest: {path}") from exc
    except yaml.YAMLError as exc:
        raise VerticalConfigurationError(f"invalid vertical manifest YAML: {path}") from exc

    try:
        manifest = VerticalManifest.model_validate(raw)
    except Exception as exc:
        raise VerticalConfigurationError(f"invalid vertical manifest: {path}") from exc

    taxonomy_path = (path.parent / manifest.taxonomy).resolve()
    if not taxonomy_path.is_file():
        raise VerticalConfigurationError(f"vertical taxonomy does not exist: {taxonomy_path}")

    data_schema_path = (path.parent / manifest.data_schema).resolve()
    if not data_schema_path.is_file():
        raise VerticalConfigurationError(f"vertical data schema does not exist: {data_schema_path}")
    try:
        data_schema = json.loads(data_schema_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerticalConfigurationError(f"invalid vertical data schema: {data_schema_path}") from exc
    if not isinstance(data_schema, dict):
        raise VerticalConfigurationError(f"vertical data schema must be an object: {data_schema_path}")

    rules = _load_asset(path, manifest.rules, "rules")
    rag = _load_asset(path, manifest.rag, "rag")
    target_profiles = _load_asset(path, manifest.target_profiles, "target_profiles")
    signals = _load_asset(path, manifest.signals, "signals")
    preprocessing = _load_asset(path, manifest.preprocessing, "preprocessing")
    retrieval_cleaning = _load_asset(path, manifest.retrieval_cleaning, "retrieval_cleaning")

    return LoadedVertical(
        manifest=manifest,
        taxonomy_path=taxonomy_path,
        data_schema_path=data_schema_path,
        data_schema=data_schema,
        rules=rules,
        rag=rag,
        target_profiles=target_profiles,
        signals=signals,
        preprocessing=preprocessing,
        retrieval_cleaning=retrieval_cleaning,
    )


def load_runtime_factory(loaded: LoadedVertical) -> Any:
    """Import a legacy manifest runtime factory during the migration window."""

    if loaded.manifest.runtime_factory is None:
        raise VerticalConfigurationError(
            "vertical profile has no legacy runtime_factory; resolve its installed plugin instead"
        )

    module_name, separator, attribute_name = loaded.manifest.runtime_factory.partition(":")
    if not separator or not module_name or not attribute_name:
        raise VerticalConfigurationError(
            "vertical runtime_factory must use 'module.path:callable_name' format"
        )
    try:
        factory = getattr(import_module(module_name), attribute_name)
    except (ImportError, AttributeError) as exc:
        raise VerticalConfigurationError(
            f"cannot load vertical runtime factory {loaded.manifest.runtime_factory}"
        ) from exc
    if not callable(factory):
        raise VerticalConfigurationError(
            f"vertical runtime factory is not callable: {loaded.manifest.runtime_factory}"
        )
    return factory
