"""Target profile loader for label-scoped vector retrieval (P0).

Declares a list of high-value labels whose vector confirmation step in
``FusionOrchestrator`` (``rule_vector_confirmed``) should run against a
restricted ``label_l1`` scope instead of the global sample table. P0 only
carries ``label`` + ``vector_scope``; P1 fields (``confirm_window`` /
``accept_threshold`` / ``dry_run``) are ignored via ``extra="ignore"`` for
forward compatibility.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mailagent.domain.versioning import ValidatedAssetSnapshot, digest_named_assets
from mailagent.classification.taxonomy import TaxonomyLoader, TaxonomyTree

logger = logging.getLogger(__name__)


class TargetProfile(BaseModel):
    """One target label and the code scope for its vector confirmation."""

    label: str = Field(min_length=1)  # flat taxonomy code declared by the active vertical
    vector_scope: tuple[str, ...] = Field(min_length=1)  # codes for scoped knn_search

    model_config = ConfigDict(extra="ignore", frozen=True)  # P1 forward compatibility

    @field_validator("label")
    @classmethod
    def normalize_label(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("target profile label must not be blank")
        return normalized

    @field_validator("vector_scope")
    @classmethod
    def normalize_vector_scope(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(value.strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("target profile vector_scope entries must not be blank")
        return normalized


@dataclass(frozen=True, slots=True)
class TargetProfileSet:
    """Immutable validated target profiles bound to one taxonomy snapshot."""

    targets: tuple[TargetProfile, ...]
    taxonomy_version: str | None


def _target_profile_snapshot(
    targets: tuple[TargetProfile, ...],
    taxonomy_version: str | None,
) -> ValidatedAssetSnapshot[TargetProfileSet]:
    canonical = json.dumps(
        {"targets": [target.model_dump(mode="json") for target in targets]},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return ValidatedAssetSnapshot(
        value=TargetProfileSet(
            targets=targets,
            taxonomy_version=taxonomy_version,
        ),
        version=digest_named_assets([("target_profiles:targets", canonical)]),
    )


class TargetProfileLoader:
    """Hot-reloadable loader for ``target_profiles.yaml`` (mtime polling, 5s).

    Behavior:
        - Missing file → empty target list + INFO log (feature off).
        - Empty ``targets: []`` → empty target list (feature off).
        - Any invalid entry rejects the complete candidate snapshot.
        - YAML/validation failure keeps the previous complete snapshot.
    """

    def __init__(
        self,
        config_path: Path,
        taxonomy_loader: TaxonomyLoader,
        poll_interval: float = 5.0,
    ) -> None:
        self._path = Path(config_path)
        self._taxonomy_loader = taxonomy_loader
        self._poll_interval = poll_interval
        self._last_mtime_ns: int = 0
        self._last_check: float = 0.0
        self._snapshot = _target_profile_snapshot((), None)
        self._has_loaded_file = False
        self._load()

    def _load(
        self,
        taxonomy_snapshot: ValidatedAssetSnapshot[TaxonomyTree] | None = None,
    ) -> None:
        if not self._path.exists():
            if self._has_loaded_file:
                logger.warning(
                    "target_profiles.yaml not found: %s, keeping previous snapshot",
                    self._path,
                )
            else:
                logger.info("target_profiles.yaml not found: %s, feature off", self._path)
            return

        try:
            raw = yaml.safe_load(self._path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("top-level value must be a mapping")
            entries = raw.get("targets", [])
            if not isinstance(entries, list):
                raise ValueError("'targets' must be a list")
            if taxonomy_snapshot is None:
                taxonomy_snapshot = self._taxonomy_loader.get_snapshot()
            valid_codes = taxonomy_snapshot.value.all_codes()
            new_targets: list[TargetProfile] = []
            for idx, entry in enumerate(entries):
                if not isinstance(entry, dict):
                    raise ValueError(f"entry[{idx}] must be a mapping")
                profile = TargetProfile.model_validate(entry)
                if profile.label not in valid_codes:
                    raise ValueError(
                        f"entry[{idx}] label '{profile.label}' is not in taxonomy"
                    )
                new_targets.append(profile)
            mtime_ns = self._path.stat().st_mtime_ns
            candidate = _target_profile_snapshot(
                tuple(new_targets),
                taxonomy_snapshot.version,
            )
        except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
            logger.warning(
                "failed to validate target_profiles.yaml: %s, keeping previous snapshot",
                exc,
            )
            return

        self._snapshot = candidate
        self._last_mtime_ns = mtime_ns
        self._has_loaded_file = True
        logger.info("loaded %d target profiles from %s", len(new_targets), self._path)

    def _maybe_reload(
        self,
        taxonomy_snapshot: ValidatedAssetSnapshot[TaxonomyTree] | None = None,
    ) -> None:
        if taxonomy_snapshot is None:
            taxonomy_snapshot = self._taxonomy_loader.get_snapshot()
        taxonomy_changed = (
            taxonomy_snapshot.version != self._snapshot.value.taxonomy_version
        )
        now = time.monotonic()
        if not taxonomy_changed and now - self._last_check < self._poll_interval:
            return
        self._last_check = now
        try:
            mtime_ns = self._path.stat().st_mtime_ns if self._path.exists() else 0
        except OSError:
            return
        if mtime_ns != self._last_mtime_ns or taxonomy_changed:
            logger.info("target_profiles.yaml changed, reloading: %s", self._path)
            self._load(taxonomy_snapshot)

    def get_snapshot(
        self,
        taxonomy_snapshot: ValidatedAssetSnapshot[TaxonomyTree] | None = None,
    ) -> ValidatedAssetSnapshot[TargetProfileSet]:
        """Return the exact complete target-profile state active for a run."""

        self._maybe_reload(taxonomy_snapshot)
        return self._snapshot

    def get_targets(self) -> list[TargetProfile]:
        """Return current targets (triggers hot-reload check)."""

        return list(self.get_snapshot().value.targets)

    def find_match(
        self,
        label_code: str,
        *,
        snapshot: ValidatedAssetSnapshot[TargetProfileSet] | None = None,
    ) -> TargetProfile | None:
        """Find an exact flat-code target (P0: no confidence window)."""

        if not label_code:
            return None
        targets = (
            snapshot.value.targets
            if snapshot is not None
            else self.get_snapshot().value.targets
        )
        for target in targets:
            if target.label == label_code:
                return target
        return None
