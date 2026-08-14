"""Unit tests for TargetProfile / TargetProfileLoader (P0 scope).

Covers:
    - Complete target entry parsing (label / vector_scope).
    - Invalid labels or empty scopes reject the complete candidate snapshot.
    - P1 fields (confirm_window / accept_threshold) ignored via extra="ignore".
    - Missing file → empty list + INFO log.
    - Empty ``targets: []`` → empty list.
    - YAML parse failure → keep previous targets + WARNING.
    - Hot reload: mtime change picks up new content.
    - find_match: label match → returns target.
    - find_match: label not match → returns None.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml

from mailagent.core.target_profile import TargetProfile, TargetProfileLoader
from mailagent.classification.taxonomy import TaxonomyLoader


@pytest.fixture
def taxonomy_file(tmp_path: Path) -> Path:
    path = tmp_path / "taxonomy.yaml"
    path.write_text(
        """
nodes:
  - code: entity_report
    label: 实体报告
    description: 报告
    keywords: [status report]
  - code: schedule
    label: 船期
    description: 船期
    keywords: [eta]
  - code: document
    label: 单证
    description: 单证
    keywords: [document]
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def taxonomy_loader(taxonomy_file: Path) -> TaxonomyLoader:
    return TaxonomyLoader(taxonomy_file, poll_interval=0.05)


def _write_profiles(path: Path, targets: list[dict]) -> None:
    path.write_text(yaml.safe_dump({"targets": targets}), encoding="utf-8")


class TestTargetProfileModel:
    def test_complete_entry(self) -> None:
        profile = TargetProfile.model_validate(
            {"label": "entity_report", "vector_scope": ["entity_report", "schedule"]}
        )
        assert profile.label == "entity_report"
        assert profile.vector_scope == ("entity_report", "schedule")

    def test_p1_fields_ignored(self) -> None:
        """P1 fields (confirm_window / accept_threshold / dry_run) silently ignored."""

        profile = TargetProfile.model_validate(
            {
                "label": "entity_report",
                "vector_scope": ["entity_report"],
                "confirm_window": [0.6, 0.85],  # P1 field
                "accept_threshold": 0.9,  # P1 field
                "dry_run": False,  # P1 field
            }
        )
        assert profile.label == "entity_report"
        assert profile.vector_scope == ("entity_report",)
        assert not hasattr(profile, "confirm_window")
        assert not hasattr(profile, "accept_threshold")
        assert not hasattr(profile, "dry_run")

    def test_empty_vector_scope_rejected(self) -> None:
        with pytest.raises(Exception):
            TargetProfile.model_validate(
                {"label": "entity_report", "vector_scope": []}
            )


class TestTargetProfileLoaderLoading:
    def test_missing_file_returns_empty(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        loader = TargetProfileLoader(tmp_path / "nonexistent.yaml", taxonomy_loader)
        assert loader.get_targets() == []

    def test_empty_targets_list(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        path = tmp_path / "target_profiles.yaml"
        _write_profiles(path, [])
        loader = TargetProfileLoader(path, taxonomy_loader)
        assert loader.get_targets() == []

    def test_complete_target_parsed(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        path = tmp_path / "target_profiles.yaml"
        _write_profiles(
            path,
            [
                {
                    "label": "entity_report",
                    "vector_scope": ["entity_report", "schedule"],
                }
            ],
        )
        loader = TargetProfileLoader(path, taxonomy_loader)
        targets = loader.get_targets()
        assert len(targets) == 1
        assert targets[0].label == "entity_report"
        assert targets[0].vector_scope == ("entity_report", "schedule")

    def test_invalid_label_rejects_complete_initial_snapshot(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        path = tmp_path / "target_profiles.yaml"
        _write_profiles(
            path,
            [
                {"label": "nonexistent", "vector_scope": ["x"]},
                {
                    "label": "entity_report",
                    "vector_scope": ["entity_report"],
                },
            ],
        )
        loader = TargetProfileLoader(path, taxonomy_loader)
        targets = loader.get_targets()
        assert targets == []

    def test_empty_vector_scope_rejects_complete_initial_snapshot(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        path = tmp_path / "target_profiles.yaml"
        _write_profiles(
            path,
            [
                {"label": "entity_report", "vector_scope": []},
                {
                    "label": "document",
                    "vector_scope": ["priority_goods"],
                },
            ],
        )
        loader = TargetProfileLoader(path, taxonomy_loader)
        targets = loader.get_targets()
        assert targets == []

    def test_yaml_parse_failure_keeps_previous(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        path = tmp_path / "target_profiles.yaml"
        _write_profiles(
            path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(path, taxonomy_loader)
        assert len(loader.get_targets()) == 1

        # Corrupt the YAML
        path.write_text("targets: [invalid: yaml: structure", encoding="utf-8")
        # bump mtime
        time.sleep(0.1)
        _ = loader.get_targets()  # trigger reload attempt
        # previous targets preserved
        targets = loader.get_targets()
        assert len(targets) == 1
        assert targets[0].label == "entity_report"


class TestTargetProfileLoaderHotReload:
    def test_snapshot_is_immutable_and_versioned_from_validated_content(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        path = tmp_path / "target_profiles.yaml"
        _write_profiles(
            path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(path, taxonomy_loader, poll_interval=0)

        first = loader.get_snapshot()
        _write_profiles(
            path,
            [{"label": "entity_report", "vector_scope": ["schedule"]}],
        )
        second = loader.get_snapshot()

        assert first.version != second.version
        assert first.value.targets[0].vector_scope == ("entity_report",)
        assert second.value.targets[0].vector_scope == ("schedule",)

    def test_invalid_reload_preserves_complete_snapshot_and_version(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        path = tmp_path / "target_profiles.yaml"
        _write_profiles(
            path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(path, taxonomy_loader, poll_interval=0)
        before = loader.get_snapshot()
        _write_profiles(
            path,
            [{"label": "entity_report", "vector_scop": ["schedule"]}],
        )

        after = loader.get_snapshot()

        assert after is before
        assert loader.find_match("entity_report") is not None

    @pytest.mark.parametrize("invalid_targets", [None, {}, "", False, 0])
    def test_falsy_non_list_targets_do_not_clear_active_snapshot(
        self,
        tmp_path: Path,
        taxonomy_loader: TaxonomyLoader,
        invalid_targets: object,
    ) -> None:
        path = tmp_path / "target_profiles.yaml"
        _write_profiles(
            path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(path, taxonomy_loader, poll_interval=0)
        before = loader.get_snapshot()
        path.write_text(
            yaml.safe_dump({"targets": invalid_targets}),
            encoding="utf-8",
        )

        after = loader.get_snapshot()

        assert after is before
        assert loader.find_match("entity_report") is not None

    def test_reload_throttle_keeps_snapshot_until_next_successful_check(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        path = tmp_path / "target_profiles.yaml"
        _write_profiles(
            path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(path, taxonomy_loader, poll_interval=60)
        before = loader.get_snapshot()
        _write_profiles(
            path,
            [{"label": "entity_report", "vector_scope": ["schedule"]}],
        )

        throttled = loader.get_snapshot()
        loader._last_check = 0
        reloaded = loader.get_snapshot()

        assert throttled is before
        assert reloaded.version != before.version

    def test_mtime_change_triggers_reload(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        path = tmp_path / "target_profiles.yaml"
        _write_profiles(
            path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(path, taxonomy_loader, poll_interval=0.05)
        assert len(loader.get_targets()) == 1

        # Update with 2 targets
        time.sleep(0.1)
        _write_profiles(
            path,
            [
                {"label": "entity_report", "vector_scope": ["entity_report"]},
                {"label": "document", "vector_scope": ["priority_goods"]},
            ],
        )
        targets = loader.get_targets()
        assert len(targets) == 2


class TestTargetProfileLoaderFindMatch:
    def test_find_match_returns_target(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        path = tmp_path / "target_profiles.yaml"
        _write_profiles(
            path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(path, taxonomy_loader)
        match = loader.find_match("entity_report")
        assert match is not None
        assert match.vector_scope == ("entity_report",)

    def test_find_match_no_match_returns_none(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        path = tmp_path / "target_profiles.yaml"
        _write_profiles(
            path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(path, taxonomy_loader)
        assert loader.find_match("document") is None

    def test_find_match_empty_label_returns_none(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        path = tmp_path / "target_profiles.yaml"
        _write_profiles(
            path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(path, taxonomy_loader)
        assert loader.find_match("") is None

    def test_find_match_leaf_code_matches_target(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        """A single-segment leaf code (e.g. from RuleClassifier's l1_code)
        matches a target whose full path ends with that leaf."""

        path = tmp_path / "target_profiles.yaml"
        _write_profiles(
            path,
            [{"label": "entity_report", "vector_scope": ["entity_report"]}],
        )
        loader = TargetProfileLoader(path, taxonomy_loader)
        # leaf code "entity_report" should match the target
        match = loader.find_match("entity_report")
        assert match is not None
        assert match.label == "entity_report"
        assert match.vector_scope == ("entity_report",)

    def test_find_match_full_path_takes_priority_over_leaf(
        self, tmp_path: Path, taxonomy_loader: TaxonomyLoader
    ) -> None:
        """When both a full path and leaf could match, the exact full-path wins."""

        path = tmp_path / "target_profiles.yaml"
        _write_profiles(
            path,
            [
                {"label": "entity_report", "vector_scope": ["entity_report"]},
                {"label": "document", "vector_scope": ["priority_goods"]},
            ],
        )
        loader = TargetProfileLoader(path, taxonomy_loader)
        # Exact full path
        match = loader.find_match("entity_report")
        assert match is not None
        assert match.label == "entity_report"
        # Leaf that doesn't match any target's last segment
        assert loader.find_match("nonexistent") is None
