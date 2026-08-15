"""Unit tests for vector-similarity-path-b queue integration (Section 9 + 16).

Covers:
- 9.1 / 9.2: ``build_mail_understanding_pipeline`` feature-flag selection
  between ``FusionOrchestrator`` (Path B) and ``CascadeClassificationOrchestrator``
  (Path A), and component injection.
- 9.5: backward compatibility (default config walks the original Cascade path).
- 16.2 / 16.3 / 16.4: cron job placeholders and their schedule expressions.
- 16.6: cron registration in ``WorkerSettings`` and feature-flag switching.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Import mailagent.infra.* before mailagent.core to avoid a pre-existing
# circular import (infra.__init__ -> bootstrap -> core.rule_classifier ->
# preprocessing -> llm.embedding). When infra is imported first, core is
# fully initialized before preprocessing needs EmbeddingClient.
from mailagent.infra.config import (
    ClassificationSettings,
    FusionSettings,
    RulesSettings,
    Settings,
)
from mailagent.infra.queue import (
    ARCHIVE_JOB_NAME,
    CLEANUP_JOB_NAME,
    CLUSTERING_JOB_NAME,
    OUTBOX_DISPATCH_JOB_NAME,
    RULE_LEARN_JOB_NAME,
    WORKER_HEARTBEAT_JOB_NAME,
    archive_job,
    build_mail_understanding_pipeline,
    classify_job,
    clustering_job,
    cron_jobs,
    rule_learn_job,
)
from mailagent.core import CascadeClassificationOrchestrator, FusionOrchestrator
from mailagent.verticals.loader import (
    LoadedVertical,
    VerticalAsset,
    VerticalLLMSettings,
    VerticalManifest,
)
from mailagent.verticals.runtime import VerticalRuntime

_EXAMPLE_TRIAGE_RULES_DIR = (
    Path(__file__).parents[3] / "verticals" / "example_triage" / "rules"
)


def _make_fake_loaded() -> LoadedVertical:
    """Build a minimal LoadedVertical that skips enrichers."""
    return LoadedVertical(
        manifest=VerticalManifest(
            id="test-vertical",
            namespace="test",
            data_schema_version="1.0.0",
            taxonomy="taxonomy.yaml",
            data_schema="schema.json",
            runtime_factory="mailagent.verticals.runtime:build_empty_runtime",
            enrichers=[],
        ),
        taxonomy_path=Path("/nonexistent/taxonomy.yaml"),
        data_schema_path=Path("/nonexistent/schema.json"),
        data_schema={},
        rules=VerticalAsset(path=_EXAMPLE_TRIAGE_RULES_DIR, version="1"),
        rag=VerticalAsset(path=Path("/tmp/rag"), version="1"),
    )


def _patch_vertical_and_taxonomy():
    """Patch selected profile and TaxonomyLoader so the builder avoids file IO."""
    taxonomy_loader = MagicMock()
    taxonomy_loader.get_tree.return_value.all_codes.return_value = {
        "action_required",
        "notification",
        "noise",
    }
    stack = (
        patch(
            "mailagent.verticals.load_selected_vertical",
            return_value=SimpleNamespace(assets=_make_fake_loaded()),
        ),
        patch(
            "mailagent.classification.taxonomy.TaxonomyLoader",
            return_value=taxonomy_loader,
        ),
    )
    return stack


class TestBuildPipelineFeatureFlag:
    """9.1 / 9.2 / 9.5: feature flag toggles orchestrator; backward compat."""

    def test_default_fusion_disabled_uses_cascade(self) -> None:
        """Default config (fusion.enabled=False) walks the original Cascade path."""
        settings = Settings()
        p1, p2 = _patch_vertical_and_taxonomy()
        with p1, p2:
            pipeline = build_mail_understanding_pipeline(
                settings, MagicMock(), VerticalRuntime()
            )
        assert isinstance(pipeline._orchestrator, CascadeClassificationOrchestrator)

    def test_fusion_enabled_uses_fusion_orchestrator(self, tmp_path: Path) -> None:
        """fusion.enabled=True switches to FusionOrchestrator (Path B)."""
        settings = Settings(
            fusion=FusionSettings(enabled=True),
            rules=RulesSettings(rules_dir=str(tmp_path)),
        )
        p1, p2 = _patch_vertical_and_taxonomy()
        with p1, p2:
            pipeline = build_mail_understanding_pipeline(
                settings,
                MagicMock(),
                VerticalRuntime(),
                embedding_client=MagicMock(),
                vector_store=MagicMock(),
            )
        assert isinstance(pipeline._orchestrator, FusionOrchestrator)

    def test_fusion_injects_all_three_classifiers(self, tmp_path: Path) -> None:
        """When embedding_client + vector_store are provided, all three paths register."""
        settings = Settings(
            fusion=FusionSettings(enabled=True),
            rules=RulesSettings(rules_dir=str(tmp_path)),
        )
        p1, p2 = _patch_vertical_and_taxonomy()
        with p1, p2:
            pipeline = build_mail_understanding_pipeline(
                settings,
                MagicMock(),
                VerticalRuntime(),
                embedding_client=MagicMock(),
                vector_store=MagicMock(),
            )
        classifiers = pipeline._orchestrator._classifiers
        assert set(classifiers.keys()) == {"rules", "vector", "llm"}
        assert classifiers["rules"]._taxonomy_loader is not None
        assert pipeline._version_provider._rule_classifier is classifiers["rules"]

    def test_fusion_degrades_without_embedding_and_vector_store(
        self, tmp_path: Path
    ) -> None:
        """Missing embedding/vector_store degrades to rule + llm only (no crash)."""
        settings = Settings(
            fusion=FusionSettings(enabled=True),
            rules=RulesSettings(rules_dir=str(tmp_path)),
        )
        p1, p2 = _patch_vertical_and_taxonomy()
        with p1, p2:
            pipeline = build_mail_understanding_pipeline(
                settings, MagicMock(), VerticalRuntime()
            )
        classifiers = pipeline._orchestrator._classifiers
        assert "vector" not in classifiers
        assert set(classifiers.keys()) == {"rules", "llm"}

    def test_backward_compat_signature_accepts_three_args(self) -> None:
        """Original 3-arg callers (no embedding/vector_store) still work with Cascade."""
        settings = Settings()
        p1, p2 = _patch_vertical_and_taxonomy()
        with p1, p2:
            pipeline = build_mail_understanding_pipeline(
                settings, MagicMock(), VerticalRuntime()
            )
        assert isinstance(pipeline._orchestrator, CascadeClassificationOrchestrator)

    def test_pipeline_wires_auto_accept_setting(self) -> None:
        settings = Settings(
            classification=ClassificationSettings(auto_accept_enabled=True)
        )
        p1, p2 = _patch_vertical_and_taxonomy()
        with p1, p2:
            pipeline = build_mail_understanding_pipeline(
                settings, MagicMock(), VerticalRuntime()
            )

        assert pipeline._auto_accept_enabled is True
        assert pipeline._version_provider is not None

    def test_fusion_without_target_profiles_does_not_error(
        self, tmp_path: Path
    ) -> None:
        """5.6: target_profiles.yaml missing (loaded.target_profiles=None) → no error,
        orchestrator gets target_profile_loader=None (feature off).
        """
        settings = Settings(
            fusion=FusionSettings(enabled=True),
            rules=RulesSettings(rules_dir=str(tmp_path)),
        )
        p1, p2 = _patch_vertical_and_taxonomy()
        with p1, p2:
            pipeline = build_mail_understanding_pipeline(
                settings,
                MagicMock(),
                VerticalRuntime(),
                embedding_client=MagicMock(),
                vector_store=MagicMock(),
            )
        assert isinstance(pipeline._orchestrator, FusionOrchestrator)
        # target_profile_loader defaults to None when target_profiles asset absent
        assert pipeline._orchestrator._target_profile_loader is None

    def test_fusion_with_target_profiles_constructs_loader(
        self, tmp_path: Path
    ) -> None:
        """5.6: when loaded.target_profiles is set, a TargetProfileLoader is wired in."""
        # Write a minimal target_profiles.yaml + taxonomy for validation.
        target_profiles_path = tmp_path / "target_profiles.yaml"
        target_profiles_path.write_text(
            "targets:\n"
            "  - label: entity_report\n"
            "    vector_scope:\n"
            "      - entity_report\n",
            encoding="utf-8",
        )
        # Minimal taxonomy containing the target label.
        taxonomy_path = tmp_path / "taxonomy.yaml"
        taxonomy_path.write_text(
            "nodes:\n"
            "  - code: entity_report\n"
            "    label: 实体报告\n"
            "    description: x\n",
            encoding="utf-8",
        )
        rules_path = tmp_path / "rules"
        rules_path.mkdir()
        for filename in (
            "sender_domains.yaml",
            "subject_patterns.yaml",
            "body_keywords.yaml",
            "structural.yaml",
        ):
            (rules_path / filename).write_text("[]", encoding="utf-8")
        loaded = LoadedVertical(
            manifest=VerticalManifest(
                id="test-vertical",
                namespace="test",
                data_schema_version="1.0.0",
                taxonomy="taxonomy.yaml",
                data_schema="schema.json",
                runtime_factory="mailagent.verticals.runtime:build_empty_runtime",
                enrichers=[],
            ),
            taxonomy_path=taxonomy_path,
            data_schema_path=Path("/nonexistent/schema.json"),
            data_schema={},
            rules=VerticalAsset(path=rules_path, version="1"),
            rag=VerticalAsset(path=Path("/tmp/rag"), version="1"),
            target_profiles=VerticalAsset(path=target_profiles_path, version="1"),
        )
        settings = Settings(
            fusion=FusionSettings(enabled=True),
            rules=RulesSettings(rules_dir=str(tmp_path)),
        )
        with patch("mailagent.verticals.load_vertical", return_value=loaded):
            pipeline = build_mail_understanding_pipeline(
                settings,
                MagicMock(),
                VerticalRuntime(),
                embedding_client=MagicMock(),
                vector_store=MagicMock(),
                loaded_vertical=loaded,
            )
        assert isinstance(pipeline._orchestrator, FusionOrchestrator)
        loader = pipeline._orchestrator._target_profile_loader
        assert loader is not None
        targets = loader.get_targets()
        assert len(targets) == 1
        assert targets[0].label == "entity_report"

    def test_pipeline_honors_vertical_llm_and_retrieval_dependencies(
        self, tmp_path: Path
    ) -> None:
        taxonomy_path = tmp_path / "taxonomy.yaml"
        taxonomy_path.write_text(
            "nodes:\n"
            "  - code: entity_report\n"
            "    label: 实体报告\n"
            "    description: x\n",
            encoding="utf-8",
        )
        loaded = LoadedVertical(
            manifest=VerticalManifest(
                id="test-vertical",
                namespace="test",
                data_schema_version="1",
                taxonomy="taxonomy.yaml",
                data_schema="schema.json",
                runtime_factory="mailagent.verticals.runtime:build_empty_runtime",
                llm=VerticalLLMSettings(body_max_chars=4000),
                enrichers=[],
            ),
            taxonomy_path=taxonomy_path,
            data_schema_path=tmp_path / "schema.json",
            data_schema={},
            rag=VerticalAsset(path=tmp_path, version="1"),
        )
        cleaning_policy = MagicMock()
        preprocessing_extension = MagicMock()
        runtime = VerticalRuntime(
            retrieval_cleaning_policy=cleaning_policy,
            preprocessing_extension=preprocessing_extension,
        )
        settings = Settings(fusion=FusionSettings(enabled=True))

        pipeline = build_mail_understanding_pipeline(
            settings,
            MagicMock(),
            runtime,
            embedding_client=MagicMock(),
            vector_store=MagicMock(),
            loaded_vertical=loaded,
        )

        classifiers = pipeline._orchestrator._classifiers
        assert classifiers["llm"].body_max_chars == 4000
        assert classifiers["vector"]._cleaning_policy is cleaning_policy
        assert classifiers["vector"]._preprocessing_extension is preprocessing_extension
        assert classifiers["vector"]._taxonomy_loader is not None
        provider = pipeline._version_provider
        assert provider is not None
        assert provider._taxonomy_loader is classifiers["llm"].taxonomy
        assert provider._rule_classifier is None
        assert provider._preprocessing_extension is preprocessing_extension
        assert provider._retrieval_cleaning_policy is cleaning_policy


class TestCronJobsSchedule:
    """16.6: cron expressions are correct for each job."""

    def test_six_cron_jobs_registered(self) -> None:
        assert {job.name for job in cron_jobs} == {
            WORKER_HEARTBEAT_JOB_NAME,
            OUTBOX_DISPATCH_JOB_NAME,
            CLUSTERING_JOB_NAME,
            RULE_LEARN_JOB_NAME,
            ARCHIVE_JOB_NAME,
            CLEANUP_JOB_NAME,
        }

    def test_clustering_job_schedule(self) -> None:
        job = next(job for job in cron_jobs if job.name == CLUSTERING_JOB_NAME)
        assert job.name == CLUSTERING_JOB_NAME
        assert job.weekday == "sun"
        assert job.hour == 2
        assert job.minute == 0

    def test_rule_learn_job_schedule(self) -> None:
        job = next(job for job in cron_jobs if job.name == RULE_LEARN_JOB_NAME)
        assert job.name == RULE_LEARN_JOB_NAME
        assert job.weekday == "sun"
        assert job.hour == 3
        assert job.minute == 0

    def test_archive_job_schedule(self) -> None:
        job = next(job for job in cron_jobs if job.name == ARCHIVE_JOB_NAME)
        assert job.name == ARCHIVE_JOB_NAME
        assert job.day == 1
        assert job.hour == 2
        assert job.minute == 0
        assert job.weekday is None

    def test_cleanup_job_schedule(self) -> None:
        job = next(job for job in cron_jobs if job.name == CLEANUP_JOB_NAME)
        assert job.name == CLEANUP_JOB_NAME
        assert job.hour == 3
        assert job.minute == 30
        assert job.weekday is None
        assert job.day is None


class TestCronJobsCallable:
    """16.2–16.6: cron jobs invoke their startup-provided services."""

    async def test_clustering_job_calls_context_engine(self) -> None:
        engine = AsyncMock()
        engine.run_weekly_clustering.return_value = "reports/intent.md"

        assert (
            await clustering_job({"clustering_engine": engine}) == "reports/intent.md"
        )
        engine.run_weekly_clustering.assert_awaited_once_with()

    async def test_rule_learn_job_calls_context_engine(self) -> None:
        learner = AsyncMock()
        learner.run_weekly_scan.return_value = "reports/rules.md"

        assert await rule_learn_job({"rule_learner": learner}) == "reports/rules.md"
        learner.run_weekly_scan.assert_awaited_once_with()

    async def test_archive_job_calls_context_pipeline(self) -> None:
        pipeline = AsyncMock()
        pipeline.archive_old_samples.return_value = 3
        settings = Settings()
        settings.vector_store.archive_window_months = 18

        assert (
            await archive_job({"bootstrap_pipeline": pipeline, "settings": settings})
            == "archived: 3"
        )
        pipeline.archive_old_samples.assert_awaited_once_with(18)

    @pytest.mark.parametrize("job", [clustering_job, rule_learn_job, archive_job])
    async def test_jobs_explicitly_skip_when_fusion_is_disabled(self, job) -> None:
        ctx = {"settings": Settings(fusion=FusionSettings(enabled=False))}

        assert await job(ctx) == "skipped: fusion disabled"


class TestWorkerSettingsRegistration:
    """16.6: cron_jobs registered in WorkerSettings; classify_job unchanged."""

    def test_worker_settings_has_cron_jobs(self) -> None:
        from mailagent.infra.worker import WorkerSettings

        # The three base cron jobs (clustering/rule_learn/archive) are always
        # registered; a mail_poll cron is appended when a gateway is enabled.
        # Compare by name because arq CronJob does not define __eq__.
        actual_names = [job.name for job in WorkerSettings.cron_jobs]
        for base_job in cron_jobs:
            assert base_job.name in actual_names
        assert len(WorkerSettings.cron_jobs) >= 3

    def test_worker_settings_functions_unchanged(self) -> None:
        """classify_job is always the primary foreground function; mail_poll_job
        is appended when a gateway is enabled (16.1 no-op check on the fusion flag)."""
        from mailagent.infra.worker import WorkerSettings

        assert WorkerSettings.functions[0] is classify_job
        if len(WorkerSettings.functions) > 1:
            from mailagent.gateway.runner import mail_poll_job

            assert WorkerSettings.functions[1] is mail_poll_job

    def test_classify_job_is_pipeline_agnostic(self) -> None:
        """16.1: classify_job does not branch on the fusion flag — it consumes the
        pipeline opaquely. Verify the function references the pipeline, not a flag.
        """
        import inspect

        source = inspect.getsource(classify_job)
        assert "mail_understanding_pipeline" in source
        assert "fusion" not in source
