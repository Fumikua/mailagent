"""Unit tests for vector-similarity-path-b config settings."""
from __future__ import annotations


from mailagent.infra.config import (
    BootstrapSettings,
    ClusteringSettings,
    ClassificationFeedbackSettings,
    EmbeddingSettings,
    FusionSettings,
    RulesSettings,
    Settings,
    VectorStoreSettings,
)


class TestEmbeddingSettings:
    def test_defaults(self):
        s = EmbeddingSettings()
        assert s.provider == "tei"
        assert s.model_name == "Qwen3-Embedding-8B"
        assert s.dimension == 4096
        assert s.timeout == 30

    def test_custom_values(self):
        s = EmbeddingSettings(api_base="http://tei:9090", dimension=768)
        assert s.api_base == "http://tei:9090"
        assert s.dimension == 768


class TestVectorStoreSettings:
    def test_defaults(self):
        s = VectorStoreSettings()
        assert s.top_k == 5
        assert s.similarity_threshold == 0.85
        assert s.archive_window_months == 12
        assert s.stratified_sample_threshold == 50000


class TestFusionSettings:
    def test_defaults(self):
        s = FusionSettings()
        assert s.rule_confidence_threshold == 0.9
        assert s.vector_confidence_threshold == 0.85
        assert s.enabled is False  # backward compat feature flag

    def test_enabled_flag(self):
        s = FusionSettings(enabled=True)
        assert s.enabled is True


class TestRulesSettings:
    def test_defaults(self):
        s = RulesSettings()
        assert s.rules_dir == "./verticals/rules"
        assert s.enable_autolearn is True
        assert s.autolearn_min_samples == 5
        assert s.autolearn_min_ratio == 0.8


class TestClusteringSettings:
    def test_defaults(self):
        s = ClusteringSettings()
        assert s.min_cluster_size == 5
        assert s.min_samples == 3
        assert s.metric == "cosine"
        assert s.max_samples == 50000
        assert s.window_days == 30


class TestBootstrapSettings:
    def test_defaults(self):
        s = BootstrapSettings()
        assert s.weekly_batch_size == 4200
        assert s.default_batch_size == 50
        assert s.reports_dir == "./reports"


class TestSettingsAggregation:
    def test_all_new_settings_in_main_settings(self):
        s = Settings()
        assert isinstance(s.embedding, EmbeddingSettings)
        assert isinstance(s.vector_store, VectorStoreSettings)
        assert isinstance(s.fusion, FusionSettings)
        assert isinstance(s.rules, RulesSettings)
        assert isinstance(s.clustering, ClusteringSettings)
        assert isinstance(s.bootstrap, BootstrapSettings)
        assert isinstance(s.classification_feedback, ClassificationFeedbackSettings)
        assert s.classification_feedback.mode == "disabled"

    def test_env_var_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("MAILAGENT_FUSION__ENABLED", "true")
        monkeypatch.setenv("MAILAGENT_EMBEDDING__DIMENSION", "768")
        monkeypatch.setenv(
            "MAILAGENT_CLASSIFICATION_FEEDBACK__MODE", "trusted_internal"
        )
        s = Settings()
        assert s.fusion.enabled is True
        assert s.embedding.dimension == 768
        assert s.classification_feedback.mode == "trusted_internal"


# pytest fixture needed for monkeypatch
import pytest  # noqa: E402
