"""End-to-end live-LLM test for Path B (Section 18.10).

Runs ONLY when the environment variable ``MAILAGENT_LIVE_LLM=1`` is set,
otherwise the entire module is skipped. When enabled, exercises the real TEI
embedding service and LongCat-2.0 LLM against a small synthetic corpus.

Required environment variables when enabled:
  - ``MAILAGENT_LIVE_LLM=1``
  - ``OPENAI_API_KEY`` (or whatever LLM_API_KEY_ENV the deployment uses)
  - ``MAILAGENT_MODEL__BASE_URL`` / ``MAILAGENT_MODEL__MODEL_NAME``
  - ``MAILAGENT_EMBEDDING__API_BASE``

The test is intentionally light on assertions — its primary purpose is to
provide a runnable smoke test for the live Path B stack, not to assert
specific LLM outputs.
"""
from __future__ import annotations

import os
from email.message import EmailMessage
from pathlib import Path

import pytest

_LIVE_LLM_ENABLED = os.getenv("MAILAGENT_LIVE_LLM", "0") == "1"

# Skip the entire module when live LLM is not enabled.
pytestmark = pytest.mark.skipif(
    not _LIVE_LLM_ENABLED,
    reason="Requires MAILAGENT_LIVE_LLM=1 plus live TEI + LongCat-2.0 endpoints",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_eml(path: Path, subject: str, sender: str = "ops@example.com") -> None:
    """Write a minimal .eml file."""
    msg = EmailMessage()
    msg["From"] = sender
    msg["To"] = "agency@port.com"
    msg["Subject"] = subject
    msg["Date"] = "Mon, 21 Jul 2026 10:00:00 +0000"
    msg["Message-ID"] = f"<{path.stem}@example.com>"
    msg.set_content(f"Body content for {subject}.")
    path.write_bytes(bytes(msg))


@pytest.fixture
def live_eml_corpus(tmp_path: Path) -> Path:
    """Create a small directory of .eml files for live Path B processing.

    The directory ships with up to 24 synthetic .eml files mirroring real
    business patterns (Example STATUS, Status Report, Location Request, etc.).
    """
    cases = [
        "Berlin Example STATUS Update",
        "Status Report Jul 21",
        "Location Request Shanghai",
        "Re[2]: Berlin Example STATUS Update",
        "Staff Update Confirmation",
        "DG Compliance Certificate",
        "Safety Notice Maritime",
        "Re: Status Report",
        "MSC Geneva STATUS Singapore",
        "CMA CGM Antoine Schedule",
        "Reply: Location Plan Update",
        "Forward: Crew List",
    ]
    for i, subject in enumerate(cases):
        _write_eml(tmp_path / f"eml_{i:02d}.eml", subject)
    return tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPathBLiveLLM:
    """Live TEI + LongCat-2.0 end-to-end on a synthetic .eml corpus."""

    async def test_live_seed_runs_full_pipeline(self, live_eml_corpus: Path) -> None:
        """Stage 1 seed runs end-to-end against live TEI + LLM services."""
        # Imports are deferred so the module can be collected without the
        # optional live dependencies being installed.
        from sqlalchemy.ext.asyncio import create_async_engine
        from sqlalchemy.pool import StaticPool

        from mailagent.domain.models import RuleResult
        from mailagent.infra.bootstrap import BootstrapPipeline
        from mailagent.infra.config import BootstrapSettings, VectorStoreSettings
        from mailagent.infra.store import Base
        from mailagent.infra.vector_store import VectorStore
        from mailagent.llm.client import LLMClient
        from mailagent.llm.embedding import EmbeddingClient
        from mailagent.infra.config import Settings

        # Load live settings from environment / config file.
        settings = Settings.from_yaml("config.yml")
        embedding_settings = settings.embedding
        model_settings = settings.model

        # Build live clients.
        embedding_client = EmbeddingClient(embedding_settings)
        llm_client = LLMClient(
            base_url=model_settings.base_url,
            api_key=os.getenv(model_settings.api_key_env, ""),
            model=model_settings.model_name,
        )

        # Wrap LLMClient into a Classifier-compatible adapter by reusing the
        # existing LLMClassifier (Path A) — its classify() returns an attempt.
        from mailagent.classification.llm_classifier import LLMClassifier

        llm_classifier = LLMClassifier(
            llm_client=llm_client,
            taxonomy_path=settings.classification.taxonomy_path,
            model_name=model_settings.model_name,
        )

        # In-memory SQLite for live run (no persistence between runs).
        eng = create_async_engine(
            "sqlite+aiosqlite:///:memory:",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        async with eng.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        vector_store = VectorStore(VectorStoreSettings(), eng)

        rule_clf = type("_EmptyRules", (), {})()  # noqa: ANN202
        rule_clf._sender_domain_rules = []
        rule_clf._subject_pattern_rules = []
        rule_clf._body_keyword_rules = []
        rule_clf._structural_rules = []
        rule_clf.match = lambda *a, **k: RuleResult(matches=[], selected=None)

        bootstrap_settings = BootstrapSettings(
            weekly_batch_size=4200,
            default_batch_size=50,
            reports_dir=str(Path("./reports")),
        )

        pipeline = BootstrapPipeline(
            rule_classifier=rule_clf,
            llm_classifier=llm_classifier,
            vector_store=vector_store,
            embedding_client=embedding_client,
            settings=bootstrap_settings,
        )

        # Stage 1 seed on the live corpus.
        report_id = await pipeline.seed(
            live_eml_corpus, force=True, no_rules=True
        )

        assert isinstance(report_id, str)
        assert len(report_id) == 12

        # Confirm tier 3 (all live-LLM annotated) — should persist samples.
        count = await pipeline.confirm_tier(report_id, tier=3, all_=False)
        assert count > 0

        # Verify samples landed in the DB with embeddings.
        samples = await vector_store.get_samples()
        assert len(samples) > 0

        await eng.dispose()
