"""Real Click-boundary integration for Bootstrap stage-2 workflows."""
from __future__ import annotations

import asyncio
import json
import re
from collections import Counter
from dataclasses import dataclass
from email.message import EmailMessage
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from click.testing import CliRunner
from rich.prompt import Prompt
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine
from sqlalchemy.pool import NullPool

from mailagent.infra.bootstrap import BootstrapPipeline
from mailagent.infra.cli import main
from mailagent.infra.config import BootstrapSettings, VectorStoreSettings
from mailagent.infra.store import Base
from mailagent.infra.vector_store import VectorStore
from mailagent.classification import AttemptStatus, ClassificationAttempt
from mailagent.classification.rule_classifier import RuleClassifier
from mailagent.domain.models import SampleRecord, TaxonomyLabel
from mailagent.llm.embedding import EmbeddingClient
from mailagent.classification.taxonomy import TaxonomyLoader

_DIMENSION = 8


@dataclass(frozen=True)
class _CliBootstrapHarness:
    pipeline: BootstrapPipeline
    vector_store: VectorStore
    reports_dir: Path
    eml_dir: Path


def _write_eml(path: Path, sender: str, subject: str, body: str) -> None:
    message = EmailMessage()
    message["From"] = sender
    message["To"] = "recipient@example.com"
    message["Subject"] = subject
    message["Date"] = "Mon, 21 Jul 2026 10:00:00 +0000"
    message["Message-ID"] = f"<{uuid4().hex}@example.com>"
    message.set_content(body)
    path.write_bytes(bytes(message))


async def _create_schema(engine: AsyncEngine) -> None:
    async with engine.begin() as connection:
        await connection.run_sync(Base.metadata.create_all)


async def _dispose(engine: AsyncEngine) -> None:
    await engine.dispose()


async def _embedding_batch(texts: list[str]) -> list[list[float]]:
    return [
        [0.1 * (index + 1) + offset * 0.01 for offset in range(_DIMENSION)]
        for index, _text in enumerate(texts)
    ]


def _flat_llm_attempt() -> ClassificationAttempt:
    return ClassificationAttempt(
        source="llm",
        status=AttemptStatus.SUCCESS,
        labels=[
            TaxonomyLabel(
                l1_code="schedule",
                l1_label="Schedule",
                confidence=0.88,
                reasoning="synthetic flat-category classification",
            )
        ],
        confidence=0.88,
    )


@pytest.fixture
def cli_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[_CliBootstrapHarness]:
    """Real pipeline whose only test doubles are external model seams."""
    taxonomy_path = tmp_path / "taxonomy.yaml"
    taxonomy_path.write_text(
        "nodes:\n"
        "  - code: schedule\n"
        "    label: Schedule\n"
        "  - code: operation\n"
        "    label: Operation\n",
        encoding="utf-8",
    )
    taxonomy_loader = TaxonomyLoader(taxonomy_path)

    rules_dir = tmp_path / "rules"
    rules_dir.mkdir()
    (rules_dir / "sender_domains.yaml").write_text(
        "- domain: highconf.com\n"
        "  label: schedule\n"
        "  confidence: 0.95\n",
        encoding="utf-8",
    )
    (rules_dir / "subject_patterns.yaml").write_text(
        "- pattern: Location Plan\n"
        "  label: operation\n"
        "  confidence: 0.80\n",
        encoding="utf-8",
    )
    for filename in ("body_keywords.yaml", "structural.yaml"):
        (rules_dir / filename).write_text("[]\n", encoding="utf-8")

    database_path = tmp_path / "bootstrap.sqlite3"
    engine = create_async_engine(
        f"sqlite+aiosqlite:///{database_path}",
        poolclass=NullPool,
    )
    asyncio.run(_create_schema(engine))
    vector_store = VectorStore(VectorStoreSettings(), engine)

    embedding_client = MagicMock(spec=EmbeddingClient)
    embedding_client.embed_batch = AsyncMock(side_effect=_embedding_batch)
    llm_classifier = SimpleNamespace(
        source="llm",
        classify=AsyncMock(return_value=_flat_llm_attempt()),
    )
    reports_dir = tmp_path / "reports"
    pipeline = BootstrapPipeline(
        rule_classifier=RuleClassifier(rules_dir, taxonomy_loader),
        llm_classifier=llm_classifier,
        vector_store=vector_store,
        embedding_client=embedding_client,
        settings=BootstrapSettings(reports_dir=str(reports_dir)),
        taxonomy_loader=taxonomy_loader,
    )
    monkeypatch.setattr("mailagent.infra.cli._build_pipeline", lambda: pipeline)

    eml_dir = tmp_path / "emails"
    eml_dir.mkdir()
    _write_eml(
        eml_dir / "tier1.eml",
        "ops@highconf.com",
        "STATUS Update",
        "The entity schedule has a confirmed arrival update for tomorrow.",
    )
    _write_eml(
        eml_dir / "tier2.eml",
        "ops@midconf.com",
        "Location Plan",
        "Please arrange the location operation and confirm the assigned team.",
    )
    _write_eml(
        eml_dir / "tier3.eml",
        "ops@unmatched.com",
        "General Inquiry",
        "Please review this unmatched operational inquiry and advise next steps.",
    )

    yield _CliBootstrapHarness(pipeline, vector_store, reports_dir, eml_dir)
    asyncio.run(_dispose(engine))


def _sample_count(harness: _CliBootstrapHarness) -> int:
    return asyncio.run(harness.vector_store.count_samples())


def _stored_samples(harness: _CliBootstrapHarness) -> list[SampleRecord]:
    return asyncio.run(harness.vector_store.get_samples())


def test_bootstrap_import_confirm_and_review_through_real_cli(
    cli_bootstrap: _CliBootstrapHarness,
) -> None:
    """Click commands preserve tier evidence and enforce reviewed persistence."""
    runner = CliRunner()

    import_result = runner.invoke(
        main,
        [
            "bootstrap",
            "import",
            "--dir",
            str(cli_bootstrap.eml_dir),
            "--batch-size",
            "2",
        ],
    )
    assert import_result.exit_code == 0, import_result.output
    report_id_match = re.search(r"Report ID: ([0-9a-f]{12})", import_result.output)
    assert report_id_match is not None
    report_id = report_id_match.group(1)

    report_path = cli_bootstrap.reports_dir / f"bootstrap_{report_id}.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["stage"] == "import"
    assert report["input_count"] == 3
    assert Counter(sample["tier"] for sample in report["samples"]) == {
        "tier1": 1,
        "tier2": 1,
        "tier3": 1,
    }
    tier2 = next(sample for sample in report["samples"] if sample["tier"] == "tier2")
    assert tier2["rule_match"]["label"] == "operation"
    assert tier2["llm_match"]["l1"] == "schedule"
    assert tier2["verification_status"] == "disagreed"
    assert _sample_count(cli_bootstrap) == 0

    confirm_args = [
        "bootstrap",
        "confirm",
        "--report-id",
        report_id,
        "--tier",
        "1",
        "--all",
    ]
    first_confirm = runner.invoke(main, confirm_args)
    assert first_confirm.exit_code == 0, first_confirm.output
    assert _sample_count(cli_bootstrap) == 1

    second_confirm = runner.invoke(main, confirm_args)
    assert second_confirm.exit_code == 0, second_confirm.output
    assert _sample_count(cli_bootstrap) == 1

    rejected_confirm = runner.invoke(
        main,
        [
            "bootstrap",
            "confirm",
            "--report-id",
            report_id,
            "--tier",
            "2",
        ],
    )
    assert rejected_confirm.exit_code == 1
    assert "Tier 2 requires individual review" in rejected_confirm.output
    assert _sample_count(cli_bootstrap) == 1

    with patch.object(Prompt, "ask", return_value=""):
        review_result = runner.invoke(
            main,
            [
                "bootstrap",
                "review",
                "--report-id",
                report_id,
                "--tier",
                "2",
            ],
        )
    assert review_result.exit_code == 0, review_result.output
    samples = _stored_samples(cli_bootstrap)
    assert len(samples) == 2
    reviewed_tier2 = next(sample for sample in samples if sample.source == "rule_tier2")
    assert reviewed_tier2.label_l1 == "operation"
    assert reviewed_tier2.reviewed is True
    assert reviewed_tier2.taxonomy_schema_version == "flat-v1"
