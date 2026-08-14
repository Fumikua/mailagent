"""Unit tests for the Click-based CLI module.

Tests cover:
  - bootstrap seed invokes BootstrapPipeline.seed
  - bootstrap confirm --tier 2 --all exits with error
  - samples list --label filter
  - rules add --from-report parses and appends rules
  - interactive review keyboard sequence (mocked Prompt)
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import yaml
from click.testing import CliRunner
from rich.console import Console

from mailagent.infra.cli import _display_sample, _run_review, main
from mailagent.infra.config import Settings, VerticalSettings


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_report(
    stage: str = "import",
    samples: list[dict] | None = None,
) -> dict:
    """Build a minimal bootstrap report dict."""
    return {
        "job_id": "test123",
        "created_at": "2026-07-21T10:00:00+00:00",
        "stage": stage,
        "input_count": len(samples or []),
        "tiers": {
            "tier1": {"count": 0, "auto_confirmed": 0},
            "tier2": {"count": 0, "auto_confirmed": 0},
            "tier3": {"count": 0, "auto_confirmed": 0},
        },
        "samples": samples or [],
    }


def test_vertical_validate_json_reports_current_profile(
    monkeypatch,
) -> None:
    settings = Settings(
        vertical=VerticalSettings(
            id="example-triage",
            verticals_path=str(Path(__file__).parents[3] / "verticals"),
        )
    )
    monkeypatch.setattr("mailagent.infra.cli._build_settings", lambda: settings)

    result = CliRunner().invoke(main, ["vertical", "validate", "--json"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["valid"] is True
    assert {check["component"] for check in payload["checks"]} >= {
        "taxonomy",
        "rules",
    }


def _make_consistent_sample(sender: str, domain: str, label: str) -> dict:
    """Build a sample entry where rule and LLM agree (consistency=True).

    扁平 taxonomy：label 是单个 code（如 "schedule"），llm_match 的 l1/l2/l3
    均填该 code 以保持与 _label_dict_to_fields 的回退逻辑一致。
    """
    return {
        "id": str(uuid4()),
        "tier": "tier1",
        "subject": f"Email from {domain}",
        "sender": sender,
        "mail_hash": f"hash-{uuid4().hex[:12]}",
        "rule_match": {
            "label": label,
            "confidence": 0.95,
            "rule_type": "sender_domains",
            "matched_pattern": f"domain={domain}",
        },
        "llm_match": {
            "l1": label,
            "l2": label,
            "l3": label,
            "confidence": 0.95,
            "reasoning": "test",
        },
        "consistency": True,
        "action": "confirmed",
        "embedding_thread": [0.1, 0.2],
        "embedding_segment_0": [0.1, 0.2],
        "body_preview": "Body preview",
        "mail_event": {
            "message_id": f"<{uuid4().hex}@example.com>",
            "subject": f"Email from {domain}",
            "sender": sender,
            "body": "Full body text",
            "recipients": ["recipient@example.com"],
            "received_at": "2026-07-21T10:00:00+00:00",
        },
        "file_path": "/tmp/test.eml",
    }


def _write_evaluation_inputs(tmp_path: Path) -> tuple[Path, Path, object]:
    """Write metadata-only CLI fixtures plus an active vertical taxonomy."""
    vertical_root = tmp_path / "verticals"
    vertical_dir = vertical_root / "example_triage"
    vertical_dir.mkdir(parents=True)
    (vertical_dir / "manifest.yaml").write_text(
        "\n".join(
            [
                "id: example-triage",
                "namespace: example_triage",
                'data_schema_version: "1"',
                "taxonomy: taxonomy.yaml",
                "data_schema: data-schema.json",
                "runtime_factory: mailagent.verticals.runtime:build_empty_runtime",
            ]
        ),
        encoding="utf-8",
    )
    (vertical_dir / "taxonomy.yaml").write_text(
        "nodes:\n"
        "  - code: schedule\n"
        "    label: Schedule\n"
        "  - code: noise\n"
        "    label: Noise\n",
        encoding="utf-8",
    )
    (vertical_dir / "data-schema.json").write_text("{}", encoding="utf-8")

    manifest_path = tmp_path / "gold.yaml"
    manifest_path.write_text(
        "corpus_version: corpus-v1\n"
        "taxonomy_version: taxonomy-v1\n"
        "examples:\n"
        "  - sample_id: sample-1\n"
        "    thread_id: thread-1\n"
        "    labels: [schedule]\n"
        "    split: test\n"
        "    annotation_refs: [annotation-a, annotation-b]\n"
        "    adjudicated: true\n",
        encoding="utf-8",
    )
    predictions_path = tmp_path / "predictions.json"
    predictions_path.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample-1",
                    "labels": ["schedule"],
                    "needs_human_review": False,
                    "strategy": "rule_only",
                    "versions": {
                        "taxonomy": "taxonomy-v1",
                        "rules": "rules-v1",
                        "prompt": None,
                        "model": None,
                        "embedding": None,
                        "preprocessing": "preprocessing-v1",
                    },
                }
            ]
        ),
        encoding="utf-8",
    )
    settings = SimpleNamespace(
        vertical=SimpleNamespace(id="example-triage", verticals_path=str(vertical_root))
    )
    return manifest_path, predictions_path, settings


def _write_rules_cli_settings(tmp_path: Path) -> tuple[Path, object]:
    """Create one selected vertical with an active taxonomy and rules asset."""
    vertical_root = tmp_path / "verticals"
    vertical_dir = vertical_root / "example_triage"
    rules_dir = vertical_dir / "rules"
    rules_dir.mkdir(parents=True)
    (vertical_dir / "manifest.yaml").write_text(
        "\n".join(
            [
                "id: example-triage",
                "namespace: example_triage",
                'data_schema_version: "1"',
                "taxonomy: taxonomy.yaml",
                "data_schema: data-schema.json",
                "runtime_factory: mailagent.verticals.runtime:build_empty_runtime",
                "rules:",
                "  path: rules",
                '  version: "1"',
            ]
        ),
        encoding="utf-8",
    )
    (vertical_dir / "taxonomy.yaml").write_text(
        "nodes:\n"
        "  - code: schedule\n"
        "    label: Schedule\n",
        encoding="utf-8",
    )
    (vertical_dir / "data-schema.json").write_text("{}", encoding="utf-8")
    settings = SimpleNamespace(
        vertical=SimpleNamespace(id="example-triage", verticals_path=str(vertical_root)),
        # Deliberately point the legacy global setting elsewhere: rules add must
        # resolve the selected vertical's manifest-owned rules asset.
        rules=SimpleNamespace(rules_dir=str(tmp_path / "legacy-rules")),
    )
    return rules_dir, settings


# ---------------------------------------------------------------------------
# Test: bootstrap seed
# ---------------------------------------------------------------------------


class TestBootstrapSeed:
    def test_seed_invokes_pipeline_seed(self, tmp_path: Path) -> None:
        """bootstrap seed --dir calls pipeline.seed and prints the report ID."""
        eml_dir = tmp_path / "emails"
        eml_dir.mkdir()

        mock_pipeline = MagicMock()
        mock_pipeline.seed = AsyncMock(return_value="abc123def456")

        runner = CliRunner()
        with patch("mailagent.infra.cli._build_pipeline", return_value=mock_pipeline):
            result = runner.invoke(main, ["bootstrap", "seed", "--dir", str(eml_dir)])

        assert result.exit_code == 0
        mock_pipeline.seed.assert_awaited_once()
        # The dir argument should match the tmp_path.
        called_args = mock_pipeline.seed.call_args
        assert called_args.args[0] == eml_dir


# ---------------------------------------------------------------------------
# Test: bootstrap confirm --tier 2 --all
# ---------------------------------------------------------------------------


class TestBootstrapConfirm:
    def test_confirm_tier2_all_exits_with_error(self) -> None:
        """bootstrap confirm --tier 2 --all should exit(1) with validation message."""
        runner = CliRunner()
        result = runner.invoke(
            main,
            ["bootstrap", "confirm", "--report-id", "test", "--tier", "2", "--all"],
        )
        assert result.exit_code == 1
        assert "--all is only allowed for --tier 1" in result.output

    def test_confirm_tier1_all_succeeds(self) -> None:
        """bootstrap confirm --tier 1 --all should call pipeline.confirm_tier."""
        mock_pipeline = MagicMock()
        mock_pipeline.confirm_tier = AsyncMock(return_value=5)

        runner = CliRunner()
        with patch("mailagent.infra.cli._build_pipeline", return_value=mock_pipeline):
            result = runner.invoke(
                main,
                ["bootstrap", "confirm", "--report-id", "test", "--tier", "1", "--all"],
            )
        assert result.exit_code == 0
        mock_pipeline.confirm_tier.assert_awaited_once()
        assert "Confirmed 5 samples" in result.output


# ---------------------------------------------------------------------------
# Test: samples list --label
# ---------------------------------------------------------------------------


class TestSamplesList:
    def test_list_with_label_filter(self) -> None:
        """samples list --label calls store.get_samples with the label filter."""
        mock_store = MagicMock()
        mock_store.get_samples = AsyncMock(return_value=[])

        runner = CliRunner()
        with patch("mailagent.infra.cli._build_vector_store", return_value=mock_store):
            result = runner.invoke(
                main, ["samples", "list", "--label", "schedule"]
            )

        assert result.exit_code == 0
        mock_store.get_samples.assert_awaited_once()
        call_kwargs = mock_store.get_samples.call_args.kwargs
        assert call_kwargs.get("label") == "schedule"


class TestSamplesStats:
    def test_stats_calls_get_quality_stats(self) -> None:
        """samples stats invokes get_quality_stats and prints disposition counts."""
        mock_store = MagicMock()
        mock_store.count_samples = AsyncMock(return_value=5)
        mock_store.get_samples = AsyncMock(return_value=[])
        mock_store.get_quality_stats = AsyncMock(
            return_value={
                "by_disposition": {"accepted": 3, "rejected": 2},
                "by_taxonomy_schema": {"flat-v1": 5},
                "by_retrieval_policy": {"example-triage-v1": 5},
                "by_label_l1": {"schedule": 3, "operation": 2},
                "duplicate_fingerprint_rows": 0,
            }
        )

        runner = CliRunner()
        with patch("mailagent.infra.cli._build_vector_store", return_value=mock_store):
            result = runner.invoke(main, ["samples", "stats"])

        assert result.exit_code == 0
        mock_store.get_quality_stats.assert_awaited_once()
        assert "accepted" in result.output
        assert "flat-v1" in result.output


class TestSamplesReembed:
    def test_dry_run_reports_candidate_count_without_modifying_rows(self) -> None:
        """samples reembed --dry-run lists candidates and exits without write calls."""
        mock_store = MagicMock()
        mock_store.get_reembed_candidates = AsyncMock(
            return_value=[uuid4(), uuid4(), uuid4()]
        )
        mock_store.mark_reembed_complete = AsyncMock()

        runner = CliRunner()
        with patch("mailagent.infra.cli._build_vector_store", return_value=mock_store):
            result = runner.invoke(
                main,
                [
                    "samples",
                    "reembed",
                    "--policy-version",
                    "example-triage-v2",
                    "--dry-run",
                ],
            )

        assert result.exit_code == 0
        mock_store.get_reembed_candidates.assert_awaited_once()
        mock_store.mark_reembed_complete.assert_not_awaited()
        assert "Candidates in this batch" in result.output
        assert "3" in result.output
        assert "DRY RUN" in result.output

    def test_dry_run_with_no_candidates_reports_zero(self) -> None:
        """When no samples need re-embedding, dry-run reports 0 candidates."""
        mock_store = MagicMock()
        mock_store.get_reembed_candidates = AsyncMock(return_value=[])

        runner = CliRunner()
        with patch("mailagent.infra.cli._build_vector_store", return_value=mock_store):
            result = runner.invoke(
                main,
                [
                    "samples",
                    "reembed",
                    "--policy-version",
                    "example-triage-v1",
                    "--dry-run",
                ],
            )

        assert result.exit_code == 0
        assert "0" in result.output


class TestClassificationEvaluate:
    def test_classification_evaluate_writes_json_and_markdown(
        self, tmp_path: Path
    ) -> None:
        """Report-only evaluation writes both stable artifacts and exits zero."""
        manifest, predictions, settings = _write_evaluation_inputs(tmp_path)
        output_dir = tmp_path / "evaluation"

        runner = CliRunner()
        with patch("mailagent.infra.cli._build_settings", return_value=settings):
            result = runner.invoke(
                main,
                [
                    "classification",
                    "evaluate",
                    "--manifest",
                    str(manifest),
                    "--predictions",
                    str(predictions),
                    "--output-dir",
                    str(output_dir),
                ],
            )

        assert result.exit_code == 0, result.output
        json_path = output_dir / "classification-evaluation.json"
        markdown_path = output_dir / "classification-evaluation.md"
        assert json_path.exists()
        assert markdown_path.exists()
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["corpus_version"] == "corpus-v1"
        assert payload["evaluated_split"] == "test"
        assert payload["excluded_non_test_examples"] == 0
        assert payload["micro"]["precision"] == 1.0
        markdown = markdown_path.read_text(encoding="utf-8")
        assert "sample-1" not in markdown
        assert "Evaluated split: `test`" in markdown
        assert "Wilson 95% lower bound" in markdown
        assert "insufficient_volume" in result.output

    def test_classification_evaluate_enforces_failed_gates_only_when_requested(
        self, tmp_path: Path
    ) -> None:
        """Gate enforcement exits one after writing diagnostics; report mode does not."""
        manifest, predictions, settings = _write_evaluation_inputs(tmp_path)
        output_dir = tmp_path / "evaluation"

        runner = CliRunner()
        with patch("mailagent.infra.cli._build_settings", return_value=settings):
            result = runner.invoke(
                main,
                [
                    "classification",
                    "evaluate",
                    "--manifest",
                    str(manifest),
                    "--predictions",
                    str(predictions),
                    "--output-dir",
                    str(output_dir),
                    "--enforce-gates",
                ],
            )

        assert result.exit_code == 1
        assert (output_dir / "classification-evaluation.json").exists()
        assert (output_dir / "classification-evaluation.md").exists()
        assert "insufficient_volume" in result.output


# ---------------------------------------------------------------------------
# Test: rules add --from-report
# ---------------------------------------------------------------------------


class TestRulesAdd:
    def test_rules_add_from_report_applies_checked_markdown_once(
        self, tmp_path: Path
    ) -> None:
        """A checked Markdown proposal is added once and then deduplicated."""
        report_path = tmp_path / "rule_proposals.md"
        report_path.write_text(
            """- [x] 确认添加 (Confirm and append)

```yaml
- domain: confirmed.example
  label: schedule
  confidence: 0.92
```
""",
            encoding="utf-8",
        )

        rules_dir, mock_settings = _write_rules_cli_settings(tmp_path)

        runner = CliRunner()
        with patch("mailagent.infra.cli._build_settings", return_value=mock_settings):
            first = runner.invoke(
                main, ["rules", "add", "--from-report", str(report_path)]
            )
            second = runner.invoke(
                main, ["rules", "add", "--from-report", str(report_path)]
            )

        assert first.exit_code == 0
        assert "Added 1 rules" in first.output
        assert second.exit_code == 0
        assert "Added 0 rules" in second.output

        loaded = yaml.safe_load(
            (rules_dir / "sender_domains.yaml").read_text(encoding="utf-8")
        )
        assert loaded == [
            {"domain": "confirmed.example", "label": "schedule", "confidence": 0.92}
        ]

    def test_rules_add_rejects_invalid_checked_markdown_without_writing(
        self, tmp_path: Path
    ) -> None:
        """A checked malformed proposal is a usage error and preserves rules."""
        report_path = tmp_path / "rule_proposals.md"
        report_path.write_text(
            """- [x] 确认添加

```yaml
- domain: invalid.example
  label: schedule
  confidence: 2
```
""",
            encoding="utf-8",
        )
        rules_dir, mock_settings = _write_rules_cli_settings(tmp_path)
        rules_file = rules_dir / "sender_domains.yaml"
        original = "- domain: existing.example\n  label: schedule\n  confidence: 0.95\n"
        rules_file.write_text(original, encoding="utf-8")
        with patch("mailagent.infra.cli._build_settings", return_value=mock_settings):
            result = CliRunner().invoke(
                main, ["rules", "add", "--from-report", str(report_path)]
            )

        assert result.exit_code != 0
        assert rules_file.read_text(encoding="utf-8") == original

    def test_rules_add_ignores_unchecked_markdown_proposals(
        self, tmp_path: Path
    ) -> None:
        """Unchecked proposals remain review-only and produce no rules."""
        report_path = tmp_path / "rule_proposals.md"
        report_path.write_text(
            """- [ ] 确认添加

```yaml
- domain: pending.example
  label: schedule
  confidence: 0.92
```
""",
            encoding="utf-8",
        )
        rules_dir, mock_settings = _write_rules_cli_settings(tmp_path)

        with patch("mailagent.infra.cli._build_settings", return_value=mock_settings):
            result = CliRunner().invoke(
                main, ["rules", "add", "--from-report", str(report_path)]
            )

        assert result.exit_code == 0
        assert "Added 0 rules" in result.output
        assert not (rules_dir / "sender_domains.yaml").exists()

    def test_rules_add_rejects_inactive_label_and_preserves_file(
        self, tmp_path: Path
    ) -> None:
        """The selected vertical taxonomy gates the real Click write boundary."""
        report_path = tmp_path / "rule_proposals.md"
        report_path.write_text(
            """- [x] 确认添加

```yaml
- domain: invalid.example
  label: definitely_not_active
  confidence: 0.92
```
""",
            encoding="utf-8",
        )
        rules_dir, settings = _write_rules_cli_settings(tmp_path)
        rules_file = rules_dir / "sender_domains.yaml"
        original = b"- domain: existing.example\n  label: schedule\n  confidence: 0.95\n"
        rules_file.write_bytes(original)

        with patch("mailagent.infra.cli._build_settings", return_value=settings):
            result = CliRunner().invoke(
                main, ["rules", "add", "--from-report", str(report_path)]
            )

        assert result.exit_code != 0
        assert "absent from active taxonomy" in result.output
        assert rules_file.read_bytes() == original

    def test_rules_add_rejects_checked_section_without_yaml_fence(
        self, tmp_path: Path
    ) -> None:
        """A checked but truncated proposal is a controlled nonzero CLI result."""
        report_path = tmp_path / "rule_proposals.md"
        report_path.write_text(
            "- [x] 确认添加\n\nproposal truncated before YAML\n",
            encoding="utf-8",
        )
        rules_dir, settings = _write_rules_cli_settings(tmp_path)
        rules_file = rules_dir / "sender_domains.yaml"
        original = b"- domain: existing.example\n  label: schedule\n"
        rules_file.write_bytes(original)

        with patch("mailagent.infra.cli._build_settings", return_value=settings):
            result = CliRunner().invoke(
                main, ["rules", "add", "--from-report", str(report_path)]
            )

        assert result.exit_code != 0
        assert rules_file.read_bytes() == original


# ---------------------------------------------------------------------------
# Test: interactive review keyboard sequence
# ---------------------------------------------------------------------------


class TestInteractiveReview:
    def test_tier2_display_shows_both_suggestions_and_mandatory_review(self) -> None:
        """The review panel must expose disagreement before human confirmation."""
        sample = _make_consistent_sample(
            "ops@midconf.com", "midconf.com", "schedule"
        )
        sample["tier"] = "tier2"
        sample["llm_match"]["l3"] = "operation"
        sample["verification_status"] = "disagreed"
        review_console = Console(record=True, width=120)

        with patch("mailagent.infra.cli.console", review_console):
            _display_sample(sample, 1, 1)

        rendered = review_console.export_text()
        assert "Rule suggestion: schedule" in rendered
        assert "LLM suggestion:" in rendered
        assert "operation" in rendered
        assert "Verification: disagreed" in rendered
        assert "Mandatory individual review" in rendered

    def test_review_confirm_category_and_retrieval_edits_skip_discard_quit(self, tmp_path: Path) -> None:
        """Review persists edits through the re-embedding preparation path.

        Action sequence for 6 samples:
          1. Enter  → confirm (persist with original label)
          2. e + new_label → category edit (persist with human_fix source)
          3. r + retrieval text + reason → re-embed derived view only
          4. s      → skip (no persist)
          5. d      → discard (no persist, marked discarded)
          6. q      → quit (stops iteration)
        """
        samples = [
            _make_consistent_sample(
                f"ops@{d}.com", f"{d}.com", "schedule"
            )
            for d in ("s1", "s2", "s3", "s4", "s5", "s6")
        ]
        # Override tiers so all are in tier3 for this test.
        for s in samples:
            s["tier"] = "tier3"

        report = _make_report(stage="import", samples=samples)

        mock_pipeline = MagicMock()
        mock_pipeline._load_report = MagicMock(return_value=report)
        mock_pipeline._prepare_sample_for_persistence = AsyncMock(
            return_value=(MagicMock(mail_hash="hash-x"), [0.1, 0.2], [0.1, 0.2])
        )
        mock_pipeline._persist_sample = AsyncMock()

        # Prompt.ask side_effect: action, optional edit values, action, ...
        # Sample 1: "" (confirm)           → 1 call
        # Sample 2: "e" + "new_label" → 2 calls
        # Sample 3: "r" + text + reason    → 3 calls
        # Sample 4: "s" (skip)             → 1 call
        # Sample 5: "d" (discard)          → 1 call
        # Sample 6: "q" (quit)             → 1 call
        prompt_responses = [
            "",                          # s1: confirm
            "e",                         # s2: edit (action)
            "new_label",                 # s2: edit (new label)
            "r",                         # s3: retrieval text edit (action)
            "Subject: revised\nLatest message:\nKeep location request.",
            "preserve location request",    # s3: override reason
            "s",                         # s4: skip
            "d",                         # s5: discard
            "q",                         # s6: quit
        ]

        import asyncio
        from mailagent.infra import cli as cli_module

        with patch.object(cli_module.Prompt, "ask", side_effect=prompt_responses):
            asyncio.run(_run_review(mock_pipeline, "test123", "3"))

        # confirm (s1) + category edit (s2) + retrieval edit (s3) = 3.
        assert mock_pipeline._persist_sample.await_count == 3
        assert mock_pipeline._prepare_sample_for_persistence.await_count == 3

        # Sample 5 should be marked discarded in the report entry.
        assert samples[4]["action"] == "discarded"
        # Sample 4 should be marked skipped.
        assert samples[3]["action"] == "skipped"
        # Sample 1 should be marked confirmed.
        assert samples[0]["action"] == "confirmed"
        # Sample 2 should use one flat category code after the edit.
        assert samples[1]["action"] == "edited"
        assert samples[1]["llm_match"]["l1"] == "new_label"
        assert samples[1]["llm_match"]["l2"] is None
        assert samples[1]["llm_match"]["l3"] is None
        # Sample 3 retains raw mail data and stores the derived-text override.
        assert samples[2]["action"] == "retrieval_text_edited"
        assert samples[2]["retrieval_text_override"].endswith("Keep location request.")
        assert samples[2]["override_reason"] == "preserve location request"
