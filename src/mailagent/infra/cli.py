"""Mailagent CLI — Click-based command interface.

Subcommands:
  mailagent vertical  validate
  mailagent bootstrap {seed|import|review|confirm|archive}
  mailagent samples   {list|delete|fix|audit|stats}
  mailagent rules     {add}

Factory functions ``_build_pipeline`` and ``_build_vector_store`` are
module-level so tests can monkeypatch them without touching real I/O.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from mailagent.infra.bootstrap import BootstrapPipeline
from mailagent.infra.config import Settings
from mailagent.infra.rule_learner import append_confirmed_sender_domain_rules
from mailagent.infra.vector_store import VectorStore

console = Console()


# ---------------------------------------------------------------------------
# Factory functions (patch these in tests)
# ---------------------------------------------------------------------------


def _build_settings() -> Settings:
    """Load settings from config.yml or environment."""
    return Settings.from_yaml()


def _build_pipeline() -> BootstrapPipeline:
    """Construct a BootstrapPipeline from settings.

    Tests should monkeypatch this function to inject a mock pipeline.
    """
    settings = _build_settings()

    from mailagent.classification.llm_classifier import LLMClassifier
    from mailagent.classification.rule_classifier import RuleClassifier
    from mailagent.llm.client import LLMClient
    from mailagent.llm.embedding import EmbeddingClient

    rules_dir = Path(settings.rules.rules_dir)
    rule_classifier = RuleClassifier(rules_dir)

    api_key = os.getenv(settings.model.api_key_env, "")
    llm_client = LLMClient(
        base_url=settings.model.base_url,
        api_key=api_key,
        model=settings.model.model_name,
    )

    taxonomy_loader: Any | None = None
    try:
        from mailagent.classification.taxonomy import TaxonomyLoader

        taxonomy_loader = TaxonomyLoader(settings.classification.taxonomy_path)
        llm_classifier: Any = LLMClassifier(llm_client, taxonomy_loader, settings.model.model_name)
    except Exception:
        llm_classifier = LLMClassifier(llm_client, None, settings.model.model_name)  # type: ignore[arg-type]

    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(settings.database.url)
    vector_store = VectorStore(settings.vector_store, engine)

    embedding_client = EmbeddingClient(settings.embedding)

    return BootstrapPipeline(
        rule_classifier=rule_classifier,
        llm_classifier=llm_classifier,
        vector_store=vector_store,
        embedding_client=embedding_client,
        settings=settings.bootstrap,
        taxonomy_loader=taxonomy_loader,
    )


def _build_vector_store() -> VectorStore:
    """Construct a VectorStore from settings.

    Tests should monkeypatch this function to inject a mock store.
    """
    settings = _build_settings()
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(settings.database.url)
    return VectorStore(settings.vector_store, engine)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


@click.group()
def main() -> None:
    """Mailagent CLI — bootstrap, samples, and rules management."""


# ---------------------------------------------------------------------------
# vertical subgroup
# ---------------------------------------------------------------------------


@main.group()
def vertical() -> None:
    """Validate and inspect installed vertical business profiles."""


@vertical.command("validate")
@click.option(
    "--json",
    "json_output",
    is_flag=True,
    help="Emit a machine-readable validation report.",
)
def vertical_validate(json_output: bool) -> None:
    """Statically validate the selected plugin and its external profile."""

    from mailagent.verticals.validation import validate_vertical_profile

    report = validate_vertical_profile(_build_settings().vertical)
    if json_output:
        click.echo(json.dumps(report.as_dict(), ensure_ascii=False, indent=2))
    else:
        table = Table(title=f"Vertical profile: {report.vertical_id}")
        table.add_column("Status", width=8)
        table.add_column("Component")
        table.add_column("Detail")
        table.add_column("Path")
        for check in report.checks:
            table.add_row(
                "[green]PASS[/green]" if check.passed else "[red]FAIL[/red]",
                check.component,
                check.detail,
                check.path or "-",
            )
        console.print(table)
        console.print(
            "[green]Validation passed.[/green]"
            if report.valid
            else "[red]Validation failed.[/red]"
        )
    if not report.valid:
        raise click.exceptions.Exit(1)


# ---------------------------------------------------------------------------
# classification subgroup
# ---------------------------------------------------------------------------


@main.group()
def classification() -> None:
    """Classification evaluation and governance commands."""


@classification.command("evaluate")
@click.option(
    "--manifest",
    "manifest_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--predictions",
    "predictions_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    required=True,
)
@click.option("--enforce-gates", is_flag=True, default=False)
def classification_evaluate(
    manifest_path: Path,
    predictions_path: Path,
    output_dir: Path,
    enforce_gates: bool,
) -> None:
    """Recompute deterministic metrics from a saved prediction snapshot."""

    from mailagent.evaluation import (
        ReleaseGate,
        evaluate_predictions,
        load_gold_manifest,
        load_prediction_snapshot,
        write_evaluation_reports,
    )
    from mailagent.classification.taxonomy import load_taxonomy
    from mailagent.verticals import load_selected_vertical

    settings = _build_settings()
    loaded_vertical = load_selected_vertical(settings.vertical).assets
    taxonomy = load_taxonomy(loaded_vertical.taxonomy_path)
    valid_labels = taxonomy.all_codes()

    manifest = load_gold_manifest(manifest_path, valid_labels)
    predictions = load_prediction_snapshot(predictions_path, valid_labels)
    report = evaluate_predictions(manifest, predictions, ReleaseGate())
    json_path, markdown_path = write_evaluation_reports(report, output_dir)

    reasons = ", ".join(report.gate.reason_codes) or "none"
    console.print(
        f"Classification evaluation gate: [bold]{report.gate.status.upper()}[/bold] "
        f"(eligible decisions: {report.gate.eligible_decisions}; reasons: {reasons})"
    )
    console.print(f"JSON report: {json_path}")
    console.print(f"Markdown report: {markdown_path}")
    if enforce_gates and not report.gate.passed:
        raise click.exceptions.Exit(1)


# ---------------------------------------------------------------------------
# bootstrap subgroup
# ---------------------------------------------------------------------------


@main.group()
def bootstrap() -> None:
    """Bootstrap pipeline commands (seed, import, review, confirm, archive)."""


@bootstrap.command("seed")
@click.option("--dir", "dir_", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--force", is_flag=True, default=False, help="Overwrite existing mail_hash samples.")
@click.option("--no-rules", "no_rules", is_flag=True, default=False, help="Skip empty-rules warning.")
def bootstrap_seed(dir_: Path, force: bool, no_rules: bool) -> None:
    """Stage 1: seed labeling with LLM full annotation."""
    pipeline = _build_pipeline()
    report_id = asyncio.run(pipeline.seed(dir_, force=force, no_rules=no_rules))
    console.print(f"[green]Seed complete.[/green] Report ID: {report_id}")


@bootstrap.command("import")
@click.option("--dir", "dir_", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--batch-size", "batch_size", type=int, default=50, help="Emails per batch.")
def bootstrap_import(dir_: Path, batch_size: int) -> None:
    """Stage 2: incremental import with tiered labeling."""
    pipeline = _build_pipeline()
    report_id = asyncio.run(pipeline.import_history(dir_, batch_size=batch_size))
    console.print(f"[green]Import complete.[/green] Report ID: {report_id}")


@bootstrap.command("review")
@click.option("--report-id", "report_id", required=True, help="Report ID from seed/import.")
@click.option("--tier", "tier", required=True, help="Tier to review (1, 2, or 3).")
def bootstrap_review(report_id: str, tier: str) -> None:
    """Interactive review of bootstrap samples (like git rebase -i)."""
    pipeline = _build_pipeline()
    asyncio.run(_run_review(pipeline, report_id, tier))


@bootstrap.command("confirm")
@click.option("--report-id", "report_id", required=True, help="Report ID from seed/import.")
@click.option("--tier", "tier", type=int, required=True, help="Tier to confirm (1, 2, or 3).")
@click.option("--all", "all_", is_flag=True, default=False, help="Batch-confirm all tier samples.")
@click.option("--dry-run", "dry_run", is_flag=True, default=False, help="Preview without DB writes.")
def bootstrap_confirm(report_id: str, tier: int, all_: bool, dry_run: bool) -> None:
    """Confirm samples from a bootstrap report."""
    if tier == 2 or (all_ and tier == 3):
        if all_:
            console.print("[red]Error:[/red] --all is only allowed for --tier 1.")
        else:
            console.print("[red]Error:[/red] Tier 2 requires individual review.")
        console.print(
            "Tier 2/3 样本必须逐条审核, 请使用 "
            "[bold]mailagent bootstrap review --tier 2,3[/bold]"
        )
        sys.exit(1)

    pipeline = _build_pipeline()
    try:
        count = asyncio.run(
            pipeline.confirm_tier(report_id, tier, all_=all_, dry_run=dry_run)
        )
    except ValueError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    if dry_run:
        console.print(f"[yellow]Dry run:[/yellow] {count} samples would be confirmed.")
    else:
        console.print(f"[green]Confirmed {count} samples.[/green]")


@bootstrap.command("archive")
@click.option("--months", type=int, default=12, help="Archive samples older than N months.")
def bootstrap_archive(months: int) -> None:
    """Move old samples to the archive table."""
    pipeline = _build_pipeline()
    count = asyncio.run(pipeline.archive_old_samples(months=months))
    console.print(f"[green]Archived {count} samples[/green] (older than {months} months).")


# ---------------------------------------------------------------------------
# samples subgroup
# ---------------------------------------------------------------------------


@main.group()
def samples() -> None:
    """Sample management commands (list, delete, fix, audit, stats)."""


@samples.command("list")
@click.option("--label", "label", default=None, help="Filter by label_l3.")
@click.option("--source", "source", default=None, help="Filter by source.")
@click.option("--page", "page", type=int, default=1, help="Page number.")
def samples_list(label: str | None, source: str | None, page: int) -> None:
    """List samples with optional filters."""
    store = _build_vector_store()
    results = asyncio.run(store.get_samples(label=label, source=source, page=page))

    table = Table(title=f"Samples (page {page})")
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Subject")
    table.add_column("Sender")
    table.add_column("Label L3", style="green")
    table.add_column("Confidence")
    table.add_column("Source")
    table.add_column("Reviewed")

    for s in results:
        table.add_row(
            str(s.id)[:8],
            s.subject_raw[:40],
            s.sender[:30],
            s.label_l3,
            f"{s.confidence:.2f}",
            s.source,
            "✓" if s.reviewed else "✗",
        )

    console.print(table)


@samples.command("delete")
@click.option("--id", "id_", required=True, help="Sample UUID to delete.")
def samples_delete(id_: str) -> None:
    """Delete a sample by ID."""
    store = _build_vector_store()
    asyncio.run(store.delete_sample(uuid.UUID(id_)))
    console.print(f"[green]Deleted sample:[/green] {id_}")


@samples.command("fix")
@click.option("--id", "id_", required=True, help="Sample UUID to fix.")
@click.option("--label", "label", required=True, help="New label_l3 value.")
def samples_fix(id_: str, label: str) -> None:
    """Fix a sample's label (source=human_fix, confidence=1.0)."""
    store = _build_vector_store()
    asyncio.run(store.update_sample_label(uuid.UUID(id_), label, source="human_fix", confidence=1.0))
    console.print(f"[green]Fixed sample[/green] {id_} → label_l3={label}")


@samples.command("audit")
@click.option("--ratio", type=float, default=0.1, help="Fraction of unreviewed samples to audit.")
def samples_audit(ratio: float) -> None:
    """Audit a random fraction of unreviewed samples."""
    import random

    store = _build_vector_store()
    unreviewed = asyncio.run(store.get_samples(source=None))
    unreviewed = [s for s in unreviewed if not s.reviewed]
    sample_size = max(1, int(len(unreviewed) * ratio))
    selected = random.sample(unreviewed, min(sample_size, len(unreviewed)))

    table = Table(title=f"Audit — {len(selected)} of {len(unreviewed)} unreviewed")
    table.add_column("ID", style="cyan")
    table.add_column("Subject")
    table.add_column("Label L3", style="green")
    table.add_column("Source")

    for s in selected:
        table.add_row(str(s.id)[:8], s.subject_raw[:40], s.label_l3, s.source)

    console.print(table)


@samples.command("stats")
def samples_stats() -> None:
    """Show sample statistics."""
    store = _build_vector_store()
    total = asyncio.run(store.count_samples())

    all_samples = asyncio.run(store.get_samples(page=1, page_size=10000))

    by_source: dict[str, int] = {}
    by_l1: dict[str, int] = {}
    reviewed = 0
    unreviewed = 0
    oldest: str | None = None

    for s in all_samples:
        by_source[s.source] = by_source.get(s.source, 0) + 1
        by_l1[s.label_l1] = by_l1.get(s.label_l1, 0) + 1
        if s.reviewed:
            reviewed += 1
        else:
            unreviewed += 1
        if oldest is None or s.created_at.isoformat() < oldest:
            oldest = s.created_at.isoformat()

    console.print(Panel.fit(f"[bold]Total samples:[/bold] {total}", title="Statistics"))

    table = Table(title="By Source")
    table.add_column("Source", style="cyan")
    table.add_column("Count")
    for src, cnt in sorted(by_source.items()):
        table.add_row(src, str(cnt))
    console.print(table)

    table2 = Table(title="By L1 Label")
    table2.add_column("Label L1", style="green")
    table2.add_column("Count")
    for lbl, cnt in sorted(by_l1.items()):
        table2.add_row(lbl, str(cnt))
    console.print(table2)

    console.print(f"Reviewed: {reviewed}  |  Unreviewed: {unreviewed}")
    console.print(f"Oldest sample: {oldest or 'N/A'}")

    # Quality / taxonomy / policy distribution (improve-vector-rag-sample-quality).
    quality_stats = asyncio.run(store.get_quality_stats())

    qtable = Table(title="Quality Disposition")
    qtable.add_column("Disposition", style="magenta")
    qtable.add_column("Count")
    for disp, cnt in sorted(quality_stats["by_disposition"].items()):
        qtable.add_row(disp, str(cnt))
    console.print(qtable)

    ttable = Table(title="Taxonomy Schema Version")
    ttable.add_column("Schema", style="yellow")
    ttable.add_column("Count")
    for ver, cnt in sorted(quality_stats["by_taxonomy_schema"].items()):
        ttable.add_row(ver, str(cnt))
    console.print(ttable)

    ptable = Table(title="Retrieval Policy Version")
    ptable.add_column("Policy", style="blue")
    ptable.add_column("Count")
    for ver, cnt in sorted(quality_stats["by_retrieval_policy"].items()):
        ptable.add_row(ver, str(cnt))
    console.print(ptable)

    dup = quality_stats["duplicate_fingerprint_rows"]
    if dup:
        console.print(f"[red]Duplicate fingerprint rows:[/red] {dup}")
    else:
        console.print("[green]No duplicate fingerprints detected.[/green]")


@samples.command(name="reembed")
@click.option(
    "--policy-version",
    required=True,
    help="Target retrieval_policy_version for re-embedding.",
)
@click.option(
    "--batch-size",
    type=int,
    default=100,
    help="Number of samples per batch (default 100).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Report affected samples without altering embeddings.",
)
def samples_reembed(policy_version: str, batch_size: int, dry_run: bool) -> None:
    """Re-embed samples whose retrieval_policy_version differs from the target.

    Resumable: each completed sample is marked with the target policy version,
    so a subsequent run after a crash skips them automatically. Use --dry-run
    to preview the migration scope without modifying any rows.
    """
    store = _build_vector_store()
    candidate_ids = asyncio.run(
        store.get_reembed_candidates(policy_version, batch_size=batch_size)
    )

    console.print(
        Panel.fit(
            f"[bold]Policy version:[/bold] {policy_version}\n"
            f"[bold]Batch size:[/bold] {batch_size}\n"
            f"[bold]Mode:[/bold] {'DRY RUN' if dry_run else 'APPLY'}\n"
            f"[bold]Candidates in this batch:[/bold] {len(candidate_ids)}",
            title="Re-embedding",
        )
    )

    if dry_run:
        console.print(
            "[yellow]Dry run:[/yellow] no embeddings will be modified. "
            "Re-run without --dry-run to apply."
        )
        return

    if not candidate_ids:
        console.print("[green]No samples need re-embedding.[/green]")
        return

    console.print(
        "[red]Re-embedding not yet implemented for apply mode.[/red] "
        "Use --dry-run to preview scope. The apply path requires a live "
        "EmbeddingClient and will be added once the worker integration "
        "is finalized."
    )


# ---------------------------------------------------------------------------
# rules subgroup
# ---------------------------------------------------------------------------


@main.group()
def rules() -> None:
    """Rules management commands."""


@rules.command("add")
@click.option("--from-report", "from_report", required=True, type=click.Path(exists=True, path_type=Path))
def rules_add(from_report: Path) -> None:
    """Apply checked sender-domain proposals from a Markdown report."""
    settings = _build_settings()
    try:
        from mailagent.classification.taxonomy import TaxonomyLoader
        from mailagent.verticals import load_selected_vertical

        loaded_vertical = load_selected_vertical(settings.vertical).assets
        if loaded_vertical.rules is None:
            raise ValueError(
                f"selected vertical {loaded_vertical.manifest.id} has no rules asset"
            )
        rules_dir = loaded_vertical.rules.path
        taxonomy_snapshot = TaxonomyLoader(
            loaded_vertical.taxonomy_path
        ).get_snapshot()
        added = append_confirmed_sender_domain_rules(
            from_report, rules_dir, taxonomy_snapshot
        )
    except ValueError as exc:
        raise click.UsageError(str(exc)) from exc
    console.print(f"[green]Added {added} rules[/green] to {rules_dir / 'sender_domains.yaml'}")


# ---------------------------------------------------------------------------
# Interactive review implementation
# ---------------------------------------------------------------------------


async def _run_review(pipeline: BootstrapPipeline, report_id: str, tier: str) -> None:
    """Run the interactive review loop for one tier of a report."""
    report = pipeline._load_report(report_id)
    tier_key = f"tier{tier}"
    tier_samples = [s for s in report["samples"] if s["tier"] == tier_key]

    if not tier_samples:
        console.print(f"[yellow]No samples found for {tier_key} in report {report_id}.[/yellow]")
        return

    console.print(f"[bold]Reviewing {len(tier_samples)} samples for {tier_key}[/bold]")
    console.print(
        "[dim]Actions: [Enter]=confirm  [e]=edit category  [r]=edit retrieval text "
        "[s]=skip  [d]=discard  [q]=quit[/dim]\n"
    )

    confirmed = 0
    discarded = 0
    skipped = 0

    for i, entry in enumerate(tier_samples):
        _display_sample(entry, i + 1, len(tier_samples))

        action = Prompt.ask(
            "Action",
            choices=["", "e", "r", "s", "d", "q"],
            default="",
            show_choices=False,
        )

        if action == "q":
            console.print(f"\n[yellow]Save and quit.[/yellow] "
                          f"Confirmed: {confirmed}, Discarded: {discarded}, Skipped: {skipped}")
            return

        if action == "d":
            entry["action"] = "discarded"
            discarded += 1
            console.print("  [red]Discarded.[/red]\n")
            continue

        if action == "s":
            entry["action"] = "skipped"
            skipped += 1
            console.print("  [dim]Skipped.[/dim]\n")
            continue

        if action == "e":
            new_label = Prompt.ask("  New flat category code")
            entry["llm_match"] = {
                "l1": new_label,
                "l2": None,
                "l3": None,
                "confidence": 1.0,
                "reasoning": "human_edit",
            }
            entry["action"] = "edited"
            await _persist_entry(pipeline, entry, report.get("stage", "import"), is_edit=True)
            confirmed += 1
            console.print(f"  [green]Confirmed with edit:[/green] {new_label}\n")
            continue

        if action == "r":
            entry["retrieval_text_override"] = Prompt.ask("  New retrieval text")
            entry["override_reason"] = Prompt.ask("  Override reason")
            entry["action"] = "retrieval_text_edited"
            await _persist_entry(pipeline, entry, report.get("stage", "import"), is_edit=True)
            confirmed += 1
            console.print("  [green]Confirmed with retrieval-text override.[/green]\n")
            continue

        # Enter (default) — confirm
        entry["action"] = "confirmed"
        await _persist_entry(pipeline, entry, report.get("stage", "import"))
        confirmed += 1
        console.print("  [green]Confirmed.[/green]\n")

    console.print(
        f"\n[bold]Review complete.[/bold] "
        f"Confirmed: {confirmed}, Discarded: {discarded}, Skipped: {skipped}"
    )


def _display_sample(entry: dict[str, Any], index: int, total: int) -> None:
    """Display one sample's metadata in a rich panel."""
    rule_match = entry.get("rule_match")
    llm_match = entry.get("llm_match")

    lines: list[str] = []
    lines.append(f"[bold]Subject:[/bold] {entry.get('subject', 'N/A')}")
    lines.append(f"[bold]Sender:[/bold] {entry.get('sender', 'N/A')}")
    lines.append(f"[bold]Tier:[/bold] {entry.get('tier', 'N/A')}")

    if rule_match:
        lines.append(
            f"[bold]Rule suggestion:[/bold] {rule_match.get('label', 'N/A')} "
            f"(conf={rule_match.get('confidence', 0):.2f}, type={rule_match.get('rule_type', '')})"
        )
    else:
        lines.append("[bold]Rule suggestion:[/bold] [dim]none[/dim]")

    if llm_match:
        lines.append(
            f"[bold]LLM suggestion:[/bold] l1={llm_match.get('l1', 'N/A')} "
            f"l2={llm_match.get('l2', 'N/A')} l3={llm_match.get('l3', 'N/A')} "
            f"(conf={llm_match.get('confidence', 0):.2f})"
        )
    else:
        lines.append("[bold]LLM suggestion:[/bold] [dim]unavailable[/dim]")

    lines.append(f"[bold]Consistency:[/bold] {'✓' if entry.get('consistency') else '✗'}")
    if entry.get("tier") == "tier2":
        lines.append(
            f"[bold]Verification:[/bold] "
            f"{entry.get('verification_status', 'unavailable')}"
        )
        verification_detail = entry.get("verification_detail")
        if verification_detail:
            lines.append(
                f"[bold]Verification detail:[/bold] {verification_detail}"
            )
        lines.append(
            "[yellow]Mandatory individual review before persistence.[/yellow]"
        )

    body_preview = entry.get("body_preview", "")
    lines.append(f"\n[dim]Body preview:[/dim]\n{body_preview}")

    panel = Panel(
        "\n".join(lines),
        title=f"Sample {index}/{total} — {entry.get('mail_hash', '')[:12]}",
        border_style="blue",
    )
    console.print(panel)


async def _persist_entry(
    pipeline: BootstrapPipeline,
    entry: dict[str, Any],
    stage: str,
    is_edit: bool = False,
) -> None:
    """Persist a single sample entry via the pipeline."""
    source = "human_fix" if is_edit else _determine_source(entry, stage)
    sample, emb_thread, emb_seg0 = await pipeline._prepare_sample_for_persistence(
        entry, source=source
    )
    await pipeline._persist_sample(sample, emb_thread, emb_seg0)


def _determine_source(entry: dict[str, Any], stage: str) -> str:
    """Determine the source field for a confirmed sample."""
    if stage == "seed":
        return "seed"
    tier = entry.get("tier", "tier3")
    if tier == "tier1":
        return "rule_tier1"
    if tier == "tier2":
        return "rule_tier2"
    return "llm"


if __name__ == "__main__":
    main()
