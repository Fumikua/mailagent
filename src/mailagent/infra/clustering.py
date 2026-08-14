"""Clustering engine for weekly intent discovery using HDBSCAN.

Runs as an arq cron job, pulling recent embeddings from the vector store,
clustering them with HDBSCAN, classifying clusters against taxonomy centroids
(known / drift / new_intent), and generating a markdown report for human review.

When ``hdbscan`` (and its dependency ``numpy``) are not installed, the engine
degrades gracefully — ``run_weekly_clustering`` returns a skip message and logs
a warning.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

from ..domain.models import SampleRecord
from ..llm.client import LLMClient
from ..llm.taxonomy import TaxonomyLoader
from .config import ClusteringSettings
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

# Conditional import: hdbscan depends on numpy, so both are available together.
try:
    import hdbscan  # type: ignore[import-not-found]
    import numpy as np  # type: ignore[import-not-found]

    HAS_HDBSCAN = True
except ImportError:
    hdbscan = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    HAS_HDBSCAN = False


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Pure-Python cosine similarity — no numpy required."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        norm_a += x * x
        norm_b += y * y
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))


def _compute_centroid(embeddings: list[list[float]]) -> list[float]:
    """Compute element-wise mean of a list of embedding vectors (pure Python)."""
    if not embeddings:
        return []
    dim = len(embeddings[0])
    centroid = [0.0] * dim
    for emb in embeddings:
        for i, val in enumerate(emb):
            centroid[i] += val
    n = len(embeddings)
    return [v / n for v in centroid]


class ClusteringEngine:
    """Weekly HDBSCAN clustering for intent discovery and drift detection."""

    def __init__(
        self,
        vector_store: VectorStore,
        taxonomy_loader: TaxonomyLoader,
        llm_client: LLMClient,
        settings: ClusteringSettings,
    ) -> None:
        self.vector_store = vector_store
        self.taxonomy_loader = taxonomy_loader
        self.llm_client = llm_client
        self.settings = settings
        self._reports_dir = Path("./reports")
        self._total_samples = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run_weekly_clustering(self) -> str:
        """Run HDBSCAN clustering on recent embeddings and generate a report.

        Returns the report file path, or a skip message if hdbscan is not
        installed or insufficient data is available.
        """
        if not HAS_HDBSCAN:
            logger.warning("hdbscan not installed, clustering skipped")
            return "clustering skipped: hdbscan not installed"

        assert hdbscan is not None  # noqa: S101
        assert np is not None  # noqa: S101

        # Fetch embeddings for the configured time window.
        embeddings_data = await self.vector_store.get_embeddings(
            days=self.settings.window_days
        )

        # Stratified sampling when dataset exceeds max_samples.
        if len(embeddings_data) > self.settings.max_samples:
            unique_labels = {label for _, _, label in embeddings_data}
            num_labels = max(len(unique_labels), 1)
            max_per_label = self.settings.max_samples // num_labels
            logger.info(
                "embeddings count %d > max_samples %d, stratified sampling "
                "(%d labels, max %d per label)",
                len(embeddings_data),
                self.settings.max_samples,
                num_labels,
                max_per_label,
            )
            embeddings_data = await self.vector_store.stratified_sample(
                days=self.settings.window_days,
                max_per_label=max_per_label,
            )

        self._total_samples = len(embeddings_data)

        if self._total_samples < self.settings.min_cluster_size:
            logger.info(
                "not enough samples for clustering (%d < %d), skipping",
                self._total_samples,
                self.settings.min_cluster_size,
            )
            self._reports_dir.mkdir(parents=True, exist_ok=True)
            report_path = (
                self._reports_dir
                / f"intent_discovery_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md"
            )
            self._generate_report([], [], report_path)
            return str(report_path)

        # Run HDBSCAN clustering.
        embeddings_array = np.array([emb for _, emb, _ in embeddings_data])
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.settings.min_cluster_size,
            min_samples=self.settings.min_samples,
            metric=self.settings.metric,
        )
        labels = clusterer.fit_predict(embeddings_array)
        labels_list = list(labels)

        # Fetch taxonomy centroids for cluster classification.
        centroids = await self.vector_store.get_centroids()

        # Fetch full SampleRecords for representative extraction.
        sample_ids = {uid for uid, _, _ in embeddings_data}
        samples_by_id = await self._fetch_samples_by_ids(sample_ids)

        # Classify clusters and collect new intents / drifts.
        new_intents: list[dict] = []
        drifts: list[dict] = []

        unique_cluster_labels = sorted(set(labels_list))
        for cluster_label in unique_cluster_labels:
            if cluster_label == -1:
                continue  # noise points

            cluster_indices = [
                i for i, lbl in enumerate(labels_list) if lbl == cluster_label
            ]
            cluster_embeddings = [embeddings_data[i][1] for i in cluster_indices]
            cluster_sample_ids = [embeddings_data[i][0] for i in cluster_indices]
            cluster_samples = [
                samples_by_id[sid]
                for sid in cluster_sample_ids
                if sid in samples_by_id
            ]

            cluster_type, max_sim = self._classify_cluster(
                cluster_embeddings, centroids
            )

            if cluster_type == "known":
                continue

            centroid = _compute_centroid(cluster_embeddings)
            representatives = self._extract_representatives(
                cluster_samples, cluster_embeddings, centroid
            )
            intent = await self._llm_describe_intent(representatives)

            entry: dict = {
                "cluster_id": int(cluster_label),
                "sample_count": len(cluster_samples),
                "intent": intent,
                "representatives": representatives,
            }

            if cluster_type == "drift":
                entry["max_similarity"] = max_sim
                drifts.append(entry)
            else:
                new_intents.append(entry)

        # Generate report.
        self._reports_dir.mkdir(parents=True, exist_ok=True)
        report_path = (
            self._reports_dir
            / f"intent_discovery_{datetime.now(timezone.utc).strftime('%Y%m%d')}.md"
        )
        self._generate_report(new_intents, drifts, report_path)
        logger.info(
            "clustering complete: %d new intents, %d drifts, report at %s",
            len(new_intents),
            len(drifts),
            report_path,
        )
        return str(report_path)

    async def backfill_after_taxonomy_change(self, new_code: str) -> int:
        """Backfill samples whose label_l3 is no longer in the taxonomy.

        Finds samples with stale labels (not in the current taxonomy tree) and
        rewrites them to ``new_code``. Idempotent — a second call finds no stale
        samples and returns 0.
        """
        valid_codes = self.taxonomy_loader.get_tree().all_codes()

        stale_ids: list[UUID] = []
        page = 1
        page_size = 1000
        while True:
            batch = await self.vector_store.get_samples(page=page, page_size=page_size)
            if not batch:
                break
            for s in batch:
                if s.label_l3 not in valid_codes:
                    stale_ids.append(s.id)
            if len(batch) < page_size:
                break
            page += 1

        if stale_ids:
            await self.vector_store.backfill_samples_label(new_code, stale_ids)
        return len(stale_ids)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _classify_cluster(
        self,
        cluster_embeddings: list[list[float]],
        centroids: dict[str, list[float]],
    ) -> tuple[str, float]:
        """Classify a cluster by comparing its centroid to taxonomy centroids.

        Returns ``(cluster_type, max_similarity)`` where cluster_type is one of
        ``known`` (>= 0.85), ``drift`` (0.6–0.85), or ``new_intent`` (< 0.6).
        """
        if not cluster_embeddings or not centroids:
            return "new_intent", 0.0

        cluster_centroid = _compute_centroid(cluster_embeddings)

        max_sim = 0.0
        for tax_centroid in centroids.values():
            sim = _cosine_similarity(cluster_centroid, tax_centroid)
            if sim > max_sim:
                max_sim = sim

        if max_sim >= 0.85:
            return "known", max_sim
        elif max_sim >= 0.6:
            return "drift", max_sim
        else:
            return "new_intent", max_sim

    def _extract_representatives(
        self,
        cluster_samples: list[SampleRecord],
        cluster_embeddings: list[list[float]],
        centroid: list[float],
        k: int = 5,
    ) -> list[SampleRecord]:
        """Return the ``k`` samples closest to the cluster centroid."""
        if not cluster_samples or not cluster_embeddings:
            return []

        scored: list[tuple[float, SampleRecord]] = []
        for sample, emb in zip(cluster_samples, cluster_embeddings, strict=True):
            sim = _cosine_similarity(emb, centroid)
            scored.append((sim, sample))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [s for _, s in scored[:k]]

    async def _llm_describe_intent(
        self, representatives: list[SampleRecord]
    ) -> str:
        """Use the LLM to generate a 1-2 sentence intent description."""
        if not representatives:
            return "Unknown intent"

        email_blocks: list[str] = []
        for i, r in enumerate(representatives, 1):
            body_preview = (r.body or "")[:200]
            email_blocks.append(
                f"Email {i}:\n  Subject: {r.subject_raw}\n  Preview: {body_preview}"
            )

        prompt = (
            "You are analyzing a cluster of similar emails to identify their "
            "common intent.\n"
            "Below are representative emails from this cluster:\n\n"
            + "\n\n".join(email_blocks)
            + "\n\nDescribe the common intent of these emails in 1-2 sentences."
        )

        try:
            response = await self.llm_client.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response["choices"][0]["message"]["content"]
            return str(content).strip()
        except Exception as exc:
            logger.warning("LLM describe intent failed: %s", exc)
            return "Unknown intent"

    def _generate_report(
        self,
        new_intents: list[dict],
        drifts: list[dict],
        output_path: Path,
    ) -> None:
        """Write a markdown report of new intents and drifts."""
        now = datetime.now(timezone.utc)
        lines: list[str] = [
            f"# Intent Discovery Report — {now.strftime('%Y-%m-%d')}",
            "",
            f"**Date**: {now.strftime('%Y-%m-%d %H:%M:%S UTC')}",
            f"**Total samples analyzed**: {self._total_samples}",
            (
                f"**Clustering parameters**: "
                f"min_cluster_size={self.settings.min_cluster_size}, "
                f"min_samples={self.settings.min_samples}, "
                f"metric={self.settings.metric}"
            ),
            "",
        ]

        # New intents section
        lines.append("## New Intents")
        lines.append("")
        if new_intents:
            for ni in new_intents:
                lines.append(
                    f"### Cluster #{ni['cluster_id']} "
                    f"({ni['sample_count']} samples)"
                )
                lines.append("")
                lines.append(f"- [ ] **Intent**: {ni['intent']}")
                lines.append("- [ ] **Proposed action**: Add to taxonomy")
                lines.append("- 建议标签名: [___________]")
                lines.append("")
                lines.append("**Representative emails**:")
                for r in ni["representatives"]:
                    lines.append(f"- [ ] `{r.subject_raw}` (from {r.sender})")
                lines.append("")
        else:
            lines.append("无新意图候选 (No new intent candidates).")
            lines.append("")

        # Drifts section
        lines.append("## Drifts")
        lines.append("")
        if drifts:
            for d in drifts:
                lines.append(
                    f"### Cluster #{d['cluster_id']} "
                    f"({d['sample_count']} samples)"
                )
                lines.append("")
                lines.append(f"- [ ] **Intent**: {d['intent']}")
                lines.append(
                    f"- [ ] **Max similarity**: {d['max_similarity']:.4f}"
                )
                lines.append("- [ ] **Proposed action**: Review label boundary")
                lines.append("")
                lines.append("**Representative emails**:")
                for r in d["representatives"]:
                    lines.append(f"- [ ] `{r.subject_raw}` (from {r.sender})")
                lines.append("")
        else:
            lines.append("无漂移检测 (No drifts detected).")
            lines.append("")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines), encoding="utf-8")

    async def _fetch_samples_by_ids(
        self, sample_ids: set[UUID]
    ) -> dict[UUID, SampleRecord]:
        """Paginate through ``get_samples`` to build a UUID → SampleRecord map."""
        result: dict[UUID, SampleRecord] = {}
        if not sample_ids:
            return result
        page = 1
        page_size = 1000
        while len(result) < len(sample_ids):
            batch = await self.vector_store.get_samples(
                page=page, page_size=page_size
            )
            if not batch:
                break
            for s in batch:
                if s.id in sample_ids:
                    result[s.id] = s
            if len(batch) < page_size:
                break
            page += 1
        return result
