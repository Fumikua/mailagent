"""Bind one run to the exact validated assets consumed by classification."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from mailagent.domain.models import ClassificationVersions
from mailagent.domain.versioning import (
    ValidatedAssetSnapshot,
    digest_named_assets,
)
from mailagent.preprocessing.retrieval_models import (
    RetrievalCleaningPolicy,
    validated_policy_version,
)

if TYPE_CHECKING:
    from mailagent.classification.rule_classifier import RuleClassifier
    from mailagent.core.target_profile import TargetProfileLoader
    from mailagent.classification.taxonomy import TaxonomyLoader
    from mailagent.preprocessing.contracts import MailPreprocessingExtension


def _digest(named_assets: Iterable[tuple[str, bytes]]) -> str:
    """Hash validated values under explicit, unique logical identities."""

    return digest_named_assets(named_assets)


@dataclass(frozen=True, slots=True)
class ClassificationAssetBinding:
    """One run's immutable versions and the snapshots that produced them."""

    versions: ClassificationVersions
    asset_snapshots: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "asset_snapshots",
            MappingProxyType(dict(self.asset_snapshots)),
        )


class ClassificationVersionProvider:
    """Atomically capture versions from live validated component snapshots."""

    def __init__(
        self,
        *,
        taxonomy_loader: TaxonomyLoader,
        rule_classifier: RuleClassifier | None = None,
        preprocessing_extension: MailPreprocessingExtension | None = None,
        retrieval_cleaning_policy: object | None = None,
        target_profile_loader: TargetProfileLoader | None = None,
        prompt_version: str | None,
        model_version: str | None,
        embedding_version: str | None,
    ) -> None:
        self._taxonomy_loader = taxonomy_loader
        self._rule_classifier = rule_classifier
        self._preprocessing_extension = preprocessing_extension
        self._retrieval_cleaning_policy = retrieval_cleaning_policy
        self._target_profile_loader = target_profile_loader
        self._prompt_version = prompt_version
        self._model_version = model_version
        self._embedding_version = embedding_version

    def bind(self) -> ClassificationAssetBinding:
        """Capture once; classifiers must consume the returned snapshots."""

        taxonomy_snapshot = self._taxonomy_loader.get_snapshot()
        assets: dict[str, object] = {"taxonomy": taxonomy_snapshot}

        rules_version: str | None = None
        if self._rule_classifier is not None:
            rules_snapshot = self._rule_classifier.get_snapshot(taxonomy_snapshot)
            if rules_snapshot.value.taxonomy_version in {
                None,
                taxonomy_snapshot.version,
            }:
                assets["rules"] = rules_snapshot
                rules_version = rules_snapshot.version
            else:
                # A taxonomy reload can invalidate the preceding rule snapshot.
                # Preserve it inside RuleClassifier, but do not use it for this run.
                assets["rules"] = None

        target_profiles_version: str | None = None
        if self._target_profile_loader is not None:
            target_profiles_snapshot = self._target_profile_loader.get_snapshot(
                taxonomy_snapshot
            )
            if target_profiles_snapshot.value.taxonomy_version in {
                None,
                taxonomy_snapshot.version,
            }:
                assets["target_profiles"] = target_profiles_snapshot
                target_profiles_version = target_profiles_snapshot.version
            else:
                assets["target_profiles"] = None

        preprocessing_versions: list[tuple[str, bytes]] = []
        if self._preprocessing_extension is not None:
            get_snapshot = getattr(
                self._preprocessing_extension,
                "get_snapshot",
                None,
            )
            if not callable(get_snapshot):
                raise TypeError(
                    "configured preprocessing extension does not expose "
                    "a validated snapshot"
                )
            preprocessing_snapshot = get_snapshot()
            if not isinstance(preprocessing_snapshot, ValidatedAssetSnapshot):
                raise TypeError("invalid preprocessing asset snapshot")
            assets["preprocessing"] = preprocessing_snapshot
            preprocessing_versions.append(
                (
                    "preprocessing:extension",
                    preprocessing_snapshot.version.encode("utf-8"),
                )
            )

        if self._retrieval_cleaning_policy is not None:
            policy = cast(
                RetrievalCleaningPolicy,
                self._retrieval_cleaning_policy,
            )
            policy_snapshot = ValidatedAssetSnapshot(
                value=policy,
                version=validated_policy_version(policy),
            )
            assets["retrieval_cleaning"] = policy_snapshot
            preprocessing_versions.append(
                (
                    "preprocessing:retrieval_cleaning",
                    policy_snapshot.version.encode("utf-8"),
                )
            )

        versions = ClassificationVersions(
            taxonomy=taxonomy_snapshot.version,
            rules=rules_version,
            target_profiles=target_profiles_version,
            prompt=self._prompt_version,
            model=self._model_version,
            embedding=self._embedding_version,
            preprocessing=(
                _digest(preprocessing_versions)
                if preprocessing_versions
                else "none"
            ),
        )
        return ClassificationAssetBinding(
            versions=versions,
            asset_snapshots=assets,
        )

    def snapshot(self) -> ClassificationVersions:
        """Compatibility accessor for callers that only need active versions."""

        return self.bind().versions
