"""Immutable validated asset snapshots and portable content identities."""
from __future__ import annotations

import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Generic, TypeVar

_AssetT = TypeVar("_AssetT")


@dataclass(frozen=True, slots=True)
class ValidatedAssetSnapshot(Generic[_AssetT]):
    """One immutable-by-contract validated value paired with its identity."""

    value: _AssetT
    version: str


def digest_named_assets(named_assets: Iterable[tuple[str, bytes]]) -> str:
    """Hash explicit logical names and validated canonical values."""

    assets = list(named_assets)
    logical_names = [logical_name for logical_name, _ in assets]
    if len(set(logical_names)) != len(logical_names):
        raise ValueError("duplicate logical asset name")

    digest = hashlib.sha256()
    for logical_name, value in sorted(assets, key=lambda item: item[0]):
        digest.update(logical_name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(value)
        digest.update(b"\0")
    return f"sha256:{digest.hexdigest()}"
