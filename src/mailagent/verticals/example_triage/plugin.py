"""Installable descriptor for the example-triage vertical.

This is the simplest possible vertical: no enrichers, no signals, no RAG. It
exists so contributors can fork a working classification-only vertical and
grow it. New verticals should live in their own pip package and register the
same ``mailagent.verticals`` entry point.
"""
from __future__ import annotations

from ..plugin import VerticalPlugin
from ..runtime import build_empty_runtime

plugin = VerticalPlugin(
    id="example-triage",
    namespace="example_triage",
    build_runtime=build_empty_runtime,
)
