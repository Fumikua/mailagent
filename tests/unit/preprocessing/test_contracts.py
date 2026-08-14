from __future__ import annotations

from importlib.util import find_spec

def test_extension_contract_module_is_available() -> None:
    """Generic preprocessing offers a vertical extension seam."""

    assert find_spec("mailagent.preprocessing.contracts") is not None
