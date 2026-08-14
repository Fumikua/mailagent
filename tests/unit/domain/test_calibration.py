"""锚点映射置信度校准测试。"""
from __future__ import annotations

import pytest

from mailagent.domain.calibration import calibrate, calibration_log


@pytest.mark.parametrize(
    "raw,expected",
    [
        (0.96, 0.975),
        (1.0, 0.975),
        (0.95, 0.975),
        (0.94, 0.87),
        (0.80, 0.87),
        (0.79, 0.70),
        (0.60, 0.70),
        (0.59, 0.30),
        (0.0, 0.30),
    ],
)
def test_calibrate_anchor_boundaries(raw: float, expected: float) -> None:
    assert calibrate(raw) == pytest.approx(expected)


def test_calibrate_invalid_below_zero_clamps_to_zero() -> None:
    assert calibrate(-0.5) == 0.30


def test_calibrate_invalid_above_one_clamps_to_one() -> None:
    assert calibrate(1.5) == 0.975


def test_calibration_log_returns_triple() -> None:
    log = calibration_log(0.92)
    assert log.raw == 0.92
    assert log.calibrated == 0.87
    assert log.anchor == "fairly certain"


def test_calibration_log_anchor_for_high_confidence() -> None:
    log = calibration_log(0.99)
    assert log.calibrated == 0.975
    assert log.anchor == "very certain"


def test_calibration_log_anchor_for_directional() -> None:
    log = calibration_log(0.65)
    assert log.calibrated == 0.70
    assert log.anchor == "directional"


def test_calibration_log_anchor_for_uncertain() -> None:
    log = calibration_log(0.30)
    assert log.calibrated == 0.30
    assert log.anchor == "uncertain"


def test_calibration_log_clamps_invalid_input() -> None:
    log = calibration_log(-0.5)
    assert log.raw == 0.0
    assert log.calibrated == 0.30
    assert log.anchor == "uncertain"
