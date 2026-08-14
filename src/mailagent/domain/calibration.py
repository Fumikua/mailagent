"""锚点映射置信度校准。

LLM 输出 raw_confidence (0-1) → 校准区间中点：

| raw_confidence 区间 | 校准中点 | anchor |
|---|---|---|
| 0.95 - 1.0 | 0.975 | very certain |
| 0.80 - 0.94 | 0.87  | fairly certain |
| 0.60 - 0.79 | 0.70  | directional |
| < 0.60     | 0.30  | uncertain |
"""
from __future__ import annotations

from .models import CalibrationLog

_ANCHORS: list[tuple[float, float, str]] = [
    (0.95, 0.975, "very certain"),
    (0.80, 0.87, "fairly certain"),
    (0.60, 0.70, "directional"),
    (0.00, 0.30, "uncertain"),
]


def calibrate(raw: float) -> float:
    """锚点映射：raw_confidence → 校准区间中点

    Args:
        raw: LLM 自报的 raw_confidence (0-1)，自动 clamp

    Returns:
        float: 校准后的 confidence (0.30 / 0.70 / 0.87 / 0.975)
    """
    clamped = max(0.0, min(1.0, raw))
    for threshold, calibrated, _ in _ANCHORS:
        if clamped >= threshold:
            return calibrated
    return 0.30  # 不可达，兜底


def calibration_log(raw: float) -> CalibrationLog:
    """生成完整 calibration_log 记录

    Args:
        raw: LLM 自报的 raw_confidence (0-1)

    Returns:
        CalibrationLog: {raw, calibrated, anchor} 三元组
    """
    clamped = max(0.0, min(1.0, raw))
    for threshold, calibrated, anchor in _ANCHORS:
        if clamped >= threshold:
            return CalibrationLog(raw=clamped, calibrated=calibrated, anchor=anchor)
    return CalibrationLog(raw=clamped, calibrated=0.30, anchor="uncertain")
