#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Pure policy helpers for noise-aware candidate throughput validation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from quark.experimental.torch.quant_perf.evaluation.throughput import ThroughputMeasurement


class RetestDisposition(StrEnum):
    KEEP = "keep"
    DROP_CONFIRMED = "drop_confirmed"
    CONFIRM_ABA = "confirm_aba"
    CONFIRM_ABBA = "confirm_abba"
    CONFIRM_NEGATIVE = "confirm_negative"
    NEEDS_REVIEW = "needs_review"
    RETRYABLE_FAULT = "retryable_fault"


@dataclass(frozen=True)
class RetestScreen:
    disposition: RetestDisposition
    gain: float
    effective_floor: float
    strong_threshold: float


def effective_keep_floor(
    keep_floor: float,
    *measurements: ThroughputMeasurement,
) -> float:
    noise = max(
        (measurement.relative_mad for measurement in measurements),
        default=0.0,
    )
    return max(float(keep_floor), 2.0 * noise)


def screen_retest_mode(
    anchor: ThroughputMeasurement,
    candidate: ThroughputMeasurement,
    *,
    keep_floor: float,
    trace_rank: int | None,
    micro_speedup: float | None,
) -> RetestScreen:
    floor = effective_keep_floor(keep_floor, anchor, candidate)
    strong = max(0.05, 3.0 * floor)
    gain = candidate.median_tps / anchor.median_tps - 1.0
    if not anchor.stable or not candidate.stable:
        disposition = RetestDisposition.RETRYABLE_FAULT
    elif gain >= strong:
        disposition = RetestDisposition.CONFIRM_ABA
    elif gain > floor:
        disposition = RetestDisposition.CONFIRM_ABBA
    elif abs(gain) <= floor:
        high_priority = (micro_speedup is not None and micro_speedup >= 1.05) or (
            trace_rank is not None and trace_rank <= 3
        )
        disposition = RetestDisposition.CONFIRM_ABBA if high_priority else RetestDisposition.NEEDS_REVIEW
    elif gain <= -strong:
        disposition = RetestDisposition.CONFIRM_NEGATIVE
    else:
        disposition = RetestDisposition.NEEDS_REVIEW
    return RetestScreen(disposition, gain, floor, strong)


def aba_gain(a1_tps: float, b1_tps: float, a2_tps: float) -> float:
    return b1_tps / math.sqrt(a1_tps * a2_tps)


def abba_gain(
    a1_tps: float,
    b1_tps: float,
    b2_tps: float,
    a2_tps: float,
) -> float:
    return math.sqrt((b1_tps * b2_tps) / (a1_tps * a2_tps))


def decide_confirmed_retest(
    gain_multiplier: float,
    effective_floor: float,
) -> RetestDisposition:
    incremental = gain_multiplier - 1.0
    if incremental > effective_floor:
        return RetestDisposition.KEEP
    if incremental <= 0.0:
        return RetestDisposition.DROP_CONFIRMED
    return RetestDisposition.NEEDS_REVIEW
