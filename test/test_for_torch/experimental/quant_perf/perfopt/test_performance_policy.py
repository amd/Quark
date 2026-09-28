#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from quark.experimental.torch.quant_perf.evaluation.throughput import ThroughputMeasurement
from quark.experimental.torch.quant_perf.pipeline.candidate_validation import run_candidate_retest
from quark.experimental.torch.quant_perf.pipeline.performance_policy import (
    RetestDisposition,
    aba_gain,
    abba_gain,
    decide_confirmed_retest,
    effective_keep_floor,
    screen_retest_mode,
)


def _measurement(tps: float, relative_mad: float = 0.002, stable: bool = True):
    return ThroughputMeasurement(
        samples_tps=(tps - 1.0, tps, tps + 1.0),
        median_tps=tps,
        mad_tps=1.0,
        relative_mad=relative_mad,
        warmup_tps=tps - 2.0,
        stable=stable,
    )


def test_effective_floor_respects_user_floor_and_measured_noise():
    assert effective_keep_floor(
        0.01,
        _measurement(100.0, relative_mad=0.002),
        _measurement(102.0, relative_mad=0.008),
    ) == pytest.approx(0.016)


def test_screen_routes_candidates_by_gain_and_priority():
    cases = (
        (106.0, 5, None, RetestDisposition.CONFIRM_ABA),
        (102.5, 5, 1.08, RetestDisposition.CONFIRM_ABBA),
        (100.4, 2, None, RetestDisposition.CONFIRM_ABBA),
        (100.4, 6, None, RetestDisposition.NEEDS_REVIEW),
    )

    for candidate_tps, trace_rank, micro_speedup, expected in cases:
        result = screen_retest_mode(
            _measurement(100.0),
            _measurement(candidate_tps),
            keep_floor=0.01,
            trace_rank=trace_rank,
            micro_speedup=micro_speedup,
        )
        assert result.disposition is expected


def test_aba_and_abba_use_drift_corrected_geometric_gain():
    assert aba_gain(100.0, 110.0, 102.0) == pytest.approx(110.0 / (100.0 * 102.0) ** 0.5)
    assert abba_gain(100.0, 110.0, 112.0, 102.0) == pytest.approx(((110.0 * 112.0) / (100.0 * 102.0)) ** 0.5)


def test_confirmed_retest_keeps_only_past_effective_floor():
    assert decide_confirmed_retest(1.04, 0.02) is RetestDisposition.KEEP
    assert decide_confirmed_retest(1.01, 0.02) is RetestDisposition.NEEDS_REVIEW
    assert decide_confirmed_retest(0.99, 0.02) is RetestDisposition.DROP_CONFIRMED


def test_strong_stable_gain_uses_aba_to_recheck_anchor_drift():
    anchor_first = _measurement(100.0)
    candidate_first = _measurement(108.0)
    anchor_second = _measurement(101.0)
    deactivate = MagicMock()
    measure_candidate = MagicMock()

    result = run_candidate_retest(
        anchor_first,
        candidate_first,
        keep_floor=0.01,
        trace_rank=5,
        micro_speedup=None,
        deactivate=deactivate,
        measure_anchor=MagicMock(return_value=anchor_second),
        measure_candidate=measure_candidate,
    )

    assert result.mode == "aba"
    assert result.anchor_measurements == (anchor_first, anchor_second)
    assert result.candidate_measurements == (candidate_first,)
    deactivate.assert_called_once()
    measure_candidate.assert_not_called()
