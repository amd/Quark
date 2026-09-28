#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for the deterministic Amdahl preflight veto."""

from __future__ import annotations

from quark.experimental.torch.quant_perf.perfopt.guardrails import (
    amdahl_ceiling,
    should_attempt_geak,
)


def test_amdahl_ceiling_full_program_optimizable_equals_speedup():
    assert amdahl_ceiling(p=1.0, s=2.0) == 2.0


def test_amdahl_ceiling_nothing_optimizable_is_one():
    assert amdahl_ceiling(p=0.0, s=100.0) == 1.0


def test_amdahl_ceiling_zero_speedup_is_safe_fallback():
    assert amdahl_ceiling(p=0.5, s=0.0) == 1.0


def test_should_attempt_geak_vetoes_tiny_kernel_share():
    # a kernel that's 0.01% of the profiled window can't meaningfully move E2E
    # even with 1.3x per-kernel speedup (Amdahl ceiling < min_e2e_gain).
    assert should_attempt_geak(kernel_time_us=1, total_time_us=1_000_000) is False


def test_should_attempt_geak_allows_dominant_kernel():
    assert should_attempt_geak(kernel_time_us=800, total_time_us=1000) is True


def test_should_attempt_geak_fails_open_when_total_time_unknown():
    assert should_attempt_geak(kernel_time_us=10, total_time_us=0) is True
