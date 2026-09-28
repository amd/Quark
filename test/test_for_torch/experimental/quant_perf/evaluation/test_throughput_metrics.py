#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for the pure throughput-metric helpers in eval_utils
(num_prompts_for, throughput_gain). These carry the benchmark-methodology
contract and must stay correct independent of any GPU."""

from __future__ import annotations

from quark.experimental.torch.quant_perf.evaluation.throughput import (
    num_prompts_for,
    throughput_gain,
)


def test_num_prompts_factor_schedule_by_seq_cost():
    # seq_cost <= 1024 -> factor 10
    assert num_prompts_for(64, 512, 512) == 640
    # <= 4096 -> factor 5  (1024+1024=2048)
    assert num_prompts_for(64, 1024, 1024) == 320
    # <= 16384 -> factor 3 (8192+1024=9216, a long prefill scenario)
    assert num_prompts_for(64, 8192, 1024) == 192
    # else -> factor 2 (very long, e.g. 16384+8192)
    assert num_prompts_for(64, 16384, 8192) == 128


def test_num_prompts_never_below_concurrency():
    # boundary exactly 1024 uses factor 10, but the floor still holds generally
    assert num_prompts_for(100, 8192, 1024) >= 100
    assert num_prompts_for(1, 32768, 8192) >= 1


def test_throughput_gain_basic_ratio():
    assert throughput_gain(1159.8, 1059.5) == 1159.8 / 1059.5


def test_throughput_gain_guards_invalid_inputs():
    # a failed/zero measurement must read as "no gain" (0.0), never as success
    assert throughput_gain(0.0, 1000.0) == 0.0
    assert throughput_gain(1000.0, 0.0) == 0.0
    assert throughput_gain(-5.0, 1000.0) == 0.0
    assert throughput_gain(1000.0, -1.0) == 0.0


def test_throughput_gain_below_and_above_one():
    assert throughput_gain(900.0, 1000.0) < 1.0  # regression
    assert throughput_gain(1500.0, 1000.0) == 1.5  # +50%
