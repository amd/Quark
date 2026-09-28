#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for the pure (no GPU/subprocess) functions in perfopt/*.

Design ref: IMPL_SPEC §4.2/decision 3.
"""

from __future__ import annotations

import gzip
import json
import subprocess
import sys

import pytest

from quark.experimental.torch.quant_perf.perfopt.bottleneck_analysis import (
    find_best_existing_trace,
    find_comparable_traces,
)
from quark.experimental.torch.quant_perf.perfopt.keep import (
    aggregate_gain,
    normalize_kernel_sig,
    summarize_patch,
)
from quark.experimental.torch.quant_perf.perfopt.kernel_source import (
    _demangle,
    classify_kernel,
)
from quark.experimental.torch.quant_perf.perfopt.service import (
    OptimizationService,
    _dominant_bound,
)


def _write_kernel_trace(path, events):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": events}, handle)


# -- perfopt/__init__.py ------------------------------------------------------


@pytest.mark.parametrize(
    ("bottlenecks", "expected"),
    [
        ([], ""),
        (
            [
                {"roofline_bound": "COMPUTE_BOUND"},
                {"roofline_bound": "COMPUTE_BOUND"},
                {"roofline_bound": "MEMORY_BOUND"},
            ],
            "COMPUTE_BOUND",
        ),
        ([{"op_name": "a"}, {"op_name": "b"}], ""),
    ],
)
def test_dominant_bound(bottlenecks, expected):
    assert _dominant_bound(bottlenecks) == expected


def test_find_comparable_traces_uses_same_trace_kind_on_both_sides(tmp_path):
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import quark.experimental.torch.quant_perf.perfopt.keep; "
                "assert 'quark.experimental.torch.quant_perf.perfopt.service' not in sys.modules"
            ),
        ],
        capture_output=True,
        text=True,
    )
    assert probe.returncode == 0, probe.stderr

    quant_dir = tmp_path / "quant"
    baseline_dir = tmp_path / "baseline"
    quant_dir.mkdir()
    baseline_dir.mkdir()

    quant_raw = quant_dir / "dp0_rank0.quant.pt.trace.json.gz"
    baseline_raw = baseline_dir / "dp0_rank0.baseline.pt.trace.json.gz"
    baseline_decode = baseline_dir / "decode_only_steady_state_rank0.baseline.json.gz"
    quant_raw.touch()
    baseline_raw.touch()
    baseline_decode.touch()

    quant_trace, baseline_trace, trace_kind = find_comparable_traces(quant_dir, baseline_dir)

    assert trace_kind == "raw"
    assert quant_trace == quant_raw
    assert baseline_trace == baseline_raw


def test_existing_trace_selection_ignores_async_frontend(tmp_path):
    frontend = tmp_path / "host_123.async_llm.1.pt.trace.json.gz"
    engine = tmp_path / "dp0_pp0_tp0_dcp0_ep0_rank0.2.pt.trace.json.gz"
    frontend.touch()
    engine.touch()

    assert find_best_existing_trace(tmp_path) == engine


def test_comparable_trace_selection_rejects_frontend_only_side(tmp_path):
    quant_dir = tmp_path / "quant"
    baseline_dir = tmp_path / "baseline"
    quant_dir.mkdir()
    baseline_dir.mkdir()
    (quant_dir / "host_123.async_llm.1.pt.trace.json.gz").touch()
    (baseline_dir / "dp0_pp0_tp0_dcp0_ep0_rank0.2.pt.trace.json.gz").touch()

    quant_trace, baseline_trace, trace_kind = find_comparable_traces(quant_dir, baseline_dir)

    assert quant_trace is None
    assert baseline_trace is None
    assert trace_kind == ""


def test_absolute_bottleneck_analysis_preserves_ranked_candidates():
    from quark.experimental.torch.quant_perf.perfopt.bottleneck_analysis import (
        BottleneckMode,
        analyze_bottlenecks,
    )

    candidates = [
        {"op_name": "slow", "kernel_time_us": 200.0},
        {"op_name": "fast", "kernel_time_us": 100.0},
    ]

    result = analyze_bottlenecks(
        mode=BottleneckMode.ABSOLUTE,
        quantized_trace="/trace/quant.json.gz",
        absolute_candidates=candidates,
        top_n=2,
    )

    assert result.requested_mode is BottleneckMode.ABSOLUTE
    assert result.effective_mode == "absolute"
    assert [row["op_name"] for row in result.candidates] == [
        "slow",
        "fast",
    ]
    assert {row["selection_mode"] for row in result.candidates} == {"absolute"}
    assert result.baseline_trace == ""


def test_differential_bottleneck_analysis_preserves_current_ranking(
    tmp_path,
):
    from quark.experimental.torch.quant_perf.perfopt.bottleneck_analysis import (
        BottleneckMode,
        analyze_bottlenecks,
    )

    quantized = tmp_path / "quant.json.gz"
    baseline = tmp_path / "baseline.json.gz"
    _write_kernel_trace(
        quantized,
        [
            {
                "cat": "kernel",
                "name": "shared.kd",
                "dur": 120.0,
            },
            {
                "cat": "kernel",
                "name": "quant_only.kd",
                "dur": 80.0,
            },
        ],
    )
    _write_kernel_trace(
        baseline,
        [
            {
                "cat": "kernel",
                "name": "shared.kd",
                "dur": 100.0,
            }
        ],
    )

    result = analyze_bottlenecks(
        mode=BottleneckMode.DIFFERENTIAL,
        quantized_trace=str(quantized),
        baseline_trace=str(baseline),
        trace_kind="raw",
        absolute_candidates=[{"op_name": "fallback", "kernel_time_us": 999.0}],
        top_n=5,
    )

    assert result.effective_mode == "differential"
    assert result.status == "completed"
    assert [row["op_name"] for row in result.candidates] == [
        "shared.kd",
        "quant_only.kd",
    ]
    assert result.candidates[0]["differential_type"] == ("slower_in_quantized")
    assert result.candidates[1]["differential_type"] == ("unique_to_quantized")


def test_empty_differential_analysis_keeps_existing_single_fallback(
    tmp_path,
):
    from quark.experimental.torch.quant_perf.perfopt.bottleneck_analysis import (
        BottleneckMode,
        analyze_bottlenecks,
    )

    quantized = tmp_path / "quant.json.gz"
    baseline = tmp_path / "baseline.json.gz"
    events = [{"cat": "kernel", "name": "shared.kd", "dur": 100.0}]
    _write_kernel_trace(quantized, events)
    _write_kernel_trace(baseline, events)

    result = analyze_bottlenecks(
        mode=BottleneckMode.DIFFERENTIAL,
        quantized_trace=str(quantized),
        baseline_trace=str(baseline),
        trace_kind="raw",
        absolute_candidates=[
            {"op_name": "first", "kernel_time_us": 200.0},
            {"op_name": "second", "kernel_time_us": 100.0},
        ],
        top_n=5,
    )

    assert result.effective_mode == "absolute_fallback"
    assert result.status == "degraded"
    assert result.reason == "no_differential_candidates"
    assert [row["op_name"] for row in result.candidates] == ["first"]


def test_geak_candidate_eligibility_is_kernel_level_only(tmp_path):
    perfopt = OptimizationService(experience_store=None)
    patch_file = tmp_path / "candidate.diff"
    patch_file.write_text("diff --git a/kernel.py b/kernel.py\n")

    assert perfopt._keep(
        {
            "best_patch": str(patch_file),
            "verified_speedup": 1.08,
            "round_evaluation": {"correctness": {"success": True}},
        }
    )
    assert not perfopt._keep(
        {
            "best_patch": str(patch_file),
            "verified_speedup": 1.08,
            "round_evaluation": {"correctness": {"success": False}},
        }
    )
    assert not perfopt._keep(
        {
            "best_patch": str(patch_file),
            "verified_speedup": 1.08,
            "round_evaluation": {},
        }
    )
    assert not perfopt._keep(
        {
            "best_patch": str(patch_file),
            "verified_speedup": 1.08,
            "round_evaluation": {"correctness": {"success": None}},
        }
    )
    assert perfopt._keep(
        {
            "best_patch": str(patch_file),
            "verified_speedup": None,
            "micro_speedup_source": "unmeasured_verified_patch",
            "round_evaluation": {"correctness": {"success": True}},
        }
    )
    assert not perfopt._keep({"best_patch": "", "verified_speedup": 1.08})


# -- kernel_source.py --------------------------------------------------------


def test_demangle_strips_hash_suffix():
    assert _demangle("my_gemm_kernel_0a1b2c3d4e") == "my_gemm_kernel"


def test_demangle_strips_namespace_prefix():
    assert _demangle("hipDeviceLib::rms_norm_kernel") == "rms_norm_kernel"


def test_classify_kernel_not_rewritable_for_torch_compile_generated():
    # torch.compile-generated Triton kernels have no stable source file to patch
    rewritable, reason = classify_kernel("triton_poi_fused_add_relu_0")
    assert not rewritable
    assert "torch.compile" in reason
    rewritable, _ = classify_kernel("triton_red_fused_sum_1")
    assert not rewritable


def test_classify_kernel_not_rewritable_for_vendor_libraries():
    # hipBLAS/rocBLAS kernels are precompiled vendor binaries
    rewritable, reason = classify_kernel("hipblaslt_gemm_kernel")
    assert not rewritable
    assert reason


# -- keep.py --------------------------------------------------------------


def test_normalize_kernel_sig_strips_hash_and_namespace():
    assert normalize_kernel_sig("hipDeviceLib::my_gemm_kernel_0a1b2c3d4e") == "my_gemm_kernel"


def test_normalize_kernel_sig_strips_shape_bracket_suffix():
    assert normalize_kernel_sig("my_gemm_kernel[128,4096,4096]") == "my_gemm_kernel"


def test_summarize_patch_returns_first_non_diff_line(tmp_path):
    patch = tmp_path / "patch.diff"
    patch.write_text(
        "# Use tl.dot for the inner loop\n"
        "diff --git a/x b/x\nindex 111..222\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-old\n+new\n"
    )
    assert summarize_patch(str(patch)) == "Use tl.dot for the inner loop"


def test_summarize_patch_missing_file_returns_fallback(tmp_path):
    assert summarize_patch(str(tmp_path / "missing.diff")) == "no summary available"


def test_aggregate_gain_multiplies_speedups():
    results = [{"verified_speedup": 1.2}, {"verified_speedup": 1.5}]
    assert aggregate_gain(results) == pytest.approx(1.8)


def test_aggregate_gain_empty_results_is_one():
    assert aggregate_gain([]) == 1.0


def test_aggregate_gain_ignores_unmeasured_verified_patches():
    assert aggregate_gain(
        [
            {"verified_speedup": None},
            {"verified_speedup": 1.2},
        ]
    ) == pytest.approx(1.2)
