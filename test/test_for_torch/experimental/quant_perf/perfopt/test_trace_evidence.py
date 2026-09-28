#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import gzip
import json

from quark.experimental.torch.quant_perf.perfopt.trace_evidence import (
    compare_differential_evidence,
    enrich_bottlenecks_with_eager_launchers,
    enrich_bottlenecks_with_kernel_evidence,
    extract_kernel_evidence,
)


def _write_trace(path, events):
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": events}, handle)


def test_extract_kernel_evidence_preserves_parent_op_without_input_dims(tmp_path):
    trace = tmp_path / "trace.json.gz"
    _write_trace(
        trace,
        [
            {
                "cat": "cpu_op",
                "name": "vllm::rocm_aiter_flydsl_gemm_a8w4_per_tensor",
                "args": {"External id": 487},
            },
            {
                "cat": "kernel",
                "name": "kernel_gemm_0.kd",
                "dur": 29.8,
                "args": {"External id": 487},
            },
        ],
    )

    evidence = extract_kernel_evidence(trace)["kernel_gemm_0.kd"]

    assert evidence.primary_parent_op == ("vllm::rocm_aiter_flydsl_gemm_a8w4_per_tensor")
    assert evidence.parent_op_names == ("vllm::rocm_aiter_flydsl_gemm_a8w4_per_tensor",)
    assert evidence.external_ids == (487,)
    assert evidence.call_count == 1
    assert evidence.total_time_us == 29.8


def test_differential_comparison_preserves_quantized_evidence(tmp_path):
    quant_trace = tmp_path / "quant.json.gz"
    base_trace = tmp_path / "base.json.gz"
    _write_trace(
        quant_trace,
        [
            {
                "cat": "cpu_op",
                "name": "aiter::fused_moe_",
                "args": {
                    "External id": 751,
                    "Input Dims": [[16, 10, 4096]],
                    "Input type": ["BFloat16"],
                },
            },
            {
                "cat": "kernel",
                "name": "moe_reduction_kernel_plain_bf16_topk10_md4096.kd",
                "dur": 30.0,
                "args": {"External id": 751},
            },
        ],
    )
    _write_trace(base_trace, [])

    differential = compare_differential_evidence(
        extract_kernel_evidence(quant_trace),
        extract_kernel_evidence(base_trace),
        top_n=5,
    )

    assert len(differential) == 1
    candidate = differential[0]
    assert candidate["op_name"] == ("moe_reduction_kernel_plain_bf16_topk10_md4096.kd")
    assert candidate["parent_op_name"] == "aiter::fused_moe_"
    assert candidate["parent_op_names"] == ["aiter::fused_moe_"]
    assert candidate["call_count"] == 1
    assert candidate["differential_type"] == "unique_to_quantized"
    assert candidate["baseline_time_us"] == 0.0
    assert candidate["shape_cases"]
    assert candidate["dtypes"] == ["BFloat16"]


def test_exact_sidecar_evidence_adds_gemm_shape_without_replacing_timing(
    tmp_path,
):
    trace = tmp_path / "sidecar.json.gz"
    _write_trace(
        trace,
        [
            {
                "cat": "cpu_op",
                "name": "aten::_scaled_mm",
                "args": {
                    "External id": 8,
                    "Input Dims": [[64, 2048], [12288, 2048]],
                    "Input type": ["Float8_e4m3fn", "Float8_e4m3fn"],
                },
            },
            {
                "cat": "kernel",
                "name": "Cijk_selected.kd",
                "dur": 12.0,
                "args": {"External id": 8},
            },
        ],
    )
    bottlenecks = [
        {
            "op_name": "Cijk_selected.kd",
            "kernel_time_us": 900.0,
            "baseline_time_us": 0.0,
        }
    ]

    enrich_bottlenecks_with_kernel_evidence(
        bottlenecks,
        extract_kernel_evidence(trace),
        source="eager_sidecar",
        trace_path=trace,
    )

    assert bottlenecks[0]["kernel_time_us"] == 900.0
    assert bottlenecks[0]["parent_op_name"] == "aten::_scaled_mm"
    assert bottlenecks[0]["gemm_shape"] == {
        "M": 64,
        "N": 12288,
        "K": 2048,
    }
    assert bottlenecks[0]["gemm_shapes"] == [{"M": 64, "N": 12288, "K": 2048}]
    assert bottlenecks[0]["gemm_shape_evidence"]["status"] == "exact"
    assert bottlenecks[0]["gemm_shape_evidence"]["source"] == ("eager_sidecar")


def test_multiple_scaled_mm_shapes_are_exact_workloads(tmp_path):
    trace = tmp_path / "sidecar.json.gz"
    _write_trace(
        trace,
        [
            {
                "cat": "cpu_op",
                "name": "aten::_scaled_mm",
                "args": {
                    "External id": 8,
                    "Input Dims": [
                        [32, 2048],
                        [2048, 12288],
                        [32, 1],
                        [1, 12288],
                    ],
                },
            },
            {
                "cat": "kernel",
                "name": "Cijk_selected.kd",
                "dur": 12.0,
                "args": {"External id": 8},
            },
            {
                "cat": "cpu_op",
                "name": "aten::_scaled_mm",
                "args": {
                    "External id": 9,
                    "Input Dims": [
                        [64, 2048],
                        [2048, 12288],
                        [64, 1],
                        [1, 12288],
                    ],
                },
            },
            {
                "cat": "kernel",
                "name": "Cijk_selected.kd",
                "dur": 13.0,
                "args": {"External id": 9},
            },
        ],
    )
    bottlenecks = [{"op_name": "Cijk_selected.kd"}]

    enrich_bottlenecks_with_kernel_evidence(
        bottlenecks,
        extract_kernel_evidence(trace),
        source="eager_sidecar",
        trace_path=trace,
    )

    assert bottlenecks[0]["gemm_shape"] == {
        "M": None,
        "N": None,
        "K": None,
    }
    assert bottlenecks[0]["gemm_shapes"] == [
        {"M": 32, "N": 12288, "K": 2048},
        {"M": 64, "N": 12288, "K": 2048},
    ]
    evidence = bottlenecks[0]["gemm_shape_evidence"]
    assert evidence["status"] == "exact"
    assert evidence["shapes"] == [
        {"M": 32, "N": 12288, "K": 2048},
        {"M": 64, "N": 12288, "K": 2048},
    ]
    assert evidence["alternatives"] == []


def test_generic_shape_case_with_multiple_matrix_pairs_is_ambiguous(
    tmp_path,
):
    trace = tmp_path / "sidecar.json.gz"
    _write_trace(
        trace,
        [
            {
                "cat": "cpu_op",
                "name": "custom::gemm_like",
                "args": {
                    "External id": 8,
                    "Input Dims": [
                        [32, 2048],
                        [2048, 12288],
                        [32, 1],
                        [1, 12288],
                    ],
                },
            },
            {
                "cat": "kernel",
                "name": "generic_selected.kd",
                "dur": 12.0,
                "args": {"External id": 8},
            },
        ],
    )
    bottlenecks = [{"op_name": "generic_selected.kd"}]

    enrich_bottlenecks_with_kernel_evidence(
        bottlenecks,
        extract_kernel_evidence(trace),
        source="eager_sidecar",
        trace_path=trace,
    )

    assert bottlenecks[0]["gemm_shapes"] == []
    assert bottlenecks[0]["gemm_shape_evidence"]["status"] == ("ambiguous")


def test_eager_launcher_evidence_records_innermost_user_frame(tmp_path):
    trace = tmp_path / "trace.json.gz"
    _write_trace(
        trace,
        [
            {
                "cat": "python_function",
                "name": "/repo/model.py(10): forward",
                "pid": 1,
                "tid": 7,
                "ts": 100.0,
                "dur": 100.0,
            },
            {
                "cat": "python_function",
                "name": "/repo/kernels/gemm.py(42): launch_gemm",
                "pid": 1,
                "tid": 7,
                "ts": 110.0,
                "dur": 80.0,
            },
            {
                "cat": "python_function",
                "name": "flydsl/compiler/jit_executor.py(210): __call__",
                "pid": 1,
                "tid": 7,
                "ts": 120.0,
                "dur": 60.0,
            },
            {
                "cat": "cuda_runtime",
                "name": "hipModuleLaunchKernel",
                "pid": 1,
                "tid": 7,
                "ts": 150.0,
                "dur": 1.0,
                "args": {"correlation": 9},
            },
            {
                "cat": "kernel",
                "name": "kernel_gemm_0.kd",
                "ts": 160.0,
                "dur": 10.0,
                "args": {"correlation": 9},
            },
        ],
    )
    bottlenecks = [
        {
            "op_name": "kernel_gemm_0.kd",
            "kernel_names": ["kernel_gemm_0.kd"],
        }
    ]

    enrich_bottlenecks_with_eager_launchers(bottlenecks, trace)

    assert bottlenecks[0]["launcher_source_file"] == ("/repo/kernels/gemm.py")
    assert bottlenecks[0]["launcher_line"] == 42
    assert bottlenecks[0]["launcher_symbol"] == "launch_gemm"
    assert bottlenecks[0]["launcher_evidence_method"] == ("eager_trace_python_stack")


def test_graph_replay_does_not_create_launcher_evidence(tmp_path):
    trace = tmp_path / "trace.json.gz"
    _write_trace(
        trace,
        [
            {
                "cat": "python_function",
                "name": "/repo/model.py(10): forward",
                "pid": 1,
                "tid": 7,
                "ts": 100.0,
                "dur": 100.0,
            },
            {
                "cat": "cuda_runtime",
                "name": "hipGraphLaunch",
                "pid": 1,
                "tid": 7,
                "ts": 150.0,
                "dur": 1.0,
                "args": {"correlation": 9},
            },
            {
                "cat": "kernel",
                "name": "kernel_gemm_0.kd",
                "ts": 160.0,
                "dur": 10.0,
                "args": {"correlation": 9},
            },
        ],
    )
    bottlenecks = [
        {
            "op_name": "kernel_gemm_0.kd",
            "kernel_names": ["kernel_gemm_0.kd"],
        }
    ]

    enrich_bottlenecks_with_eager_launchers(bottlenecks, trace)

    assert "launcher_source_file" not in bottlenecks[0]
    assert bottlenecks[0]["trace_parent_status"] == ("graph_replay_unavailable")


def test_graph_replay_marks_every_kernel_sharing_one_correlation(
    tmp_path,
):
    trace = tmp_path / "trace.json.gz"
    _write_trace(
        trace,
        [
            {
                "cat": "cuda_runtime",
                "name": "hipGraphLaunch",
                "pid": 1,
                "tid": 7,
                "ts": 150.0,
                "dur": 1.0,
                "args": {"correlation": 9},
            },
            {
                "cat": "kernel",
                "name": "kernel_gemm_0.kd",
                "ts": 160.0,
                "dur": 10.0,
                "args": {"correlation": 9},
            },
            {
                "cat": "kernel",
                "name": "quant_kernel.kd",
                "ts": 171.0,
                "dur": 5.0,
                "args": {"correlation": 9},
            },
        ],
    )
    bottlenecks = [
        {"op_name": "kernel_gemm_0.kd", "kernel_names": []},
        {"op_name": "quant_kernel.kd", "kernel_names": []},
    ]

    enrich_bottlenecks_with_eager_launchers(bottlenecks, trace)

    assert {row.get("trace_parent_status") for row in bottlenecks} == {"graph_replay_unavailable"}
