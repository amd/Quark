#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for perfopt/geak.py's pure functions (build_geak_task,
get_quant_detail, _match_layer_config), plus a delegation check for run_geak
(now a thin adapter over the single-kernel kernel_workflow driver).

Design ref: IMPL_SPEC §2.3.1/§4.2.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from quark.experimental.torch.quant_perf.perfopt.geak import (
    _match_layer_config,
    build_geak_task,
    get_quant_detail,
    run_geak,
)
from quark.experimental.torch.quant_perf.perfopt.service import OptimizationService
from quark.experimental.torch.quant_perf.perfopt.workload_contract import WorkloadContract


@pytest.mark.parametrize(
    "symbol,blocked",
    [
        ("_ZN5aiter40allreduce_fusion_kernel_2stage_per_groupIDF16bDB8_Li8EEEv", True),
        ("reduce_scatter_cross_device_store<bf16,8>", True),
        ("void ck::kernel_moe<ck::AllReduce>(ck::Argument)", False),
        ("_ZN2ck10kernel_moeINS_9AllReduceEEEv", False),
        ("_ZunknownAllReduce", False),
    ],
)
def test_geak_checks_kernel_execution_requirements(monkeypatch, tmp_path, symbol, blocked):
    from quark.experimental.torch.quant_perf.perfopt import kernel_workflow

    calls = []
    monkeypatch.setattr(kernel_workflow, "run_kernel_workflow", lambda **kwargs: calls.append(kwargs) or {})
    report = run_geak("/aiter/kernel.cuh", "Model uses TP8", "model", 0, str(tmp_path), kernel_name=symbol)
    assert bool(calls) is not blocked
    if blocked:
        assert report["execution_status"] == "skipped"
        assert report["verified_speedup"] is None
        assert "multi-GPU" in report["error"]


@patch("quark.experimental.torch.quant_perf.perfopt.kernel_workflow.run_kernel_workflow")
@patch("quark.experimental.torch.quant_perf.perfopt.kernel_workflow.gen_workload_spec")
def test_run_geak_delegates_to_kernel_workflow(mock_genspec, mock_run_kw, tmp_path):
    """run_geak resolves a workload spec from the run's trace and hands the
    SPECIFIC kernel to kernel_workflow (no whole-model run_e2e handoff)."""
    run_dir = tmp_path / "run"
    mock_genspec.return_value = str(run_dir / "workload.json")
    mock_run_kw.return_value = {
        "verified_speedup": 1.2,
        "best_patch": "/x/current_best.diff",
        "watchdog_status": "success",
    }

    binding = {
        "source_file": "/vllm/csrc/common.cu",
        "source_symbol": "scaled_fp8_quant_kernel",
        "method": "unique_definition",
    }
    out = run_geak(
        "/vllm/csrc/common.cu",
        "task-hint",
        "model",
        2,
        str(run_dir),
        session_dir=str(tmp_path / "sess"),
        kernel_name="scaled_fp8_quant_kernel",
        source_binding=binding,
        knowledge_ids=["kernel.flydsl.productionization.v1"],
        budget=3,
        timeout_s=120,
    )

    # delegates to the single-kernel driver with the chosen kernel as target
    assert mock_run_kw.called
    kwargs = mock_run_kw.call_args.kwargs
    assert kwargs["kernel_path"] == "/vllm/csrc/common.cu"
    assert kwargs["source_repo"] == ""
    assert kwargs["budget"] == 3
    assert kwargs["hard_timeout_s"] == 120
    assert out["verified_speedup"] == 1.2 and out["best_patch"].endswith("current_best.diff")
    assert json.loads((run_dir / "source_binding.json").read_text()) == binding
    audit = json.loads((tmp_path / "sess" / "llm_calls.jsonl").read_text())
    assert audit["call_type"] == "kernel_optimization"
    assert audit["knowledge_ids"] == ["kernel.flydsl.productionization.v1"]


@patch("quark.experimental.torch.quant_perf.perfopt.kernel_workflow.run_kernel_workflow")
@patch("quark.experimental.torch.quant_perf.perfopt.kernel_workflow.gen_workload_spec")
def test_run_geak_prefers_eager_shape_evidence(
    mock_genspec,
    mock_run_kw,
    tmp_path,
):
    session_dir = tmp_path / "session"
    (session_dir / "trace_shape_evidence").mkdir(parents=True)
    (session_dir / "trace").mkdir()
    run_dir = tmp_path / "run"
    workload = run_dir / "workload.json"

    def generate(trace_dir, _kernel_name, _out_path):
        if trace_dir == str(session_dir / "trace_shape_evidence"):
            workload.parent.mkdir(parents=True, exist_ok=True)
            workload.write_text(
                json.dumps(
                    {
                        "schema": "workload-v1",
                        "kernels": [
                            {
                                "name": "generic_kernel.kd",
                                "cases": [
                                    {
                                        "dims": [[32, 2048]],
                                        "count": 1,
                                    }
                                ],
                            }
                        ],
                    }
                )
            )
            return str(workload)
        return None

    mock_genspec.side_effect = generate
    mock_run_kw.return_value = {
        "verified_speedup": 1.1,
        "best_patch": "/x/current_best.diff",
        "watchdog_status": "success",
    }

    run_geak(
        "/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py",
        "task-hint",
        "model",
        0,
        str(run_dir),
        session_dir=str(session_dir),
        kernel_name="mfma_moe1_silu_mul_afp4_wfp4_bf16",
    )

    assert mock_genspec.call_args_list[0].args[0] == str(session_dir / "trace_shape_evidence")
    contract = mock_run_kw.call_args.kwargs["workload_contract"]
    assert isinstance(contract, WorkloadContract)
    assert contract.spec_path == str(workload)


@patch("quark.experimental.torch.quant_perf.perfopt.kernel_workflow.run_kernel_workflow")
@patch("quark.experimental.torch.quant_perf.perfopt.kernel_workflow.gen_workload_spec")
def test_run_geak_injects_authoritative_workload_and_source_contract(
    mock_genspec,
    mock_run_kw,
    tmp_path,
):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    workload = run_dir / "workload.json"
    workload.write_text(
        json.dumps(
            {
                "schema": "workload-v1",
                "kernels": [
                    {
                        "name": "mfma_moe2.kd",
                        "cases": [
                            {
                                "dims": [
                                    [32, 2048],
                                    [256, 1024, 1024],
                                    [256, 2048, 256],
                                    [32, 8],
                                ],
                                "dtypes": [
                                    "c10::BFloat16",
                                    "c10::Float4_e2m1fn_x2",
                                    "c10::Float4_e2m1fn_x2",
                                    "float",
                                ],
                                "count": 440,
                                "weight_source": "trace",
                            }
                        ],
                    }
                ],
            }
        )
    )
    mock_genspec.return_value = str(workload)
    mock_run_kw.return_value = {
        "verified_speedup": 1.0,
        "best_patch": "",
        "watchdog_status": "success",
    }
    binding = {
        "source_repo": "/managed/aiter",
        "source_file": ("/managed/aiter/aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py"),
        "source_relpath": ("aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py"),
        "compiler": "flydsl",
        "builder_symbol": "compile_mixed_moe_gemm",
        "launcher_source_file": ("/managed/aiter/aiter/ops/flydsl/moe_kernels.py"),
    }

    run_geak(
        binding["source_file"],
        "optimize the selected kernel",
        "model",
        0,
        str(run_dir),
        session_dir=str(tmp_path / "session"),
        kernel_name="mfma_moe2.kd",
        source_binding=binding,
    )

    task = mock_run_kw.call_args.kwargs["task"]
    assert "Authoritative workload contract" in task
    assert ('"dims": [[32, 2048], [256, 1024, 1024], [256, 2048, 256], [32, 8]]') in task
    assert '"count": 440' in task
    assert '"operator_family": "flydsl_mixed_moe_stage2"' in task
    assert '"model_dim": 2048' in task
    assert '"inter_dim": 512' in task
    assert '"experts": 256' in task
    assert '"topk": 8' in task
    assert '"token_buckets": [32]' in task
    assert "override source examples and default model dimensions" in task
    assert "within 25%" in task
    assert "workload_aligned=false" in task
    assert "/managed/aiter/aiter/ops/flydsl/moe_kernels.py" in task
    assert "Do not use another checkout" in task
    contract = mock_run_kw.call_args.kwargs["workload_contract"]
    assert isinstance(contract, WorkloadContract)
    assert contract.logical_semantics == {
        "operator_family": "flydsl_mixed_moe_stage2",
        "model_dim": 2048,
        "inter_dim": 512,
        "experts": 256,
        "topk": 8,
        "token_buckets": [32],
        "packed_fp4x2_last_dim_multiplier": 2,
    }


def test_workload_contract_keeps_generic_cases_without_family_semantics(
    tmp_path,
):
    workload = tmp_path / "workload.json"
    workload.write_text(
        json.dumps(
            {
                "schema": "workload-v1",
                "kernels": [
                    {
                        "name": "kernel_paged_attention_2d.kd",
                        "cases": [
                            {
                                "dims": [[32, 128], [32, 128]],
                                "dtypes": ["bf16", "bf16"],
                                "count": 100,
                            }
                        ],
                    }
                ],
            }
        )
    )

    contract = WorkloadContract.from_spec(
        str(workload),
        {
            "compiler": "triton",
            "source_relpath": "vllm/attention/paged_attention.py",
        },
    )

    assert contract is not None
    assert contract.logical_semantics == {}
    assert contract.cases[0]["count"] == 100
    assert "Authoritative workload contract" in contract.prompt_suffix()


def test_perfopt_keep_accepts_verified_patch_without_numeric_micro(tmp_path):
    patch = tmp_path / "candidate.diff"
    patch.write_text("diff --git a/kernel.py b/kernel.py\n+optimized\n")

    assert OptimizationService._keep(
        {
            "best_patch": str(patch),
            "verified_speedup": None,
            "micro_speedup_source": "unmeasured_verified_patch",
            "round_evaluation": {"correctness": {"success": True}},
        }
    )


def test_match_layer_config_finds_pattern_by_substring():
    qcfg = {"layer_quant_config": {"model.layers.0.mlp.gate_proj": {"dtype": "fp8"}}}
    assert _match_layer_config(qcfg, "model.layers.0.mlp.gate_proj") == {"dtype": "fp8"}


def test_match_layer_config_returns_none_when_no_match():
    qcfg = {"layer_quant_config": {"model.layers.0.mlp.gate_proj": {"dtype": "fp8"}}}
    assert _match_layer_config(qcfg, "totally_unrelated_op") is None


def test_get_quant_detail_uses_matched_layer_config(tmp_path):
    cfg = {
        "quantization_config": {
            "layer_quant_config": {"gate_proj": {"dtype": "fp8", "qscheme": "per_channel", "symmetric": True}},
            "global_quant_config": {"dtype": "bf16"},
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    detail = get_quant_detail(str(tmp_path), "gate_proj")
    assert "fp8" in detail
    assert "per_channel" in detail
    assert "symmetric" in detail


def test_get_quant_detail_falls_back_to_global_config(tmp_path):
    cfg = {
        "quantization_config": {
            "layer_quant_config": {},
            "global_quant_config": {
                "dtype": "mxfp4",
                "qscheme": "per_group",
                "symmetric": False,
                "group_size": 32,
            },
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    detail = get_quant_detail(str(tmp_path), "unmatched_op")
    assert "mxfp4" in detail
    assert "asymmetric" in detail
    assert "group_size=32" in detail


def test_build_geak_task_includes_bound_quant_and_shape(tmp_path):
    cfg = {"quantization_config": {"global_quant_config": {"dtype": "fp8", "symmetric": True}}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    bn = {
        "op_name": "my_gemm_kernel",
        "roofline_bound": "COMPUTE_BOUND",
        "sol_pct": 40.0,
        "gemm_shape": {"M": 128, "N": 4096, "K": 4096},
        "kernel_time_us": 12.3,
    }
    task = build_geak_task(
        bn,
        "fp8",
        str(tmp_path),
    )
    assert "my_gemm_kernel" in task
    assert "COMPUTE_BOUND" in task
    assert "tensor core" in task
    assert "fp8" in task
    assert "M=128" in task
    assert "N=4096" in task
    assert "K=4096" in task
    assert "kernel_time_us=12.3" in task
    assert "harness USER TASK CONTEXT" not in task


def test_build_geak_task_memory_bound_hint(tmp_path):
    cfg = {"quantization_config": {"global_quant_config": {"dtype": "bf16"}}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    bn = {"op_name": "my_norm_kernel", "roofline_bound": "MEMORY_BOUND", "sol_pct": 60.0}
    task = build_geak_task(bn, None, str(tmp_path))
    assert "HBM bandwidth" in task
    assert "Historical note" not in task


def test_build_geak_task_includes_structured_knowledge_context(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"global_quant_config": {"dtype": "mxfp4"}}})
    )
    task = build_geak_task(
        {
            "op_name": "flydsl_gemm",
            "roofline_bound": "COMPUTE_BOUND",
        },
        "mxfp4",
        str(tmp_path),
        knowledge_text=("## Retrieved knowledge\n- [kernel.flydsl.productionization.v1] prewarm every shape"),
    )

    assert "kernel.flydsl.productionization.v1" in task
    assert "prewarm every shape" in task


def test_build_geak_task_includes_source_binding_evidence(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"quantization_config": {"global_quant_config": {"dtype": "mxfp4"}}})
    )
    bn = {
        "op_name": "kernel_gemm_0.kd",
        "kernel_names": ["kernel_gemm_0.kd"],
        "parent_op_name": ("vllm::rocm_aiter_flydsl_gemm_a8w4_per_tensor"),
        "source_resolution": {
            "source_symbol": "kernel_gemm",
            "builder_symbol": "launch_gemm",
            "launcher_source_file": "/aiter/batched_gemm_mxfp4.py",
            "live_call_seam": ("vllm::rocm_aiter_flydsl_gemm_a8w4_per_tensor"),
            "compiler": "flydsl",
            "method": "operator_rule",
            "confidence": "operator_rule",
        },
    }

    task = build_geak_task(
        bn,
        "mxfp4_fp8",
        str(tmp_path),
        source_file="/aiter/mxfp4_preshuffle.py",
    )

    assert "Parent CPU operator" in task
    assert "vllm::rocm_aiter_flydsl_gemm_a8w4_per_tensor" in task
    assert "Source symbol: `kernel_gemm`" in task
    assert "Builder symbol: `launch_gemm`" in task
    assert "Resolution: `operator_rule` / `operator_rule`" in task
