#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import gzip
import hashlib
import json
import pickle
import tempfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from quark.experimental.torch.quant_perf.orchestration.orchestrator import Orchestrator
from quark.experimental.torch.quant_perf.perfopt.keep import make_kernel_id
from quark.experimental.torch.quant_perf.perfopt.kernel_source import (
    MappingKind,
    ResolutionConfidence,
    SourceResolution,
)
from quark.experimental.torch.quant_perf.perfopt.service import OptimizationService
from quark.experimental.torch.quant_perf.session.spec import Checkpoint, Spec

from ..testing import init_git_repo
from ..testing import run_git as _git


def _repo(tmp_path: Path) -> Path:
    return init_git_repo(
        tmp_path / "aiter",
        {"kernel.cu": "__global__ void selected_kernel(float* x) { x[0] = x[0]; }\n"},
    )


def _spec(tmp_path: Path, repo: Path) -> Spec:
    return Spec(
        model_dir="/models/quant",
        base_model="/models/base",
        framework="vllm",
        gpu_type="mi350x",
        gpu_arch="MI355X",
        isl=128,
        osl=128,
        quant_strategy="mxfp4",
        session_dir=str(tmp_path / "session"),
        kernel_repo=str(repo),
        top_kernels=1,
    )


def _write_trace(path: Path, events: list[dict]) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"traceEvents": events}, handle)


@patch("quark.experimental.torch.quant_perf.perfopt.service.collect_trace")
@patch("quark.experimental.torch.quant_perf.perfopt.service.landing.load")
def test_graph_replayed_kernel_uses_shared_eager_shape_sidecar(
    mock_load,
    mock_collect_trace,
    tmp_path,
):
    spec = _spec(tmp_path, tmp_path / "aiter")
    quant = Path(spec.quant_ckpt_dir)
    quant.mkdir(parents=True)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["quant_ckpt_dir"] = str(quant)
    trace = tmp_path / "shape-sidecar.json.gz"
    kernel_name = "mfma_moe2_afp4_wfp4_bf16_cshuffle_t32x128x256_vscale_fix3_fp4opt_v1_pm1_acc0.kd"
    _write_trace(
        trace,
        [
            {
                "cat": "cpu_op",
                "name": "aiter::fused_moe_",
                "args": {
                    "External id": 11,
                    "Input Dims": [
                        [32, 2048],
                        [256, 1024, 1024],
                        [256, 2048, 256],
                        [32, 8],
                    ],
                    "Input type": [
                        "BFloat16",
                        "Float4_e2m1fn_x2",
                        "Float4_e2m1fn_x2",
                        "Float",
                    ],
                },
            },
            {
                "cat": "kernel",
                "name": kernel_name,
                "dur": 12.0,
                "args": {"External id": 11},
            },
        ],
    )
    captured = {}
    server = MagicMock(port=19080)

    def fake_load(model_dir, sidecar_spec, profiler_dir):
        captured["model_dir"] = model_dir
        captured["spec"] = sidecar_spec
        captured["profiler_dir"] = profiler_dir
        return server

    mock_load.side_effect = fake_load
    mock_collect_trace.return_value = trace
    bottlenecks = [
        {
            "op_name": kernel_name,
            "kernel_names": [kernel_name],
            "shape_cases": [],
            "trace_parent_status": "graph_replay_unavailable",
        }
    ]

    OptimizationService(experience_store=None)._enrich_selected_kernel_shape_evidence(
        bottlenecks,
        spec,
        ckpt,
    )

    assert captured["model_dir"] == str(quant)
    assert "--enforce-eager" in captured["spec"].vllm_extra_args
    mock_collect_trace.assert_called_once_with(
        19080,
        profiler_dir=captured["profiler_dir"],
        isl=spec.isl,
        osl=16,
        warmup_steps=2,
        profile_steps=32,
        extract_steady_state=False,
    )
    server.stop.assert_called_once()
    assert bottlenecks[0]["shape_cases"] == [
        [
            [32, 2048],
            [256, 1024, 1024],
            [256, 2048, 256],
            [32, 8],
        ]
    ]


@patch("quark.experimental.torch.quant_perf.perfopt.service.collect_trace")
@patch("quark.experimental.torch.quant_perf.perfopt.service.landing.load")
def test_vendor_shape_sidecar_uses_eager_trace_selected_signatures(
    mock_load,
    mock_collect_trace,
    tmp_path,
):
    spec = _spec(tmp_path, tmp_path / "aiter")
    quant = Path(spec.quant_ckpt_dir)
    quant.mkdir(parents=True)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["quant_ckpt_dir"] = str(quant)
    trace = tmp_path / "shape-sidecar.json.gz"
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
    captured = {}
    server = MagicMock(port=19080)

    def fake_load(model_dir, sidecar_spec, profiler_dir):
        captured["model_dir"] = model_dir
        captured["spec"] = sidecar_spec
        captured["profiler_dir"] = profiler_dir
        output = Path(sidecar_spec.runtime_env["PYTORCH_TUNABLEOP_UNTUNED_FILENAME"])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_name(f"{output.stem}0{output.suffix}").write_text(
            "ScaledGemmTunableOp_Float8_e4m3fn_Float8_e4m3fn_"
            "BFloat16_TN,"
            "tn_64_12288_2048_ld_2048_2048_64_rw_1_bias_None\n"
            "ScaledGemmTunableOp_Float8_e4m3fn_Float8_e4m3fn_"
            "BFloat16_TN,"
            "tn_32_12288_2048_ld_2048_2048_32_rw_1_bias_None\n"
        )
        return server

    mock_load.side_effect = fake_load
    mock_collect_trace.return_value = trace
    bottlenecks = [
        {
            "op_name": "Cijk_selected.kd",
            "kernel_time_us": 900.0,
        }
    ]

    OptimizationService(experience_store=None)._enrich_selected_kernel_shape_evidence(
        bottlenecks,
        spec,
        ckpt,
    )

    assert captured["model_dir"] == str(quant)
    assert "--enforce-eager" in captured["spec"].vllm_extra_args
    assert captured["spec"].runtime_env["PYTORCH_TUNABLEOP_RECORD_UNTUNED"] == "1"
    mock_collect_trace.assert_called_once_with(
        19080,
        profiler_dir=captured["profiler_dir"],
        isl=spec.isl,
        osl=16,
        warmup_steps=2,
        profile_steps=32,
        extract_steady_state=False,
    )
    server.stop.assert_called_once()
    assert bottlenecks[0]["gemm_shape"] == {
        "M": 64,
        "N": 12288,
        "K": 2048,
    }
    assert bottlenecks[0]["gemm_shapes"] == [{"M": 64, "N": 12288, "K": 2048}]
    selected = Path(bottlenecks[0]["tunableop_input"])
    assert selected.is_file()
    assert len(selected.read_text().splitlines()) == 1

    from quark.experimental.torch.quant_perf.perfopt.bottleneck_analysis import (
        BottleneckAnalysisResult,
        BottleneckMode,
    )
    from quark.experimental.torch.quant_perf.perfopt.journey import record_bottleneck_analysis

    record_bottleneck_analysis(
        ckpt,
        BottleneckAnalysisResult(
            requested_mode=BottleneckMode.DIFFERENTIAL,
            effective_mode="differential",
            status="completed",
            reason="",
            candidates=tuple(bottlenecks),
            quantized_trace=str(trace),
            baseline_trace="/trace/baseline.json.gz",
            trace_kind="raw",
        ),
    )
    stored = ckpt.state["bottleneck_analysis"]["candidates"][0]
    assert stored["gemm_shape"] == {
        "M": 64,
        "N": 12288,
        "K": 2048,
    }
    assert stored["gemm_shapes"] == [{"M": 64, "N": 12288, "K": 2048}]
    assert stored["gemm_shape_evidence"]["status"] == "exact"
    assert stored["tunableop_input"] == str(selected)


def test_source_context_tags_require_active_dense_mxfp4_flydsl(tmp_path):
    from quark.experimental.torch.quant_perf.perfopt.service import _source_context_tags

    spec = replace(
        _spec(tmp_path, tmp_path / "aiter"),
        quant_strategy=None,
        mxfp4_gemm_backend="flydsl",
        w4a8_gemm_backend="flydsl",
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["best_candidate"] = {
        "linear_attn_mode": "mxfp4_fp8",
        "self_attn_mode": "mxfp4_fp8",
        "mlp_mode": "mxfp4",
        "kv_cache_mode": "native",
    }

    assert _source_context_tags(spec, ckpt) == ("flydsl_dense_mxfp4",)

    spec = replace(
        spec,
        mxfp4_gemm_backend="triton",
        w4a8_gemm_backend="triton",
    )
    assert _source_context_tags(spec, ckpt) == ()


@patch("quark.experimental.torch.quant_perf.perfopt.service.landing.load")
@patch("quark.experimental.torch.quant_perf.perfopt.service.locate_bottlenecks")
@patch("quark.experimental.torch.quant_perf.perfopt.service.collect_trace")
def test_absolute_bottleneck_mode_never_loads_baseline_server(
    mock_collect_trace,
    mock_locate,
    mock_load,
    tmp_path,
):
    spec = replace(
        _spec(tmp_path, tmp_path / "aiter"),
        bottleneck_mode="absolute",
    )
    ckpt = Checkpoint.fresh(spec)
    trace = tmp_path / "quantized.json.gz"
    trace.touch()
    mock_collect_trace.return_value = trace
    mock_locate.return_value = [
        {
            "op_name": "slow_kernel.kd",
            "kernel_time_us": 100.0,
            "roofline_bound": "UNKNOWN",
        }
    ]
    server = MagicMock(port=19080)

    analysis, _ = OptimizationService(experience_store=None)._collect_and_analyze_bottlenecks(
        server,
        spec,
        ckpt,
    )

    assert analysis.effective_mode == "absolute"
    assert analysis.candidates[0]["op_name"] == "slow_kernel.kd"
    mock_load.assert_not_called()
    server.stop.assert_called_once()


def test_external_source_roots_exclude_managed_runtime_origins(tmp_path):
    from quark.experimental.torch.quant_perf.perfopt.service import _external_source_roots

    spec = _spec(tmp_path, tmp_path / "aiter")
    spec.runtime.kernel_worktree = str(tmp_path / "managed-aiter")
    spec.runtime.runtime_origins = {
        "aiter": spec.kernel_worktree,
        "quark": str(tmp_path / "Quark"),
        "torch": "/usr/local/lib/python3.12/dist-packages/torch",
    }

    assert _external_source_roots(spec) == {
        "quantizer": str(tmp_path / "Quark"),
        "torch": "/usr/local/lib/python3.12/dist-packages/torch",
    }


def test_managed_runtime_uses_session_local_flydsl_cache(tmp_path):
    repo = _repo(tmp_path)
    spec = replace(
        _spec(tmp_path, repo),
        mxfp4_gemm_backend="flydsl",
    )
    quant_config = Path(spec.quant_ckpt_dir) / "config.json"
    quant_config.parent.mkdir(parents=True)
    quant_config.write_text(json.dumps({"quantization_config": {"global_quant_config": {"dtype": "mxfp4"}}}))
    ckpt = Checkpoint.fresh(spec)

    Orchestrator()._prepare_managed_workspaces(spec, ckpt)

    expected = (Path(spec.session_dir) / "runtime" / "cache" / "flydsl").resolve()
    assert spec.runtime_env["FLYDSL_RUNTIME_ENABLE_CACHE"] == "1"
    assert Path(spec.runtime_env["FLYDSL_RUNTIME_CACHE_DIR"]).resolve() == expected


@pytest.mark.parametrize(
    "kernel_name,source_symbol",
    [("reduce_scatter_cross_device_store<bf16,8>", "selected_kernel"), ("selected_kernel", "allreduce_kernel")],
)
def test_unsupported_kernel_skips_preparation_on_resume(tmp_path, monkeypatch, kernel_name, source_symbol):
    repo = _repo(tmp_path)
    spec = _spec(tmp_path, repo)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["kernel_journey"] = [{"kernel_id": make_kernel_id(kernel_name), "name": kernel_name}]
    bottlenecks = [{"op_name": kernel_name, "kernel_time_us": 100.0}]
    resolution = SourceResolution(
        mapping_kind=MappingKind.EDITABLE_SOURCE,
        source_file=str(repo / "kernel.cu"),
        source_repo=str(repo),
        source_symbol=source_symbol,
        confidence=ResolutionConfidence.EXACT,
    )
    monkeypatch.setattr(
        "quark.experimental.torch.quant_perf.perfopt.service.resolve_kernel_source_repo", lambda *args, **kw: resolution
    )

    def unexpected_preparation(*args, **kwargs):
        pytest.fail("unsupported kernel reached worktree, knowledge lookup or GEAK")

    monkeypatch.setattr(
        "quark.experimental.torch.quant_perf.workspace.manager.RepoWorkspaceManager.candidate", unexpected_preparation
    )
    monkeypatch.setattr(
        "quark.experimental.torch.quant_perf.perfopt.knowledge_context.build_kernel_knowledge", unexpected_preparation
    )
    monkeypatch.setattr("quark.experimental.torch.quant_perf.perfopt.service.run_geak", unexpected_preparation)
    service = OptimizationService(experience_store=None)
    for _ in range(2):  # Reenter the candidate loop with its saved skip record.
        assert service._optimize_rewritable_kernels_with_geak(bottlenecks, bottlenecks, [], spec, ckpt) == []
        journey = ckpt.state["kernel_journey"][0]
        assert journey["outcome"] == "skipped"
        assert "multi-GPU" in journey["skip_reason"]
        assert journey["source_mapping"]["patchable"] is True
        assert not ckpt.state.get("geak_patches")


@patch("quark.experimental.torch.quant_perf.perfopt.service.run_geak")
@patch("quark.experimental.torch.quant_perf.perfopt.service.locate_bottlenecks")
@patch("quark.experimental.torch.quant_perf.perfopt.service.collect_trace")
def test_generated_kernel_skip_records_artifact_provenance(
    mock_collect_trace,
    mock_locate,
    mock_run_geak,
    tmp_path,
):
    repo = _repo(tmp_path)
    spec = _spec(tmp_path, repo)
    ckpt = Checkpoint.fresh(spec)
    Orchestrator()._prepare_managed_workspaces(spec, ckpt)
    cache = Path(spec.runtime_env["TRITON_CACHE_DIR"])
    artifact_dir = cache / "HASH"
    origin = tmp_path / "torchinductor" / "graph.py"
    origin.parent.mkdir()
    origin.write_text("def triton_red_fused_example_1():\n    pass\n")
    source = artifact_dir / "triton_red_fused_example_1.source"
    source.parent.mkdir(parents=True)
    source.write_text(f'#loc = loc("{origin}":18:0)\nmodule {{}}\n')
    trace = tmp_path / "trace.json.gz"
    trace.touch()
    mock_collect_trace.return_value = trace
    bottleneck = {
        "op_name": "triton_red_fused_example_1.kd",
        "kernel_names": ["triton_red_fused_example_1.kd"],
        "kernel_time_us": 100.0,
        "roofline_bound": "UNKNOWN",
        "sol_pct": None,
        "gemm_shape": {"M": None, "N": None, "K": None},
    }
    mock_locate.return_value = [bottleneck]
    spec = replace(spec, bottleneck_mode="absolute")
    perfopt = OptimizationService(experience_store=MagicMock())

    result = perfopt.generate_optimization_candidates(
        MagicMock(),
        spec,
        ckpt=ckpt,
        quant_gain=1.0,
    )

    mapping = ckpt.state["kernel_journey"][0]["source_mapping"]
    assert result.patches == []
    mock_run_geak.assert_not_called()
    assert mapping["mapping_kind"] == "generated_artifact"
    assert mapping["patchable"] is False
    assert mapping["generated_source_file"] == str(source)
    assert mapping["generated_origin_file"] == str(origin)


@patch("quark.experimental.torch.quant_perf.perfopt.service.resolve_kernel_source_repo")
@patch("quark.experimental.torch.quant_perf.perfopt.service.run_geak")
@patch("quark.experimental.torch.quant_perf.perfopt.service.locate_bottlenecks")
@patch("quark.experimental.torch.quant_perf.perfopt.service.collect_trace")
def test_dependency_source_is_reported_without_running_geak(
    mock_collect_trace,
    mock_locate,
    mock_run_geak,
    mock_resolve_source,
    tmp_path,
    monkeypatch,
):
    repo = _repo(tmp_path)
    spec = _spec(tmp_path, repo)
    ckpt = Checkpoint.fresh(spec)
    Orchestrator()._prepare_managed_workspaces(spec, ckpt)
    source = Path(spec.active_kernel_repo) / "third_party" / "kernel.hpp"
    source.parent.mkdir()
    source.write_text("struct DependencyKernel {};\n")
    trace = tmp_path / "trace.json.gz"
    trace.touch()
    mock_collect_trace.return_value = trace
    mock_locate.return_value = [
        {
            "op_name": "dependency_kernel.kd",
            "kernel_names": ["dependency_kernel.kd"],
            "kernel_time_us": 100.0,
            "roofline_bound": "UNKNOWN",
            "sol_pct": None,
            "gemm_shape": {"M": 1, "N": 1, "K": 1},
        }
    ]
    mock_resolve_source.return_value = SourceResolution(
        mapping_kind=MappingKind.DEPENDENCY_SOURCE,
        source_file=str(source),
        source_repo=str(source.parent),
        source_repo_role="composable_kernel",
        source_symbol="DependencyKernel",
        method="template_kernel_type",
        confidence=ResolutionConfidence.EXACT,
        reason="composable_kernel dependency is not managed",
    )
    spec = replace(spec, bottleneck_mode="absolute")
    monkeypatch.setattr(
        OptimizationService,
        "_enrich_selected_kernel_shape_evidence",
        lambda *_args, **_kwargs: None,
    )

    result = OptimizationService(
        experience_store=MagicMock(),
    ).generate_optimization_candidates(
        MagicMock(),
        spec,
        ckpt=ckpt,
        quant_gain=1.0,
    )

    journey = ckpt.state["kernel_journey"][0]
    assert result.patches == []
    mock_run_geak.assert_not_called()
    assert journey["outcome"] == "skipped"
    assert journey["skip_reason"] == ("dependency_source: composable_kernel dependency is not managed")
    assert journey["source_mapping"]["patchable"] is False


@patch("quark.experimental.torch.quant_perf.perfopt.service.run_geak")
@patch("quark.experimental.torch.quant_perf.perfopt.service.locate_bottlenecks")
@patch("quark.experimental.torch.quant_perf.perfopt.service.collect_trace")
def test_perfopt_collects_flydsl_cache_provenance_before_resolution(
    mock_collect_trace,
    mock_locate,
    mock_run_geak,
    tmp_path,
):
    repo = _repo(tmp_path)
    source = repo / "aiter" / "ops" / "flydsl" / "kernels" / "anonymous.py"
    source.parent.mkdir(parents=True)
    source.write_text("@flyc.kernel\ndef anonymous_kernel(x):\n    return x\n")
    (repo / "aiter" / "__init__.py").write_text("")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "add anonymous kernel")
    spec = replace(
        _spec(tmp_path, repo),
        mxfp4_gemm_backend="flydsl",
    )
    quant_config = Path(spec.quant_ckpt_dir) / "config.json"
    quant_config.parent.mkdir(parents=True)
    quant_config.write_text(json.dumps({"quantization_config": {"global_quant_config": {"dtype": "mxfp4"}}}))
    ckpt = Checkpoint.fresh(spec)
    Orchestrator()._prepare_managed_workspaces(spec, ckpt)
    managed_source = Path(spec.active_kernel_repo) / source.relative_to(repo)
    cache = Path(spec.runtime_env["FLYDSL_RUNTIME_CACHE_DIR"])
    artifact = cache / "launch_anonymous_0123456789abcdef0123456789abcdef" / "abc123def4567890.pkl"
    artifact.parent.mkdir(parents=True)
    source_ir = (
        f'#loc1 = loc("{managed_source}":2:0)\n'
        "module {\n"
        "  gpu.module @kernels {\n"
        "    gpu.func @anonymous_kernel_0() kernel loc(#loc1)\n"
        "  }\n"
        "}\n"
    )
    artifact.write_bytes(pickle.dumps({"source_ir": source_ir}))
    trace = tmp_path / "trace.json.gz"
    trace.touch()
    mock_collect_trace.return_value = trace
    bottleneck = {
        "op_name": "anonymous_kernel_0.kd",
        "kernel_names": ["anonymous_kernel_0.kd"],
        "parent_op_names": [],
        "kernel_time_us": 100.0,
        "roofline_bound": "UNKNOWN",
        "sol_pct": None,
        "gemm_shape": {"M": 1, "N": 1, "K": 1},
    }
    mock_locate.return_value = [bottleneck]
    mock_run_geak.return_value = {
        "verified_speedup": 1.0,
        "best_patch": "",
        "round_evaluation": {},
    }
    spec = replace(spec, bottleneck_mode="absolute")
    perfopt = OptimizationService(experience_store=MagicMock())

    perfopt.generate_optimization_candidates(
        MagicMock(),
        spec,
        ckpt=ckpt,
        quant_gain=1.0,
    )

    mapping = ckpt.state["kernel_journey"][0]["source_mapping"]
    assert mapping["method"] == "compiler_manifest"
    assert mapping["source_file"] == str(managed_source)
    assert (Path(spec.session_dir) / "kernel_provenance.json").is_file()
    assert (Path(spec.session_dir) / "kernel_source_resolution.json").is_file()


@patch("quark.experimental.torch.quant_perf.perfopt.service.locate_bottlenecks")
@patch("quark.experimental.torch.quant_perf.perfopt.service.collect_trace")
@patch("quark.experimental.torch.quant_perf.perfopt.service.run_geak")
@pytest.mark.parametrize("ending", ["completed", "incomplete", "rejected", "save_error", "changed_patch"])
def test_geak_runs_in_candidate_and_persists_patch_before_cleanup(
    mock_run_geak,
    mock_collect_trace,
    mock_locate,
    tmp_path,
    monkeypatch,
    ending,
):
    repo = _repo(tmp_path)
    spec = _spec(tmp_path, repo)
    quant_config = Path(spec.quant_ckpt_dir) / "config.json"
    quant_config.parent.mkdir(parents=True)
    quant_config.write_text(json.dumps({"quantization_config": {"global_quant_config": {"dtype": "mxfp4"}}}))
    ckpt = Checkpoint.fresh(spec)
    Orchestrator()._prepare_managed_workspaces(spec, ckpt)
    integration_source = Path(spec.active_kernel_repo) / "kernel.cu"
    original_text = integration_source.read_text()

    trace = tmp_path / "trace.json.gz"
    trace.touch()
    mock_collect_trace.return_value = trace
    bottleneck = {
        "op_name": "selected_kernel",
        "kernel_names": ["selected_kernel"],
        "kernel_time_us": 100.0,
        "roofline_bound": "UNKNOWN",
        "sol_pct": None,
        "gemm_shape": {"M": 1, "N": 1, "K": 1},
        "shape_cases": [{"M": 1, "N": 1, "K": 1}],
    }
    mock_locate.return_value = [bottleneck]
    eval_dirs = []

    def _run_geak(candidate_src, *_args, **_kwargs):
        assert Path(candidate_src).is_file()
        assert str(candidate_src).startswith(str(Path(spec.session_dir) / "workspaces" / "managed"))
        assert not str(candidate_src).startswith(spec.active_kernel_repo)
        Path(candidate_src).write_text("__global__ void selected_kernel(float* x) { x[0] += 1; }\n")
        Path("/tmp/quark_quant_perf_kw_eval").mkdir(parents=True, exist_ok=True)
        eval_dir = Path(tempfile.mkdtemp(prefix="unit-", dir="/tmp/quark_quant_perf_kw_eval"))
        eval_dirs.append(eval_dir)
        patch = eval_dir / "final_patch.diff"
        patch.write_text(
            "diff --git a/kernel.cu b/kernel.cu\n"
            "--- a/kernel.cu\n"
            "+++ b/kernel.cu\n"
            "@@ -1 +1 @@\n"
            "-__global__ void selected_kernel(float* x) { x[0] = x[0]; }\n"
            "+__global__ void selected_kernel(float* x) { x[0] += 1; }\n"
        )
        (eval_dir / "correctness.json").write_text(json.dumps({"success": True}))
        (eval_dir / "benchmark.json").write_text(json.dumps({"speedup": 1.1}))
        report = {
            "verified_speedup": 1.1,
            "best_patch": str(patch),
            "eval_dir": str(eval_dir),
            "round_evaluation": {"correctness": {"success": None if ending == "incomplete" else True}},
            "execution_status": "timed_out" if ending == "incomplete" else "completed",
            "timed_out": ending == "incomplete",
        }
        report["patch_sha256"] = hashlib.sha256(patch.read_bytes()).hexdigest()
        if ending == "rejected":
            report.update(
                best_patch="",
                verified_speedup=0.0,
                micro_speedup_source="invalid_workload",
                execution_status="incomplete",
            )
        if ending == "changed_patch":
            patch.write_text(patch.read_text().replace("+= 1", "+= 2"))
        return report

    mock_run_geak.side_effect = _run_geak
    spec = replace(spec, bottleneck_mode="absolute")
    perfopt = OptimizationService(experience_store=MagicMock())
    model_server = MagicMock()

    if ending == "save_error":

        def fail_save(*args, **kwargs):
            raise OSError("disk full")

        monkeypatch.setattr("quark.experimental.torch.quant_perf.perfopt.kernel_workflow.write_json_atomic", fail_save)
    if ending in {"save_error", "changed_patch"}:
        with pytest.raises((OSError, ValueError), match="disk full|patch changed"):
            perfopt.generate_optimization_candidates(model_server, spec, ckpt=ckpt)
        assert eval_dirs[0].is_dir()
        assert integration_source.read_text() == original_text
        return

    result = perfopt.generate_optimization_candidates(
        model_server,
        spec,
        ckpt=ckpt,
        quant_gain=1.0,
    )

    assert integration_source.read_text() == original_text
    assert eval_dirs[0].exists() is (ending in {"incomplete", "rejected"})
    entry = ckpt.state["geak_patches"][0]
    assert entry["status"] == (ending if ending in {"incomplete", "rejected"} else "candidate")
    assert entry["timed_out"] is (ending == "incomplete")
    saved = json.loads((Path(entry["artifacts_dir"]) / "result.json").read_text())
    assert saved["execution_status"] == {"incomplete": "timed_out", "rejected": "incomplete"}.get(ending, "completed")
    if ending in {"incomplete", "rejected"}:
        assert not result.patches
        return
    assert len(result.patches) == 1
    assert Path(result.patches[0]).is_file()
    assert "/artifacts/final_patch.diff" in result.patches[0]
    assert ckpt.state["transient_resources"] == []
    assert not list(
        (Path("/tmp/quark_quant_perf_worktrees") / ckpt.state["session_id"] / "kernel" / "candidates").glob(
            "*/worktree"
        )
    )


@patch("quark.experimental.torch.quant_perf.perfopt.service.locate_bottlenecks")
@patch("quark.experimental.torch.quant_perf.perfopt.service.collect_trace")
@patch("quark.experimental.torch.quant_perf.perfopt.service.run_geak")
def test_unvalidated_micro_gain_does_not_stop_remaining_geak_candidates(
    mock_run_geak,
    mock_collect_trace,
    mock_locate,
    tmp_path,
):
    repo = _repo(tmp_path)
    source = repo / "kernel.cu"
    source.write_text(
        "__global__ void selected_kernel_one(float* x) { x[0] = x[0]; }\n"
        "__global__ void selected_kernel_two(float* x) { x[0] = x[0]; }\n"
    )
    _git(repo, "add", "kernel.cu")
    _git(repo, "commit", "-m", "add second kernel")

    spec = _spec(tmp_path, repo)
    spec = replace(
        spec,
        top_kernels=2,
        target_gain=1.05,
    )
    quant_config = Path(spec.quant_ckpt_dir) / "config.json"
    quant_config.parent.mkdir(parents=True)
    quant_config.write_text(json.dumps({"quantization_config": {"global_quant_config": {"dtype": "mxfp4"}}}))
    ckpt = Checkpoint.fresh(spec)
    Orchestrator()._prepare_managed_workspaces(spec, ckpt)
    trace = tmp_path / "trace.json.gz"
    trace.touch()
    mock_collect_trace.return_value = trace
    bottlenecks = [
        {
            "op_name": name,
            "kernel_names": [name],
            "kernel_time_us": 100.0,
            "roofline_bound": "UNKNOWN",
            "sol_pct": None,
            "gemm_shape": {"M": 1, "N": 1, "K": 1},
            "shape_cases": [{"M": 1, "N": 1, "K": 1}],
        }
        for name in ("selected_kernel_one", "selected_kernel_two")
    ]
    mock_locate.return_value = bottlenecks

    def _run_geak(candidate_src, *_args, **_kwargs):
        Path("/tmp/quark_quant_perf_kw_eval").mkdir(parents=True, exist_ok=True)
        eval_dir = Path(tempfile.mkdtemp(prefix="unit-", dir="/tmp/quark_quant_perf_kw_eval"))
        patch = eval_dir / "final_patch.diff"
        patch.write_text(
            "diff --git a/kernel.cu b/kernel.cu\n"
            "--- a/kernel.cu\n"
            "+++ b/kernel.cu\n"
            "@@ -1,2 +1,3 @@\n"
            " __global__ void selected_kernel_one(float* x) { x[0] = x[0]; }\n"
            f"+# optimized {Path(candidate_src).name}\n"
            " __global__ void selected_kernel_two(float* x) { x[0] = x[0]; }\n"
        )
        (eval_dir / "correctness.json").write_text(json.dumps({"success": True}))
        (eval_dir / "benchmark.json").write_text(json.dumps({"speedup": 1.1}))
        return {
            "verified_speedup": 1.1,
            "best_patch": str(patch),
            "eval_dir": str(eval_dir),
            "round_evaluation": {"correctness": {"success": True}},
        }

    mock_run_geak.side_effect = _run_geak
    spec = replace(spec, bottleneck_mode="absolute")
    perfopt = OptimizationService(experience_store=MagicMock())

    result = perfopt.generate_optimization_candidates(
        MagicMock(),
        spec,
        ckpt=ckpt,
        quant_gain=1.0,
    )

    assert mock_run_geak.call_count == 2
    assert len(result.patches) == 2


@patch("quark.experimental.torch.quant_perf.perfopt.service.locate_bottlenecks")
@patch("quark.experimental.torch.quant_perf.perfopt.service.collect_trace")
@patch("quark.experimental.torch.quant_perf.perfopt.vendor_gemm.run_vendor_gemm_tuning")
def test_vendor_gemm_candidate_flows_into_perf_result(
    mock_vendor_tuning,
    mock_collect_trace,
    mock_locate,
    tmp_path,
):
    repo = _repo(tmp_path)
    spec = _spec(tmp_path, repo)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["vendor_gemm_tuning"] = {
        "status": "failed",
        "reason": "legacy model analysis failure",
        "capability_version": 2,
        "attempt_count": 1,
        "attempts": [
            {
                "attempt": 1,
                "status": "failed",
                "capability_version": 2,
            }
        ],
    }
    Orchestrator()._prepare_managed_workspaces(spec, ckpt)
    trace = tmp_path / "trace.json.gz"
    trace.touch()
    mock_collect_trace.return_value = trace
    bottleneck = {
        "op_name": "Cijk_vendor_gemm",
        "kernel_names": ["Cijk_vendor_gemm"],
        "kernel_time_us": 500.0,
        "roofline_bound": "COMPUTE_BOUND",
        "sol_pct": 70.0,
        "gemm_shape": {"M": 16, "N": 4096, "K": 4096},
    }
    mock_locate.return_value = [bottleneck]
    candidate = {
        "name": "forge_vendor_gemm",
        "runtime_env": {"AITER_CONFIG_GEMM_A4W4": "/tmp/tuned.csv"},
        "artifacts": {"tuned_csv": "/tmp/tuned.csv"},
        "micro_speedup": 1.1,
    }
    mock_vendor_tuning.return_value = {
        "status": "candidate",
        "candidates": [candidate],
        "capability_version": 4,
        "attempt_count": 2,
    }

    spec = replace(spec, bottleneck_mode="absolute")
    perfopt = OptimizationService(experience_store=MagicMock())
    result = perfopt.generate_optimization_candidates(
        MagicMock(),
        spec,
        ckpt=ckpt,
        quant_gain=1.0,
    )

    assert result.runtime_candidates == [candidate]
    assert ckpt.state["vendor_gemm_tuning"]["status"] == "candidate"
    assert mock_vendor_tuning.call_args.kwargs["attempt_count"] == 2
    assert len(ckpt.state["vendor_gemm_tuning"]["attempts"]) == 2
