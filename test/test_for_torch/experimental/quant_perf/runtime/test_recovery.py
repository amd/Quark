#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for baseline/performance failure classification and fingerprints."""

import json
from dataclasses import replace

import pytest

from quark.experimental.torch.quant_perf.evaluation.gsm8k import EvaluationFailure
from quark.experimental.torch.quant_perf.evaluation.throughput import BenchmarkFailure
from quark.experimental.torch.quant_perf.runtime.recovery import (
    build_accuracy_fingerprint,
    build_runtime_fingerprint,
    classify_failure,
)
from quark.experimental.torch.quant_perf.session.spec import EvalProfile, Spec


def _profile() -> EvalProfile:
    return EvalProfile(
        profile_id="gsm8k-chat-nothink-v1",
        profile_hash="",
        model_mode="chat",
        apply_chat_template=True,
        enable_thinking=False,
        detection_reason="test",
    ).with_computed_hash()


def _spec(tmp_path, **overrides):
    defaults = dict(
        model_dir=str(tmp_path),
        base_model=str(tmp_path),
        framework="vllm",
        gpu_type="mi355x",
        gpu_arch="MI355X",
        isl=1024,
        osl=1024,
        quant_strategy="mxfp4",
        vllm_extra_args=[
            "--tensor-parallel-size=4",
            "--gpu-memory-utilization=0.8",
        ],
        eval_profile=_profile(),
        session_dir=str(tmp_path / "run"),
    )
    defaults.update(overrides)
    return Spec(**defaults)


def test_inference_fingerprints_ignore_search_backend_but_track_inference(tmp_path):
    spec = _spec(tmp_path)

    def fingerprints(value):
        return (
            build_accuracy_fingerprint(value, str(tmp_path), framework_commit="v", kernel_commit="a", runtime_env={}),
            build_runtime_fingerprint(value, framework_commit="v", runtime_env={}, stage="baseline"),
        )

    assert fingerprints(spec) == fingerprints(replace(spec, search_moe_backend="aiter"))
    assert fingerprints(spec) != fingerprints(replace(spec, inference_moe_backend="triton"))
    assert fingerprints(spec) != fingerprints(replace(spec, kv_cache_scheme="fp8"))


@pytest.mark.parametrize(
    ("error", "repos", "expected", "has_evidence"),
    [
        pytest.param(
            "torch.OutOfMemoryError: HIP out of memory\n"
            "unquantized_fused_moe_method.py in _maybe_pad_weight\n"
            "weight = F.pad(weight, (0, num_pad), 'constant', 0)",
            {},
            {
                "failure_class": "resource_capacity",
                "cache_policy": "soft",
                "code": "rocm_moe_padding_oom",
                "recovery": {"action": "set_env", "key": "VLLM_ROCM_MOE_PADDING", "value": "0"},
            },
            False,
            id="padding-oom",
        ),
        pytest.param(
            "ValueError: No available memory for the cache blocks. Try increasing `gpu_memory_utilization`.",
            {},
            {
                "failure_class": "resource_capacity",
                "cache_policy": "soft",
                "code": "kv_cache_no_memory",
                "recovery": {"action": "increase_gpu_memory_utilization", "step": 0.05, "maximum": 0.95},
            },
            False,
            id="kv-cache-shortage",
        ),
        pytest.param(
            "RuntimeError: The size of tensor a (4096) must match the size of tensor b (32)",
            {},
            {"failure_class": "framework", "cache_policy": "hard", "repair_eligible": True},
            False,
            id="shape-mismatch",
        ),
        pytest.param(
            "RuntimeError: W4A8 FlyDSL backend is unavailable in the installed AITER build.",
            {},
            {
                "failure_class": "kernel",
                "cache_policy": "hard",
                "code": "kernel_backend_unavailable",
                "repair_eligible": True,
                "target_role": "kernel",
            },
            True,
            id="missing-flydsl",
        ),
        pytest.param(
            "vllm/model_executor/layers/quantization/quark/quark_moe.py\n"
            "AttributeError: 'NoneType' object has no attribute 'to'",
            {},
            {
                "failure_class": "framework",
                "cache_policy": "hard",
                "code": "quark_biasless_moe",
                "repair_eligible": True,
            },
            False,
            id="biasless-moe",
        ),
        pytest.param(
            'File "/work/vllm/layer.py", in _load_per_tensor_weight_scale\n'
            "param_data[expert_id] = loaded_weight\n"
            "RuntimeError: expand(torch.ByteTensor{[4096, 32]}, size=[])",
            {},
            {
                "failure_class": "framework",
                "code": "quark_mxfp4_block_scale_as_scalar",
                "repair_eligible": True,
                "target_role": "framework",
                "reason": "known_signature",
            },
            True,
            id="mxfp4-block-scale",
        ),
        pytest.param(
            'File "/work/vllm/vllm/model_executor/custom_loader.py", line 44, in load\n'
            "RuntimeError: new loader failure",
            {"framework_repo": "/work/vllm", "kernel_repo": "/work/aiter"},
            {
                "failure_class": "framework",
                "code": "managed_source_failure",
                "repair_eligible": True,
                "target_role": "framework",
                "reason": "managed_source",
            },
            True,
            id="managed-source",
        ),
        pytest.param(
            'File "/work/vllm/vllm/model_executor/custom_loader.py", line 44, in <module>\n'
            "    from vllm.model_executor.layers.new_api import Loader\n"
            "ModuleNotFoundError: No module named 'vllm.model_executor.layers.new_api'",
            {"framework_repo": "/work/vllm", "kernel_repo": "/work/aiter"},
            {
                "failure_class": "framework",
                "code": "managed_source_failure",
                "repair_eligible": True,
                "target_role": "framework",
                "reason": "managed_source",
            },
            True,
            id="managed-source-missing-internal-module",
        ),
        pytest.param(
            'File "/work/vllm/vllm/model_executor/custom_loader.py", line 44, in <module>\n'
            "    import compressed_tensors\n"
            "ModuleNotFoundError: No module named 'compressed_tensors'",
            {"framework_repo": "/work/vllm", "kernel_repo": "/work/aiter"},
            {
                "failure_class": "dependency",
                "cache_policy": "hard",
                "code": "dependency_error",
                "repair_eligible": False,
            },
            False,
            id="managed-source-missing-external-dependency",
        ),
        pytest.param(
            'File "/usr/local/lib/python3.12/site-packages/other/runtime.py", line 7, in run\n'
            "RuntimeError: external failure",
            {"framework_repo": "/work/vllm", "kernel_repo": "/work/aiter"},
            {"failure_class": "unknown", "repair_eligible": False, "target_role": ""},
            True,
            id="external-source",
        ),
        pytest.param(
            "torch.OutOfMemoryError: HIP out of memory while loading weights",
            {},
            {
                "failure_class": "resource_capacity",
                "cache_policy": "soft",
                "code": "capacity_oom",
                "recovery": None,
                "repair_eligible": False,
            },
            False,
            id="generic-oom",
        ),
        pytest.param(
            BenchmarkFailure("/model", timed_out=True, bench_error="timeout"),
            {},
            {
                "failure_class": "timeout",
                "cache_policy": "soft",
                "code": "timeout",
                "recovery": {"action": "retry_same"},
            },
            False,
            id="typed-timeout",
        ),
        pytest.param(
            EvaluationFailure(
                "/model",
                returncode=1,
                stderr="HIP error: hipErrorStreamCaptureInvalidated during capture",
            ),
            {},
            {
                "failure_class": "kernel",
                "cache_policy": "soft",
                "code": "cuda_graph_capture",
                "recovery": {"action": "retry_same"},
                "repair_eligible": False,
            },
            False,
            id="cuda-graph-capture",
        ),
    ],
)
def test_failure_classification(error, repos, expected, has_evidence):
    diagnosis = classify_failure(error, **repos)

    for field, value in expected.items():
        assert getattr(diagnosis, field) == value
    assert bool(diagnosis.evidence_signature) is has_evidence


def test_runtime_fingerprint_is_stable_and_changes_with_tp_env_and_profile(
    tmp_path,
):
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5_moe"}')
    (tmp_path / "model.safetensors.index.json").write_text('{"weight_map":{"a":"model-1.safetensors"}}')
    spec = _spec(tmp_path)
    env = {"VLLM_ROCM_MOE_PADDING": "1"}

    first = build_runtime_fingerprint(
        spec,
        framework_commit="abc",
        runtime_env=env,
        stage="baseline_health",
    )
    second = build_runtime_fingerprint(
        spec,
        framework_commit="abc",
        runtime_env=env,
        stage="baseline_health",
    )
    tp_changed = build_runtime_fingerprint(
        _spec(
            tmp_path,
            vllm_extra_args=[
                "--tensor-parallel-size=8",
                "--gpu-memory-utilization=0.8",
            ],
        ),
        framework_commit="abc",
        runtime_env=env,
        stage="baseline_health",
    )
    env_changed = build_runtime_fingerprint(
        spec,
        framework_commit="abc",
        runtime_env={"VLLM_ROCM_MOE_PADDING": "0"},
        stage="baseline_health",
    )
    vendor_gemm_changed = build_runtime_fingerprint(
        spec,
        framework_commit="abc",
        runtime_env={
            "VLLM_ROCM_MOE_PADDING": "1",
            "AITER_CONFIG_GEMM_A4W4": "/tmp/tuned.csv",
        },
        stage="baseline_health",
    )
    w4a8_backend_changed = build_runtime_fingerprint(
        spec,
        framework_commit="abc",
        runtime_env={
            "VLLM_ROCM_MOE_PADDING": "1",
            "VLLM_ROCM_W4A8_GEMM_BACKEND": "flydsl",
        },
        stage="baseline_health",
    )
    mxfp4_backend_changed = build_runtime_fingerprint(
        spec,
        framework_commit="abc",
        runtime_env={
            "VLLM_ROCM_MOE_PADDING": "1",
            "VLLM_ROCM_MXFP4_GEMM_BACKEND": "flydsl",
        },
        stage="baseline_health",
    )
    mxfp4_asm_changed = build_runtime_fingerprint(
        spec,
        framework_commit="abc",
        runtime_env={
            "VLLM_ROCM_MOE_PADDING": "1",
            "VLLM_ROCM_MXFP4_GEMM_BACKEND": "triton",
            "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": "1",
        },
        stage="baseline_health",
    )
    profile_changed = build_runtime_fingerprint(
        _spec(
            tmp_path,
            eval_profile=replace(
                spec.eval_profile,
                max_gen_toks=2048,
            ).with_computed_hash(),
        ),
        framework_commit="abc",
        runtime_env=env,
        stage="baseline_health",
    )
    sample_count_changed = build_runtime_fingerprint(
        _spec(
            tmp_path,
            gsm8k_num_samples=50,
        ),
        framework_commit="abc",
        runtime_env=env,
        stage="baseline_health",
    )

    assert first == second
    assert first != tp_changed
    assert first != env_changed
    assert first != vendor_gemm_changed
    assert first != w4a8_backend_changed
    assert first != mxfp4_backend_changed
    assert first != mxfp4_asm_changed
    assert first != profile_changed
    assert first != sample_count_changed


def test_accuracy_fingerprint_binds_checkpoint_tp_profile_and_repos(
    tmp_path,
):
    base = tmp_path / "base"
    quant = tmp_path / "quant"
    base.mkdir()
    quant.mkdir()
    (base / "config.json").write_text('{"model_type":"qwen3_5_moe"}')
    (quant / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "num_experts": 256,
                "quantization_config": {
                    "global_quant_config": {
                        "weight": {"dtype": "fp4"},
                        "input_tensors": {"dtype": "fp4"},
                    }
                },
            }
        )
    )
    (quant / "model.safetensors.index.json").write_text('{"weight_map":{"a":"model-1.safetensors"}}')
    spec = _spec(tmp_path, model_dir=str(base), base_model=str(base))

    first = build_accuracy_fingerprint(
        spec,
        str(quant),
        framework_commit="fw-a",
        kernel_commit="kernel-a",
        runtime_env={"VLLM_ROCM_MOE_PADDING": "0"},
    )
    same = build_accuracy_fingerprint(
        spec,
        str(quant),
        framework_commit="fw-a",
        kernel_commit="kernel-a",
        runtime_env={"VLLM_ROCM_MOE_PADDING": "0"},
    )
    tp_changed = build_accuracy_fingerprint(
        _spec(
            tmp_path,
            model_dir=str(base),
            base_model=str(base),
            vllm_extra_args=["--tensor-parallel-size=8"],
        ),
        str(quant),
        framework_commit="fw-a",
        kernel_commit="kernel-a",
        runtime_env={"VLLM_ROCM_MOE_PADDING": "0"},
    )
    repo_changed = build_accuracy_fingerprint(
        spec,
        str(quant),
        framework_commit="fw-b",
        kernel_commit="kernel-a",
        runtime_env={"VLLM_ROCM_MOE_PADDING": "0"},
    )
    explicit_single_k = build_accuracy_fingerprint(
        spec,
        str(quant),
        framework_commit="fw-a",
        kernel_commit="kernel-a",
        runtime_env={
            "VLLM_ROCM_MOE_PADDING": "0",
            "AITER_KSPLIT": "1",
        },
    )
    non_aiter_backend = build_accuracy_fingerprint(
        _spec(
            tmp_path,
            model_dir=str(base),
            base_model=str(base),
            mxfp4_moe_backend="triton",
        ),
        str(quant),
        framework_commit="fw-a",
        kernel_commit="kernel-a",
        runtime_env={"VLLM_ROCM_MOE_PADDING": "0"},
    )

    assert first == same
    assert first == explicit_single_k
    assert first != tp_changed
    assert first != repo_changed
    assert first != non_aiter_backend
