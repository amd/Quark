#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""GPU-free tests for Quark public mixed-precision search integration."""

from __future__ import annotations

import fnmatch
import json
import shutil
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quark.experimental.torch.quant_perf.quantize.search import (
    _candidate_queue_from_result,
    _effective_exclude_patterns,
    _hardware_target,
    _resolve_layer_precision_candidates,
    _search_runtime_args,
)
from quark.experimental.torch.quant_perf.session.spec import Checkpoint, RuntimeContext, Spec, StageError

from .quant_artifact_fixtures import (
    write_valid_quant_checkpoint,
)


def _runtime_spec(tmp_path, **overrides):
    return Spec(
        model_dir=str(tmp_path),
        base_model=str(tmp_path),
        framework="vllm",
        gpu_type="mi355x",
        gpu_arch="MI355X",
        isl=128,
        osl=128,
        quant_strategy=None,
        session_dir=str(tmp_path),
        **overrides,
    )


@pytest.mark.parametrize("backend", [None, "triton", "aiter", "aiter_mxfp4_bf16"])
def test_search_worker_isolates_inference_moe_environment(tmp_path, monkeypatch, backend):
    from quark.experimental.torch.quant_perf.quantize import isolated_search

    spec = _runtime_spec(
        tmp_path,
        search_moe_backend=backend,
        runtime=RuntimeContext(runtime_env={"VLLM_ROCM_USE_AITER_MOE": "1"}),
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["retained_runtime_env"] = {"VLLM_ROCM_USE_AITER_MOE": "1"}
    ckpt.save()
    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_FLYDSL_MOE", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE", "1")
    monkeypatch.setenv("AITER_FLYDSL_FORCE", "1")
    monkeypatch.setattr(isolated_search, "activate_runtime", lambda spec: None)

    def check_search(spec, checkpoint):
        assert _search_runtime_args(spec) == ([f"--moe-backend={backend}"] if backend else []) + [
            "--gpu-memory-utilization=0.75"
        ]
        assert isolated_search.os.environ["VLLM_ROCM_USE_AITER"] == "1"
        assert isolated_search.os.environ["VLLM_ROCM_USE_AITER_MOE"] == (
            "1" if backend and backend.startswith("aiter") else "0"
        )
        for key in ("VLLM_ROCM_USE_AITER_FLYDSL_MOE", "VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE", "AITER_FLYDSL_FORCE"):
            assert key not in isolated_search.os.environ
        return "/quant"

    monkeypatch.setattr(isolated_search, "run_module_search", check_search)
    assert isolated_search._worker_payload(tmp_path)["quant_ckpt_dir"] == "/quant"


def test_maps_gpu_type_to_quark_hardware_name():
    assert _hardware_target("mi300x") == "mi300"
    assert _hardware_target("mi325x") == "mi325"
    assert _hardware_target("mi350x") == "mi355"
    assert _hardware_target("mi355x") == "mi355"


@pytest.mark.parametrize("route", ["standard", "explicit", "packed_source", "memory"])
@pytest.mark.parametrize("has_compatible", [False, True])
def test_completed_search_can_change_export_route_without_repeating_search(
    tmp_path, monkeypatch, route, has_compatible
):
    from quark.experimental.torch.mix_precision import MixPrecisionQuantizer
    from quark.experimental.torch.mix_precision import quantizer as implementation
    from quark.experimental.torch.quant_perf.quantize import search

    candidate = {"self_attn_mode": "native", "routed_moe_mode": "mxfp4", "kv_cache_mode": "native"}
    static = {**candidate, "routed_moe_mode": "mxfp4_fp8"}
    candidates = [static, candidate] if has_compatible else [static, {"mlp_mode": "native"}]
    calls = []
    if route == "memory":
        monkeypatch.setattr(implementation, "_file_to_file_memory_reason", lambda _: "estimated memory exceeds budget")
    if route == "packed_source":
        (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
        (tmp_path / "model.safetensors").touch()
        monkeypatch.setattr(implementation, "has_packed_mxfp4_source", lambda _: True)

    def export(self, model_path, config, destination):
        calls.append((config, self.config.file2file_quantization))
        write_valid_quant_checkpoint(destination)

    monkeypatch.setattr(MixPrecisionQuantizer, "search", lambda *_args, **_kwargs: pytest.fail("Search must be reused"))
    monkeypatch.setattr(MixPrecisionQuantizer, "_export_best_standard", export)
    monkeypatch.setattr(MixPrecisionQuantizer, "_export_best_file_to_file", export)

    monkeypatch.setattr(search, "_empty_all_cuda_caches", lambda: None)
    spec = SimpleNamespace(base_model=str(tmp_path), session_dir=str(tmp_path), file2file_export=route == "explicit")
    monkeypatch.setattr(search, "_search_runtime_args", lambda _spec: [])
    state = {
        "mix_precision_search": {
            "status": "completed",
            "config": {},
            "result": {"total_configs_evaluated": 5},
            "candidate_queue": [
                {"config": config, "status": "pending", "search_metrics": {"gsm8k": 0.8}} for config in candidates
            ],
            "candidate_cursor": 0,
        }
    }
    ckpt = SimpleNamespace(state=state, save=lambda: None)
    if route != "standard" and not has_compatible:
        with pytest.raises(StageError, match="quantized candidate"):
            search.run_module_search(spec, ckpt)
        assert calls == []
    else:
        assert search.run_module_search(spec, ckpt) == str(tmp_path / "quant_ckpt")
        assert calls == [(static if route == "standard" else candidate, route != "standard")]
        assert state["mix_precision_search"]["candidate_cursor"] == (0 if route == "standard" else 1)
        if route != "standard":
            assert state["mix_precision_search"]["candidate_queue"][0]["status"] == "skipped"
    assert [row["config"] for row in state["mix_precision_search"]["candidate_queue"]] == candidates
    assert all(row["search_metrics"] == {"gsm8k": 0.8} for row in state["mix_precision_search"]["candidate_queue"])
    assert state["mix_precision_search"]["result"]["total_configs_evaluated"] == 5


def test_unknown_gpu_type_raises_value_error():
    with pytest.raises(ValueError, match="Unknown gpu_type"):
        _hardware_target("mi9000x")


def test_default_layer_precision_candidates_are_arch_aware_and_exclude_mxfp6():
    assert _resolve_layer_precision_candidates(None, "mi300") == [
        "native",
        "fp8",
        "ptpc_fp8",
    ]
    assert _resolve_layer_precision_candidates(None, "mi355") == [
        "native",
        "fp8",
        "ptpc_fp8",
        "mxfp4",
        "mxfp4_fp8",
    ]


def test_explicit_layer_precision_candidates_win_verbatim():
    assert _resolve_layer_precision_candidates(
        ["native", "mxfp6_e2m3"],
        "mi355",
    ) == ["native", "mxfp6_e2m3"]


def test_explicit_layer_precision_candidates_add_native_when_omitted():
    assert _resolve_layer_precision_candidates(
        ["mxfp4", "mxfp4_fp8", "fp8"],
        "mi355",
    ) == ["native", "mxfp4", "mxfp4_fp8", "fp8"]


def test_search_runtime_args_inherit_eval_profile_max_model_len():
    spec = SimpleNamespace(
        effective_search_moe_backend="triton",
        search_vllm_args=["--tensor-parallel-size", "4", "--moe-backend=triton"],
        eval_profile=SimpleNamespace(max_model_len=8192),
    )

    assert _search_runtime_args(spec) == [
        "--tensor-parallel-size",
        "4",
        "--moe-backend=triton",
        "--gpu-memory-utilization=0.75",
        "--max-model-len",
        "8192",
    ]


def test_search_runtime_args_preserve_explicit_max_model_len():
    spec = SimpleNamespace(
        effective_search_moe_backend="triton",
        search_vllm_args=[
            "--tensor-parallel-size",
            "4",
            "--max-model-len=4096",
            "--moe-backend=triton",
        ],
        eval_profile=SimpleNamespace(max_model_len=8192),
    )

    assert _search_runtime_args(spec) == spec.search_vllm_args + ["--gpu-memory-utilization=0.75"]


def test_gemma4_search_keeps_multimodal_towers_and_connectors_native():
    spec = SimpleNamespace(model_arch="gemma4", exclude_layers=None)

    excludes = _effective_exclude_patterns(spec)

    assert excludes is not None
    assert "lm_head" in excludes
    for module_name in (
        "model.vision_tower.encoder.layers.0.mlp.down_proj.linear",
        "vision_tower.encoder.layers.0.self_attn.q_proj",
        "model.embed_vision.embedding_projection",
        "embed_vision.embedding_projection",
        "model.audio_tower.encoder.layers.0",
        "audio_tower.encoder.layers.0",
        "model.embed_audio.embedding_projection",
        "embed_audio.embedding_projection",
        "model.language_model.layers.0.router.proj",
    ):
        assert any(fnmatch.fnmatch(module_name, pattern) for pattern in excludes), module_name


def test_default_quark_excludes_are_not_overridden_without_additions():
    spec = SimpleNamespace(model_arch="qwen3", exclude_layers=None)

    assert _effective_exclude_patterns(spec) is None


def test_effective_excludes_preserve_user_patterns_without_duplicates():
    spec = SimpleNamespace(
        model_arch="gemma4",
        exclude_layers=["lm_head", "vision_tower"],
    )

    excludes = _effective_exclude_patterns(spec)

    assert excludes[:2] == ["lm_head", "vision_tower"]
    assert excludes.count("vision_tower") == 1
    assert any("vision_tower" in pattern for pattern in excludes)
    assert any("embed_vision" in pattern for pattern in excludes)


def test_latest_quark_public_api_is_the_only_required_search_surface(
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.quantize import search

    modules = {}
    for name in (
        "quark",
        "quark.experimental",
        "quark.experimental.torch",
        "quark.experimental.torch.mix_precision",
    ):
        module = ModuleType(name)
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        if "." in name:
            parent_name, attr = name.rsplit(".", 1)
            setattr(modules[parent_name], attr, module)

    public = modules["quark.experimental.torch.mix_precision"]
    public.MixPrecisionConfig = type("MixPrecisionConfig", (), {})
    public.MixPrecisionQuantizer = type("MixPrecisionQuantizer", (), {})

    api = search._mix_precision_api()

    assert api.MixPrecisionConfig is public.MixPrecisionConfig
    assert api.MixPrecisionQuantizer is public.MixPrecisionQuantizer


def test_mix_precision_config_mapping_preserves_quant_perf_contract():
    from quark.experimental.torch.quant_perf.quantize import search

    spec = SimpleNamespace(
        gpu_type="mi355x",
        accuracy_gap=0.05,
        search_gsm8k_num_samples=64,
        gsm8k_num_samples=200,
        num_calib_data=512,
        calib_seqlen=512,
        layer_precision_candidates=["ptpc_fp8", "fp8", "mxfp4", "mxfp4_fp8"],
        kv_cache_precision_candidates=["native"],
        max_search_candidates=0,
        model_arch="gemma4",
        exclude_layers=["lm_head"],
    )

    config = search._build_mix_precision_config_kwargs(spec)

    assert config == {
        "granularity": "module",
        "hardware": "mi355",
        "eval_metrics": ["gsm8k"],
        "eval_threshold": 1.05,
        "eval_num_samples": 64,
        "eval_max_new_tokens": 256,
        "num_calib_samples": 512,
        "calib_seq_len": 512,
        "exclude_patterns": [
            "lm_head",
            "*vision_tower*",
            "*embed_vision*",
            "*audio_tower*",
            "*embed_audio*",
            "*router*",
        ],
        "max_configs": None,
        "early_stop": False,
        "search_modes": [
            "native",
            "ptpc_fp8",
            "fp8",
            "mxfp4",
            "mxfp4_fp8",
        ],
        "kv_cache_quant": False,
        "file2file_quantization": False,
    }


def test_native_and_fp8_kv_cache_maps_to_public_boolean():
    from quark.experimental.torch.quant_perf.quantize import search

    spec = SimpleNamespace(
        gpu_type="mi355x",
        accuracy_gap=0.03,
        gsm8k_num_samples=20,
        num_calib_data=8,
        calib_seqlen=128,
        layer_precision_candidates=["native", "fp8"],
        kv_cache_precision_candidates=["native", "fp8"],
        max_search_candidates=2,
        model_arch="qwen3",
        exclude_layers=None,
    )

    config = search._build_mix_precision_config_kwargs(spec)

    assert config["kv_cache_quant"] is True
    assert config["max_configs"] == 2
    assert config["early_stop"] is False


def test_fp8_only_kv_cache_search_fails_instead_of_widening_space():
    from quark.experimental.torch.quant_perf.quantize import search

    spec = SimpleNamespace(
        gpu_type="mi355x",
        accuracy_gap=0.03,
        gsm8k_num_samples=20,
        num_calib_data=8,
        calib_seqlen=128,
        layer_precision_candidates=["native", "fp8"],
        kv_cache_precision_candidates=["fp8"],
        max_search_candidates=2,
        model_arch="qwen3",
        exclude_layers=None,
    )

    with pytest.raises(ValueError, match="fp8-only KV-cache"):
        search._build_mix_precision_config_kwargs(spec)


def test_missing_requested_flydsl_dense_mxfp4_fails_before_search(
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.quantize import search

    monkeypatch.setattr(
        search,
        "_flydsl_dense_mxfp4_available",
        lambda: False,
    )
    spec = SimpleNamespace(mxfp4_gemm_backend="flydsl")

    with pytest.raises(
        StageError,
        match="does not expose its required A4W4 entry points",
    ):
        search._validate_runtime_capabilities(
            spec,
            {"search_modes": ["native", "mxfp4"]},
        )


def test_export_cache_cleanup_visits_every_visible_cuda_device(
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.quantize import search

    cleared = []
    active = [None]
    torch = ModuleType("torch")

    class DeviceContext:
        def __init__(self, index):
            self.index = index

        def __enter__(self):
            active[0] = self.index

        def __exit__(self, exc_type, exc, tb):
            active[0] = None

    class Cuda:
        @staticmethod
        def device_count():
            return 2

        @staticmethod
        def device(index):
            return DeviceContext(index)

        @staticmethod
        def empty_cache():
            cleared.append(active[0])

    torch.cuda = Cuda
    monkeypatch.setitem(sys.modules, "torch", torch)

    search._empty_all_cuda_caches()

    assert cleared == [0, 1]


def test_isolated_search_reloads_worker_checkpoint_state(
    monkeypatch,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.quantize import isolated_search

    monkeypatch.setenv("VLLM_ROCM_USE_AITER", "1")
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_MOE", "1")
    spec = Spec(
        model_dir="model",
        base_model="model",
        framework="vllm",
        gpu_type="mi355x",
        gpu_arch="MI355X",
        isl=128,
        osl=128,
        quant_strategy=None,
        search_timeout_s=123,
        inference_moe_backend="aiter",
        session_dir=str(tmp_path),
        runtime=RuntimeContext(runtime_python=sys.executable),
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.save()

    def fake_run(command, **kwargs):
        assert kwargs["timeout"] == 123
        assert kwargs["capture_output"] is False
        assert _search_runtime_args(spec) == ["--gpu-memory-utilization=0.75"]
        assert kwargs["env"]["VLLM_ROCM_USE_AITER"] == "1"
        assert kwargs["env"]["VLLM_ROCM_USE_AITER_MOE"] == "0"
        worker_ckpt = Checkpoint.load(tmp_path)
        assert worker_ckpt is not None
        worker_ckpt.state["best_candidate"] = {"mlp_mode": "mxfp4"}
        worker_ckpt.state["quant_ckpt_dir"] = "/quant"
        worker_ckpt.save()
        isolated_search.result_path(tmp_path).write_text(
            json.dumps(
                {
                    "status": "success",
                    "quant_ckpt_dir": "/quant",
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(
        isolated_search,
        "run_isolated_subprocess",
        fake_run,
        raising=False,
    )
    monkeypatch.setattr(
        isolated_search.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("legacy subprocess.run used")),
    )

    result = isolated_search.run_isolated_module_search(spec, ckpt)

    assert result == "/quant"
    assert ckpt.state["best_candidate"] == {"mlp_mode": "mxfp4"}
    assert ckpt.state["quant_ckpt_dir"] == "/quant"
    assert isolated_search.os.environ["VLLM_ROCM_USE_AITER_MOE"] == "1"


def test_isolated_search_propagates_worker_stage_error(
    monkeypatch,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.quantize import isolated_search

    spec = Spec(
        model_dir="model",
        base_model="model",
        framework="vllm",
        gpu_type="mi355x",
        gpu_arch="MI355X",
        isl=128,
        osl=128,
        quant_strategy=None,
        session_dir=str(tmp_path),
        runtime=RuntimeContext(runtime_python=sys.executable),
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.save()

    def fake_run(command, **kwargs):
        isolated_search.result_path(tmp_path).write_text(
            json.dumps(
                {
                    "status": "failed",
                    "stage": "quantize",
                    "message": "worker failed",
                    "code": "worker_error",
                    "diagnostic": "trace",
                }
            )
        )
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(
        isolated_search,
        "run_isolated_subprocess",
        fake_run,
        raising=False,
    )

    with pytest.raises(StageError, match="worker failed") as captured:
        isolated_search.run_isolated_module_search(spec, ckpt)

    assert captured.value.code == "worker_error"
    assert captured.value.diagnostic == "trace"


def test_isolated_search_timeout_is_reported_as_stage_error(
    monkeypatch,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.quantize import isolated_search

    spec = Spec(
        model_dir="model",
        base_model="model",
        framework="vllm",
        gpu_type="mi355x",
        gpu_arch="MI355X",
        isl=128,
        osl=128,
        quant_strategy=None,
        search_timeout_s=123,
        session_dir=str(tmp_path),
        runtime=RuntimeContext(runtime_python=sys.executable),
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.save()

    def timeout(*_args, **_kwargs):
        raise isolated_search.subprocess.TimeoutExpired(["python"], 123)

    monkeypatch.setattr(
        isolated_search,
        "run_isolated_subprocess",
        timeout,
        raising=False,
    )

    with pytest.raises(StageError, match="timed out after 123") as captured:
        isolated_search.run_isolated_module_search(spec, ckpt)

    assert captured.value.code == "search_worker_timeout"


@pytest.mark.parametrize(
    ("initial_status", "expected_status", "expected_reason"),
    [
        ("running", "partial_timeout", "search_timeout"),
        ("completed", "completed", None),
    ],
)
def test_isolated_search_timeout_retries_export_from_persisted_candidate(
    monkeypatch,
    tmp_path,
    initial_status,
    expected_status,
    expected_reason,
):
    from quark.experimental.torch.quant_perf.quantize import isolated_search

    spec = Spec(
        model_dir="model",
        base_model="model",
        framework="vllm",
        gpu_type="mi355x",
        gpu_arch="MI355X",
        isl=128,
        osl=128,
        quant_strategy=None,
        search_timeout_s=123,
        session_dir=str(tmp_path),
        runtime=RuntimeContext(runtime_python=sys.executable),
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["mix_precision_search"] = {
        "schema_version": 2,
        "status": initial_status,
        "config": {},
        "result": {
            "best_config": {"mlp_mode": "mxfp4"},
            "all_results": [
                {
                    "config": {"mlp_mode": "mxfp4"},
                    "metrics": {"gsm8k": 0.8},
                    "relative_change": {"gsm8k": -0.01},
                    "is_valid": True,
                    "rank": 2,
                }
            ],
            "baseline_metrics": {"gsm8k": 0.81},
            "total_configs_evaluated": 1,
            "total_configs_available": 4,
            "search_time_seconds": 30.0,
            "granularity": "module",
            "hardware": "mi355",
        },
        "candidate_queue": [
            {
                "config": {"mlp_mode": "mxfp4"},
                "search_metrics": {"gsm8k": 0.8},
                "relative_change": {"gsm8k": -0.01},
                "quark_rank": 2,
                "status": "pending",
            }
        ],
        "candidate_cursor": 0,
        "candidate_order_source": "quark_reverse_evaluation_order",
    }
    ckpt.save()
    calls = 0

    def fake_run(command, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise isolated_search.subprocess.TimeoutExpired(command, 123)
        worker_ckpt = Checkpoint.load(tmp_path)
        assert worker_ckpt is not None
        assert worker_ckpt.state["mix_precision_search"]["status"] == expected_status
        isolated_search.result_path(tmp_path).write_text(
            json.dumps(
                {
                    "status": "success",
                    "quant_ckpt_dir": str(tmp_path / "quant_ckpt"),
                }
            )
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(isolated_search, "run_isolated_subprocess", fake_run)

    assert isolated_search.run_isolated_module_search(spec, ckpt) == str(tmp_path / "quant_ckpt")
    assert calls == 2
    assert ckpt.state["mix_precision_search"]["status"] == expected_status
    if expected_reason is None:
        assert "termination_reason" not in ckpt.state["mix_precision_search"]
    else:
        assert ckpt.state["mix_precision_search"]["termination_reason"] == expected_reason


def test_isolated_search_reloads_checkpoint_when_worker_returns_no_result(
    monkeypatch,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.quantize import isolated_search

    spec = Spec(
        model_dir="model",
        base_model="model",
        framework="vllm",
        gpu_type="mi355x",
        gpu_arch="MI355X",
        isl=128,
        osl=128,
        quant_strategy=None,
        session_dir=str(tmp_path),
        runtime=RuntimeContext(runtime_python=sys.executable),
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.save()

    def fake_run(command, **kwargs):
        worker_ckpt = Checkpoint.load(tmp_path)
        assert worker_ckpt is not None
        worker_ckpt.state["best_candidate"] = {"mlp_mode": "mxfp4"}
        worker_ckpt.save()
        return SimpleNamespace(returncode=-9)

    monkeypatch.setattr(
        isolated_search,
        "run_isolated_subprocess",
        fake_run,
    )

    with pytest.raises(StageError, match="without a result"):
        isolated_search.run_isolated_module_search(spec, ckpt)

    assert ckpt.state["best_candidate"] == {"mlp_mode": "mxfp4"}


def test_candidate_queue_uses_best_then_reverse_quark_evaluation_order():
    conservative = {"mlp_mode": "ptpc_fp8"}
    invalid = {"mlp_mode": "mxfp4"}
    winner = {"mlp_mode": "mxfp4_fp8"}
    late_valid = {"mlp_mode": "fp8"}
    result = SimpleNamespace(
        best_config=winner,
        all_results=[
            SimpleNamespace(
                config=conservative,
                metrics={"gsm8k": 0.80},
                relative_change={"gsm8k": -0.01},
                is_valid=True,
                rank=1,
            ),
            SimpleNamespace(
                config=invalid,
                metrics={"gsm8k": 0.50},
                relative_change={"gsm8k": -0.30},
                is_valid=False,
                rank=4,
            ),
            SimpleNamespace(
                config=winner,
                metrics={"gsm8k": 0.79},
                relative_change={"gsm8k": -0.02},
                is_valid=True,
                rank=3,
            ),
            SimpleNamespace(
                config=late_valid,
                metrics={"gsm8k": 0.795},
                relative_change={"gsm8k": -0.015},
                is_valid=True,
                rank=2,
            ),
        ],
    )

    queue = _candidate_queue_from_result(result)

    assert [entry["config"] for entry in queue] == [
        winner,
        late_valid,
        conservative,
    ]
    assert all(entry["status"] == "pending" for entry in queue)


def test_advancing_search_candidate_persists_real_gate_rejection():
    from quark.experimental.torch.quant_perf.quantize import search

    state = {
        "mix_precision_search": {
            "status": "completed",
            "candidate_cursor": 0,
            "candidate_queue": [
                {"config": {"mlp_mode": "mxfp4"}, "status": "exported"},
                {"config": {"mlp_mode": "fp8"}, "status": "pending"},
            ],
        },
        "best_candidate": {"mlp_mode": "mxfp4"},
    }
    attempt = {
        "baseline": 0.8,
        "quantized": 0.6,
        "gap": 0.25,
        "passed": False,
    }

    assert search.advance_search_candidate(state, attempt=attempt)
    search_state = state["mix_precision_search"]
    assert search_state["candidate_cursor"] == 1
    assert search_state["candidate_queue"][0]["status"] == "rejected"
    assert search_state["candidate_queue"][0]["real_gate"] == attempt
    assert search_state["candidate_queue"][1]["status"] == "pending"
    assert state["best_candidate"] == {"mlp_mode": "mxfp4"}


def test_advancing_load_failure_prefers_minimal_config_change():
    from quark.experimental.torch.quant_perf.quantize import search

    winner = {
        "self_attn_mode": "fp8",
        "mlp_mode": "mxfp4",
        "shared_expert_mode": "mxfp4",
    }
    nearest = {
        "self_attn_mode": "native",
        "mlp_mode": "mxfp4",
        "shared_expert_mode": "mxfp4",
    }
    state = {
        "mix_precision_search": {
            "status": "completed",
            "candidate_cursor": 0,
            "candidate_queue": [
                {"config": winner, "status": "exported"},
                {
                    "config": {
                        "self_attn_mode": "fp8",
                        "mlp_mode": "fp8",
                        "shared_expert_mode": "fp8",
                    },
                    "status": "pending",
                },
                {"config": nearest, "status": "pending"},
            ],
        },
        "best_candidate": winner,
    }

    assert search.advance_search_candidate(
        state,
        attempt={"passed": False},
        prefer_minimal_change=True,
    )

    search_state = state["mix_precision_search"]
    assert search_state["candidate_cursor"] == 1
    assert search_state["candidate_queue"][1]["config"] == nearest
    assert state["best_candidate"] == winner


@pytest.mark.parametrize(
    "persisted_status",
    ["completed", "partial_timeout", "HIP out of memory", "No available memory for the cache blocks", "RPC timed out"],
)
def test_persisted_search_reexports_preprocessed_model_without_repeating_search(
    monkeypatch,
    tmp_path,
    persisted_status,
):
    from quark.experimental.torch.quant_perf.quantize import search

    result = SimpleNamespace(
        best_config={"mlp_mode": "mxfp4_fp8", "kv_cache_mode": "native"},
        all_results=[
            SimpleNamespace(
                config={"mlp_mode": "fp8", "kv_cache_mode": "native"},
                metrics={"gsm8k": 0.81},
                relative_change={"gsm8k": -0.01},
                is_valid=True,
                rank=1,
            ),
            SimpleNamespace(
                config={
                    "mlp_mode": "mxfp4_fp8",
                    "kv_cache_mode": "native",
                },
                metrics={"gsm8k": 0.80},
                relative_change={"gsm8k": -0.02},
                is_valid=True,
                rank=2,
            ),
        ],
        baseline_metrics={"gsm8k": 0.82},
        total_configs_evaluated=2,
        total_configs_available=2,
        search_time_seconds=12.5,
        granularity=SimpleNamespace(value="module"),
        hardware=SimpleNamespace(value="mi355"),
    )
    calls = {"search": 0, "exports": []}
    persisted_during_search = []

    class MixPrecisionConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class MixPrecisionQuantizer:
        def __init__(self, config):
            self.config = config
            self.result = None
            self.model_path = None

        export_reason = None

        def prepare_export(self, *_args, **_kwargs):
            return False

        def search(self, model_path, runtime_args, *, progress_callback):
            calls["search"] += 1
            self.model_path = model_path
            self.search_moe_backend_resolution = {"requested": "auto", "selected": "triton"}
            self.search_runtime_args = ["--moe-backend=triton"]
            progress_callback(
                SimpleNamespace(
                    best_config=result.all_results[0].config,
                    all_results=result.all_results[:1],
                    baseline_metrics=result.baseline_metrics,
                    total_configs_evaluated=1,
                    total_configs_available=result.total_configs_available,
                    search_time_seconds=6.0,
                    granularity=result.granularity,
                    hardware=result.hardware,
                )
            )
            persisted_during_search.append(dict(ckpt.state["mix_precision_search"]))
            if persisted_status not in {"completed", "partial_timeout"}:
                raise RuntimeError(persisted_status)
            self.result = result
            return result

        def export_best(self, output_dir):
            destination = Path(output_dir)
            write_valid_quant_checkpoint(destination)
            candidate = dict(self.result.best_config)
            (destination / "candidate.json").write_text(json.dumps(candidate, sort_keys=True))
            calls["exports"].append(candidate)
            return destination

    public = ModuleType("quark.experimental.torch.mix_precision")
    public.MixPrecisionConfig = MixPrecisionConfig
    public.MixPrecisionQuantizer = MixPrecisionQuantizer
    monkeypatch.setitem(
        sys.modules,
        "quark.experimental.torch.mix_precision",
        public,
    )

    spec = SimpleNamespace(
        gpu_type="mi355x",
        accuracy_gap=0.05,
        gsm8k_num_samples=20,
        num_calib_data=8,
        calib_seqlen=128,
        layer_precision_candidates=["native", "fp8", "mxfp4_fp8"],
        kv_cache_precision_candidates=["native"],
        max_search_candidates=0,
        model_arch="qwen3",
        exclude_layers=None,
        base_model="/models/qwen3-8b",
        effective_search_moe_backend="triton",
        search_vllm_args=["--tensor-parallel-size", "1", "--moe-backend=triton"],
        session_dir=str(tmp_path),
        mxfp4_gemm_backend="triton",
        w4a8_gemm_backend="triton",
        mxfp4_moe_backend="aiter",
    )

    class Checkpoint:
        def __init__(self):
            self.state = {
                "mix_precision_search": {
                    "schema_version": 1,
                    "status": "pending",
                    "candidate_queue": [],
                    "candidate_cursor": 0,
                }
            }

        def save(self):
            pass

    ckpt = Checkpoint()

    if persisted_status not in {"completed", "partial_timeout"}:
        with pytest.raises(StageError, match=persisted_status) as caught:
            search.run_module_search(spec, ckpt)
        assert ("--search-gpu-memory-utilization" in caught.value.message) == (persisted_status == "HIP out of memory")
        assert persisted_status in caught.value.diagnostic
        assert calls["exports"] == []
        assert ckpt.state["mix_precision_search"]["status"] == "failed"
        from quark.experimental.torch.quant_perf.quantize.isolated_search import _prepare_timeout_salvage

        assert not _prepare_timeout_salvage(ckpt)
        return

    first = search.run_module_search(spec, ckpt)

    assert calls["search"] == 1
    assert Path(first, "candidate.json").exists()
    assert persisted_during_search[0]["status"] == "running"
    assert persisted_during_search[0]["result"]["total_configs_evaluated"] == 1
    assert persisted_during_search[0]["candidate_queue"][0]["config"] == result.all_results[0].config
    assert ckpt.state["mix_precision_search"]["status"] == "completed"
    assert ckpt.state["mix_precision_search"]["moe_backend_resolution"]["selected"] == "triton"
    assert ckpt.state["mix_precision_search"]["resolved_runtime_args"] == ["--moe-backend=triton"]
    assert ckpt.state["best_candidate"] == result.best_config

    shutil.rmtree(first)
    ckpt.state["mix_precision_search"]["status"] = persisted_status
    second = search.run_module_search(spec, ckpt)

    assert calls["search"] == 1
    assert Path(second, "candidate.json").exists()
    assert calls["exports"] == [result.best_config, result.best_config]
    assert ckpt.state["mix_precision_search"]["status"] == persisted_status
    spec.search_vllm_args.append("--max-model-len=4096")
    with pytest.raises(StageError, match="runtime arguments changed"):
        search.run_module_search(spec, ckpt)


def test_search_persists_resolved_backend_when_engine_fails(tmp_path, monkeypatch):
    from quark.experimental.torch.quant_perf.quantize import search

    class FailingQuantizer:
        def __init__(self, config):
            self.search_moe_backend_resolution = None
            self.export_reason = None

        def prepare_export(self, model_path):
            return False

        def search(self, *args, **kwargs):
            self.search_moe_backend_resolution = {"requested": "auto", "selected": "triton_unfused"}
            self.search_runtime_args = ["--moe-backend=triton_unfused"]
            raise RuntimeError("engine startup failed")

    monkeypatch.setattr(
        search,
        "_mix_precision_api",
        lambda: SimpleNamespace(MixPrecisionConfig=lambda **kwargs: kwargs, MixPrecisionQuantizer=FailingQuantizer),
    )
    spec = _runtime_spec(tmp_path, search_gpu_memory_utilization=0.68)
    ckpt = Checkpoint.fresh(spec)
    with pytest.raises(StageError, match="engine startup failed") as failure:
        search.run_module_search(spec, ckpt)
    assert failure.value.code == "search_execution_failed"
    saved = Checkpoint.load(tmp_path).state["mix_precision_search"]
    assert saved["runtime_args"] == ["--gpu-memory-utilization=0.68"]
    assert saved["resolved_runtime_args"] == ["--moe-backend=triton_unfused"]
    assert saved["moe_backend_resolution"]["selected"] == "triton_unfused"


def test_fallback_export_preserves_rejected_checkpoint_when_space_is_insufficient(
    monkeypatch,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.quantize import search

    destination = tmp_path / "quant_ckpt"
    destination.mkdir()
    (destination / "rejected.bin").write_bytes(b"old")
    rejected = {
        "self_attn_mode": "fp8",
        "mlp_mode": "mxfp4",
        "shared_expert_mode": "mxfp4",
    }
    unrelated = {
        "self_attn_mode": "fp8",
        "mlp_mode": "fp8",
        "shared_expert_mode": "fp8",
    }
    fallback = {
        "self_attn_mode": "native",
        "mlp_mode": "mxfp4",
        "shared_expert_mode": "mxfp4",
    }

    class MixPrecisionConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class MixPrecisionQuantizer:
        def __init__(self, config):
            self.result = None
            self.model_path = None

        export_reason = None

        def prepare_export(self, *_args, **_kwargs):
            return False

        def export_best(self, output_dir):
            raise AssertionError("export must not start without enough free space")

    monkeypatch.setattr(
        search,
        "_mix_precision_api",
        lambda: SimpleNamespace(
            MixPrecisionConfig=MixPrecisionConfig,
            MixPrecisionQuantizer=MixPrecisionQuantizer,
        ),
    )
    monkeypatch.setattr(search, "_empty_all_cuda_caches", lambda: None)
    monkeypatch.setattr(
        search.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=0),
    )
    spec = SimpleNamespace(
        base_model="/models/glm",
        session_dir=str(tmp_path),
        effective_search_moe_backend="triton",
        search_vllm_args=["--moe-backend=triton"],
    )
    ckpt = SimpleNamespace(
        state={
            "best_candidate": rejected,
            "mix_precision_search": {
                "status": "completed",
                "config": {},
                "candidate_cursor": 1,
                "candidate_queue": [
                    {
                        "config": rejected,
                        "status": "rejected",
                        "real_gate": {
                            "reason": "quantized_accuracy_load_failed",
                        },
                    },
                    {"config": unrelated, "status": "pending"},
                    {"config": fallback, "status": "pending"},
                ],
            },
        },
        save=lambda: None,
    )

    with pytest.raises(
        StageError,
        match="insufficient free space",
    ) as captured:
        search.run_module_search(spec, ckpt)

    assert captured.value.code == "insufficient_storage_for_candidate_fallback"
    assert destination.is_dir()
    assert (destination / "rejected.bin").read_bytes() == b"old"
    assert ckpt.state["best_candidate"] == rejected
