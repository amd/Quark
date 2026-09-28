#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for quark.experimental.torch.quant_perf.session.spec: Spec round-trip, Checkpoint atomic
save/load, and SessionLock mutual exclusion.
"""

import concurrent.futures
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pytest

from quark.experimental.torch.quant_perf.session.spec import (
    SCHEMA_VERSION,
    Checkpoint,
    DeployPackage,
    EvalProfile,
    SessionLock,
    Spec,
    StageError,
)


def make_spec(tmp_path: Path, **overrides) -> Spec:
    defaults = dict(
        model_dir="m",
        base_model="b",
        framework="atom",
        gpu_type="mi300x",
        gpu_arch="MI300X",
        isl=128,
        osl=128,
        quant_strategy=None,
        session_dir=str(tmp_path),
    )
    defaults.update(overrides)
    return Spec(**defaults)


def test_spec_round_trip(tmp_path):
    spec = make_spec(
        tmp_path,
        quant_strategy="fp8",
        performance_mode="optimize",
        target_gain=1.5,
        exclude_layers=["lm_head"],
        search_moe_backend="triton",
        inference_moe_backend="aiter",
        eval_profile=EvalProfile(
            profile_id="gsm8k-base-default-v1",
            profile_hash="abc",
            model_mode="base",
            apply_chat_template=False,
            enable_thinking=None,
            detection_reason="fallback_base",
        ),
    )
    restored = Spec.from_dict(spec.to_dict())
    assert restored == spec


@pytest.mark.parametrize("phase", ["search", "inference"])
def test_conflicting_legacy_and_phase_backend_is_rejected(tmp_path, phase):
    with pytest.raises(ValueError, match="conflict"):
        make_spec(tmp_path, vllm_extra_args=["--moe-backend=aiter"], **{f"{phase}_moe_backend": "triton"})


@pytest.mark.parametrize(
    ("config", "hip", "scheme", "extra", "expected"),
    [
        ({"model_type": "llama"}, "7.2", None, [], "auto"),
        ({"model_type": "llama"}, "7.2", "fp8", [], "fp8"),
        ({"model_type": "llama"}, None, None, ["--kv_cache_dtype", "bfloat16"], "bfloat16"),
        ({"model_type": "deepseek_v4"}, "7.2", None, [], "fp8_ds_mla"),
        ({"text_config": {"model_type": "deepseek_v4"}}, "7.2", None, [], "fp8_ds_mla"),
        ({"model_type": "deepseek_v4"}, None, None, [], "auto"),
        ({"model_type": "deepseek_v4"}, "7.2", "fp8", ["--kv-cache-dtype=fp8"], "fp8_ds_mla"),
        ({}, "7.2", None, ["--kv-cache-dtype fp8", "--kv-cache-dtype=fp8"], "fp8"),
    ],
)
def test_kv_dtype_is_shared_without_changing_native_search(tmp_path, monkeypatch, config, hip, scheme, extra, expected):
    import torch

    monkeypatch.setattr(torch.version, "hip", hip)
    (tmp_path / "config.json").write_text(json.dumps(config))
    spec = make_spec(
        tmp_path,
        framework="vllm",
        base_model=str(tmp_path),
        kv_cache_scheme=scheme,
        vllm_extra_args=extra,
        kv_cache_precision_candidates=["native"],
    )
    expected_args = [] if expected == "auto" else [f"--kv-cache-dtype={expected}"]
    assert spec.search_vllm_args == spec.inference_vllm_args == expected_args
    assert spec.kv_cache_precision_candidates == ["native"]
    assert spec.kv_cache_scheme == scheme


@pytest.mark.parametrize(
    ("scheme", "extra"),
    [
        ("fp8", ["--kv-cache-dtype=bfloat16"]),
        (None, ["--kv-cache-dtype=fp8", "--kv-cache-dtype=auto"]),
        (None, ["--kv-cache-dtype"]),
        (None, ["--kv-cache-dtype="]),
    ],
)
def test_conflicting_or_incomplete_kv_dtype_is_rejected(tmp_path, scheme, extra):
    with pytest.raises(ValueError, match="kv-cache"):
        make_spec(tmp_path, kv_cache_scheme=scheme, vllm_extra_args=extra)


def test_deepseek_rocm_rejects_explicit_bf16_kv_before_launch(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "hip", "7.2")
    (tmp_path / "config.json").write_text('{"model_type": "deepseek_v4"}')
    spec = make_spec(tmp_path, base_model=str(tmp_path), vllm_extra_args=["--kv-cache-dtype=bfloat16"])
    with pytest.raises(ValueError, match="DeepSeek V4.*fp8"):
        _ = spec.inference_vllm_args


def test_native_kv_uses_cached_hub_config(tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.version, "hip", "7.2")
    config = tmp_path / "config.json"
    config.write_text('{"model_type": "deepseek_v4"}')
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *args: str(config))
    spec = make_spec(tmp_path, base_model="deepseek-ai/DeepSeek-V4-Flash-0731")
    assert spec.inference_vllm_args == ["--kv-cache-dtype=fp8_ds_mla"]


def test_spec_from_dict_preserves_legacy_performance_behavior(tmp_path):
    data = make_spec(tmp_path).to_dict()
    data.pop("performance_mode", None)
    data["target_gain"] = 1.2

    restored = Spec.from_dict(data)

    assert restored.performance_mode == "optimize"
    assert restored.target_gain == 1.2


def test_spec_from_dict_ignores_unknown_fields(tmp_path):
    spec = make_spec(tmp_path)
    data = spec.to_dict()
    data["some_future_field"] = "ignored"
    data["algorithm"] = "awq"
    data["geak_bench_concurrency"] = 16
    restored = Spec.from_dict(data)
    assert restored == spec


def test_spec_is_immutable_after_construction(tmp_path):
    spec = make_spec(tmp_path)

    with pytest.raises(FrozenInstanceError):
        spec.model_dir = "changed"


def test_active_repo_properties_prefer_runtime_worktrees(tmp_path):
    spec = make_spec(
        tmp_path,
        framework_repo="/src/vllm",
        kernel_repo="/src/aiter",
    )

    assert spec.active_framework_repo == "/src/vllm"
    assert spec.active_kernel_repo == "/src/aiter"

    spec.runtime.framework_worktree = "/tmp/session/vllm"
    spec.runtime.kernel_worktree = "/tmp/session/aiter"

    assert spec.active_framework_repo == "/tmp/session/vllm"
    assert spec.active_kernel_repo == "/tmp/session/aiter"
    assert "runtime" not in spec.to_dict()
    assert "framework_worktree" not in spec.to_dict()


def test_tp_parses_equals_form(tmp_path):
    spec = make_spec(
        tmp_path,
        vllm_extra_args=["--tensor-parallel-size=4"],
    )
    assert spec.tp == 4
    assert spec.vllm_passthrough_args == []


def test_server_host_owns_vllm_host_and_port_arguments(tmp_path):
    spec = make_spec(
        tmp_path,
        vllm_extra_args=[
            "--host=0.0.0.0",
            "--port",
            "9001",
            "--disable-log-requests",
        ],
    )

    assert spec.server_host == "127.0.0.1"
    assert spec.vllm_passthrough_args == ["--disable-log-requests"]


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], 1),
        (["--tensor-parallel-size", "4"], 4),
        (["-tp", "8"], 8),
        (["--tensor-parallel-size", "invalid"], 1),
    ],
)
def test_tp_parses_supported_argument_forms(
    tmp_path,
    extra_args,
    expected,
):
    spec = make_spec(tmp_path, vllm_extra_args=extra_args)
    assert spec.tp == expected


def test_quant_ckpt_dir_is_inside_session(tmp_path):
    assert make_spec(tmp_path).quant_ckpt_dir == f"{tmp_path}/quant_ckpt"


@pytest.mark.parametrize(
    ("extra_args", "expected"),
    [
        ([], None),
        (["--max-num-seqs 64"], 64),
        (["--max-num-seqs", "32"], 32),
        (["--max-num-seqs=128"], 128),
        (["--max_num_seqs 16"], 16),
        (["--max_num_seqs=8"], 8),
        (["--max-num-seqs 1024", "--max_num_seqs=64"], 64),
    ],
)
def test_vllm_max_num_seqs_survives_spec_round_trip(tmp_path, extra_args, expected):
    spec = make_spec(tmp_path, vllm_extra_args=extra_args)
    restored = Spec.from_dict(spec.to_dict())

    assert restored.vllm_max_num_seqs == expected


@pytest.mark.parametrize("value", ["0", "-1", "1.5", "invalid", ""])
def test_vllm_max_num_seqs_rejects_invalid_limits(tmp_path, value):
    spec = make_spec(tmp_path, vllm_extra_args=[f"--max-num-seqs {value}"])

    with pytest.raises(ValueError, match="positive integer"):
        _ = spec.vllm_max_num_seqs


def test_checkpoint_fresh_state_shape(tmp_path):
    spec = make_spec(tmp_path, quant_strategy="fp8")
    ckpt = Checkpoint.fresh(spec)
    assert ckpt.state["schema_version"] == SCHEMA_VERSION == 9
    assert ckpt.state["runtime_context"] == {}
    assert ckpt.state["path"] == "direct_ptq"
    assert ckpt.state["stage"] == "quantize"
    assert ckpt.state["accuracy_attempts"] == []
    assert ckpt.state["performance_measurements"] == []
    assert ckpt.state["accuracy_validation"] is None
    assert ckpt.state["post_repair_rechecked_configs"] == []
    assert ckpt.state["vendor_gemm_tuning"] == {}
    assert ckpt.state["retained_runtime_env"] == {}
    assert ckpt.state["phase_timeline"] == []
    assert ckpt.state["kernel_journey"] == []
    assert ckpt.state["retain_trials"] == []
    assert ckpt.state["reporting"]["status"] == "pending"
    assert ckpt.state["repo_workspaces"] == {}
    assert ckpt.state["transient_resources"] == []
    assert ckpt.state["cleanup"]["status"] == "pending"
    assert ckpt.state["eval_profile"] is None
    assert ckpt.state["eval_profile_hash"] is None
    assert ckpt.state["baseline_reference"] is None
    assert ckpt.state["baseline_runtime_health"] is None
    assert ckpt.state["performance_runtime"] is None
    assert ckpt.state["recovery_attempts"] == []
    assert ckpt.state["performance_validation"]["policy_version"] == 1
    assert ckpt.state["repair_journey"] == []
    assert ckpt.state["change_ledger"] == []
    assert ckpt.state["mix_precision_search"] == {
        "schema_version": 1,
        "status": "pending",
        "api": "quark.experimental.torch.mix_precision",
        "config": {},
        "result": {},
        "candidate_queue": [],
        "candidate_cursor": 0,
        "candidate_order_source": "quark_reverse_evaluation_order",
    }
    ckpt.record_phase_event(
        "baseline_health",
        "passed",
        baseline=0.8,
        ignored=None,
    )
    assert ckpt.state["phase_timeline"][-1]["action"] == "baseline_health"
    assert ckpt.state["phase_timeline"][-1]["status"] == "passed"
    assert ckpt.state["phase_timeline"][-1]["baseline"] == 0.8
    assert "ignored" not in ckpt.state["phase_timeline"][-1]


def test_checkpoint_path_a_has_no_quant_strategy(tmp_path):
    spec = make_spec(tmp_path, quant_strategy=None)
    ckpt = Checkpoint.fresh(spec)
    assert ckpt.state["path"] == "mix_precision_search"


def test_checkpoint_atomic_save_and_load(tmp_path):
    spec = make_spec(tmp_path)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["mix_precision_search"]["candidate_queue"].append({"config": {"mode": "fp8"}, "status": "pending"})
    ckpt.save()

    reloaded = Checkpoint.load(tmp_path)
    assert reloaded is not None
    assert reloaded.state == ckpt.state

    # No leftover tmp file after a successful save.
    assert list(tmp_path.glob("state.tmp.*")) == []


def test_checkpoint_load_missing_returns_none(tmp_path):
    assert Checkpoint.load(tmp_path) is None


def test_checkpoint_save_is_never_partially_visible(tmp_path):
    """Concurrent writers must never leave state.json truncated/corrupt --
    each save() is a single atomic os.replace, so every read sees either the
    old or the new complete content, never a half-written file."""
    spec = make_spec(tmp_path)
    ckpt = Checkpoint.fresh(spec)
    ckpt.save()

    def write_round(i: int) -> None:
        c = Checkpoint.load(tmp_path)
        c.state["phase_timeline"].append({"action": "concurrent_write", "round": i})
        c.save()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(write_round, range(20)))

    # Every read after the storm must parse as valid, complete JSON.
    final = json.loads((tmp_path / "state.json").read_text())
    assert final["schema_version"] == SCHEMA_VERSION


def test_checkpoint_rejects_unknown_schema_version(tmp_path):
    spec = make_spec(tmp_path)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["schema_version"] = 999
    ckpt.save()

    with pytest.raises(StageError):
        Checkpoint.load(tmp_path)


def test_checkpoint_migrates_v8_runtime_fields_without_losing_explicit_repos(
    tmp_path,
):
    spec = make_spec(
        tmp_path,
        framework_repo="/explicit/vllm",
        kernel_repo="/explicit/aiter",
    )
    state = Checkpoint.fresh(spec).state
    state["schema_version"] = 8
    state["run_spec"].update(
        {
            "framework_repo": "/auto/vllm",
            "kernel_repo": "/auto/aiter",
            "framework_worktree": "/work/vllm",
            "kernel_worktree": "/work/aiter",
            "framework_branch": "quant-perf/framework",
            "kernel_branch": "quant-perf/kernel",
            "framework_version": "framework-sha",
            "kernel_version": "kernel-sha",
            "runtime_python": "/venv/bin/python",
            "runtime_env": {"PYTHONPATH": "/work/vllm"},
            "runtime_origins": {"vllm": "/work/vllm"},
        }
    )
    state["invocation"]["spec"]["framework_repo"] = "/explicit/vllm"
    state["invocation"]["spec"]["kernel_repo"] = "/explicit/aiter"
    state.pop("runtime_context", None)
    (tmp_path / "state.json").write_text(json.dumps(state))

    migrated = Checkpoint.load(tmp_path)

    assert migrated is not None
    assert migrated.state["schema_version"] == 9
    assert migrated.state["run_spec"]["framework_repo"] == "/explicit/vllm"
    assert migrated.state["run_spec"]["kernel_repo"] == "/explicit/aiter"
    assert "framework_worktree" not in migrated.state["run_spec"]
    assert migrated.state["runtime_context"] == {
        "resolved_framework_repo": "/auto/vllm",
        "resolved_kernel_repo": "/auto/aiter",
        "runtime_python": "/venv/bin/python",
        "runtime_env": {"PYTHONPATH": "/work/vllm"},
        "runtime_origins": {"vllm": "/work/vllm"},
        "framework_branch": "quant-perf/framework",
        "framework_version": "framework-sha",
        "kernel_branch": "quant-perf/kernel",
        "kernel_version": "kernel-sha",
        "framework_worktree": "/work/vllm",
        "kernel_worktree": "/work/aiter",
    }


def test_stage_error_preserves_code_and_diagnostic():
    error = StageError(
        "land",
        "server failed",
        code="address_in_use",
        diagnostic="OSError: [Errno 98] Address already in use",
    )

    assert error.code == "address_in_use"
    assert error.diagnostic.endswith("Address already in use")


def test_deploy_package_exposes_report_status_and_paths():
    package = DeployPackage(
        status="success",
        report_status="complete",
        report_paths={"final_json": "/session/reports/final.json"},
    )
    assert package.report_status == "complete"
    assert package.report_paths["final_json"].endswith("final.json")


def test_session_lock_rejects_second_acquire(tmp_path):
    with SessionLock(tmp_path), pytest.raises(StageError), SessionLock(tmp_path):
        pass  # pragma: no cover


def test_session_lock_reacquirable_after_release(tmp_path):
    with SessionLock(tmp_path):
        pass
    with SessionLock(tmp_path):
        pass  # should not raise -- the first lock was released
