#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for quark.experimental.torch.quant_perf.landing: command construction (no GPU/subprocess
actually run) and the framework dispatcher."""

import json
import os
import signal
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quark.experimental.torch.quant_perf.landing import load
from quark.experimental.torch.quant_perf.landing.base import wait_ready
from quark.experimental.torch.quant_perf.session.spec import RuntimeContext, ServerHandle, Spec, StageError


def make_spec(**overrides) -> Spec:
    defaults = dict(
        model_dir="m",
        base_model="m",
        framework="atom",
        gpu_type="mi300x",
        gpu_arch="MI300X",
        isl=128,
        osl=128,
        quant_strategy="fp8",
    )
    runtime_fields = set(RuntimeContext.__dataclass_fields__)
    runtime_overrides = {key: overrides.pop(key) for key in list(overrides) if key in runtime_fields}
    defaults.update(overrides)
    return Spec(**defaults, runtime=RuntimeContext(**runtime_overrides))


def _write_w4a4_moe_config(model_dir: Path) -> None:
    (model_dir / "config.json").write_text(
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


@patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.subprocess.Popen")
def test_inference_triton_override_wins_over_flydsl_preference(mock_popen, tmp_path, monkeypatch):
    from quark.experimental.torch.quant_perf.landing.vllm_adapter import start_vllm_server

    _write_w4a4_moe_config(tmp_path)
    monkeypatch.setenv("QUARK_QUANT_PERF_MXFP4_MOE_BACKEND", "flydsl")
    start_vllm_server(str(tmp_path), tp=1, port=9000, kv_cache_scheme=None, extra_args=["--moe-backend=triton"])

    cmd = mock_popen.call_args.args[0]
    env = mock_popen.call_args.kwargs["env"]
    assert "--moe-backend=triton" in cmd
    assert "--moe-backend" not in cmd
    assert env["VLLM_ROCM_USE_AITER_MOE"] == "0"


def test_dispatch_rejects_unknown_framework():
    spec = make_spec(framework="unsupported")
    try:
        load("/tmp/quant_ckpt", spec)
        raise AssertionError("should have raised")
    except StageError as e:
        assert "unsupported" in str(e)


@patch("quark.experimental.torch.quant_perf.landing._serve")
def test_landing_propagates_failure_without_nested_repair(mock_serve):
    mock_serve.side_effect = StageError("land", "load failed")
    spec = make_spec(framework="vllm", framework_repo="/repo")

    with pytest.raises(StageError, match="load failed"):
        load("/tmp/quant_ckpt", spec)

    mock_serve.assert_called_once()


@pytest.mark.parametrize("profiling,override", [(False, False), (True, False), (True, True)])
@patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.subprocess.Popen")
def test_dispatch_routes_vllm_to_vllm_adapter(mock_popen, profiling, override):
    mock_popen.return_value = MagicMock()
    spec = make_spec(
        framework="vllm",
        server_port=9100,
        vllm_extra_args=[
            "--tensor-parallel-size 2",
            "--disable-log-requests",
        ],
    )
    if override:
        spec.vllm_extra_args.extend(["--max-model-len 4096", "--max-num-seqs=32", "--gpu-memory-utilization=0.7"])
    with (
        patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.ensure_port_available"),
        patch(
            "quark.experimental.torch.quant_perf.landing.vllm_adapter.wait_ready",
            return_value=True,
        ),
    ):
        handle = load("/tmp/quant_ckpt", spec, profiler_dir="/trace" if profiling else None)
    assert handle.port == 9100
    cmd = mock_popen.call_args.args[0]
    assert cmd[:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert cmd.count("--tensor-parallel-size") == 1
    assert cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
    assert "--disable-log-requests" in cmd
    env = mock_popen.call_args.kwargs["env"]
    assert env["ROCR_VISIBLE_DEVICES"] == "0,1"
    if override:
        assert cmd[cmd.index("--max-model-len") + 1] == "4096"
        assert "--max-num-seqs=32" in cmd and "--max-num-seqs" not in cmd
        assert "--gpu-memory-utilization=0.7" in cmd and "--gpu-memory-utilization" not in cmd
    elif profiling:
        assert cmd[cmd.index("--max-model-len") + 1] == "512"
        assert cmd[cmd.index("--max-num-seqs") + 1] == str(spec.bench_concurrency)
        assert cmd[cmd.index("--gpu-memory-utilization") + 1] == "0.85"
    else:
        assert "--max-model-len" not in cmd and "--max-num-seqs" not in cmd


@patch("quark.experimental.torch.quant_perf.landing.atom_adapter.subprocess.Popen")
def test_dispatch_routes_spec_tp_to_atom_adapter(mock_popen):
    mock_popen.return_value = MagicMock()
    spec = make_spec(
        framework="atom",
        server_port=9100,
        vllm_extra_args=["--tensor-parallel-size 2"],
    )
    with (
        patch("quark.experimental.torch.quant_perf.landing.atom_adapter.ensure_port_available"),
        patch(
            "quark.experimental.torch.quant_perf.landing.atom_adapter.wait_ready",
            return_value=True,
        ),
    ):
        load("/tmp/quant_ckpt", spec)
    cmd = mock_popen.call_args.args[0]
    assert cmd[cmd.index("-tp") + 1] == "2"
    env = mock_popen.call_args.kwargs["env"]
    assert env["ROCR_VISIBLE_DEVICES"] == "0,1"


@pytest.mark.parametrize(
    ("framework", "adapter"),
    [
        ("vllm", "vllm_adapter"),
        ("atom", "atom_adapter"),
    ],
)
@pytest.mark.parametrize("returncode", [None, 1])
def test_readiness_failure_stops_server_process_group(
    framework,
    adapter,
    returncode,
):
    proc = MagicMock()
    proc.poll.return_value = returncode
    module = f"quark.experimental.torch.quant_perf.landing.{adapter}"
    start_name = f"start_{framework}_server"

    with (
        patch(f"{module}.ensure_port_available"),
        patch(f"{module}.{start_name}", return_value=proc),
        patch(f"{module}.wait_ready", return_value=False),
        patch.object(ServerHandle, "stop", autospec=True) as stop,
        pytest.raises(StageError, match="never became ready"),
    ):
        load("/tmp/quant_ckpt", make_spec(framework=framework))

    stop.assert_called_once()
    assert stop.call_args.args[0].process_group_id == proc.pid


def test_server_handle_stop_uses_captured_process_group_after_leader_exit():
    proc = MagicMock(pid=1234)
    handle = ServerHandle(port=9100, proc=proc, process_group_id=1234)

    with (
        patch("quark.experimental.torch.quant_perf.session.spec.os.getpgid") as getpgid,
        patch("quark.experimental.torch.quant_perf.session.spec.os.killpg") as killpg,
    ):
        handle.stop()

    getpgid.assert_not_called()
    killpg.assert_called_once_with(1234, signal.SIGTERM)
    proc.terminate.assert_not_called()
    proc.wait.assert_called_once_with(timeout=30)


@patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.subprocess.Popen")
def test_start_vllm_server_uses_openai_api_server_module(mock_popen):
    from quark.experimental.torch.quant_perf.landing.vllm_adapter import start_vllm_server

    mock_popen.return_value = MagicMock()
    start_vllm_server(model_dir="/quant_ckpt", tp=2, port=9000, kv_cache_scheme="fp8")

    cmd = mock_popen.call_args.args[0]
    assert cmd[:3] == [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    assert "--tensor-parallel-size" in cmd and cmd[cmd.index("--tensor-parallel-size") + 1] == "2"
    assert "--kv-cache-dtype" in cmd and cmd[cmd.index("--kv-cache-dtype") + 1] == "fp8"


@patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.subprocess.Popen")
def test_start_vllm_server_uses_session_runtime(mock_popen):
    from quark.experimental.torch.quant_perf.landing.vllm_adapter import start_vllm_server

    mock_popen.return_value = MagicMock()
    start_vllm_server(
        model_dir="/quant_ckpt",
        tp=1,
        port=9000,
        kv_cache_scheme=None,
        python_exe="/session/venv/bin/python",
        runtime_env={"PYTHONPATH": "/session/overlay"},
    )

    cmd = mock_popen.call_args.args[0]
    env = mock_popen.call_args.kwargs["env"]
    assert cmd[:3] == [
        "/session/venv/bin/python",
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]
    assert env["PYTHONPATH"] == "/session/overlay"


@patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.subprocess.Popen")
@pytest.mark.parametrize(
    ("model_type", "hip", "expected"),
    [("llama", "7.2", "auto"), ("deepseek_v4", "7.2", "fp8_ds_mla"), ("deepseek_v4", None, "auto")],
)
def test_start_vllm_server_resolves_native_kv_cache(mock_popen, tmp_path, monkeypatch, model_type, hip, expected):
    import torch

    from quark.experimental.torch.quant_perf.landing.vllm_adapter import start_vllm_server

    monkeypatch.setattr(torch.version, "hip", hip)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": model_type}))
    mock_popen.return_value = MagicMock()
    start_vllm_server(
        model_dir=str(tmp_path), tp=1, port=9000, kv_cache_scheme=None, extra_args=["--kv-cache-dtype=auto"]
    )

    cmd = mock_popen.call_args.args[0]
    assert cmd[cmd.index("--kv-cache-dtype") + 1] == expected
    assert sum(arg.startswith("--kv-cache-dtype") for arg in cmd) == 1


@patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.subprocess.Popen")
def test_start_vllm_server_profiles_engine_shapes_without_frontend(
    mock_popen,
):
    from quark.experimental.torch.quant_perf.landing.vllm_adapter import start_vllm_server

    mock_popen.return_value = MagicMock()
    start_vllm_server(
        model_dir="/quant_ckpt",
        tp=1,
        port=9000,
        kv_cache_scheme=None,
        profiler_dir="/tmp/trace",
    )

    cmd = mock_popen.call_args.args[0]
    profiler = json.loads(cmd[cmd.index("--profiler-config") + 1])
    assert profiler["ignore_frontend"] is True
    assert profiler["torch_profiler_record_shapes"] is True


@patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.subprocess.Popen")
def test_start_vllm_server_rocr_visible_devices_starts_at_gpu_id(mock_popen):
    from quark.experimental.torch.quant_perf.landing.vllm_adapter import start_vllm_server

    mock_popen.return_value = MagicMock()
    start_vllm_server(model_dir="/quant_ckpt", tp=2, port=9000, kv_cache_scheme="fp8", gpu_id=4)

    env = mock_popen.call_args.kwargs["env"]
    assert env["ROCR_VISIBLE_DEVICES"] == "4,5"


@patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.subprocess.Popen")
def test_start_vllm_server_disables_atom_plugin_auto_registration(mock_popen):
    """ATOM registers itself as a global vLLM platform plugin; on this host
    that plugin references a module that doesn't exist in the installed vLLM
    version, breaking `import vllm` entirely unless plugin auto-discovery is
    disabled (confirmed via tests/spikes/README.md's 2026-07-07 cross-check)."""
    from quark.experimental.torch.quant_perf.landing.vllm_adapter import start_vllm_server

    mock_popen.return_value = MagicMock()
    start_vllm_server(model_dir="/quant_ckpt", tp=1, port=9000, kv_cache_scheme="fp8")

    env = mock_popen.call_args.kwargs["env"]
    assert env["VLLM_PLUGINS"] == ""


@patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.subprocess.Popen")
def test_start_vllm_server_uses_checkpoint_scoped_compile_cache(
    mock_popen,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.landing.vllm_adapter import start_vllm_server

    mock_popen.return_value = MagicMock()
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_5_moe","quantization_config":{"scheme":"mxfp4"}}')

    start_vllm_server(
        model_dir=str(tmp_path),
        tp=1,
        port=9000,
        kv_cache_scheme=None,
    )

    env = mock_popen.call_args.kwargs["env"]
    assert env["VLLM_CACHE_ROOT"].startswith("/tmp/quark_quant_perf_vllm_cache/")


@patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.subprocess.Popen")
def test_start_vllm_server_aiter_backend_uses_safe_single_k_with_cuda_graphs(
    mock_popen,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.landing.vllm_adapter import start_vllm_server

    mock_popen.return_value = MagicMock()
    _write_w4a4_moe_config(tmp_path)
    with (
        patch.dict(os.environ, {"VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE": "1"}, clear=False),
        patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.is_mxfp4_moe_model", return_value=True),
        patch("quark.experimental.torch.quant_perf.runtime.backends.mxfp4_moe_backend", return_value="aiter"),
    ):
        start_vllm_server(
            model_dir=str(tmp_path),
            tp=1,
            port=9000,
            kv_cache_scheme="fp8",
        )

    env = mock_popen.call_args.kwargs["env"]
    assert "VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE" not in env
    cmd = mock_popen.call_args.args[0]
    assert "--compilation-config" not in cmd
    assert env["AITER_KSPLIT"] == "1"


@patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.subprocess.Popen")
def test_start_vllm_server_respects_explicit_non_aiter_moe_backend(mock_popen, tmp_path):
    from quark.experimental.torch.quant_perf.landing.vllm_adapter import start_vllm_server

    mock_popen.return_value = MagicMock()
    _write_w4a4_moe_config(tmp_path)
    with patch("quark.experimental.torch.quant_perf.runtime.backends.mxfp4_moe_backend", return_value="aiter"):
        start_vllm_server(
            model_dir=str(tmp_path),
            tp=1,
            port=9000,
            kv_cache_scheme="fp8",
            extra_args=["--moe-backend", "triton"],
        )

    env = mock_popen.call_args.kwargs["env"]
    assert "AITER_KSPLIT" not in env


@patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.subprocess.Popen")
def test_start_vllm_server_preserves_aiter_for_fp8_glm_dsa(mock_popen, tmp_path):
    from quark.experimental.torch.quant_perf.landing.vllm_adapter import start_vllm_server

    mock_popen.return_value = MagicMock()
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "glm_moe_dsa",
                "quantization_config": {
                    "quant_method": "fp8",
                },
            }
        )
    )

    with (
        patch.dict(os.environ, {}, clear=True),
        patch("quark.experimental.torch.quant_perf.runtime.backends.mxfp4_moe_backend", return_value="flydsl"),
    ):
        start_vllm_server(
            model_dir=str(tmp_path),
            tp=1,
            port=9000,
            kv_cache_scheme=None,
        )

    cmd = mock_popen.call_args.args[0]
    env = mock_popen.call_args.kwargs["env"]
    assert env["VLLM_ROCM_USE_AITER"] == "1"
    assert env["VLLM_ROCM_USE_AITER_MOE"] == "1"
    assert "VLLM_ROCM_USE_AITER_FLYDSL_MOE" not in env
    assert "AITER_FLYDSL_FORCE" not in env
    assert "AITER_KSPLIT" not in env
    assert "--moe-backend" not in cmd
    assert "--no-enable-prefix-caching" not in cmd
    assert "--compilation-config" not in cmd


@patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.subprocess.Popen")
def test_start_vllm_server_flydsl_backend_uses_aiter_and_tuned_config(mock_popen):
    from quark.experimental.torch.quant_perf.landing.vllm_adapter import start_vllm_server

    mock_popen.return_value = MagicMock()
    with (
        patch.dict(
            os.environ,
            {"AITER_CONFIG_FMOE": "/configs/qwen35.csv"},
            clear=False,
        ),
        patch("quark.experimental.torch.quant_perf.landing.vllm_adapter.is_mxfp4_moe_model", return_value=True),
        patch("quark.experimental.torch.quant_perf.runtime.backends.mxfp4_moe_backend", return_value="flydsl"),
    ):
        start_vllm_server(
            model_dir="/quant_ckpt",
            tp=1,
            port=9000,
            kv_cache_scheme=None,
        )

    cmd = mock_popen.call_args.args[0]
    env = mock_popen.call_args.kwargs["env"]
    assert cmd[cmd.index("--moe-backend") + 1] == "aiter"
    assert env["AITER_CONFIG_FMOE"] == "/configs/qwen35.csv"
    assert env["AITER_FLYDSL_FORCE"] == "1"
    assert env["VLLM_ROCM_USE_AITER_FLYDSL_MOE"] == "1"


@patch("quark.experimental.torch.quant_perf.landing.atom_adapter.subprocess.Popen")
def test_start_atom_server_uses_module_invocation_not_console_script(mock_popen):
    from quark.experimental.torch.quant_perf.landing.atom_adapter import start_atom_server

    mock_popen.return_value = MagicMock()
    start_atom_server(model_dir="/quant_ckpt", tp=2, port=9000, kv_cache_scheme="fp8")

    cmd = mock_popen.call_args.args[0]
    assert cmd[:3] == [sys.executable, "-m", "atom.entrypoints.openai_server"]
    assert "atom" not in cmd  # no bare "atom" console-script entry
    assert "serve" not in cmd  # no "serve" subcommand -- verified not to exist
    assert "-tp" in cmd and cmd[cmd.index("-tp") + 1] == "2"
    assert "--kv_cache_dtype" in cmd and cmd[cmd.index("--kv_cache_dtype") + 1] == "fp8"


@patch("quark.experimental.torch.quant_perf.landing.atom_adapter.subprocess.Popen")
def test_start_atom_server_rocr_visible_devices_starts_at_gpu_id(mock_popen):
    """On a shared multi-GPU host, GPU0 can be busy -- ROCR_VISIBLE_DEVICES
    must be able to target a non-zero-indexed physical GPU, not always start
    from 0 (confirmed by a real E2E run that failed with 'No HIP GPUs are
    available' when GPU0-3 were busy and gpu_id wasn't threaded through)."""
    from quark.experimental.torch.quant_perf.landing.atom_adapter import start_atom_server

    mock_popen.return_value = MagicMock()
    start_atom_server(model_dir="/quant_ckpt", tp=2, port=9000, kv_cache_scheme="fp8", gpu_id=4)

    env = mock_popen.call_args.kwargs["env"]
    assert env["ROCR_VISIBLE_DEVICES"] == "4,5"


@patch("quark.experimental.torch.quant_perf.landing.atom_adapter.subprocess.Popen")
def test_start_atom_server_defaults_kv_cache_to_bf16(mock_popen):
    from quark.experimental.torch.quant_perf.landing.atom_adapter import start_atom_server

    mock_popen.return_value = MagicMock()
    start_atom_server(model_dir="/quant_ckpt", tp=1, port=9000, kv_cache_scheme=None)

    cmd = mock_popen.call_args.args[0]
    assert cmd[cmd.index("--kv_cache_dtype") + 1] == "bf16"


@patch("quark.experimental.torch.quant_perf.landing.base.requests")
@patch("quark.experimental.torch.quant_perf.landing.base.time")
def test_wait_ready_fails_fast_when_proc_already_exited(mock_time, mock_requests):
    """A server that crashes during startup (e.g. a missing dependency
    surfaced by a real ATOM run) must not be retried for the full timeout --
    proc.poll() returning non-None means it's dead, so give up immediately."""
    mock_time.time.side_effect = [0, 1]  # one loop iteration before "deadline"
    dead_proc = MagicMock()
    dead_proc.poll.return_value = 1  # exited with code 1

    assert wait_ready(9000, timeout_s=600, proc=dead_proc) is False
    mock_requests.get.assert_not_called()  # never bothers probing a dead server


@patch("quark.experimental.torch.quant_perf.landing.base.requests")
@patch("quark.experimental.torch.quant_perf.landing.base.time")
def test_wait_ready_retries_transient_v1_models_failure(mock_time, mock_requests):
    """A real E2E run showed /health can return 200 (the FastAPI app is up)
    before the EngineCore subprocess has registered the loaded model, so
    /v1/models transiently returns no data right after /health first
    succeeds. All three tiers must be retried together until timeout_s, not
    just tier 1 -- otherwise this transient state is reported as
    'never became ready' seconds before the server actually finishes."""
    mock_time.time.side_effect = [0, 1, 2, 3, 4, 5]

    health_resp = MagicMock(status_code=200)
    empty_models_resp = MagicMock()
    empty_models_resp.json.return_value = {"data": []}
    ready_models_resp = MagicMock()
    ready_models_resp.json.return_value = {"data": [{"id": "m1"}]}
    completion_resp = MagicMock(status_code=200)

    mock_requests.get.side_effect = [health_resp, empty_models_resp, health_resp, ready_models_resp]
    mock_requests.post.return_value = completion_resp

    assert wait_ready(9000, timeout_s=600) is True
    assert mock_requests.get.call_count == 4


@patch("quark.experimental.torch.quant_perf.landing.atom_adapter.subprocess.Popen")
def test_start_atom_server_only_sets_profiler_more_when_profiling(mock_popen):
    from quark.experimental.torch.quant_perf.landing.atom_adapter import start_atom_server

    mock_popen.return_value = MagicMock()
    start_atom_server(model_dir="/quant_ckpt", tp=1, port=9000, kv_cache_scheme="fp8")
    env = mock_popen.call_args.kwargs["env"]
    assert "ATOM_PROFILER_MORE" not in env

    start_atom_server(model_dir="/quant_ckpt", tp=1, port=9000, kv_cache_scheme="fp8", profiler_dir="/tmp/trace")
    env = mock_popen.call_args.kwargs["env"]
    assert env["ATOM_PROFILER_MORE"] == "1"
    cmd = mock_popen.call_args.args[0]
    assert "--torch-profiler-dir" in cmd
