#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for Quark Quant-Perf's offline GSM8K and throughput helpers."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quark.experimental.torch.quant_perf.evaluation import execution, gsm8k, throughput
from quark.experimental.torch.quant_perf.runtime import backends as runtime_backends
from quark.experimental.torch.quant_perf.session.spec import EvalProfile


def _profile(**overrides):
    fields = dict(
        profile_id="gsm8k-chat-nothink-v1",
        profile_hash="hash",
        model_mode="chat",
        apply_chat_template=True,
        enable_thinking=False,
        detection_reason="chat_template",
    )
    fields.update(overrides)
    return EvalProfile(**fields)


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


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        ("fp4", True),
        ("mxfp4", True),
        ("fp8_e4m3fn", False),
    ],
)
def test_is_mxfp4_model_detects_weight_dtype(tmp_path, dtype, expected):
    cfg = {"quantization_config": {"global_quant_config": {"weight": {"dtype": dtype}}}}
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert runtime_backends.is_mxfp4_model(str(tmp_path)) is expected


def test_is_mxfp4_model_returns_false_without_config(tmp_path):
    assert runtime_backends.is_mxfp4_model(str(tmp_path)) is False
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "llama"}))
    assert runtime_backends.is_mxfp4_model(str(tmp_path)) is False


def test_is_w4a8_mxfp4_model_detects_static_fp8_activation(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "global_quant_config": {
                        "weight": {"dtype": "fp4"},
                        "input_tensors": {
                            "dtype": "fp8_e4m3",
                            "qscheme": "per_tensor",
                            "is_dynamic": False,
                        },
                    }
                }
            }
        )
    )

    assert runtime_backends.is_w4a8_mxfp4_model(str(tmp_path)) is True


def test_aiter_w4a4_moe_uses_safe_single_k_runtime(tmp_path):
    _write_w4a4_moe_config(tmp_path)

    env = {"AITER_KSPLIT": "4"}
    runtime_backends.configure_aiter_mxfp4_moe_ksplit(env, str(tmp_path), "aiter")
    assert env["AITER_KSPLIT"] == "1"

    env = {}
    runtime_backends.configure_aiter_mxfp4_moe_ksplit(env, str(tmp_path), "flydsl")
    assert "AITER_KSPLIT" not in env

    config = json.loads((tmp_path / "config.json").read_text())
    config["quantization_config"]["global_quant_config"]["input_tensors"]["dtype"] = "fp8_e4m3"
    (tmp_path / "config.json").write_text(json.dumps(config))
    env = {}
    runtime_backends.configure_aiter_mxfp4_moe_ksplit(env, str(tmp_path), "aiter")
    assert "AITER_KSPLIT" not in env

    config["model_type"] = "qwen3"
    config.pop("num_experts")
    config["quantization_config"]["global_quant_config"]["input_tensors"]["dtype"] = "fp4"
    (tmp_path / "config.json").write_text(json.dumps(config))
    env = {}
    runtime_backends.configure_aiter_mxfp4_moe_ksplit(env, str(tmp_path), "aiter")
    assert "AITER_KSPLIT" not in env


def test_mxfp4_moe_detection_does_not_classify_dense_w4a8(tmp_path):
    quantization_config = {
        "global_quant_config": {
            "weight": {"dtype": "fp4"},
            "input_tensors": {"dtype": "fp8_e4m3"},
        }
    }
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3",
                "architectures": ["Qwen3ForCausalLM"],
                "quantization_config": quantization_config,
            }
        )
    )
    assert runtime_backends.is_mxfp4_moe_model(str(tmp_path)) is False

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_moe",
                "num_experts": 128,
                "quantization_config": quantization_config,
            }
        )
    )
    assert runtime_backends.is_mxfp4_moe_model(str(tmp_path)) is True


def test_dense_asm_backend_preserves_aiter_for_non_moe_model():
    env = {
        "QUARK_QUANT_PERF_MXFP4_GEMM_BACKEND": "asm",
        "VLLM_ROCM_MXFP4_GEMM_BACKEND": "triton",
        "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM": "1",
    }

    backend = runtime_backends.configure_mxfp4_runtime_env(
        env,
        enable_aiter_moe=False,
        select_mxfp4_moe_backend=False,
    )

    assert backend == ""
    assert env["VLLM_ROCM_USE_AITER"] == "1"
    assert env["VLLM_ROCM_USE_AITER_FP4_ASM_GEMM"] == "1"


@pytest.mark.parametrize(
    ("task", "score"),
    [
        ("gsm8k", 0.83),
        ("gsm8k_cot_zeroshot", 0.75),
    ],
)
def test_read_gsm8k_score_accepts_task_variants(
    tmp_path,
    task,
    score,
):
    data = {"results": {task: {"exact_match,flexible-extract": score}}}
    (tmp_path / "results.json").write_text(json.dumps(data))
    assert gsm8k.read_gsm8k_score(tmp_path) == pytest.approx(score)


def test_read_gsm8k_score_uses_newest_result_file(tmp_path):
    older = tmp_path / "older" / "results.json"
    newer = tmp_path / "newer" / "results.json"
    older.parent.mkdir()
    newer.parent.mkdir()
    older.write_text(json.dumps({"results": {"gsm8k": {"exact_match,flexible-extract": 0.1}}}))
    newer.write_text(json.dumps({"results": {"gsm8k": {"exact_match,flexible-extract": 0.9}}}))
    os.utime(older, (1, 1))
    os.utime(newer, (2, 2))

    with patch.object(Path, "rglob", return_value=[older, newer]):
        assert gsm8k.read_gsm8k_score(tmp_path) == pytest.approx(0.9)


def test_read_gsm8k_score_rejects_missing_results(tmp_path):
    with pytest.raises(RuntimeError, match="no results"):
        gsm8k.read_gsm8k_score(tmp_path)


def test_read_gsm8k_score_rejects_non_gsm8k_results(tmp_path):
    data = {"results": {"other_task": {"exact_match,flexible-extract": 0.5}}}
    (tmp_path / "results.json").write_text(json.dumps(data))
    with pytest.raises(RuntimeError, match="no gsm8k task"):
        gsm8k.read_gsm8k_score(tmp_path)


def test_run_isolated_subprocess_terminates_process_group_on_timeout():
    process = MagicMock()
    process.pid = 1234
    process.communicate.side_effect = [
        subprocess.TimeoutExpired(["python"], 1),
        ("complete stdout", "complete stderr"),
    ]

    with (
        patch.object(execution.subprocess, "Popen", return_value=process) as popen,
        patch.object(execution.os, "killpg") as killpg,
        pytest.raises(subprocess.TimeoutExpired) as exc_info,
    ):
        execution.run_isolated_subprocess(
            ["python", "-c", "pass"],
            timeout=1,
            env={"KEY": "value"},
        )

    assert popen.call_args.kwargs["start_new_session"] is True
    killpg.assert_called_once_with(process.pid, signal.SIGTERM)
    assert exc_info.value.stdout == "complete stdout"
    assert exc_info.value.stderr == "complete stderr"


def test_run_isolated_subprocess_can_inherit_output_streams():
    process = MagicMock()
    process.returncode = 0
    process.communicate.return_value = (None, None)

    with patch.object(execution.subprocess, "Popen", return_value=process) as popen:
        result = execution.run_isolated_subprocess(
            ["python", "-c", "pass"],
            timeout=1,
            capture_output=False,
        )

    assert popen.call_args.kwargs["stdout"] is None
    assert popen.call_args.kwargs["stderr"] is None
    assert result.stdout == ""
    assert result.stderr == ""


def test_remove_stale_aiter_jit_locks_is_scoped_to_lock_files(tmp_path):
    jit_dir = tmp_path / "aiter_jit"
    outer_lock = jit_dir / "build" / "lock_module_quant"
    inner_lock = jit_dir / "build" / "module_quant" / "build" / "lock"
    compiled_object = jit_dir / "build" / "module_quant" / "build" / "kernel.o"
    outer_lock.parent.mkdir(parents=True)
    inner_lock.parent.mkdir(parents=True)
    outer_lock.touch()
    inner_lock.touch()
    compiled_object.write_bytes(b"object")

    removed = execution.remove_stale_aiter_jit_locks({"AITER_JIT_DIR": str(jit_dir)})

    assert removed == [str(outer_lock), str(inner_lock)]
    assert not outer_lock.exists()
    assert not inner_lock.exists()
    assert compiled_object.read_bytes() == b"object"


def test_model_startup_timeout_scales_with_checkpoint_size(tmp_path):
    shard = tmp_path / "model-00001-of-00001.safetensors"
    with shard.open("wb") as handle:
        handle.truncate(700 * 1024**3)

    assert (
        execution.model_startup_timeout_s(
            str(tmp_path),
            minimum_s=600,
        )
        == 3100
    )


def test_gsm8k_passes_explicit_runtime_settings(tmp_path):
    _write_w4a4_moe_config(tmp_path)
    with (
        patch(
            "quark.experimental.torch.quant_perf.evaluation.gsm8k.is_mxfp4_moe_model",
            return_value=True,
        ),
        patch("quark.experimental.torch.quant_perf.runtime.backends.mxfp4_moe_backend", return_value="aiter"),
        patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.run_isolated_subprocess") as mock_run,
        patch(
            "quark.experimental.torch.quant_perf.evaluation.gsm8k.read_gsm8k_score",
            return_value=0.8,
        ),
    ):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        gsm8k.gsm8k_eval_offline(
            str(tmp_path),
            num_questions=1,
            profile=_profile(),
            moe_backend="triton",
            kv_cache_dtype="fp8",
        )

    env = mock_run.call_args.kwargs["env"]
    argv = mock_run.call_args.args[0]
    model_args = argv[argv.index("--model_args") + 1]
    assert "moe_backend=triton" in model_args
    assert "kv_cache_dtype=fp8" in model_args
    assert "AITER_KSPLIT" not in env


def test_gsm8k_aiter_backend_uses_safe_single_k_with_cuda_graphs(tmp_path):
    _write_w4a4_moe_config(tmp_path)
    with (
        patch.dict(os.environ, {"VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE": "1"}, clear=False),
        patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.is_mxfp4_moe_model", return_value=True),
    ):
        with patch("quark.experimental.torch.quant_perf.runtime.backends.mxfp4_moe_backend", return_value="aiter"):
            with patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.run_isolated_subprocess") as mock_run:
                mock_run.return_value.returncode = 0
                with patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.read_gsm8k_score", return_value=0.8):
                    gsm8k.gsm8k_eval_offline(
                        str(tmp_path),
                        num_questions=1,
                        profile=_profile(),
                    )

    env = mock_run.call_args.kwargs["env"]
    argv = mock_run.call_args.args[0]
    model_args = argv[argv.index("--model_args") + 1]
    assert "VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE" not in env
    assert "enforce_eager=True" not in model_args
    assert "compilation_config=" not in model_args
    assert env["AITER_KSPLIT"] == "1"


@pytest.mark.parametrize("model_type", ["glm_moe_dsa", "deepseek_v4"])
@pytest.mark.parametrize(
    ("backend", "disable_aiter_moe", "moe_enabled"),
    [("", False, "1"), ("", True, "0"), ("triton", False, "0"), ("triton", True, "0"), ("aiter", True, "1")],
)
def test_gsm8k_preserves_aiter_for_sparse_attention_with_moe_override(
    tmp_path, model_type, backend, disable_aiter_moe, moe_enabled
):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "quantization_config": {
                    "quant_method": "fp8",
                    "weight_block_size": [128, 128],
                },
            }
        )
    )
    runtime_env = {
        "VLLM_ROCM_USE_AITER": "1",
    }
    if disable_aiter_moe:
        runtime_env["VLLM_ROCM_USE_AITER_MOE"] = "0"
    with (
        patch("quark.experimental.torch.quant_perf.runtime.backends.mxfp4_moe_backend", return_value="flydsl"),
        patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.run_isolated_subprocess") as mock_run,
        patch(
            "quark.experimental.torch.quant_perf.evaluation.gsm8k.read_gsm8k_score",
            return_value=0.8,
        ),
    ):
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""

        gsm8k.gsm8k_eval_offline(
            str(tmp_path),
            num_questions=1,
            profile=_profile(),
            runtime_env=runtime_env,
            moe_backend=backend,
        )

    env = mock_run.call_args.kwargs["env"]
    argv = mock_run.call_args.args[0]
    model_args = argv[argv.index("--model_args") + 1]
    assert env["VLLM_ROCM_USE_AITER"] == "1"
    assert env["VLLM_ROCM_USE_AITER_MOE"] == moe_enabled
    assert "VLLM_ROCM_USE_AITER_FLYDSL_MOE" not in env
    assert "AITER_FLYDSL_FORCE" not in env
    assert "AITER_KSPLIT" not in env
    assert "enable_prefix_caching=False" not in model_args
    assert "enforce_eager=True" not in model_args
    assert "compilation_config=" not in model_args


@pytest.mark.parametrize("tp", [1, 8])
@pytest.mark.parametrize("trust_remote_code", [False, True])
def test_gsm8k_uses_native_backend_and_session_runtime(tp, trust_remote_code):
    with (
        patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.is_mxfp4_moe_model", return_value=False),
        patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.run_isolated_subprocess") as mock_run,
        patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.read_gsm8k_score", return_value=0.8),
    ):
        mock_run.return_value.returncode = 0
        gsm8k.gsm8k_eval_offline(
            "model",
            num_questions=1,
            tp=tp,
            profile=_profile(),
            runtime_python="/session/venv/bin/python",
            runtime_env={"PYTHONPATH": "/session/overlay"},
            trust_remote_code=trust_remote_code,
        )

    cmd = mock_run.call_args.args[0]
    env = mock_run.call_args.kwargs["env"]
    launcher = "quark.experimental.torch.quant_perf.evaluation.lm_eval_launcher" if trust_remote_code else "lm_eval"
    assert cmd[:5] == ["/session/venv/bin/python", "-m", launcher, "--model", "vllm"]
    executor = "PreparedUniProcExecutor" if tp == 1 else "PreparedMultiprocExecutor"
    model_args = cmd[cmd.index("--model_args") + 1]
    assert f"evaluation.vllm_executor.{executor}" in model_args
    assert ("trust_remote_code=True" in model_args) is trust_remote_code
    assert env["PYTHONPATH"] == "/session/overlay"


def test_throughput_aiter_backend_uses_safe_single_k_with_cuda_graphs(tmp_path):
    _write_w4a4_moe_config(tmp_path)
    with (
        patch.dict(os.environ, {"VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE": "1"}, clear=False),
        patch("quark.experimental.torch.quant_perf.evaluation.throughput.is_mxfp4_moe_model", return_value=True),
    ):
        with patch("quark.experimental.torch.quant_perf.runtime.backends.mxfp4_moe_backend", return_value="aiter"):
            with patch("quark.experimental.torch.quant_perf.evaluation.throughput.run_isolated_subprocess") as mock_run:
                mock_run.return_value.stdout = "THROUGHPUT=123.0\n"
                mock_run.return_value.stderr = ""
                throughput.throughput_benchmark(str(tmp_path), isl=1, osl=1, num_prompts=1, concurrency=1)

    env = mock_run.call_args.kwargs["env"]
    script = mock_run.call_args.args[0][2]
    assert "VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE" not in env
    assert "enforce_eager=True" not in script
    assert "compilation_config=" not in script
    assert env["AITER_KSPLIT"] == "1"


@pytest.mark.parametrize("model_type", ["glm_moe_dsa", "deepseek_v4"])
def test_throughput_preserves_aiter_for_fp8_sparse_attention(tmp_path, model_type):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "quantization_config": {
                    "quant_method": "fp8",
                },
            }
        )
    )
    with (
        patch.dict(os.environ, {}, clear=True),
        patch("quark.experimental.torch.quant_perf.runtime.backends.mxfp4_moe_backend", return_value="flydsl"),
        patch("quark.experimental.torch.quant_perf.evaluation.throughput.run_isolated_subprocess") as mock_run,
    ):
        mock_run.return_value.stdout = "THROUGHPUT=123.0\n"
        mock_run.return_value.stderr = ""

        throughput.throughput_benchmark(
            str(tmp_path),
            isl=1,
            osl=1,
            num_prompts=1,
            concurrency=1,
        )

    env = mock_run.call_args.kwargs["env"]
    script = mock_run.call_args.args[0][2]
    assert env["VLLM_ROCM_USE_AITER"] == "1"
    assert env["VLLM_ROCM_USE_AITER_MOE"] == "1"
    assert "VLLM_ROCM_USE_AITER_FLYDSL_MOE" not in env
    assert "AITER_FLYDSL_FORCE" not in env
    assert "AITER_KSPLIT" not in env
    assert "moe_backend='aiter'" not in script
    assert "compilation_config=" not in script


def test_throughput_uses_session_runtime_python_and_env():
    with (
        patch("quark.experimental.torch.quant_perf.evaluation.throughput.is_mxfp4_moe_model", return_value=False),
        patch("quark.experimental.torch.quant_perf.evaluation.throughput.run_isolated_subprocess") as mock_run,
    ):
        mock_run.return_value.stdout = "THROUGHPUT=123.0\n"
        mock_run.return_value.stderr = ""
        throughput.throughput_benchmark(
            "model",
            isl=1,
            osl=1,
            num_prompts=1,
            concurrency=1,
            runtime_python="/session/venv/bin/python",
            runtime_env={"PYTHONPATH": "/session/overlay"},
        )

    cmd = mock_run.call_args.args[0]
    env = mock_run.call_args.kwargs["env"]
    assert cmd[:2] == ["/session/venv/bin/python", "-c"]
    assert env["PYTHONPATH"] == "/session/overlay"


def test_gsm8k_flydsl_backend_uses_tuned_aiter_with_cuda_graphs():
    with (
        patch.dict(
            os.environ,
            {
                "AITER_CONFIG_FMOE": "/configs/qwen35.csv",
                "VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE": "1",
            },
            clear=False,
        ),
        patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.is_mxfp4_moe_model", return_value=True),
    ):
        with patch("quark.experimental.torch.quant_perf.runtime.backends.mxfp4_moe_backend", return_value="flydsl"):
            with patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.run_isolated_subprocess") as mock_run:
                mock_run.return_value.returncode = 0
                with patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.read_gsm8k_score", return_value=0.8):
                    gsm8k.gsm8k_eval_offline(
                        "model",
                        num_questions=1,
                        profile=_profile(),
                    )

    env = mock_run.call_args.kwargs["env"]
    cmd = mock_run.call_args.args[0]
    model_args = cmd[cmd.index("--model_args") + 1]
    assert env["AITER_CONFIG_FMOE"] == "/configs/qwen35.csv"
    assert env["AITER_FLYDSL_FORCE"] == "1"
    assert env["VLLM_ROCM_USE_AITER"] == "1"
    assert env["VLLM_ROCM_USE_AITER_MOE"] == "1"
    assert env["VLLM_ROCM_USE_AITER_FLYDSL_MOE"] == "1"
    assert "VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE" not in env
    assert "moe_backend=aiter" in model_args
    assert "enable_prefix_caching=False" in model_args
    assert "enforce_eager=True" not in model_args


def test_vllm_cache_root_isolated_by_checkpoint_quant_config(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "quantization_config": {"layer_quant_config": {"self_attn": {"scheme": "mxfp4"}}},
            }
        )
    )
    (second / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "quantization_config": {"layer_quant_config": {"self_attn": {"scheme": "mxfp4_fp8"}}},
            }
        )
    )

    first_env = {}
    second_env = {}
    execution.configure_vllm_cache_env(first_env, str(first))
    execution.configure_vllm_cache_env(second_env, str(second))

    assert first_env["VLLM_CACHE_ROOT"] != second_env["VLLM_CACHE_ROOT"]
    assert first_env["VLLM_CACHE_ROOT"].startswith("/tmp/quark_quant_perf_vllm_cache/")
    assert second_env["VLLM_CACHE_ROOT"].startswith("/tmp/quark_quant_perf_vllm_cache/")


def test_accuracy_and_throughput_share_cache_for_same_checkpoint(tmp_path):
    model = tmp_path / "quant_ckpt"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5_moe",
                "quantization_config": {"layer_quant_config": {"self_attn": {"scheme": "mxfp4_fp8"}}},
            }
        )
    )
    cache_roots = []

    def run(*_args, **kwargs):
        cache_roots.append(kwargs["env"]["VLLM_CACHE_ROOT"])
        return subprocess.CompletedProcess(
            ["test"],
            returncode=0,
            stdout="THROUGHPUT=123.0\n",
            stderr="",
        )

    with (
        patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.is_mxfp4_moe_model", return_value=False),
        patch(
            "quark.experimental.torch.quant_perf.evaluation.throughput.is_mxfp4_moe_model",
            return_value=False,
        ),
        patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.run_isolated_subprocess", side_effect=run),
        patch("quark.experimental.torch.quant_perf.evaluation.throughput.run_isolated_subprocess", side_effect=run),
        patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.read_gsm8k_score", return_value=0.8),
    ):
        gsm8k.gsm8k_eval_offline(
            str(model),
            num_questions=1,
            profile=_profile(),
        )
        throughput.throughput_benchmark(
            str(model),
            isl=1,
            osl=1,
            num_prompts=1,
            concurrency=1,
        )

    assert cache_roots[0] == cache_roots[1]


def test_throughput_flydsl_backend_uses_tuned_aiter_with_cuda_graphs():
    with (
        patch.dict(
            os.environ,
            {"AITER_CONFIG_FMOE": "/configs/qwen35.csv"},
            clear=False,
        ),
        patch("quark.experimental.torch.quant_perf.evaluation.throughput.is_mxfp4_moe_model", return_value=True),
    ):
        with patch("quark.experimental.torch.quant_perf.runtime.backends.mxfp4_moe_backend", return_value="flydsl"):
            with patch("quark.experimental.torch.quant_perf.evaluation.throughput.run_isolated_subprocess") as mock_run:
                mock_run.return_value.stdout = "THROUGHPUT=123.0\n"
                mock_run.return_value.stderr = ""
                throughput.throughput_benchmark("model", isl=1, osl=1, num_prompts=1, concurrency=1)

    env = mock_run.call_args.kwargs["env"]
    script = mock_run.call_args.args[0][2]
    assert env["AITER_CONFIG_FMOE"] == "/configs/qwen35.csv"
    assert env["AITER_FLYDSL_FORCE"] == "1"
    assert env["VLLM_ROCM_USE_AITER_FLYDSL_MOE"] == "1"
    assert "moe_backend='aiter'" in script
    assert "enforce_eager=True" not in script


def test_throughput_failure_preserves_stdout_root_cause():
    with patch("quark.experimental.torch.quant_perf.evaluation.throughput.is_mxfp4_moe_model", return_value=False):
        with patch("quark.experimental.torch.quant_perf.evaluation.throughput.run_isolated_subprocess") as mock_run:
            mock_run.return_value.stdout = "worker rank root cause from stdout\n"
            mock_run.return_value.stderr = "engine wrapper failure from stderr\n"

            with pytest.raises(RuntimeError) as error:
                throughput.throughput_benchmark("model", isl=1, osl=1, num_prompts=1, concurrency=1)

    message = str(error.value)
    assert "worker rank root cause from stdout" in message
    assert "engine wrapper failure from stderr" in message


def test_throughput_uses_requested_gpu_memory_utilization():
    with patch("quark.experimental.torch.quant_perf.evaluation.throughput.is_mxfp4_moe_model", return_value=False):
        with patch("quark.experimental.torch.quant_perf.evaluation.throughput.run_isolated_subprocess") as mock_run:
            mock_run.return_value.stdout = "THROUGHPUT=123.0\n"
            mock_run.return_value.stderr = ""

            throughput.throughput_benchmark(
                "model",
                isl=1,
                osl=1,
                num_prompts=1,
                concurrency=1,
                gpu_memory_utilization=0.75,
            )

    script = mock_run.call_args.args[0][2]
    assert "gpu_memory_utilization=0.75" in script


def test_throughput_passes_explicit_runtime_settings(tmp_path):
    _write_w4a4_moe_config(tmp_path)
    with (
        patch(
            "quark.experimental.torch.quant_perf.evaluation.throughput.is_mxfp4_moe_model",
            return_value=True,
        ),
        patch("quark.experimental.torch.quant_perf.runtime.backends.mxfp4_moe_backend", return_value="aiter"),
        patch("quark.experimental.torch.quant_perf.evaluation.throughput.run_isolated_subprocess") as mock_run,
    ):
        mock_run.return_value.stdout = "THROUGHPUT=123.0\n"
        mock_run.return_value.stderr = ""

        throughput.throughput_benchmark(
            str(tmp_path),
            isl=1,
            osl=1,
            num_prompts=1,
            concurrency=1,
            moe_backend="triton",
            kv_cache_dtype="fp8",
            trust_remote_code=True,
            max_num_seqs=8,
        )

    env = mock_run.call_args.kwargs["env"]
    script = mock_run.call_args.args[0][2]
    assert "moe_backend='triton'" in script
    assert "kv_cache_dtype='fp8'" in script
    assert "trust_remote_code=True" in script
    assert "max_num_seqs=8" in script
    assert "AITER_KSPLIT" not in env


def test_throughput_timeout_raises_structured_failure():
    timeout = subprocess.TimeoutExpired(
        cmd=["python3"],
        timeout=3600,
        output="partial stdout",
        stderr="partial stderr",
    )
    with (
        patch("quark.experimental.torch.quant_perf.evaluation.throughput.is_mxfp4_moe_model", return_value=False),
        patch(
            "quark.experimental.torch.quant_perf.evaluation.throughput.run_isolated_subprocess",
            side_effect=timeout,
        ) as mock_run,
        pytest.raises(throughput.BenchmarkFailure) as error,
    ):
        throughput.throughput_benchmark("model", isl=1, osl=1, num_prompts=1, concurrency=1)

    assert error.value.timed_out is True
    assert error.value.stdout == "partial stdout"
    assert error.value.stderr == "partial stderr"
    assert mock_run.call_args.kwargs["timeout"] == 3600
    assert mock_run.call_args.kwargs["capture_output"] is True


def test_measure_throughput_returns_robust_sample_summary():
    payload = {
        "samples_tps": [100.0, 102.0, 101.0],
        "warmup_tps": 99.0,
        "stable": True,
    }
    with patch("quark.experimental.torch.quant_perf.evaluation.throughput.is_mxfp4_moe_model", return_value=False):
        with patch("quark.experimental.torch.quant_perf.evaluation.throughput.run_isolated_subprocess") as mock_run:
            mock_run.return_value.stdout = "THROUGHPUT_RESULT=" + json.dumps(payload) + "\n"
            mock_run.return_value.stderr = ""

            measurement = throughput.measure_throughput("model", isl=1, osl=1, num_prompts=1, concurrency=1)
            assert throughput.throughput_benchmark(
                "model", isl=1, osl=1, num_prompts=1, concurrency=1
            ) == pytest.approx(101.0)

    assert measurement.samples_tps == (100.0, 102.0, 101.0)
    assert measurement.median_tps == pytest.approx(101.0)
    assert measurement.mad_tps == pytest.approx(1.0)
    assert measurement.relative_mad == pytest.approx(1.0 / 101.0)
    assert measurement.warmup_tps == pytest.approx(99.0)
    assert measurement.stable is True


def test_gsm8k_failure_preserves_process_output_and_graph_error():
    failure = subprocess.CompletedProcess(
        ["lm_eval"],
        returncode=1,
        stdout="worker startup failed",
        stderr="hipErrorStreamCaptureInvalidated during capture",
    )
    dependency_probe = subprocess.CompletedProcess(
        ["/session/venv/bin/python", "-c", "import vllm"],
        returncode=0,
        stdout="",
        stderr="",
    )
    with patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.is_mxfp4_moe_model", return_value=False):
        with patch(
            "quark.experimental.torch.quant_perf.evaluation.gsm8k.run_isolated_subprocess", return_value=failure
        ):
            with (
                patch(
                    "quark.experimental.torch.quant_perf.evaluation.gsm8k.subprocess.run",
                    return_value=dependency_probe,
                ) as mock_probe,
                pytest.raises(gsm8k.EvaluationFailure) as error,
            ):
                gsm8k.gsm8k_eval_offline(
                    "model",
                    num_questions=1,
                    profile=_profile(),
                    runtime_python="/session/venv/bin/python",
                )

    assert error.value.returncode == 1
    assert "worker startup failed" in error.value.stdout
    assert "hipErrorStreamCaptureInvalidated" in error.value.stderr
    assert mock_probe.call_args.args[0][:2] == [
        "/session/venv/bin/python",
        "-c",
    ]


def test_gsm8k_missing_result_after_zero_exit_is_structured_failure():
    completed = subprocess.CompletedProcess(
        ["lm_eval"],
        returncode=0,
        stdout="evaluation ended without result artifact",
        stderr="hipErrorStreamCaptureInvalidated",
    )
    with patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.is_mxfp4_moe_model", return_value=False):
        with patch(
            "quark.experimental.torch.quant_perf.evaluation.gsm8k.run_isolated_subprocess", return_value=completed
        ):
            with patch(
                "quark.experimental.torch.quant_perf.evaluation.gsm8k.read_gsm8k_score",
                side_effect=RuntimeError("lm_eval produced no results*.json"),
            ):
                with pytest.raises(gsm8k.EvaluationFailure) as error:
                    gsm8k.gsm8k_eval_offline(
                        "model",
                        num_questions=1,
                        profile=_profile(),
                    )

    assert error.value.returncode == 0
    assert "evaluation ended without result artifact" in error.value.stdout
    assert "hipErrorStreamCaptureInvalidated" in error.value.stderr


def test_gsm8k_command_uses_chat_thinking_profile():
    with patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.is_mxfp4_moe_model", return_value=False):
        with patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.run_isolated_subprocess") as mock_run:
            mock_run.return_value.returncode = 0
            with patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.read_gsm8k_score", return_value=0.8):
                gsm8k.gsm8k_eval_offline(
                    "model",
                    num_questions=1,
                    profile=_profile(),
                )

    cmd = mock_run.call_args.args[0]
    model_args = cmd[cmd.index("--model_args") + 1]
    assert "--apply_chat_template" in cmd
    assert "enable_thinking=False" in model_args
    assert "max_model_len=8192" in model_args
    assert cmd[cmd.index("--tasks") + 1] == "gsm8k_cot_zeroshot"


def test_gsm8k_command_omits_chat_settings_for_base_profile():
    with patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.is_mxfp4_moe_model", return_value=False):
        with patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.run_isolated_subprocess") as mock_run:
            mock_run.return_value.returncode = 0
            with patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.read_gsm8k_score", return_value=0.8):
                gsm8k.gsm8k_eval_offline(
                    "model",
                    num_questions=1,
                    profile=_profile(
                        profile_id="gsm8k-base-default-v1",
                        model_mode="base",
                        apply_chat_template=False,
                        enable_thinking=None,
                        max_model_len=4096,
                    ),
                )

    cmd = mock_run.call_args.args[0]
    model_args = cmd[cmd.index("--model_args") + 1]
    assert "--apply_chat_template" not in cmd
    assert "enable_thinking" not in model_args
    assert "max_model_len=4096" in model_args


def test_gsm8k_persists_profile_command_and_process_logs(tmp_path):
    completed = subprocess.CompletedProcess(
        ["lm_eval"],
        returncode=0,
        stdout="eval stdout",
        stderr="eval stderr",
    )
    output_dir = tmp_path / "evaluation"
    profile = _profile(
        schema_version=2,
        policy_version="quark-quant-perf-gsm8k-profile-v2",
        task="gsm8k",
        num_fewshot=5,
        prompting_strategy="cot",
        gen_kwargs={"temperature": 0, "top_p": 1},
        settings_source="quark_policy_default",
        source_reference="quark-quant-perf-gsm8k-v2",
    ).with_computed_hash()

    with (
        patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.is_mxfp4_moe_model", return_value=False),
        patch(
            "quark.experimental.torch.quant_perf.evaluation.gsm8k.run_isolated_subprocess",
            return_value=completed,
        ) as mock_run,
        patch(
            "quark.experimental.torch.quant_perf.evaluation.gsm8k.read_gsm8k_score",
            return_value=0.8,
        ),
    ):
        score = gsm8k.gsm8k_eval_offline(
            "model",
            num_questions=20,
            profile=profile,
            output_dir=output_dir,
        )

    assert score == 0.8
    cmd = mock_run.call_args.args[0]
    assert cmd[cmd.index("--num_fewshot") + 1] == "5"
    assert cmd[cmd.index("--gen_kwargs") + 1] == "temperature=0,top_p=1"
    assert "--log_samples" in cmd
    assert cmd[cmd.index("--output_path") + 1] == str(output_dir)
    assert mock_run.call_args.kwargs["timeout"] == 7200
    assert mock_run.call_args.kwargs["preparation_timeout"] == 2400
    assert mock_run.call_args.kwargs["preparation_status_path"] == output_dir / "preparation.json"
    assert mock_run.call_args.kwargs["capture_output"] is True
    assert json.loads((output_dir / "command.json").read_text())["argv"] == cmd
    assert json.loads((output_dir / "profile.json").read_text())["profile_hash"] == profile.profile_hash
    assert (output_dir / "stdout.log").read_text() == "eval stdout"
    assert (output_dir / "stderr.log").read_text() == "eval stderr"
