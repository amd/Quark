#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import socket
from unittest.mock import MagicMock

import pytest

from quark.experimental.torch.quant_perf.evaluation.gsm8k import EvaluationFailure
from quark.experimental.torch.quant_perf.pipeline import AccuracyStage, LandingStage
from quark.experimental.torch.quant_perf.repair.source_router import resolve_repair_target
from quark.experimental.torch.quant_perf.session.spec import RuntimeContext, Spec, StageError


def _spec(tmp_path, **overrides) -> Spec:
    values = {
        "model_dir": "model",
        "base_model": "model",
        "framework": "vllm",
        "gpu_type": "mi355x",
        "gpu_arch": "MI355X",
        "isl": 1024,
        "osl": 1024,
        "quant_strategy": None,
        "framework_repo": "/source/vllm",
        "session_dir": str(tmp_path),
    }
    runtime_fields = set(RuntimeContext.__dataclass_fields__)
    runtime_overrides = {
        "framework_worktree": "/work/vllm",
        **{key: overrides.pop(key) for key in list(overrides) if key in runtime_fields},
    }
    values.update(overrides)
    return Spec(**values, runtime=RuntimeContext(**runtime_overrides))


def test_port_preflight_reports_structured_runtime_failure():
    from quark.experimental.torch.quant_perf.landing.base import ensure_port_available

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    try:
        with pytest.raises(StageError) as captured:
            ensure_port_available(port)
    finally:
        listener.close()

    assert captured.value.code == "address_in_use"
    assert "already in use" in captured.value.diagnostic


def test_port_preflight_allows_recently_closed_listener():
    from quark.experimental.torch.quant_perf.landing.base import ensure_port_available

    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    port = listener.getsockname()[1]
    client = socket.create_connection(("127.0.0.1", port))
    accepted, _ = listener.accept()
    accepted.close()
    listener.close()
    client.close()

    ensure_port_available(port)


def test_landing_stage_does_not_repair_runtime_failure(tmp_path):
    repair = MagicMock()
    loader = MagicMock(
        side_effect=StageError(
            "land",
            "server could not start",
            code="address_in_use",
            diagnostic="OSError: [Errno 98] Address already in use",
        )
    )

    with pytest.raises(StageError, match="server could not start"):
        LandingStage(repair, loader=loader).load_model_with_repair(
            _spec(tmp_path),
            "/quant",
        )

    repair.repair.assert_not_called()


def test_landing_stage_repairs_explicit_framework_failure(tmp_path):
    repair = MagicMock()
    repair.repair.return_value = MagicMock(status="fixed")
    server = MagicMock()
    loader = MagicMock(
        side_effect=[
            StageError(
                "land",
                "quantized model failed",
                code="tensor_shape_mismatch",
                diagnostic=("RuntimeError: size of tensor a must match tensor b"),
            ),
            server,
        ]
    )

    result = LandingStage(
        repair,
        loader=loader,
    ).load_model_with_repair(
        _spec(tmp_path),
        "/quant",
    )

    assert result is server
    request = repair.repair.call_args.args[0]
    assert request.error == ("RuntimeError: size of tensor a must match tensor b")


def test_landing_stage_routes_missing_aiter_backend_to_kernel_repair(tmp_path):
    handle = MagicMock()
    loader = MagicMock(
        side_effect=[
            StageError(
                "land",
                "load failed",
                diagnostic="W4A8 FlyDSL backend is unavailable in the installed AITER build.",
            ),
            handle,
        ]
    )
    repair = MagicMock()
    repair.repair.return_value.status = "fixed"

    result = LandingStage(
        repair,
        loader=loader,
    ).load_model_with_repair(
        _spec(
            tmp_path,
            framework_repo="",
            kernel_repo="/source/aiter",
            kernel_worktree="/managed/aiter",
        ),
        "/quant",
    )

    request = repair.repair.call_args.args[0]
    target = resolve_repair_target(request)
    assert result is handle
    assert target.role == "kernel"
    assert target.repo == "/managed/aiter"
    assert loader.call_count == 2


def test_accuracy_stage_retries_after_verified_repair(tmp_path):
    repair = MagicMock()
    repair.repair.return_value = MagicMock(status="fixed")
    expected = MagicMock(passed=True, gap=0.01)
    gate = MagicMock()
    gate.eval_quantized.side_effect = [
        RuntimeError("size of tensor a must match tensor b"),
        expected,
    ]

    result = AccuracyStage(repair).evaluate_quantized_checkpoint(
        _spec(tmp_path),
        gate,
        "/quant",
    )

    assert result is expected
    assert gate.eval_quantized.call_count == 2
    assert repair.repair.call_args.args[0].failure_class == "load_run"


def test_accuracy_stage_repairs_unknown_failure_from_managed_framework(
    tmp_path,
):
    repair = MagicMock()
    repair.repair.return_value = MagicMock(status="fixed")
    expected = MagicMock(passed=True, gap=0.01)
    gate = MagicMock()
    gate.eval_quantized.side_effect = [
        RuntimeError(
            'File "/work/vllm/vllm/model_executor/custom_loader.py", '
            "line 44, in load\n"
            '    raise RuntimeError("new loader failure")\n'
            "RuntimeError: new loader failure"
        ),
        expected,
    ]

    result = AccuracyStage(repair).evaluate_quantized_checkpoint(
        _spec(
            tmp_path,
            framework_worktree="/work/vllm",
            kernel_worktree="/work/aiter",
        ),
        gate,
        "/quant",
    )

    assert result is expected
    assert repair.repair.call_count == 1


@pytest.mark.parametrize("long_worker_logs", [False, True])
def test_accuracy_stage_applies_two_distinct_verified_repairs(tmp_path, long_worker_logs):
    repair = MagicMock()
    repair.repair.side_effect = [
        MagicMock(status="fixed"),
        MagicMock(status="fixed"),
    ]
    expected = MagicMock(passed=True, gap=0.01)
    gate = MagicMock()
    errors = [
        RuntimeError("size of tensor a must match tensor b"),
        RuntimeError("kernel launch failed: invalid device function"),
    ]
    if long_worker_logs:
        errors = [
            EvaluationFailure(
                "/quant",
                stderr=(
                    'File "/work/vllm/loader.py", line 1, in load\n'
                    f"KeyError: missing {name}\n" + "worker shutdown\n" * 2000
                ),
            )
            for name in ("weight_scale", "input_scale")
        ]
        assert str(errors[0]) == str(errors[1])
    gate.eval_quantized.side_effect = [*errors, expected]

    result = AccuracyStage(repair).evaluate_quantized_checkpoint(
        _spec(tmp_path),
        gate,
        "/quant",
    )

    assert result is expected
    assert gate.eval_quantized.call_count == 3
    assert repair.repair.call_count == 2


def test_accuracy_stage_surfaces_failed_repair(tmp_path):
    repair = MagicMock()
    repair.repair.return_value = MagicMock(status="not_fixed")
    gate = MagicMock()
    gate.eval_quantized.side_effect = RuntimeError("size of tensor a must match tensor b")

    with pytest.raises(StageError, match="tensor a must match"):
        AccuracyStage(repair).evaluate_quantized_checkpoint(
            _spec(tmp_path),
            gate,
            "/quant",
        )


def test_accuracy_stage_retries_contention_without_repair(tmp_path):
    repair = MagicMock()
    expected = MagicMock(passed=True, gap=0.01)
    gate = MagicMock()
    gate.eval_quantized.side_effect = [
        EvaluationFailure(
            "/quant",
            returncode=0,
            stdout=(
                "ValueError: Free memory on device cuda:2 "
                "(200.9/251.98 GiB) on startup is less than desired "
                "GPU memory utilization (0.8, 201.59 GiB)."
            ),
        ),
        expected,
    ]
    stage = AccuracyStage(repair)

    result = stage.evaluate_quantized_checkpoint(
        _spec(
            tmp_path,
            vllm_extra_args=["--gpu-memory-utilization=0.8"],
        ),
        gate,
        "/quant",
    )

    assert result is expected
    assert gate.eval_quantized.call_args_list[1].kwargs == {
        "gpu_memory_utilization": 0.79,
    }
    assert stage.last_recovery["code"] == "gpu_memory_occupied"
    repair.repair.assert_not_called()


def test_accuracy_stage_reclassifies_resource_retry_failure_for_repair(tmp_path):
    repair = MagicMock()
    repair.can_repair.return_value = True
    repair.repair.return_value = MagicMock(status="fixed")
    expected = MagicMock(passed=True, gap=0.01)
    gate = MagicMock()
    gate.eval_quantized.side_effect = [
        EvaluationFailure(
            "/quant",
            returncode=1,
            stdout=(
                "ValueError: Free memory on device cuda:7 "
                "(122.44/251.98 GiB) on startup is less than desired "
                "GPU memory utilization (0.60, 151.19 GiB)."
            ),
        ),
        RuntimeError(
            'File "/work/vllm/vllm/model_executor/models/deepseek_v2.py", '
            "line 1459, in load_weights\n"
            "KeyError: 'layers.0.self_attn.indexer.wk_weights_proj.input_scale'"
        ),
        expected,
    ]
    stage = AccuracyStage(repair)

    result = stage.evaluate_quantized_checkpoint(
        _spec(
            tmp_path,
            vllm_extra_args=["--gpu-memory-utilization=0.60"],
        ),
        gate,
        "/quant",
    )

    assert result is expected
    assert gate.eval_quantized.call_count == 3
    repair.repair.assert_called_once()
    request = repair.repair.call_args.args[0]
    assert request.failure_code == "managed_source_failure"
    assert "wk_weights_proj.input_scale" in request.error
    assert stage.last_recovery["outcome"] == "reclassified"


def test_accuracy_stage_bounds_repeated_resource_contention(tmp_path):
    repair = MagicMock()
    gate = MagicMock()
    gate.eval_quantized.side_effect = EvaluationFailure(
        "/quant",
        returncode=1,
        stdout=(
            "ValueError: Free memory on device cuda:7 "
            "(122.44/251.98 GiB) on startup is less than desired "
            "GPU memory utilization (0.60, 151.19 GiB)."
        ),
    )

    with pytest.raises(StageError, match="resource retry failed"):
        AccuracyStage(repair).evaluate_quantized_checkpoint(
            _spec(
                tmp_path,
                vllm_extra_args=["--gpu-memory-utilization=0.60"],
            ),
            gate,
            "/quant",
        )

    assert gate.eval_quantized.call_count == 2
    repair.repair.assert_not_called()


def test_accuracy_stage_does_not_repair_capacity_failure(tmp_path):
    repair = MagicMock()
    gate = MagicMock()
    gate.eval_quantized.side_effect = RuntimeError("GPU out of memory")

    with pytest.raises(StageError, match="GPU out of memory"):
        AccuracyStage(repair).evaluate_quantized_checkpoint(
            _spec(tmp_path),
            gate,
            "/quant",
        )

    repair.repair.assert_not_called()
