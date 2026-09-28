#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for the optional deterministic vendor GEMM tuning backend."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_forge_model_view_normalizes_optional_null_without_mutating_source(
    tmp_path,
):
    from quark.experimental.torch.quant_perf.perfopt import vendor_gemm

    model = tmp_path / "quant_ckpt"
    model.mkdir()
    config_path = model / "config.json"
    original = json.dumps(
        {
            "architectures": ["Gemma4ForConditionalGeneration"],
            "text_config": {
                "hidden_size": 5376,
                "intermediate_size": 21504,
                "moe_intermediate_size": None,
                "num_experts": None,
            },
        },
        sort_keys=True,
    ).encode()
    config_path.write_bytes(original)
    weights = model / "model.safetensors"
    weights.write_bytes(b"weights")

    with vendor_gemm._forge_model_view(
        model,
        tmp_path / "vendor_gemm",
    ) as (forge_model, metadata):
        overlay = Path(forge_model)
        assert overlay != model
        normalized = json.loads((overlay / "config.json").read_text())
        assert normalized["text_config"]["moe_intermediate_size"] == 0
        assert normalized["text_config"]["num_experts"] == 0
        assert (overlay / "model.safetensors").is_symlink()
        assert (overlay / "model.safetensors").resolve() == weights
        assert config_path.read_bytes() == original
        assert metadata["used"] is True
        assert metadata["normalized_fields"] == [
            "text_config.moe_intermediate_size",
            "text_config.num_experts",
        ]

    assert not overlay.exists()
    assert config_path.read_bytes() == original


def test_forge_model_view_reuses_compatible_checkpoint(tmp_path):
    from quark.experimental.torch.quant_perf.perfopt import vendor_gemm

    model = tmp_path / "quant_ckpt"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "hidden_size": 5376,
                    "intermediate_size": 21504,
                    "moe_intermediate_size": 0,
                }
            }
        )
    )

    with vendor_gemm._forge_model_view(
        model,
        tmp_path / "vendor_gemm",
    ) as (forge_model, metadata):
        assert Path(forge_model) == model
        assert metadata["used"] is False
        assert metadata["normalized_fields"] == []

    assert not (tmp_path / "vendor_gemm" / "model_view").exists()


def test_forge_model_view_cleans_partial_overlay_on_setup_error(
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.perfopt import vendor_gemm

    model = tmp_path / "quant_ckpt"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "moe_intermediate_size": None,
                }
            }
        )
    )
    (model / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        vendor_gemm.os,
        "symlink",
        MagicMock(side_effect=OSError("link failed")),
    )

    with (
        pytest.raises(OSError, match="link failed"),
        vendor_gemm._forge_model_view(
            model,
            tmp_path / "vendor_gemm",
        ),
    ):
        pass

    assert not (tmp_path / "vendor_gemm" / "model_view").exists()


def test_vendor_lane_uses_runtime_kernel_evidence():
    from quark.experimental.torch.quant_perf.perfopt import vendor_gemm

    fp8 = vendor_gemm._classify_vendor_lane(
        {
            "op_name": "Cijk_Alik_Bljk_F8BS_kernel.kd",
            "dtypes": [],
        }
    )
    assert fp8 is not None
    assert fp8.key == "fp8_blockscale"
    assert fp8.precision == "fp8"
    assert fp8.quant_type == "blockscale"

    a4w4 = vendor_gemm._classify_vendor_lane({"op_name": "aiter::gemm_a4w4_blockscale"})
    assert a4w4 is not None
    assert a4w4.key == "a4w4_blockscale"

    assert vendor_gemm._classify_vendor_lane({"op_name": "Cijk_vendor_kernel"}) is None


def test_vendor_result_retry_is_bounded_and_versioned():
    from quark.experimental.torch.quant_perf.perfopt import vendor_gemm

    assert vendor_gemm.vendor_result_needs_run({})
    assert vendor_gemm.vendor_result_needs_run(
        {
            "status": "failed",
            "capability_version": 2,
            "attempt_count": 1,
        }
    )
    assert vendor_gemm.vendor_result_needs_run(
        {
            "status": "failed",
            "capability_version": 4,
            "attempt_count": 1,
            "retryable": True,
        }
    )
    assert not vendor_gemm.vendor_result_needs_run(
        {
            "status": "failed",
            "capability_version": 4,
            "attempt_count": 2,
            "retryable": True,
        }
    )
    assert not vendor_gemm.vendor_result_needs_run(
        {
            "status": "failed",
            "capability_version": 4,
            "attempt_count": 1,
            "retryable": False,
        }
    )
    assert not vendor_gemm.vendor_result_needs_run(
        {
            "status": "candidate",
            "capability_version": 4,
            "attempt_count": 1,
        }
    )


def test_extract_gemm_shapes_deduplicates_valid_shapes():
    from quark.experimental.torch.quant_perf.perfopt.vendor_gemm import extract_gemm_shapes

    shapes = extract_gemm_shapes(
        [
            {
                "gemm_shape": {"M": 16, "N": 4096, "K": 4096},
                "kernel_time_us": 100.0,
            },
            {
                "gemm_shape": {"M": 16, "N": 4096, "K": 4096},
                "kernel_time_us": 50.0,
            },
            {
                "gemm_shape": {"M": None, "N": 1, "K": 1},
                "kernel_time_us": 500.0,
            },
        ]
    )

    assert shapes == [{"M": 16, "N": 4096, "K": 4096, "weight": 150.0}]


def test_extract_gemm_shapes_supports_multiple_shapes_per_kernel():
    from quark.experimental.torch.quant_perf.perfopt.vendor_gemm import extract_gemm_shapes

    shapes = extract_gemm_shapes(
        [
            {
                "gemm_shapes": [
                    {"M": 31, "N": 12288, "K": 2048},
                    {"M": 32, "N": 12288, "K": 2048},
                ],
                "kernel_time_us": 100.0,
            }
        ]
    )

    assert shapes == [
        {"M": 31, "N": 12288, "K": 2048, "weight": 50.0},
        {"M": 32, "N": 12288, "K": 2048, "weight": 50.0},
    ]


def test_tuner_unavailable_is_a_nonfatal_skip(tmp_path, monkeypatch):
    from quark.experimental.torch.quant_perf.perfopt import vendor_gemm

    monkeypatch.setattr(vendor_gemm, "resolve_forge_command", lambda: None)
    result = vendor_gemm.run_vendor_gemm_tuning(
        [{"gemm_shape": {"M": 1, "N": 2, "K": 3}}],
        SimpleNamespace(
            quant_ckpt_dir="/q",
            framework="vllm",
            gpu_type="mi355x",
            tp=1,
            bench_concurrency=64,
            active_kernel_repo="",
        ),
        tmp_path,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "tuner_unavailable"
    assert result["candidates"] == []


def test_flydsl_w4a8_routes_vendor_mm_to_vllm_tunableop(
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.perfopt import vendor_gemm

    monkeypatch.setattr(
        vendor_gemm,
        "resolve_forge_command",
        lambda: ["forge-gemm-tune"],
    )
    tunable_input = tmp_path / "untuned.csv"
    tunable_input.write_text("GemmTunableOp,params,Default,0.0\n")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        (output_dir / "result.json").write_text(
            json.dumps(
                {
                    "micro_decision": "candidate",
                    "best_speedup": 1.05,
                    "recommended_env": {"PYTORCH_TUNABLEOP_FILENAME": str(tmp_path / "tuned.csv")},
                }
            )
        )
        (tmp_path / "tuned.csv").write_text("result\n")
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(vendor_gemm.subprocess, "run", fake_run)

    result = vendor_gemm.run_vendor_gemm_tuning(
        [
            {
                "op_name": "Cijk_vendor.kd",
                "parent_op_name": "aten::mm",
                "kernel_time_us": 10.0,
                "gemm_shape": {"M": 64, "N": 2048, "K": 2048},
                "tunableop_input": str(tunable_input),
            }
        ],
        SimpleNamespace(
            quant_ckpt_dir="/q",
            framework="vllm",
            gpu_type="mi355x",
            tp=1,
            bench_concurrency=64,
            active_kernel_repo="",
            w4a8_gemm_backend="flydsl",
        ),
        tmp_path,
    )

    assert result["status"] == "candidate"
    assert result["capability_version"] == 4
    cmd = captured["cmd"]
    assert cmd[cmd.index("--tuner") + 1] == "vllm_dense_tunableop"
    assert cmd[cmd.index("--tunableop-input") + 1] == str(tunable_input)


def test_w4a8_vendor_lane_without_shape_capture_has_precise_skip(
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.perfopt import vendor_gemm

    monkeypatch.setattr(
        vendor_gemm,
        "resolve_forge_command",
        lambda: ["forge-gemm-tune"],
    )
    run = MagicMock()
    monkeypatch.setattr(vendor_gemm.subprocess, "run", run)

    result = vendor_gemm.run_vendor_gemm_tuning(
        [{"op_name": "Cijk_vendor.kd", "parent_op_name": "aten::mm"}],
        SimpleNamespace(
            quant_ckpt_dir="/q",
            framework="vllm",
            gpu_type="mi355x",
            tp=1,
            bench_concurrency=64,
            active_kernel_repo="",
            w4a8_gemm_backend="flydsl",
            gpu_id=0,
        ),
        tmp_path,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "missing_exact_gemm_shape_evidence"
    assert result["capability_version"] == 4
    run.assert_not_called()


def test_filter_tunableop_signatures_keeps_only_selected_shapes(tmp_path):
    from quark.experimental.torch.quant_perf.perfopt.vendor_gemm import (
        filter_tunableop_signatures,
    )

    source = tmp_path / "tunableop_untuned.csv"
    source.write_text(
        "ScaledGemmTunableOp_Float8_e4m3fn_Float8_e4m3fn_"
        "BFloat16_TN,tn_64_12288_2048_ld_2048_2048_64_rw_1_bias_None\n"
        "ScaledGemmTunableOp_Float8_e4m3fn_Float8_e4m3fn_"
        "BFloat16_TN,tn_32_12288_2048_ld_2048_2048_32_rw_1_bias_None\n"
        "GemmTunableOp_BFloat16_TN,"
        "tn_64_12288_2048_ld_2048_2048_64\n"
    )
    output = tmp_path / "selected.csv"

    count = filter_tunableop_signatures(
        source,
        output,
        shapes=[{"M": 64, "N": 12288, "K": 2048}],
        scaled_only=True,
    )

    assert count == 1
    assert output.read_text().splitlines() == [
        "ScaledGemmTunableOp_Float8_e4m3fn_Float8_e4m3fn_BFloat16_TN,tn_64_12288_2048_ld_2048_2048_64_rw_1_bias_None"
    ]


def test_vllm_vendor_without_shapes_skips_model_wide_tunableop(
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.perfopt import vendor_gemm

    model = tmp_path / "quant_ckpt"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "hidden_size": 5376,
                    "intermediate_size": 21504,
                    "moe_intermediate_size": None,
                }
            }
        )
    )
    (model / "model.safetensors").write_bytes(b"weights")
    monkeypatch.setattr(
        vendor_gemm,
        "resolve_forge_command",
        lambda: ["forge-gemm-tune"],
    )
    run = MagicMock()
    monkeypatch.setattr(vendor_gemm.subprocess, "run", run)

    result = vendor_gemm.run_vendor_gemm_tuning(
        [
            {
                "op_name": "Cijk_Alik_Bljk_F8BS_kernel.kd",
                "kernel_time_us": 10.0,
                "gemm_shape": {"M": None, "N": None, "K": None},
            }
        ],
        SimpleNamespace(
            quant_ckpt_dir=str(model),
            framework="vllm",
            gpu_type="mi355x",
            tp=1,
            bench_concurrency=64,
            active_kernel_repo="",
            gpu_id=0,
            mxfp4_gemm_backend="triton",
            w4a8_gemm_backend="triton",
        ),
        tmp_path / "vendor_gemm",
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "missing_exact_gemm_shape_evidence"
    assert result["capability_version"] == 4
    assert result["lanes"][0]["scope"] == "trace_selected"
    run.assert_not_called()
    assert json.loads((model / "config.json").read_text())["text_config"]["moe_intermediate_size"] is None


def test_shaped_f8bs_vendor_uses_trace_selected_tunableop_signatures(
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.perfopt import vendor_gemm

    model = tmp_path / "quant_ckpt"
    model.mkdir()
    (model / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "hidden_size": 5376,
                    "intermediate_size": 21504,
                }
            }
        )
    )
    monkeypatch.setattr(
        vendor_gemm,
        "resolve_forge_command",
        lambda: ["forge-gemm-tune"],
    )
    tunable_input = tmp_path / "selected_tunableop.csv"
    tunable_input.write_text("ScaledGemmTunableOp_Float8_e4m3fn_Float8_e4m3fn_BFloat16_TN,tn_16_5376_5376\n")
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        (output_dir / "result.json").write_text(json.dumps({"micro_decision": "no_improvement"}))
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(vendor_gemm.subprocess, "run", fake_run)

    result = vendor_gemm.run_vendor_gemm_tuning(
        [
            {
                "op_name": "Cijk_Alik_Bljk_F8BS_kernel.kd",
                "parent_op_name": "aten::_scaled_mm",
                "kernel_time_us": 10.0,
                "gemm_shape": {"M": 16, "N": 5376, "K": 5376},
                "tunableop_input": str(tunable_input),
            }
        ],
        SimpleNamespace(
            quant_ckpt_dir=str(model),
            framework="vllm",
            gpu_type="mi355x",
            tp=1,
            bench_concurrency=64,
            active_kernel_repo="",
            gpu_id=0,
            mxfp4_gemm_backend="triton",
            w4a8_gemm_backend="triton",
        ),
        tmp_path / "vendor_gemm",
    )

    cmd = captured["cmd"]
    assert cmd[cmd.index("--tuner") + 1] == "vllm_dense_tunableop"
    assert cmd[cmd.index("--tunableop-input") + 1] == str(tunable_input)
    assert "--shapes-json" not in cmd
    assert result["status"] == "no_improvement"
    assert result["lanes"][0]["lane"] == "vllm_tunableop"


def test_flydsl_a4w4_skips_incompatible_vendor_tuner(
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.perfopt import vendor_gemm

    monkeypatch.setattr(
        vendor_gemm,
        "resolve_forge_command",
        lambda: ["forge-gemm-tune"],
    )
    run = MagicMock()
    monkeypatch.setattr(vendor_gemm.subprocess, "run", run)

    result = vendor_gemm.run_vendor_gemm_tuning(
        [
            {
                "op_name": "aiter::gemm_a4w4_blockscale",
                "gemm_shape": {"M": 1, "N": 128, "K": 256},
            }
        ],
        SimpleNamespace(
            quant_ckpt_dir="/q",
            framework="vllm",
            gpu_type="mi355x",
            tp=1,
            bench_concurrency=64,
            active_kernel_repo="",
            mxfp4_gemm_backend="flydsl",
            w4a8_gemm_backend="triton",
        ),
        tmp_path,
    )

    assert result["status"] == "skipped"
    assert result["reason"] == "unsupported_flydsl_dense_backend"
    assert result["candidates"] == []
    assert result["capability_version"] == 4
    run.assert_not_called()


def test_forge_candidate_normalizes_runtime_environment(
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.perfopt import vendor_gemm

    monkeypatch.setattr(
        vendor_gemm,
        "resolve_forge_command",
        lambda: ["forge-gemm-tune"],
    )
    tuned = tmp_path / "tuned.csv"
    tuned.write_text("M,N,K\n1,2,3\n")
    result_payload = {
        "status": "ok",
        "micro_decision": "candidate",
        "best_speedup": 1.12,
        "recommended_env": {"AITER_CONFIG_GEMM_A4W4": str(tuned)},
        "artifacts": {"tuned_csv": str(tuned)},
    }

    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        output_dir = Path(cmd[cmd.index("--output-dir") + 1])
        output = output_dir / "result.json"
        output.write_text(json.dumps(result_payload))
        return MagicMock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(vendor_gemm.subprocess, "run", fake_run)
    result = vendor_gemm.run_vendor_gemm_tuning(
        [
            {
                "op_name": "aiter::gemm_a4w4_blockscale",
                "gemm_shape": {"M": 1, "N": 2, "K": 3},
            }
        ],
        SimpleNamespace(
            quant_ckpt_dir="/q",
            framework="vllm",
            gpu_type="mi355x",
            tp=1,
            bench_concurrency=64,
            active_kernel_repo="",
        ),
        tmp_path,
    )

    assert result["status"] == "candidate"
    assert captured["cmd"][captured["cmd"].index("--precision") + 1] == "fp4"
    assert captured["cmd"][captured["cmd"].index("--quant-type") + 1] == "mxfp4"
    assert captured["cmd"][captured["cmd"].index("--tuner") + 1] == "a4w4_blockscale"
    assert captured["cmd"][captured["cmd"].index("--gpu-ids") + 1] == "0"
    assert result["candidates"] == [
        {
            "name": "forge_vendor_gemm:a4w4_blockscale",
            "runtime_env": {"AITER_CONFIG_GEMM_A4W4": str(tuned)},
            "artifacts": {"tuned_csv": str(tuned)},
            "micro_speedup": 1.12,
        }
    ]


def test_tunableop_measured_zero_improvement_is_not_a_candidate(
    tmp_path,
):
    from quark.experimental.torch.quant_perf.perfopt.vendor_gemm import _candidate_from_result

    tuned = tmp_path / "tuned.csv"
    tuned.write_text("result\n")
    result = {
        "micro_decision": "candidate",
        "best_speedup": 1.0,
        "recommended_env": {
            "PYTORCH_TUNABLEOP_FILENAME": str(tuned),
        },
        "tuners_run": [
            {
                "tuner": "vllm_dense_tunableop",
                "artifact": str(tuned),
                "candidate": True,
                "total_shapes": 107,
                "improved_shapes": 0,
                "best_micro_speedup": 1.0,
            }
        ],
    }

    assert _candidate_from_result(result, "vllm_tunableop") == []


def test_tunableop_unmeasured_artifact_requires_e2e_screen(tmp_path):
    from quark.experimental.torch.quant_perf.perfopt.vendor_gemm import _candidate_from_result

    tuned = tmp_path / "tuned.csv"
    tuned.write_text("result\n")
    result = {
        "micro_decision": "candidate",
        "best_speedup": 1.0,
        "recommended_env": {
            "PYTORCH_TUNABLEOP_FILENAME": str(tuned),
        },
        "tuners_run": [
            {
                "tuner": "vllm_dense_tunableop",
                "artifact": str(tuned),
                "candidate": True,
                "total_shapes": 0,
            }
        ],
    }

    assert _candidate_from_result(result, "vllm_tunableop") == [
        {
            "name": "forge_vendor_gemm:vllm_tunableop",
            "runtime_env": {
                "PYTORCH_TUNABLEOP_FILENAME": str(tuned),
            },
            "artifacts": {},
            "micro_speedup": 1.0,
            "requires_screen": True,
        }
    ]


def test_non_tunableop_zero_improved_shapes_is_not_a_candidate(tmp_path):
    from quark.experimental.torch.quant_perf.perfopt.vendor_gemm import _candidate_from_result

    tuned = tmp_path / "tuned.csv"
    tuned.write_text("result\n")
    result = {
        "micro_decision": "candidate",
        "best_speedup": 1.0,
        "recommended_env": {
            "AITER_CONFIG_GEMM_A4W4": str(tuned),
        },
        "tuners_run": [
            {
                "tuner": "a4w4_blockscale",
                "artifact": str(tuned),
                "candidate": True,
                "total_shapes": 107,
                "improved_shapes": 0,
                "best_micro_speedup": 1.0,
            }
        ],
    }

    assert _candidate_from_result(result, "vllm_tunableop") == []


def test_resolve_untuned_csv_requires_model_matching_k(tmp_path):
    from quark.experimental.torch.quant_perf.perfopt.vendor_gemm import resolve_untuned_csv

    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text(json.dumps({"hidden_size": 4096}))
    repo = tmp_path / "aiter"
    configs = repo / "aiter" / "configs"
    configs.mkdir(parents=True)
    csv_path = configs / "a4w4_blockscale_untuned_gemm.csv"
    csv_path.write_text("M,N,K\n16,4096,2048\n")

    assert resolve_untuned_csv(repo, "mxfp4", model) == ""

    csv_path.write_text("M,N,K\n16,4096,4096\n")
    assert resolve_untuned_csv(repo, "mxfp4", model) == str(csv_path)
