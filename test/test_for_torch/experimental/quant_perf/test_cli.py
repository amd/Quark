#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for quark.experimental.torch.quant_perf.cli.build_spec: session_dir must always come out
absolute. Direct PTQ's run_ptq() embeds session_dir in a prompt executed by a
subprocess whose cwd is config.quark_root(), not Quark Quant-Perf's cwd -- a relative
session_dir would silently resolve under the wrong repo (confirmed by a real
E2E run whose quant_ckpt landed under Quark/ instead of Quark Quant-Perf/)."""

import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import quark.experimental.torch.quant_perf.cli as cli
from quark.experimental.torch.quant_perf import config
from quark.experimental.torch.quant_perf.cli import _prepend_pythonpath, build_parser, build_spec, main
from quark.experimental.torch.quant_perf.evaluation.cli import (
    build_eval_parser,
    eval_spec_from_args,
    run_eval_command,
    runtime_drift_errors,
)
from quark.experimental.torch.quant_perf.session.spec import (
    Checkpoint,
    DeployPackage,
    EvalProfile,
    RuntimeContext,
    SessionLock,
)


def test_default_session_dir_is_absolute():
    args = build_parser().parse_args(["--model", "some/model"])
    spec = build_spec(args)
    assert os.path.isabs(spec.session_dir)


def test_default_framework_is_vllm():
    spec = build_spec(build_parser().parse_args(["--model", "some/model"]))
    assert spec.framework == "vllm"


@pytest.mark.parametrize("enabled", [False, True])
def test_file2file_export_option_survives_worker_spec_roundtrip(enabled):
    from quark.experimental.torch.quant_perf.session.spec import Spec

    extra = ["--file2file-export"] if enabled else []
    spec = build_spec(build_parser().parse_args(["--model", "some/model", *extra]))
    assert Spec.from_dict(spec.to_dict()).file2file_export is enabled


@pytest.mark.parametrize(
    "extra,search_memory,inference_memory",
    [
        ([], 0.75, 0.85),
        (["--search-gpu-memory-utilization", "0.65"], 0.65, 0.85),
        (["--vllm-extra-arg=--gpu-memory-utilization 0.8"], 0.8, 0.8),
        (["--vllm-extra-arg=--gpu-memory-utilization=0.8", "--search-gpu-memory-utilization", "0.6"], 0.6, 0.8),
    ],
)
def test_search_memory_precedence_and_inference_isolation(extra, search_memory, inference_memory):
    from quark.experimental.torch.quant_perf.quantize.search import _search_runtime_args
    from quark.experimental.torch.quant_perf.session.spec import Spec

    spec = build_spec(build_parser().parse_args(["--model", "some/model", *extra]))
    restored = Spec.from_dict(spec.to_dict())
    assert f"--gpu-memory-utilization={search_memory}" in _search_runtime_args(restored)
    assert restored.vllm_gpu_memory_utilization == inference_memory


def test_search_memory_rejects_invalid_override():
    args = build_parser().parse_args(["--model", "some/model", "--search-gpu-memory-utilization=nan"])
    with pytest.raises(ValueError, match="search.gpu.memory.utilization"):
        build_spec(args)


def test_default_gpu_type_is_mi355x():
    spec = build_spec(build_parser().parse_args(["--model", "some/model"]))

    assert spec.gpu_type == "mi355x"
    assert spec.gpu_arch == "MI355X"


def test_standalone_eval_defaults_to_mi355x(tmp_path):
    base = tmp_path / "base"
    quant = tmp_path / "quant"
    base.mkdir()
    quant.mkdir()
    (base / "config.json").write_text('{"model_type": "llama"}')

    args = build_eval_parser().parse_args(
        [
            "--model",
            str(quant),
            "--base-model",
            str(base),
            "--session-dir",
            str(tmp_path / "eval"),
            "--inference-moe-backend",
            "aiter",
        ]
    )
    spec, _ = eval_spec_from_args(args)

    assert spec.gpu_type == "mi355x"
    assert spec.gpu_arch == "MI355X"
    assert spec.vllm_moe_backend == "aiter"


def test_search_candidate_lists_use_canonical_cli_names():
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "some/model",
                "--layer-precision-candidates",
                "native",
                "fp8",
                "mxfp4",
                "--kv-cache-precision-candidates",
                "native",
                "fp8",
                "--max-search-candidates",
                "12",
                "--search-timeout",
                "1234",
            ]
        )
    )

    assert spec.layer_precision_candidates == ["native", "fp8", "mxfp4"]
    assert spec.kv_cache_precision_candidates == ["native", "fp8"]
    assert spec.max_search_candidates == 12
    assert spec.search_timeout_s == 1234


@pytest.mark.parametrize(
    "legacy_args",
    [
        ["--layer-mode", "native"],
        ["--kv-cache-mode", "native"],
        ["--max-rounds", "8"],
        ["--geak-budget", "2"],
    ],
)
def test_removed_legacy_search_cli_names_are_rejected(legacy_args):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", "some/model", *legacy_args])


def test_geak_direction_budget_is_the_canonical_cli_name():
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "some/model",
                "--geak-direction-budget",
                "5",
            ]
        )
    )

    assert spec.geak_direction_budget == 5


def test_resolved_config_summary_uses_canonical_search_names():
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "some/model",
                "--layer-precision-candidates",
                "native",
                "mxfp4",
                "--kv-cache-precision-candidates",
                "native",
                "--max-search-candidates",
                "6",
                "--search-gsm8k-num-samples",
                "64",
                "--gsm8k-num-samples",
                "1319",
                "--geak-direction-budget",
                "4",
            ]
        )
    )

    summary = cli._resolved_config_summary(spec)

    assert summary["gpu"]["type"] == "mi355x"
    assert summary["quantization"]["layer_precision_candidates"] == ["native", "mxfp4"]
    assert summary["quantization"]["kv_cache_precision_candidates"] == ["native"]
    assert summary["quantization"]["max_search_candidates"] == 6
    assert summary["evaluation"]["search_gsm8k_num_samples"] == 64
    assert summary["evaluation"]["accuracy_gsm8k_num_samples"] == 1319
    assert summary["optimization"]["geak_direction_budget"] == 4


def test_atom_framework_is_marked_experimental_in_help():
    action = next(action for action in build_parser()._actions if action.dest == "framework")
    assert action.help
    assert "experimental" in action.help.lower()


def test_sglang_is_not_exposed_as_a_cli_framework():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", "some/model", "--framework", "sglang"])


def test_default_benchmark_is_1k_1k_at_concurrency_64():
    spec = build_spec(build_parser().parse_args(["--model", "some/model"]))
    assert spec.isl == 1024
    assert spec.osl == 1024
    assert spec.bench_concurrency == 64


def test_performance_is_opt_in_by_default():
    spec = build_spec(build_parser().parse_args(["--model", "some/model"]))

    assert spec.performance_mode == "off"
    assert spec.target_gain is None


def test_target_gain_enables_performance_optimization():
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "some/model",
                "--target-gain",
                "1.4",
            ]
        )
    )

    assert spec.performance_mode == "optimize"
    assert spec.target_gain == 1.4


def test_measure_performance_does_not_require_a_target():
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "some/model",
                "--performance-mode",
                "measure",
            ]
        )
    )

    assert spec.performance_mode == "measure"
    assert spec.target_gain is None


def test_non_optimizing_performance_mode_rejects_target_gain():
    args = build_parser().parse_args(
        [
            "--model",
            "some/model",
            "--performance-mode",
            "off",
            "--target-gain",
            "1.2",
        ]
    )

    with pytest.raises(ValueError, match="requires --performance-mode optimize"):
        build_spec(args)


def test_server_port_is_configurable():
    spec = build_spec(build_parser().parse_args(["--model", "some/model", "--server-port", "18080"]))

    assert spec.server_port == 18080


@pytest.mark.parametrize(
    "removed_args",
    [
        ["--algorithm", "awq"],
        ["--geak-bench-concurrency", "16"],
        ["--granularity", "module"],
    ],
)
def test_removed_noop_cli_flags_are_rejected(removed_args):
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--model", "some/model", *removed_args])


def test_vllm_extra_args_split_tp_from_server_passthrough():
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "some/model",
                "--vllm-extra-arg",
                "--tensor-parallel-size 2",
                "--vllm-extra-arg=--disable-log-requests",
            ]
        )
    )
    assert spec.tp == 2
    assert spec.vllm_passthrough_args == ["--disable-log-requests"]


@pytest.mark.parametrize(
    ("options", "search", "inference"),
    [
        ([], "auto", ""),
        (["--search-moe-backend=auto"], "auto", ""),
        (["--search-moe-backend=triton", "--inference-moe-backend=aiter"], "triton", "aiter"),
        (["--search-moe-backend=triton_unfused"], "triton_unfused", ""),
        (["--vllm-extra-arg=--moe-backend=triton"], "triton", "triton"),
        (["--vllm-extra-arg=--moe_backend triton"], "triton", "triton"),
    ],
)
def test_vllm_moe_backend_is_parsed_from_passthrough_args(options, search, inference):
    spec = build_spec(build_parser().parse_args(["--model", "some/model", *options]))

    assert spec.effective_search_moe_backend == search
    assert spec.search_vllm_args == ([f"--moe-backend={search}"] if search != "auto" else [])
    assert spec.vllm_moe_backend == inference
    assert spec.vllm_passthrough_args == ([f"--moe-backend={inference}"] if inference else [])


def test_explicit_relative_session_dir_is_resolved_to_absolute():
    args = build_parser().parse_args(["--model", "some/model", "--session-dir", "runs/my-session"])
    spec = build_spec(args)
    assert os.path.isabs(spec.session_dir)
    assert spec.session_dir.endswith("runs/my-session")


def test_explicit_absolute_session_dir_is_unchanged():
    args = build_parser().parse_args(["--model", "some/model", "--session-dir", "/tmp/some-session"])
    spec = build_spec(args)
    assert spec.session_dir == "/tmp/some-session"


def test_mi350x_gpu_type_aliases_to_mi355x_arch():
    # MI350 and MI355X are the same gfx950 silicon; mi350x is accepted as a
    # --gpu-type and resolves to the MI355X hardware profile downstream.
    args = build_parser().parse_args(["--model", "some/model", "--gpu-type", "mi350x"])
    spec = build_spec(args)
    assert spec.gpu_type == "mi350x"
    assert spec.gpu_arch == "MI355X"


def test_other_gpu_types_uppercase_unchanged():
    for gt in ("mi300x", "mi325x", "mi355x"):
        spec = build_spec(build_parser().parse_args(["--model", "m", "--gpu-type", gt]))
        assert spec.gpu_arch == gt.upper()


def test_tracelens_gpu_arch_json_is_resolved_and_preserved(tmp_path):
    arch_json = tmp_path / "MI355X.json"
    arch_json.write_text('{"name": "MI355X"}')

    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "m",
                "--tracelens-gpu-arch-json",
                str(arch_json),
            ]
        )
    )

    assert spec.tracelens_gpu_arch_json == str(arch_json.resolve())


def test_kernel_repo_is_preserved_in_spec():
    args = build_parser().parse_args(["--model", "m", "--framework-repo", "/fw", "--kernel-repo", "/aiter"])
    spec = build_spec(args)
    assert spec.framework_repo == "/fw"
    assert spec.kernel_repo == "/aiter"


def test_workspace_source_defaults_to_auto():
    spec = build_spec(build_parser().parse_args(["--model", "m"]))

    assert spec.workspace_source == "auto"


def test_workspace_source_auto_is_preserved_in_spec():
    spec = build_spec(build_parser().parse_args(["--model", "m", "--workspace-source", "auto"]))

    assert spec.workspace_source == "auto"


def test_recheck_baseline_is_preserved_in_spec():
    spec = build_spec(build_parser().parse_args(["--model", "m", "--recheck-baseline"]))

    assert spec.recheck_baseline is True


def test_retry_accuracy_gate_is_preserved_in_spec():
    spec = build_spec(build_parser().parse_args(["--model", "m", "--retry-accuracy-gate"]))

    assert spec.retry_accuracy_gate is True


def test_retry_perfopt_is_preserved_in_spec():
    spec = build_spec(build_parser().parse_args(["--model", "m", "--retry-perfopt"]))

    assert spec.retry_perfopt is True


def test_bottleneck_mode_defaults_to_differential_and_accepts_absolute():
    default_spec = build_spec(build_parser().parse_args(["--model", "m"]))
    absolute_spec = build_spec(build_parser().parse_args(["--model", "m", "--bottleneck-mode", "absolute"]))

    assert default_spec.bottleneck_mode == "differential"
    assert absolute_spec.bottleneck_mode == "absolute"


def test_eval_profile_options_are_preserved_in_spec():
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "m",
                "--eval-discovery",
                "online",
                "--eval-no-llm",
                "--eval-task",
                "gsm8k_cot_zeroshot",
                "--eval-num-fewshot",
                "0",
                "--eval-prompting-strategy",
                "cot",
                "--eval-thinking-mode",
                "disabled",
                "--eval-max-gen-toks",
                "768",
            ]
        )
    )

    assert spec.eval_discovery == "online"
    assert spec.eval_allow_llm is False
    assert spec.eval_task == "gsm8k_cot_zeroshot"
    assert spec.eval_num_fewshot == 0
    assert spec.eval_prompting_strategy == "cot"
    assert spec.eval_thinking_mode == "disabled"
    assert spec.eval_max_gen_toks == 768


def test_eval_subcommand_dispatches_managed_entrypoint(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "quark.experimental.torch.quant_perf.evaluation.cli.run_eval_command",
        lambda argv, **kwargs: calls.append((list(argv), kwargs["preflight_framework_imports"])) or 0,
    )

    result = cli.main(
        [
            "eval",
            "--model",
            "/models/quant",
            "--base-model",
            "/models/base",
            "--session-dir",
            "/runs/eval",
        ]
    )

    assert result == 0
    assert calls == [
        (
            [
                "--model",
                "/models/quant",
                "--base-model",
                "/models/base",
                "--session-dir",
                "/runs/eval",
            ],
            cli._preflight_framework_imports,
        )
    ]


@patch(
    "quark.experimental.torch.quant_perf.perfopt.bottleneck_analysis.locate_bottlenecks",
    return_value=[
        {
            "op_name": "slow_kernel.kd",
            "kernel_time_us": 100.0,
            "roofline_bound": "UNKNOWN",
        }
    ],
)
def test_bottlenecks_subcommand_reads_existing_trace_without_writing_state(
    mock_locate,
    tmp_path,
    capsys,
):
    session = tmp_path / "session"
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "/models/qwen",
                "--gpu-type",
                "mi350x",
                "--session-dir",
                str(session),
            ]
        )
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.save()
    trace = session / "trace" / "dp0_pp0_tp0_dcp0_ep0_rank0.1.pt.trace.json.gz"
    trace.parent.mkdir()
    trace.touch()
    before = (session / "state.json").read_text()

    result = main(
        [
            "bottlenecks",
            "--session",
            str(session),
            "--mode",
            "absolute",
            "--top-kernels",
            "1",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["requested_mode"] == "absolute"
    assert payload["effective_mode"] == "absolute"
    assert payload["candidates"][0]["op_name"] == "slow_kernel.kd"
    assert (session / "state.json").read_text() == before
    mock_locate.assert_called_once()


def test_eval_subcommand_builds_spec_and_runs_service(tmp_path):
    base = tmp_path / "base"
    quant = tmp_path / "quant"
    kernel = tmp_path / "kernel"
    output = tmp_path / "eval"
    base.mkdir()
    quant.mkdir()
    (kernel / "aiter").mkdir(parents=True)
    (base / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
            }
        )
    )
    (base / "tokenizer_config.json").write_text("{}")
    (quant / "config.json").write_text("{}")

    with (
        patch(
            "quark.experimental.torch.quant_perf.evaluation.cli.verify_runtime_origins",
            return_value={},
        ),
        patch(
            "quark.experimental.torch.quant_perf.evaluation.service.EvaluationService.run",
            return_value={"status": "success"},
        ) as mock_run,
    ):
        result = run_eval_command(
            [
                "--model",
                str(quant),
                "--base-model",
                str(base),
                "--session-dir",
                str(output),
                "--gpu-type",
                "mi355x",
                "--kernel-repo",
                str(kernel),
                "--mxfp4-gemm-backend",
                "flydsl",
                "--gsm8k-num-samples",
                "20",
                "--eval-no-llm",
            ],
            preflight_framework_imports=cli._preflight_framework_imports,
        )

    assert result == 0
    spec, quant_model = mock_run.call_args.args
    assert quant_model == str(quant)
    assert spec.base_model == str(base)
    assert spec.model_dir == str(quant)
    assert spec.session_dir == str(output.resolve())
    assert spec.gpu_arch == "MI355X"
    assert spec.gsm8k_num_samples == 20
    assert spec.eval_allow_llm is False
    assert spec.runtime_env["AITER_ROOT_DIR"] == str(kernel)
    assert spec.runtime_env["FLYDSL_RUNTIME_ENABLE_CACHE"] == "1"


def test_eval_subcommand_passes_runtime_origin_evidence(tmp_path):
    base = tmp_path / "base"
    quant = tmp_path / "quant"
    output = tmp_path / "eval"
    base.mkdir()
    quant.mkdir()
    (base / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
            }
        )
    )
    (base / "tokenizer_config.json").write_text("{}")
    (quant / "config.json").write_text("{}")
    runtime = MagicMock()
    evidence = {
        "vllm": {
            "origin": "/runtime/vllm/__init__.py",
            "matched": True,
        }
    }

    with (
        patch(
            "quark.experimental.torch.quant_perf.evaluation.cli.activate_runtime",
            return_value=runtime,
        ),
        patch(
            "quark.experimental.torch.quant_perf.evaluation.cli.verify_runtime_origins",
            return_value=evidence,
            create=True,
        ) as verify_origins,
        patch(
            "quark.experimental.torch.quant_perf.evaluation.service.EvaluationService.run",
            return_value={"status": "success"},
        ) as run_eval,
    ):
        result = run_eval_command(
            [
                "--model",
                str(quant),
                "--base-model",
                str(base),
                "--session-dir",
                str(output),
                "--eval-no-llm",
            ],
            preflight_framework_imports=cli._preflight_framework_imports,
        )

    assert result == 0
    verify_origins.assert_called_once_with(runtime)
    assert run_eval.call_args.kwargs["runtime_origin_evidence"] == evidence


def test_eval_runtime_drift_compares_prior_source_heads(tmp_path):
    source_session = tmp_path / "source"
    reports = source_session / "reports"
    reports.mkdir(parents=True)
    (reports / "final.json").write_text(
        json.dumps(
            {
                "repositories": {
                    "framework": {"source_head": "framework-old"},
                    "kernel": {"source_head": "kernel-old"},
                }
            }
        )
    )

    errors = runtime_drift_errors(
        source_session,
        {
            "framework": "framework-new",
            "kernel": "kernel-old",
        },
    )

    assert errors == ["framework source HEAD changed: framework-old -> framework-new"]


def test_eval_subcommand_from_session_uses_existing_checkpoint(tmp_path):
    source = tmp_path / "source"
    output = tmp_path / "eval"
    quant = source / "quant_ckpt"
    framework = tmp_path / "framework"
    kernel = tmp_path / "kernel"
    quant.mkdir(parents=True)
    (framework / "vllm").mkdir(parents=True)
    (kernel / "aiter").mkdir(parents=True)
    spec = cli.Spec(
        model_dir="/models/base",
        base_model="/models/base",
        framework="vllm",
        gpu_type="mi355x",
        gpu_arch="MI355X",
        isl=128,
        osl=128,
        quant_strategy=None,
        session_dir=str(source),
        arch_fingerprint="abc",
        gsm8k_num_samples=20,
        framework_repo=str(framework),
        kernel_repo=str(kernel),
        runtime=RuntimeContext(
            framework_worktree="/removed/framework-worktree",
            kernel_worktree="/removed/kernel-worktree",
        ),
        eval_profile=EvalProfile(
            profile_id="custom-profile",
            profile_hash="custom-hash",
            model_mode="chat",
            apply_chat_template=True,
            enable_thinking=False,
            detection_reason="user",
            schema_version=2,
            policy_version="custom-policy",
            task="gsm8k_cot_zeroshot",
            num_fewshot=0,
            prompting_strategy="cot",
            settings_source="user",
        ),
        eval_num_fewshot=0,
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["run_spec"] = spec.to_dict()
    ckpt.state["quant_ckpt_dir"] = str(quant)
    ckpt.save()

    with (
        patch(
            "quark.experimental.torch.quant_perf.evaluation.cli.verify_runtime_origins",
            return_value={},
        ),
        patch(
            "quark.experimental.torch.quant_perf.evaluation.service.EvaluationService.run",
            return_value={"status": "success"},
        ) as mock_run,
        patch.object(cli, "_preflight_framework_imports") as preflight,
    ):
        result = run_eval_command(
            [
                "--from-session",
                str(source),
                "--session-dir",
                str(output),
            ],
            preflight_framework_imports=preflight,
        )

    assert result == 0
    eval_spec, quant_model = mock_run.call_args.args
    assert quant_model == str(quant)
    assert eval_spec.base_model == "/models/base"
    assert eval_spec.model_dir == str(quant)
    assert eval_spec.session_dir == str(output.resolve())
    assert eval_spec.eval_profile is not None
    assert eval_spec.eval_profile.profile_id == "custom-profile"
    assert eval_spec.eval_num_fewshot == 0
    assert eval_spec.framework_worktree == ""
    assert eval_spec.kernel_worktree == ""
    assert eval_spec.active_framework_repo == str(framework)
    assert eval_spec.active_kernel_repo == str(kernel)
    assert eval_spec.runtime_env["AITER_ROOT_DIR"] == str(kernel)


def test_prepend_pythonpath_adds_kernel_repo_once(monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/existing")
    monkeypatch.setattr(sys, "path", list(sys.path))
    _prepend_pythonpath("/aiter")
    _prepend_pythonpath("/aiter")
    assert os.environ["PYTHONPATH"].split(os.pathsep) == ["/aiter", "/existing"]
    assert sys.path[0] == str(Path("/aiter").resolve())


def test_atom_preflight_fails_before_work_when_root_is_missing(
    monkeypatch,
    tmp_path,
):
    preflight = getattr(
        cli,
        "_preflight_atom",
        lambda: pytest.fail("_preflight_atom is not implemented"),
    )
    missing = tmp_path / "missing-atom"
    monkeypatch.setattr(config, "atom_root", lambda: str(missing))

    with pytest.raises(SystemExit):
        preflight()


def test_atom_preflight_probes_configured_root(monkeypatch, tmp_path):
    preflight = getattr(
        cli,
        "_preflight_atom",
        lambda: pytest.fail("_preflight_atom is not implemented"),
    )
    root = tmp_path / "atom"
    root.mkdir()
    monkeypatch.setattr(config, "atom_root", lambda: str(root))
    result = MagicMock(returncode=0, stdout="", stderr="")

    with patch.object(cli, "subprocess", create=True) as mock_subprocess:
        mock_subprocess.run.return_value = result
        preflight()

    mock_run = mock_subprocess.run
    assert mock_run.call_args.kwargs["cwd"] == str(root.resolve())
    env = mock_run.call_args.kwargs["env"]
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(root.resolve())


def test_main_runs_atom_preflight_before_model_intake():
    with (
        patch.object(
            cli,
            "_preflight_atom",
            side_effect=SystemExit(1),
        ) as mock_preflight,
        patch("quark.experimental.torch.quant_perf.orchestration.intake.build_spec") as mock_build,
    ):
        with pytest.raises(SystemExit):
            main(["--model", "m", "--framework", "atom"])

    mock_preflight.assert_called_once()
    mock_build.assert_not_called()


def test_gc_subcommand_reports_owned_workspace_scan(tmp_path, capsys):
    root = tmp_path / "workspaces"
    root.mkdir()

    rc = main(
        [
            "gc",
            "--dry-run",
            "--root",
            str(root),
            "--max-age-hours",
            "0",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["roots"] == [str(root.resolve())]


@patch(
    "quark.experimental.torch.quant_perf.orchestration.backend_probe.run_backend_probe",
    return_value={"status": "complete", "selected_backend": "asm"},
)
def test_backend_probe_subcommand_reports_result(
    mock_probe,
    tmp_path,
    capsys,
):
    session = tmp_path / "session"
    session.mkdir()

    rc = cli.main(
        [
            "backend-probe",
            "--session",
            str(session),
        ]
    )

    assert rc == 0
    assert json.loads(capsys.readouterr().out)["selected_backend"] == "asm"
    mock_probe.assert_called_once_with(session.resolve())


def test_flydsl_backend_and_tuned_fmoe_path_are_preserved():
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "m",
                "--mxfp4-moe-backend",
                "flydsl",
                "--aiter-config-fmoe",
                "/configs/qwen35.csv",
            ]
        )
    )
    assert spec.mxfp4_moe_backend == "flydsl"
    assert spec.aiter_config_fmoe == "/configs/qwen35.csv"


def test_w4a8_flydsl_backend_is_preserved():
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "m",
                "--w4a8-gemm-backend",
                "flydsl",
            ]
        )
    )

    assert spec.w4a8_gemm_backend == "flydsl"


def test_mxfp4_flydsl_dense_backend_is_preserved():
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "m",
                "--mxfp4-gemm-backend",
                "flydsl",
            ]
        )
    )

    assert spec.mxfp4_gemm_backend == "flydsl"


def test_mxfp4_asm_dense_backend_is_preserved():
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "m",
                "--mxfp4-gemm-backend",
                "asm",
            ]
        )
    )

    assert spec.mxfp4_gemm_backend == "asm"


def test_runtime_env_configures_vllm_w4a8_backend(monkeypatch):
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "m",
                "--w4a8-gemm-backend",
                "flydsl",
            ]
        )
    )
    monkeypatch.delenv("QUARK_QUANT_PERF_W4A8_GEMM_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_ROCM_W4A8_GEMM_BACKEND", raising=False)

    cli._configure_runtime_env(spec)

    assert os.environ["QUARK_QUANT_PERF_W4A8_GEMM_BACKEND"] == "flydsl"
    assert os.environ["VLLM_ROCM_W4A8_GEMM_BACKEND"] == "flydsl"


def test_runtime_env_configures_vllm_mxfp4_backend(monkeypatch):
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "m",
                "--mxfp4-gemm-backend",
                "flydsl",
            ]
        )
    )
    monkeypatch.delenv("QUARK_QUANT_PERF_MXFP4_GEMM_BACKEND", raising=False)
    monkeypatch.delenv("VLLM_ROCM_MXFP4_GEMM_BACKEND", raising=False)

    cli._configure_runtime_env(spec)

    assert os.environ["QUARK_QUANT_PERF_MXFP4_GEMM_BACKEND"] == "flydsl"
    assert os.environ["VLLM_ROCM_MXFP4_GEMM_BACKEND"] == "flydsl"


def test_runtime_env_configures_asm_and_clears_it_for_flydsl(monkeypatch):
    asm_spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "m",
                "--mxfp4-gemm-backend",
                "asm",
            ]
        )
    )
    flydsl_spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "m",
                "--mxfp4-gemm-backend",
                "flydsl",
            ]
        )
    )
    monkeypatch.delenv(
        "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM",
        raising=False,
    )

    cli._configure_runtime_env(asm_spec)

    assert os.environ["QUARK_QUANT_PERF_MXFP4_GEMM_BACKEND"] == "asm"
    assert os.environ["VLLM_ROCM_MXFP4_GEMM_BACKEND"] == "triton"
    assert os.environ["VLLM_ROCM_USE_AITER_FP4_ASM_GEMM"] == "1"

    cli._configure_runtime_env(flydsl_spec)

    assert os.environ["QUARK_QUANT_PERF_MXFP4_GEMM_BACKEND"] == "flydsl"
    assert os.environ["VLLM_ROCM_MXFP4_GEMM_BACKEND"] == "flydsl"
    assert os.environ["VLLM_ROCM_USE_AITER_FP4_ASM_GEMM"] == "0"


def test_runtime_env_configures_flydsl_moe_backend(monkeypatch):
    spec = build_spec(
        build_parser().parse_args(
            [
                "--model",
                "m",
                "--mxfp4-moe-backend",
                "flydsl",
            ]
        )
    )
    monkeypatch.setenv("VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE", "1")
    for key in (
        "VLLM_ROCM_USE_AITER",
        "VLLM_ROCM_USE_AITER_MOE",
        "VLLM_ROCM_USE_AITER_FLYDSL_MOE",
        "AITER_FLYDSL_FORCE",
    ):
        monkeypatch.delenv(key, raising=False)

    cli._configure_runtime_env(spec)

    assert os.environ["VLLM_ROCM_USE_AITER"] == "1"
    assert os.environ["VLLM_ROCM_USE_AITER_MOE"] == "1"
    assert os.environ["VLLM_ROCM_USE_AITER_FLYDSL_MOE"] == "1"
    assert os.environ["AITER_FLYDSL_FORCE"] == "1"
    assert "VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE" not in os.environ


def test_runtime_env_preserves_explicit_aiter_moe_opt_out():
    spec = build_spec(build_parser().parse_args(["--model", "m"]))
    environment = {"VLLM_ROCM_USE_AITER": "1", "VLLM_ROCM_USE_AITER_MOE": "0"}

    cli._configure_runtime_env(spec, environment)

    assert environment["VLLM_ROCM_USE_AITER"] == "1"
    assert environment["VLLM_ROCM_USE_AITER_MOE"] == "0"
    assert "VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE" not in environment
    assert "VLLM_ROCM_USE_AITER_FLYDSL_MOE" not in environment


def test_report_subcommand_regenerates_terminal_artifacts(
    tmp_path,
    capsys,
    monkeypatch,
):
    monkeypatch.setenv(
        "QUARK_QUANT_PERF_EXPERIENCE_STORE_PATH",
        str(tmp_path / "experience.sqlite"),
    )
    spec = build_spec(build_parser().parse_args(["--model", "m", "--session-dir", str(tmp_path)]))
    ckpt = Checkpoint.fresh(spec)
    package = DeployPackage(
        status="perf_below_target",
        quant_ckpt_dir="/q",
        message="below target",
    )
    ckpt.state.update(
        {
            "stage": "perf_failed",
            "terminal_stage": "perf_failed",
            "terminal_result": {
                **package.__dict__,
                "perf": None,
            },
        }
    )
    ckpt.save()

    rc = main(["report", "--session", str(tmp_path)])

    output = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert output["status"] == "complete"
    assert Path(output["paths"]["final_json"]).is_file()
    assert Path(output["paths"]["final_md"]).is_file()
    assert Path(output["paths"]["session_breakdown_json"]).is_file()
    assert Path(output["paths"]["session_report_md"]).is_file()
    state = Checkpoint.load(tmp_path).state
    assert state["reporting"]["status"] == "complete"
    assert state["experience_capture"] == {
        "quantization": 0,
        "repair": 0,
        "kernel_optimization": 0,
        "reviewable": 0,
    }
    progress = json.loads((tmp_path / "progress.json").read_text())
    assert progress["report_status"] == "complete"
    assert progress["reports"] == output["paths"]


@patch("quark.experimental.torch.quant_perf.reporting.service.write_final_artifacts")
def test_report_subcommand_rejects_nonterminal_session_without_mutating_state(
    mock_write_reports,
    tmp_path,
    capsys,
):
    spec = build_spec(build_parser().parse_args(["--model", "m", "--session-dir", str(tmp_path)]))
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["stage"] = "benchmark"
    ckpt.save()
    state_path = tmp_path / "state.json"
    state_before = state_path.read_bytes()

    rc = main(["report", "--session", str(tmp_path)])

    assert rc == 1
    assert "not terminal" in capsys.readouterr().err
    assert state_path.read_bytes() == state_before
    mock_write_reports.assert_not_called()


@patch("quark.experimental.torch.quant_perf.reporting.service.write_final_artifacts")
def test_report_subcommand_respects_active_session_lock(
    mock_write_reports,
    tmp_path,
    capsys,
):
    spec = build_spec(build_parser().parse_args(["--model", "m", "--session-dir", str(tmp_path)]))
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["stage"] = "perf_failed"
    ckpt.state["terminal_stage"] = "perf_failed"
    ckpt.state["terminal_result"] = {
        **DeployPackage(status="perf_below_target", quant_ckpt_dir="/q").__dict__,
        "perf": None,
    }
    ckpt.save()
    state_path = tmp_path / "state.json"
    state_before = state_path.read_bytes()

    with SessionLock(tmp_path):
        rc = main(["report", "--session", str(tmp_path)])

    assert rc == 1
    assert "already holds the lock" in capsys.readouterr().err
    assert state_path.read_bytes() == state_before
    mock_write_reports.assert_not_called()


def test_knowledge_subcommand_lists_and_exports_review_candidate(
    tmp_path,
    monkeypatch,
    capsys,
):
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore

    db_path = tmp_path / "experience.sqlite"
    monkeypatch.setenv(
        "QUARK_QUANT_PERF_EXPERIENCE_STORE_PATH",
        str(db_path),
    )
    with ExperienceStore(db_path) as store:
        store.record_repair_experience(
            record_id="session-1:repair:1",
            source_session_id="session-1",
            context_fingerprint="fingerprint",
            framework="vllm",
            framework_version="abc",
            arch_fingerprint="arch",
            quant_signature="mlp=mxfp4",
            failure_mode="load_run",
            error_signature="v2|AttributeError|||none.to",
            outcome="fixed",
            verification_status="verified",
            payload={"approach_summary": "guard optional bias"},
        )

    session_dir = tmp_path / "session"
    session_dir.mkdir()
    (session_dir / "state.json").write_text(json.dumps({"session_id": "session-1"}))

    assert main(["knowledge", "list", "--session", str(session_dir)]) == 0
    listed = json.loads(capsys.readouterr().out)
    key = listed[0]["experience_key"]

    candidate = tmp_path / "candidate.yaml"
    assert (
        main(
            [
                "knowledge",
                "review",
                key,
                "--output",
                str(candidate),
            ]
        )
        == 0
    )
    assert candidate.is_file()
    assert "status: proposed" in candidate.read_text()


def test_knowledge_validate_rejects_duplicate_yaml_keys(tmp_path):
    candidate = tmp_path / "candidate.yaml"
    candidate.write_text(
        "\n".join(
            [
                "schema_version: 1",
                "id: repair.one.v1",
                "id: repair.two.v1",
                "domain: repair",
                "kind: diagnostic_playbook",
                "status: proposed",
                "applicability: {}",
                "match: {}",
                "summary: duplicate",
                "guidance: []",
                "required_checks: []",
                "evidence: {level: E1}",
                "provenance: {kind: manual}",
            ]
        )
    )

    with pytest.raises(ValueError, match="duplicate YAML key"):
        main(["knowledge", "validate", str(candidate)])
