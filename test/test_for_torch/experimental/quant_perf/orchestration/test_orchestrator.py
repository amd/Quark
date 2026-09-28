#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for quark.experimental.torch.quant_perf.orchestration.orchestrator: the state machine, with
run_ptq/landing/AccuracyGate/throughput_benchmark all mocked out (no GPU)."""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from quark.experimental.torch.quant_perf.evaluation.gsm8k import EvaluationFailure
from quark.experimental.torch.quant_perf.evaluation.throughput import (
    BenchmarkFailure,
    ThroughputMeasurement,
)
from quark.experimental.torch.quant_perf.orchestration.orchestrator import Orchestrator
from quark.experimental.torch.quant_perf.pipeline.accuracy_stage import (
    AccuracyStage,
    BaselineHealthStage,
)
from quark.experimental.torch.quant_perf.pipeline.benchmarking import BenchmarkCoordinator
from quark.experimental.torch.quant_perf.pipeline.retention import CandidateRetentionService
from quark.experimental.torch.quant_perf.runtime.recovery import (
    build_accuracy_fingerprint,
    classify_failure,
)
from quark.experimental.torch.quant_perf.session.spec import (
    AccuracyResult,
    Checkpoint,
    EvalProfile,
    PerfResult,
    RuntimeContext,
    ServerHandle,
    Spec,
    StageError,
)

from ..quantize.quant_artifact_fixtures import (
    write_valid_quant_checkpoint,
)
from ..testing import init_git_repo
from ..testing import run_git as _git


def make_spec(tmp_path: Path, **overrides) -> Spec:
    defaults = dict(
        model_dir="m",
        base_model="m",
        framework="atom",
        gpu_type="mi300x",
        gpu_arch="MI300X",
        isl=128,
        osl=128,
        quant_strategy="fp8",
        performance_mode="optimize",
        target_gain=1.2,
        session_dir=str(tmp_path),
        eval_profile=EvalProfile(
            profile_id="gsm8k-chat-nothink-v1",
            profile_hash="",
            model_mode="chat",
            apply_chat_template=True,
            enable_thinking=False,
            detection_reason="chat_template",
        ).with_computed_hash(),
    )
    runtime_fields = set(RuntimeContext.__dataclass_fields__)
    runtime_overrides = {key: overrides.pop(key) for key in list(overrides) if key in runtime_fields}
    defaults.update(overrides)
    return Spec(**defaults, runtime=RuntimeContext(**runtime_overrides))


def _throughput_measurement(tps, relative_mad=0.002, stable=True):
    return ThroughputMeasurement(
        samples_tps=(tps - 1.0, tps, tps + 1.0),
        median_tps=tps,
        mad_tps=1.0,
        relative_mad=relative_mad,
        warmup_tps=tps - 2.0,
        stable=stable,
    )


def fake_server() -> ServerHandle:
    proc = MagicMock()
    server = ServerHandle(
        port=9000,
        proc=proc,
        process_group_id=proc.pid,
    )
    server.stop = MagicMock()
    return server


def _git_repo(path: Path) -> Path:
    return init_git_repo(path, {"module.py": "VALUE = 1\n"})


def _good_gate_mock(mock_gate_cls):
    """Configure a gate mock that passes the baseline-health and accuracy checks."""
    mock_gate_cls.return_value._source_cache = 0.8
    mock_gate_cls.return_value.check_baseline_health.return_value = (True, "", 0.8)
    mock_gate_cls.return_value.eval_quantized.return_value = AccuracyResult(
        gap=0.01, source_gsm8k=0.8, quantized_gsm8k=0.79, passed=True
    )


def _benchmark_coordinator(
    orchestrator: Orchestrator | None = None,
) -> BenchmarkCoordinator:
    if orchestrator is None:
        orchestrator = Orchestrator()
        orchestrator.repair_service.allow_in_place_repair = True
    return BenchmarkCoordinator(orchestrator.repair_service)


def _mark_accuracy_valid(
    ckpt,
    spec: Spec,
    quant_ckpt_dir: str,
    *,
    baseline: float = 0.8,
    quantized: float = 0.8,
) -> None:
    ckpt.state["accuracy_validation"] = {
        "fingerprint": build_accuracy_fingerprint(
            spec,
            quant_ckpt_dir,
            framework_commit=Orchestrator._framework_commit(spec),
            kernel_commit=Orchestrator._kernel_commit(spec),
            runtime_env=os.environ,
        ),
        "profile_hash": spec.eval_profile.profile_hash,
        "tp": spec.tp,
        "gsm8k_num_samples": spec.gsm8k_num_samples,
        "baseline": baseline,
        "quantized": quantized,
        "gap": 0.0,
        "passed": True,
    }


def test_accuracy_stage_routes_missing_aiter_backend_to_kernel_repair(
    tmp_path,
):
    expected = AccuracyResult(
        gap=0.0,
        source_gsm8k=0.8,
        quantized_gsm8k=0.8,
        passed=True,
    )
    gate = MagicMock()
    gate.eval_quantized.side_effect = [
        RuntimeError("W4A8 FlyDSL backend is unavailable in the installed AITER build."),
        expected,
    ]
    repair_service = MagicMock()
    repair_service.repair.return_value.status = "fixed"
    spec = make_spec(
        tmp_path,
        framework_repo="",
        kernel_repo="/source/aiter",
        kernel_worktree="/managed/aiter",
    )

    result = AccuracyStage(repair_service).evaluate_quantized_checkpoint(
        spec,
        gate,
        "/quant",
    )

    request = repair_service.repair.call_args.args[0]
    from quark.experimental.torch.quant_perf.repair.source_router import resolve_repair_target

    assert result is expected
    assert resolve_repair_target(request).role == "kernel"
    assert gate.eval_quantized.call_count == 2


def test_prepare_repo_workspaces_keeps_sources_on_original_branches(
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    framework = _git_repo(tmp_path / "vllm")
    kernel = _git_repo(tmp_path / "aiter")
    session = tmp_path / "session"
    spec = make_spec(
        session,
        framework_repo=str(framework),
        kernel_repo=str(kernel),
    )
    ckpt = Checkpoint.fresh(spec)

    manager = Orchestrator()._prepare_managed_workspaces(spec, ckpt)

    assert _git(framework, "branch", "--show-current") == "main"
    assert _git(kernel, "branch", "--show-current") == "main"
    assert spec.framework_worktree != str(framework)
    assert spec.kernel_worktree != str(kernel)
    assert Path(spec.framework_worktree).is_dir()
    assert Path(spec.kernel_worktree).is_dir()
    assert spec.active_framework_repo == spec.framework_worktree
    assert spec.active_kernel_repo == spec.kernel_worktree
    assert sys.path[:2] == [
        spec.framework_worktree,
        spec.kernel_worktree,
    ]
    assert spec.runtime_env["PYTHONPATH"].split(os.pathsep)[:2] == [
        spec.framework_worktree,
        spec.kernel_worktree,
    ]
    assert "quark" not in spec.runtime_origins
    assert ckpt.state["fw_original_branch"] == "main"
    assert ckpt.state["kernel_original_branch"] == "main"
    assert spec.framework_version == _git(framework, "rev-parse", "HEAD")
    assert spec.kernel_version == _git(kernel, "rev-parse", "HEAD")
    assert ckpt.state["runtime_inventory"]["packages"]["quark"]["origin"]
    assert manager is not None


def test_perf_failed_resume_reopens_vendor_tuning_after_capability_upgrade():
    state = {
        "vendor_gemm_tuning": {
            "status": "failed",
            "capability_version": 2,
            "attempt_count": 1,
        },
        "kernel_journey": [
            {
                "outcome": "skipped",
                "skip_reason": ("vendor library kernel (precompiled binary, not rewritable)"),
            }
        ],
    }

    assert Orchestrator._has_pending_kernel_candidates(state)

    state["vendor_gemm_tuning"].update(
        {
            "capability_version": 4,
            "retryable": False,
        }
    )
    assert not Orchestrator._has_pending_kernel_candidates(state)


def test_perf_failed_resume_reopens_retryable_source_after_resolver_upgrade():
    state = {
        "kernel_journey": [
            {
                "outcome": "skipped",
                "source_mapping": {
                    "method": "definition_search",
                    "confidence": "ambiguous",
                    "resolver_version": 2,
                    "retryable": True,
                },
            }
        ]
    }

    assert Orchestrator._has_pending_kernel_candidates(state)


def test_auto_workspace_materializes_installed_framework_overlay(
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint
    from quark.experimental.torch.quant_perf.workspace import sources as workspace_sources

    monkeypatch.setattr(sys, "path", list(sys.path))
    site = tmp_path / "site-packages"
    package = site / "vllm"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    monkeypatch.setattr(
        workspace_sources,
        "_package_origin",
        lambda name: package if name == "vllm" else None,
    )
    monkeypatch.setattr(
        workspace_sources,
        "_editable_project_root",
        lambda _name: None,
    )
    spec = make_spec(
        tmp_path / "session",
        framework="vllm",
        workspace_source="auto",
    )
    ckpt = Checkpoint.fresh(spec)

    manager = Orchestrator()._prepare_managed_workspaces(spec, ckpt)

    assert spec.framework_source_kind == "installed_overlay"
    assert "/workspaces/framework-overlay/source" in spec.framework_source_repo
    assert Path(spec.framework_worktree).is_dir()
    assert spec.runtime_origins["vllm"] == spec.framework_worktree
    assert spec.runtime_env["PYTHONPATH"].split(os.pathsep)[0] == (spec.framework_worktree)
    assert sys.path[0] == spec.framework_worktree
    assert spec.runtime_env["TRITON_CACHE_DIR"].startswith(str(Path(spec.session_dir) / "runtime"))
    assert "AITER_ROOT_DIR" not in spec.runtime_env
    assert ckpt.state["resolved_sources"]["framework"]["kind"] == ("installed_overlay")
    assert manager.root_dir.is_relative_to(Path(spec.session_dir))
    assert manager.init_submodules is True


def test_auto_workspace_uses_kernel_source_root_and_isolated_jit_cache(
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint
    from quark.experimental.torch.quant_perf.workspace import sources as workspace_sources

    site = tmp_path / "site-packages"
    aiter = site / "aiter"
    aiter_meta = site / "aiter_meta"
    aiter.mkdir(parents=True)
    (aiter / "__init__.py").write_text("")
    (aiter_meta / "csrc").mkdir(parents=True)
    (aiter_meta / "__init__.py").write_text("")
    (aiter_meta / "csrc" / "kernel.h").write_text("// installed metadata\n")
    monkeypatch.setattr(
        workspace_sources,
        "_package_origin",
        lambda name: {
            "aiter": aiter,
            "aiter_meta": aiter_meta,
        }.get(name),
    )
    monkeypatch.setattr(
        workspace_sources,
        "_editable_project_root",
        lambda _name: None,
    )
    spec = make_spec(
        tmp_path / "session",
        framework="vllm",
        workspace_source="auto",
    )
    ckpt = Checkpoint.fresh(spec)

    Orchestrator()._prepare_managed_workspaces(spec, ckpt)

    kernel_root = Path(spec.kernel_worktree)
    assert spec.kernel_source_kind == "installed_overlay"
    assert spec.runtime_env["AITER_ROOT_DIR"] == str(kernel_root)
    assert spec.runtime_env["AITER_META_DIR"] == str(kernel_root / "aiter_meta")
    assert spec.runtime_env["AITER_JIT_DIR"].startswith(str(Path(spec.session_dir) / "runtime"))
    assert ckpt.state["runtime_origin_evidence"]["aiter_meta"]["matched"] is True


def test_readonly_workspace_does_not_create_source_branches(tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    framework = _git_repo(tmp_path / "vllm")
    spec = make_spec(
        tmp_path / "session",
        framework="vllm",
        framework_repo=str(framework),
        workspace_source="readonly",
    )
    ckpt = Checkpoint.fresh(spec)

    Orchestrator()._prepare_managed_workspaces(spec, ckpt)

    assert spec.framework_worktree == ""
    assert spec.can_modify_framework is False
    assert _git(framework, "branch", "--format=%(refname:short)") == "main"
    assert ckpt.state["resolved_sources"]["framework"]["kind"] == "readonly"


@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_dirty_repo_is_rejected_before_quantization(mock_run_ptq, tmp_path):
    from quark.experimental.torch.quant_perf.workspace.manager import WorkspaceError

    framework = _git_repo(tmp_path / "vllm")
    (framework / "local.txt").write_text("user work\n")
    spec = make_spec(
        tmp_path / "session",
        framework_repo=str(framework),
    )

    with pytest.raises(WorkspaceError, match="commit or clean"):
        Orchestrator().run(spec)

    mock_run_ptq.assert_not_called()
    assert (framework / "local.txt").read_text() == "user work\n"


@patch("quark.experimental.torch.quant_perf.reporting.service.write_final_artifacts", return_value={})
def test_final_stage_cleans_worktrees_before_reporting(mock_reports, tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint, DeployPackage

    framework = _git_repo(tmp_path / "vllm")
    session = tmp_path / "session"
    spec = make_spec(session, framework_repo=str(framework))
    ckpt = Checkpoint.fresh(spec)
    orchestrator = Orchestrator()
    orchestrator._prepare_managed_workspaces(spec, ckpt)
    worktree = Path(spec.framework_worktree)
    (worktree / "module.py").write_text("VALUE = 2\n")
    _git(worktree, "add", "module.py")
    _git(worktree, "commit", "-m", "keep optimization")
    branch = spec.framework_branch

    result = orchestrator._cleanup_workspaces_and_generate_final_reports(
        spec,
        ckpt,
        DeployPackage(status="success", framework_branch=branch),
    )

    assert result.status == "success"
    assert not worktree.exists()
    assert _git(framework, "branch", "--show-current") == "main"
    assert branch in _git(framework, "branch", "--format=%(refname:short)").splitlines()
    assert ckpt.state["cleanup"]["status"] == "complete"
    mock_reports.assert_called_once()


@patch("quark.experimental.torch.quant_perf.reporting.service.write_final_artifacts", return_value={})
def test_final_stage_completes_cleanup_when_no_workspaces_exist(
    mock_reports,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint, DeployPackage

    spec = make_spec(tmp_path / "session", workspace_source="readonly")
    ckpt = Checkpoint.fresh(spec)

    result = Orchestrator()._cleanup_workspaces_and_generate_final_reports(
        spec,
        ckpt,
        DeployPackage(status="success"),
    )

    assert result.status == "success"
    assert ckpt.state["repo_workspaces"] == {}
    assert ckpt.state["cleanup"] == {
        "status": "complete",
        "removed": [],
        "errors": [],
    }
    mock_reports.assert_called_once()


@patch("quark.experimental.torch.quant_perf.reporting.service.write_final_artifacts", return_value={})
@patch(
    "quark.experimental.torch.quant_perf.orchestration.orchestrator.RepoWorkspaceManager.cleanup_terminal",
    side_effect=RuntimeError("cleanup failed"),
)
def test_cleanup_failure_does_not_mask_terminal_result(mock_cleanup, mock_reports, tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint, DeployPackage

    spec = make_spec(tmp_path)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["repo_workspaces"] = {"framework": {"source_repo": "/repo/vllm"}}

    result = Orchestrator()._cleanup_workspaces_and_generate_final_reports(
        spec, ckpt, DeployPackage(status="perf_below_target")
    )

    assert result.status == "perf_below_target"
    assert "cleanup failed" in ckpt.state["report_warnings"][0]
    mock_cleanup.assert_called_once()
    mock_reports.assert_called_once()


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_module_search")
def test_path_a_runs_quark_module_search(mock_search, mock_landing, mock_gate_cls, mock_tps, tmp_path):
    mock_search.return_value = "/quant_ckpt"
    mock_landing.load.return_value = fake_server()
    _good_gate_mock(mock_gate_cls)

    spec = make_spec(tmp_path, quant_strategy=None, target_gain=1.0)
    kb = MagicMock()
    kb.base_unhealthy_seen.return_value = False
    result = Orchestrator(
        experience_store=kb,
        perfopt=None,
    ).run(spec)

    mock_search.assert_called_once()
    args, _ = mock_search.call_args
    assert args[0].runtime is spec.runtime
    assert args[0].to_dict() == {
        **spec.to_dict(),
        "eval_profile": args[0].eval_profile.to_dict(),
    }
    assert args[1].state["session_id"]
    assert result.status == "success"
    assert result.quant_ckpt_dir == "/quant_ckpt"


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_accuracy_only_run_skips_performance_pipeline(
    mock_run_ptq,
    _mock_landing,
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    mock_run_ptq.return_value = {
        "status": "success",
        "quantized_model_dir": "/quant_ckpt",
    }
    _good_gate_mock(mock_gate_cls)
    perfopt = MagicMock()

    result = Orchestrator(perfopt=perfopt).run(
        make_spec(
            tmp_path,
            performance_mode="off",
            target_gain=None,
        )
    )

    assert result.status == "success"
    assert result.perf is None
    mock_tps.assert_not_called()
    perfopt.generate_optimization_candidates.assert_not_called()
    state = Checkpoint.load(tmp_path).state
    assert state["performance_status"] == "not_requested"
    assert state["stage"] == "done"


@patch(
    "quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark",
    side_effect=[1000.0, 800.0],
)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_measure_only_run_skips_perfopt_even_when_quantized_is_slower(
    mock_run_ptq,
    _mock_landing,
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    mock_run_ptq.return_value = {
        "status": "success",
        "quantized_model_dir": "/quant_ckpt",
    }
    _good_gate_mock(mock_gate_cls)
    perfopt = MagicMock()

    result = Orchestrator(perfopt=perfopt).run(
        make_spec(
            tmp_path,
            performance_mode="measure",
            target_gain=None,
        )
    )

    assert result.status == "success"
    assert result.perf is not None
    assert result.perf.gain == 0.8
    assert mock_tps.call_count == 2
    perfopt.generate_optimization_candidates.assert_not_called()
    state = Checkpoint.load(tmp_path).state
    assert state["performance_status"] == "measured"
    assert state["stage"] == "done"


@pytest.mark.parametrize(
    ("performance_mode", "target_gain"),
    [
        ("measure", None),
        ("optimize", 1.2),
    ],
)
@patch(
    "quark.experimental.torch.quant_perf.reporting.service.write_final_artifacts",
    return_value={},
)
@patch.object(
    Orchestrator,
    "_execute_checkpointed_pipeline",
    side_effect=StageError("throughput", "baseline throughput failed"),
)
@patch.object(
    Orchestrator,
    "_resolve_and_freeze_eval_profile",
    side_effect=lambda spec, _ckpt: spec,
)
def test_throughput_failure_is_not_reported_as_target_miss(
    _mock_profile,
    _mock_pipeline,
    _mock_reports,
    tmp_path,
    performance_mode,
    target_gain,
):
    result = Orchestrator().run(
        make_spec(
            tmp_path,
            performance_mode=performance_mode,
            target_gain=target_gain,
        )
    )

    assert result.status == "performance_failed"
    state = Checkpoint.load(tmp_path).state
    assert state["stage"] == "failed"
    assert state["performance_status"] == "measurement_failed"
    assert state["terminal_result"]["status"] == "performance_failed"


@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_quantize_failure_returns_accuracy_failed(
    mock_run_ptq,
    mock_gate_cls,
    tmp_path,
):
    _good_gate_mock(mock_gate_cls)
    mock_run_ptq.return_value = {"status": "failed", "agent_summary": "no safetensors written"}
    spec = make_spec(tmp_path)
    result = Orchestrator().run(spec)
    assert result.status == "accuracy_failed"
    assert "no safetensors" in result.message


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_exact_baseline_hard_failure_fail_fasts_without_recheck(
    mock_run_ptq,
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    kb = MagicMock()
    kb.baseline_hard_failure.return_value = {
        "diagnosis": "unsupported architecture",
        "failure_class": "framework",
    }
    result = Orchestrator(experience_store=kb).run(make_spec(tmp_path))

    assert result.status == "base_unhealthy"
    mock_gate_cls.return_value.check_baseline_health.assert_not_called()
    mock_run_ptq.assert_not_called()


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_recheck_baseline_bypasses_hard_failure(
    mock_run_ptq,
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    kb = MagicMock()
    kb.baseline_hard_failure.return_value = {
        "diagnosis": "unsupported architecture",
        "failure_class": "framework",
    }
    _good_gate_mock(mock_gate_cls)
    mock_run_ptq.return_value = {
        "status": "success",
        "quantized_model_dir": "/quant_ckpt",
    }

    result = Orchestrator(experience_store=kb).run(
        make_spec(
            tmp_path,
            recheck_baseline=True,
            target_gain=1.0,
        )
    )

    assert result.status == "success"
    mock_gate_cls.return_value.check_baseline_health.assert_called_once()


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_fix_base_framework_bypasses_exact_hard_failure(
    mock_run_ptq,
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    kb = MagicMock()
    kb.baseline_hard_failure.return_value = {
        "diagnosis": "unsupported architecture",
        "failure_class": "framework",
    }
    _good_gate_mock(mock_gate_cls)
    mock_run_ptq.return_value = {
        "status": "success",
        "quantized_model_dir": "/quant_ckpt",
    }

    result = Orchestrator(experience_store=kb).run(
        make_spec(
            tmp_path,
            fix_base_framework=True,
            target_gain=1.0,
        )
    )

    assert result.status == "success"
    mock_gate_cls.return_value.check_baseline_health.assert_called_once()


@pytest.mark.parametrize("fix_base_framework", [False, True])
def test_baseline_padding_oom_recovery_requires_optin(tmp_path, monkeypatch, fix_base_framework):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    monkeypatch.delenv("VLLM_ROCM_MOE_PADDING", raising=False)
    spec = make_spec(tmp_path, fix_base_framework=fix_base_framework)
    ckpt = Checkpoint.fresh(spec)
    gate = MagicMock()
    gate._source_cache = 0.8
    gate.check_baseline_health.side_effect = [
        (
            False,
            "OutOfMemoryError in _maybe_pad_weight F.pad",
            0.0,
        ),
        (True, "", 0.8),
    ]

    orchestrator = Orchestrator()
    diagnosis = orchestrator.baseline_health_stage.check_baseline_before_quantization(
        spec,
        ckpt,
        gate,
        runtime_fingerprint="before",
        framework_commit="abc",
    )

    if not fix_base_framework:
        assert "OutOfMemoryError" in diagnosis
        gate.check_baseline_health.assert_called_once()
        assert not ckpt.state["retained_runtime_env"]
        assert not ckpt.state["recovery_attempts"]
        assert ckpt.state["baseline_runtime_health"]["status"] == "unhealthy"
        return
    assert diagnosis == ""
    assert gate.check_baseline_health.call_count == 2
    assert ckpt.state["retained_runtime_env"]["VLLM_ROCM_MOE_PADDING"] == "0"
    assert ckpt.state["baseline_runtime_health"]["status"] == "healthy"
    assert ckpt.state["recovery_attempts"][0]["code"] == "rocm_moe_padding_oom"


def test_baseline_health_is_not_reused_after_fingerprint_change(
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(tmp_path)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["baseline_gsm8k"] = 0.8
    ckpt.state["baseline_runtime_health"] = {
        "status": "healthy",
        "fingerprint": "old",
    }
    gate = MagicMock()
    gate._source_cache = 0.8
    gate.check_baseline_health.return_value = (True, "", 0.8)

    orchestrator = Orchestrator()
    diagnosis = orchestrator.baseline_health_stage.check_baseline_before_quantization(
        spec,
        ckpt,
        gate,
        runtime_fingerprint="new",
        framework_commit="abc",
    )

    assert diagnosis == ""
    gate.check_baseline_health.assert_called_once()


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark")
def test_benchmark_pair_applies_bounded_padding_and_kv_recovery(
    mock_tps,
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    monkeypatch.delenv("VLLM_ROCM_MOE_PADDING", raising=False)
    mock_tps.side_effect = [
        BenchmarkFailure(
            "base",
            stdout="OutOfMemoryError in _maybe_pad_weight F.pad",
        ),
        BenchmarkFailure(
            "base",
            stdout="No available memory for the cache blocks",
        ),
        1000.0,
        1500.0,
    ]
    spec = make_spec(
        tmp_path,
        base_model="/base",
        vllm_extra_args=[
            "--tensor-parallel-size=4",
            "--gpu-memory-utilization=0.75",
        ],
    )
    ckpt = Checkpoint.fresh(spec)

    baseline, quantized, gain = _benchmark_coordinator().measure_baseline_and_quantized_throughput(
        spec,
        ckpt,
        "/quant",
    )

    assert (baseline, quantized, gain) == (1000.0, 1500.0, 1.5)
    assert [call.kwargs["gpu_memory_utilization"] for call in mock_tps.call_args_list] == [0.75, 0.75, 0.8, 0.8]
    assert [call.args[0] for call in mock_tps.call_args_list] == [
        "/base",
        "/base",
        "/base",
        "/quant",
    ]
    assert ckpt.state["retained_runtime_env"]["VLLM_ROCM_MOE_PADDING"] == "0"
    assert ckpt.state["performance_runtime"]["effective_gpu_memory_utilization"] == 0.8
    assert ckpt.state["performance_runtime"]["fingerprint"]
    assert len(ckpt.state["recovery_attempts"]) == 2


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark")
def test_quantized_recovery_restarts_complete_pair(
    mock_tps,
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    monkeypatch.delenv("VLLM_ROCM_MOE_PADDING", raising=False)
    mock_tps.side_effect = [
        1000.0,
        BenchmarkFailure(
            "quant",
            stdout="OutOfMemoryError in _maybe_pad_weight F.pad",
        ),
        900.0,
        1500.0,
    ]
    spec = make_spec(tmp_path, base_model="/base")
    ckpt = Checkpoint.fresh(spec)

    baseline, quantized, _ = _benchmark_coordinator().measure_baseline_and_quantized_throughput(
        spec,
        ckpt,
        "/quant",
    )

    assert (baseline, quantized) == (900.0, 1500.0)
    assert [call.args[0] for call in mock_tps.call_args_list] == [
        "/base",
        "/quant",
        "/base",
        "/quant",
    ]


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark")
def test_benchmark_timeout_retries_same_pair_once(
    mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    mock_tps.side_effect = [
        BenchmarkFailure(
            "/base",
            timed_out=True,
            bench_error="timeout",
        ),
        1000.0,
        1500.0,
    ]
    spec = make_spec(tmp_path, base_model="/base")
    ckpt = Checkpoint.fresh(spec)

    result = _benchmark_coordinator().measure_baseline_and_quantized_throughput(
        spec,
        ckpt,
        "/quant",
    )

    assert result == (1000.0, 1500.0, 1.5)
    assert [call.args[0] for call in mock_tps.call_args_list] == [
        "/base",
        "/base",
        "/quant",
    ]
    assert ckpt.state["recovery_attempts"][0]["code"] == "timeout"


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark")
def test_benchmark_pair_stops_on_generic_capacity_oom(
    mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    mock_tps.side_effect = BenchmarkFailure(
        "base",
        stdout="torch.OutOfMemoryError: HIP out of memory while loading weights",
    )
    spec = make_spec(tmp_path, base_model="/base")
    ckpt = Checkpoint.fresh(spec)

    with pytest.raises(StageError, match="capacity_oom") as exc_info:
        (
            _benchmark_coordinator().measure_baseline_and_quantized_throughput(
                spec,
                ckpt,
                "/quant",
            )
        )

    assert exc_info.value.stage == "throughput"
    assert len(mock_tps.call_args_list) == 1


@patch.object(
    BenchmarkCoordinator,
    "_repair_and_remeasure_throughput",
)
@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark")
def test_generic_oom_with_framework_repo_does_not_use_performance_repair(
    mock_tps,
    mock_repair,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    mock_tps.side_effect = BenchmarkFailure(
        "base",
        stdout="torch.OutOfMemoryError: HIP out of memory while loading weights",
    )
    spec = make_spec(
        tmp_path,
        base_model="/base",
        framework="vllm",
        framework_repo="/repo",
    )
    ckpt = Checkpoint.fresh(spec)

    with pytest.raises(StageError, match="capacity_oom"):
        (
            _benchmark_coordinator().measure_baseline_and_quantized_throughput(
                spec,
                ckpt,
                "/quant",
            )
        )

    mock_repair.assert_not_called()


@patch.object(
    BenchmarkCoordinator,
    "_repair_and_remeasure_throughput",
    return_value=(1000.0, 1500.0, 1.5),
)
@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark")
def test_framework_benchmark_failure_uses_performance_repair(
    mock_tps,
    mock_repair,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    mock_tps.side_effect = BenchmarkFailure(
        "base",
        stdout=("RuntimeError: The size of tensor a (4096) must match the size of tensor b (32)"),
    )
    spec = make_spec(
        tmp_path,
        base_model="/base",
        framework="vllm",
        framework_repo="/repo",
    )
    ckpt = Checkpoint.fresh(spec)

    result = _benchmark_coordinator().measure_baseline_and_quantized_throughput(
        spec,
        ckpt,
        "/quant",
    )

    assert result == (1000.0, 1500.0, 1.5)
    mock_repair.assert_called_once()


@patch.object(
    BenchmarkCoordinator,
    "_repair_and_remeasure_throughput",
    return_value=(1000.0, 1500.0, 1.5),
)
@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark")
def test_kernel_benchmark_failure_uses_aiter_repair_without_framework_repo(
    mock_tps,
    mock_repair,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    mock_tps.side_effect = BenchmarkFailure(
        "quant",
        stdout=("/managed/aiter/aiter/ops/flydsl/gemm.py\nRuntimeError: invalid device function"),
    )
    spec = make_spec(
        tmp_path,
        base_model="/base",
        framework="vllm",
        kernel_repo="/source/aiter",
        kernel_worktree="/managed/aiter",
    )
    ckpt = Checkpoint.fresh(spec)

    result = _benchmark_coordinator().measure_baseline_and_quantized_throughput(
        spec,
        ckpt,
        "/quant",
    )

    assert result == (1000.0, 1500.0, 1.5)
    mock_repair.assert_called_once()


@patch("quark.experimental.torch.quant_perf.repair.llm_repair.attempt_benchmark_repair")
@patch("quark.experimental.torch.quant_perf.pipeline.benchmarking.AccuracyGate")
def test_performance_repair_verification_reruns_accuracy_and_complete_pair(
    mock_gate_cls,
    mock_fix,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(
        tmp_path,
        framework="vllm",
        framework_repo="/repo",
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["baseline_gsm8k"] = 0.8
    ckpt.state["baseline_runtime_fingerprint"] = "baseline-runtime"
    ckpt.state["performance_runtime"] = {
        "fingerprint": "throughput-runtime",
    }
    mock_gate_cls.return_value.eval_quantized.return_value = AccuracyResult(
        gap=0.01,
        source_gsm8k=0.8,
        quantized_gsm8k=0.79,
        passed=True,
    )
    coordinator = _benchmark_coordinator()
    coordinator.measure_baseline_and_quantized_throughput = MagicMock(return_value=(1000.0, 1500.0, 1.5))

    def run_fix(**kwargs):
        ok, _ = kwargs["verify"]()
        return ok

    mock_fix.side_effect = run_fix

    result = coordinator._repair_and_remeasure_throughput(
        spec,
        ckpt,
        "/quant",
        error="shape mismatch",
        role="baseline",
        diagnosis=classify_failure(
            "shape mismatch",
            framework_repo=spec.framework_repo,
        ),
    )

    assert result == (1000.0, 1500.0, 1.5)
    mock_gate_cls.return_value.eval_quantized.assert_called_once_with("/quant")
    assert coordinator.measure_baseline_and_quantized_throughput.call_args.kwargs["allow_repair"] is False
    assert mock_fix.call_args.kwargs["runtime_fingerprint"] == "throughput-runtime"
    recovery = ckpt.state["recovery_attempts"][-1]
    assert recovery["code"] == "performance_repair"
    assert recovery["action"] == "repair"


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark")
def test_benchmark_resume_reuses_effective_runtime_and_retained_env(
    mock_tps,
    tmp_path,
    monkeypatch,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    monkeypatch.delenv("VLLM_ROCM_MOE_PADDING", raising=False)
    mock_tps.side_effect = [1000.0, 1500.0]
    spec = make_spec(
        tmp_path,
        base_model="/base",
        vllm_extra_args=["--gpu-memory-utilization=0.75"],
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["performance_runtime"] = {
        "effective_gpu_memory_utilization": 0.85,
    }
    ckpt.state["retained_runtime_env"] = {
        "VLLM_ROCM_MOE_PADDING": "0",
    }
    ckpt.save()

    result = _benchmark_coordinator().measure_baseline_and_quantized_throughput(
        spec,
        ckpt,
        "/quant",
    )

    assert result == (1000.0, 1500.0, 1.5)
    assert [call.kwargs["gpu_memory_utilization"] for call in mock_tps.call_args_list] == [0.85, 0.85]
    assert os.environ["VLLM_ROCM_MOE_PADDING"] == "0"
    assert ckpt.state["performance_runtime"]["effective_gpu_memory_utilization"] == 0.85


@patch(
    "quark.experimental.torch.quant_perf.reporting.service.write_final_artifacts",
    return_value={
        "final_json": "/s/reports/final.json",
        "final_md": "/s/reports/final.md",
        "session_breakdown_json": "/s/session_breakdown.json",
        "session_report_md": "/s/session_report.md",
    },
)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_module_search")
def test_search_stage_error_reports_terminal_result_and_cleans_worktree(
    mock_search,
    mock_gate_cls,
    mock_write_reports,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    framework = _git_repo(tmp_path / "vllm")
    session = tmp_path / "session"
    spec = make_spec(
        session,
        quant_strategy=None,
        framework_repo=str(framework),
    )
    _good_gate_mock(mock_gate_cls)
    mock_search.side_effect = StageError(
        "quantize",
        "mix_precision_search produced no candidate",
    )

    result = Orchestrator().run(spec)

    state = Checkpoint.load(session).state
    worktree = Path(state["repo_workspaces"]["framework"]["integration_path"])
    assert result.status == "accuracy_failed"
    assert "produced no candidate" in result.message
    assert result.report_status == "complete"
    assert state["stage"] == "failed"
    assert state["terminal_result"]["status"] == "accuracy_failed"
    assert state["cleanup"]["status"] == "complete"
    assert not worktree.exists()
    mock_write_reports.assert_called_once()


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_accuracy_failure_returns_before_landing_and_benchmark(
    mock_run_ptq, mock_landing, mock_gate_cls, mock_tps, tmp_path
):
    mock_run_ptq.return_value = {"status": "success", "quantized_model_dir": "/quant_ckpt"}
    mock_gate_cls.return_value._source_cache = 0.8
    mock_gate_cls.return_value.check_baseline_health.return_value = (True, "", 0.8)
    mock_gate_cls.return_value.eval_quantized.return_value = AccuracyResult(
        gap=0.2, source_gsm8k=0.8, quantized_gsm8k=0.6, passed=False
    )

    spec = make_spec(tmp_path, accuracy_gap=0.02)
    result = Orchestrator().run(spec)

    assert result.status == "accuracy_failed"
    assert "0.2" in result.message
    mock_landing.load.assert_not_called()
    mock_tps.assert_not_called()


@patch(
    "quark.experimental.torch.quant_perf.repair.llm_repair.classify_accuracy_failure",
    return_value="framework",
)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_accuracy_repair_crash_keeps_stage_at_land(
    mock_run_ptq,
    mock_gate_cls,
    _mock_classify,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    framework = _git_repo(tmp_path / "vllm")
    session = tmp_path / "session"
    mock_run_ptq.return_value = {
        "status": "success",
        "quantized_model_dir": "/quant_ckpt",
    }
    mock_gate_cls.return_value._source_cache = 0.8
    mock_gate_cls.return_value.check_baseline_health.return_value = (
        True,
        "",
        0.8,
    )
    mock_gate_cls.return_value.eval_quantized.return_value = AccuracyResult(
        gap=1.0,
        source_gsm8k=0.8,
        quantized_gsm8k=0.0,
        passed=False,
    )
    repair_service = MagicMock()
    repair_service.repair.side_effect = RuntimeError("repair crashed")
    spec = make_spec(
        session,
        accuracy_gap=0.02,
        framework_repo=str(framework),
    )

    with pytest.raises(RuntimeError, match="repair crashed"):
        Orchestrator(repair_service=repair_service).run(spec)

    assert Checkpoint.load(session).state["stage"] == "land"


@patch(
    "quark.experimental.torch.quant_perf.repair.llm_repair.classify_accuracy_failure",
    return_value="framework",
)
@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_accuracy_repair_verifies_with_full_gate_and_reuses_result(
    mock_run_ptq,
    mock_gate_cls,
    _mock_tps,
    _mock_classify,
    tmp_path,
):
    framework = _git_repo(tmp_path / "vllm")
    mock_run_ptq.return_value = {
        "status": "success",
        "quantized_model_dir": "/quant_ckpt",
    }
    gate = mock_gate_cls.return_value
    gate._source_cache = 0.8
    gate.check_baseline_health.return_value = (True, "", 0.8)
    failed = AccuracyResult(
        gap=1.0,
        source_gsm8k=0.8,
        quantized_gsm8k=0.0,
        passed=False,
    )
    repaired = AccuracyResult(
        gap=0.0,
        source_gsm8k=0.8,
        quantized_gsm8k=0.81,
        passed=True,
    )
    gate.eval_quantized.side_effect = [failed, repaired]
    repair_service = MagicMock()

    def repair(request):
        assert callable(request.verifier)
        passed, failure = request.verifier()
        assert passed is True
        assert failure == ""
        return MagicMock(status="fixed")

    repair_service.repair.side_effect = repair

    result = Orchestrator(repair_service=repair_service).run(
        make_spec(
            tmp_path,
            framework_repo=str(framework),
            accuracy_gap=0.03,
            target_gain=1.0,
        )
    )

    assert result.status == "success"
    assert gate.eval_quantized.call_count == 2


@patch(
    "quark.experimental.torch.quant_perf.repair.llm_repair.classify_accuracy_failure",
    return_value="framework",
)
@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_accuracy_repair_reruns_gate_after_stale_failed_candidate_result(
    mock_run_ptq,
    mock_gate_cls,
    _mock_tps,
    _mock_classify,
    tmp_path,
):
    framework = _git_repo(tmp_path / "vllm")
    mock_run_ptq.return_value = {
        "status": "success",
        "quantized_model_dir": "/quant_ckpt",
    }
    gate = mock_gate_cls.return_value
    gate._source_cache = 0.8
    gate.check_baseline_health.return_value = (True, "", 0.8)
    failed = AccuracyResult(
        gap=1.0,
        source_gsm8k=0.8,
        quantized_gsm8k=0.0,
        passed=False,
    )
    repaired = AccuracyResult(
        gap=0.0,
        source_gsm8k=0.8,
        quantized_gsm8k=0.81,
        passed=True,
    )
    gate.eval_quantized.side_effect = [failed, failed, repaired]
    repair_service = MagicMock()

    def repair(request):
        passed, _failure = request.verifier()
        assert passed is False
        return MagicMock(status="fixed")

    repair_service.repair.side_effect = repair

    result = Orchestrator(repair_service=repair_service).run(
        make_spec(
            tmp_path,
            framework_repo=str(framework),
            accuracy_gap=0.03,
            target_gain=1.0,
        )
    )

    assert result.status == "success"
    assert gate.eval_quantized.call_count == 3


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_quant_only_below_target_without_perfopt_is_not_success(
    mock_run_ptq,
    mock_landing,
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    mock_run_ptq.return_value = {"status": "success", "quantized_model_dir": "/quant_ckpt"}
    _good_gate_mock(mock_gate_cls)

    spec = make_spec(tmp_path, target_gain=1.2)
    result = Orchestrator(perfopt=None).run(spec)

    assert result.status == "perf_below_target"
    assert result.quant_ckpt_dir == "/quant_ckpt"
    assert result.perf is None
    assert mock_tps.call_count == 2
    mock_landing.load.assert_not_called()
    state = Checkpoint.load(tmp_path).state
    assert state["stage"] == "perf_failed"
    assert state["terminal_result"]["status"] == "perf_below_target"


@patch("quark.experimental.torch.quant_perf.reporting.service.write_final_artifacts", return_value={})
@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", side_effect=[1000.0, 1500.0])
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_success_persists_accuracy_and_performance_attempts(
    mock_run_ptq,
    mock_landing,
    mock_gate_cls,
    mock_tps,
    mock_reports,
    tmp_path,
):
    mock_run_ptq.return_value = {
        "status": "success",
        "quantized_model_dir": "/quant_ckpt",
    }
    mock_gate_cls.return_value._source_cache = 0.85
    mock_gate_cls.return_value.check_baseline_health.return_value = (
        True,
        "",
        0.85,
    )
    mock_gate_cls.return_value.eval_quantized.return_value = AccuracyResult(
        gap=(0.85 - 0.84) / 0.85,
        source_gsm8k=0.85,
        quantized_gsm8k=0.84,
        passed=True,
    )

    result = Orchestrator(perfopt=None).run(
        make_spec(
            tmp_path,
            target_gain=1.4,
            vllm_extra_args=["--gpu-memory-utilization=0.75"],
        )
    )

    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    state = Checkpoint.load(tmp_path).state
    assert result.status == "success"
    assert state["baseline_runtime_health"]["status"] == "healthy"
    assert state["baseline_gsm8k"] == 0.85
    accuracy_attempt = state["accuracy_attempts"][0]
    assert accuracy_attempt["attempt"] == 1
    assert accuracy_attempt["baseline"] == 0.85
    assert accuracy_attempt["quantized"] == 0.84
    assert accuracy_attempt["gap"] == (0.85 - 0.84) / 0.85
    assert accuracy_attempt["passed"] is True
    assert accuracy_attempt["profile_hash"] == state["eval_profile_hash"]
    assert accuracy_attempt["ts"]
    performance_attempt = state["performance_measurements"][0]
    assert performance_attempt["attempt"] == 1
    assert performance_attempt["role"] == "quant_only"
    assert performance_attempt["baseline_tps"] == 1000.0
    assert performance_attempt["quantized_tps"] == 1500.0
    assert performance_attempt["gain"] == 1.5
    assert performance_attempt["isl"] == 128
    assert performance_attempt["osl"] == 128
    assert performance_attempt["concurrency"] == 64
    assert performance_attempt["ts"]
    assert [call.kwargs["gpu_memory_utilization"] for call in mock_tps.call_args_list[:2]] == [0.75, 0.75]
    actions = [event["action"] for event in state["phase_timeline"]]
    assert "quantization" in actions
    assert "accuracy_gate" in actions
    assert "throughput_benchmark" in actions
    assert "terminal_decision" in actions
    assert actions[-1] == "final"


def test_freeze_eval_profile_reuses_state_profile(tmp_path):
    from quark.experimental.torch.quant_perf.evaluation.profile import (
        EVAL_PROFILE_RESOLVER_VERSION,
        eval_profile_input_hash,
    )
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(tmp_path)
    ckpt = Checkpoint.fresh(spec)
    state_profile = EvalProfile(
        profile_id="state-profile",
        profile_hash="",
        model_mode="base",
        apply_chat_template=False,
        enable_thinking=None,
        detection_reason="fallback_base",
    ).with_computed_hash()
    ckpt.state["eval_profile"] = state_profile.to_dict()
    ckpt.state["eval_profile_hash"] = state_profile.profile_hash
    ckpt.state["eval_profile_resolver_version"] = EVAL_PROFILE_RESOLVER_VERSION
    ckpt.state["eval_profile_input_hash"] = eval_profile_input_hash(
        spec.base_model,
        discovery=spec.eval_discovery,
        allow_llm=spec.eval_allow_llm,
        overrides={
            "task": spec.eval_task,
            "num_fewshot": spec.eval_num_fewshot,
            "prompting_strategy": spec.eval_prompting_strategy,
            "enable_thinking": None,
            "max_gen_toks": spec.eval_max_gen_toks,
        },
    )

    spec = Orchestrator()._resolve_and_freeze_eval_profile(spec, ckpt)

    assert spec.eval_profile == state_profile
    assert ckpt.state["eval_profile_hash"] == state_profile.profile_hash


def test_freeze_eval_profile_recomputes_stale_hash(tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(tmp_path)
    ckpt = Checkpoint.fresh(spec)
    stored = spec.eval_profile.to_dict()
    stored["profile_hash"] = "stale"
    ckpt.state["eval_profile"] = stored
    ckpt.state["eval_profile_hash"] = "stale"

    spec = Orchestrator()._resolve_and_freeze_eval_profile(spec, ckpt)

    assert spec.eval_profile.profile_hash != "stale"
    assert ckpt.state["eval_profile_hash"] == spec.eval_profile.profile_hash


def test_freeze_eval_profile_resolves_new_profile_before_gpu_work(
    monkeypatch,
    tmp_path,
):
    import quark.experimental.torch.quant_perf.orchestration.orchestrator as orchestrator_module
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(
        tmp_path,
        eval_profile=None,
        eval_discovery="online",
        eval_allow_llm=False,
        eval_task="gsm8k_cot_zeroshot",
        eval_num_fewshot=0,
        eval_prompting_strategy="cot",
        eval_thinking_mode="disabled",
        eval_max_gen_toks=768,
    )
    ckpt = Checkpoint.fresh(spec)
    calls = []
    resolved = EvalProfile(
        profile_id="resolved-v2",
        profile_hash="",
        model_mode="chat",
        apply_chat_template=True,
        enable_thinking=False,
        detection_reason="chat_template",
        schema_version=2,
        policy_version="quark-quant-perf-gsm8k-profile-v2",
        task="gsm8k_cot_zeroshot",
        num_fewshot=0,
        prompting_strategy="cot",
        max_gen_toks=768,
        settings_source="user",
    ).with_computed_hash()

    def resolve(model_ref, **kwargs):
        calls.append((model_ref, kwargs))
        return resolved

    monkeypatch.setattr(
        orchestrator_module,
        "resolve_eval_profile",
        resolve,
        raising=False,
    )

    spec = Orchestrator()._resolve_and_freeze_eval_profile(spec, ckpt)

    assert calls == [
        (
            spec.base_model,
            {
                "discovery": "online",
                "allow_llm": False,
                "overrides": {
                    "task": "gsm8k_cot_zeroshot",
                    "num_fewshot": 0,
                    "prompting_strategy": "cot",
                    "enable_thinking": False,
                    "max_gen_toks": 768,
                },
                "artifact_dir": Path(spec.session_dir) / "evaluation" / "profile",
            },
        )
    ]
    assert spec.eval_profile == resolved


def test_stale_eval_profile_refreshes_active_session_and_invalidates_accuracy(
    monkeypatch,
    tmp_path,
):
    import quark.experimental.torch.quant_perf.orchestration.orchestrator as orchestrator_module
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(tmp_path)
    ckpt = Checkpoint.fresh(spec)
    old_profile = spec.eval_profile
    refreshed = EvalProfile(
        profile_id="refreshed-profile",
        profile_hash="",
        model_mode="chat",
        apply_chat_template=True,
        enable_thinking=False,
        detection_reason="chat_template",
        schema_version=2,
        policy_version="quark-quant-perf-gsm8k-profile-v2",
    ).with_computed_hash()
    ckpt.state.update(
        {
            "stage": "perfopt",
            "quant_ckpt_dir": str(tmp_path / "quant_ckpt"),
            "eval_profile": old_profile.to_dict(),
            "eval_profile_hash": old_profile.profile_hash,
            "eval_profile_resolver_version": -1,
            "eval_profile_input_hash": "stale",
            "baseline_gsm8k": 0.8,
            "baseline_reference": {"score": 0.8},
            "baseline_runtime_health": {"status": "healthy"},
            "accuracy_validation": {"passed": True},
            "accuracy_attempts": [{"passed": True}],
        }
    )
    monkeypatch.setattr(
        orchestrator_module,
        "resolve_eval_profile",
        lambda *args, **kwargs: refreshed,
    )

    Orchestrator()._resolve_and_freeze_eval_profile(spec, ckpt)

    assert ckpt.state["eval_profile"] == refreshed.to_dict()
    assert ckpt.state["eval_profile_resolver_version"] > 0
    assert ckpt.state["eval_profile_input_hash"] != "stale"
    assert ckpt.state["eval_profile_history"][-1]["profile"] == old_profile.to_dict()
    assert ckpt.state["baseline_gsm8k"] is None
    assert ckpt.state["baseline_reference"] is None
    assert ckpt.state["baseline_runtime_health"] is None
    assert ckpt.state["accuracy_validation"] is None
    assert ckpt.state["accuracy_attempts"] == []
    assert ckpt.state["quant_ckpt_dir"] == str(tmp_path / "quant_ckpt")
    assert ckpt.state["stage"] == "land"


def test_stale_eval_profile_rejects_perfopt_only_retry(tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint, StageError

    spec = make_spec(tmp_path, retry_perfopt=True)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "perf_failed",
            "eval_profile_resolver_version": -1,
            "eval_profile_input_hash": "stale",
        }
    )

    with pytest.raises(
        StageError,
        match="--retry-accuracy-gate",
    ):
        Orchestrator()._resolve_and_freeze_eval_profile(spec, ckpt)


def test_stale_eval_profile_keeps_terminal_session_immutable(
    monkeypatch,
    tmp_path,
):
    import quark.experimental.torch.quant_perf.orchestration.orchestrator as orchestrator_module
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(tmp_path)
    ckpt = Checkpoint.fresh(spec)
    stored = spec.eval_profile.to_dict()
    ckpt.state.update(
        {
            "stage": "done",
            "terminal_result": {"status": "success"},
            "eval_profile": stored,
            "eval_profile_resolver_version": -1,
            "eval_profile_input_hash": "stale",
        }
    )
    resolve = MagicMock(side_effect=AssertionError("terminal profile must remain immutable"))
    monkeypatch.setattr(orchestrator_module, "resolve_eval_profile", resolve)

    restored = Orchestrator()._resolve_and_freeze_eval_profile(spec, ckpt)

    assert restored.eval_profile.to_dict() == stored
    assert ckpt.state["eval_profile_resolver_version"] == -1
    assert ckpt.state["eval_profile_history"] == []
    resolve.assert_not_called()


def test_baseline_reference_rejects_different_gsm8k_sample_count(tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(tmp_path, gsm8k_num_samples=50)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["baseline_gsm8k"] = 0.85
    ckpt.state["baseline_reference"] = {
        "score": 0.85,
        "profile_hash": spec.eval_profile.profile_hash,
        "gsm8k_num_samples": 200,
        "base_model": str(Path(spec.base_model).resolve()),
        "runtime_fingerprint": "runtime-50",
        "source": "measured",
    }

    assert not BaselineHealthStage.baseline_reference_matches_current_runtime(
        spec,
        ckpt,
        "runtime-50",
    )


def test_baseline_reference_accepts_exact_measurement_profile(tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(tmp_path, gsm8k_num_samples=50)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["baseline_gsm8k"] = 0.85
    ckpt.state["baseline_reference"] = {
        "score": 0.85,
        "profile_hash": spec.eval_profile.profile_hash,
        "gsm8k_num_samples": 50,
        "base_model": str(Path(spec.base_model).resolve()),
        "runtime_fingerprint": "runtime-50",
        "source": "measured",
    }

    assert BaselineHealthStage.baseline_reference_matches_current_runtime(
        spec,
        ckpt,
        "runtime-50",
    )


# The apply/validate/retain step is the real-GPU part; mock it to return the
# (retained_patches, retained_srcs, final_gain, final_gap) verdict so the flow
# tests stay GPU-free. final_gain is now the authoritative number the verdict
# uses.
@patch(
    "quark.experimental.torch.quant_perf.pipeline.retention.CandidateRetentionService.validate_and_retain_candidates",
    return_value=([], [], 1.05, 0.01),
)
@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_perf_below_target_after_perfopt_runs(
    mock_run_ptq, mock_landing, mock_gate_cls, mock_tps, mock_retain, tmp_path
):
    mock_run_ptq.return_value = {"status": "success", "quantized_model_dir": "/quant_ckpt"}
    mock_landing.load.return_value = fake_server()
    _good_gate_mock(mock_gate_cls)
    perfopt = MagicMock()
    perfopt.generate_optimization_candidates.return_value = PerfResult(
        patches=["/p.diff"],
        gain=1.05,
    )

    spec = make_spec(tmp_path, target_gain=1.2)
    result = Orchestrator(perfopt=perfopt).run(spec)

    # no patch cleared the floor -> real final_gain (1.05) < target -> below target
    assert result.status == "perf_below_target"
    assert result.perf.gain == 1.05
    assert result.applied_patches == []


@patch(
    "quark.experimental.torch.quant_perf.pipeline.benchmarking.BenchmarkCoordinator.measure_final_stack_with_abba",
    return_value=(1000.0, 1300.0, 1.3),
)
@patch(
    "quark.experimental.torch.quant_perf.pipeline.retention.CandidateRetentionService.validate_and_retain_candidates",
    return_value=(["/p.diff"], ["/src"], 1.3, 0.01),
)
@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_success_with_perfopt_meeting_target(
    mock_run_ptq, mock_landing, mock_gate_cls, mock_tps, mock_retain, mock_final_pair, tmp_path
):
    mock_run_ptq.return_value = {"status": "success", "quantized_model_dir": "/quant_ckpt"}
    mock_landing.load.return_value = fake_server()
    _good_gate_mock(mock_gate_cls)
    perfopt = MagicMock()
    perfopt.generate_optimization_candidates.return_value = PerfResult(
        patches=["/p.diff"],
        patch_srcs=["/src"],
        gain=1.3,
    )

    spec = make_spec(tmp_path, target_gain=1.2)
    result = Orchestrator(perfopt=perfopt).run(spec)

    # retained patch pushed real final_gain (1.3) >= target -> success
    assert result.status == "success"
    assert result.perf.gain == 1.3
    assert result.applied_patches == ["/p.diff"]


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_kernel_repo_gets_independent_work_branch(
    mock_run_ptq,
    mock_landing,
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    mock_run_ptq.return_value = {
        "status": "success",
        "quantized_model_dir": "/quant_ckpt",
    }
    _good_gate_mock(mock_gate_cls)
    framework_repo = _git_repo(tmp_path / "vllm")
    kernel_repo = _git_repo(tmp_path / "aiter")
    session = tmp_path / "session"
    spec = make_spec(
        session,
        framework_repo=str(framework_repo),
        kernel_repo=str(kernel_repo),
        target_gain=1.0,
    )

    result = Orchestrator(perfopt=None).run(spec)

    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    ckpt = Checkpoint.load(session)
    assert result.status == "success"
    assert ckpt.state["fw_original_branch"] == "main"
    assert ckpt.state["kernel_original_branch"] == "main"
    assert (
        ckpt.state["repo_workspaces"]["framework"]["integration_path"]
        != (ckpt.state["repo_workspaces"]["kernel"]["integration_path"])
    )
    assert ckpt.state["repo_workspaces"]["framework"]["branch_retained"] is False
    assert ckpt.state["repo_workspaces"]["kernel"]["branch_retained"] is False
    assert _git(framework_repo, "branch", "--show-current") == "main"
    assert _git(kernel_repo, "branch", "--show-current") == "main"


@patch("quark.experimental.torch.quant_perf.pipeline.retention.commit_selected_changes")
@patch("quark.experimental.torch.quant_perf.pipeline.retention.reset_hard_to")
@patch(
    "quark.experimental.torch.quant_perf.pipeline.retention.get_head_sha",
    return_value="base-sha",
)
@patch(
    "quark.experimental.torch.quant_perf.pipeline.retention.apply_prepared_kernel_patch",
    return_value=(True, "ok"),
)
@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline", return_value=0.8)
@patch(
    "quark.experimental.torch.quant_perf.evaluation.throughput.measure_throughput",
    side_effect=[
        _throughput_measurement(100.0),
        _throughput_measurement(120.0),
        _throughput_measurement(100.0),
    ],
)
def test_apply_validate_retain_applies_patch_to_recorded_kernel_repo(
    mock_tps,
    mock_gsm8k,
    mock_apply,
    mock_head,
    mock_reset,
    mock_commit,
    tmp_path,
):
    patch_file = tmp_path / "patch.diff"
    patch_file.write_text("diff --git a/kernel.py b/kernel.py\n")
    kernel_repo = str(tmp_path / "aiter")
    kernel_src = str(tmp_path / "aiter" / "kernel.py")
    perf = PerfResult(
        patches=[str(patch_file)],
        patch_srcs=[kernel_src],
        patch_repos=[kernel_repo],
        gain=1.2,
    )
    spec = make_spec(
        tmp_path,
        framework_repo="/source/vllm",
        framework_worktree=str(tmp_path / "vllm"),
        kernel_repo="/source/aiter",
        kernel_worktree=kernel_repo,
        keep_floor=0.01,
    )
    gate = MagicMock()
    gate._source_cache = 0.8
    ckpt = MagicMock()
    ckpt.state = {
        "baseline_tps": 100.0,
        "baseline_gsm8k": 0.8,
        "best_accuracy_gap": 0.01,
        "retain_trials": [],
        "kernel_journey": [],
        "geak_patches": [
            {
                "kernel_sig": "k1",
                "status": "candidate",
                "best_patch": str(patch_file),
                "verified_speedup": 1.2,
                "patch_changed_files": ["kernel.py"],
                "patch_requires_rebuild": False,
            }
        ],
    }

    retained, retained_srcs, gain, gap = CandidateRetentionService().validate_and_retain_candidates(
        perf,
        spec,
        1.0,
        "/quant_ckpt",
        gate,
        ckpt,
    )

    assert not (Path.cwd() / "MagicMock").exists()
    assert retained == [str(patch_file)]
    assert retained_srcs == [kernel_src]
    assert perf.patch_repos == [kernel_repo]
    assert mock_apply.call_count == 2
    mock_apply.assert_any_call(str(patch_file), kernel_repo)
    assert mock_reset.call_count == 2
    mock_reset.assert_called_with(kernel_repo, "base-sha")
    mock_commit.assert_called_once_with(
        kernel_repo,
        ["kernel.py"],
        "Quark Quant-Perf kernel patch (retained, gain 1.000->1.200x)",
    )
    trial = ckpt.state["retain_trials"][0]
    assert trial["attempt"] == 1
    assert trial["ts"]
    assert trial["kernel_id"] == "k1"
    assert trial["patch"] == str(patch_file)
    assert trial["source"] == kernel_src
    assert trial["repo"] == kernel_repo
    assert trial["accuracy_score"] == 0.8
    assert trial["accuracy_gap"] == 0.0
    assert trial["throughput_tps"] == 120.0
    assert trial["gain"] == 1.2
    assert trial["decision"] == "KEEP"
    assert trial["reason"] == "improved_past_keep_floor"
    assert ckpt.state["geak_patches"][0]["status"] == "retained"
    ckpt.record_phase_event.assert_called_with(
        "retain_patch",
        "keep",
        kernel_id="k1",
        gain=1.2,
        accuracy_gap=0.0,
        reason="improved_past_keep_floor",
    )
    assert ckpt.state["retention_stack"] == [
        {
            "kind": "patch",
            "candidate": "k1",
            "patch": str(patch_file),
            "source": kernel_src,
            "repo": kernel_repo,
            "repo_before_sha": "base-sha",
            "repo_after_sha": "base-sha",
            "changed_files": ["kernel.py"],
            "requires_rebuild": False,
            "gain_before": 1.0,
            "gain_after": 1.2,
            "accuracy_gap_before": 0.01,
            "accuracy_gap_after": 0.0,
            "retain_trial_attempt": 1,
            "active": True,
        }
    ]


@patch("quark.experimental.torch.quant_perf.pipeline.retention.rebuild_framework")
@patch("quark.experimental.torch.quant_perf.pipeline.retention.reset_hard_to")
def test_final_abba_reconciles_active_retention_prefix(
    mock_reset,
    mock_rebuild,
    monkeypatch,
    tmp_path,
):
    make_spec(tmp_path, keep_floor=0.01)
    service = CandidateRetentionService()

    def patch_entry(name, before, after, gain):
        return {
            "kind": "patch",
            "candidate": name,
            "patch": f"/{name}.diff",
            "source": f"/{name}.py",
            "repo": "/aiter",
            "repo_before_sha": before,
            "repo_after_sha": after,
            "changed_files": [f"{name}.py"],
            "requires_rebuild": False,
            "gain_before": 1.2,
            "gain_after": gain,
            "accuracy_gap_before": 0.01,
            "accuracy_gap_after": 0.01,
            "retain_trial_attempt": 1 if name == "k1" else 2,
            "active": True,
        }

    passing_ckpt = MagicMock()
    passing_ckpt.state = {
        "retention_stack": [patch_entry("k1", "base", "k1", 1.3)],
        "retain_trials": [{"attempt": 1, "patch": "/k1.diff", "decision": "KEEP", "active": True}],
        "kernel_journey": [],
        "geak_patches": [],
    }
    passing_perf = PerfResult(
        patches=["/k1.diff"],
        patch_srcs=["/k1.py"],
        patch_repos=["/aiter"],
        gain=1.3,
    )
    passing_measure = MagicMock(return_value=(100.0, 130.0, 1.3, 0.01))

    gain, gap = service.reconcile_final_stack(
        passing_perf,
        passing_ckpt,
        quant_gain=1.2,
        quant_gap=0.02,
        measure_final_stack=passing_measure,
    )

    assert gain == 1.3
    assert gap == 0.01
    assert passing_perf.patches == ["/k1.diff"]
    mock_reset.assert_not_called()

    prefix_ckpt = MagicMock()
    prefix_ckpt.state = {
        "retention_stack": [
            patch_entry("k1", "base", "k1", 1.3),
            patch_entry("k2", "k1", "k2", 1.31),
        ],
        "retain_trials": [
            {"attempt": 1, "patch": "/k1.diff", "decision": "KEEP", "active": True},
            {"attempt": 2, "patch": "/k2.diff", "decision": "KEEP", "active": True},
        ],
        "kernel_journey": [],
        "geak_patches": [
            {"best_patch": "/k1.diff"},
            {"best_patch": "/k2.diff"},
        ],
    }
    prefix_perf = PerfResult(
        patches=["/k1.diff", "/k2.diff"],
        patch_srcs=["/k1.py", "/k2.py"],
        patch_repos=["/aiter", "/aiter"],
        gain=1.31,
    )
    prefix_measure = MagicMock(
        side_effect=[
            (100.0, 120.5, 1.205, 0.01),
            (100.0, 130.0, 1.3, 0.01),
        ]
    )

    gain, gap = service.reconcile_final_stack(
        prefix_perf,
        prefix_ckpt,
        quant_gain=1.2,
        quant_gap=0.02,
        measure_final_stack=prefix_measure,
    )

    assert gain == 1.3
    assert gap == 0.01
    assert prefix_perf.patches == ["/k1.diff"]
    assert prefix_ckpt.state["retain_trials"][1]["active"] is False
    assert prefix_ckpt.state["retain_trials"][1]["final_decision"] == "REVERTED"
    assert prefix_ckpt.state["geak_patches"][1]["retention_status"] == "reverted"
    mock_reset.assert_called_once_with("/aiter", "k1")

    runtime_key = "AITER_CONFIG_GEMM_A4W4"
    monkeypatch.setenv(runtime_key, "/candidate.csv")
    quant_only_ckpt = MagicMock()
    quant_only_ckpt.state = {
        "retained_runtime_env": {runtime_key: "/candidate.csv"},
        "retention_stack": [
            {
                "kind": "runtime",
                "candidate": "vendor_gemm",
                "runtime_env_before": {},
                "runtime_env_after": {runtime_key: "/candidate.csv"},
                "runtime_snapshot": {runtime_key: None},
                "runtime_artifacts_before": {},
                "runtime_artifacts_after": {"tuned_csv": "/candidate.csv"},
                "gain_before": 1.2,
                "gain_after": 1.21,
                "accuracy_gap_before": 0.02,
                "accuracy_gap_after": 0.02,
                "active": True,
            }
        ],
        "retain_trials": [],
        "kernel_journey": [],
        "geak_patches": [],
    }
    quant_only_perf = PerfResult(
        patches=[],
        gain=1.21,
        runtime_env={runtime_key: "/candidate.csv"},
        runtime_artifacts={"tuned_csv": "/candidate.csv"},
    )
    quant_only_measure = MagicMock(
        side_effect=[
            (100.0, 120.5, 1.205, 0.01),
            (100.0, 120.0, 1.2, 0.01),
        ]
    )

    gain, gap = service.reconcile_final_stack(
        quant_only_perf,
        quant_only_ckpt,
        quant_gain=1.2,
        quant_gap=0.02,
        measure_final_stack=quant_only_measure,
    )

    assert gain == 1.2
    assert gap == 0.02
    assert quant_only_perf.runtime_env == {}
    assert quant_only_perf.runtime_artifacts == {}
    assert runtime_key not in os.environ
    assert quant_only_ckpt.state["retained_runtime_env"] == {}
    assert quant_only_ckpt.state["retention_stack"][0]["active"] is False
    mock_rebuild.assert_not_called()


@patch("quark.experimental.torch.quant_perf.pipeline.retention.apply_prepared_kernel_patch")
@patch(
    "quark.experimental.torch.quant_perf.pipeline.retention.get_head_sha",
    return_value="base-sha",
)
def test_installed_overlay_rejects_compiled_kernel_patch(
    mock_head,
    mock_apply,
    tmp_path,
):
    patch_file = tmp_path / "compiled.diff"
    patch_file.write_text("diff --git a/kernel.cpp b/kernel.cpp\n--- a/kernel.cpp\n+++ b/kernel.cpp\n")
    repo = str(tmp_path / "aiter-overlay")
    perf = PerfResult(
        patches=[str(patch_file)],
        patch_srcs=[str(tmp_path / "aiter-overlay" / "kernel.cpp")],
        patch_repos=[repo],
        gain=1.2,
    )
    spec = make_spec(
        tmp_path,
        kernel_repo=repo,
        kernel_worktree=repo,
        kernel_source_kind="installed_overlay",
    )
    gate = MagicMock()
    gate._source_cache = 0.8
    ckpt = MagicMock()
    ckpt.state = {
        "baseline_tps": 100.0,
        "baseline_gsm8k": 0.8,
        "best_accuracy_gap": 0.01,
        "retain_trials": [],
        "kernel_journey": [],
        "geak_patches": [
            {
                "kernel_sig": "k1",
                "status": "candidate",
                "best_patch": str(patch_file),
                "verified_speedup": 1.2,
                "patch_changed_files": ["kernel.cpp"],
                "patch_requires_rebuild": True,
            }
        ],
    }

    retained, _, gain, _ = CandidateRetentionService().validate_and_retain_candidates(
        perf,
        spec,
        1.0,
        "/quant_ckpt",
        gate,
        ckpt,
    )

    assert retained == []
    assert gain == 1.0
    mock_apply.assert_not_called()
    assert ckpt.state["retain_trials"][0]["reason"] == ("source_repo_required")


@patch("quark.experimental.torch.quant_perf.pipeline.retention.commit_selected_changes")
@patch("quark.experimental.torch.quant_perf.pipeline.retention.reset_hard_to")
@patch(
    "quark.experimental.torch.quant_perf.pipeline.retention.get_head_sha",
    return_value="base-sha",
)
@patch(
    "quark.experimental.torch.quant_perf.pipeline.retention.apply_prepared_kernel_patch",
    return_value=(True, "ok"),
)
@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline", return_value=0.8)
@patch(
    "quark.experimental.torch.quant_perf.evaluation.throughput.measure_throughput",
    side_effect=[
        _throughput_measurement(100.0),
        _throughput_measurement(102.5),
        _throughput_measurement(103.0),
        _throughput_measurement(100.5),
    ],
)
def test_apply_validate_retain_uses_abba_for_marginal_candidate(
    mock_measure,
    mock_gsm8k,
    mock_apply,
    mock_head,
    mock_reset,
    mock_commit,
    tmp_path,
):
    patch_file = tmp_path / "marginal.diff"
    patch_file.write_text("diff --git a/kernel.py b/kernel.py\n")
    repo = str(tmp_path / "aiter")
    src = str(tmp_path / "aiter" / "kernel.py")
    perf = PerfResult(
        patches=[str(patch_file)],
        patch_srcs=[src],
        patch_repos=[repo],
        gain=1.0,
    )
    spec = make_spec(
        tmp_path,
        kernel_repo="/source/aiter",
        kernel_worktree=repo,
        keep_floor=0.01,
    )
    gate = MagicMock()
    gate._source_cache = 0.8
    ckpt = MagicMock()
    ckpt.state = {
        "baseline_tps": 100.0,
        "baseline_gsm8k": 0.8,
        "best_accuracy_gap": 0.0,
        "retain_trials": [],
        "kernel_journey": [],
        "bottleneck_analysis": {"candidates": [{"kernel_id": "k-marginal", "rank": 5}]},
        "geak_patches": [
            {
                "kernel_sig": "k-marginal",
                "status": "candidate",
                "best_patch": str(patch_file),
                "verified_speedup": 1.08,
                "patch_changed_files": ["kernel.py"],
                "patch_requires_rebuild": False,
            }
        ],
    }

    retained, _, gain, _ = CandidateRetentionService().validate_and_retain_candidates(
        perf,
        spec,
        1.0,
        "/quant",
        gate,
        ckpt,
    )

    expected = ((102.5 * 103.0) / (100.0 * 100.5)) ** 0.5
    assert retained == [str(patch_file)]
    assert gain == pytest.approx(expected)
    assert mock_apply.call_count == 2
    trial = ckpt.state["retain_trials"][0]
    assert trial["mode"] == "abba"
    assert len(trial["candidate_measurements"]) == 2


@patch("quark.experimental.torch.quant_perf.pipeline.retention.reset_hard_to")
@patch(
    "quark.experimental.torch.quant_perf.pipeline.retention.get_head_sha",
    return_value="base-sha",
)
@patch(
    "quark.experimental.torch.quant_perf.pipeline.retention.apply_prepared_kernel_patch",
    return_value=(True, "ok"),
)
@patch(
    "quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline",
    side_effect=[
        EvaluationFailure(
            "/quant",
            stderr="hipErrorStreamCaptureInvalidated",
        ),
        EvaluationFailure(
            "/quant",
            stderr="hipErrorStreamCaptureInvalidated",
        ),
    ],
)
@patch(
    "quark.experimental.torch.quant_perf.evaluation.throughput.measure_throughput",
    return_value=_throughput_measurement(100.0),
)
def test_graph_incompatible_candidate_is_needs_review_not_drop(
    mock_measure,
    mock_gsm8k,
    mock_apply,
    mock_head,
    mock_reset,
    tmp_path,
):
    patch_file = tmp_path / "graph.diff"
    patch_file.write_text("diff --git a/kernel.py b/kernel.py\n")
    repo = str(tmp_path / "aiter")
    src = str(tmp_path / "aiter" / "kernel.py")
    perf = PerfResult(
        patches=[str(patch_file)],
        patch_srcs=[src],
        patch_repos=[repo],
        gain=1.0,
    )
    spec = make_spec(
        tmp_path,
        kernel_repo="/source/aiter",
        kernel_worktree=repo,
    )
    gate = MagicMock()
    gate._source_cache = 0.8
    ckpt = MagicMock()
    ckpt.state = {
        "baseline_tps": 100.0,
        "baseline_gsm8k": 0.8,
        "best_accuracy_gap": 0.0,
        "retain_trials": [],
        "kernel_journey": [],
        "bottleneck_analysis": {"candidates": [{"kernel_id": "k-graph", "rank": 1}]},
        "geak_patches": [
            {
                "kernel_sig": "k-graph",
                "status": "candidate",
                "best_patch": str(patch_file),
                "verified_speedup": None,
                "patch_changed_files": ["kernel.py"],
                "patch_requires_rebuild": False,
            }
        ],
    }

    retained, _, gain, _ = CandidateRetentionService().validate_and_retain_candidates(
        perf,
        spec,
        1.0,
        "/quant",
        gate,
        ckpt,
    )

    assert retained == []
    assert gain == 1.0
    trial = ckpt.state["retain_trials"][0]
    assert trial["decision"] == "NEEDS_REVIEW"
    assert trial["reason"] == "graph_incompatible"
    assert trial["accuracy_score"] is None
    assert mock_gsm8k.call_count == 2
    assert mock_measure.call_count == 1
    mock_reset.assert_called_once()


def test_apply_validate_retain_rejects_user_source_repo_target(tmp_path):
    from quark.experimental.torch.quant_perf.workspace.manager import WorkspaceError

    patch_file = tmp_path / "patch.diff"
    patch_file.write_text("diff --git a/kernel.py b/kernel.py\n")
    source_repo = str(tmp_path / "aiter")
    perf = PerfResult(
        patches=[str(patch_file)],
        patch_srcs=[str(tmp_path / "aiter" / "kernel.py")],
        patch_repos=[source_repo],
        gain=1.1,
    )
    spec = make_spec(tmp_path, kernel_repo=source_repo)
    ckpt = MagicMock()
    ckpt.state = {"best_accuracy_gap": 0.01}

    with pytest.raises(WorkspaceError, match="integration worktree"):
        CandidateRetentionService().validate_and_retain_candidates(
            perf,
            spec,
            1.0,
            "/quant_ckpt",
            MagicMock(),
            ckpt,
        )


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_server_always_stopped_even_on_perf_fail(mock_run_ptq, mock_landing, mock_gate_cls, mock_tps, tmp_path):
    mock_run_ptq.return_value = {"status": "success", "quantized_model_dir": "/quant_ckpt"}
    server = fake_server()
    mock_landing.load.return_value = server
    _good_gate_mock(mock_gate_cls)
    perfopt = MagicMock()
    perfopt.generate_optimization_candidates.return_value = PerfResult(
        patches=[],
        gain=0.5,
    )

    Orchestrator(perfopt=perfopt).run(make_spec(tmp_path, target_gain=1.2))

    server.stop.assert_called_once()


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_resume_skips_quantize_when_accuracy_fingerprint_matches(
    mock_run_ptq, mock_landing, mock_gate_cls, mock_tps, tmp_path
):
    """A prior run already validated 'land' -- resume must not re-run quantize."""
    mock_landing.load.return_value = fake_server()
    _good_gate_mock(mock_gate_cls)

    spec = make_spec(tmp_path, target_gain=1.0)
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    ckpt = Checkpoint.fresh(spec)
    ckpt.state["quant_ckpt_dir"] = "/prior/quant_ckpt"
    _mark_accuracy_valid(ckpt, spec, "/prior/quant_ckpt")
    ckpt.save()

    result = Orchestrator(perfopt=None).run(spec)

    mock_run_ptq.assert_not_called()
    assert result.status == "success"
    assert result.quant_ckpt_dir == "/prior/quant_ckpt"


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
def test_resume_rechecks_accuracy_when_tp_changes(
    mock_gate_cls,
    _mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    measured = make_spec(
        tmp_path,
        target_gain=1.0,
        vllm_extra_args=["--tensor-parallel-size=4"],
    )
    resumed = make_spec(
        tmp_path,
        target_gain=1.0,
        vllm_extra_args=["--tensor-parallel-size=8"],
    )
    ckpt = Checkpoint.fresh(measured)
    ckpt.state["quant_ckpt_dir"] = "/prior/quant_ckpt"
    _mark_accuracy_valid(ckpt, measured, "/prior/quant_ckpt")
    ckpt.save()
    _good_gate_mock(mock_gate_cls)

    result = Orchestrator(perfopt=None).run(resumed)

    assert result.status == "success"
    mock_gate_cls.return_value.eval_quantized.assert_called_once_with("/prior/quant_ckpt")


def test_resume_of_already_done_run_is_idempotent(tmp_path):
    spec = make_spec(tmp_path)
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    ckpt = Checkpoint.fresh(spec)
    ckpt.state["stage"] = "done"
    ckpt.state["quant_ckpt_dir"] = "/prior/quant_ckpt"
    ckpt.save()

    result = Orchestrator().run(spec)

    assert result.status == "success"
    assert result.quant_ckpt_dir == "/prior/quant_ckpt"


@patch("quark.experimental.torch.quant_perf.reporting.service.write_final_artifacts")
def test_completed_report_resume_preserves_terminal_progress(mock_write_reports, tmp_path):
    from quark.experimental.torch.quant_perf.session.progress import read_progress, write_progress
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(tmp_path)
    report_paths = {}
    for name in (
        "final_json",
        "final_md",
        "session_breakdown_json",
        "session_report_md",
    ):
        path = tmp_path / f"{name}.txt"
        path.write_text("ok")
        report_paths[name] = str(path)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "done",
            "terminal_stage": "done",
            "quant_ckpt_dir": "/prior/quant_ckpt",
            "terminal_result": {
                "status": "success",
                "quant_ckpt_dir": "/prior/quant_ckpt",
                "perf": None,
                "message": "",
                "applied_patches": [],
                "framework_branch": "",
                "original_branch": "",
                "revert_command": "",
                "kernel_branch": "",
                "kernel_original_branch": "",
                "kernel_revert_command": "",
                "report_status": "complete",
                "report_paths": report_paths,
            },
            "reporting": {
                "status": "complete",
                "paths": report_paths,
                "error": "",
            },
        }
    )
    ckpt.save()
    write_progress(tmp_path, stage="done", report_status="complete")

    result = Orchestrator().run(spec)

    assert result.status == "success"
    assert read_progress(tmp_path)["stage"] == "done"
    mock_write_reports.assert_not_called()


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_module_search")
@pytest.mark.parametrize("retry", [False, True])
@pytest.mark.parametrize("matching_baseline", [False, True])
def test_retry_failed_search_without_candidate_preserves_matching_baseline(
    mock_search, mock_gate_cls, mock_tps, tmp_path, retry, matching_baseline
):
    from quark.experimental.torch.quant_perf.runtime.recovery import build_runtime_fingerprint

    spec = make_spec(tmp_path, quant_strategy=None, retry_accuracy_gate=retry, performance_mode="off", target_gain=None)
    ckpt = Checkpoint.fresh(spec)
    fingerprint = build_runtime_fingerprint(spec, framework_commit="", runtime_env=os.environ, stage="baseline_health")
    ckpt.state.update(
        stage="failed",
        terminal_stage="failed",
        terminal_result={"status": "accuracy_failed", "message": "missing search dependency"},
        baseline_gsm8k=0.8,
        baseline_runtime_health={"status": "healthy", "fingerprint": fingerprint if matching_baseline else "old"},
        baseline_reference={
            "score": 0.8,
            "profile_hash": spec.eval_profile.profile_hash,
            "gsm8k_num_samples": spec.gsm8k_num_samples,
            "base_model": str(Path(spec.base_model).resolve()),
            "runtime_fingerprint": fingerprint,
            "source": "measured",
        },
    )
    ckpt.state["mix_precision_search"].update(status="failed", termination_reason="search_execution_failed")
    ckpt.save()
    _good_gate_mock(mock_gate_cls)
    mock_search.side_effect = StageError("quantize", "next search failure", code="search_execution_failed")

    result = Orchestrator().run(spec)

    assert result.status == "accuracy_failed"
    state = Checkpoint.load(tmp_path).state
    assert state["stage"] == "failed"
    assert state["reporting"]["status"] == "complete"
    assert state["baseline_gsm8k"] == 0.8
    if retry:
        mock_search.assert_called_once()
        assert "next search failure" in result.message
        assert any(event["status"] == "search_restarted" for event in state["phase_timeline"])
    else:
        mock_search.assert_not_called()
        assert result.message == "missing search dependency"
    if retry and not matching_baseline:
        mock_gate_cls.return_value.check_baseline_health.assert_called_once()
    else:
        mock_gate_cls.return_value.check_baseline_health.assert_not_called()
    mock_tps.assert_not_called()


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_module_search")
@pytest.mark.parametrize("export_started", [False, True])
@pytest.mark.parametrize("failed_before_search", [False, True])
def test_retry_accuracy_gate_reexports_missing_saved_winner(
    mock_search,
    mock_gate_cls,
    mock_tps,
    tmp_path,
    export_started,
    failed_before_search,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(
        tmp_path,
        quant_strategy=None,
        retry_accuracy_gate=True,
        target_gain=1.0,
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "failed",
            "terminal_stage": "failed",
            "terminal_result": {
                "status": "accuracy_failed",
                "quant_ckpt_dir": str(tmp_path / "missing-quant"),
            },
            "quant_ckpt_dir": str(tmp_path / "missing-quant") if export_started else None,
            "best_candidate": {
                "linear_attn_mode": "mxfp4",
                "self_attn_mode": "mxfp4",
                "mlp_mode": "mxfp4",
                "kv_cache_mode": "native",
            },
            "best_accuracy_gap": 0.01,
            "baseline_gsm8k": 0.8,
        }
    )
    ckpt.state["mix_precision_search"].update(
        {
            "status": "completed",
            "candidate_queue": [
                {
                    "config": ckpt.state["best_candidate"],
                    "status": "accepted",
                }
            ],
            "candidate_cursor": 0,
        }
    )
    if not export_started:
        ckpt.state["best_candidate"] = None
        ckpt.state["terminal_result"]["quant_ckpt_dir"] = ""
    if failed_before_search:
        ckpt.state["quant_ckpt_dir"] = None
        ckpt.state["best_candidate"] = None
        ckpt.state["terminal_result"]["quant_ckpt_dir"] = ""
        ckpt.state["mix_precision_search"].update(
            status="failed", candidate_queue=[], termination_reason="search_execution_failed"
        )
    ckpt.save()
    mock_search.return_value = "/reexported/quant"
    _good_gate_mock(mock_gate_cls)
    gate = mock_gate_cls.return_value
    gate.eval_quantized.return_value = AccuracyResult(
        gap=0.01,
        source_gsm8k=0.8,
        quantized_gsm8k=0.792,
        passed=True,
        artifacts={
            "baseline": "/eval/baseline",
            "quantized": "/eval/quantized",
        },
    )

    result = Orchestrator(perfopt=None).run(spec)

    assert result.status == "success"
    assert result.quant_ckpt_dir == "/reexported/quant"
    mock_search.assert_called_once()


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_retry_accuracy_gate_recovers_completed_direct_ptq_artifacts(
    mock_run_ptq,
    mock_landing,
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(
        tmp_path,
        retry_accuracy_gate=True,
        target_gain=1.0,
    )
    quant_dir = Path(spec.quant_ckpt_dir)
    write_valid_quant_checkpoint(quant_dir)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "failed",
            "terminal_stage": "failed",
            "terminal_result": {
                "status": "accuracy_failed",
                "quant_ckpt_dir": "",
                "message": "quantization did not produce valid artifacts",
            },
            "quant_ckpt_dir": None,
            "baseline_gsm8k": 0.8,
        }
    )
    ckpt.save()
    _good_gate_mock(mock_gate_cls)

    result = Orchestrator(perfopt=None).run(spec)

    assert result.status == "success"
    assert result.quant_ckpt_dir == str(quant_dir)
    assert Checkpoint.load(tmp_path).state["quant_ckpt_dir"] == str(quant_dir)
    mock_run_ptq.assert_not_called()
    mock_landing.load.assert_not_called()
    assert mock_tps.call_count == 2


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_module_search")
def test_retry_accuracy_gate_recovers_completed_search_artifacts_when_state_path_is_missing(
    mock_search,
    mock_gate_cls,
    _mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    winner = {
        "self_attn_mode": "fp8",
        "mlp_mode": "mxfp4",
        "kv_cache_mode": "native",
    }
    fallback = {
        "self_attn_mode": "fp8",
        "mlp_mode": "fp8",
        "kv_cache_mode": "native",
    }
    spec = make_spec(
        tmp_path,
        quant_strategy=None,
        retry_accuracy_gate=True,
        target_gain=1.0,
    )
    quant_dir = Path(spec.quant_ckpt_dir)
    write_valid_quant_checkpoint(quant_dir)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "failed",
            "path": "mix_precision_search",
            "terminal_stage": "failed",
            "terminal_result": {
                "status": "accuracy_failed",
                "quant_ckpt_dir": "",
            },
            "quant_ckpt_dir": None,
            "best_candidate": winner,
            "accuracy_attempts": [
                {
                    "baseline": 0.8,
                    "quantized": 0.0,
                    "gap": 1.0,
                    "passed": False,
                    "candidate": winner,
                }
            ],
            "baseline_gsm8k": 0.8,
        }
    )
    ckpt.state["mix_precision_search"].update(
        {
            "status": "completed",
            "candidate_queue": [
                {"config": winner, "status": "rejected"},
                {"config": fallback, "status": "pending"},
            ],
            "candidate_cursor": 1,
        }
    )
    ckpt.save()
    mock_search.return_value = str(quant_dir)
    _good_gate_mock(mock_gate_cls)

    orchestrator = Orchestrator(perfopt=None)
    orchestrator._prepare_managed_workspaces = MagicMock()
    result = orchestrator.run(spec)

    assert result.status == "success"
    assert result.quant_ckpt_dir == str(quant_dir)
    mock_search.assert_not_called()
    orchestrator._prepare_managed_workspaces.assert_called_once()
    assert Checkpoint.load(tmp_path).state["best_candidate"] == winner


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_module_search")
def test_real_accuracy_failure_automatically_exports_next_search_candidate(
    mock_search,
    mock_gate_cls,
    _mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    winner = {
        "mlp_mode": "mxfp4",
        "kv_cache_mode": "native",
    }
    fallback = {
        "mlp_mode": "fp8",
        "kv_cache_mode": "native",
    }
    quant_dir = tmp_path / "quant_ckpt"
    quant_dir.mkdir()
    spec = make_spec(
        tmp_path,
        quant_strategy=None,
        target_gain=1.0,
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["mix_precision_search"].update(
        {
            "status": "completed",
            "candidate_queue": [
                {"config": winner, "status": "exported"},
                {"config": fallback, "status": "pending"},
            ],
            "candidate_cursor": 0,
        }
    )
    ckpt.save()

    def export_current_candidate(_spec, checkpoint):
        search_state = checkpoint.state["mix_precision_search"]
        entry = search_state["candidate_queue"][search_state["candidate_cursor"]]
        checkpoint.state["best_candidate"] = dict(entry["config"])
        checkpoint.state["quant_ckpt_dir"] = str(quant_dir)
        checkpoint.save()
        return str(quant_dir)

    mock_search.side_effect = export_current_candidate
    mock_gate_cls.return_value._source_cache = 0.8
    mock_gate_cls.return_value.check_baseline_health.return_value = (
        True,
        "",
        0.8,
    )
    mock_gate_cls.return_value.eval_quantized.side_effect = [
        AccuracyResult(
            gap=0.25,
            source_gsm8k=0.8,
            quantized_gsm8k=0.6,
            passed=False,
        ),
        AccuracyResult(
            gap=0.01,
            source_gsm8k=0.8,
            quantized_gsm8k=0.792,
            passed=True,
        ),
    ]

    result = Orchestrator(perfopt=None).run(spec)

    assert result.status == "success"
    assert mock_search.call_count == 2
    state = Checkpoint.load(tmp_path).state
    search_state = state["mix_precision_search"]
    assert search_state["candidate_cursor"] == 1
    assert search_state["candidate_queue"][0]["status"] == "rejected"
    assert search_state["candidate_queue"][1]["status"] == "accepted"
    assert state["best_candidate"] == fallback


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_module_search")
def test_unrepaired_load_failure_automatically_exports_next_search_candidate(
    mock_search,
    mock_gate_cls,
    _mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    winner = {
        "self_attn_mode": "fp8",
        "mlp_mode": "mxfp4",
        "kv_cache_mode": "native",
    }
    fallback = {
        "self_attn_mode": "native",
        "mlp_mode": "mxfp4",
        "kv_cache_mode": "native",
    }
    quant_dir = tmp_path / "quant_ckpt"
    quant_dir.mkdir()
    spec = make_spec(
        tmp_path,
        quant_strategy=None,
        target_gain=1.0,
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["mix_precision_search"].update(
        {
            "status": "completed",
            "candidate_queue": [
                {"config": winner, "status": "exported"},
                {"config": fallback, "status": "pending"},
            ],
            "candidate_cursor": 0,
        }
    )
    ckpt.save()

    def export_current_candidate(_spec, checkpoint):
        search_state = checkpoint.state["mix_precision_search"]
        entry = search_state["candidate_queue"][search_state["candidate_cursor"]]
        checkpoint.state["best_candidate"] = dict(entry["config"])
        checkpoint.state["quant_ckpt_dir"] = str(quant_dir)
        checkpoint.save()
        return str(quant_dir)

    mock_search.side_effect = export_current_candidate
    mock_gate_cls.return_value._source_cache = 0.8
    mock_gate_cls.return_value.check_baseline_health.return_value = (
        True,
        "",
        0.8,
    )
    orchestrator = Orchestrator(perfopt=None)
    orchestrator._prepare_managed_workspaces = MagicMock()
    orchestrator.accuracy_stage.evaluate_quantized_checkpoint = MagicMock(
        side_effect=[
            StageError(
                "land",
                "quantized accuracy repair failed: fused scale missing",
                code="managed_source_failure",
                diagnostic="KeyError: wk_weights_proj.input_scale",
            ),
            AccuracyResult(
                gap=0.01,
                source_gsm8k=0.8,
                quantized_gsm8k=0.792,
                passed=True,
            ),
        ]
    )

    result = orchestrator.run(spec)

    assert result.status == "success"
    assert mock_search.call_count == 2
    state = Checkpoint.load(tmp_path).state
    search_state = state["mix_precision_search"]
    assert search_state["candidate_cursor"] == 1
    assert search_state["candidate_queue"][0]["status"] == "rejected"
    assert search_state["candidate_queue"][0]["real_gate"]["reason"] == ("quantized_accuracy_load_failed")
    assert search_state["candidate_queue"][1]["status"] == "accepted"
    assert state["best_candidate"] == fallback


def test_search_candidate_rejection_excludes_transient_failures(tmp_path):
    spec = make_spec(tmp_path)

    assert (
        Orchestrator._should_reject_search_candidate_for(
            StageError(
                "land",
                "HSA_STATUS_ERROR_OUT_OF_RESOURCES",
                code="hsa_out_of_resources",
            ),
            spec,
        )
        is False
    )
    assert (
        Orchestrator._should_reject_search_candidate_for(
            StageError(
                "land",
                "KeyError: wk_weights_proj.input_scale",
                code="managed_source_failure",
            ),
            spec,
        )
        is True
    )


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_module_search")
def test_retry_accuracy_gate_rechecks_promoted_accuracy_repair_once(
    mock_search,
    mock_gate_cls,
    _mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    quant_dir = tmp_path / "quant_ckpt"
    quant_dir.mkdir()
    winner = {
        "linear_attn_mode": "fp8",
        "self_attn_mode": "fp8",
        "mlp_mode": "mxfp4_fp8",
        "kv_cache_mode": "native",
    }
    spec = make_spec(
        tmp_path,
        quant_strategy=None,
        retry_accuracy_gate=True,
        target_gain=1.0,
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "failed",
            "terminal_stage": "failed",
            "terminal_result": {
                "status": "accuracy_failed",
                "quant_ckpt_dir": str(quant_dir),
            },
            "quant_ckpt_dir": str(quant_dir),
            "best_candidate": winner,
            "accuracy_attempts": [
                {
                    "baseline": 0.8,
                    "quantized": 0.0,
                    "gap": 1.0,
                    "passed": False,
                    "candidate": winner,
                }
            ],
            "repair_journey": [
                {
                    "failure_class": "accuracy_gap",
                    "status": "fixed",
                    "candidate_id": "accuracy-gap-fix",
                }
            ],
            "baseline_gsm8k": 0.8,
        }
    )
    ckpt.save()
    _good_gate_mock(mock_gate_cls)

    result = Orchestrator(perfopt=None).run(spec)

    assert result.status == "success"
    mock_search.assert_not_called()
    state = Checkpoint.load(tmp_path).state
    assert state["post_repair_rechecked_configs"] == [winner]


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_module_search")
@patch(
    "quark.experimental.torch.quant_perf.quantize.search._flydsl_dense_mxfp4_available",
    return_value=False,
)
def test_retry_accuracy_gate_reuses_repairable_w4a8_checkpoint(
    _mock_dense_available,
    mock_search,
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    quant_dir = tmp_path / "quant_ckpt"
    quant_dir.mkdir()
    winner = {
        "linear_attn_mode": "mxfp4_fp8",
        "self_attn_mode": "mxfp4_fp8",
        "mlp_mode": "mxfp4",
        "kv_cache_mode": "native",
    }
    spec = make_spec(
        tmp_path,
        quant_strategy=None,
        retry_accuracy_gate=True,
        target_gain=1.0,
        mxfp4_gemm_backend="flydsl",
        w4a8_gemm_backend="flydsl",
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "failed",
            "path": "mix_precision_search",
            "terminal_stage": "failed",
            "terminal_result": {
                "status": "accuracy_failed",
                "quant_ckpt_dir": str(quant_dir),
                "message": (
                    "land: quantized accuracy load failed: "
                    "W4A8 FlyDSL backend is unavailable in the "
                    "installed AITER build"
                ),
            },
            "quant_ckpt_dir": str(quant_dir),
            "best_candidate": winner,
            "best_accuracy_gap": 0.0,
            "baseline_gsm8k": 0.8,
        }
    )
    ckpt.save()
    _good_gate_mock(mock_gate_cls)

    result = Orchestrator(perfopt=None).run(spec)

    assert result.status == "success"
    mock_search.assert_not_called()
    mock_gate_cls.return_value.eval_quantized.assert_called_once_with(str(quant_dir))


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_module_search")
@pytest.mark.parametrize(
    ("terminal_stage", "terminal_status"),
    [("failed", "accuracy_failed"), ("perf_failed", "perf_below_target"), ("done", "success")],
)
def test_retry_accuracy_gate_reuses_checkpoint_after_terminal_result(
    mock_search,
    mock_gate_cls,
    _mock_tps,
    tmp_path,
    terminal_stage,
    terminal_status,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    failed = {
        "linear_attn_mode": "fp8",
        "self_attn_mode": "fp8",
        "mlp_mode": "mxfp4_fp8",
        "kv_cache_mode": "native",
    }
    fallback = {
        "linear_attn_mode": "ptpc_fp8",
        "self_attn_mode": "fp8",
        "mlp_mode": "mxfp4",
        "kv_cache_mode": "native",
    }
    quant_dir = tmp_path / "quant_ckpt"
    quant_dir.mkdir()
    spec = make_spec(
        tmp_path,
        quant_strategy=None,
        retry_accuracy_gate=True,
        target_gain=1.0,
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": terminal_stage,
            "terminal_stage": terminal_stage,
            "terminal_result": {
                "status": terminal_status,
                "quant_ckpt_dir": str(quant_dir),
                "message": (
                    "land: quantized accuracy repair failed: Unsupported kernel config for moe heuristic dispatch"
                ),
            },
            "quant_ckpt_dir": str(quant_dir),
            "best_candidate": failed,
            "baseline_gsm8k": 0.8,
        }
    )
    ckpt.state["mix_precision_search"].update(
        {
            "status": "completed",
            "candidate_queue": [
                {"config": failed, "status": "exported"},
                {"config": fallback, "status": "pending"},
            ],
            "candidate_cursor": 0,
        }
    )
    ckpt.save()
    mock_search.return_value = str(quant_dir)
    _good_gate_mock(mock_gate_cls)

    result = Orchestrator(perfopt=None).run(spec)

    assert result.status == "success"
    mock_search.assert_not_called()
    mock_gate_cls.return_value.eval_quantized.assert_called_once_with(str(quant_dir))
    state = Checkpoint.load(tmp_path).state
    search_state = state["mix_precision_search"]
    assert search_state["candidate_cursor"] == 0
    assert search_state["candidate_queue"][0]["status"] == "accepted"
    assert state["best_candidate"] == failed


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_module_search")
def test_retry_accuracy_gate_reuses_checkpoint_when_failure_was_other_candidate(
    mock_search,
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    quant_dir = tmp_path / "quant_ckpt"
    quant_dir.mkdir()
    rejected_winner = {
        "linear_attn_mode": "mxfp4",
        "self_attn_mode": "mxfp4",
        "mlp_mode": "mxfp4",
        "kv_cache_mode": "native",
    }
    current_winner = {
        "linear_attn_mode": "mxfp4_fp8",
        "self_attn_mode": "mxfp4_fp8",
        "mlp_mode": "mxfp4",
        "kv_cache_mode": "native",
    }
    spec = make_spec(
        tmp_path,
        quant_strategy=None,
        retry_accuracy_gate=True,
        target_gain=1.0,
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "failed",
            "terminal_stage": "failed",
            "terminal_result": {
                "status": "accuracy_failed",
                "quant_ckpt_dir": str(quant_dir),
            },
            "quant_ckpt_dir": str(quant_dir),
            "best_candidate": current_winner,
            "accuracy_attempts": [
                {
                    "baseline": 0.795,
                    "quantized": 0.615,
                    "gap": 0.226,
                    "passed": False,
                    "candidate": rejected_winner,
                }
            ],
            "baseline_gsm8k": 0.795,
        }
    )
    ckpt.state["mix_precision_search"].update(
        {
            "status": "completed",
            "candidate_queue": [
                {"config": rejected_winner, "status": "rejected"},
                {"config": current_winner, "status": "exported"},
            ],
            "candidate_cursor": 1,
        }
    )
    ckpt.save()
    _good_gate_mock(mock_gate_cls)
    gate = mock_gate_cls.return_value
    gate.eval_quantized.return_value = AccuracyResult(
        gap=0.01,
        source_gsm8k=0.795,
        quantized_gsm8k=0.787,
        passed=True,
    )

    result = Orchestrator(perfopt=None).run(spec)

    assert result.status == "success"
    assert result.quant_ckpt_dir == str(quant_dir)
    mock_search.assert_not_called()
    state = Checkpoint.load(tmp_path).state
    assert state["mix_precision_search"]["candidate_cursor"] == 1
    assert state["phase_timeline"][-1]["action"] != "quantization"


def test_real_accuracy_attempt_records_candidate_identity(tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    candidate = {
        "linear_attn_mode": "mxfp4_fp8",
        "self_attn_mode": "mxfp4_fp8",
        "mlp_mode": "mxfp4",
        "kv_cache_mode": "native",
    }
    spec = make_spec(tmp_path, target_gain=1.0)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "land",
            "quant_ckpt_dir": "/quant",
            "best_candidate": candidate,
            "baseline_gsm8k": 0.8,
        }
    )
    ckpt.save()
    gate = MagicMock()
    gate._source_cache = 0.8
    gate.check_baseline_health.return_value = (True, "", 0.8)
    gate.eval_quantized.return_value = AccuracyResult(
        gap=0.01,
        source_gsm8k=0.8,
        quantized_gsm8k=0.792,
        passed=True,
        artifacts={
            "baseline": "/eval/baseline",
            "quantized": "/eval/quantized",
        },
    )
    orchestrator = Orchestrator(perfopt=None)

    with (
        patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate", return_value=gate),
        patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0),
    ):
        result = orchestrator.run(spec)

    assert result.status == "success"
    attempt = Checkpoint.load(tmp_path).state["accuracy_attempts"][-1]
    assert attempt["candidate"] == candidate
    assert attempt["artifacts"] == {
        "baseline": "/eval/baseline",
        "quantized": "/eval/quantized",
    }


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
def test_perf_failed_retries_when_new_kernel_repo_is_added(
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    kernel_repo = _git_repo(tmp_path / "aiter")
    session = tmp_path / "session"
    spec = make_spec(session, kernel_repo=str(kernel_repo), target_gain=1.0)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "perf_failed",
            "quant_ckpt_dir": "/prior/quant_ckpt",
            "baseline_gsm8k": 0.8,
            "best_accuracy_gap": 0.01,
        }
    )
    ckpt.save()
    _good_gate_mock(mock_gate_cls)

    result = Orchestrator(perfopt=None).run(spec)

    assert result.status == "success"
    assert mock_tps.call_count == 2
    reloaded = Checkpoint.load(session)
    assert reloaded.state["kernel_repo"] == str(kernel_repo)


def test_perfopt_resume_reuses_matching_quant_only_pair(tmp_path):
    from quark.experimental.torch.quant_perf.runtime.recovery import build_runtime_fingerprint
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(
        tmp_path,
        vllm_extra_args=["--tensor-parallel-size=4"],
        isl=1024,
        osl=1024,
        bench_concurrency=64,
    )
    ckpt = Checkpoint.fresh(spec)
    orchestrator = Orchestrator()
    fingerprint = build_runtime_fingerprint(
        spec,
        framework_commit=orchestrator._framework_commit(spec),
        runtime_env=os.environ,
        stage="throughput",
        effective_gpu_memory_utilization=(spec.vllm_gpu_memory_utilization),
    )
    ckpt.state.update(
        {
            "stage": "perfopt",
            "performance_runtime": {
                "effective_gpu_memory_utilization": (spec.vllm_gpu_memory_utilization),
                "runtime_env": {},
                "fingerprint": fingerprint,
            },
            "performance_measurements": [
                {
                    "role": "quant_only",
                    "baseline_tps": 1000.0,
                    "quantized_tps": 1760.0,
                    "gain": 1.76,
                    "isl": 1024,
                    "osl": 1024,
                    "concurrency": 64,
                }
            ],
        }
    )

    assert (_benchmark_coordinator(orchestrator).reuse_matching_quantized_throughput_pair(spec, ckpt)) == (
        1000.0,
        1760.0,
        1.76,
    )


def test_perfopt_resume_rejects_pair_from_different_tp(tmp_path):
    from quark.experimental.torch.quant_perf.runtime.recovery import build_runtime_fingerprint
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    measured_spec = make_spec(
        tmp_path,
        vllm_extra_args=["--tensor-parallel-size=4"],
    )
    resumed_spec = make_spec(
        tmp_path,
        vllm_extra_args=["--tensor-parallel-size=8"],
    )
    ckpt = Checkpoint.fresh(measured_spec)
    orchestrator = Orchestrator()
    ckpt.state.update(
        {
            "stage": "perfopt",
            "performance_runtime": {
                "effective_gpu_memory_utilization": (measured_spec.vllm_gpu_memory_utilization),
                "runtime_env": {},
                "fingerprint": build_runtime_fingerprint(
                    measured_spec,
                    framework_commit=orchestrator._framework_commit(measured_spec),
                    runtime_env=os.environ,
                    stage="throughput",
                    effective_gpu_memory_utilization=(measured_spec.vllm_gpu_memory_utilization),
                ),
            },
            "performance_measurements": [
                {
                    "role": "quant_only",
                    "baseline_tps": 1000.0,
                    "quantized_tps": 1760.0,
                    "gain": 1.76,
                    "isl": measured_spec.isl,
                    "osl": measured_spec.osl,
                    "concurrency": measured_spec.bench_concurrency,
                }
            ],
        }
    )

    assert (
        _benchmark_coordinator(orchestrator).reuse_matching_quantized_throughput_pair(
            resumed_spec,
            ckpt,
        )
        is None
    )


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
def test_perfopt_resume_does_not_remeasure_matching_pair(
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.runtime.recovery import build_runtime_fingerprint
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(
        tmp_path,
        target_gain=2.0,
        vllm_extra_args=["--tensor-parallel-size=4"],
    )
    ckpt = Checkpoint.fresh(spec)
    orchestrator = Orchestrator(perfopt=None)
    orchestrator._prepare_managed_workspaces(spec, ckpt)
    runtime_env = orchestrator._effective_runtime_env(spec)
    framework_commit = orchestrator._framework_commit(spec)
    ckpt.state.update(
        {
            "stage": "perfopt",
            "quant_ckpt_dir": "/prior/quant",
            "baseline_gsm8k": 0.85,
            "best_accuracy_gap": 0.0,
            "baseline_runtime_health": {
                "status": "healthy",
                "fingerprint": build_runtime_fingerprint(
                    spec,
                    framework_commit=framework_commit,
                    runtime_env=runtime_env,
                    stage="baseline_health",
                ),
            },
            "performance_runtime": {
                "effective_gpu_memory_utilization": (spec.vllm_gpu_memory_utilization),
                "runtime_env": {},
                "fingerprint": build_runtime_fingerprint(
                    spec,
                    framework_commit=framework_commit,
                    runtime_env=runtime_env,
                    stage="throughput",
                    effective_gpu_memory_utilization=(spec.vllm_gpu_memory_utilization),
                ),
            },
            "performance_measurements": [
                {
                    "attempt": 1,
                    "role": "quant_only",
                    "baseline_tps": 1733.0,
                    "quantized_tps": 3056.0,
                    "gain": 1.763,
                    "isl": spec.isl,
                    "osl": spec.osl,
                    "concurrency": spec.bench_concurrency,
                }
            ],
        }
    )
    baseline_fingerprint = ckpt.state["baseline_runtime_health"]["fingerprint"]
    ckpt.state["baseline_reference"] = {
        "score": 0.85,
        "profile_hash": spec.eval_profile.profile_hash,
        "gsm8k_num_samples": spec.gsm8k_num_samples,
        "base_model": str(Path(spec.base_model).resolve()),
        "runtime_fingerprint": baseline_fingerprint,
        "source": "measured",
    }
    ckpt.state["accuracy_validation"] = {
        "fingerprint": build_accuracy_fingerprint(
            spec,
            "/prior/quant",
            framework_commit=orchestrator._framework_commit(spec),
            kernel_commit=orchestrator._kernel_commit(spec),
            runtime_env=runtime_env,
        ),
        "profile_hash": spec.eval_profile.profile_hash,
        "tp": spec.tp,
        "gsm8k_num_samples": spec.gsm8k_num_samples,
        "baseline": 0.85,
        "quantized": 0.85,
        "gap": 0.0,
        "passed": True,
    }
    ckpt.save()
    mock_gate_cls.return_value._source_cache = 0.85

    result = orchestrator.run(spec)

    assert result.status == "perf_below_target"
    mock_tps.assert_not_called()
    state = Checkpoint.load(tmp_path).state
    assert len(state["performance_measurements"]) == 1


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark")
def test_perf_failed_does_not_retry_with_same_kernel_repo(mock_tps, tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(tmp_path, kernel_repo="/repo/aiter")
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "perf_failed",
            "quant_ckpt_dir": "/prior/quant_ckpt",
            "kernel_repo": "/repo/aiter",
        }
    )
    ckpt.save()

    result = Orchestrator().run(spec)

    assert result.status == "perf_below_target"
    mock_tps.assert_not_called()


def test_retry_perfopt_resets_only_perfopt_state(tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    quant = tmp_path / "quant_ckpt"
    quant.mkdir()
    spec = make_spec(tmp_path, retry_perfopt=True)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "perf_failed",
            "terminal_stage": "perf_failed",
            "terminal_result": {"status": "perf_below_target"},
            "quant_ckpt_dir": str(quant),
            "baseline_gsm8k": 0.8,
            "accuracy_validation": {
                "passed": True,
                "fingerprint": "accuracy",
                "gap": 0.01,
            },
            "performance_runtime": {"fingerprint": "runtime"},
            "performance_measurements": [
                {"role": "quant_only"},
                {"role": "vendor_gemm"},
                {"role": "final_abba"},
            ],
            "bottleneck_analysis": {"candidates": [{"kernel_id": "old"}]},
            "kernel_journey": [{"kernel_id": "old"}],
            "geak_patches": [{"kernel_sig": "old"}],
            "vendor_gemm_tuning": {"status": "no_improvement"},
            "retain_trials": [{"decision": "DROP"}],
        }
    )
    trace = tmp_path / "trace"
    trace.mkdir()
    (trace / "old.json.gz").write_text("old")
    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "final.json").write_text("{}")

    Orchestrator._prepare_perfopt_retry(spec, ckpt)

    assert ckpt.state["stage"] == "perfopt"
    assert ckpt.state["quant_ckpt_dir"] == str(quant)
    assert ckpt.state["baseline_gsm8k"] == 0.8
    assert ckpt.state["accuracy_validation"]["passed"] is True
    assert ckpt.state["performance_measurements"] == [{"role": "quant_only"}]
    assert ckpt.state["bottleneck_analysis"]["candidates"] == []
    assert ckpt.state["kernel_journey"] == []
    assert ckpt.state["geak_patches"] == []
    assert ckpt.state["vendor_gemm_tuning"] == {}
    assert ckpt.state["retain_trials"] == []
    assert ckpt.state["perfopt_history"]
    history_dir = Path(ckpt.state["perfopt_history"][-1]["artifact_dir"])
    assert (history_dir / "trace" / "old.json.gz").is_file()
    assert (history_dir / "reports" / "final.json").is_file()
    assert not trace.exists()
    assert not reports.exists()


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark")
def test_retry_perfopt_never_remeasures_missing_throughput_pair(
    mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint, StageError

    spec = make_spec(tmp_path, retry_perfopt=True)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["stage"] = "perfopt"
    orchestrator = Orchestrator()
    orchestrator.benchmark_coordinator.reuse_matching_quantized_throughput_pair = MagicMock(return_value=None)

    with pytest.raises(
        StageError,
        match="reusable quant-only throughput pair",
    ):
        orchestrator._measure_or_reuse_quantized_throughput(
            spec,
            ckpt,
            "/quant",
            0.01,
        )

    mock_tps.assert_not_called()


def test_retry_perfopt_execution_bypasses_upstream_stages(tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint, DeployPackage

    spec = make_spec(tmp_path, retry_perfopt=True)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["stage"] = "perfopt"
    orchestrator = Orchestrator()
    gate = MagicMock()
    expected = DeployPackage(status="perf_below_target")
    orchestrator._resume_session_or_return_terminal_result = MagicMock(return_value=None)
    orchestrator._prepare_managed_workspaces = MagicMock()
    orchestrator._restore_perfopt_retry_context = MagicMock(return_value=(gate, "/quant", 0.01))
    orchestrator._complete_after_accuracy = MagicMock(return_value=expected)
    orchestrator._check_baseline_before_quantization = MagicMock(side_effect=AssertionError("baseline must not run"))
    orchestrator._prepare_quantized_checkpoint = MagicMock(side_effect=AssertionError("quantization must not run"))
    orchestrator._evaluate_quantized_checkpoint_and_handle_fallback = MagicMock(
        side_effect=AssertionError("accuracy must not run")
    )

    result = orchestrator._execute_checkpointed_pipeline(spec, ckpt)

    assert result is expected
    orchestrator._restore_perfopt_retry_context.assert_called_once_with(
        spec,
        ckpt,
    )
    orchestrator._complete_after_accuracy.assert_called_once_with(
        spec,
        ckpt,
        gate,
        "/quant",
        0.01,
    )


@pytest.mark.parametrize("stage", ["perfopt", "failed"])
def test_retry_perfopt_restarts_interrupted_perfopt_session(tmp_path, stage):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(tmp_path, retry_perfopt=True)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["stage"] = stage
    orchestrator = Orchestrator()
    orchestrator._prepare_perfopt_retry = MagicMock()

    result = orchestrator._resume_session_or_return_terminal_result(
        spec,
        ckpt,
    )

    assert result is None
    orchestrator._prepare_perfopt_retry.assert_called_once_with(
        spec,
        ckpt,
    )


def test_retry_perfopt_reopens_failed_terminal(tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint, DeployPackage

    spec = make_spec(tmp_path, retry_perfopt=True)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(stage="failed", terminal_result={"status": "accuracy_failed"})
    ckpt.save()
    orchestrator = Orchestrator()
    orchestrator._execute_checkpointed_pipeline = MagicMock(return_value=DeployPackage(status="perf_below_target"))
    orchestrator.run(spec)
    orchestrator._execute_checkpointed_pipeline.assert_called_once()


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", side_effect=[1000.0, 1100.0])
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
def test_perf_failed_resumes_same_repo_when_kernel_candidates_remain(
    mock_landing,
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    kernel_repo = _git_repo(tmp_path / "aiter")
    session = tmp_path / "session"
    spec = make_spec(
        session,
        kernel_repo=str(kernel_repo),
        target_gain=1.2,
    )
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "perf_failed",
            "quant_ckpt_dir": "/prior/quant_ckpt",
            "kernel_repo": str(kernel_repo),
            "baseline_gsm8k": 0.8,
            "best_accuracy_gap": 0.01,
            "terminal_stage": "perf_failed",
            "terminal_result": {
                "status": "perf_below_target",
                "quant_ckpt_dir": "/prior/quant_ckpt",
                "message": "prior run stopped below target",
            },
            "kernel_journey": [
                {
                    "kernel_id": "remaining-kernel",
                    "outcome": "selected",
                    "backend_attempts": [],
                    "e2e": {},
                }
            ],
        }
    )
    _mark_accuracy_valid(
        ckpt,
        spec,
        "/prior/quant_ckpt",
        baseline=0.8,
    )
    ckpt.save()
    _good_gate_mock(mock_gate_cls)
    mock_landing.load.return_value = fake_server()
    perfopt = MagicMock()
    perfopt.generate_optimization_candidates.return_value = PerfResult(
        patches=[],
        gain=1.0,
    )

    result = Orchestrator(perfopt=perfopt).run(spec)

    assert result.status == "perf_below_target"
    assert mock_tps.call_count == 2
    perfopt.generate_optimization_candidates.assert_called_once()


def test_final_pair_uses_abba_geometric_gain(tmp_path):
    spec = make_spec(tmp_path)
    ckpt = MagicMock()
    ckpt.state = {"performance_measurements": []}

    with patch(
        "quark.experimental.torch.quant_perf.evaluation.throughput.measure_throughput",
        side_effect=[
            _throughput_measurement(100.0),
            _throughput_measurement(180.0),
            _throughput_measurement(182.0),
            _throughput_measurement(101.0),
        ],
    ):
        baseline, final, gain, floor = BenchmarkCoordinator(
            Orchestrator().repair_service
        ).measure_final_stack_with_abba(spec, ckpt, "/quant")

    assert baseline == pytest.approx((100.0 * 101.0) ** 0.5)
    assert final == pytest.approx((180.0 * 182.0) ** 0.5)
    assert gain == pytest.approx(((180.0 * 182.0) / (100.0 * 101.0)) ** 0.5)
    assert floor == 0.01
    measurement = ckpt.state["performance_measurements"][-1]
    assert measurement["role"] == "final_abba"
    assert measurement["mode"] == "abba"
    assert measurement["effective_keep_floor"] == 0.01
    assert len(measurement["baseline_measurements"]) == 2
    assert len(measurement["final_measurements"]) == 2


@patch(
    "quark.experimental.torch.quant_perf.evaluation.throughput.measure_throughput",
    side_effect=[
        _throughput_measurement(1000.0),
        _throughput_measurement(1200.0),
        _throughput_measurement(1000.0),
    ],
)
@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline", return_value=0.79)
def test_runtime_env_candidate_is_kept_after_real_validation(
    mock_accuracy,
    mock_tps,
    monkeypatch,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    key = "AITER_CONFIG_GEMM_A4W4"
    monkeypatch.delenv(key, raising=False)
    spec = make_spec(tmp_path, accuracy_gap=0.03, keep_floor=0.01)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["baseline_tps"] = 1000.0
    ckpt.state["baseline_gsm8k"] = 0.8
    gate = MagicMock()
    gate._source_cache = 0.8
    perf = PerfResult(
        patches=[],
        gain=1.0,
        runtime_candidates=[
            {
                "name": "forge_vendor_gemm",
                "runtime_env": {key: "/tmp/tuned.csv"},
                "artifacts": {"tuned_csv": "/tmp/tuned.csv"},
            }
        ],
    )

    retained_patches, retained_sources, gain, gap = CandidateRetentionService().validate_and_retain_candidates(
        perf,
        spec,
        1.0,
        "/quant",
        gate,
        ckpt,
    )

    assert retained_patches == []
    assert retained_sources == []
    assert gain == 1.2
    assert gap == pytest.approx(0.0125)
    assert os.environ[key] == "/tmp/tuned.csv"
    assert ckpt.state["retained_runtime_env"] == {key: "/tmp/tuned.csv"}
    assert perf.runtime_env == {key: "/tmp/tuned.csv"}
    assert perf.runtime_artifacts == {"tuned_csv": "/tmp/tuned.csv"}


@patch(
    "quark.experimental.torch.quant_perf.evaluation.throughput.measure_throughput",
    side_effect=[
        _throughput_measurement(1000.0),
        _throughput_measurement(1005.0),
    ],
)
@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline", return_value=0.79)
def test_runtime_env_candidate_is_reverted_below_keep_floor(
    mock_accuracy,
    mock_tps,
    monkeypatch,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    key = "AITER_CONFIG_GEMM_A4W4"
    monkeypatch.setenv(key, "/original.csv")
    spec = make_spec(tmp_path, accuracy_gap=0.03, keep_floor=0.01)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["baseline_tps"] = 1000.0
    ckpt.state["baseline_gsm8k"] = 0.8
    gate = MagicMock()
    gate._source_cache = 0.8
    perf = PerfResult(
        patches=[],
        gain=1.0,
        runtime_candidates=[
            {
                "name": "forge_vendor_gemm",
                "runtime_env": {key: "/candidate.csv"},
                "artifacts": {},
            }
        ],
    )

    retained_patches, retained_sources, gain, gap = CandidateRetentionService().validate_and_retain_candidates(
        perf,
        spec,
        1.0,
        "/quant",
        gate,
        ckpt,
    )

    assert retained_patches == []
    assert retained_sources == []
    assert gain == 1.0
    assert os.environ[key] == "/original.csv"
    assert ckpt.state.get("retained_runtime_env") in (None, {})
    assert perf.runtime_env == {}


@patch(
    "quark.experimental.torch.quant_perf.evaluation.throughput.measure_throughput",
    side_effect=[
        _throughput_measurement(1000.0),
        _throughput_measurement(950.0),
    ],
)
@patch("quark.experimental.torch.quant_perf.evaluation.gsm8k.gsm8k_eval_offline")
def test_unmeasured_runtime_candidate_rejects_clear_screen_loss_before_accuracy(
    mock_accuracy,
    mock_tps,
    monkeypatch,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    key = "PYTORCH_TUNABLEOP_FILENAME"
    monkeypatch.delenv(key, raising=False)
    spec = make_spec(tmp_path, accuracy_gap=0.03, keep_floor=0.01, bench_concurrency=64)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["baseline_gsm8k"] = 0.8
    gate = MagicMock()
    gate._source_cache = 0.8
    perf = PerfResult(
        patches=[],
        gain=1.0,
        runtime_candidates=[
            {
                "name": "forge_vendor_gemm:vllm_tunableop",
                "runtime_env": {key: "/tmp/tuned.csv"},
                "artifacts": {},
                "requires_screen": True,
            }
        ],
    )

    _, _, gain, _ = CandidateRetentionService().validate_and_retain_candidates(
        perf,
        spec,
        1.0,
        "/quant",
        gate,
        ckpt,
    )

    assert gain == 1.0
    assert key not in os.environ
    assert mock_accuracy.call_count == 0
    assert mock_tps.call_count == 2
    for call in mock_tps.call_args_list:
        assert call.kwargs["num_prompts"] == 128
        assert call.kwargs["timed_samples"] == 2
        assert call.kwargs["max_timed_samples"] == 2
    assert key not in mock_tps.call_args_list[0].kwargs["runtime_env"]
    assert mock_tps.call_args_list[1].kwargs["runtime_env"][key] == "/tmp/tuned.csv"
    assert ckpt.state["phase_timeline"][-1]["reason"] == "screen_confirmed_no_gain"


@patch("quark.experimental.torch.quant_perf.perfopt.kernel_workflow.rebuild_framework")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.clean_untracked")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.reset_hard_to")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.get_head_sha", return_value="deadbeef")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.worktree_dirty", return_value=True)
def test_cleanup_resets_cleans_and_rebuilds_when_polluted(
    mock_dirty, mock_head, mock_reset, mock_clean, mock_rebuild, tmp_path
):
    spec = make_spec(tmp_path, framework_worktree="/repo")
    Orchestrator()._cleanup_framework_repo(spec)
    # reset targets HEAD (keeps committed patches), clean removes untracked,
    # rebuild runs because csrc was polluted.
    mock_reset.assert_called_once_with("/repo", "deadbeef")
    mock_clean.assert_called_once_with("/repo", "csrc")
    mock_rebuild.assert_called_once_with("/repo")


@patch("quark.experimental.torch.quant_perf.perfopt.kernel_workflow.rebuild_framework")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.clean_untracked")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.reset_hard_to")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.get_head_sha", return_value="deadbeef")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.worktree_dirty", return_value=False)
def test_cleanup_skips_rebuild_when_no_c_pollution(
    mock_dirty, mock_head, mock_reset, mock_clean, mock_rebuild, tmp_path
):
    spec = make_spec(tmp_path, framework_worktree="/repo")
    Orchestrator()._cleanup_framework_repo(spec)
    mock_reset.assert_called_once()
    mock_clean.assert_called_once()
    mock_rebuild.assert_not_called()  # no csrc pollution -> no needless rebuild


@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.reset_hard_to")
def test_cleanup_noop_without_framework_repo(mock_reset, tmp_path):
    Orchestrator()._cleanup_framework_repo(make_spec(tmp_path))  # framework_repo=""
    mock_reset.assert_not_called()


@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.reset_hard_to")
def test_cleanup_noop_for_readonly_framework_repo(mock_reset, tmp_path):
    spec = make_spec(
        tmp_path,
        framework_repo="/repo/vllm",
        workspace_source="readonly",
    )

    Orchestrator()._cleanup_framework_repo(spec)

    mock_reset.assert_not_called()


@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.reset_hard_to")
def test_cleanup_refuses_to_reset_user_source_repo(mock_reset, tmp_path):
    from quark.experimental.torch.quant_perf.workspace.manager import WorkspaceError

    spec = make_spec(tmp_path, framework_repo="/repo/vllm")

    with pytest.raises(WorkspaceError, match="integration worktree"):
        Orchestrator()._cleanup_framework_repo(spec)

    mock_reset.assert_not_called()


# -- baseline health gate --------------------------------------------------


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_base_unhealthy_aborts_before_quantized_measurement(
    mock_run_ptq, mock_landing, mock_gate_cls, mock_tps, tmp_path
):
    mock_run_ptq.return_value = {"status": "success", "quantized_model_dir": "/quant_ckpt"}
    mock_gate_cls.return_value.check_baseline_health.return_value = (False, "base model broken", 0.0)
    kb = MagicMock()
    kb.base_unhealthy_seen.return_value = False
    result = Orchestrator(
        experience_store=kb,
        perfopt=None,
    ).run(make_spec(tmp_path))
    assert result.status == "base_unhealthy"
    assert "broken" in result.message
    mock_run_ptq.assert_not_called()
    mock_gate_cls.return_value.eval_quantized.assert_not_called()  # no quantized measurement
    kb.record_baseline_health.assert_called_once()
    assert kb.record_baseline_health.call_args.kwargs["failure_class"] == "unknown"
    assert kb.record_baseline_health.call_args.kwargs["cache_policy"] == "none"


@patch(
    "quark.experimental.torch.quant_perf.reporting.service.write_final_artifacts",
    return_value={
        "final_json": "/s/reports/final.json",
        "final_md": "/s/reports/final.md",
        "session_breakdown_json": "/s/session_breakdown.json",
        "session_report_md": "/s/session_report.md",
    },
)
@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_throughput_recovery_budget_exhaustion_reaches_final_report(
    mock_run_ptq,
    mock_gate_cls,
    mock_tps,
    mock_write_reports,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    _good_gate_mock(mock_gate_cls)
    mock_run_ptq.return_value = {
        "status": "success",
        "quantized_model_dir": "/quant_ckpt",
    }
    mock_tps.side_effect = [
        BenchmarkFailure(
            "base",
            stdout="No available memory for the cache blocks",
        )
        for _ in range(3)
    ]

    result = Orchestrator(perfopt=None).run(
        make_spec(
            tmp_path,
            vllm_extra_args=["--gpu-memory-utilization=0.75"],
        )
    )

    state = Checkpoint.load(tmp_path).state
    assert result.status == "performance_failed"
    assert "recovery budget exhausted" in result.message
    assert result.report_status == "complete"
    assert state["stage"] == "failed"
    assert state["performance_status"] == "measurement_failed"
    assert state["terminal_result"]["status"] == "performance_failed"
    assert len(state["recovery_attempts"]) == 3
    mock_write_reports.assert_called_once()


@patch(
    "quark.experimental.torch.quant_perf.reporting.service.write_final_artifacts",
    return_value={
        "final_json": "/s/reports/final.json",
        "final_md": "/s/reports/final.md",
        "session_breakdown_json": "/s/session_breakdown.json",
        "session_report_md": "/s/session_report.md",
    },
)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
@patch("quark.experimental.torch.quant_perf.knowledge.terminal_experience.TerminalExperienceRecorder.capture")
def test_terminal_result_runs_final_stage_and_surfaces_reports(
    mock_capture_experience,
    mock_run_ptq,
    mock_gate_cls,
    mock_write_reports,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.knowledge.terminal_experience import (
        ExperienceCaptureSummary,
    )

    mock_capture_experience.return_value = ExperienceCaptureSummary(
        quantization=1,
        repair=0,
        kernel_optimization=0,
    )
    _good_gate_mock(mock_gate_cls)
    mock_run_ptq.return_value = {
        "status": "failed",
        "agent_summary": "no artifacts",
    }

    result = Orchestrator(experience_store=MagicMock()).run(make_spec(tmp_path))

    from quark.experimental.torch.quant_perf.session.progress import read_progress
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    state = Checkpoint.load(tmp_path).state
    progress = read_progress(tmp_path)
    assert result.status == "accuracy_failed"
    assert result.report_status == "complete"
    assert result.report_paths["final_json"].endswith("final.json")
    assert state["stage"] == "failed"
    assert state["terminal_stage"] == "failed"
    assert state["reporting"]["status"] == "complete"
    assert state["experience_capture"]["quantization"] == 1
    assert state["experience_capture"]["reviewable"] == 0
    assert progress["stage"] == "failed"
    assert progress["report_status"] == "complete"
    mock_capture_experience.assert_called_once()
    mock_write_reports.assert_called_once()


@patch(
    "quark.experimental.torch.quant_perf.reporting.service.write_final_artifacts",
    side_effect=RuntimeError("disk full"),
)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_report_failure_does_not_mask_terminal_result(
    mock_run_ptq,
    mock_gate_cls,
    mock_write_reports,
    tmp_path,
):
    _good_gate_mock(mock_gate_cls)
    mock_run_ptq.return_value = {
        "status": "failed",
        "agent_summary": "no artifacts",
    }

    result = Orchestrator().run(make_spec(tmp_path))

    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    state = Checkpoint.load(tmp_path).state
    assert result.status == "accuracy_failed"
    assert result.report_status == "failed"
    assert result.report_paths == {}
    assert state["stage"] == "failed"
    assert state["reporting"]["status"] == "failed"
    assert "disk full" in state["reporting"]["error"]


@patch(
    "quark.experimental.torch.quant_perf.reporting.service.write_final_artifacts",
    return_value={
        "final_json": "/s/reports/final.json",
        "final_md": "/s/reports/final.md",
        "session_breakdown_json": "/s/session_breakdown.json",
        "session_report_md": "/s/session_report.md",
    },
)
@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_resume_from_final_stage_only_regenerates_reports(mock_run_ptq, mock_tps, mock_write_reports, tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(tmp_path)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "final",
            "terminal_stage": "perf_failed",
            "run_spec": {},
            "terminal_result": {
                "status": "perf_below_target",
                "quant_ckpt_dir": "/q",
                "message": "below target",
                "applied_patches": [],
                "framework_branch": "",
                "original_branch": "",
                "revert_command": "",
                "kernel_branch": "",
                "kernel_original_branch": "",
                "kernel_revert_command": "",
                "perf": None,
                "report_status": "",
                "report_paths": {},
            },
        }
    )
    ckpt.save()

    result = Orchestrator().run(spec)

    state = Checkpoint.load(tmp_path).state
    assert result.status == "perf_below_target"
    assert result.report_status == "complete"
    assert state["stage"] == "perf_failed"
    assert state["run_spec"]["model_dir"] == "m"
    mock_run_ptq.assert_not_called()
    mock_tps.assert_not_called()
    mock_write_reports.assert_called_once()


@patch(
    "quark.experimental.torch.quant_perf.reporting.service.write_final_artifacts",
    return_value={
        "final_json": "/s/reports/final.json",
        "final_md": "/s/reports/final.md",
        "session_breakdown_json": "/s/session_breakdown.json",
        "session_report_md": "/s/session_report.md",
    },
)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_resume_from_failed_terminal_only_regenerates_reports(mock_run_ptq, mock_write_reports, tmp_path):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(tmp_path)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "failed",
            "terminal_stage": "failed",
            "terminal_result": {
                "status": "accuracy_failed",
                "quant_ckpt_dir": "",
                "perf": None,
                "message": "accuracy failed",
                "applied_patches": [],
                "framework_branch": "",
                "original_branch": "",
                "revert_command": "",
                "kernel_branch": "",
                "kernel_original_branch": "",
                "kernel_revert_command": "",
                "report_status": "failed",
                "report_paths": {},
            },
            "reporting": {
                "status": "failed",
                "paths": {},
                "error": "disk full",
            },
        }
    )
    ckpt.save()

    result = Orchestrator().run(spec)

    assert result.status == "accuracy_failed"
    assert result.report_status == "complete"
    mock_run_ptq.assert_not_called()
    mock_write_reports.assert_called_once()


@patch.object(
    Orchestrator,
    "_cleanup_workspaces_and_generate_final_reports",
)
@patch.object(Orchestrator, "_execute_checkpointed_pipeline")
def test_base_unhealthy_terminal_reenters_pipeline_when_fix_requested(
    mock_run_locked,
    mock_finalize,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint, DeployPackage

    spec = make_spec(tmp_path, fix_base_framework=True)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state.update(
        {
            "stage": "base_unhealthy",
            "terminal_stage": "base_unhealthy",
            "terminal_result": {
                "status": "base_unhealthy",
                "quant_ckpt_dir": "",
                "message": "baseline failed",
            },
        }
    )
    ckpt.save()
    expected = DeployPackage(status="success")
    mock_run_locked.return_value = expected
    mock_finalize.return_value = expected

    result = Orchestrator().run(spec)

    assert result is expected
    mock_run_locked.assert_called_once()


@patch("quark.experimental.torch.quant_perf.repair.llm_repair.attempt_load_repair", return_value=True)
@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.landing")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_fix_base_framework_optin_retries_then_proceeds(
    mock_run_ptq, mock_landing, mock_gate_cls, mock_tps, mock_repair, tmp_path
):
    mock_run_ptq.return_value = {"status": "success", "quantized_model_dir": "/quant_ckpt"}
    mock_landing.load.return_value = fake_server()
    gate = mock_gate_cls.return_value
    gate._source_cache = 0.8
    # unhealthy first, then healthy after the repair
    gate.check_baseline_health.side_effect = [(False, "broken", 0.0), (True, "", 0.8)]
    gate.eval_quantized.return_value = AccuracyResult(gap=0.01, source_gsm8k=0.8, quantized_gsm8k=0.79, passed=True)
    kb = MagicMock()
    kb.base_unhealthy_seen.return_value = False
    framework_repo = _git_repo(tmp_path / "vllm")
    spec = make_spec(
        tmp_path / "session",
        framework="vllm",
        framework_repo=str(framework_repo),
        fix_base_framework=True,
        target_gain=1.0,
    )

    def repair_with_source_change(**kwargs):
        candidate = Path(kwargs["framework_repo"])
        (candidate / "baseline_fix.py").write_text("FIXED = True\n")
        _git(candidate, "add", "baseline_fix.py")
        _git(candidate, "commit", "-m", "fix baseline")
        return True

    mock_repair.side_effect = repair_with_source_change
    result = Orchestrator(
        experience_store=kb,
        perfopt=None,
    ).run(spec)
    assert result.status == "success"
    mock_repair.assert_called_once()
    assert mock_repair.call_args.kwargs["framework_repo"] != str(framework_repo)
    assert mock_repair.call_args.kwargs["quant_ckpt_dir"] == spec.base_model
    # records the fix so a future run won't fast-fail
    assert any(c.kwargs.get("outcome") == "fixed" for c in kb.record_baseline_health.call_args_list)


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_baseline_health_completes_before_quantization(
    mock_run_ptq,
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    events = []
    gate = mock_gate_cls.return_value
    gate._source_cache = 0.8

    def _check_health():
        events.append("baseline")
        return True, "", 0.8

    async def _quantize(*_args, **_kwargs):
        events.append("quantize")
        return {
            "status": "success",
            "quantized_model_dir": "/quant_ckpt",
        }

    gate.check_baseline_health.side_effect = _check_health
    gate.eval_quantized.return_value = AccuracyResult(
        gap=0.01,
        source_gsm8k=0.8,
        quantized_gsm8k=0.792,
        passed=True,
    )
    mock_run_ptq.side_effect = _quantize

    result = Orchestrator().run(make_spec(tmp_path, target_gain=1.0))

    assert result.status == "success"
    assert events[:2] == ["baseline", "quantize"]


@patch("quark.experimental.torch.quant_perf.evaluation.throughput.throughput_benchmark", return_value=2000.0)
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.AccuracyGate")
@patch("quark.experimental.torch.quant_perf.orchestration.orchestrator.run_ptq", new_callable=AsyncMock)
def test_resume_reuses_fingerprinted_prequant_baseline(
    mock_run_ptq,
    mock_gate_cls,
    mock_tps,
    tmp_path,
):
    from quark.experimental.torch.quant_perf.runtime.recovery import build_runtime_fingerprint
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    spec = make_spec(tmp_path, target_gain=1.0)
    ckpt = Checkpoint.fresh(spec)
    ckpt.state["baseline_gsm8k"] = 0.8
    ckpt.state["baseline_runtime_health"] = {
        "status": "healthy",
        "fingerprint": build_runtime_fingerprint(
            spec,
            framework_commit="",
            runtime_env=os.environ,
            stage="baseline_health",
        ),
    }
    ckpt.state["baseline_reference"] = {
        "score": 0.8,
        "profile_hash": spec.eval_profile.profile_hash,
        "gsm8k_num_samples": spec.gsm8k_num_samples,
        "base_model": str(Path(spec.base_model).resolve()),
        "runtime_fingerprint": ckpt.state["baseline_runtime_health"]["fingerprint"],
        "source": "measured",
    }
    ckpt.save()

    mock_run_ptq.return_value = {
        "status": "success",
        "quantized_model_dir": "/quant_ckpt",
    }
    gate = mock_gate_cls.return_value
    gate.eval_quantized.return_value = AccuracyResult(
        gap=0.01,
        source_gsm8k=0.8,
        quantized_gsm8k=0.792,
        passed=True,
    )

    result = Orchestrator().run(spec)

    assert result.status == "success"
    gate.check_baseline_health.assert_not_called()
    assert gate._source_cache == 0.8
