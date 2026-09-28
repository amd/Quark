#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Orchestrator: the main pipeline (quantize -> land -> accuracy gate ->
perfopt -> package), with checkpointing and a single-run lock.

Design ref: IMPL_SPEC §4.1. mix_precision_search (§2.1.2) is delegated to
Quark's module-level public search API. PerfOpt (§4.2) still supports
`perfopt=None` to short-circuit after a passing accuracy gate, the natural
state before Phase 3's PerfOpt existed and still valid when no
--framework-repo is given.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from quark.experimental.torch.quant_perf import config, landing
from quark.experimental.torch.quant_perf.evaluation.gate import AccuracyGate
from quark.experimental.torch.quant_perf.evaluation.profile import (
    EVAL_PROFILE_RESOLVER_VERSION,
    eval_profile_input_hash,
    resolve_eval_profile,
)
from quark.experimental.torch.quant_perf.perfopt.bottleneck_analysis import (
    bottleneck_analysis_from_state,
)
from quark.experimental.torch.quant_perf.perfopt.service import OptimizationService
from quark.experimental.torch.quant_perf.pipeline import (
    AccuracyStage,
    BaselineHealthStage,
    BenchmarkCoordinator,
    CandidateRetentionService,
    LandingStage,
)
from quark.experimental.torch.quant_perf.quantize.direct_ptq import check_quant_artifacts, run_ptq
from quark.experimental.torch.quant_perf.quantize.isolated_search import (
    run_isolated_module_search as run_module_search,
)
from quark.experimental.torch.quant_perf.quantize.search import (
    advance_search_candidate,
    has_resumable_search_candidate,
)
from quark.experimental.torch.quant_perf.repair import RepairRequest, RepairService, build_repair_request
from quark.experimental.torch.quant_perf.runtime.inventory import collect_runtime_inventory
from quark.experimental.torch.quant_perf.runtime.recovery import (
    FailureDiagnosis,
    build_accuracy_fingerprint,
    build_runtime_fingerprint,
    classify_failure,
)
from quark.experimental.torch.quant_perf.session.progress import write_progress
from quark.experimental.torch.quant_perf.session.spec import (
    AccuracyResult,
    Checkpoint,
    DeployPackage,
    EvalProfile,
    PerfResult,
    RuntimeContext,
    SessionLock,
    Spec,
    StageError,
)
from quark.experimental.torch.quant_perf.session.state import SessionState, append_performance_measurement
from quark.experimental.torch.quant_perf.workspace.git import (
    clean_untracked,
    get_head_sha,
    reset_hard_to,
    worktree_dirty,
)
from quark.experimental.torch.quant_perf.workspace.manager import RepoWorkspaceManager, WorkspaceError
from quark.experimental.torch.quant_perf.workspace.sources import (
    SessionRuntime,
    activate_runtime,
    materialize_source,
    resolve_source,
    verify_runtime_origins,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore


class Orchestrator:
    def __init__(
        self,
        experience_store: ExperienceStore | None = None,
        perfopt: OptimizationService | None = None,
        repair_service: RepairService | None = None,
    ) -> None:
        self.experience_store = experience_store
        self.perfopt = perfopt
        self.repair_service = repair_service or RepairService(experience_store=experience_store)
        self.landing_stage = LandingStage(
            self.repair_service,
            loader=landing.load,
        )
        self.accuracy_stage = AccuracyStage(self.repair_service)
        self.baseline_health_stage = BaselineHealthStage(
            self.repair_service,
            experience_store=experience_store,
        )
        self.benchmark_coordinator = BenchmarkCoordinator(self.repair_service)
        self.candidate_retention_service = CandidateRetentionService()

    @staticmethod
    def _effective_runtime_env(spec: Spec) -> dict[str, str]:
        return {**os.environ, **dict(spec.runtime_env)}

    @staticmethod
    def _completed_quant_checkpoint_dir(spec: Spec) -> str:
        """Return the conventional session checkpoint when its artifacts are complete."""
        result = check_quant_artifacts(spec.quant_ckpt_dir)
        return result["quantized_model_dir"] if result["status"] == "success" else ""

    @staticmethod
    def _repair_request(
        spec: Spec,
        *,
        failure_class: str,
        error: str,
        quant_ckpt_dir: str,
        verifier_profile: str,
        quant_signature: str = "",
        metrics: dict[str, Any] | None = None,
        verifier: Callable[[], tuple[bool, str]] | None = None,
        diagnosis: FailureDiagnosis | None = None,
    ) -> RepairRequest:
        return build_repair_request(
            spec,
            failure_class=failure_class,
            error=error,
            quant_ckpt_dir=quant_ckpt_dir,
            verifier_profile=verifier_profile,
            quant_signature=quant_signature,
            metrics=metrics,
            verifier=verifier,
            diagnosis=diagnosis,
        )

    @staticmethod
    def _framework_commit(spec: Spec) -> str:
        """Best-effort framework revision for runtime fingerprints."""
        repo = spec.active_framework_repo
        if not repo or not Path(repo).is_dir():
            return spec.framework_version
        try:
            return get_head_sha(repo) or spec.framework_version
        except OSError:
            return spec.framework_version

    @staticmethod
    def _kernel_commit(spec: Spec) -> str:
        repo = spec.active_kernel_repo
        if not repo or not Path(repo).is_dir():
            return spec.kernel_version
        try:
            return get_head_sha(repo)
        except Exception:
            return spec.kernel_version

    def run(self, spec: Spec) -> DeployPackage:
        session_dir = Path(spec.session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        loaded = Checkpoint.load(session_dir)
        ckpt = loaded or Checkpoint.fresh(spec)
        if loaded is not None:
            spec = replace(
                spec,
                runtime=RuntimeContext.from_dict(dict(ckpt.state.get("runtime_context") or {})),
            )

        with SessionLock(session_dir):
            spec = self._resolve_and_freeze_eval_profile(spec, ckpt)
            if not ckpt.state.get("run_spec"):
                ckpt.state["run_spec"] = spec.to_dict()
                invocation = ckpt.state.setdefault("invocation", {})
                invocation.setdefault("argv", list(spec.invocation_argv))
                invocation["spec"] = spec.to_dict()
                ckpt.save()
            try:
                stage = ckpt.state.get("stage")
                pending_kernel_resume = stage in {"failed", "perf_failed"} and (
                    spec.retry_perfopt or (stage == "perf_failed" and self._has_pending_kernel_candidates(ckpt.state))
                )
                pending_baseline_resume = stage == "base_unhealthy" and (
                    spec.recheck_baseline or spec.fix_base_framework
                )
                pending_accuracy_resume = stage in {"failed", "perf_failed", "done"} and spec.retry_accuracy_gate
                if stage == "final" or (
                    stage in {"done", "perf_failed", "failed", "base_unhealthy"}
                    and ckpt.state.get("terminal_result")
                    and not pending_kernel_resume
                    and not pending_baseline_resume
                    and not pending_accuracy_resume
                ):
                    package = DeployPackage.from_dict(ckpt.state.get("terminal_result") or {})
                else:
                    while True:
                        package = self._execute_checkpointed_pipeline(
                            spec,
                            ckpt,
                        )
                        if package.status != "_retry_search_candidate":
                            break
            except StageError as e:
                logger.error("[%s] %s", e.stage, e.message)
                if e.stage == "throughput":
                    status = "performance_failed"
                    terminal_stage = "failed"
                    ckpt.state["performance_status"] = "measurement_failed"
                elif e.stage == "perfopt":
                    status = "perf_below_target"
                    terminal_stage = "perf_failed"
                else:
                    status = "accuracy_failed"
                    terminal_stage = "failed"
                ckpt.state["stage"] = terminal_stage
                ckpt.record_phase_event(
                    e.stage,
                    "failed",
                    reason=e.message,
                )
                ckpt.save()
                write_progress(
                    spec.session_dir,
                    stage=ckpt.state["stage"],
                    warning=f"{e.stage}: {e.message}",
                )
                package = DeployPackage(
                    status=status,
                    quant_ckpt_dir=ckpt.state.get("quant_ckpt_dir") or "",
                    message=f"{e.stage}: {e.message}",
                )
            return self._cleanup_workspaces_and_generate_final_reports(
                spec,
                ckpt,
                package,
            )

    @staticmethod
    def _terminal_stage(
        package: DeployPackage,
        state: SessionState,
    ) -> str:
        current = state.get("stage")
        if current in {"done", "perf_failed", "failed", "base_unhealthy"}:
            return current
        if package.status == "success":
            return "done"
        if package.status == "perf_below_target":
            return "perf_failed"
        if package.status == "base_unhealthy":
            return "base_unhealthy"
        return "failed"

    @staticmethod
    def _has_pending_kernel_candidates(state: SessionState) -> bool:
        from quark.experimental.torch.quant_perf.perfopt.kernel_source import SOURCE_RESOLVER_VERSION
        from quark.experimental.torch.quant_perf.perfopt.vendor_gemm import (
            vendor_result_needs_run,
        )

        has_vendor_candidates = any(
            str(row.get("skip_reason") or "").startswith("vendor library kernel")
            for row in state.get("kernel_journey") or []
        )
        vendor_result = state.get("vendor_gemm_tuning")
        if has_vendor_candidates and vendor_result and vendor_result_needs_run(vendor_result):
            return True
        for row in state.get("kernel_journey") or []:
            if str((row.get("e2e") or {}).get("decision") or "").upper() == "RETRYABLE_FAULT":
                return True
            if row.get("backend_attempts"):
                continue
            if row.get("outcome") == "selected":
                return True
            if row.get("outcome") != "skipped":
                continue
            mapping = row.get("source_mapping") or {}
            retryable = bool(mapping.get("retryable"))
            prior_version = int(mapping.get("resolver_version") or 0)
            if retryable and prior_version < SOURCE_RESOLVER_VERSION:
                return True
        return False

    def _resolve_and_freeze_eval_profile(
        self,
        spec: Spec,
        ckpt: Checkpoint,
    ) -> Spec:
        """Resolve one session-wide profile and persist it before GPU work."""
        thinking = {
            "enabled": True,
            "disabled": False,
        }.get(spec.eval_thinking_mode)
        overrides = {
            "task": spec.eval_task,
            "num_fewshot": spec.eval_num_fewshot,
            "prompting_strategy": spec.eval_prompting_strategy,
            "enable_thinking": thinking,
            "max_gen_toks": spec.eval_max_gen_toks,
        }
        input_hash = eval_profile_input_hash(
            spec.base_model,
            discovery=spec.eval_discovery,
            allow_llm=spec.eval_allow_llm,
            overrides=overrides,
        )
        stored = ckpt.state.get("eval_profile")
        stale = isinstance(stored, dict) and (
            int(ckpt.state.get("eval_profile_resolver_version") or 0) != EVAL_PROFILE_RESOLVER_VERSION
            or ckpt.state.get("eval_profile_input_hash") != input_hash
        )
        stage = str(ckpt.state.get("stage") or "")
        terminal = stage in {"done", "perf_failed", "failed", "base_unhealthy", "final"}
        accuracy_retry = spec.retry_accuracy_gate or spec.recheck_baseline or spec.fix_base_framework
        if stale and spec.retry_perfopt and not accuracy_retry:
            raise StageError(
                "evaluation",
                "the saved Eval Profile is stale; rerun with --retry-accuracy-gate before --retry-perfopt",
                code="stale_eval_profile",
            )
        if stale and terminal and not accuracy_retry:
            profile = EvalProfile.from_dict(stored)
            return replace(spec, eval_profile=profile)

        created = False
        if isinstance(stored, dict) and not stale:
            profile = EvalProfile.from_dict(stored)
        elif spec.eval_profile is not None and not stale:
            profile = spec.eval_profile
            created = True
        else:
            profile = resolve_eval_profile(
                spec.base_model,
                discovery=spec.eval_discovery,
                allow_llm=spec.eval_allow_llm,
                overrides=overrides,
                artifact_dir=Path(spec.session_dir) / "evaluation" / "profile",
            )
            created = True

        profile = profile.with_computed_hash()
        if stale:
            ckpt.state.setdefault("eval_profile_history", []).append(
                {
                    "profile": dict(stored),
                    "resolver_version": ckpt.state.get("eval_profile_resolver_version"),
                    "input_hash": ckpt.state.get("eval_profile_input_hash"),
                    "baseline_gsm8k": ckpt.state.get("baseline_gsm8k"),
                    "baseline_reference": ckpt.state.get("baseline_reference"),
                    "baseline_runtime_health": ckpt.state.get("baseline_runtime_health"),
                    "accuracy_validation": ckpt.state.get("accuracy_validation"),
                    "accuracy_attempts": list(ckpt.state.get("accuracy_attempts") or []),
                }
            )
            for key in (
                "baseline_gsm8k",
                "baseline_reference",
                "baseline_runtime_health",
                "accuracy_validation",
            ):
                ckpt.state[key] = None
            ckpt.state["accuracy_attempts"] = []
            ckpt.state["best_accuracy_gap"] = None
            if stage not in {"quantize", "failed", "base_unhealthy"}:
                self._reopen_session_at_stage(
                    ckpt,
                    "land" if ckpt.state.get("quant_ckpt_dir") else "quantize",
                    reset_cleanup=False,
                )
        spec = replace(spec, eval_profile=profile)
        ckpt.state["eval_profile"] = profile.to_dict()
        ckpt.state["eval_profile_hash"] = profile.profile_hash
        ckpt.state["eval_profile_resolver_version"] = EVAL_PROFILE_RESOLVER_VERSION
        ckpt.state["eval_profile_input_hash"] = input_hash
        if ckpt.state.get("run_spec"):
            ckpt.state["run_spec"]["eval_profile"] = profile.to_dict()
        if created:
            ckpt.record_phase_event(
                "eval_profile",
                "frozen",
                profile_id=profile.profile_id,
                profile_hash=profile.profile_hash,
            )
        ckpt.save()
        return spec

    @staticmethod
    def _activate_repo_paths(spec: Spec) -> SessionRuntime:
        return activate_runtime(spec)

    @staticmethod
    def _workspace_root(
        spec: Spec,
        ckpt: Checkpoint,
    ) -> Path:
        stored = str(ckpt.state.get("workspace_root") or "")
        if stored:
            return Path(stored)
        for record in (ckpt.state.get("repo_workspaces") or {}).values():
            path = Path(str(record.get("integration_path") or ""))
            if path.name == "worktree" and len(path.parents) >= 3:
                return path.parents[2]
        root = Path(spec.session_dir).resolve() / "workspaces" / "managed"
        ckpt.state["workspace_root"] = str(root)
        return root

    @staticmethod
    def _resolve_sources(spec: Spec, ckpt: Checkpoint) -> None:
        resolved = {}
        roles = (
            (
                "framework",
                spec.framework_repo,
                ("vllm",) if spec.framework == "vllm" else ("atom",),
            ),
            ("kernel", spec.kernel_repo, ("aiter", "aiter_meta")),
        )
        for role, explicit, packages in roles:
            source = resolve_source(
                role=role,
                mode=spec.workspace_source,
                explicit_repo=explicit,
                package_names=packages,
            )
            source = materialize_source(
                source,
                session_dir=spec.session_dir,
            )
            resolved[role] = source.to_dict()
            if role == "framework":
                spec.runtime.framework_source_kind = source.kind
                spec.runtime.framework_source_origin = source.origin_path
                spec.runtime.resolved_framework_repo = source.source_root
            else:
                spec.runtime.kernel_source_kind = source.kind
                spec.runtime.kernel_source_origin = source.origin_path
                spec.runtime.resolved_kernel_repo = source.source_root
        ckpt.state["resolved_sources"] = resolved

    @staticmethod
    def _restore_or_resolve_sources(spec: Spec, ckpt: Checkpoint) -> None:
        resolved = ckpt.state.get("resolved_sources") or {}
        if not resolved:
            Orchestrator._resolve_sources(spec, ckpt)
            return

        for role in ("framework", "kernel"):
            source = resolved.get(role) or {}
            source_root = str(source.get("source_root") or "")
            source_kind = str(source.get("kind") or "")
            source_origin = str(source.get("origin_path") or "")
            if role == "framework":
                spec.runtime.resolved_framework_repo = source_root
                spec.runtime.framework_source_kind = source_kind
                spec.runtime.framework_source_origin = source_origin
            else:
                spec.runtime.resolved_kernel_repo = source_root
                spec.runtime.kernel_source_kind = source_kind
                spec.runtime.kernel_source_origin = source_origin

    def _prepare_managed_workspaces(self, spec: Spec, ckpt: Checkpoint) -> RepoWorkspaceManager:
        self._restore_or_resolve_sources(spec, ckpt)
        manager = RepoWorkspaceManager(
            session_dir=spec.session_dir,
            session_id=ckpt.state.get("session_id", "unknown"),
            ckpt=ckpt,
            root_dir=self._workspace_root(spec, ckpt),
            init_submodules=spec.workspace_source != "readonly",
        )
        if spec.can_modify_framework:
            workspace = manager.prepare("framework", spec.framework_source_repo)
            spec.runtime.framework_worktree = str(workspace.integration_path)
            spec.runtime.framework_branch = workspace.work_branch
            spec.runtime.framework_version = workspace.base_sha
            ckpt.state["fw_branch"] = workspace.work_branch
            ckpt.state["fw_original_branch"] = workspace.original_branch
        if spec.can_modify_kernel:
            workspace = manager.prepare("kernel", spec.kernel_source_repo)
            spec.runtime.kernel_worktree = str(workspace.integration_path)
            spec.runtime.kernel_branch = workspace.work_branch
            spec.runtime.kernel_version = workspace.base_sha
            ckpt.state["kernel_repo"] = spec.kernel_source_repo
            ckpt.state["kernel_branch"] = workspace.work_branch
            ckpt.state["kernel_original_branch"] = workspace.original_branch
        self.repair_service.workspace_manager = manager
        runtime = self._activate_repo_paths(spec)
        ckpt.state["session_runtime"] = runtime.to_dict()
        if runtime.expected_origins:
            ckpt.state["runtime_origin_evidence"] = verify_runtime_origins(runtime)
        if not ckpt.state.get("runtime_inventory"):
            ckpt.state["runtime_inventory"] = collect_runtime_inventory()
        ckpt.state["run_spec"] = spec.to_dict()
        ckpt.state["runtime_context"] = spec.runtime.to_dict()
        ckpt.save()
        return manager

    def _cleanup_workspaces_and_generate_final_reports(
        self,
        spec: Spec,
        ckpt: Checkpoint,
        package: DeployPackage,
    ) -> DeployPackage:
        """Run the best-effort, resumable FINAL stage for a terminal result."""
        if not package.status:
            raise StageError("final", "terminal_result is missing a status")

        cleanup_changed = False
        if not ckpt.state.get("repo_workspaces") and (ckpt.state.get("cleanup") or {}).get("status") != "complete":
            ckpt.state["cleanup"] = {
                "status": "complete",
                "removed": [],
                "errors": [],
            }
            ckpt.save()
        if ckpt.state.get("repo_workspaces") and (ckpt.state.get("cleanup") or {}).get("status") != "complete":
            manager = RepoWorkspaceManager(
                session_dir=spec.session_dir,
                session_id=ckpt.state.get("session_id", "unknown"),
                ckpt=ckpt,
                root_dir=self._workspace_root(spec, ckpt),
                init_submodules=spec.workspace_source != "readonly",
            )
            try:
                manager.cleanup_terminal()
            except Exception as exc:  # cleanup must not hide the run result
                cleanup = ckpt.state.setdefault("cleanup", {})
                cleanup["status"] = "failed"
                cleanup.setdefault("errors", []).append(f"{type(exc).__name__}: {exc}")
                ckpt.state.setdefault("report_warnings", []).append(f"workspace cleanup failed: {exc}")
                ckpt.save()
            else:
                cleanup = ckpt.state.get("cleanup") or {}
                if cleanup.get("status") == "failed":
                    errors = "; ".join(cleanup.get("errors") or [])
                    ckpt.state.setdefault("report_warnings", []).append(f"workspace cleanup incomplete: {errors}")
                    ckpt.save()
                else:
                    self._sync_package_repo_metadata(package, ckpt.state)
            cleanup_changed = True

        if ckpt.state.get("repo_workspaces"):
            try:
                from quark.experimental.torch.quant_perf.workspace.patch_bundle import export_patch_bundle

                ckpt.state["patch_bundle"] = export_patch_bundle(
                    spec.session_dir,
                    ckpt.state,
                )
                ckpt.save()
            except Exception as exc:
                ckpt.state.setdefault("report_warnings", []).append(f"patch bundle export failed: {exc}")
                ckpt.save()

        experience_changed = False
        if self.experience_store is not None:
            try:
                from quark.experimental.torch.quant_perf.knowledge.terminal_experience import (
                    TerminalExperienceRecorder,
                )

                summary = asdict(
                    TerminalExperienceRecorder(self.experience_store).capture(
                        spec=spec,
                        state=ckpt.state,
                    )
                )
                summary["reviewable"] = len(
                    self.experience_store.list_reviewable_experience(session_id=str(ckpt.state.get("session_id") or ""))
                )
                experience_changed = ckpt.state.get("experience_capture") != summary
                ckpt.state["experience_capture"] = summary
                ckpt.save()
            except Exception as exc:
                ckpt.state.setdefault("report_warnings", []).append(f"runtime experience capture failed: {exc}")
                ckpt.save()

        terminal_stage = ckpt.state.get("terminal_stage") or self._terminal_stage(package, ckpt.state)
        prior_reporting = ckpt.state.get("reporting") or {}
        prior_paths = prior_reporting.get("paths") or {}
        if (
            not cleanup_changed
            and not experience_changed
            and prior_reporting.get("status") == "complete"
            and prior_paths
            and all(Path(p).is_file() for p in prior_paths.values())
        ):
            package.report_status = "complete"
            package.report_paths = dict(prior_paths)
            ckpt.state["stage"] = terminal_stage
            write_progress(
                spec.session_dir,
                stage=terminal_stage,
                stage_detail="final reports already complete",
                report_status="complete",
                reports=dict(prior_paths),
            )
            return package

        ckpt.state["terminal_stage"] = terminal_stage
        ckpt.state["terminal_result"] = asdict(package)
        ckpt.record_phase_event(
            "terminal_decision",
            package.status,
            terminal_stage=terminal_stage,
            message=package.message,
        )
        ckpt.state["stage"] = "final"
        ckpt.state["reporting"] = {
            "status": "running",
            "paths": {},
            "error": "",
        }
        ckpt.record_phase_event("final", "running")
        ckpt.save()
        write_progress(
            spec.session_dir,
            stage="final",
            stage_detail="generating final reports",
            report_status="running",
        )

        try:
            from quark.experimental.torch.quant_perf.reporting.service import write_final_artifacts

            paths = write_final_artifacts(spec, ckpt.state, package)
        except Exception as exc:
            package.report_status = "failed"
            package.report_paths = {}
            ckpt.state["stage"] = terminal_stage
            ckpt.state["reporting"] = {
                "status": "failed",
                "paths": {},
                "error": str(exc),
            }
            ckpt.state["terminal_result"] = asdict(package)
            ckpt.record_phase_event(
                "final",
                "failed",
                error=str(exc),
            )
            ckpt.save()
            write_progress(
                spec.session_dir,
                stage=terminal_stage,
                stage_detail="final report generation failed",
                report_status="failed",
                warning=f"final report generation failed: {exc}",
            )
            logger.warning("final report generation failed: %s", exc)
            return package

        package.report_status = "complete"
        package.report_paths = dict(paths)
        ckpt.state["stage"] = terminal_stage
        ckpt.state["reporting"] = {
            "status": "complete",
            "paths": dict(paths),
            "error": "",
            "generated_at": datetime.now(UTC).isoformat(),
        }
        ckpt.state["terminal_result"] = asdict(package)
        ckpt.record_phase_event(
            "final",
            "complete",
            paths=dict(paths),
        )
        ckpt.save()
        write_progress(
            spec.session_dir,
            stage=terminal_stage,
            stage_detail="final reports generated",
            report_status="complete",
            reports=dict(paths),
        )
        return package

    @staticmethod
    def _sync_package_repo_metadata(
        package: DeployPackage,
        state: SessionState,
    ) -> None:
        workspaces = state.get("repo_workspaces") or {}

        def _metadata(role: str) -> tuple[str, str, str]:
            record = workspaces.get(role) or {}
            if not record.get("branch_retained"):
                return "", "", ""
            source = record.get("source_repo", "")
            branch = record.get("work_branch", "")
            original = record.get("original_branch", "")
            revert = f"git -C {source} branch -D {branch}" if source and branch else ""
            return branch, original, revert

        (
            package.framework_branch,
            package.original_branch,
            package.revert_command,
        ) = _metadata("framework")
        (
            package.kernel_branch,
            package.kernel_original_branch,
            package.kernel_revert_command,
        ) = _metadata("kernel")

    @staticmethod
    def _reopen_session_at_stage(
        ckpt: Checkpoint,
        stage: str,
        *,
        reset_cleanup: bool,
    ) -> None:
        ckpt.state["stage"] = stage
        ckpt.state["terminal_stage"] = None
        ckpt.state["terminal_result"] = None
        ckpt.state["reporting"] = {
            "status": "pending",
            "paths": {},
            "error": "",
        }
        if reset_cleanup:
            ckpt.state["cleanup"] = {
                "status": "pending",
                "removed": [],
                "errors": [],
            }

    @classmethod
    def _prepare_perfopt_retry(
        cls,
        spec: Spec,
        ckpt: Checkpoint,
    ) -> None:
        quant_ckpt_dir = Path(ckpt.state.get("quant_ckpt_dir") or "")
        if not quant_ckpt_dir.is_dir():
            raise StageError(
                "perfopt",
                "--retry-perfopt requires the saved quantized checkpoint",
            )
        history = ckpt.state.setdefault("perfopt_history", [])
        history_dir = Path(spec.session_dir) / "perfopt_history" / f"attempt-{len(history) + 1}"
        history_dir.mkdir(parents=True, exist_ok=False)
        history_entry = {
            "attempt": len(history) + 1,
            "artifact_dir": str(history_dir),
            "terminal_stage": ckpt.state.get("terminal_stage"),
            "terminal_result": ckpt.state.get("terminal_result"),
            "bottleneck_analysis": bottleneck_analysis_from_state(ckpt.state),
            "kernel_journey": list(ckpt.state.get("kernel_journey") or []),
            "geak_patches": list(ckpt.state.get("geak_patches") or []),
            "vendor_gemm_tuning": dict(ckpt.state.get("vendor_gemm_tuning") or {}),
            "vendor_shape_evidence": dict(ckpt.state.get("vendor_shape_evidence") or {}),
            "retain_trials": list(ckpt.state.get("retain_trials") or []),
            "retention_stack": list(ckpt.state.get("retention_stack") or []),
            "performance_measurements": list(ckpt.state.get("performance_measurements") or []),
        }
        (history_dir / "state_snapshot.json").write_text(json.dumps(history_entry, indent=2, sort_keys=True))
        session_dir = Path(spec.session_dir)
        for name in (
            "trace",
            "trace_baseline",
            "trace_shape_evidence",
            "vendor_gemm",
            "geak",
            "reports",
        ):
            source = session_dir / name
            if source.exists():
                shutil.move(str(source), str(history_dir / name))
        for name in (
            "session_breakdown.json",
            "session_report.md",
            "kernel_provenance.json",
            "kernel_source_resolution.json",
        ):
            source = session_dir / name
            if source.exists():
                shutil.move(str(source), str(history_dir / name))
        history.append(history_entry)
        for key, empty in (
            (
                "bottleneck_analysis",
                {
                    "policy_version": 1,
                    "requested_mode": spec.bottleneck_mode,
                    "effective_mode": "",
                    "status": "pending",
                    "reason": "",
                    "candidates": [],
                    "quantized_trace": "",
                    "baseline_trace": "",
                    "trace_kind": "",
                },
            ),
            ("kernel_journey", []),
            ("geak_patches", []),
            ("vendor_gemm_tuning", {}),
            ("vendor_shape_evidence", {}),
            ("retain_trials", []),
            ("retention_stack", []),
        ):
            cast(dict[str, Any], ckpt.state)[key] = empty
        ckpt.state["performance_measurements"] = [
            measurement
            for measurement in (ckpt.state.get("performance_measurements") or [])
            if measurement.get("role") == "quant_only"
        ]
        ckpt.state["retention_base"] = {}
        ckpt.state["final_retention"] = {}
        ckpt.state["kernel_provenance_artifact"] = None
        ckpt.state["kernel_source_resolution_artifact"] = None
        ckpt.state["patch_bundle"] = {}
        runtime = ckpt.state.get("performance_runtime") or {}
        ckpt.state["retained_runtime_env"] = dict(runtime.get("runtime_env") or {})
        cls._reopen_session_at_stage(
            ckpt,
            "perfopt",
            reset_cleanup=True,
        )
        ckpt.record_phase_event(
            "perfopt_retry",
            "reopened",
            reason="explicit_retry_perfopt",
        )
        ckpt.save()

    def _restore_perfopt_retry_context(
        self,
        spec: Spec,
        ckpt: Checkpoint,
    ) -> tuple[AccuracyGate, str, float]:
        quant_ckpt_dir = str(ckpt.state.get("quant_ckpt_dir") or "")
        if not quant_ckpt_dir or not Path(quant_ckpt_dir).is_dir():
            raise StageError(
                "perfopt",
                "--retry-perfopt requires the saved quantized checkpoint",
            )
        validation = ckpt.state.get("accuracy_validation")
        if not isinstance(validation, dict) or not validation.get("passed"):
            raise StageError(
                "perfopt",
                "--retry-perfopt requires a passing saved accuracy gate",
            )
        current_fingerprint = build_accuracy_fingerprint(
            spec,
            quant_ckpt_dir,
            framework_commit=self._framework_commit(spec),
            kernel_commit=self._kernel_commit(spec),
            runtime_env=self._effective_runtime_env(spec),
        )
        if validation.get("fingerprint") != current_fingerprint:
            raise StageError(
                "perfopt",
                "--retry-perfopt accuracy fingerprint no longer matches the current runtime",
            )
        baseline = validation.get("baseline")
        if baseline is None:
            baseline = ckpt.state.get("baseline_gsm8k")
        if baseline is None:
            raise StageError(
                "perfopt",
                "--retry-perfopt requires the saved baseline score",
            )
        gate = AccuracyGate(spec)
        gate._source_cache = float(baseline)
        return (
            gate,
            quant_ckpt_dir,
            float(validation.get("gap") or 0.0),
        )

    def _resume_session_or_return_terminal_result(
        self,
        spec: Spec,
        ckpt: Checkpoint,
    ) -> DeployPackage | None:
        stage = ckpt.state.get("stage")
        if stage == "done" and not spec.retry_accuracy_gate:
            return DeployPackage(
                status="success",
                quant_ckpt_dir=ckpt.state["quant_ckpt_dir"],
            )
        if spec.retry_perfopt and (stage == "perfopt" or (stage == "failed" and not spec.retry_accuracy_gate)):
            self._prepare_perfopt_retry(spec, ckpt)
            return None
        if stage in {"failed", "perf_failed", "done"} and spec.retry_accuracy_gate:
            prior_quant_ckpt = ckpt.state.get("quant_ckpt_dir") or ""
            recovered_checkpoint_from_artifacts = False
            if not prior_quant_ckpt:
                prior_quant_ckpt = self._completed_quant_checkpoint_dir(spec)
                recovered_checkpoint_from_artifacts = bool(prior_quant_ckpt)
            accuracy_attempts = ckpt.state.get("accuracy_attempts") or []
            latest_accuracy = accuracy_attempts[-1] if accuracy_attempts else {}
            terminal_message = str((ckpt.state.get("terminal_result") or {}).get("message") or "")
            best_candidate = ckpt.state.get("best_candidate")
            load_incompatible_winner = False
            if "quantized accuracy load failed" in terminal_message and isinstance(best_candidate, dict):
                from quark.experimental.torch.quant_perf.quantize.search import (
                    _flydsl_dense_mxfp4_available,
                )

                load_incompatible_winner = bool(
                    spec.mxfp4_gemm_backend == "flydsl"
                    and any(
                        best_candidate.get(partition) == "mxfp4"
                        for partition in (
                            "linear_attn_mode",
                            "self_attn_mode",
                        )
                    )
                    and not _flydsl_dense_mxfp4_available()
                )
            accuracy_failed_winner = (
                latest_accuracy.get("passed") is False and latest_accuracy.get("candidate") == best_candidate
            )
            latest_repair = (ckpt.state.get("repair_journey") or [])[-1] if ckpt.state.get("repair_journey") else {}
            rechecked_configs = ckpt.state.setdefault(
                "post_repair_rechecked_configs",
                [],
            )
            post_repair_recheck = (
                isinstance(best_candidate, dict)
                and accuracy_failed_winner
                and latest_repair.get("failure_class") == "accuracy_gap"
                and latest_repair.get("status") == "fixed"
                and best_candidate not in rechecked_configs
            )
            if post_repair_recheck:
                rechecked_configs.append(dict(best_candidate))
                ckpt.save()
            restart_search = False
            if (
                ckpt.state.get("path") == "mix_precision_search"
                and isinstance(best_candidate, dict)
                and not post_repair_recheck
                and not recovered_checkpoint_from_artifacts
            ):
                search_state = ckpt.state.get("mix_precision_search") or {}
                queue = search_state.get("candidate_queue") or []
                cursor = int(search_state.get("candidate_cursor") or 0)
                if load_incompatible_winner:
                    restart_search = (
                        advance_search_candidate(
                            ckpt.state,
                            attempt={
                                "passed": False,
                                "candidate": best_candidate,
                                "reason": "quantized_accuracy_load_failed",
                                "message": terminal_message,
                            },
                        )
                        is not None
                    )
                elif accuracy_failed_winner and cursor < len(queue) and queue[cursor].get("config") != best_candidate:
                    restart_search = True
            checkpoint_exists = bool(prior_quant_ckpt and Path(prior_quant_ckpt).is_dir() and not restart_search)
            self._reopen_session_at_stage(
                ckpt,
                "land" if checkpoint_exists else "quantize",
                reset_cleanup=True,
            )
            ckpt.state["accuracy_validation"] = None
            if checkpoint_exists:
                ckpt.state["quant_ckpt_dir"] = prior_quant_ckpt
            else:
                ckpt.state["quant_ckpt_dir"] = None
            ckpt.record_phase_event(
                "accuracy_resume",
                "checkpoint_reused"
                if checkpoint_exists
                else (
                    "search_restarted"
                    if ckpt.state.get("path") == "mix_precision_search"
                    and (ckpt.state.get("mix_precision_search") or {}).get("status") == "failed"
                    and not has_resumable_search_candidate(ckpt.state)
                    else "winner_reexport"
                ),
                prior_quant_ckpt=prior_quant_ckpt,
            )
            ckpt.save()
            stage = ckpt.state.get("stage")
        if stage == "perf_failed":
            if spec.retry_perfopt:
                self._prepare_perfopt_retry(spec, ckpt)
                stage = "perfopt"
                return None
            requested_kernel_repo = os.path.realpath(spec.kernel_repo) if spec.kernel_repo else ""
            prior_kernel_repo = ckpt.state.get("kernel_repo") or ""
            prior_kernel_repo = os.path.realpath(prior_kernel_repo) if prior_kernel_repo else ""
            pending_kernels = self._has_pending_kernel_candidates(ckpt.state)
            if requested_kernel_repo and (requested_kernel_repo != prior_kernel_repo or pending_kernels):
                logger.info(
                    "retrying prior perf_failed session with kernel_repo=%s (pending_kernels=%s)",
                    spec.kernel_repo,
                    pending_kernels,
                )
                self._reopen_session_at_stage(
                    ckpt,
                    "perfopt",
                    reset_cleanup=True,
                )
                ckpt.save()
                stage = "perfopt"
            else:
                # PerfOpt ran with the same repo set; re-running would repeat it.
                return DeployPackage(
                    status="perf_below_target",
                    quant_ckpt_dir=ckpt.state.get("quant_ckpt_dir", ""),
                    message="resumed: perf_below_target from previous run",
                )
        if stage == "base_unhealthy" and not (spec.recheck_baseline or spec.fix_base_framework):
            # The base model was found broken in the framework; nothing downstream
            # is meaningful. Re-running would just re-discover it.
            return DeployPackage(
                status="base_unhealthy",
                message=ckpt.state.get("base_unhealthy_diag", "resumed: base model unhealthy"),
            )
        if stage == "base_unhealthy":
            self._reopen_session_at_stage(
                ckpt,
                "quantize",
                reset_cleanup=False,
            )
            ckpt.save()
        return None

    def _check_baseline_before_quantization(
        self,
        spec: Spec,
        ckpt: Checkpoint,
    ) -> tuple[AccuracyGate, DeployPackage | None]:
        framework_commit = self._framework_commit(spec)
        baseline_fingerprint = build_runtime_fingerprint(
            spec,
            framework_commit=framework_commit,
            runtime_env=self._effective_runtime_env(spec),
            stage="baseline_health",
        )
        ckpt.state["baseline_runtime_fingerprint"] = baseline_fingerprint
        ckpt.save()

        if self.experience_store is not None:
            cached = self.experience_store.baseline_hard_failure(baseline_fingerprint)
            if isinstance(cached, dict) and not (spec.recheck_baseline or spec.fix_base_framework):
                return (
                    AccuracyGate(spec),
                    self._on_base_unhealthy(
                        spec,
                        ckpt,
                        "baseline previously failed for this exact "
                        "runtime: "
                        f"{cached.get('diagnosis', 'unknown failure')}; "
                        "pass --recheck-baseline or "
                        "--fix-base-framework to retry",
                    ),
                )

        gate = AccuracyGate(spec)
        diagnosis = self.baseline_health_stage.check_baseline_before_quantization(
            spec,
            ckpt,
            gate,
            baseline_fingerprint,
            framework_commit,
        )
        if diagnosis:
            return gate, self._on_base_unhealthy(
                spec,
                ckpt,
                diagnosis,
            )
        return gate, None

    def _prepare_quantized_checkpoint(
        self,
        spec: Spec,
        ckpt: Checkpoint,
    ) -> tuple[str, DeployPackage | None]:
        quant_ckpt_dir = ckpt.state.get("quant_ckpt_dir")
        if quant_ckpt_dir:
            logger.info(
                "resuming with existing quant_ckpt_dir=%s",
                quant_ckpt_dir,
            )
            return str(quant_ckpt_dir), None

        if spec.quant_strategy:
            manager = self.repair_service.workspace_manager
            if manager is None:
                raise StageError(
                    "quantize",
                    "direct PTQ requires a checkpoint-managed workspace",
                )
            with manager.execution_worktree(
                "quantizer",
                config.quark_root(),
                "direct-ptq",
            ) as quark_workspace:
                outcome = asyncio.run(
                    run_ptq(
                        spec,
                        spec.session_dir,
                        experience_store=self.experience_store,
                        state=ckpt.state,
                        quark_root=str(quark_workspace),
                    )
                )
            if outcome["status"] != "success":
                return "", self._on_quantize_fail(
                    spec,
                    ckpt,
                    outcome,
                )
            quant_ckpt_dir = outcome["quantized_model_dir"]
        else:
            quant_ckpt_dir = run_module_search(spec, ckpt)

        ckpt.state["quant_ckpt_dir"] = quant_ckpt_dir
        ckpt.state["stage"] = "land"
        ckpt.record_phase_event(
            "quantization",
            "done",
            quant_ckpt_dir=quant_ckpt_dir,
            path=ckpt.state.get("path"),
        )
        ckpt.save()
        return str(quant_ckpt_dir), None

    def _evaluate_quantized_checkpoint_and_handle_fallback(
        self,
        spec: Spec,
        ckpt: Checkpoint,
        gate: AccuracyGate,
        quant_ckpt_dir: str,
    ) -> tuple[float, DeployPackage | None]:
        accuracy_fingerprint = build_accuracy_fingerprint(
            spec,
            quant_ckpt_dir,
            framework_commit=self._framework_commit(spec),
            kernel_commit=self._kernel_commit(spec),
            runtime_env=self._effective_runtime_env(spec),
        )
        accuracy_validation = ckpt.state.get("accuracy_validation")
        accuracy_is_current = (
            isinstance(accuracy_validation, dict)
            and accuracy_validation.get("passed") is True
            and accuracy_validation.get("fingerprint") == accuracy_fingerprint
        )
        if accuracy_is_current:
            logger.info("accuracy gate already validated for current fingerprint; skipping to PerfOpt")
            current_validation = cast(dict[str, Any], accuracy_validation)
            accuracy_gap = float(current_validation.get("gap") or 0.0)
            baseline_gsm8k = ckpt.state.get("baseline_gsm8k")
            if baseline_gsm8k is None:
                gate.warm_baseline()
            else:
                gate._source_cache = baseline_gsm8k
            write_progress(
                spec.session_dir,
                stage="perfopt",
                best_so_far={
                    "accuracy_gap": accuracy_gap,
                    "perf_gain": None,
                },
            )
            return accuracy_gap, None

        write_progress(spec.session_dir, stage="land")
        profile = spec.eval_profile
        if profile is None:
            raise StageError("accuracy", "Eval Profile is missing")
        try:
            accuracy = self.accuracy_stage.evaluate_quantized_checkpoint(
                spec,
                gate,
                quant_ckpt_dir,
            )
        except StageError as error:
            if self._should_reject_search_candidate_for(error, spec):
                attempt = {
                    "passed": False,
                    "candidate": (
                        dict(ckpt.state["best_candidate"])
                        if isinstance(ckpt.state.get("best_candidate"), dict)
                        else None
                    ),
                    "reason": "quantized_accuracy_load_failed",
                    "message": error.message,
                    "code": error.code,
                    "diagnostic": error.diagnostic,
                }
                retry = self._retry_next_search_candidate(
                    spec,
                    ckpt,
                    attempt=attempt,
                    stage_detail=("real accuracy could not load candidate; exporting next persisted Quark result"),
                    prefer_minimal_change=True,
                )
                if retry is not None:
                    return 0.0, retry
            raise
        accuracy_recovery = self.accuracy_stage.last_recovery
        if accuracy_recovery:
            recovery_attempts = ckpt.state.setdefault(
                "recovery_attempts",
                [],
            )
            recovery_attempts.append(
                {
                    "attempt": len(recovery_attempts) + 1,
                    **accuracy_recovery,
                }
            )
            ckpt.state["accuracy_runtime"] = dict(accuracy_recovery)
            ckpt.save()

        accuracy_gap = accuracy.gap
        accuracy_attempts = ckpt.state.setdefault(
            "accuracy_attempts",
            [],
        )
        accuracy_attempts.append(
            {
                "attempt": len(accuracy_attempts) + 1,
                "ts": datetime.now(UTC).isoformat(),
                "baseline": accuracy.source_gsm8k,
                "quantized": accuracy.quantized_gsm8k,
                "gap": accuracy.gap,
                "passed": accuracy.passed,
                "candidate": (
                    dict(ckpt.state["best_candidate"])
                    if isinstance(
                        ckpt.state.get("best_candidate"),
                        dict,
                    )
                    else None
                ),
                "profile_id": profile.profile_id,
                "profile_hash": profile.profile_hash,
                "artifacts": dict(accuracy.artifacts),
            }
        )
        ckpt.state["best_accuracy_gap"] = accuracy_gap
        ckpt.record_phase_event(
            "accuracy_gate",
            "passed" if accuracy.passed else "failed",
            baseline=accuracy.source_gsm8k,
            quantized=accuracy.quantized_gsm8k,
            gap=accuracy.gap,
        )
        ckpt.save()

        if not accuracy.passed and spec.can_modify_framework:
            from quark.experimental.torch.quant_perf.repair.llm_repair import (
                classify_accuracy_failure,
            )
            from quark.experimental.torch.quant_perf.repair.signatures import (
                normalize_quant_signature,
            )

            quant_signature = normalize_quant_signature(
                spec.quant_strategy or "",
                ckpt.state.get("best_candidate"),
            )
            root_cause = classify_accuracy_failure(
                source_gsm8k=gate._source_cache,
                quantized_gsm8k=accuracy.quantized_gsm8k,
                gap=accuracy_gap,
                quant_ckpt_dir=quant_ckpt_dir,
                quant_strategy=spec.quant_strategy or "",
                session_dir=spec.session_dir,
            )
            if root_cause == "framework":
                write_progress(
                    spec.session_dir,
                    stage_detail="repairing framework accuracy bug",
                )
                verified_accuracy: AccuracyResult | None = None

                def verify_accuracy_repair() -> tuple[bool, str]:
                    nonlocal verified_accuracy
                    verified_accuracy = gate.eval_quantized(quant_ckpt_dir)
                    if verified_accuracy.passed:
                        return True, ""
                    return (
                        False,
                        f"accuracy gap {verified_accuracy.gap:.4f} exceeds threshold {spec.accuracy_gap:.4f}",
                    )

                repair = self.repair_service.repair(
                    self._repair_request(
                        spec,
                        failure_class="accuracy_gap",
                        error=(f"quantized model accuracy gap {accuracy_gap:.4f}"),
                        quant_ckpt_dir=quant_ckpt_dir,
                        verifier_profile="accuracy",
                        quant_signature=quant_signature,
                        metrics={
                            "source_gsm8k": gate._source_cache,
                            "quantized_gsm8k": (accuracy.quantized_gsm8k),
                            "gap": accuracy_gap,
                        },
                        verifier=verify_accuracy_repair,
                    )
                )
                if repair.status == "fixed":
                    accuracy = (
                        verified_accuracy
                        if (verified_accuracy is not None and verified_accuracy.passed)
                        else gate.eval_quantized(quant_ckpt_dir)
                    )
                    accuracy_gap = accuracy.gap
                    accuracy_attempts.append(
                        {
                            "attempt": len(accuracy_attempts) + 1,
                            "ts": (datetime.now(UTC).isoformat()),
                            "baseline": accuracy.source_gsm8k,
                            "quantized": accuracy.quantized_gsm8k,
                            "gap": accuracy.gap,
                            "passed": accuracy.passed,
                            "candidate": (
                                dict(ckpt.state["best_candidate"])
                                if isinstance(
                                    ckpt.state.get("best_candidate"),
                                    dict,
                                )
                                else None
                            ),
                            "profile_id": profile.profile_id,
                            "profile_hash": profile.profile_hash,
                            "artifacts": dict(accuracy.artifacts),
                        }
                    )
                    ckpt.state["best_accuracy_gap"] = accuracy_gap
                    ckpt.record_phase_event(
                        "accuracy_gate",
                        "passed" if accuracy.passed else "failed",
                        baseline=accuracy.source_gsm8k,
                        quantized=accuracy.quantized_gsm8k,
                        gap=accuracy.gap,
                        retry="post_repair",
                    )
                    ckpt.save()

        if not accuracy.passed:
            return accuracy_gap, self._on_accuracy_fail(
                spec,
                ckpt,
                accuracy_gap,
            )

        if ckpt.state.get("path") == "mix_precision_search":
            from quark.experimental.torch.quant_perf.quantize.search import (
                accept_current_search_candidate,
            )

            accept_current_search_candidate(
                ckpt.state,
                attempt=accuracy_attempts[-1],
            )
        ckpt.state["accuracy_validation"] = {
            "fingerprint": build_accuracy_fingerprint(
                spec,
                quant_ckpt_dir,
                framework_commit=self._framework_commit(spec),
                kernel_commit=self._kernel_commit(spec),
                runtime_env=self._effective_runtime_env(spec),
            ),
            "profile_hash": profile.profile_hash,
            "tp": spec.tp,
            "gsm8k_num_samples": spec.gsm8k_num_samples,
            "baseline": accuracy.source_gsm8k,
            "quantized": accuracy.quantized_gsm8k,
            "gap": accuracy.gap,
            "passed": True,
        }
        next_stage = "done" if spec.effective_performance_mode == "off" else "perfopt"
        ckpt.state["stage"] = next_stage
        ckpt.save()
        write_progress(
            spec.session_dir,
            stage=next_stage,
            stage_detail=("accuracy gate passed; performance not requested" if next_stage == "done" else ""),
            best_so_far={
                "accuracy_gap": accuracy_gap,
                "perf_gain": None,
            },
        )
        return accuracy_gap, None

    def _measure_or_reuse_quantized_throughput(
        self,
        spec: Spec,
        ckpt: Checkpoint,
        quant_ckpt_dir: str,
        accuracy_gap: float,
    ) -> tuple[float, float, float]:
        resumed_pair = self.benchmark_coordinator.reuse_matching_quantized_throughput_pair(spec, ckpt)
        if resumed_pair is not None:
            baseline_tps, quantized_tps, gain = resumed_pair
            logger.info(
                "[throughput] reusing completed pair: baseline=%.1f tok/s, quantized=%.1f tok/s, gain=%.3fx",
                baseline_tps,
                quantized_tps,
                gain,
            )
            write_progress(
                spec.session_dir,
                stage="perfopt",
                stage_detail=("reusing completed quant-only throughput pair"),
                best_so_far={
                    "accuracy_gap": accuracy_gap,
                    "perf_gain": gain,
                },
            )
            return resumed_pair
        if spec.retry_perfopt:
            raise StageError(
                "perfopt",
                "--retry-perfopt requires a reusable quant-only throughput pair for the current runtime fingerprint",
            )

        write_progress(
            spec.session_dir,
            stage_detail="throughput benchmark: baseline",
        )
        baseline_tps, quantized_tps, gain = self.benchmark_coordinator.measure_baseline_and_quantized_throughput(
            spec,
            ckpt,
            quant_ckpt_dir,
        )
        logger.info(
            "[throughput] baseline=%.1f tok/s, quantized=%.1f tok/s, gain=%.3fx",
            baseline_tps,
            quantized_tps,
            gain,
        )
        ckpt.state["baseline_tps"] = baseline_tps
        ckpt.state["quant_tps"] = quantized_tps
        append_performance_measurement(
            ckpt.state,
            {
                "ts": datetime.now(UTC).isoformat(),
                "role": "quant_only",
                "baseline_tps": baseline_tps,
                "quantized_tps": quantized_tps,
                "gain": gain,
                "isl": spec.isl,
                "osl": spec.osl,
                "concurrency": spec.bench_concurrency,
            },
        )
        ckpt.record_phase_event(
            "throughput_benchmark",
            "done",
            baseline_tps=baseline_tps,
            quantized_tps=quantized_tps,
            gain=gain,
        )
        ckpt.save()
        return baseline_tps, quantized_tps, gain

    def _build_quantization_only_deploy_package(
        self,
        spec: Spec,
        quant_ckpt_dir: str,
        *,
        status: str,
        message: str,
        perf: PerfResult | None = None,
    ) -> DeployPackage:
        return DeployPackage(
            status=status,
            quant_ckpt_dir=quant_ckpt_dir,
            perf=perf,
            framework_branch=(spec.framework_branch if spec.can_modify_framework else ""),
            original_branch=(spec.framework_version if spec.can_modify_framework else ""),
            revert_command=self._repo_revert_command(
                spec.framework_repo,
                spec.framework_branch,
                spec.framework_version,
            ),
            kernel_branch=(spec.kernel_branch if spec.can_modify_kernel else ""),
            kernel_original_branch=(spec.kernel_version if spec.can_modify_kernel else ""),
            kernel_revert_command=self._kernel_revert_command(
                spec,
                spec.kernel_branch,
                spec.kernel_version,
            ),
            message=message,
        )

    def _run_perfopt_and_validate_candidates(
        self,
        spec: Spec,
        ckpt: Checkpoint,
        gate: AccuracyGate,
        quant_ckpt_dir: str,
        quant_gain: float,
    ) -> tuple[PerfResult, list[str], float, float]:
        perfopt = self.perfopt
        if perfopt is None:
            raise StageError(
                "perfopt",
                "performance optimization service is not configured",
            )
        try:
            perfopt.preflight()
        except Exception as exc:
            raise StageError(
                "perfopt",
                f"TraceLens preflight failed: {exc}",
            ) from exc
        profiler_dir = f"{spec.session_dir}/trace"
        server = self.landing_stage.load_model_with_repair(
            spec,
            quant_ckpt_dir,
            profiler_dir=profiler_dir,
        )
        spec = replace(spec, server_port=server.port)
        try:
            perf = perfopt.generate_optimization_candidates(
                server,
                spec,
                ckpt=ckpt,
                quant_gain=quant_gain,
            )
        finally:
            server.stop()

        retained, retained_sources, final_gain, final_gap = (
            self.candidate_retention_service.validate_and_retain_candidates(
                perf,
                spec,
                quant_gain,
                quant_ckpt_dir,
                gate,
                ckpt,
            )
        )
        self._cleanup_framework_repo(spec)
        perf.patches = retained
        perf.patch_srcs = retained_sources
        if ckpt.state.get("retention_stack"):

            def measure_final_stack() -> tuple[float, float, float, float]:
                result = self.benchmark_coordinator.measure_final_stack_with_abba(
                    spec,
                    ckpt,
                    quant_ckpt_dir,
                )
                ckpt.state["baseline_tps"] = result[0]
                ckpt.state["quant_tps"] = result[1]
                return result

            final_gain, final_gap = self.candidate_retention_service.reconcile_final_stack(
                perf,
                ckpt,
                quant_gain=quant_gain,
                quant_gap=float((ckpt.state.get("retention_base") or {}).get("accuracy_gap", final_gap)),
                measure_final_stack=measure_final_stack,
            )
            retained = list(perf.patches)
            retained_sources = list(perf.patch_srcs)
        perf.gain = final_gain
        return perf, retained, final_gain, final_gap

    def _build_optimized_deploy_package(
        self,
        spec: Spec,
        quant_ckpt_dir: str,
        perf: PerfResult,
        retained: list[str],
        final_gain: float,
        final_gap: float,
        quant_gain: float,
    ) -> DeployPackage:
        target_gain = spec.target_gain
        if target_gain is None:
            raise StageError(
                "perfopt",
                "optimized deployment requires a performance target",
            )
        target_met = final_gain >= target_gain
        retained_repositories = set(perf.patch_repos)
        framework_retained = bool(spec.active_framework_repo and spec.active_framework_repo in retained_repositories)
        kernel_retained = bool(spec.active_kernel_repo and spec.active_kernel_repo in retained_repositories)
        revert_command = self._repo_revert_command(
            spec.framework_repo,
            spec.framework_branch,
            spec.framework_version,
        )
        kernel_revert_command = self._kernel_revert_command(
            spec,
            spec.kernel_branch,
            spec.kernel_version,
        )

        if target_met:
            message = (
                f"target met: final_gain={final_gain:.3f}x >= "
                f"{target_gain}x, accuracy_gap={final_gap:.4f}, "
                f"{len(retained)} kernel patch(es) retained "
                f"(quant-only {quant_gain:.3f}x)."
            )
            status = "success"
        elif retained:
            message = (
                f"final_gain={final_gain:.3f}x < "
                f"target={target_gain}x, but {len(retained)} "
                "validated kernel patch(es) retained "
                f"(accuracy_gap={final_gap:.4f}, "
                f"quant-only {quant_gain:.3f}x). "
                f"framework_branch={spec.framework_branch or 'n/a'}, "
                f"kernel_branch={spec.kernel_branch or 'n/a'}"
            )
            status = "perf_below_target"
        else:
            message = (
                f"final_gain={final_gain:.3f}x < "
                f"target={target_gain}x; no kernel patch improved "
                f"real throughput past the {spec.keep_floor:.0%} floor "
                f"(quant-only {quant_gain:.3f}x)."
            )
            status = "perf_below_target"

        return DeployPackage(
            status=status,
            quant_ckpt_dir=quant_ckpt_dir,
            perf=perf,
            applied_patches=retained,
            framework_branch=(spec.framework_branch if framework_retained else ""),
            original_branch=(spec.framework_version if framework_retained else ""),
            revert_command=(revert_command if framework_retained else ""),
            kernel_branch=(spec.kernel_branch if kernel_retained or spec.kernel_repo else ""),
            kernel_original_branch=(spec.kernel_version if kernel_retained or spec.kernel_repo else ""),
            kernel_revert_command=kernel_revert_command,
            message=message,
        )

    def _complete_after_accuracy(
        self,
        spec: Spec,
        ckpt: Checkpoint,
        gate: AccuracyGate,
        quant_ckpt_dir: str,
        accuracy_gap: float,
    ) -> DeployPackage:
        performance_mode = spec.effective_performance_mode
        if performance_mode == "off":
            ckpt.state["performance_status"] = "not_requested"
            ckpt.state["stage"] = "done"
            ckpt.record_phase_event(
                "performance",
                "skipped",
                reason="not_requested",
            )
            ckpt.save()
            write_progress(
                spec.session_dir,
                stage="done",
                stage_detail="accuracy target met; performance not requested",
                best_so_far={
                    "accuracy_gap": accuracy_gap,
                    "perf_gain": None,
                },
            )
            return self._build_quantization_only_deploy_package(
                spec,
                quant_ckpt_dir,
                status="success",
                message="accuracy target met; performance evaluation was not requested",
            )

        _, _, quant_gain = self._measure_or_reuse_quantized_throughput(
            spec,
            ckpt,
            quant_ckpt_dir,
            accuracy_gap,
        )

        if performance_mode == "measure":
            ckpt.state["performance_status"] = "measured"
            ckpt.state["stage"] = "done"
            ckpt.save()
            write_progress(
                spec.session_dir,
                stage="done",
                stage_detail="throughput measured; optimization not requested",
                best_so_far={
                    "accuracy_gap": accuracy_gap,
                    "perf_gain": quant_gain,
                },
            )
            return self._build_quantization_only_deploy_package(
                spec,
                quant_ckpt_dir,
                status="success",
                message=f"throughput measured: quantization gain {quant_gain:.3f}x; optimization was not requested",
                perf=PerfResult(patches=[], gain=quant_gain),
            )

        if spec.target_gain is None:
            raise StageError(
                "perfopt",
                "performance_mode='optimize' requires target_gain",
            )

        if quant_gain >= spec.target_gain:
            ckpt.state["performance_status"] = "target_met"
            ckpt.state["stage"] = "done"
            ckpt.save()
            write_progress(
                spec.session_dir,
                stage="done",
                best_so_far={
                    "accuracy_gap": accuracy_gap,
                    "perf_gain": quant_gain,
                },
            )
            return self._build_quantization_only_deploy_package(
                spec,
                quant_ckpt_dir,
                status="success",
                message=(f"quantization gain {quant_gain:.3f}x >= target {spec.target_gain}x (no kernel opt needed)"),
                perf=PerfResult(patches=[], gain=quant_gain),
            )

        if self.perfopt is None:
            ckpt.state["performance_status"] = "target_not_met"
            ckpt.state["stage"] = "perf_failed"
            ckpt.save()
            write_progress(
                spec.session_dir,
                stage="perf_failed",
                best_so_far={
                    "accuracy_gap": accuracy_gap,
                    "perf_gain": quant_gain,
                },
            )
            return self._build_quantization_only_deploy_package(
                spec,
                quant_ckpt_dir,
                status="perf_below_target",
                message=(
                    f"quantization gain {quant_gain:.3f}x < target "
                    f"{spec.target_gain}x; PerfOpt is unavailable because no "
                    "modifiable framework or kernel source was configured."
                ),
            )

        perf, retained, final_gain, final_gap = self._run_perfopt_and_validate_candidates(
            spec,
            ckpt,
            gate,
            quant_ckpt_dir,
            quant_gain,
        )
        target_met = final_gain >= spec.target_gain
        ckpt.state["performance_status"] = "target_met" if target_met else "target_not_met"
        ckpt.state["stage"] = "done" if target_met else "perf_failed"
        ckpt.state["best_accuracy_gap"] = final_gap
        ckpt.save()
        write_progress(
            spec.session_dir,
            stage=("done" if target_met else "perf_failed"),
            best_so_far={
                "accuracy_gap": final_gap,
                "perf_gain": final_gain,
            },
        )
        return self._build_optimized_deploy_package(
            spec,
            quant_ckpt_dir,
            perf,
            retained,
            final_gain,
            final_gap,
            quant_gain,
        )

    def _execute_checkpointed_pipeline(
        self,
        spec: Spec,
        ckpt: Checkpoint,
    ) -> DeployPackage:
        write_progress(spec.session_dir, stage="quantize")
        terminal = self._resume_session_or_return_terminal_result(
            spec,
            ckpt,
        )
        if terminal is not None:
            return terminal
        self._prepare_managed_workspaces(spec, ckpt)
        for key, value in (ckpt.state.get("retained_runtime_env") or {}).items():
            os.environ[str(key)] = str(value)
        if spec.retry_perfopt and ckpt.state.get("stage") == "perfopt":
            gate, quant_ckpt_dir, accuracy_gap = self._restore_perfopt_retry_context(spec, ckpt)
            return self._complete_after_accuracy(
                spec,
                ckpt,
                gate,
                quant_ckpt_dir,
                accuracy_gap,
            )
        gate, baseline_terminal = self._check_baseline_before_quantization(spec, ckpt)
        if baseline_terminal is not None:
            return baseline_terminal
        quant_ckpt_dir, quantization_terminal = self._prepare_quantized_checkpoint(spec, ckpt)
        if quantization_terminal is not None:
            return quantization_terminal

        accuracy_gap, accuracy_terminal = self._evaluate_quantized_checkpoint_and_handle_fallback(
            spec,
            ckpt,
            gate,
            quant_ckpt_dir,
        )
        if accuracy_terminal is not None:
            return accuracy_terminal
        return self._complete_after_accuracy(
            spec,
            ckpt,
            gate,
            quant_ckpt_dir,
            accuracy_gap,
        )

    @staticmethod
    def _repo_revert_command(repo: str, branch: str, original: str) -> str:
        if not repo or not branch or not original or branch == original:
            return ""
        return f"git -C {repo} checkout {original} && git -C {repo} branch -D {branch}"

    @classmethod
    def _kernel_revert_command(cls, spec: Spec, branch: str, original: str) -> str:
        if (
            spec.framework_repo
            and spec.kernel_repo
            and os.path.realpath(spec.framework_repo) == os.path.realpath(spec.kernel_repo)
        ):
            return ""
        return cls._repo_revert_command(spec.kernel_repo, branch, original)

    def _cleanup_framework_repo(self, spec: Spec) -> None:
        """Restore the framework repo to its committed branch state after PerfOpt.

        GEAK swaps the repo's kernel sources to symlinks into its /tmp workspace
        and builds in-place to benchmark its authored kernels, but its teardown
        does not restore them; the greedy's per-patch `git reset` handles tracked
        typechanges but never removes untracked build artifacts (hipify headers).
        So the repo is left polluted with dangling symlinks + untracked files.
        This wipes that transient mess:
        - `git reset --hard HEAD` -> drops any uncommitted in-place source swaps
          while PRESERVING committed repair and retained optimization patches:
          reset targets the current branch tip, never the original branch.
        - `git clean -fd csrc` -> removes untracked build artifacts reset can't.
        - rebuild ONLY if csrc had pollution, so the extension .so matches the
          restored clean source (GEAK's in-place build otherwise leaves a stale
          .so); skipped when no C sources were touched to avoid a needless build.
        """
        if not spec.active_framework_repo:
            return
        if spec.workspace_source == "readonly":
            return
        if spec.framework_repo and not spec.framework_worktree:
            raise WorkspaceError("framework cleanup requires an Quark Quant-Perf integration worktree")
        from quark.experimental.torch.quant_perf.perfopt.kernel_workflow import rebuild_framework

        repo = spec.active_framework_repo
        had_c_pollution = worktree_dirty(repo, "csrc")
        head = get_head_sha(repo)
        if head:
            reset_hard_to(repo, head)
        clean_untracked(repo, "csrc")
        if had_c_pollution:
            logger.info(
                "[cleanup] GEAK left in-place C-source pollution; rebuilding "
                "extension so the deployed .so matches the clean source"
            )
            rebuild_framework(repo)
        logger.info("[cleanup] framework repo restored to committed branch state")

    def _on_quantize_fail(
        self,
        spec: Spec,
        ckpt: Checkpoint,
        outcome: dict[str, Any],
    ) -> DeployPackage:
        """direct_ptq's skill run failed to produce valid artifacts (§4.6:
        Quark Quant-Perf's own criterion, not the skill's self-reported status)."""
        ckpt.state["stage"] = "failed"
        ckpt.record_phase_event(
            "quantization",
            "failed",
            reason=outcome.get("agent_summary", ""),
        )
        ckpt.save()
        write_progress(spec.session_dir, stage="failed", warning="quantize: no valid artifacts")
        return DeployPackage(
            status="accuracy_failed",
            message=f"quantization did not produce valid artifacts: {outcome.get('agent_summary', '')}",
        )

    def _on_accuracy_fail(self, spec: Spec, ckpt: Checkpoint, gap: float) -> DeployPackage:
        attempts = ckpt.state.get("accuracy_attempts") or []
        attempt = (
            attempts[-1]
            if attempts
            else {
                "gap": gap,
                "passed": False,
                "candidate": ckpt.state.get("best_candidate"),
            }
        )
        retry = self._retry_next_search_candidate(
            spec,
            ckpt,
            attempt=attempt,
            stage_detail="real accuracy rejected candidate; exporting next persisted Quark result",
        )
        if retry is not None:
            return retry
        ckpt.state["stage"] = "failed"
        ckpt.save()
        write_progress(spec.session_dir, stage="failed", warning=f"accuracy gap {gap:.4f} exceeded threshold")
        return DeployPackage(
            status="accuracy_failed",
            message=f"accuracy gap {gap:.4f} exceeds threshold {spec.accuracy_gap}",
        )

    @staticmethod
    def _should_reject_search_candidate_for(
        error: StageError,
        spec: Spec,
    ) -> bool:
        if error.stage != "land" or not error.code:
            return False
        diagnosis = classify_failure(
            error,
            framework_repo=spec.active_framework_repo,
            kernel_repo=spec.active_kernel_repo,
        )
        return diagnosis.failure_class not in {
            "resource_contention",
            "resource_capacity",
            "transient",
            "timeout",
        }

    @staticmethod
    def _retry_next_search_candidate(
        spec: Spec,
        ckpt: Checkpoint,
        *,
        attempt: dict[str, Any],
        stage_detail: str,
        prefer_minimal_change: bool = False,
    ) -> DeployPackage | None:
        if ckpt.state.get("path") != "mix_precision_search":
            return None
        next_candidate = advance_search_candidate(
            ckpt.state,
            attempt=attempt,
            prefer_minimal_change=prefer_minimal_change,
        )
        if next_candidate is None:
            return None
        ckpt.state["stage"] = "quantize"
        ckpt.state["quant_ckpt_dir"] = None
        ckpt.state["accuracy_validation"] = None
        event = {
            "rejected_candidate": attempt.get("candidate"),
            "next_candidate": next_candidate,
        }
        for key in ("reason", "code"):
            if attempt.get(key):
                event[key] = attempt[key]
        ckpt.record_phase_event(
            "accuracy_candidate_fallback",
            "retry",
            **event,
        )
        ckpt.save()
        write_progress(
            spec.session_dir,
            stage="quantize",
            stage_detail=stage_detail,
        )
        return DeployPackage(status="_retry_search_candidate")

    def _on_base_unhealthy(self, spec: Spec, ckpt: Checkpoint, diagnosis: str) -> DeployPackage:
        """The BASE (unquantized) model does not run correctly in the framework,
        so the accuracy gap and throughput gain are undefined -- abort before any
        quantized measurement. This is a framework/model-support problem upstream
        of Quark Quant-Perf; reported for the user to fix (or retried once when
        --fix-base-framework is set). Distinct from accuracy_failed, which would
        wrongly imply quantization hurt accuracy."""
        ckpt.state["stage"] = "base_unhealthy"
        ckpt.state["base_unhealthy_diag"] = diagnosis
        ckpt.save()
        write_progress(spec.session_dir, stage="failed", warning=f"base model unhealthy: {diagnosis}")
        logger.error("[orchestrator] base model unhealthy in %s -- aborting: %s", spec.framework, diagnosis)
        return DeployPackage(status="base_unhealthy", message=diagnosis)
