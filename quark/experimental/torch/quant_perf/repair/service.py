#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quark.experimental.torch.quant_perf.session.progress import write_progress
from quark.experimental.torch.quant_perf.workspace.git import (
    reset_hard_to,
)
from quark.experimental.torch.quant_perf.workspace.manager import RepoWorkspaceManager
from quark.experimental.torch.quant_perf.workspace.sources import compiled_changes

from . import llm_repair
from .artifacts import export_patch
from .evidence import extract_failure_evidence, render_failure_evidence, save_failure_evidence
from .knowledge_context import build_repair_knowledge
from .source_router import resolve_repair_target
from .types import FailureEvidence, RepairKnowledgeQuery, RepairRequest, RepairResult, VerificationResult
from .verifiers import (
    DEFAULT_VERIFIER_TIMEOUT_S,
    load_inference_timeout_s,
    verify_load_and_inference,
)

if TYPE_CHECKING:
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore

RepairVerifier = Callable[[str, RepairRequest, int], tuple[bool, str]]


class RepairService:
    """Coordinate isolated replay, LLM repair, verification, and promotion."""

    def __init__(
        self,
        experience_store: ExperienceStore | None = None,
        workspace_manager: RepoWorkspaceManager | None = None,
        verifier: RepairVerifier | None = None,
        allow_in_place_repair: bool = False,
    ) -> None:
        self.experience_store = experience_store
        self.workspace_manager = workspace_manager
        self.verifier = verifier
        self.allow_in_place_repair = allow_in_place_repair

    @staticmethod
    def can_repair(request: RepairRequest) -> bool:
        target = resolve_repair_target(request)
        return bool(target.repo)

    def repair(self, request: RepairRequest) -> RepairResult:
        started = time.monotonic()
        if self.workspace_manager is None and not self.allow_in_place_repair:
            raise RuntimeError(
                "RepairService requires a managed workspace; "
                "set allow_in_place_repair=True only for an isolated disposable checkout"
            )
        if request.session_dir:
            request.evidence = save_failure_evidence(
                extract_failure_evidence(request),
                Path(request.session_dir) / "repair" / f"failure-{time.time_ns()}",
            )
        target = resolve_repair_target(request)
        candidate_id = ""
        promoted = False
        attempts: list[dict[str, Any]] = []
        artifacts: list[dict[str, Any]] = []
        verifier_results: list[VerificationResult] = []
        knowledge_ids: list[str] = []
        changed_files: list[str] = []

        def knowledge_for_round(round_num: int, evidence: FailureEvidence) -> tuple[str, list[str]]:
            text, ids = self._knowledge_for_round(request, round_num=round_num, evidence=evidence)
            knowledge_ids.extend(item for item in ids if item not in knowledge_ids)
            return text, ids

        if self.workspace_manager is None:
            baseline_sha = self._repo_head(target.repo)
            fixed, verification = self._dispatch(
                request,
                target.repo,
                target.role,
                knowledge_for_round=knowledge_for_round,
                attempts_out=attempts,
            )
            if verification is not None:
                verifier_results.append(verification)
            if fixed and baseline_sha:
                patch_path, patch_files = export_patch(
                    target.repo,
                    baseline_sha,
                    session_dir=request.session_dir,
                    label=f"{request.failure_class}-repair",
                )
                changed_files = patch_files
                if patch_path:
                    artifacts.append(
                        {
                            "kind": "repair_patch",
                            "patch_path": patch_path,
                            "changed_files": changed_files,
                        }
                    )
        else:
            candidate_id = (
                f"{request.failure_class}-"
                f"{hashlib.sha256(extract_failure_evidence(request).signature.encode()).hexdigest()[:12]}"
            )
            with self.workspace_manager.candidate(
                target.role,
                candidate_id,
                intent="repair",
            ) as candidate:
                with self._candidate_pythonpath(candidate.path):
                    fixed, verification = self._dispatch(
                        request,
                        str(candidate.path),
                        target.role,
                        knowledge_for_round=knowledge_for_round,
                        attempts_out=attempts,
                    )
                    if verification is not None:
                        verifier_results.append(verification)
                    if fixed:
                        patch_path, patch_files = export_patch(
                            str(candidate.path),
                            candidate.base_sha,
                            session_dir=request.session_dir,
                            label=f"{request.failure_class}-repair",
                        )
                        changed_files = patch_files
                        if patch_path:
                            artifacts.append(
                                {
                                    "kind": "repair_patch",
                                    "patch_path": patch_path,
                                    "changed_files": changed_files,
                                }
                            )
                if fixed:
                    source_kind = str(request.stack_fingerprint.get(f"{target.role}_source_kind") or "")
                    compiled = compiled_changes(
                        candidate.path,
                        candidate.base_sha,
                    )
                    if source_kind == "installed_overlay" and compiled:
                        reset_hard_to(
                            str(candidate.path),
                            candidate.base_sha,
                        )
                        fixed = False
                        attempts.append(
                            {
                                "kind": "source_capability_gate",
                                "outcome": "source_repo_required",
                                "changed_files": compiled,
                            }
                        )
                    else:
                        promoted = self.workspace_manager.promote_candidate(
                            candidate,
                            message=f"Quark Quant-Perf {request.failure_class} repair",
                        )
                        if not promoted:
                            fixed = False
                            attempts.append(
                                {
                                    "kind": "source_capability_gate",
                                    "outcome": "no_source_change",
                                }
                            )
        result = RepairResult(
            status="fixed" if fixed else "not_fixed",
            target_repos=[target.repo],
            attempts=attempts,
            artifacts=artifacts,
            verifier_results=verifier_results,
            knowledge_ids=knowledge_ids,
        )
        self._record_journey(
            request=request,
            target_role=target.role,
            target_repo=target.repo,
            candidate_id=candidate_id,
            result=result,
            promoted=promoted,
            changed_files=changed_files,
            elapsed_seconds=time.monotonic() - started,
        )
        return result

    @staticmethod
    def _repo_head(repo: str) -> str:
        from quark.experimental.torch.quant_perf.workspace.git import get_head_sha

        if not repo or not os.path.isdir(repo):
            return ""
        return get_head_sha(repo)

    def _verify(
        self,
        candidate_repo: str,
        request: RepairRequest,
        *,
        timeout_s: int,
    ) -> tuple[bool, str]:
        result = self._verification_result(
            candidate_repo,
            request,
            timeout_s=timeout_s,
        )
        return result.passed, result.failure

    def _verification_result(
        self,
        candidate_repo: str,
        request: RepairRequest,
        *,
        timeout_s: int,
    ) -> VerificationResult:
        diagnostic_dir = (
            Path(request.session_dir) / "repair" / f"verification-{time.time_ns()}" if request.session_dir else None
        )
        try:
            result = self._run_verification(candidate_repo, request, timeout_s=timeout_s, diagnostic_dir=diagnostic_dir)
        except Exception as error:
            evidence = extract_failure_evidence(error)
            result = VerificationResult(request.verifier_profile, False, failure=str(error), evidence=evidence)
        paths = tuple(str(path) for path in sorted(diagnostic_dir.rglob("*.log"))) if diagnostic_dir else ()
        if result.passed:
            return replace(result, evidence_paths=paths)
        evidence = result.evidence or extract_failure_evidence(result.failure)
        evidence = replace(evidence, evidence_paths=paths)
        if diagnostic_dir is not None:
            evidence = save_failure_evidence(evidence, diagnostic_dir)
        return replace(
            result, failure=render_failure_evidence(evidence), evidence=evidence, evidence_paths=evidence.evidence_paths
        )

    def _run_verification(
        self,
        candidate_repo: str,
        request: RepairRequest,
        *,
        timeout_s: int,
        diagnostic_dir: Path | None,
    ) -> VerificationResult:
        if self.verifier is not None:
            passed, failure = self.verifier(
                candidate_repo,
                request,
                timeout_s,
            )
            return VerificationResult(
                verifier=request.verifier_profile,
                passed=passed,
                failure=failure,
            )
        if request.verifier_profile == "accuracy":
            from quark.experimental.torch.quant_perf.evaluation.gsm8k import gsm8k_eval_offline

            source = float(request.metrics["source_gsm8k"])
            score = gsm8k_eval_offline(
                request.quant_ckpt_dir,
                gpu_id=int(request.workload.get("gpu_id") or 0),
                num_questions=int(request.immutable_constraints.get("accuracy_repair_num_samples", 50)),
                tp=int(request.workload.get("tp") or 1),
                gpu_memory_utilization=float(request.immutable_constraints.get("gpu_memory_utilization", 0.85)),
                profile=request.immutable_constraints.get("eval_profile"),
                trust_remote_code=bool(request.immutable_constraints.get("trust_remote_code", False)),
                max_num_seqs=request.immutable_constraints.get("max_num_seqs"),
                runtime_python=str(request.immutable_constraints.get("runtime_python") or ""),
                kv_cache_dtype=request.immutable_constraints.get("kv_cache_dtype"),
                runtime_env=self._candidate_runtime_env(candidate_repo, request),
            )
            gap = max(0.0, (source - score) / max(source, 1e-9))
            threshold = float(request.immutable_constraints.get("accuracy_gap", 0.0))
            passed = gap <= threshold
            return VerificationResult(
                verifier="accuracy",
                passed=passed,
                metrics={
                    "score": score,
                    "gap": gap,
                    "threshold": threshold,
                },
                failure=("" if passed else (f"accuracy gap {gap:.4f} exceeds threshold {threshold:.4f}")),
            )
        if request.verifier is not None:
            passed, failure = request.verifier()
            return VerificationResult(
                verifier=request.verifier_profile,
                passed=passed,
                failure=failure,
            )
        if request.verifier_profile == "load_inference":
            runtime_env = self._candidate_runtime_env(
                candidate_repo,
                request,
            )
            gpu_memory_utilization = float(
                request.immutable_constraints.get(
                    "gpu_memory_utilization",
                    0.85,
                )
            )
            max_model_len = int(
                request.immutable_constraints.get(
                    "max_model_len",
                    4096,
                )
            )
            common = {
                "quant_ckpt_dir": request.quant_ckpt_dir,
                "gpu_id": int(request.workload.get("gpu_id") or 0),
                "tp": int(request.workload.get("tp") or 1),
                "max_model_len": max_model_len,
                "max_num_seqs": request.immutable_constraints.get("max_num_seqs"),
                "trust_remote_code": bool(request.immutable_constraints.get("trust_remote_code", False)),
                "timeout_s": timeout_s,
                "runtime_python": str(request.immutable_constraints.get("runtime_python") or ""),
                "kv_cache_dtype": request.immutable_constraints.get("kv_cache_dtype"),
                "runtime_env": runtime_env,
            }
            original_enforce_eager = bool(
                request.immutable_constraints.get(
                    "original_enforce_eager",
                    False,
                )
            )
            # Eager mode still loads the same weights and needs KV cache space.
            # Preserve the runtime budget so large models can reach inference.
            eager_passed, eager_failure = verify_load_and_inference(
                **common,
                enforce_eager=True,
                gpu_memory_utilization=gpu_memory_utilization,
                diagnostic_dir=diagnostic_dir / "eager" if diagnostic_dir else None,
            )
            checks = [
                {
                    "mode": "eager",
                    "passed": eager_passed,
                    "failure": render_failure_evidence(extract_failure_evidence(eager_failure))
                    if eager_failure
                    else "",
                }
            ]
            if not eager_passed:
                return VerificationResult(
                    verifier="load_inference",
                    passed=False,
                    metrics={"checks": checks},
                    failure=f"eager runtime: {eager_failure}",
                )

            original_passed: bool
            original_failure: str
            if original_enforce_eager:
                original_passed, original_failure = (
                    eager_passed,
                    eager_failure,
                )
            else:
                original_passed, original_failure = verify_load_and_inference(
                    **common,
                    enforce_eager=False,
                    gpu_memory_utilization=gpu_memory_utilization,
                    diagnostic_dir=diagnostic_dir / "original" if diagnostic_dir else None,
                )
            checks.append(
                {
                    "mode": "original",
                    "passed": original_passed,
                    "failure": render_failure_evidence(extract_failure_evidence(original_failure))
                    if original_failure
                    else "",
                }
            )
            return VerificationResult(
                verifier="load_inference",
                passed=original_passed,
                metrics={"checks": checks},
                failure=("" if original_passed else f"original runtime: {original_failure}"),
            )
        raise ValueError(f"unsupported verifier profile: {request.verifier_profile}")

    @staticmethod
    def _candidate_runtime_env(
        candidate_repo: str,
        request: RepairRequest,
    ) -> dict[str, str]:
        runtime_env = {
            str(key): str(value) for key, value in (request.immutable_constraints.get("runtime_env") or {}).items()
        }
        entries = [
            candidate_repo,
            *[
                entry
                for entry in runtime_env.get(
                    "PYTHONPATH",
                    "",
                ).split(os.pathsep)
                if entry and entry != candidate_repo
            ],
        ]
        runtime_env["PYTHONPATH"] = os.pathsep.join(entries)
        return runtime_env

    @staticmethod
    @contextmanager
    def _candidate_pythonpath(
        path: str | Path,
    ) -> Iterator[None]:
        previous = os.environ.get("PYTHONPATH")
        entries = [str(path)]
        if previous:
            entries.append(previous)
        os.environ["PYTHONPATH"] = os.pathsep.join(entries)
        try:
            yield
        finally:
            if previous is None:
                os.environ.pop("PYTHONPATH", None)
            else:
                os.environ["PYTHONPATH"] = previous

    def _dispatch(
        self,
        request: RepairRequest,
        target_repo: str,
        target_role: str,
        *,
        knowledge_for_round: RepairKnowledgeQuery,
        attempts_out: list[dict[str, Any]],
    ) -> tuple[bool, VerificationResult | None]:
        workload = request.workload
        candidate_runtime_env = self._candidate_runtime_env(
            target_repo,
            request,
        )
        verification: VerificationResult | None = None

        def verify() -> tuple[bool, str | FailureEvidence]:
            nonlocal verification
            timeout_s = (
                load_inference_timeout_s(request.quant_ckpt_dir)
                if request.verifier_profile == "load_inference"
                else DEFAULT_VERIFIER_TIMEOUT_S
            )
            verification = self._verification_result(target_repo, request, timeout_s=timeout_s)
            return verification.passed, verification.evidence or verification.failure

        def record_phase(event: dict[str, Any]) -> None:
            if self.workspace_manager is not None:
                ckpt = self.workspace_manager.ckpt
                ckpt.record_phase_event("repair", **event)
                ckpt.save()

        common = {
            "error": extract_failure_evidence(request),
            "framework": request.framework,
            "framework_repo": target_repo,
            "quant_ckpt_dir": request.quant_ckpt_dir,
            "arch_fingerprint": str(request.stack_fingerprint.get("arch_fingerprint") or ""),
            "framework_version": str(request.stack_fingerprint.get("framework_version") or ""),
            "session_dir": request.session_dir,
            "knowledge_for_round": knowledge_for_round,
            "attempts_out": attempts_out,
            "phase_callback": record_phase,
        }
        if request.failure_class == "load_run":
            fixed = llm_repair.attempt_load_repair(
                **common,
                role=target_role,
                gpu_id=int(workload.get("gpu_id") or 0),
                tp=int(workload.get("tp") or 1),
                quant_signature=request.quant_signature,
                verify=verify,
                runtime_python=str(request.immutable_constraints.get("runtime_python") or ""),
                kv_cache_dtype=request.immutable_constraints.get("kv_cache_dtype"),
                runtime_env=candidate_runtime_env,
            )
        elif request.failure_class == "benchmark_execution":
            if request.verifier is None and self.verifier is None:
                raise ValueError("benchmark_execution repair requires verifier")
            fixed = llm_repair.attempt_benchmark_repair(
                **common,
                tp=int(workload.get("tp") or 1),
                isl=int(workload.get("isl") or 0),
                osl=int(workload.get("osl") or 0),
                concurrency=int(workload.get("concurrency") or 0),
                verify=verify,
                role=target_role,
                runtime_fingerprint=request.quant_signature,
            )
        elif request.failure_class == "accuracy_gap":
            fixed = llm_repair.attempt_accuracy_repair(
                source_gsm8k=float(request.metrics["source_gsm8k"]),
                quantized_gsm8k=float(request.metrics["quantized_gsm8k"]),
                gap=float(request.metrics["gap"]),
                framework=request.framework,
                framework_repo=target_repo,
                quant_ckpt_dir=request.quant_ckpt_dir,
                gpu_id=int(workload.get("gpu_id") or 0),
                tp=int(workload.get("tp") or 1),
                arch_fingerprint=str(request.stack_fingerprint.get("arch_fingerprint") or ""),
                framework_version=str(request.stack_fingerprint.get("framework_version") or ""),
                quant_signature=request.quant_signature,
                eval_profile=request.immutable_constraints.get("eval_profile"),
                trust_remote_code=bool(request.immutable_constraints.get("trust_remote_code", False)),
                max_num_seqs=request.immutable_constraints.get("max_num_seqs"),
                gpu_memory_utilization=float(
                    request.immutable_constraints.get(
                        "gpu_memory_utilization",
                        0.85,
                    )
                ),
                session_dir=request.session_dir,
                knowledge_for_round=knowledge_for_round,
                verify_callback=verify,
                runtime_python=str(request.immutable_constraints.get("runtime_python") or ""),
                kv_cache_dtype=request.immutable_constraints.get("kv_cache_dtype"),
                runtime_env=candidate_runtime_env,
                attempts_out=attempts_out,
                phase_callback=record_phase,
            )
        else:
            raise ValueError(f"unsupported repair failure_class: {request.failure_class}")
        return fixed, verification

    def _knowledge_for_round(
        self,
        request: RepairRequest,
        *,
        round_num: int,
        evidence: FailureEvidence,
    ) -> tuple[str, list[str]]:
        stage, consumer = {
            "load_run": ("land", "runtime_repair"),
            "benchmark_execution": (
                "benchmark",
                "benchmark_execution_repair",
            ),
            "accuracy_gap": ("accuracy", "accuracy_repair"),
        }[request.failure_class]
        state = self.workspace_manager.ckpt.state if self.workspace_manager is not None else None
        bundle, rendered = build_repair_knowledge(
            experience_store=self.experience_store,
            state=state,
            session_dir=request.session_dir,
            consumer=consumer,
            stage=stage,
            framework=request.framework,
            framework_version=str(request.stack_fingerprint.get("framework_version") or ""),
            arch_fingerprint=str(request.stack_fingerprint.get("arch_fingerprint") or ""),
            quant_signature=request.quant_signature,
            failure_class=request.failure_class,
            error_signature=evidence.signature,
            error_text=f"{evidence.exception_type}: {evidence.exception_message}\n{evidence.root_source}",
            workload=request.workload,
            round_id=round_num,
        )
        return rendered, list(bundle.source_ids)

    def _record_journey(
        self,
        *,
        request: RepairRequest,
        target_role: str,
        target_repo: str,
        candidate_id: str,
        result: RepairResult,
        promoted: bool,
        changed_files: list[str],
        elapsed_seconds: float,
    ) -> None:
        rounds = max((int(attempt.get("round", 0)) for attempt in result.attempts), default=0)
        if request.session_dir:
            write_progress(
                Path(request.session_dir),
                detail=f"Repair {result.status} after {rounds} rounds",
                repair={
                    "phase": "finished",
                    "status": result.status,
                    "round": rounds,
                    "target_role": target_role,
                    "total_elapsed_seconds": elapsed_seconds,
                },
            )
        if self.workspace_manager is None:
            return
        ckpt = self.workspace_manager.ckpt
        state = ckpt.state
        evidence = extract_failure_evidence(request)
        state.setdefault("repair_journey", []).append(
            {
                "failure_class": request.failure_class,
                "status": result.status,
                "target_role": target_role,
                "target_repo": target_repo,
                "candidate_id": candidate_id,
                "error": render_failure_evidence(evidence, limit=2000),
                "error_signature": evidence.signature,
                "failure_evidence": {key: value for key, value in asdict(evidence).items() if key != "full_error"},
                "quant_signature": request.quant_signature,
                "knowledge_ids": list(result.knowledge_ids),
                "attempts": list(result.attempts),
                "rounds": rounds,
                "elapsed_seconds": elapsed_seconds,
                "verifier_results": [
                    {
                        "verifier": verification.verifier,
                        "passed": verification.passed,
                        "metrics": dict(verification.metrics),
                        "evidence_paths": list(verification.evidence_paths),
                        "failure": verification.failure,
                    }
                    for verification in result.verifier_results
                ],
                "promoted": promoted,
                "changed_files": list(changed_files),
            }
        )
        if promoted:
            state.setdefault("change_ledger", []).append(
                {
                    "intent": "repair",
                    "candidate_id": candidate_id,
                    "target_role": target_role,
                    "target_repo": target_repo,
                    "status": "promoted",
                }
            )
        ckpt.save()
