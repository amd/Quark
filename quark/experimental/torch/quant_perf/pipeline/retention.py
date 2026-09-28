#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Real accuracy and end-to-end validation for optimization candidates."""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.evaluation.gate import AccuracyGate
from quark.experimental.torch.quant_perf.evaluation.throughput import ThroughputMeasurement
from quark.experimental.torch.quant_perf.perfopt.bottleneck_analysis import (
    bottleneck_candidates_from_state,
)
from quark.experimental.torch.quant_perf.perfopt.journey import (
    record_candidate_retention_decision,
)
from quark.experimental.torch.quant_perf.perfopt.keep import kernel_record_id
from quark.experimental.torch.quant_perf.perfopt.kernel_workflow import (
    apply_prepared_kernel_patch,
    rebuild_framework,
)
from quark.experimental.torch.quant_perf.pipeline.candidate_validation import (
    evaluate_candidate_accuracy,
    measure_quantized_candidate,
    run_candidate_retest,
    validate_runtime_environment_candidates,
)
from quark.experimental.torch.quant_perf.pipeline.performance_policy import RetestDisposition
from quark.experimental.torch.quant_perf.session.spec import Checkpoint, PerfResult, Spec, StageError
from quark.experimental.torch.quant_perf.workspace.git import (
    commit_selected_changes,
    get_head_sha,
    reset_hard_to,
)
from quark.experimental.torch.quant_perf.workspace.manager import WorkspaceError


def _restore_repo_revision(
    repo: str,
    revision: str,
    *,
    requires_rebuild: bool,
) -> None:
    """Restore source and compiled artifacts to the same repository revision."""
    reset_hard_to(repo, revision)
    if not requires_rebuild:
        return
    rebuilt, message = rebuild_framework(repo)
    if not rebuilt:
        raise StageError(
            "perfopt",
            f"rollback rebuild failed: {message}",
        )


class CandidateRetentionService:
    """Apply final accuracy and E2E retention policy to all candidates."""

    @staticmethod
    def _measure_quantized_checkpoint(
        spec: Spec,
        model_dir: str,
    ) -> ThroughputMeasurement:
        return measure_quantized_candidate(spec, model_dir)

    @staticmethod
    def _evaluate_quantized_checkpoint_accuracy(
        spec: Spec,
        model_dir: str,
    ) -> tuple[str, float | None, str]:
        return evaluate_candidate_accuracy(spec, model_dir)

    def validate_and_retain_candidates(
        self,
        perf: PerfResult,
        spec: Spec,
        quant_gain: float,
        quant_ckpt_dir: str,
        gate: AccuracyGate,
        ckpt: Checkpoint,
    ) -> tuple[list[str], list[str], float, float]:
        """Validate runtime candidates, then source patch candidates."""
        ckpt.state["retention_stack"] = []
        ckpt.state["retention_base"] = {
            "gain": quant_gain,
            "accuracy_gap": (ckpt.state.get("best_accuracy_gap") or 0.0),
        }
        (
            _,
            best_gain,
            best_gap,
            runtime_entries,
        ) = validate_runtime_environment_candidates(
            perf,
            spec,
            quant_gain=quant_gain,
            quant_ckpt_dir=quant_ckpt_dir,
            gate=gate,
            ckpt=ckpt,
        )
        ckpt.state["retention_stack"].extend(runtime_entries)
        ckpt.save()
        return self._validate_kernel_patch_candidates(
            perf,
            spec,
            quant_ckpt_dir=quant_ckpt_dir,
            gate=gate,
            ckpt=ckpt,
            initial_gain=best_gain,
            initial_gap=best_gap,
        )

    def _validate_kernel_patch_candidates(
        self,
        perf: PerfResult,
        spec: Spec,
        *,
        quant_ckpt_dir: str,
        gate: AccuracyGate,
        ckpt: Checkpoint,
        initial_gain: float,
        initial_gap: float,
    ) -> tuple[list[str], list[str], float, float]:
        retained: list[str] = []
        retained_srcs: list[str] = []
        retained_repos: list[str] = []
        best_gain = initial_gain
        best_gap = initial_gap
        if not perf.patches:
            perf.patch_repos = []
            return retained, retained_srcs, best_gain, best_gap

        baseline_score = ckpt.state.get("baseline_gsm8k") or gate._source_cache or 1e-9
        repos = perf.patch_repos or [spec.active_framework_repo or spec.active_kernel_repo] * len(perf.patches)
        active_worktrees = {
            os.path.realpath(path)
            for path in (
                spec.framework_worktree,
                spec.kernel_worktree,
            )
            if path
        }
        invalid_repos = [repo for repo in repos if repo and os.path.realpath(repo) not in active_worktrees]
        if invalid_repos:
            raise WorkspaceError(
                "kernel patch retention requires an Quark Quant-Perf integration "
                f"worktree, not a user source repository: {invalid_repos[0]}"
            )
        base_shas = {repo: get_head_sha(repo) for repo in repos if repo}
        speeds = {
            entry.get("best_patch"): entry.get("verified_speedup")
            for entry in (ckpt.state.get("geak_patches") or [])
            if entry.get("status") == "candidate"
        }
        kernel_ids = {
            entry.get("best_patch"): kernel_record_id(entry)
            for entry in (ckpt.state.get("geak_patches") or [])
            if entry.get("best_patch")
        }
        patch_metadata = {
            entry.get("best_patch"): entry
            for entry in (ckpt.state.get("geak_patches") or [])
            if entry.get("best_patch")
        }
        srcs = perf.patch_srcs or [""] * len(perf.patches)
        ordered_candidates = sorted(
            zip(perf.patches, srcs, repos, strict=False),
            key=lambda item: (
                speeds.get(item[0]) is not None,
                float(speeds.get(item[0]) or 0.0),
            ),
            reverse=True,
        )
        ranks = {kernel_record_id(row): row.get("rank") for row in bottleneck_candidates_from_state(ckpt.state)}

        eligible_candidates = []
        for patch, source, repo in ordered_candidates:
            metadata = patch_metadata.get(patch) or {}
            changed_files = [str(path) for path in (metadata.get("patch_changed_files") or []) if path]
            if not changed_files:
                record_candidate_retention_decision(
                    ckpt,
                    kernel_ids,
                    patch=patch,
                    source=source,
                    repo=repo,
                    decision="SKIP",
                    reason="patch_metadata_missing",
                )
                continue
            needs_rebuild = bool(metadata.get("patch_requires_rebuild"))
            source_kind = (
                spec.kernel_source_kind
                if (spec.active_kernel_repo and os.path.realpath(repo) == os.path.realpath(spec.active_kernel_repo))
                else spec.framework_source_kind
            )
            if source_kind == "installed_overlay" and needs_rebuild:
                record_candidate_retention_decision(
                    ckpt,
                    kernel_ids,
                    patch=patch,
                    source=source,
                    repo=repo,
                    decision="DROP",
                    reason="source_repo_required",
                )
                continue
            eligible_candidates.append(
                (
                    patch,
                    source,
                    repo,
                    changed_files,
                    needs_rebuild,
                )
            )

        if not eligible_candidates:
            perf.patch_repos = []
            return retained, retained_srcs, best_gain, best_gap

        try:
            current_anchor = self._measure_quantized_checkpoint(
                spec,
                quant_ckpt_dir,
            )
        except Exception as exc:
            raise StageError(
                "perfopt",
                f"fresh current-stack anchor failed: {str(exc)[-2000:]}",
            ) from exc

        for (
            patch,
            source,
            target_repo,
            changed_files,
            needs_rebuild,
        ) in eligible_candidates:
            (
                kept,
                current_anchor,
                candidate_gain,
                candidate_gap,
            ) = self._validate_kernel_patch_candidate(
                spec,
                ckpt,
                quant_ckpt_dir=quant_ckpt_dir,
                baseline_score=baseline_score,
                patch=patch,
                source=source,
                target_repo=target_repo,
                base_shas=base_shas,
                kernel_ids=kernel_ids,
                ranks=ranks,
                speeds=speeds,
                current_anchor=current_anchor,
                current_gain=best_gain,
                current_gap=best_gap,
                changed_files=changed_files,
                needs_rebuild=needs_rebuild,
            )
            if not kept:
                continue
            best_gain = candidate_gain
            best_gap = candidate_gap
            retained.append(patch)
            retained_srcs.append(source)
            retained_repos.append(target_repo)

        perf.patch_repos = retained_repos
        return retained, retained_srcs, best_gain, best_gap

    def _validate_kernel_patch_candidate(
        self,
        spec: Spec,
        ckpt: Checkpoint,
        *,
        quant_ckpt_dir: str,
        baseline_score: float,
        patch: str,
        source: str,
        target_repo: str,
        base_shas: dict[str, str],
        kernel_ids: dict[str, str],
        ranks: dict[str, int],
        speeds: dict[str, float | None],
        current_anchor: ThroughputMeasurement,
        current_gain: float,
        current_gap: float,
        changed_files: list[str],
        needs_rebuild: bool,
    ) -> tuple[bool, ThroughputMeasurement, float, float]:
        def revert(needs_rebuild: bool) -> None:
            _restore_repo_revision(
                target_repo,
                base_shas[target_repo],
                requires_rebuild=needs_rebuild,
            )

        if not patch or not Path(patch).exists():
            record_candidate_retention_decision(
                ckpt,
                kernel_ids,
                patch=patch,
                source=source,
                repo=target_repo,
                decision="SKIP",
                reason="patch_missing",
            )
            return False, current_anchor, current_gain, 0.0
        if not source:
            record_candidate_retention_decision(
                ckpt,
                kernel_ids,
                patch=patch,
                source=source,
                repo=target_repo,
                decision="SKIP",
                reason="source_missing",
            )
            return False, current_anchor, current_gain, 0.0
        if not target_repo:
            record_candidate_retention_decision(
                ckpt,
                kernel_ids,
                patch=patch,
                source=source,
                repo=target_repo,
                decision="SKIP",
                reason="repo_missing",
            )
            return False, current_anchor, current_gain, 0.0

        applied, apply_message = apply_prepared_kernel_patch(
            patch,
            target_repo,
        )
        if not applied:
            record_candidate_retention_decision(
                ckpt,
                kernel_ids,
                patch=patch,
                source=source,
                repo=target_repo,
                decision="SKIP",
                reason=f"apply_failed: {apply_message}",
            )
            return False, current_anchor, current_gain, 0.0

        if needs_rebuild:
            rebuilt, rebuild_message = rebuild_framework(target_repo)
            if not rebuilt:
                revert(True)
                record_candidate_retention_decision(
                    ckpt,
                    kernel_ids,
                    patch=patch,
                    source=source,
                    repo=target_repo,
                    decision="RETRYABLE_FAULT",
                    reason=f"rebuild_failed: {rebuild_message}",
                )
                return False, current_anchor, current_gain, 0.0

        accuracy_status, score, accuracy_reason = self._evaluate_quantized_checkpoint_accuracy(
            spec,
            quant_ckpt_dir,
        )
        if accuracy_status != "passed" or score is None:
            revert(needs_rebuild)
            record_candidate_retention_decision(
                ckpt,
                kernel_ids,
                patch=patch,
                source=source,
                repo=target_repo,
                decision=("NEEDS_REVIEW" if accuracy_status == "needs_review" else "RETRYABLE_FAULT"),
                reason=accuracy_reason,
                score=score,
            )
            return False, current_anchor, current_gain, 0.0

        gap = max(
            0.0,
            (baseline_score - score) / max(baseline_score, 1e-9),
        )
        if gap > spec.accuracy_gap:
            revert(needs_rebuild)
            record_candidate_retention_decision(
                ckpt,
                kernel_ids,
                patch=patch,
                source=source,
                repo=target_repo,
                score=score,
                gap=gap,
                decision="DROP_CONFIRMED",
                reason="accuracy_regression",
            )
            return False, current_anchor, current_gain, gap

        try:
            candidate_first = self._measure_quantized_checkpoint(
                spec,
                quant_ckpt_dir,
            )
        except Exception as exc:
            revert(needs_rebuild)
            record_candidate_retention_decision(
                ckpt,
                kernel_ids,
                patch=patch,
                source=source,
                repo=target_repo,
                score=score,
                gap=gap,
                decision="RETRYABLE_FAULT",
                reason=f"throughput_error:{str(exc)[-1000:]}",
            )
            return False, current_anchor, current_gain, gap

        try:
            retest = run_candidate_retest(
                current_anchor,
                candidate_first,
                keep_floor=spec.keep_floor,
                trace_rank=ranks.get(kernel_ids.get(patch, "")),
                micro_speedup=speeds.get(patch),
                deactivate=lambda: revert(needs_rebuild),
                measure_anchor=lambda: self._measure_quantized_checkpoint(
                    spec,
                    quant_ckpt_dir,
                ),
                measure_candidate=lambda: self._measure_quantized_checkpoint(
                    spec,
                    quant_ckpt_dir,
                ),
            )
        except StageError:
            raise
        except Exception as exc:
            revert(needs_rebuild)
            record_candidate_retention_decision(
                ckpt,
                kernel_ids,
                patch=patch,
                source=source,
                repo=target_repo,
                score=score,
                gap=gap,
                decision="RETRYABLE_FAULT",
                reason=f"retest_error:{str(exc)[-1000:]}",
                mode="retest",
                evidence={
                    "anchor_measurements": [],
                    "candidate_measurements": [candidate_first.to_dict()],
                },
            )
            return False, current_anchor, current_gain, gap

        candidate_measurements = list(retest.candidate_measurements)
        new_gain = current_gain * retest.multiplier
        evidence = {
            "incremental_multiplier": retest.multiplier,
            "effective_keep_floor": retest.effective_floor,
            "screen_gain": retest.screen_gain,
            "anchor_measurements": [item.to_dict() for item in retest.anchor_measurements],
            "candidate_measurements": [item.to_dict() for item in candidate_measurements],
        }
        if retest.disposition is RetestDisposition.KEEP:
            repo_before_sha = base_shas[target_repo]
            reset_hard_to(target_repo, repo_before_sha)
            applied, apply_message = apply_prepared_kernel_patch(
                patch,
                target_repo,
            )
            if not applied:
                record_candidate_retention_decision(
                    ckpt,
                    kernel_ids,
                    patch=patch,
                    source=source,
                    repo=target_repo,
                    score=score,
                    gap=gap,
                    decision="RETRYABLE_FAULT",
                    reason=f"reapply_failed:{apply_message}",
                    mode=retest.mode,
                    evidence=evidence,
                )
                return False, current_anchor, current_gain, gap
            if needs_rebuild:
                rebuilt, rebuild_message = rebuild_framework(target_repo)
                if not rebuilt:
                    revert(True)
                    record_candidate_retention_decision(
                        ckpt,
                        kernel_ids,
                        patch=patch,
                        source=source,
                        repo=target_repo,
                        score=score,
                        gap=gap,
                        decision="RETRYABLE_FAULT",
                        reason=f"rebuild_failed: {rebuild_message}",
                        mode=retest.mode,
                        evidence=evidence,
                    )
                    return False, current_anchor, current_gain, gap
            committed = commit_selected_changes(
                target_repo,
                changed_files,
                f"Quark Quant-Perf kernel patch (retained, gain {current_gain:.3f}->{new_gain:.3f}x)",
            )
            if not committed:
                revert(needs_rebuild)
                record_candidate_retention_decision(
                    ckpt,
                    kernel_ids,
                    patch=patch,
                    source=source,
                    repo=target_repo,
                    score=score,
                    gap=gap,
                    decision="RETRYABLE_FAULT",
                    reason="commit_failed",
                    mode=retest.mode,
                    evidence=evidence,
                )
                return False, current_anchor, current_gain, gap
            repo_after_sha = get_head_sha(target_repo)
            base_shas[target_repo] = repo_after_sha
            new_anchor = candidate_measurements[-1]
            trial = record_candidate_retention_decision(
                ckpt,
                kernel_ids,
                patch=patch,
                source=source,
                repo=target_repo,
                score=score,
                gap=gap,
                tps=new_anchor.median_tps,
                gain=new_gain,
                decision="KEEP",
                reason="improved_past_keep_floor",
                mode=retest.mode,
                evidence=evidence,
            )
            ckpt.state.setdefault("retention_stack", []).append(
                {
                    "kind": "patch",
                    "candidate": kernel_ids.get(patch, ""),
                    "patch": patch,
                    "source": source,
                    "repo": target_repo,
                    "repo_before_sha": repo_before_sha,
                    "repo_after_sha": repo_after_sha,
                    "changed_files": list(changed_files),
                    "requires_rebuild": needs_rebuild,
                    "gain_before": current_gain,
                    "gain_after": new_gain,
                    "accuracy_gap_before": current_gap,
                    "accuracy_gap_after": gap,
                    "retain_trial_attempt": trial["attempt"],
                    "active": True,
                }
            )
            ckpt.save()
            return True, new_anchor, new_gain, gap

        if retest.candidate_active:
            revert(needs_rebuild)
        decision = (
            "DROP_CONFIRMED"
            if retest.disposition is RetestDisposition.DROP_CONFIRMED
            else "RETRYABLE_FAULT"
            if retest.disposition is RetestDisposition.RETRYABLE_FAULT
            else "NEEDS_REVIEW"
        )
        record_candidate_retention_decision(
            ckpt,
            kernel_ids,
            patch=patch,
            source=source,
            repo=target_repo,
            score=score,
            gap=gap,
            tps=candidate_first.median_tps,
            gain=new_gain,
            decision=decision,
            reason=(
                "confirmed_no_gain"
                if decision == "DROP_CONFIRMED"
                else "within_measurement_noise"
                if decision == "NEEDS_REVIEW"
                else "unstable_measurement"
            ),
            mode=retest.mode,
            evidence=evidence,
        )
        return False, current_anchor, current_gain, gap

    def reconcile_final_stack(
        self,
        perf: PerfResult,
        ckpt: Checkpoint,
        *,
        quant_gain: float,
        quant_gap: float,
        measure_final_stack: Callable[[], tuple[float, float, float, float]],
    ) -> tuple[float, float]:
        """Make final ABBA authoritative by reverting the weakest suffix."""
        stack = ckpt.state.setdefault("retention_stack", [])
        while True:
            _, _, final_gain, effective_floor = measure_final_stack()
            perf.gain = final_gain
            active = [entry for entry in stack if entry.get("active")]
            final_gap = float(active[-1].get("accuracy_gap_after") or 0.0) if active else quant_gap
            incremental = final_gain / quant_gain - 1.0 if quant_gain > 0.0 else 0.0
            ckpt.state["final_retention"] = {
                "quant_only_gain": quant_gain,
                "final_gain": final_gain,
                "incremental_gain": incremental,
                "effective_keep_floor": effective_floor,
                "active_candidates": len(active),
                "status": (
                    "accepted"
                    if active and incremental > effective_floor
                    else "quant_only"
                    if not active
                    else "reconciling"
                ),
            }
            ckpt.save()
            if not active or incremental > effective_floor:
                return final_gain, final_gap

            reverted = active[-1]
            self._rollback_retention_entry(reverted, perf)
            self._mark_retention_entry_reverted(ckpt, reverted)
            ckpt.save()

    @staticmethod
    def _rollback_retention_entry(
        entry: dict[str, Any],
        perf: PerfResult,
    ) -> None:
        if entry.get("kind") == "runtime":
            for key, value in (entry.get("runtime_snapshot") or {}).items():
                if value is None:
                    os.environ.pop(str(key), None)
                else:
                    os.environ[str(key)] = str(value)
            perf.runtime_env = dict(entry.get("runtime_env_before") or {})
            perf.runtime_artifacts = dict(entry.get("runtime_artifacts_before") or {})
            return

        repo = str(entry.get("repo") or "")
        before_sha = str(entry.get("repo_before_sha") or "")
        if not repo or not before_sha:
            raise StageError(
                "perfopt",
                "retained patch is missing rollback repository metadata",
            )
        _restore_repo_revision(
            repo,
            before_sha,
            requires_rebuild=bool(entry.get("requires_rebuild")),
        )
        patch = str(entry.get("patch") or "")
        kept = [
            row
            for row in zip(
                perf.patches,
                perf.patch_srcs,
                perf.patch_repos,
                strict=False,
            )
            if row[0] != patch
        ]
        perf.patches = [row[0] for row in kept]
        perf.patch_srcs = [row[1] for row in kept]
        perf.patch_repos = [row[2] for row in kept]

    @staticmethod
    def _mark_retention_entry_reverted(
        ckpt: Checkpoint,
        entry: dict[str, Any],
    ) -> None:
        entry["active"] = False
        entry["final_decision"] = "REVERTED"
        entry["revert_reason"] = "final_abba_below_keep_floor"
        if entry.get("kind") == "runtime":
            ckpt.state["retained_runtime_env"] = dict(entry.get("runtime_env_before") or {})
            ckpt.record_phase_event(
                "vendor_gemm",
                "reverted",
                candidate=entry.get("candidate"),
                reason="final_abba_below_keep_floor",
            )
            return

        attempt = entry.get("retain_trial_attempt")
        for trial in ckpt.state.get("retain_trials") or []:
            if trial.get("attempt") != attempt:
                continue
            trial["active"] = False
            trial["final_decision"] = "REVERTED"
            trial["revert_reason"] = "final_abba_below_keep_floor"
            break
        patch = entry.get("patch")
        for candidate in ckpt.state.get("geak_patches") or []:
            if candidate.get("best_patch") == patch:
                candidate["retention_status"] = "reverted"
                candidate["retained"] = False
                break
        kernel_id = entry.get("candidate")
        for journey in ckpt.state.get("kernel_journey") or []:
            if journey.get("kernel_id") != kernel_id:
                continue
            e2e = journey.setdefault("e2e", {})
            e2e.update(
                {
                    "validated": False,
                    "decision": "REVERTED",
                    "reason": "final_abba_below_keep_floor",
                }
            )
            journey["outcome"] = "rejected"
            break
