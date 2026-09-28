#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""PerfOpt service for generating optimization candidates.

Locate bottleneck kernels in an already-landed, accuracy-passing
model, hand each to GEAK with quantization-aware context, and keep only the
patches that survive an independent re-measurement + accuracy re-check.

Design ref: IMPL_SPEC §4.2.
"""

from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quark.experimental.torch.quant_perf import landing
from quark.experimental.torch.quant_perf.perfopt.bottleneck_analysis import (
    BottleneckAnalysisResult,
    BottleneckMode,
    analyze_bottlenecks,
    find_best_existing_trace,
    find_comparable_traces,
)
from quark.experimental.torch.quant_perf.perfopt.collect import collect_trace
from quark.experimental.torch.quant_perf.perfopt.geak import build_geak_task, run_geak
from quark.experimental.torch.quant_perf.perfopt.guardrails import geak_execution_skip_reason, should_attempt_geak
from quark.experimental.torch.quant_perf.perfopt.journey import (
    load_geak_resume_state as _load_geak_resume_state,
)
from quark.experimental.torch.quant_perf.perfopt.journey import (
    record_bottleneck_analysis as _record_bottleneck_analysis,
)
from quark.experimental.torch.quant_perf.perfopt.journey import (
    record_kernel_attempt as _record_kernel_attempt,
)
from quark.experimental.torch.quant_perf.perfopt.journey import (
    record_kernel_skip as _record_kernel_skip,
)
from quark.experimental.torch.quant_perf.perfopt.keep import (
    aggregate_gain,
    is_kernel_candidate,
    kernel_candidate_status,
    make_kernel_id,
)
from quark.experimental.torch.quant_perf.perfopt.kernel_provenance import (
    collect_flydsl_cache_provenance,
    write_kernel_provenance,
)
from quark.experimental.torch.quant_perf.perfopt.kernel_source import (
    SOURCE_RESOLVER_VERSION,
    classify_kernel,
    generated_kernel_provenance,
    resolve_kernel_source_repo,
)
from quark.experimental.torch.quant_perf.perfopt.locate import locate_bottlenecks
from quark.experimental.torch.quant_perf.perfopt.trace_evidence import (
    enrich_bottlenecks_with_eager_launchers,
    enrich_bottlenecks_with_kernel_evidence,
    extract_kernel_evidence,
)
from quark.experimental.torch.quant_perf.session.progress import write_progress
from quark.experimental.torch.quant_perf.session.spec import Checkpoint, PerfResult, ServerHandle, Spec
from quark.experimental.torch.quant_perf.workspace.manager import RepoWorkspaceManager

if TYPE_CHECKING:
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore

logger = logging.getLogger(__name__)


def _resolve_quant_ckpt_dir(
    spec: Spec,
    ckpt: Checkpoint | None,
) -> str:
    if ckpt is not None:
        checkpoint = str(ckpt.state.get("quant_ckpt_dir") or "")
        if checkpoint:
            return checkpoint
    return spec.quant_ckpt_dir


def _source_context_tags(
    spec: Spec,
    ckpt: Checkpoint | None,
) -> tuple[str, ...]:
    candidate = ckpt.state.get("best_candidate") if ckpt is not None else None
    candidate = candidate or {}
    modes = {str(value).lower() for value in candidate.values() if value}
    strategy = str(spec.quant_strategy or "").lower()
    uses_a4w4 = "mxfp4" in modes or ("mxfp4" in strategy and "mxfp4_fp8" not in strategy and "w4a8" not in strategy)
    uses_w4a8 = "mxfp4_fp8" in modes or "mxfp4_fp8" in strategy or "w4a8" in strategy
    if (uses_a4w4 and spec.mxfp4_gemm_backend == "flydsl") or (uses_w4a8 and spec.w4a8_gemm_backend == "flydsl"):
        return ("flydsl_dense_mxfp4",)
    return ()


def _external_source_roots(spec: Spec) -> dict[str, str]:
    managed = {
        os.path.realpath(repo)
        for repo in (
            spec.active_framework_repo,
            spec.active_kernel_repo,
        )
        if repo
    }
    roots: dict[str, str] = {}
    for package, root in dict(spec.runtime_origins or {}).items():
        if not root or os.path.realpath(root) in managed:
            continue
        role = "quantizer" if package == "quark" else str(package)
        roots[role] = str(root)
    return roots


def _generated_cache_roots(spec: Spec) -> tuple[Path, ...]:
    runtime_env = dict(spec.runtime_env)
    roots = [
        runtime_env.get("TRITON_CACHE_DIR"),
        runtime_env.get("TORCHINDUCTOR_CACHE_DIR"),
        runtime_env.get("QUARK_QUANT_PERF_VLLM_CACHE_BASE"),
        str(Path(spec.session_dir) / "runtime" / "cache" / "triton"),
    ]
    return tuple(dict.fromkeys(Path(root).resolve() for root in roots if root and Path(root).is_dir()))


class OptimizationService:
    def __init__(
        self,
        experience_store: ExperienceStore | None,
    ) -> None:
        self.experience_store = experience_store

    @staticmethod
    def preflight() -> None:
        from TraceLens.Reporting.reporting_utils import resolve_gpu_arch  # type: ignore[import-untyped]

        if not callable(resolve_gpu_arch):
            raise RuntimeError("TraceLens reporting API is unavailable")

    def _enrich_selected_kernel_shape_evidence(
        self,
        bottlenecks: list[dict[str, Any]],
        spec: Spec,
        ckpt: Checkpoint,
    ) -> None:
        """Collect one eager trace for selected graph-replayed kernels."""
        if not bottlenecks or spec.framework != "vllm":
            return
        from quark.experimental.torch.quant_perf.perfopt.vendor_gemm import (
            extract_gemm_shapes,
            filter_tunableop_signatures,
            merge_tunableop_untuned_files,
        )

        vendor_bottlenecks = [
            bottleneck
            for bottleneck in bottlenecks
            if classify_kernel(bottleneck["op_name"])[1].startswith("vendor library kernel")
        ]
        vendor_evidence_complete = bool(vendor_bottlenecks) and all(
            extract_gemm_shapes([bottleneck]) and bottleneck.get("tunableop_input") for bottleneck in vendor_bottlenecks
        )
        missing_shape_cases = any(not bottleneck.get("shape_cases") for bottleneck in bottlenecks)
        if not missing_shape_cases and (not vendor_bottlenecks or vendor_evidence_complete):
            return

        run_dir = Path(spec.session_dir) / "trace_shape_evidence"
        run_dir.mkdir(parents=True, exist_ok=True)
        untuned = run_dir / "tunableop_untuned.csv"
        selected = run_dir / "trace_selected_tunableop.csv"
        if vendor_bottlenecks:
            for stale in run_dir.glob(f"{untuned.stem}*{untuned.suffix}"):
                stale.unlink(missing_ok=True)
            selected.unlink(missing_ok=True)

        sidecar_args = list(spec.vllm_extra_args)
        if "--enforce-eager" not in spec.expanded_vllm_args:
            sidecar_args.append("--enforce-eager")
        sidecar_env = dict(spec.runtime_env)
        if vendor_bottlenecks:
            sidecar_env.update(
                {
                    "PYTORCH_TUNABLEOP_ENABLED": "1",
                    "PYTORCH_TUNABLEOP_TUNING": "0",
                    "PYTORCH_TUNABLEOP_RECORD_UNTUNED": "1",
                    "PYTORCH_TUNABLEOP_UNTUNED_FILENAME": str(untuned),
                    "PYTORCH_TUNABLEOP_FILENAME": str(run_dir / "tunableop_results.csv"),
                }
            )
        sidecar_spec = replace(
            spec,
            vllm_extra_args=sidecar_args,
            runtime=replace(
                spec.runtime,
                runtime_env=sidecar_env,
            ),
        )
        server = None
        trace = None
        try:
            write_progress(
                spec.session_dir,
                stage_detail=("collecting eager shape evidence for graph-replayed kernels"),
            )
            server = landing.load(
                _resolve_quant_ckpt_dir(spec, ckpt),
                sidecar_spec,
                profiler_dir=str(run_dir),
            )
            trace = collect_trace(
                server.port,
                profiler_dir=str(run_dir),
                isl=spec.isl,
                osl=min(spec.osl, 16),
                warmup_steps=2,
                profile_steps=32,
                extract_steady_state=False,
            )
        except Exception as exc:
            logger.warning(
                "kernel shape sidecar failed: %s",
                exc,
            )
            for bottleneck in vendor_bottlenecks:
                bottleneck.setdefault(
                    "gemm_shape_evidence",
                    {
                        "status": "missing",
                        "source": "eager_sidecar",
                        "trace_path": "",
                        "alternatives": [],
                        "reason": str(exc),
                    },
                )
            return
        finally:
            if server is not None:
                server.stop()

        enrich_bottlenecks_with_kernel_evidence(
            bottlenecks,
            extract_kernel_evidence(trace),
            source="eager_sidecar",
            trace_path=trace,
        )
        if not vendor_bottlenecks:
            return

        merged_count = merge_tunableop_untuned_files(untuned)
        exact_shapes = extract_gemm_shapes(vendor_bottlenecks)
        scaled_only = all(
            str(bottleneck.get("parent_op_name") or "") == "aten::_scaled_mm"
            for bottleneck in vendor_bottlenecks
            if extract_gemm_shapes([bottleneck])
        )
        selected_count = (
            filter_tunableop_signatures(
                untuned,
                selected,
                shapes=exact_shapes,
                scaled_only=scaled_only,
            )
            if merged_count and exact_shapes
            else 0
        )
        if selected_count:
            for bottleneck in vendor_bottlenecks:
                if extract_gemm_shapes([bottleneck]):
                    bottleneck["tunableop_input"] = str(selected)
        ckpt.state["vendor_shape_evidence"] = {
            "trace": str(trace),
            "status": ("exact" if exact_shapes else "unresolved"),
            "shape_count": len(exact_shapes),
            "recorded_signature_count": merged_count,
            "selected_signature_count": selected_count,
            "tunableop_input": (str(selected) if selected_count else ""),
        }
        ckpt.save()

    def _collect_and_analyze_bottlenecks(
        self,
        model_server: ServerHandle,
        spec: Spec,
        ckpt: Checkpoint,
    ) -> tuple[BottleneckAnalysisResult, list[dict[str, Any]]]:
        profiler_dir = f"{spec.session_dir}/trace"
        existing_trace = find_best_existing_trace(Path(profiler_dir))
        if existing_trace:
            trace = existing_trace
            logger.info("reusing existing trace: %s", trace)
        else:
            trace = collect_trace(
                model_server.port,
                profiler_dir=profiler_dir,
                isl=spec.isl,
                osl=spec.osl,
            )
        absolute_candidates = locate_bottlenecks(
            trace,
            gpu=spec.gpu_arch,
            top_n=spec.top_kernels,
            gpu_arch_json_path=spec.tracelens_gpu_arch_json,
        )
        logger.info(
            "located %d bottleneck ops from trace",
            len(absolute_candidates),
        )

        logger.info("stopping quantized server before bottleneck analysis/GEAK (serialize GPU use)")
        model_server.stop()

        requested_mode = BottleneckMode(spec.bottleneck_mode)
        baseline_trace = ""
        trace_kind = ""
        fallback_reason = ""
        quantized_analysis_trace = str(trace)
        if requested_mode is BottleneckMode.DIFFERENTIAL:
            baseline_dir = f"{profiler_dir}_baseline"
            quantized_trace, baseline_trace_path, trace_kind = find_comparable_traces(
                profiler_dir,
                baseline_dir,
            )
            if quantized_trace is None or baseline_trace_path is None:
                logger.info("collecting baseline trace for differential analysis")
                write_progress(
                    spec.session_dir,
                    stage_detail=("collecting baseline trace for differential kernel analysis"),
                )
                try:
                    baseline_server = landing.load(
                        spec.base_model,
                        spec,
                        profiler_dir=baseline_dir,
                    )
                    try:
                        collect_trace(
                            baseline_server.port,
                            profiler_dir=baseline_dir,
                            isl=spec.isl,
                            osl=spec.osl,
                        )
                    finally:
                        baseline_server.stop()
                except Exception as exc:
                    logger.warning(
                        "baseline trace collection failed (%s); using absolute fallback",
                        exc,
                    )
                    fallback_reason = "baseline_trace_collection_failed"
                quantized_trace, baseline_trace_path, trace_kind = find_comparable_traces(
                    profiler_dir,
                    baseline_dir,
                )
            if quantized_trace is not None and baseline_trace_path is not None:
                quantized_analysis_trace = str(quantized_trace)
                baseline_trace = str(baseline_trace_path)
            elif not fallback_reason:
                fallback_reason = "no_comparable_baseline_trace"

        analysis = analyze_bottlenecks(
            mode=requested_mode,
            quantized_trace=quantized_analysis_trace,
            baseline_trace=baseline_trace,
            trace_kind=trace_kind,
            absolute_candidates=absolute_candidates,
            top_n=spec.top_kernels,
            fallback_reason=fallback_reason,
        )
        bottlenecks = list(analysis.candidates)
        enrich_bottlenecks_with_eager_launchers(
            bottlenecks,
            analysis.quantized_trace,
        )
        provenance_records = collect_flydsl_cache_provenance(
            spec.runtime_env.get("FLYDSL_RUNTIME_CACHE_DIR", ""),
            source_roots={
                role: repo
                for role, repo in (
                    ("framework", spec.active_framework_repo),
                    ("kernel", spec.active_kernel_repo),
                )
                if repo
            },
            gpu_arch=spec.gpu_arch,
        )
        provenance_path = write_kernel_provenance(
            spec.session_dir,
            provenance_records,
        )
        ckpt.state["kernel_provenance_artifact"] = str(provenance_path)
        ckpt.save()
        logger.info(
            "bottleneck analysis mode=%s effective=%s selected=%d",
            analysis.requested_mode.value,
            analysis.effective_mode,
            len(bottlenecks),
        )
        if bottlenecks != list(analysis.candidates):
            analysis = replace(
                analysis,
                candidates=tuple(bottlenecks),
            )
        return analysis, provenance_records

    @staticmethod
    def _classify_kernel_candidates(
        bottlenecks: list[dict[str, Any]],
        spec: Spec,
        ckpt: Checkpoint,
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        rewritable: list[dict[str, Any]] = []
        vendor: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for bottleneck in bottlenecks:
            can_rewrite, reason = classify_kernel(bottleneck["op_name"])
            if can_rewrite:
                rewritable.append(bottleneck)
                continue
            if reason.startswith("vendor library kernel"):
                vendor.append(bottleneck)
            skipped.append(
                {
                    "kernel": bottleneck["op_name"],
                    "reason": reason,
                }
            )
            logger.info(
                "SKIP (not rewritable): %s — %s",
                bottleneck["op_name"][:60],
                reason,
            )
            write_progress(
                spec.session_dir,
                warning=(f"SKIP {bottleneck['op_name'][:60]}: {reason}"),
            )
            if reason.startswith("torch.compile generated kernel"):
                source_mapping = generated_kernel_provenance(
                    bottleneck["op_name"],
                    cache_roots=_generated_cache_roots(spec),
                )
            else:
                source_mapping = {
                    "mapping_kind": "precompiled_binary",
                    "patchable": False,
                    "method": "runtime_classification",
                    "confidence": "non_rewritable",
                    "reason": reason,
                    "resolver_version": SOURCE_RESOLVER_VERSION,
                    "retryable": False,
                }
            _record_kernel_skip(
                ckpt,
                make_kernel_id(bottleneck["op_name"]),
                reason,
                source_mapping,
            )
        return rewritable, vendor, skipped

    @staticmethod
    def _tune_vendor_gemm_candidates(
        vendor_bottlenecks: list[dict[str, Any]],
        spec: Spec,
        ckpt: Checkpoint,
    ) -> list[dict[str, Any]]:
        vendor_result = dict(ckpt.state.get("vendor_gemm_tuning") or {})
        if not vendor_bottlenecks:
            return list(vendor_result.get("candidates") or [])

        from quark.experimental.torch.quant_perf.perfopt.vendor_gemm import (
            run_vendor_gemm_tuning,
            vendor_result_needs_run,
        )

        if vendor_result_needs_run(vendor_result):
            write_progress(
                spec.session_dir,
                stage_detail=("tuning trace-selected vendor GEMM workloads"),
            )
            prior_attempts = list(vendor_result.get("attempts") or [])
            attempt_count = int(vendor_result.get("attempt_count") or 0) + 1
            vendor_result = run_vendor_gemm_tuning(
                vendor_bottlenecks,
                spec,
                Path(spec.session_dir) / "vendor_gemm",
                attempt_count=attempt_count,
            )
            vendor_result["attempts"] = [
                *prior_attempts,
                {
                    "attempt": attempt_count,
                    "status": vendor_result.get("status"),
                    "reason": vendor_result.get("reason", ""),
                    "capability_version": vendor_result.get("capability_version"),
                },
            ]
            ckpt.state["vendor_gemm_tuning"] = dict(vendor_result)
            ckpt.save()
        return list(vendor_result.get("candidates") or [])

    def _optimize_rewritable_kernels_with_geak(
        self,
        bottlenecks: list[dict[str, Any]],
        rewritable_bottlenecks: list[dict[str, Any]],
        provenance_records: list[dict[str, Any]],
        spec: Spec,
        ckpt: Checkpoint,
    ) -> list[dict[str, Any]]:
        attempted_signatures, results = _load_geak_resume_state(
            ckpt,
            active_repos=[
                repo
                for repo in (
                    spec.active_kernel_repo,
                    spec.active_framework_repo,
                )
                if repo
            ],
        )
        if attempted_signatures:
            logger.info(
                "resume: %d GEAK kernel(s) already attempted (%d kept), skipping them",
                len(attempted_signatures),
                len(results),
            )
        quant_ckpt_dir = _resolve_quant_ckpt_dir(spec, ckpt)
        total_time_us = sum(bottleneck["kernel_time_us"] for bottleneck in bottlenecks)
        workspace_manager = RepoWorkspaceManager(
            session_dir=spec.session_dir,
            session_id=ckpt.state.get("session_id", "unknown"),
            ckpt=ckpt,
            root_dir=(ckpt.state.get("workspace_root") or Path(spec.session_dir) / "workspaces" / "managed"),
            init_submodules=spec.workspace_source != "readonly",
        )
        source_context_tags = _source_context_tags(spec, ckpt)
        total_candidates = len(rewritable_bottlenecks)

        for index, bottleneck in enumerate(
            rewritable_bottlenecks,
            start=1,
        ):
            signature = make_kernel_id(bottleneck["op_name"])
            if signature in attempted_signatures:
                logger.info(
                    "resume: kernel %s already attempted, skipping",
                    signature,
                )
                continue
            resolution = resolve_kernel_source_repo(
                bottleneck,
                spec.active_framework_repo,
                spec.active_kernel_repo,
                context_tags=source_context_tags,
                provenance_records=provenance_records,
                gpu_arch=spec.gpu_arch,
                external_source_roots=_external_source_roots(spec),
            )
            if bottleneck.get("launcher_source_file") and not resolution.launcher_source_file:
                resolution = replace(
                    resolution,
                    launcher_source_file=str(bottleneck["launcher_source_file"]),
                    launcher_symbol=str(bottleneck.get("launcher_symbol") or ""),
                )
            if not resolution.runtime_kernel_name:
                resolution = replace(
                    resolution,
                    runtime_kernel_name=str(bottleneck.get("device_kernel_name") or bottleneck.get("op_name") or ""),
                )
            if not resolution.gpu_arch:
                resolution = replace(
                    resolution,
                    gpu_arch=spec.gpu_arch,
                )

            source_file = resolution.source_file
            source_repo = resolution.source_repo
            source_reason = resolution.reason
            if source_file is None or not resolution.patchable:
                mapping = resolution.to_dict()
                skip_reason = f"{mapping['mapping_kind']}: {source_reason}"
                write_progress(
                    spec.session_dir,
                    warning=(f"SKIP {bottleneck['op_name'][:60]}: {skip_reason}"),
                )
                _record_kernel_skip(
                    ckpt,
                    signature,
                    skip_reason,
                    mapping,
                )
                continue

            skip_reason = geak_execution_skip_reason(bottleneck["op_name"], resolution.source_symbol)
            if skip_reason:
                _record_kernel_skip(ckpt, signature, skip_reason, resolution.to_dict())
                write_progress(spec.session_dir, warning=skip_reason)
                continue

            if not should_attempt_geak(
                bottleneck["kernel_time_us"],
                total_time_us,
            ):
                write_progress(
                    spec.session_dir,
                    warning=(f"kernel {bottleneck['op_name']} vetoed by Amdahl preflight"),
                )
                _record_kernel_skip(
                    ckpt,
                    signature,
                    "Amdahl preflight veto",
                    resolution.to_dict(),
                )
                continue

            role = (
                "kernel"
                if (
                    spec.active_kernel_repo
                    and os.path.realpath(source_repo) == os.path.realpath(spec.active_kernel_repo)
                )
                else "framework"
            )
            can_modify = spec.can_modify_kernel if role == "kernel" else spec.can_modify_framework
            if not can_modify:
                reason = f"{role} source is readonly"
                _record_kernel_skip(
                    ckpt,
                    signature,
                    reason,
                    resolution.to_dict(),
                )
                write_progress(
                    spec.session_dir,
                    warning=(f"SKIP {bottleneck['op_name'][:60]}: {reason}"),
                )
                continue

            run_dir = f"{spec.session_dir}/geak/{signature}"
            with workspace_manager.candidate(
                role,
                signature,
            ) as candidate:
                relative_source = Path(source_file).resolve().relative_to(Path(source_repo).resolve())
                candidate_source = str(candidate.path / relative_source)
                resolution_dict = resolution.to_dict()
                explicit_patch_files = []
                for field in (
                    "launcher_source_file",
                    "config_file",
                ):
                    value = str(resolution_dict.get(field) or "")
                    if not value:
                        continue
                    try:
                        relative = Path(value).resolve().relative_to(Path(source_repo).resolve())
                    except ValueError:
                        continue
                    explicit_patch_files.append(str(relative).replace(os.sep, "/"))
                source_binding = dict(resolution_dict)
                source_binding.update(
                    {
                        "source_file": candidate_source,
                        "source_repo": str(candidate.path),
                        "source_relpath": str(relative_source).replace(
                            os.sep,
                            "/",
                        ),
                    }
                )
                write_progress(
                    spec.session_dir,
                    stage_detail=(f"optimizing kernel {index}/{total_candidates} ({bottleneck['op_name']}) via GEAK"),
                )
                from quark.experimental.torch.quant_perf.perfopt.knowledge_context import (
                    build_kernel_knowledge,
                )

                knowledge_bundle, knowledge_text = build_kernel_knowledge(
                    experience_store=self.experience_store,
                    state=ckpt.state,
                    spec=spec,
                    bottleneck=bottleneck,
                )
                task = build_geak_task(
                    {
                        **bottleneck,
                        "source_resolution": source_binding,
                    },
                    spec.quant_strategy,
                    quant_ckpt_dir,
                    source_file=candidate_source,
                    gpu_arch=spec.gpu_arch,
                    knowledge_text=knowledge_text,
                )
                report = run_geak(
                    candidate_source,
                    task,
                    spec.geak_model,
                    spec.gpu_id,
                    run_dir,
                    session_dir=spec.session_dir,
                    kernel_name=bottleneck["op_name"],
                    source_binding=source_binding,
                    knowledge_ids=list(knowledge_bundle.source_ids),
                    budget=spec.geak_direction_budget,
                    timeout_s=spec.geak_timeout_s,
                )
                if report.get("execution_status") == "skipped":
                    _record_kernel_skip(ckpt, signature, report["error"], resolution_dict)
                    write_progress(spec.session_dir, warning=report["error"])
                    continue
                from quark.experimental.torch.quant_perf.perfopt.kernel_workflow import (
                    cleanup_kernel_workflow_eval_dir,
                    persist_kernel_workflow_artifacts,
                )

                report = persist_kernel_workflow_artifacts(
                    report,
                    run_dir,
                    kernel_src=candidate_source,
                    source_repo=str(candidate.path),
                    explicitly_allowed_files=tuple(explicit_patch_files),
                )
                report["candidate_status"] = self._candidate_status(report)
                # Preserve evidence from interrupted runs and unfinished validation,
                # including an early workload rejection before a final result exists.
                if (
                    report["candidate_status"] != "incomplete"
                    and report.get("execution_status", "completed") == "completed"
                ):
                    cleanup_kernel_workflow_eval_dir(report.get("eval_dir", ""))

            report.setdefault(
                "kernel_name",
                bottleneck["op_name"],
            )
            report["kernel_src"] = source_file
            report["kernel_repo"] = source_repo
            report["knowledge_ids"] = list(knowledge_bundle.source_ids)
            kept = report["candidate_status"] == "candidate"
            _record_kernel_attempt(
                ckpt,
                kernel_id=signature,
                source_file=source_file,
                source_repo=source_repo,
                source_reason=source_reason,
                report=report,
                kept=kept,
                source_mapping=resolution_dict,
            )
            ckpt.state.setdefault("geak_patches", []).append(
                {
                    "kernel_sig": signature,
                    "status": report["candidate_status"],
                    "execution_status": report.get("execution_status"),
                    "timed_out": report.get("timed_out", False),
                    "error": report.get("error"),
                    "artifacts_dir": report.get("artifacts_dir", ""),
                    "best_patch": report.get("best_patch", ""),
                    "verified_speedup": report.get("verified_speedup"),
                    "micro_speedup_source": report.get(
                        "micro_speedup_source",
                        "",
                    ),
                    "kernel_name": bottleneck["op_name"],
                    "kernel_src": source_file,
                    "kernel_repo": source_repo,
                    "kernel_relpath": resolution_dict.get(
                        "source_relpath",
                        "",
                    ),
                    "source_mapping": resolution_dict,
                    "patch_changed_files": list(report.get("patch_changed_files") or []),
                    "patch_requires_rebuild": bool(report.get("patch_requires_rebuild")),
                    "patch_validation": dict(report.get("patch_validation") or {}),
                }
            )
            ckpt.save()
            if not kept:
                continue
            results.append(report)
        return results

    def generate_optimization_candidates(
        self,
        model_server: ServerHandle,
        spec: Spec,
        ckpt: Checkpoint,
        quant_gain: float = 1.0,
    ) -> PerfResult:
        """Run resumable kernel optimization under checkpoint ownership."""
        analysis, provenance_records = self._collect_and_analyze_bottlenecks(
            model_server,
            spec,
            ckpt,
        )
        bottlenecks = list(analysis.candidates)
        self._enrich_selected_kernel_shape_evidence(
            bottlenecks,
            spec,
            ckpt,
        )
        _record_bottleneck_analysis(ckpt, analysis)

        (
            rewritable_bottlenecks,
            vendor_bottlenecks,
            skipped_kernels,
        ) = self._classify_kernel_candidates(
            bottlenecks,
            spec,
            ckpt,
        )
        runtime_candidates = self._tune_vendor_gemm_candidates(
            vendor_bottlenecks,
            spec,
            ckpt,
        )

        if skipped_kernels:
            logger.info(
                "%d/%d bottleneck kernels skipped (vendor/non-rewritable): %s",
                len(skipped_kernels),
                len(bottlenecks),
                [s["kernel"][:40] for s in skipped_kernels],
            )

        results = self._optimize_rewritable_kernels_with_geak(
            bottlenecks,
            rewritable_bottlenecks,
            provenance_records,
            spec,
            ckpt,
        )

        dominant = _dominant_bound(rewritable_bottlenecks or bottlenecks)
        # model_server was already stopped before the GEAK loop; nothing to
        # tear down here. The orchestrator's finally: server.stop() is a
        # harmless no-op on the already-stopped handle.
        return PerfResult(
            patches=[r["best_patch"] for r in results],
            patch_srcs=[r.get("kernel_src", "") for r in results],
            patch_repos=[r.get("kernel_repo", "") for r in results],
            gain=aggregate_gain(results) if results else 1.0,
            dominant_bound=dominant,
            runtime_candidates=runtime_candidates,
        )

    @staticmethod
    def _candidate_status(report: dict[str, Any]) -> str:
        """Separate an explicit rejection from missing validation evidence."""
        return kernel_candidate_status(report)

    @staticmethod
    def _keep(report: dict[str, Any]) -> bool:
        """Return whether a GEAK result is eligible for full E2E retention."""
        return is_kernel_candidate(report)


def _dominant_bound(
    bottlenecks: list[dict[str, Any]],
) -> str:
    if not bottlenecks:
        return ""
    counts: dict[str, int] = {}
    for bn in bottlenecks:
        bound = bn.get("roofline_bound", "")
        counts[bound] = counts.get(bound, 0) + 1
    return max(counts, key=counts.get)
