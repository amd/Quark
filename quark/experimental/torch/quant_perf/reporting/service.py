#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Deterministic FINAL-stage report generation.

``session_breakdown.json`` is the complete fact source. ``reports/final.json``
is its compact projection; both Markdown files are rendered from those JSON
objects without re-parsing console logs.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import os
import platform
import shlex
import subprocess
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf import __version__ as quant_perf_version
from quark.experimental.torch.quant_perf.perfopt.bottleneck_analysis import (
    bottleneck_analysis_from_state,
)
from quark.experimental.torch.quant_perf.session.progress import read_progress
from quark.experimental.torch.quant_perf.session.spec import DeployPackage, Spec

BREAKDOWN_SCHEMA_VERSION = "quark.quant_perf.session_breakdown.v5"
FINAL_SCHEMA_VERSION = "quark.quant_perf.final.v5"
EXPORTER_VERSION = "quark-quant-perf-reporting-2.1.0"

_STOP_REASON_EXPLANATIONS = {
    "success": "The run completed successfully and satisfied its terminal criteria.",
    "performance_failed": "The run stopped because throughput measurement could not be completed.",
    "perf_below_target": "The run completed, but the validated final gain remained below the target.",
    "accuracy_failed": "The run stopped because the quantized model did not pass the accuracy gate.",
    "base_unhealthy": "The run stopped because the unquantized baseline was not healthy in the framework.",
}


def _utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    tmp_path = Path(tmp)
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception:
        with contextlib.suppress(OSError):
            tmp_path.unlink()
        raise


def _repo_snapshot(path: str, state: dict[str, Any], prefix: str) -> dict[str, Any]:
    if not path:
        return {}
    role = "framework" if prefix == "fw" else "kernel"
    workspace = (state.get("repo_workspaces") or {}).get(role) or {}
    repo = Path(workspace.get("source_repo") or path)

    def _git(*args: str) -> str:
        if not repo.is_dir():
            return ""
        try:
            result = subprocess.run(
                ["git", *args],
                cwd=repo,
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception:
            return ""
        return result.stdout.strip() if result.returncode == 0 else ""

    status = _git("status", "--porcelain")
    return {
        "path": str(repo),
        "source_kind": ((state.get("resolved_sources") or {}).get(role) or {}).get("kind", ""),
        "source_origin": ((state.get("resolved_sources") or {}).get(role) or {}).get("origin_path", ""),
        "original_branch": (workspace.get("original_branch") or state.get(f"{prefix}_original_branch") or ""),
        "work_branch": (
            workspace.get("work_branch") or state.get(f"{prefix}_branch") or _git("rev-parse", "--abbrev-ref", "HEAD")
        ),
        "head": workspace.get("final_sha") or _git("rev-parse", "HEAD"),
        "source_head": _git("rev-parse", "HEAD"),
        "branch_retained": workspace.get("branch_retained"),
        "integration_path": workspace.get("integration_path", ""),
        "workspace_status": workspace.get("status", ""),
        "worktree_dirty": bool(status),
        "worktree_change_count": len(status.splitlines()) if status else 0,
    }


def _model_info(model_dir: str) -> dict[str, Any]:
    path = Path(model_dir) / "config.json"
    if not path.is_file():
        return {}
    try:
        cfg = json.loads(path.read_text())
    except Exception:
        return {}
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    return {
        "model_type": text.get("model_type") or cfg.get("model_type"),
        "architectures": cfg.get("architectures") or [],
        "hidden_size": text.get("hidden_size"),
        "num_hidden_layers": text.get("num_hidden_layers"),
        "num_attention_heads": text.get("num_attention_heads"),
        "num_key_value_heads": text.get("num_key_value_heads"),
        "num_experts": text.get("num_experts") or text.get("num_local_experts"),
        "num_experts_per_tok": text.get("num_experts_per_tok"),
        "torch_dtype": text.get("torch_dtype") or cfg.get("torch_dtype"),
    }


def _accepted_accuracy(attempts: list[dict[str, Any]]) -> dict[str, Any]:
    passed = [a for a in attempts if a.get("passed")]
    return dict((passed or attempts)[-1]) if attempts else {}


def _candidate_configs(search_state: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(entry["config"])
        for entry in (search_state.get("candidate_queue") or [])
        if isinstance(entry, dict) and isinstance(entry.get("config"), dict)
    ]


def _reproduction(spec: Spec, state: dict[str, Any]) -> dict[str, Any]:
    """Build copy-ready commands from the persisted run specification."""
    session_dir = str(Path(spec.session_dir).resolve())
    quant_ckpt = str(state.get("quant_ckpt_dir") or spec.quant_ckpt_dir)
    runtime_env = {
        **dict((state.get("invocation") or {}).get("env") or {}),
        **dict((state.get("performance_runtime") or {}).get("runtime_env") or {}),
        **dict(spec.runtime_env),
        "VLLM_PLUGINS": "",
        "ROCR_VISIBLE_DEVICES": ",".join(str(spec.gpu_id + index) for index in range(spec.tp)),
    }
    runtime_python = spec.runtime_python or "python3"
    effective_utilization = float(
        (state.get("performance_runtime") or {}).get(
            "effective_gpu_memory_utilization",
            spec.vllm_gpu_memory_utilization,
        )
    )
    serve_script = (
        "from quark.experimental.torch.quant_perf.landing.vllm_adapter "
        "import start_vllm_server; "
        "start_vllm_server("
        f"model_dir={quant_ckpt!r}, "
        f"tp={spec.tp!r}, "
        f"port={spec.server_port!r}, "
        f"kv_cache_scheme={spec.vllm_kv_cache_dtype!r}, "
        f"extra_args={spec.vllm_passthrough_args!r}, "
        f"gpu_id={spec.gpu_id!r}, "
        f"python_exe={runtime_python!r}, "
        f"runtime_env={runtime_env!r}"
        ").wait()"
    )
    benchmark_script = (
        "import json; "
        "from quark.experimental.torch.quant_perf.evaluation.throughput "
        "import measure_throughput; "
        "result = measure_throughput("
        f"{quant_ckpt!r}, "
        f"gpu_id={spec.gpu_id!r}, "
        f"isl={spec.isl!r}, "
        f"osl={spec.osl!r}, "
        f"tp={spec.tp!r}, "
        f"concurrency={spec.bench_concurrency!r}, "
        f"gpu_memory_utilization={effective_utilization!r}, "
        f"moe_backend={spec.vllm_moe_backend!r}, "
        f"trust_remote_code={spec.vllm_trust_remote_code!r}, "
        f"max_num_seqs={spec.vllm_max_num_seqs!r}, "
        f"kv_cache_dtype={spec.vllm_kv_cache_dtype!r}, "
        f"runtime_python={runtime_python!r}, "
        f"runtime_env={runtime_env!r}"
        "); "
        "print(json.dumps(result.to_dict(), indent=2))"
    )
    return {
        "environment": runtime_env,
        "commands": {
            "original": list((state.get("invocation") or {}).get("argv") or spec.invocation_argv),
            "serve": [runtime_python, "-c", serve_script],
            "evaluate": [
                "quark-quant-perf",
                "eval",
                "--from-session",
                session_dir,
                "--session-dir",
                "<new-eval-session>",
            ],
            "benchmark": [runtime_python, "-c", benchmark_script],
        },
        "replace_before_running": [
            "`<new-eval-session>` with a writable output directory.",
            "Recorded model/repository paths if the session is copied to another host.",
            f"GPU IDs `{runtime_env['ROCR_VISIBLE_DEVICES']}` and port `{spec.server_port}` if unavailable.",
        ],
    }


def _final_performance(state: dict[str, Any]) -> dict[str, Any]:
    measurements = list(state.get("performance_measurements") or [])
    quant_only_rows = [row for row in measurements if row.get("role") == "quant_only"]
    quant_only = dict(quant_only_rows[-1]) if quant_only_rows else dict(measurements[-1]) if measurements else {}
    final_rows = [row for row in measurements if row.get("role") == "final_abba"]
    authoritative = dict(final_rows[-1]) if final_rows else quant_only
    baseline = authoritative.get(
        "baseline_tps",
        quant_only.get("baseline_tps", state.get("baseline_tps")),
    )
    quantized = quant_only.get("quantized_tps", state.get("quant_tps"))
    final_tps = authoritative.get("final_tps", authoritative.get("quantized_tps", quantized))
    gain = authoritative.get("gain")
    if gain is None and baseline and final_tps:
        gain = final_tps / baseline
    return {
        "baseline_tps": baseline,
        "quantized_tps": quantized,
        "final_tps": final_tps,
        "quant_only_gain": quant_only.get("quant_only_gain", quant_only.get("gain", gain)),
        "final_gain": gain,
        "measurement_role": authoritative.get("role"),
        "measurement_attempt": authoritative.get("attempt"),
    }


def _capability_summary(state: dict[str, Any]) -> dict[str, Any]:
    accuracy = list(state.get("accuracy_attempts") or [])
    perf = list(state.get("performance_measurements") or [])
    kernels = list(state.get("kernel_journey") or [])
    retained = [t for t in (state.get("retain_trials") or []) if t.get("decision") == "KEEP" and t.get("active", True)]
    repairs = list(state.get("repair_journey") or [])
    performance_status = str(state.get("performance_status") or "")
    search_state = state.get("mix_precision_search") or {}
    search_result = search_state.get("result") or {}
    return {
        "quantization_search": {
            "status": "completed" if state.get("quant_ckpt_dir") else "not_completed",
            "attempts": int(
                search_result.get("total_configs_evaluated") or len(search_state.get("candidate_queue") or [])
            ),
        },
        "accuracy_gate": {
            "status": "passed" if any(a.get("passed") for a in accuracy) else "not_passed",
            "attempts": len(accuracy),
        },
        "performance_benchmark": {
            "status": (
                performance_status
                if performance_status in {"not_requested", "measured", "measurement_failed"}
                else "completed"
                if perf or state.get("quant_tps")
                else "not_completed"
            ),
            "attempts": len(perf),
        },
        "kernel_optimization": {
            "status": (
                "not_requested"
                if performance_status in {"not_requested", "measured"}
                else "incomplete"
                if any(row.get("status") == "incomplete" for row in state.get("geak_patches") or [])
                else "completed"
                if kernels or state.get("geak_patches")
                else "not_attempted"
            ),
            "attempts": len(state.get("geak_patches") or []),
            "retained": len(retained),
        },
        "runtime_repair": {
            "status": (
                "completed"
                if any(row.get("status") == "fixed" for row in repairs)
                else "not_fixed"
                if repairs
                else "not_attempted"
            ),
            "attempts": len(repairs),
        },
    }


def _active_retained_patches(
    state: dict[str, Any],
    package: DeployPackage,
) -> list[str]:
    stack = state.get("retention_stack")
    if isinstance(stack, list) and stack:
        paths = [
            str(entry.get("patch"))
            for entry in stack
            if (entry.get("kind") == "patch" and entry.get("active") and entry.get("patch"))
        ]
    else:
        paths = list(package.applied_patches or [])
    return list(dict.fromkeys(path for path in paths if path))


def _source_files(session_dir: Path) -> dict[str, str]:
    candidates = {
        "state_json": session_dir / "state.json",
        "progress_json": session_dir / "progress.json",
        "quant_checkpoint": session_dir / "quant_ckpt",
        "quant_trace": session_dir / "trace",
        "baseline_trace": session_dir / "trace_baseline",
        "geak_runs": session_dir / "geak",
        "kernel_provenance": session_dir / "kernel_provenance.json",
        "kernel_source_resolution": (session_dir / "kernel_source_resolution.json"),
        "knowledge_audit": session_dir / "knowledge" / "query_audit.jsonl",
        "llm_calls": session_dir / "llm_calls.jsonl",
    }
    return {key: str(path) for key, path in candidates.items()}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            rows.append(value)
    return rows


def _data_provenance(source_files: dict[str, str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for name, value in source_files.items():
        path = Path(value)
        size = path.stat().st_size if path.is_file() else None
        digest = ""
        if size is not None and size <= 64 * 1024 * 1024:
            hasher = hashlib.sha256()
            with path.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                    hasher.update(chunk)
            digest = hasher.hexdigest()
        out.append(
            {
                "source": name,
                "path": value,
                "found": path.exists(),
                "kind": "directory" if path.is_dir() else "file",
                "size_bytes": size,
                "snapshot_sha256": digest or None,
            }
        )
    return out


def build_session_breakdown(
    spec: Spec,
    state: dict[str, Any],
    package: DeployPackage,
) -> dict[str, Any]:
    """Build the complete, versioned machine-readable session record."""
    session_dir = Path(spec.session_dir).resolve()
    progress = read_progress(session_dir) or {}
    attempts = list(state.get("accuracy_attempts") or [])
    accepted_accuracy = _accepted_accuracy(attempts)
    final_perf = _final_performance(state)
    final_gain = final_perf.get("final_gain")
    quant_only_gain = final_perf.get("quant_only_gain")
    retained = _active_retained_patches(state, package)
    source_files = _source_files(session_dir)
    provenance = _data_provenance(source_files)
    warnings = list(progress.get("warnings") or [])
    warnings.extend(state.get("report_warnings") or [])
    terminal_stage = state.get("terminal_stage") or state.get("stage") or ""
    package_dict = asdict(package)
    knowledge_queries = list(state.get("knowledge_audit") or [])
    if not knowledge_queries:
        knowledge_queries = _load_jsonl(session_dir / "knowledge" / "query_audit.jsonl")
    llm_calls = list(state.get("llm_calls") or [])
    if not llm_calls:
        llm_calls = _load_jsonl(session_dir / "llm_calls.jsonl")

    search_state = state.get("mix_precision_search") or {}
    search_result = search_state.get("result") or {}
    performance_mode = spec.effective_performance_mode
    performance_target_met = (
        bool(final_gain is not None and final_gain >= spec.target_gain)
        if performance_mode == "optimize" and spec.target_gain is not None
        else None
    )
    return {
        "schema_version": BREAKDOWN_SCHEMA_VERSION,
        "exporter_version": EXPORTER_VERSION,
        "exported_at_utc": _utc_now(),
        "session": {
            "session_id": state.get("session_id"),
            "session_dir": str(session_dir),
            "created_at_utc": progress.get("started_at"),
            "ended_at_utc": progress.get("updated_at"),
            "elapsed_seconds": progress.get("elapsed_seconds"),
            "terminal_stage": terminal_stage,
            "stop_reason": package.status,
            "stop_reason_explanation": _STOP_REASON_EXPLANATIONS.get(package.status, ""),
            "message": package.message,
        },
        "workload": {
            "model": spec.model_dir,
            "base_model": spec.base_model,
            "framework": spec.framework,
            "gpu_type": spec.gpu_type,
            "gpu_arch": spec.gpu_arch,
            "gpu_id": spec.gpu_id,
            "tp": spec.tp,
            "isl": spec.isl,
            "osl": spec.osl,
            "benchmark_concurrency": spec.bench_concurrency,
            "precision": state.get("best_candidate") or spec.quant_strategy,
            "objective": {
                "performance_mode": performance_mode,
                "target_gain": spec.target_gain,
                "accuracy_gap_threshold": spec.accuracy_gap,
            },
            "runtime": {
                "search_moe_backend": spec.effective_search_moe_backend,
                "inference_moe_backend": spec.effective_inference_moe_backend,
                "mxfp4_moe_backend": spec.mxfp4_moe_backend,
                "mxfp4_gemm_backend": spec.mxfp4_gemm_backend,
                "w4a8_gemm_backend": spec.w4a8_gemm_backend,
                "aiter_config_fmoe": spec.aiter_config_fmoe,
                "kv_cache_precision_candidates": spec.kv_cache_precision_candidates,
                "layer_precision_candidates": spec.layer_precision_candidates,
                "gsm8k_samples": spec.gsm8k_num_samples,
                "workspace_source": spec.workspace_source,
                "session_runtime": dict(state.get("session_runtime") or {}),
                "runtime_origin_evidence": dict(state.get("runtime_origin_evidence") or {}),
            },
        },
        "model_info": _model_info(spec.base_model),
        "invocation": dict(state.get("invocation") or {"spec": spec.to_dict()}),
        "reproduction": _reproduction(spec, state),
        "repositories": {
            "framework": _repo_snapshot(spec.framework_source_repo, state, "fw"),
            "kernel": _repo_snapshot(spec.kernel_source_repo, state, "kernel"),
        },
        "cleanup": dict(state.get("cleanup") or {}),
        "recovery_details": {
            "attempt_count": len(state.get("recovery_attempts") or []),
            "attempts": list(state.get("recovery_attempts") or []),
            "baseline_reference": dict(state.get("baseline_reference") or {}),
            "baseline_runtime_health": dict(state.get("baseline_runtime_health") or {}),
            "performance_runtime": dict(state.get("performance_runtime") or {}),
        },
        "quantization_search": {
            "path": state.get("path"),
            "winner": state.get("best_candidate"),
            "status": search_state.get("status"),
            "termination_reason": search_state.get("termination_reason"),
            "api": search_state.get("api"),
            "config": dict(search_state.get("config") or {}),
            "baseline_metrics": dict(search_result.get("baseline_metrics") or {}),
            "all_results": list(search_result.get("all_results") or []),
            "total_configs_evaluated": int(search_result.get("total_configs_evaluated") or 0),
            "total_configs_available": int(search_result.get("total_configs_available") or 0),
            "search_time_seconds": search_result.get("search_time_seconds"),
            "candidate_queue": list(search_state.get("candidate_queue") or []),
            "candidate_cursor": int(search_state.get("candidate_cursor") or 0),
            "candidate_order_source": search_state.get("candidate_order_source"),
            "quant_checkpoint": state.get("quant_ckpt_dir"),
        },
        "accuracy": {
            "attempts": attempts,
            "accepted": accepted_accuracy,
            "threshold": spec.accuracy_gap,
            "profile": dict(state.get("eval_profile") or {}),
        },
        "evaluation_protocols": {
            "search_eval": {
                "evaluator": "quark_fakequant",
                "authoritative": False,
                "task": (
                    (search_state.get("config") or {}).get("eval_metrics", [None])[0]
                    if (search_state.get("config") or {}).get("eval_metrics")
                    else None
                ),
                "num_samples": (search_state.get("config") or {}).get("eval_num_samples"),
                "max_new_tokens": (search_state.get("config") or {}).get("eval_max_new_tokens"),
            },
            "real_accuracy_gate": {
                "evaluator": "lm_eval_vllm_exported_checkpoint",
                "authoritative": True,
                "task": (state.get("eval_profile") or {}).get("task"),
                "num_samples": spec.gsm8k_num_samples,
                "num_fewshot": (state.get("eval_profile") or {}).get("num_fewshot"),
                "prompting_strategy": (state.get("eval_profile") or {}).get("prompting_strategy"),
                "profile_hash": (state.get("eval_profile") or {}).get("profile_hash"),
            },
        },
        "performance": {
            "status": state.get("performance_status") or "not_completed",
            "measurements": list(state.get("performance_measurements") or []),
            "validation_policy": dict(state.get("performance_validation") or {}),
            "vendor_gemm_tuning": dict(state.get("vendor_gemm_tuning") or {}),
            "vendor_shape_evidence": dict(state.get("vendor_shape_evidence") or {}),
            "final": final_perf,
            "target_gain": spec.target_gain,
            "target_met": performance_target_met,
        },
        "capability_summary": _capability_summary(state),
        "phase_timeline": list(state.get("phase_timeline") or []),
        "bottleneck_analysis": bottleneck_analysis_from_state(state),
        "kernel_journey": list(state.get("kernel_journey") or []),
        "repair_journey": list(state.get("repair_journey") or []),
        "knowledge": {
            "queries": knowledge_queries,
            "experience_capture": dict(state.get("experience_capture") or {}),
        },
        "llm_calls": llm_calls,
        "change_ledger": list(state.get("change_ledger") or []),
        "retain_trials": list(state.get("retain_trials") or []),
        "patch_bundle": dict(state.get("patch_bundle") or {}),
        "attribution": {
            "quantization_gain": final_perf.get("quant_only_gain"),
            "final_gain": final_gain,
            "kernel_incremental_multiplier": (
                float(final_gain) / float(quant_only_gain) if final_gain is not None and quant_only_gain else None
            ),
            "kernel_retained_count": len(retained),
            "method": "validated" if final_perf.get("final_gain") is not None else "missing",
        },
        "final": {
            "status": package.status,
            "terminal_stage": terminal_stage,
            "accuracy": accepted_accuracy,
            "performance": final_perf,
            "target_met": performance_target_met,
            "retained_patches": retained,
            "deploy_package": package_dict,
        },
        "reporting": dict(state.get("reporting") or {}),
        "versions": {
            "quant_perf": quant_perf_version,
            "reporting": EXPORTER_VERSION,
            "python": platform.python_version(),
            "runtime": dict(state.get("runtime_inventory") or {}),
        },
        "warnings": list(dict.fromkeys(str(w) for w in warnings if w)),
        "source_files": source_files,
        "data_provenance": provenance,
        "artifact_manifest": {
            "found_count": sum(1 for item in provenance if item.get("found")),
            "missing_count": sum(1 for item in provenance if not item.get("found")),
            "found_sources": [item["source"] for item in provenance if item.get("found")],
            "missing_sources": [item["source"] for item in provenance if not item.get("found")],
            "known_file_bytes": sum(int(item.get("size_bytes") or 0) for item in provenance),
        },
    }


def build_final_summary(breakdown: dict[str, Any]) -> dict[str, Any]:
    """Project the breakdown into a compact dashboard/CLI result."""
    final = breakdown.get("final") or {}
    search = breakdown.get("quantization_search") or {}
    accuracy = final.get("accuracy") or {}
    performance = final.get("performance") or {}
    performance_summary = breakdown.get("performance") or {}
    capabilities = breakdown.get("capability_summary") or {}
    kernel_cap = capabilities.get("kernel_optimization") or {}
    bottleneck_analysis = breakdown.get("bottleneck_analysis") or {}
    profile = (breakdown.get("accuracy") or {}).get("profile") or {}
    return {
        "schema_version": FINAL_SCHEMA_VERSION,
        "generated_at_utc": _utc_now(),
        "session": breakdown.get("session") or {},
        "workload": breakdown.get("workload") or {},
        "quantization": {
            "winner": search.get("winner"),
            "candidates": _candidate_configs(search),
            "checkpoint": search.get("quant_checkpoint"),
            "search_status": search.get("status"),
            "termination_reason": search.get("termination_reason"),
            "total_configs_evaluated": search.get("total_configs_evaluated"),
            "total_configs_available": search.get("total_configs_available"),
        },
        "accuracy": {
            "baseline": accuracy.get("baseline"),
            "quantized": accuracy.get("quantized"),
            "gap": accuracy.get("gap"),
            "passed": accuracy.get("passed"),
            "threshold": (breakdown.get("accuracy") or {}).get("threshold"),
            "profile_id": ((breakdown.get("accuracy") or {}).get("profile") or {}).get("profile_id"),
            "profile_hash": (profile.get("profile_hash")),
            "evidence_hash": profile.get("evidence_hash"),
            "num_fewshot": profile.get("num_fewshot"),
            "prompting_strategy": profile.get("prompting_strategy"),
            "settings_source": profile.get("settings_source"),
            "source_reference": profile.get("source_reference"),
            "artifacts": dict(accuracy.get("artifacts") or {}),
        },
        "evaluation_protocols": (breakdown.get("evaluation_protocols") or {}),
        "performance": {
            **performance,
            "status": performance_summary.get("status"),
            "target_gain": performance_summary.get("target_gain"),
            "target_met": performance_summary.get("target_met"),
        },
        "optimization": {
            "kernels_attempted": kernel_cap.get("attempts", 0),
            "patches_retained": len(final.get("retained_patches") or []),
            "requested_bottleneck_mode": bottleneck_analysis.get("requested_mode"),
            "effective_bottleneck_mode": bottleneck_analysis.get("effective_mode"),
            "bottleneck_analysis_status": bottleneck_analysis.get("status"),
            "bottleneck_analysis_reason": bottleneck_analysis.get("reason"),
        },
        "recovery": {
            "attempt_count": (breakdown.get("recovery_details") or {}).get("attempt_count", 0),
            "effective_gpu_memory_utilization": (
                (breakdown.get("recovery_details") or {}).get("performance_runtime") or {}
            ).get("effective_gpu_memory_utilization"),
            "runtime_env": dict(
                ((breakdown.get("recovery_details") or {}).get("performance_runtime") or {}).get("runtime_env") or {}
            ),
            "baseline_source": ((breakdown.get("recovery_details") or {}).get("baseline_reference") or {}).get(
                "source"
            ),
        },
        "repositories": breakdown.get("repositories") or {},
        "reproduction": breakdown.get("reproduction") or {},
        "versions": breakdown.get("versions") or {},
        "warnings": breakdown.get("warnings") or [],
        "artifacts": {
            "patch_bundle": dict(breakdown.get("patch_bundle") or {}),
        },
    }


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_final_markdown(final: dict[str, Any]) -> str:
    session = final.get("session") or {}
    accuracy = final.get("accuracy") or {}
    perf = final.get("performance") or {}
    opt = final.get("optimization") or {}
    quant = final.get("quantization") or {}
    runtime = (final.get("versions") or {}).get("runtime") or {}
    runtime_packages = runtime.get("packages") or {}
    accelerator = runtime.get("accelerator") or {}
    performance_status = perf.get("status") or "not_completed"
    if performance_status == "not_requested":
        performance_summary = "- Performance: `not_requested`."
    elif performance_status == "measurement_failed":
        performance_summary = "- Performance: throughput measurement failed."
    else:
        performance_summary = (
            f"- Final gain: `{_fmt(perf.get('final_gain'))}x` "
            f"(target `{_fmt(perf.get('target_gain'))}x`, met={perf.get('target_met')})."
        )
    lines = [
        f"# Quark Quant-Perf Final Report — {session.get('session_id') or '(unknown)'}",
        "",
        "## Executive Summary",
        "",
        f"- Status: `{session.get('stop_reason') or 'unknown'}`.",
        f"- Winner: `{quant.get('winner')}`.",
        f"- Search: `{quant.get('search_status') or 'unknown'}` "
        f"(reason `{quant.get('termination_reason') or 'none'}`, "
        f"evaluated `{quant.get('total_configs_evaluated') or 0}`/"
        f"`{quant.get('total_configs_available') or 0}`).",
        f"- Accuracy gap: `{_fmt(accuracy.get('gap'))}` "
        f"(threshold `{_fmt(accuracy.get('threshold'))}`, passed={accuracy.get('passed')}).",
        f"- Accuracy profile: `{accuracy.get('profile_id') or 'unknown'}` "
        f"(hash `{accuracy.get('profile_hash') or 'unknown'}`).",
        f"- Eval settings: `{accuracy.get('num_fewshot')}`-shot "
        f"`{accuracy.get('prompting_strategy') or 'unknown'}` "
        f"(source `{accuracy.get('settings_source') or 'unknown'}`).",
        performance_summary,
        f"- Retained kernel patches: `{opt.get('patches_retained', 0)}`.",
        f"- Recovery attempts: `{(final.get('recovery') or {}).get('attempt_count', 0)}`.",
        f"- Runtime: vLLM `{(runtime_packages.get('vllm') or {}).get('version') or 'unknown'}`, "
        f"PyTorch `{(runtime_packages.get('torch') or {}).get('version') or 'unknown'}`, "
        f"Transformers `{(runtime_packages.get('transformers') or {}).get('version') or 'unknown'}`, "
        f"ROCm `{accelerator.get('rocm') or 'unknown'}`.",
        "",
        "## Key Results",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Baseline GSM8K | {_fmt(accuracy.get('baseline'))} |",
        f"| Quantized GSM8K | {_fmt(accuracy.get('quantized'))} |",
        f"| Baseline TPS | {_fmt(perf.get('baseline_tps'))} |",
        f"| Final TPS | {_fmt(perf.get('final_tps'))} |",
        f"| Final gain | {_fmt(perf.get('final_gain'))}x |",
        f"| Kernel attempts | {opt.get('kernels_attempted', 0)} |",
        f"| Retained patches | {opt.get('patches_retained', 0)} |",
    ]
    lines.extend(["", "## Repository Versions", ""])
    repositories = final.get("repositories") or {}
    if repositories:
        for role, repo in repositories.items():
            if repo:
                lines.append(f"- {role}: branch=`{repo.get('work_branch') or ''}`, commit=`{repo.get('head') or ''}`")
    else:
        lines.append("- No repository metadata recorded.")
    reproduction = final.get("reproduction") or {}
    commands = reproduction.get("commands") or {}
    lines.extend(["", "## Reproduce / Serve / Evaluate", ""])
    environment = reproduction.get("environment") or {}
    if environment:
        lines.append("Environment used by serving and benchmark commands:")
        lines.append("")
        lines.append("```bash")
        lines.extend(f"export {key}={shlex.quote(str(value))}" for key, value in sorted(environment.items()))
        lines.append("```")
    for label in ("original", "serve", "evaluate", "benchmark"):
        command = commands.get(label) or []
        if not command:
            continue
        lines.extend(
            [
                "",
                f"{label.capitalize()}:",
                "",
                "```bash",
                shlex.join(str(part) for part in command),
                "```",
            ]
        )
    replacements = reproduction.get("replace_before_running") or []
    if replacements:
        lines.extend(["", "Replace before running:", ""])
        lines.extend(f"- {item}" for item in replacements)
    patch_bundle = (final.get("artifacts") or {}).get("patch_bundle") or {}
    if patch_bundle:
        lines.extend(["", "Retained patch artifacts:", ""])
        lines.extend(f"- {role}: `{path}`" for role, path in patch_bundle.items())
    lines.extend(
        [
            "",
            "## Reports",
            "",
            "- Full structured breakdown: `../session_breakdown.json`",
            "- Full technical report: `../session_report.md`",
        ]
    )
    warnings = final.get("warnings") or []
    if warnings:
        lines.extend(["", "## Data Quality Notes", ""])
        lines.extend(f"- {w}" for w in warnings[:10])
    return "\n".join(lines).rstrip() + "\n"


def render_session_markdown(breakdown: dict[str, Any]) -> str:
    session = breakdown.get("session") or {}
    workload = breakdown.get("workload") or {}
    final = breakdown.get("final") or {}
    lines = [
        f"# Quark Quant-Perf Session Report — {session.get('session_id') or '(unknown)'}",
        "",
        "## Executive Summary",
        "",
        f"- Stop reason: `{session.get('stop_reason')}`.",
        f"- Terminal stage: `{session.get('terminal_stage')}`.",
        f"- Target met: `{final.get('target_met')}`.",
        "",
        "## Session & Workload",
        "",
        f"- Model: `{workload.get('model')}`",
        f"- Framework: `{workload.get('framework')}`",
        f"- GPU: `{workload.get('gpu_type')}` / `{workload.get('gpu_arch')}`",
        f"- Shape: TP={workload.get('tp')}, ISL={workload.get('isl')}, "
        f"OSL={workload.get('osl')}, concurrency={workload.get('benchmark_concurrency')}",
        "",
        "## Repositories & Versions",
        "",
        "| Role | Path | Original branch | Work branch | HEAD | Dirty |",
        "|---|---|---|---|---|---|",
    ]
    for role, repo in (breakdown.get("repositories") or {}).items():
        if not repo:
            continue
        lines.append(
            f"| {role} | `{repo.get('path') or ''}` | "
            f"`{repo.get('original_branch') or ''}` | "
            f"`{repo.get('work_branch') or ''}` | "
            f"`{repo.get('head') or ''}` | {repo.get('worktree_dirty')} |"
        )
    runtime = (breakdown.get("versions") or {}).get("runtime") or {}
    accelerator = runtime.get("accelerator") or {}
    lines.extend(
        [
            "",
            "## Runtime Environment",
            "",
            f"- Python: `{runtime.get('python') or 'unknown'}`",
            f"- Platform: `{runtime.get('platform') or 'unknown'}`",
            f"- ROCm: `{accelerator.get('rocm') or 'unknown'}`",
            f"- CUDA: `{accelerator.get('cuda') or 'unknown'}`",
            f"- GPUs: `{', '.join(accelerator.get('gpu_models') or []) or 'unknown'}`",
            "",
            "| Package | Version | Import origin | Source revision |",
            "|---|---|---|---|",
        ]
    )
    for name, package in (runtime.get("packages") or {}).items():
        lines.append(
            f"| {name} | `{package.get('version') or 'not installed'}` | "
            f"`{package.get('origin') or ''}` | "
            f"`{package.get('git_description') or package.get('git_sha') or ''}` |"
        )
    lines.extend(
        [
            "",
            "## Quantization Search",
            "",
            f"- Winner: `{(breakdown.get('quantization_search') or {}).get('winner')}`",
            f"- Status: `{(breakdown.get('quantization_search') or {}).get('status') or 'unknown'}`",
            f"- Termination reason: "
            f"`{(breakdown.get('quantization_search') or {}).get('termination_reason') or 'none'}`",
            f"- Candidates evaluated: `{(breakdown.get('quantization_search') or {}).get('total_configs_evaluated') or 0}`",
            f"- Candidate order: `{(breakdown.get('quantization_search') or {}).get('candidate_order_source') or 'unknown'}`",
            f"- Candidate cursor: `{(breakdown.get('quantization_search') or {}).get('candidate_cursor') or 0}`",
            "",
            "| # | Configuration | Search metrics | Status |",
            "|---:|---|---|---|",
        ]
    )
    for index, entry in enumerate(
        (breakdown.get("quantization_search") or {}).get("candidate_queue") or [],
        1,
    ):
        lines.append(
            f"| {index} | `{entry.get('config')}` | "
            f"`{entry.get('search_metrics') or {}}` | "
            f"{entry.get('status') or 'unknown'} |"
        )
    lines.extend(
        [
            "",
            "## Accuracy",
            "",
            f"- Profile: `{((breakdown.get('accuracy') or {}).get('profile') or {}).get('profile_id') or 'unknown'}`",
            f"- Profile hash: `{((breakdown.get('accuracy') or {}).get('profile') or {}).get('profile_hash') or 'unknown'}`",
            "",
            "| Attempt | Baseline | Quantized | Gap | Passed |",
            "|---:|---:|---:|---:|---|",
        ]
    )
    for i, attempt in enumerate((breakdown.get("accuracy") or {}).get("attempts") or [], 1):
        lines.append(
            f"| {i} | {_fmt(attempt.get('baseline'))} | "
            f"{_fmt(attempt.get('quantized'))} | {_fmt(attempt.get('gap'))} | "
            f"{attempt.get('passed')} |"
        )
    lines.extend(
        [
            "",
            "## Performance Results",
            "",
            f"- Status: `{(breakdown.get('performance') or {}).get('status') or 'not_completed'}`",
            "",
            "| Role | Baseline TPS | Quantized/Final TPS | Gain |",
            "|---|---:|---:|---:|",
        ]
    )
    for measurement in (breakdown.get("performance") or {}).get("measurements") or []:
        lines.append(
            f"| {measurement.get('role') or 'measurement'} | "
            f"{_fmt(measurement.get('baseline_tps'))} | "
            f"{_fmt(measurement.get('final_tps', measurement.get('quantized_tps')))} | "
            f"{_fmt(measurement.get('gain'))}x |"
        )
    lines.extend(
        [
            "",
            "## Capability Summary",
            "",
            "| Capability | Status | Attempts | Retained |",
            "|---|---|---:|---:|",
        ]
    )
    for name, capability in (breakdown.get("capability_summary") or {}).items():
        lines.append(
            f"| {name} | {capability.get('status') or '—'} | "
            f"{capability.get('attempts', 0)} | "
            f"{capability.get('retained', '—')} |"
        )
    lines.extend(["", "## Repair Journey", ""])
    repairs = breakdown.get("repair_journey") or []
    if repairs:
        lines.extend(
            [
                "| Failure class | Target | Status | Candidate | Rounds | Generation (s) | Verification (s) | Total (s) |",
                "|---|---|---|---|---:|---:|---:|---:|",
            ]
        )
        for repair in repairs:
            attempts = repair.get("attempts") or []
            generation = [attempt["generation_seconds"] for attempt in attempts if "generation_seconds" in attempt]
            verification = [
                attempt["verification_seconds"] for attempt in attempts if "verification_seconds" in attempt
            ]
            lines.append(
                f"| {repair.get('failure_class') or '—'} | "
                f"{repair.get('target_role') or '—'} | "
                f"{repair.get('status') or '—'} | "
                f"`{repair.get('candidate_id') or ''}` | "
                f"{_fmt(repair.get('rounds'))} | "
                f"{_fmt(sum(generation) if generation else None, 2)} | "
                f"{_fmt(sum(verification) if verification else None, 2)} | "
                f"{_fmt(repair.get('elapsed_seconds'), 2)} |"
            )
    else:
        lines.append("_No repair journey recorded._")
    lines.extend(["", "## Knowledge and LLM Audit", ""])
    knowledge = breakdown.get("knowledge") or {}
    queries = knowledge.get("queries") or []
    experience = knowledge.get("experience_capture") or {}
    recorded = sum(
        int(experience.get(key) or 0)
        for key in (
            "quantization",
            "repair",
            "kernel_optimization",
        )
    )
    llm_calls = breakdown.get("llm_calls") or []
    lines.append(f"- Knowledge queries: `{len(queries)}`")
    lines.append(f"- Runtime experiences recorded: `{recorded}`")
    lines.append(f"- Reviewable experiences: `{int(experience.get('reviewable') or 0)}`")
    lines.append(f"- LLM calls recorded: `{len(llm_calls)}`")
    lines.extend(["", "## Kernel Optimization", ""])
    journeys = breakdown.get("kernel_journey") or []
    if journeys:
        lines.extend(
            [
                "| Kernel | Source | Resolution | Outcome | Execution | Compile | Correctness | Micro speedup | E2E decision |",
                "|---|---|---|---|---|---|---|---:|---|",
            ]
        )
        for item in journeys:
            attempts = item.get("backend_attempts") or []
            attempt = attempts[-1] if attempts else {}
            speedup = attempt.get("micro_speedup")
            mapping = item.get("source_mapping") or {}
            source = mapping.get("source_file") or mapping.get("generated_source_file") or ""
            resolution = " / ".join(
                value
                for value in (
                    str(mapping.get("method") or ""),
                    str(mapping.get("confidence") or ""),
                )
                if value
            )
            if mapping.get("retryable"):
                resolution = f"{resolution} (retryable)".strip()
            execution = attempt.get("execution_status") or "—"
            if attempt.get("error"):
                execution += ": " + str(attempt["error"]).replace("|", "\\|").replace("\n", " ")
            lines.append(
                f"| `{item.get('name') or item.get('kernel_id')}` | "
                f"`{source}` | "
                f"{resolution or '—'} | "
                f"{item.get('outcome') or '—'} | "
                f"{execution} | "
                f"{attempt.get('compile_passed')} | "
                f"{attempt.get('correctness_passed')} | "
                f"{_fmt(speedup) + 'x' if speedup is not None else '—'} | "
                f"{(item.get('e2e') or {}).get('decision') or '—'} |"
            )
    else:
        lines.append("_No kernel optimization journey recorded._")
    lines.extend(["", "## Retain Trials", ""])
    retain_trials = breakdown.get("retain_trials") or []
    if retain_trials:
        lines.extend(
            [
                "| Kernel | Mode | Accuracy gap | TPS | Gain | Incremental | Effective floor | "
                "Initial decision | Final decision | Reason |",
                "|---|---|---:|---:|---:|---:|---:|---|---|---|",
            ]
        )
        for trial in retain_trials:
            lines.append(
                f"| `{trial.get('kernel_id') or ''}` | "
                f"{trial.get('mode') or '—'} | "
                f"{_fmt(trial.get('accuracy_gap'))} | "
                f"{_fmt(trial.get('throughput_tps'))} | "
                f"{_fmt(trial.get('gain'))}x | "
                f"{_fmt(trial.get('incremental_multiplier'))}x | "
                f"{_fmt(trial.get('effective_keep_floor'))} | "
                f"{trial.get('decision') or '—'} | "
                f"{trial.get('final_decision') or trial.get('decision') or '—'} | "
                f"{trial.get('reason') or '—'} |"
            )
    else:
        lines.append("_No final retain trials recorded._")
    attribution = breakdown.get("attribution") or {}
    lines.extend(
        [
            "",
            "## Gain Attribution",
            "",
            f"- Method: `{attribution.get('method') or 'missing'}`",
            f"- Quantization gain: `{_fmt(attribution.get('quantization_gain'))}x`",
            f"- Final gain: `{_fmt(attribution.get('final_gain'))}x`",
            f"- Kernel incremental multiplier: `{_fmt(attribution.get('kernel_incremental_multiplier'))}x`",
            f"- Retained kernel patches: `{attribution.get('kernel_retained_count', 0)}`",
        ]
    )
    lines.extend(["", "## Phase Timeline", ""])
    for event in breakdown.get("phase_timeline") or []:
        lines.append(f"- `{event.get('ts') or ''}` `{event.get('action') or ''}`: {event.get('status') or ''}")
    lines.extend(["", "## Source Artifacts", ""])
    for key, value in (breakdown.get("source_files") or {}).items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Data Provenance", ""])
    provenance = breakdown.get("data_provenance") or []
    if provenance:
        lines.extend(
            [
                "| Source | Found | Kind | Size | Path |",
                "|---|---|---|---:|---|",
            ]
        )
        for item in provenance:
            lines.append(
                f"| {item.get('source')} | {item.get('found')} | "
                f"{item.get('kind')} | {_fmt(item.get('size_bytes'), 0)} | "
                f"`{item.get('path')}` |"
            )
    else:
        lines.append("_No provenance records._")
    warnings = breakdown.get("warnings") or []
    if warnings:
        lines.extend(["", "## Data Quality Notes", ""])
        lines.extend(f"- {w}" for w in warnings)
    return "\n".join(lines).rstrip() + "\n"


def write_final_artifacts(
    spec: Spec,
    state: dict[str, Any],
    package: DeployPackage,
) -> dict[str, str]:
    """Atomically write the four FINAL-stage artifacts and return their paths."""
    session_dir = Path(spec.session_dir).resolve()
    reports_dir = session_dir / "reports"
    breakdown_path = session_dir / "session_breakdown.json"
    session_md_path = session_dir / "session_report.md"
    final_json_path = reports_dir / "final.json"
    final_md_path = reports_dir / "final.md"

    paths = {
        "final_json": str(final_json_path),
        "final_md": str(final_md_path),
        "session_breakdown_json": str(breakdown_path),
        "session_report_md": str(session_md_path),
    }
    kernel_provenance = Path(str(state.get("kernel_provenance_artifact") or session_dir / "kernel_provenance.json"))
    kernel_resolution = session_dir / "kernel_source_resolution.json"
    if kernel_provenance.is_file():
        paths["kernel_provenance_json"] = str(kernel_provenance.resolve())
    if kernel_resolution.is_file():
        paths["kernel_source_resolution_json"] = str(kernel_resolution.resolve())
    report_state = copy.deepcopy(state)
    terminal_stage = report_state.get("terminal_stage") or report_state.get("stage") or ""
    report_state["stage"] = terminal_stage
    report_state["reporting"] = {
        "status": "complete",
        "paths": dict(paths),
        "error": "",
    }
    timeline = report_state.setdefault("phase_timeline", [])
    if not timeline or not (timeline[-1].get("action") == "final" and timeline[-1].get("status") == "complete"):
        timeline.append(
            {
                "ts": _utc_now(),
                "action": "final",
                "status": "complete",
                "paths": dict(paths),
            }
        )
    report_package = copy.deepcopy(package)
    report_package.report_status = "complete"
    report_package.report_paths = dict(paths)
    breakdown = build_session_breakdown(spec, report_state, report_package)
    breakdown["report_artifacts"] = dict(paths)
    final = build_final_summary(breakdown)
    final["artifacts"] = paths

    _atomic_write_text(
        breakdown_path,
        json.dumps(breakdown, indent=2, sort_keys=True, default=str),
    )
    _atomic_write_text(
        final_json_path,
        json.dumps(final, indent=2, sort_keys=True, default=str),
    )
    _atomic_write_text(final_md_path, render_final_markdown(final))
    _atomic_write_text(session_md_path, render_session_markdown(breakdown))
    return paths
