#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.perfopt.bottleneck_analysis import BottleneckAnalysisResult
from quark.experimental.torch.quant_perf.perfopt.keep import is_kernel_candidate, kernel_record_id, make_kernel_id
from quark.experimental.torch.quant_perf.perfopt.kernel_provenance import (
    write_kernel_source_resolution,
)
from quark.experimental.torch.quant_perf.perfopt.kernel_workflow import parse_director_validation
from quark.experimental.torch.quant_perf.session.spec import Checkpoint


def _save(ckpt: Checkpoint) -> None:
    session_dir = getattr(ckpt, "session_dir", None)
    if isinstance(session_dir, str | Path):
        path = write_kernel_source_resolution(
            session_dir,
            list(ckpt.state.get("kernel_journey") or []),
        )
        ckpt.state["kernel_source_resolution_artifact"] = str(path)
    ckpt.save()


def load_geak_resume_state(
    ckpt: Checkpoint,
    *,
    active_repos: list[str] | None = None,
) -> tuple[set[str], list[dict[str, Any]]]:
    entries = ckpt.state.get("geak_patches") or []
    attempted = {kernel_record_id(e) for e in entries if kernel_record_id(e)}
    e2e_dropped = {
        kernel_record_id(row)
        for row in (ckpt.state.get("kernel_journey") or [])
        if str((row.get("e2e") or {}).get("decision") or "").upper() in {"DROP", "DROP_CONFIRMED", "NEEDS_REVIEW"}
    }
    e2e_adopted = {
        kernel_record_id(row)
        for row in (ckpt.state.get("kernel_journey") or [])
        if (
            str((row.get("e2e") or {}).get("decision") or "").upper() == "KEEP"
            and bool((row.get("e2e") or {}).get("validated"))
        )
    }
    kept = []
    for entry in entries:
        if not (
            entry.get("status") == "candidate"
            and entry.get("best_patch")
            and kernel_record_id(entry) not in e2e_dropped
            and kernel_record_id(entry) not in e2e_adopted
        ):
            continue
        # The ledger is a cache. Re-read the matching persisted validation and
        # verify the delivered bytes before reusing a candidate after restart.
        patch = Path(entry["best_patch"])
        artifacts = Path(entry.get("artifacts_dir") or patch.parent)
        entry["verified_speedup"] = None
        try:
            saved = json.loads((artifacts / "result.json").read_text())
            validation = saved.get("round_evaluation", {}).get("correctness", {}).get("result")
            if (
                not isinstance(validation, dict)
                or saved.get("micro_speedup_source") == "invalid_workload"
                or saved.get("patch_sha256") != hashlib.sha256(patch.read_bytes()).hexdigest()
            ):
                continue
        except (OSError, ValueError):
            continue
        evidence = parse_director_validation(validation)
        if not is_kernel_candidate({**evidence, "best_patch": str(patch)}):
            continue
        entry["verified_speedup"] = evidence["verified_speedup"]
        entry["micro_speedup_source"] = evidence["micro_speedup_source"]
        row = {
            "best_patch": entry["best_patch"],
            "verified_speedup": entry["verified_speedup"],
            "micro_speedup_source": entry.get("micro_speedup_source", ""),
            "kernel_name": entry.get("kernel_name", ""),
            "kernel_src": entry.get("kernel_src", ""),
            "kernel_repo": entry.get("kernel_repo", ""),
        }
        if active_repos:
            relpath = str(entry.get("kernel_relpath") or "")
            if not relpath:
                old_src = str(entry.get("kernel_src") or "")
                old_repo = str(entry.get("kernel_repo") or "")
                if old_src and old_repo:
                    candidate = os.path.relpath(old_src, old_repo)
                    if not candidate.startswith(".."):
                        relpath = candidate.replace(os.sep, "/")
            if relpath:
                for repo in active_repos:
                    current = Path(repo) / relpath
                    if current.is_file():
                        row.update(
                            {
                                "kernel_src": str(current),
                                "kernel_repo": str(Path(repo).resolve()),
                                "kernel_relpath": relpath,
                            }
                        )
                        break
        kept.append(row)
    return attempted, kept


def _record_kernel_discovery(
    ckpt: Checkpoint,
    bottlenecks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    existing = {kernel_record_id(row): row for row in (ckpt.state.get("kernel_journey") or []) if kernel_record_id(row)}
    candidates = []
    journeys = list(existing.values())
    for rank, bottleneck in enumerate(bottlenecks, start=1):
        kernel_id = make_kernel_id(bottleneck.get("op_name", ""))
        candidate = {
            "rank": rank,
            "kernel_id": kernel_id,
            "name": bottleneck.get("op_name", ""),
            "kernel_time_us": bottleneck.get("kernel_time_us"),
            "baseline_time_us": bottleneck.get("baseline_time_us"),
            "differential_type": bottleneck.get("differential_type"),
            "roofline_bound": bottleneck.get("roofline_bound"),
        }
        for key in (
            "device_kernel_name",
            "parent_op_name",
            "parent_op_names",
            "external_ids",
            "call_count",
            "shape_cases",
            "dtypes",
            "gemm_shape",
            "gemm_shapes",
            "gemm_shape_evidence",
            "tunableop_input",
            "source_file",
            "launcher_source_file",
            "launcher_line",
            "launcher_symbol",
            "launcher_sample_count",
            "launcher_launch_api",
            "launcher_evidence_method",
            "trace_parent_status",
        ):
            if bottleneck.get(key) not in (None, "", [], {}):
                candidate[key] = bottleneck[key]
        candidates.append(candidate)
        journey = existing.get(kernel_id)
        if journey is None:
            journey = {
                "kernel_id": kernel_id,
                "name": bottleneck.get("op_name", ""),
                "discovery": dict(candidate),
                "source_mapping": {},
                "backend_attempts": [],
                "e2e": {},
                "outcome": "selected",
            }
            journeys.append(journey)
            existing[kernel_id] = journey
        journey["kernel_id"] = kernel_id
        journey["discovery"] = dict(candidate)
    ckpt.state["kernel_journey"] = journeys
    return candidates


def record_bottleneck_analysis(
    ckpt: Checkpoint,
    analysis: BottleneckAnalysisResult,
) -> None:
    candidates = _record_kernel_discovery(
        ckpt,
        list(analysis.candidates),
    )
    value = analysis.to_dict()
    value["candidates"] = candidates
    ckpt.state["bottleneck_analysis"] = value
    ckpt.state.setdefault("phase_timeline", []).append(
        {
            "ts": datetime.now(UTC).isoformat(),
            "action": "bottleneck_analysis",
            "status": analysis.status,
            "requested_mode": analysis.requested_mode.value,
            "effective_mode": analysis.effective_mode,
            "reason": analysis.reason,
            "candidate_count": len(candidates),
        }
    )
    _save(ckpt)


def _journey_for(
    ckpt: Checkpoint,
    kernel_id: str,
) -> dict[str, Any] | None:
    for row in ckpt.state.get("kernel_journey") or []:
        if kernel_record_id(row) == kernel_id:
            return row
    return None


def record_kernel_skip(
    ckpt: Checkpoint,
    kernel_id: str,
    reason: str,
    source_mapping: dict[str, Any] | None = None,
) -> None:
    journey = _journey_for(ckpt, kernel_id)
    if journey is None:
        return
    journey["outcome"] = "skipped"
    journey["skip_reason"] = reason
    if source_mapping is not None:
        journey["source_mapping"] = dict(source_mapping)
    _save(ckpt)


def record_kernel_attempt(
    ckpt: Checkpoint,
    *,
    kernel_id: str,
    source_file: str,
    source_repo: str,
    source_reason: str,
    report: dict[str, Any],
    kept: bool,
    source_mapping: dict[str, Any] | None = None,
) -> None:
    journey = _journey_for(ckpt, kernel_id)
    if journey is None:
        return
    round_evaluation = report.get("round_evaluation") or {}
    correctness = round_evaluation.get("correctness", {}).get("success")
    compilation = round_evaluation.get("compilation", {}).get("success")
    status = report.get("candidate_status") or ("candidate" if kept else "rejected")
    journey["source_mapping"] = dict(
        source_mapping
        or {
            "source_file": source_file,
            "source_repo": source_repo,
            "reason": source_reason,
        }
    )
    attempts = journey.setdefault("backend_attempts", [])
    attempts.append(
        {
            "attempt": len(attempts) + 1,
            "ts": datetime.now(UTC).isoformat(),
            "backend": "geak",
            "micro_speedup": report.get("verified_speedup"),
            "compile_passed": compilation,
            "correctness_passed": correctness,
            "best_patch": report.get("best_patch", ""),
            "candidate_status": status,
            "execution_status": report.get("execution_status"),
            "timed_out": report.get("timed_out", False),
            "error": report.get("error"),
            "knowledge_ids": list(report.get("knowledge_ids") or []),
        }
    )
    journey["outcome"] = status
    ckpt.state.setdefault("phase_timeline", []).append(
        {
            "ts": datetime.now(UTC).isoformat(),
            "action": "geak",
            "status": status,
            "execution_status": report.get("execution_status"),
            "timed_out": report.get("timed_out", False),
            "kernel_id": kernel_id,
            "micro_speedup": report.get("verified_speedup"),
        }
    )
    _save(ckpt)


def record_candidate_retention_decision(
    ckpt: Checkpoint,
    kernel_ids: dict[str, str],
    *,
    patch: str,
    source: str,
    repo: str,
    decision: str,
    reason: str,
    score: float | None = None,
    gap: float | None = None,
    tps: float | None = None,
    gain: float | None = None,
    mode: str = "",
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    kernel_id = kernel_ids.get(patch, "")
    retain_trials = ckpt.state.setdefault("retain_trials", [])
    trial = {
        "attempt": len(retain_trials) + 1,
        "ts": datetime.now(UTC).isoformat(),
        "kernel_id": kernel_id,
        "patch": patch,
        "source": source,
        "repo": repo,
        "accuracy_score": score,
        "accuracy_gap": gap,
        "throughput_tps": tps,
        "gain": gain,
        "decision": decision,
        "reason": reason,
        "active": decision == "KEEP",
        "final_decision": decision,
    }
    if mode:
        trial["mode"] = mode
    if evidence:
        trial.update(evidence)
    retain_trials.append(trial)

    for journey in ckpt.state.get("kernel_journey") or []:
        if kernel_record_id(journey) != kernel_id:
            continue
        journey["e2e"] = {
            "accuracy_score": score,
            "accuracy_gap": gap,
            "throughput_tps": tps,
            "gain": gain,
            "validated": decision == "KEEP",
            "decision": decision,
            "reason": reason,
            "patch_path": patch or None,
        }
        if decision == "KEEP":
            journey["outcome"] = "adopted"
        elif decision in {"NEEDS_REVIEW", "RETRYABLE_FAULT"}:
            journey["outcome"] = "deferred"
        else:
            journey["outcome"] = "rejected"
        break

    for entry in ckpt.state.get("geak_patches") or []:
        if entry.get("best_patch") != patch:
            continue
        if decision == "KEEP":
            entry["status"] = "retained"
        elif decision == "NEEDS_REVIEW":
            entry["status"] = "needs_review"
        elif decision == "RETRYABLE_FAULT":
            entry["status"] = "candidate"
        else:
            entry["status"] = "rejected"
        break

    ckpt.record_phase_event(
        "retain_patch",
        decision.lower(),
        kernel_id=kernel_id,
        gain=gain,
        accuracy_gap=gap,
        reason=reason,
    )
    _save(ckpt)
    return trial
