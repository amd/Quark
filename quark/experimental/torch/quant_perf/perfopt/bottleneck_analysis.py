#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Mode-aware bottleneck selection over already collected traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.perfopt.collect import select_engine_rank0_trace
from quark.experimental.torch.quant_perf.perfopt.locate import locate_bottlenecks
from quark.experimental.torch.quant_perf.perfopt.trace_evidence import (
    compare_differential_evidence,
    extract_kernel_evidence,
)
from quark.experimental.torch.quant_perf.session.state import SessionState


class BottleneckMode(StrEnum):
    DIFFERENTIAL = "differential"
    ABSOLUTE = "absolute"


@dataclass(frozen=True)
class BottleneckAnalysisResult:
    requested_mode: BottleneckMode
    effective_mode: str
    status: str
    reason: str
    candidates: tuple[dict[str, Any], ...]
    quantized_trace: str
    baseline_trace: str = ""
    trace_kind: str = ""
    policy_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["requested_mode"] = self.requested_mode.value
        value["candidates"] = [dict(candidate) for candidate in self.candidates]
        return value


def find_best_existing_trace(trace_dir: str | Path) -> Path | None:
    root = Path(trace_dir)
    if not root.exists():
        return None
    for pattern in (
        "decode_only_steady_state*.json.gz",
        "mixed_steady_state*.json.gz",
        "*.pt.trace.json.gz",
    ):
        selected = select_engine_rank0_trace(sorted(root.rglob(pattern)))
        if selected is not None:
            return selected
    return None


def find_comparable_traces(
    quantized_dir: str | Path,
    baseline_dir: str | Path,
) -> tuple[Path | None, Path | None, str]:
    quantized_root = Path(quantized_dir)
    baseline_root = Path(baseline_dir)
    for trace_kind, pattern in (
        ("decode_only", "decode_only_steady_state*.json.gz"),
        ("mixed", "mixed_steady_state*.json.gz"),
        ("raw", "*.pt.trace.json.gz"),
    ):
        quantized = select_engine_rank0_trace(sorted(quantized_root.rglob(pattern)) if quantized_root.exists() else [])
        baseline = select_engine_rank0_trace(sorted(baseline_root.rglob(pattern)) if baseline_root.exists() else [])
        if quantized is not None and baseline is not None:
            return quantized, baseline, trace_kind
    return None, None, ""


def analyze_existing_session_bottlenecks(
    session_dir: str | Path,
    *,
    mode: BottleneckMode | str,
    gpu_arch: str,
    top_n: int,
    gpu_arch_json_path: str | Path = "",
) -> BottleneckAnalysisResult:
    session = Path(session_dir)
    quantized_dir = session / "trace"
    baseline_dir = session / "trace_baseline"
    requested_mode = BottleneckMode(mode)

    if requested_mode is BottleneckMode.DIFFERENTIAL:
        quantized_trace, baseline_trace, trace_kind = find_comparable_traces(
            quantized_dir,
            baseline_dir,
        )
        if quantized_trace is None:
            quantized_trace = find_best_existing_trace(quantized_dir)
        if quantized_trace is None:
            raise ValueError(f"quantized engine trace not found under {quantized_dir}")
    else:
        quantized_trace = find_best_existing_trace(quantized_dir)
        baseline_trace = None
        trace_kind = ""
        if quantized_trace is None:
            raise ValueError(f"quantized engine trace not found under {quantized_dir}")

    absolute_candidates = locate_bottlenecks(
        quantized_trace,
        gpu=gpu_arch,
        top_n=top_n,
        gpu_arch_json_path=gpu_arch_json_path,
    )
    return analyze_bottlenecks(
        mode=requested_mode,
        quantized_trace=quantized_trace,
        baseline_trace=baseline_trace or "",
        trace_kind=trace_kind,
        absolute_candidates=absolute_candidates,
        top_n=top_n,
        fallback_reason=(
            "no_comparable_baseline_trace"
            if (requested_mode is BottleneckMode.DIFFERENTIAL and baseline_trace is None)
            else ""
        ),
    )


def bottleneck_analysis_from_state(state: SessionState) -> dict[str, Any]:
    current = state.get("bottleneck_analysis")
    if isinstance(current, dict) and "candidates" in current:
        return dict(current)

    legacy = list(state.get("differential_candidates") or [])
    if not legacy:
        return {}
    is_differential = all(candidate.get("differential_type") for candidate in legacy)
    return {
        "policy_version": 1,
        "requested_mode": "differential",
        "effective_mode": ("differential" if is_differential else "absolute_fallback"),
        "status": "completed" if is_differential else "degraded",
        "reason": "" if is_differential else "legacy_absolute_fallback",
        "candidates": legacy,
        "quantized_trace": "",
        "baseline_trace": "",
        "trace_kind": "",
    }


def bottleneck_candidates_from_state(
    state: SessionState,
) -> list[dict[str, Any]]:
    return list(bottleneck_analysis_from_state(state).get("candidates") or [])


def _tag_candidates(
    candidates: list[dict[str, Any]],
    *,
    selection_mode: str,
    selection_reason: str,
    top_n: int,
) -> tuple[dict[str, Any], ...]:
    tagged: list[dict[str, Any]] = []
    for candidate in candidates[:top_n]:
        row = dict(candidate)
        row["selection_mode"] = selection_mode
        row["selection_reason"] = selection_reason
        tagged.append(row)
    return tuple(tagged)


def analyze_bottlenecks(
    *,
    mode: BottleneckMode | str,
    quantized_trace: str | Path,
    absolute_candidates: list[dict[str, Any]],
    top_n: int,
    baseline_trace: str | Path = "",
    trace_kind: str = "",
    fallback_reason: str = "",
    slowdown_threshold: float = 1.15,
) -> BottleneckAnalysisResult:
    requested_mode = BottleneckMode(mode)
    quantized_path = str(quantized_trace)
    baseline_path = str(baseline_trace) if baseline_trace else ""

    if requested_mode is BottleneckMode.ABSOLUTE:
        return BottleneckAnalysisResult(
            requested_mode=requested_mode,
            effective_mode="absolute",
            status="completed",
            reason="",
            candidates=_tag_candidates(
                absolute_candidates,
                selection_mode="absolute",
                selection_reason="top_quantized_kernel_time",
                top_n=top_n,
            ),
            quantized_trace=quantized_path,
        )

    if not baseline_path:
        return BottleneckAnalysisResult(
            requested_mode=requested_mode,
            effective_mode="absolute_fallback",
            status="degraded",
            reason=fallback_reason or "baseline_trace_unavailable",
            candidates=_tag_candidates(
                absolute_candidates,
                selection_mode="absolute_fallback",
                selection_reason=(fallback_reason or "baseline_trace_unavailable"),
                top_n=top_n,
            ),
            quantized_trace=quantized_path,
            trace_kind=trace_kind,
        )

    try:
        differential = compare_differential_evidence(
            extract_kernel_evidence(quantized_path),
            extract_kernel_evidence(baseline_path),
            slowdown_threshold=slowdown_threshold,
            top_n=top_n,
        )
    except Exception:
        return BottleneckAnalysisResult(
            requested_mode=requested_mode,
            effective_mode="absolute_fallback",
            status="degraded",
            reason="differential_trace_parse_failed",
            candidates=_tag_candidates(
                absolute_candidates,
                selection_mode="absolute_fallback",
                selection_reason="differential_trace_parse_failed",
                top_n=top_n,
            ),
            quantized_trace=quantized_path,
            baseline_trace=baseline_path,
            trace_kind=trace_kind,
        )

    if not differential:
        return BottleneckAnalysisResult(
            requested_mode=requested_mode,
            effective_mode="absolute_fallback",
            status="degraded",
            reason="no_differential_candidates",
            candidates=_tag_candidates(
                absolute_candidates[:1],
                selection_mode="absolute_fallback",
                selection_reason="no_differential_candidates",
                top_n=top_n,
            ),
            quantized_trace=quantized_path,
            baseline_trace=baseline_path,
            trace_kind=trace_kind,
        )

    return BottleneckAnalysisResult(
        requested_mode=requested_mode,
        effective_mode="differential",
        status="completed",
        reason="",
        candidates=_tag_candidates(
            differential,
            selection_mode="differential",
            selection_reason="unique_or_slower_in_quantized",
            top_n=top_n,
        ),
        quantized_trace=quantized_path,
        baseline_trace=baseline_path,
        trace_kind=trace_kind,
    )
