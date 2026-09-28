#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Static types for the persisted Quant-Perf session state."""

from __future__ import annotations

from typing import Any, Required, TypeAlias, TypedDict

StageName: TypeAlias = str


class PerformanceMeasurement(TypedDict, total=False):
    attempt: int
    ts: str
    role: str
    mode: str
    baseline_tps: float
    quantized_tps: float
    final_tps: float
    gain: float
    isl: int
    osl: int
    concurrency: int
    effective_keep_floor: float
    candidate: str
    incremental_multiplier: float
    baseline_measurements: list[dict[str, Any]]
    final_measurements: list[dict[str, Any]]
    anchor_measurements: list[dict[str, Any]]
    candidate_measurements: list[dict[str, Any]]


class EvalProfileHistoryEntry(TypedDict, total=False):
    profile: dict[str, Any]
    resolver_version: int | None
    input_hash: str | None
    baseline_gsm8k: float | None
    baseline_reference: dict[str, Any] | None
    baseline_runtime_health: dict[str, Any] | None
    accuracy_validation: dict[str, Any] | None
    accuracy_attempts: list[dict[str, Any]]


class SessionState(TypedDict, total=False):
    schema_version: Required[int]
    session_id: Required[str]
    stage: Required[StageName]
    path: Required[str]
    run_spec: Required[dict[str, Any]]
    runtime_context: Required[dict[str, Any]]
    invocation: Required[dict[str, Any]]
    quant_ckpt_dir: str | None
    best_candidate: dict[str, Any] | None
    best_accuracy_gap: float | None
    mix_precision_search: dict[str, Any]
    baseline_gsm8k: float | None
    baseline_reference: dict[str, Any] | None
    baseline_runtime_fingerprint: str
    baseline_runtime_health: dict[str, Any] | None
    accuracy_validation: dict[str, Any] | None
    accuracy_runtime: dict[str, Any]
    accuracy_attempts: list[dict[str, Any]]
    post_repair_rechecked_configs: list[dict[str, Any]]
    eval_profile: dict[str, Any] | None
    eval_profile_hash: str | None
    eval_profile_resolver_version: int | None
    eval_profile_input_hash: str | None
    eval_profile_history: list[EvalProfileHistoryEntry]
    baseline_tps: float
    quant_tps: float
    performance_measurements: list[PerformanceMeasurement]
    performance_runtime: dict[str, Any] | None
    performance_status: str
    performance_validation: dict[str, Any]
    bottleneck_analysis: dict[str, Any]
    differential_candidates: list[dict[str, Any]]
    kernel_journey: list[dict[str, Any]]
    kernel_provenance_artifact: str | None
    kernel_source_resolution_artifact: str | None
    geak_patches: list[dict[str, Any]]
    vendor_gemm_tuning: dict[str, Any]
    vendor_shape_evidence: dict[str, Any]
    perfopt_history: list[dict[str, Any]]
    retained_runtime_env: dict[str, str]
    retention_stack: list[dict[str, Any]]
    retention_base: dict[str, Any]
    final_retention: dict[str, Any]
    retain_trials: list[dict[str, Any]]
    repair_journey: list[dict[str, Any]]
    recovery_attempts: list[dict[str, Any]]
    change_ledger: list[dict[str, Any]]
    phase_timeline: list[dict[str, Any]]
    runtime_inventory: dict[str, Any]
    runtime_origin_evidence: dict[str, Any]
    session_runtime: dict[str, Any]
    resolved_sources: dict[str, dict[str, Any]]
    repo_workspaces: dict[str, dict[str, Any]]
    workspace_root: str
    transient_resources: list[dict[str, Any]]
    cleanup: dict[str, Any]
    patch_bundle: dict[str, Any]
    fw_branch: str
    fw_original_branch: str
    kernel_repo: str | None
    kernel_branch: str | None
    kernel_original_branch: str | None
    base_unhealthy_diag: str
    terminal_stage: StageName | None
    terminal_result: dict[str, Any] | None
    reporting: dict[str, Any]
    report_paths: dict[str, str]
    experience_capture: dict[str, Any]
    knowledge_audit: list[dict[str, Any]]
    llm_calls: list[dict[str, Any]]
    backend_probes: list[dict[str, Any]]
    report_warnings: list[str]


def append_performance_measurement(
    state: SessionState,
    measurement: PerformanceMeasurement,
) -> None:
    measurements = state.setdefault("performance_measurements", [])
    measurement["attempt"] = len(measurements) + 1
    measurements.append(measurement)
