#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Trace-derived workload contracts for GEAK kernel optimization."""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_BASELINE_RELATIVE_TOLERANCE = 0.25
_CASE_FIELDS = (
    "dims",
    "dtypes",
    "count",
    "baseline_latency_ms",
    "weight",
    "weight_source",
)
_CASE_MISMATCH = (
    "GEAK workload case dimensions/counts or latencies are not valid for the supplied workload specification"
)

_CaseEvidence = tuple[Counter[str], int, float | None]


def _measured_case_matches_expected(
    measured: _CaseEvidence,
    expected: _CaseEvidence,
) -> bool:
    """Match the kernel operands reported by a harness to a full trace case."""
    measured_dims, measured_count, _ = measured
    expected_dims, expected_count, _ = expected
    # A trace can include parent-op inputs that the isolated kernel harness
    # does not consume. Every measured shape must still come from the trace.
    return measured_count == expected_count and measured_dims <= expected_dims


def _decode_flydsl_mixed_moe_stage2(
    kernel: dict[str, Any],
    source_binding: dict[str, Any],
) -> dict[str, Any]:
    compiler = str(source_binding.get("compiler") or "").lower()
    builder = str(source_binding.get("builder_symbol") or "")
    source = str(source_binding.get("source_relpath") or source_binding.get("source_file") or "")
    runtime_name = str(kernel.get("name") or "").lower()
    if (
        compiler != "flydsl"
        or builder != "compile_mixed_moe_gemm"
        or not source.endswith("mixed_moe_gemm_2stage.py")
        or "moe2" not in runtime_name
    ):
        return {}

    cases = kernel.get("cases")
    if not isinstance(cases, list) or not cases:
        return {}

    token_buckets: list[int] = []
    fixed_geometry: tuple[int, int, int, int] | None = None
    for case in cases:
        if not isinstance(case, dict):
            return {}
        dims = case.get("dims")
        dtypes = case.get("dtypes")
        if not isinstance(dims, list) or len(dims) < 4 or not isinstance(dtypes, list) or len(dtypes) < 3:
            return {}
        activation, _w1_packed, w2_packed, routing = dims[:4]
        if (
            not isinstance(activation, list)
            or len(activation) != 2
            or not isinstance(w2_packed, list)
            or len(w2_packed) != 3
            or not isinstance(routing, list)
            or len(routing) != 2
            or "float4_e2m1fn_x2" not in str(dtypes[2]).lower()
        ):
            return {}

        token, model_dim = activation
        experts, w2_model_dim, packed_inter_dim = w2_packed
        routing_token, topk = routing
        if token != routing_token or model_dim != w2_model_dim:
            return {}
        geometry = (
            int(model_dim),
            int(packed_inter_dim) * 2,
            int(experts),
            int(topk),
        )
        if fixed_geometry is None:
            fixed_geometry = geometry
        elif fixed_geometry != geometry:
            return {}
        token_buckets.append(int(token))

    if fixed_geometry is None:
        return {}
    model_dim, inter_dim, experts, topk = fixed_geometry
    return {
        "operator_family": "flydsl_mixed_moe_stage2",
        "model_dim": model_dim,
        "inter_dim": inter_dim,
        "experts": experts,
        "topk": topk,
        "token_buckets": token_buckets,
        "packed_fp4x2_last_dim_multiplier": 2,
    }


@dataclass(frozen=True)
class WorkloadContract:
    spec_path: str
    schema: str
    kernel_name: str
    cases: tuple[dict[str, Any], ...]
    logical_semantics: dict[str, Any]
    managed_sources: tuple[str, ...]
    baseline_relative_tolerance: float = _BASELINE_RELATIVE_TOLERANCE

    @classmethod
    def from_spec(
        cls,
        workload_spec_path: str,
        source_binding: dict[str, Any] | None,
    ) -> WorkloadContract | None:
        if not workload_spec_path:
            return None
        try:
            workload = json.loads(Path(workload_spec_path).read_text())
            kernel = workload["kernels"][0]
            raw_cases = kernel["cases"]
        except (KeyError, IndexError, OSError, TypeError, ValueError):
            return None
        if not isinstance(raw_cases, list):
            return None

        cases = tuple(
            {key: case[key] for key in _CASE_FIELDS if key in case}
            for case in raw_cases
            if isinstance(case, dict) and case.get("dims")
        )
        if not cases:
            return None

        binding = source_binding or {}
        managed_sources = tuple(
            str(binding[key])
            for key in (
                "source_repo",
                "source_file",
                "launcher_source_file",
            )
            if binding.get(key)
        )
        return cls(
            spec_path=str(workload_spec_path),
            schema=str(workload.get("schema") or "workload-v1"),
            kernel_name=str(kernel.get("name") or ""),
            cases=cases,
            logical_semantics=_decode_flydsl_mixed_moe_stage2(
                kernel,
                binding,
            ),
            managed_sources=managed_sources,
        )

    def prompt_suffix(self) -> str:
        contract: dict[str, Any] = {
            "schema": self.schema,
            "kernel": self.kernel_name,
            "cases": list(self.cases),
        }
        if self.logical_semantics:
            contract["logical_semantics"] = self.logical_semantics
        provenance = (
            "\nManaged source provenance:\n- " + "\n- ".join(self.managed_sources) if self.managed_sources else ""
        )
        tolerance_pct = int(self.baseline_relative_tolerance * 100)
        return (
            "\nAuthoritative workload contract (trace-derived):\n"
            "These cases override source examples and default model "
            "dimensions. When logical_semantics is present, use it directly "
            "instead of re-decoding packed storage dimensions. Build and "
            "benchmark the harness with these exact tensor dimensions and "
            "call counts. Before optimization, each pristine baseline "
            f"latency must be within {tolerance_pct}% of the trace "
            "baseline_latency_ms for that case. If not, fix the harness or "
            "record workload_aligned=false and stop optimization.\n"
            f"```json\n{json.dumps(contract)}\n```"
            f"{provenance}\n"
            "Do not use another checkout, installed package source, or "
            "example model dimensions in place of the managed source "
            "provenance above."
        )

    def validate_alignment(
        self,
        eval_dir: str,
    ) -> tuple[bool, str]:
        timing_path = Path(eval_dir) / "baseline_timing.json"
        try:
            timing = json.loads(timing_path.read_text())
        except (OSError, ValueError) as exc:
            return False, f"missing workload alignment evidence: {exc}"
        alignment_confirmed = timing.get("workload_aligned") is True
        unconfirmed_reason = "GEAK did not confirm workload_aligned=true for the supplied workload specification"
        measured_cases = timing.get("test_cases")
        if not isinstance(measured_cases, list):
            return False, _CASE_MISMATCH if alignment_confirmed else unconfirmed_reason

        expected = [self._case_evidence(case, "baseline_latency_ms") for case in self.cases]
        measured = [self._case_evidence(case, "latency_ms") for case in measured_cases]
        if any(item is None for item in expected + measured):
            return False, _CASE_MISMATCH
        if len(expected) != len(measured):
            return False, _CASE_MISMATCH

        unmatched = list(expected)
        for case_index, measured_case in enumerate(measured):
            case = measured_cases[case_index]
            case_description = f"dims={case['dims']}, count={case['count']}"
            _, _, measured_latency = measured_case
            match_index = next(
                (
                    index
                    for index, expected_case in enumerate(unmatched)
                    if _measured_case_matches_expected(measured_case, expected_case)
                ),
                None,
            )
            if match_index is None:
                expected_cases = [{"dims": case["dims"], "count": case["count"]} for case in self.cases]
                return False, f"{_CASE_MISMATCH}: {case_description}; expected={expected_cases}"
            _, _, expected_latency = unmatched.pop(match_index)
            if expected_latency is not None and measured_latency is None:
                return False, (
                    "GEAK baseline latency is missing for a trace workload case: "
                    f"{case_description}, trace={expected_latency * 1000:.6f} us"
                )
            if (
                expected_latency is not None
                and measured_latency is not None
                and abs(measured_latency - expected_latency) / expected_latency > self.baseline_relative_tolerance
            ):
                return False, (
                    "GEAK baseline latency differs by more than "
                    f"{int(self.baseline_relative_tolerance * 100)}% from "
                    f"the trace workload baseline: {case_description}, "
                    f"trace={expected_latency * 1000:.6f} us, measured={measured_latency * 1000:.6f} us, "
                    f"deviation={(measured_latency / expected_latency - 1) * 100:+.4f}%, "
                    f"tolerance=+/-{self.baseline_relative_tolerance * 100:g}%"
                )
        # Report measured violations before falling back to GEAK's alignment flag.
        if not alignment_confirmed:
            return False, unconfirmed_reason
        return True, "GEAK workload alignment confirmed"

    def analysis_violation(self, eval_dir: str) -> str:
        if not self.logical_semantics:
            return ""
        analysis_path = Path(eval_dir) / "analysis.json"
        if not analysis_path.is_file():
            return ""
        try:
            analysis = json.loads(analysis_path.read_text())
        except (OSError, ValueError):
            return ""

        missing = object()

        def find_value(
            value: Any,
            keys: tuple[str, ...],
        ) -> Any:
            if isinstance(value, dict):
                for key in keys:
                    if key in value:
                        return value[key]
                for child in value.values():
                    found = find_value(child, keys)
                    if found is not missing:
                        return found
            elif isinstance(value, list):
                for child in value:
                    found = find_value(child, keys)
                    if found is not missing:
                        return found
            return missing

        scalar_checks = (
            ("model_dim", ("model_dim_N", "model_dim")),
            ("inter_dim", ("inter_dim_K", "inter_dim")),
            ("experts", ("experts", "num_experts")),
            ("topk", ("topk", "top_k")),
        )
        for semantic_key, analysis_keys in scalar_checks:
            expected = self.logical_semantics.get(semantic_key)
            actual = find_value(analysis, analysis_keys)
            if expected is None or actual is missing:
                continue
            try:
                normalized_actual = int(actual)
            except (TypeError, ValueError):
                continue
            if normalized_actual != int(expected):
                return (
                    f"analysis {semantic_key}={normalized_actual} conflicts "
                    f"with workload {semantic_key}={int(expected)}"
                )

        expected_tokens = self.logical_semantics.get("token_buckets")
        actual_tokens = find_value(
            analysis,
            ("cases_tokens", "token_buckets"),
        )
        if expected_tokens and actual_tokens is not missing:
            try:
                normalized_actual_tokens = sorted(int(value) for value in actual_tokens)
                normalized_expected_tokens = sorted(int(value) for value in expected_tokens)
            except (TypeError, ValueError):
                normalized_actual_tokens = []
                normalized_expected_tokens = []
            if normalized_actual_tokens != normalized_expected_tokens:
                return (
                    "analysis token_buckets="
                    f"{normalized_actual_tokens} conflicts with workload "
                    f"token_buckets={normalized_expected_tokens}"
                )
        return ""

    def baseline_violation(self, eval_dir: str) -> str:
        timing_path = Path(eval_dir) / "baseline_timing.json"
        if not timing_path.is_file():
            return ""
        valid, reason = self.validate_alignment(eval_dir)
        return "" if valid else reason

    @staticmethod
    def _case_evidence(
        case: dict[str, Any],
        latency_key: str,
    ) -> tuple[Counter[str], int, float | None] | None:
        if not isinstance(case, dict):
            return None
        dims = case.get("dims")
        count = case.get("count")
        if not isinstance(dims, list) or not dims or count is None:
            return None
        try:
            normalized_count = int(count)
        except (TypeError, ValueError):
            return None
        latency = case.get(latency_key)
        if latency is not None:
            if isinstance(latency, bool):
                return None
            try:
                latency = float(latency)
            except (TypeError, ValueError):
                return None
            if not math.isfinite(latency) or latency <= 0:
                return None
        return (
            Counter(json.dumps(dim, separators=(",", ":")) for dim in dims),
            normalized_count,
            latency,
        )
