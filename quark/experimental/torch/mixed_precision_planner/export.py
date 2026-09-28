#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from torch import nn

from quark.torch.quantization.config.config import QConfig

from ._serialization import StrictSchema, atomic_write_json, load_json_object
from .data import TokenDataset
from .decision_space import DecisionSpace
from .errors import PlanSelectionError, SchemaValidationError
from .mixed_precision_plan import MixedPrecisionPlan
from .ppl import PplResult
from .qconfig_builder import QConfigAudit, audit_qconfig
from .sensitivity_profile import model_behavior_fingerprint, model_state_fingerprint


@dataclass(frozen=True, slots=True)
class ReloadMeasurement(StrictSchema):
    """Measurements produced by the fresh reload subprocess."""

    reloaded_hf_ppl: PplResult
    ppl_token_hash: str
    qconfig_hash: str
    meta_parameters: int
    meta_buffers: int

    def __post_init__(self) -> None:
        if not self.ppl_token_hash.startswith("sha256:") or not self.qconfig_hash.startswith("sha256:"):
            raise SchemaValidationError("Reload measurement hashes must be SHA-256 values.")
        if self.meta_parameters < 0 or self.meta_buffers < 0:
            raise SchemaValidationError("Reload measurement meta tensor counts must be non-negative.")

    @classmethod
    def load(cls, path: str | Path) -> ReloadMeasurement:
        return cls.from_dict(load_json_object(path))

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


@dataclass(frozen=True, slots=True)
class ReloadValidationResult(StrictSchema):
    """Same-backend HF comparison for the source and freshly reloaded export."""

    source_hf_ppl: PplResult
    reloaded_hf_ppl: PplResult
    degradation: float
    max_degradation: float
    ppl_token_hash: str
    qconfig_hash: str
    meta_parameters: int
    meta_buffers: int

    def __post_init__(self) -> None:
        if self.source_hf_ppl.token_count != self.reloaded_hf_ppl.token_count:
            raise SchemaValidationError("Reload validation PPL results must score the same token count.")
        expected_degradation = self.reloaded_hf_ppl.ppl / self.source_hf_ppl.ppl - 1.0
        if not math.isfinite(self.degradation) or not math.isclose(
            self.degradation,
            expected_degradation,
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise SchemaValidationError("Reload validation degradation does not match its HF PPL results.")
        if (
            not math.isfinite(self.max_degradation)
            or self.max_degradation < 0
            or self.degradation > self.max_degradation
        ):
            raise SchemaValidationError("Reload validation result does not pass its recorded quality gate.")
        if not self.ppl_token_hash.startswith("sha256:") or not self.qconfig_hash.startswith("sha256:"):
            raise SchemaValidationError("Reload validation hashes must be SHA-256 values.")
        if self.meta_parameters != 0 or self.meta_buffers != 0:
            raise SchemaValidationError("Reloaded model contains unresolved meta tensors.")

    @classmethod
    def load(cls, path: str | Path) -> ReloadValidationResult:
        return cls.from_dict(load_json_object(path))

    def save(self, path: str | Path) -> None:
        atomic_write_json(path, self.to_dict())


def _selected_qconfig_hash(plan: MixedPrecisionPlan) -> str:
    selected_evaluation = next(
        evaluation
        for evaluation in plan.payload.evaluations
        if evaluation.candidate_id == plan.payload.selected_candidate.candidate_id
    )
    assert selected_evaluation.qconfig_hash is not None
    return selected_evaluation.qconfig_hash


def audit_export_binding(
    model: nn.Module,
    qconfig: QConfig,
    decision_space: DecisionSpace,
    plan: MixedPrecisionPlan,
    calibration_tokens: TokenDataset,
    ppl_tokens: TokenDataset,
) -> QConfigAudit:
    """Prove that the final source model and QConfig still bind to the selected Plan."""
    decision_space.validate_integrity()
    plan.validate_integrity()
    if (
        plan.artifact.upstream.get("decision_space") != decision_space.artifact_id
        or plan.artifact.model_fingerprint != decision_space.model_fingerprint
    ):
        raise PlanSelectionError("Selected Plan does not belong to the supplied Decision Space.")
    if model_state_fingerprint(model) != plan.payload.model_state_fingerprint:
        raise PlanSelectionError("Final model weights differ from the profiled checkpoint.")
    if model_behavior_fingerprint(model) != plan.payload.model_behavior_fingerprint:
        raise PlanSelectionError("Final model behavior differs from the profiled checkpoint.")
    if (
        calibration_tokens.token_hash != plan.payload.calibration_token_hash
        or ppl_tokens.token_hash != plan.payload.ppl_token_hash
    ):
        raise PlanSelectionError("Final export token manifests differ from the selected Plan.")
    audit = audit_qconfig(model, decision_space, plan.payload.selected_candidate, qconfig)
    if audit.qconfig_hash != _selected_qconfig_hash(plan):
        raise PlanSelectionError("Final QConfig recipe differs from the evaluated Plan.")
    return audit


def verify_export_roundtrip(
    *,
    model_dir: str | Path,
    ppl_tokens: TokenDataset,
    source_hf_ppl: PplResult,
    expected_qconfig_hash: str,
    max_degradation: float,
    device: str,
    trust_remote_code: bool = False,
) -> ReloadValidationResult:
    """Reload an exported model in a fresh process and enforce its HF PPL gate."""
    if not expected_qconfig_hash.startswith("sha256:"):
        raise SchemaValidationError("Expected export QConfig hash must be a SHA-256 value.")
    if not math.isfinite(max_degradation) or max_degradation < 0:
        raise SchemaValidationError("Export max_degradation must be finite and non-negative.")

    export_dir = Path(model_dir)
    if not export_dir.is_dir():
        raise PlanSelectionError(f"Exported model directory does not exist: {export_dir}.")
    ppl_path = export_dir / "ppl_tokens.json"
    measurement_path = export_dir / "reload_measurement.json"
    result_path = export_dir / "reload_validation.json"
    measurement_path.unlink(missing_ok=True)
    result_path.unlink(missing_ok=True)
    ppl_tokens.save(ppl_path)

    command = [
        sys.executable,
        "-m",
        "quark.experimental.torch.mixed_precision_planner.reload_validation",
        "--model-dir",
        str(export_dir),
        "--tokens",
        str(ppl_path),
        "--device",
        device,
        "--output",
        str(measurement_path),
    ]
    if trust_remote_code:
        command.append("--trust-remote-code")
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise PlanSelectionError(
                f"Fresh-process reload failed with exit code {completed.returncode}: {completed.stderr.strip()}"
            )
        measurement = ReloadMeasurement.load(measurement_path)
        if measurement.ppl_token_hash != ppl_tokens.token_hash:
            raise PlanSelectionError("Fresh-process reload used a different PPL token manifest.")
        if measurement.qconfig_hash != expected_qconfig_hash:
            raise PlanSelectionError("Reloaded model QConfig differs from the exported assignment.")
        if measurement.meta_parameters != 0 or measurement.meta_buffers != 0:
            raise PlanSelectionError("Reloaded model contains unresolved meta tensors.")

        degradation = measurement.reloaded_hf_ppl.ppl / source_hf_ppl.ppl - 1.0
        if degradation > max_degradation:
            raise PlanSelectionError(f"Exported model PPL degradation {degradation:.6g} exceeds {max_degradation:.6g}.")
        result = ReloadValidationResult(
            source_hf_ppl=source_hf_ppl,
            reloaded_hf_ppl=measurement.reloaded_hf_ppl,
            degradation=degradation,
            max_degradation=max_degradation,
            ppl_token_hash=ppl_tokens.token_hash,
            qconfig_hash=measurement.qconfig_hash,
            meta_parameters=measurement.meta_parameters,
            meta_buffers=measurement.meta_buffers,
        )
        result.save(result_path)
        return result
    finally:
        measurement_path.unlink(missing_ok=True)


__all__ = [
    "ReloadMeasurement",
    "ReloadValidationResult",
    "audit_export_binding",
    "verify_export_roundtrip",
]
