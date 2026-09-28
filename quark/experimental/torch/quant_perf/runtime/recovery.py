#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Deterministic failure classification and runtime fingerprints."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf.evaluation.preparation import EVALUATION_RUNTIME_VERSION
from quark.experimental.torch.quant_perf.repair.evidence import (
    extract_failure_evidence,
    is_missing_flydsl_backend_failure,
)
from quark.experimental.torch.quant_perf.repair.types import FailureEvidence
from quark.experimental.torch.quant_perf.runtime.backends import configure_aiter_mxfp4_moe_ksplit
from quark.experimental.torch.quant_perf.session.spec import Spec


@dataclass(frozen=True)
class FailureDiagnosis:
    failure_class: str
    cache_policy: str
    code: str
    repair_eligible: bool = False
    recovery: dict[str, Any] | None = None
    target_role: str = ""
    reason: str = ""
    evidence_signature: str = ""
    evidence: FailureEvidence | None = None


def _repairable_failure(
    failure_class: str,
    code: str,
    *,
    evidence_signature: str = "",
    target_role: str = "",
    reason: str = "",
) -> FailureDiagnosis:
    return FailureDiagnosis(
        failure_class,
        "hard",
        code,
        repair_eligible=True,
        target_role=target_role,
        reason=reason,
        evidence_signature=evidence_signature,
    )


def _managed_source_context(
    evidence: FailureEvidence,
    *,
    framework_repo: str,
    kernel_repo: str,
) -> tuple[str, str] | None:
    root_file = evidence.root_file.replace("\\", "/").lower()
    for role, repo in (
        ("kernel", kernel_repo),
        ("framework", framework_repo),
    ):
        normalized_repo = str(repo or "").replace("\\", "/").rstrip("/").lower()
        if normalized_repo and root_file.startswith(normalized_repo + "/"):
            relative_path = root_file[len(normalized_repo) :].lstrip("/")
            package_name = relative_path.split("/", 1)[0]
            return role, package_name
    return None


def _managed_source_failure(
    evidence: FailureEvidence,
    *,
    role: str,
) -> FailureDiagnosis:
    return _repairable_failure(
        role,
        "managed_source_failure",
        target_role=role,
        reason="managed_source",
        evidence_signature=evidence.signature,
    )


def _missing_module_root(evidence: FailureEvidence) -> str:
    if evidence.exception_type != "ModuleNotFoundError":
        return ""
    match = re.search(
        r"no module named ['\"]([^'\"]+)['\"]",
        evidence.exception_message,
        flags=re.IGNORECASE,
    )
    return match.group(1).split(".", 1)[0].lower() if match else ""


def classify_failure(
    error: object,
    *,
    framework_repo: str = "",
    kernel_repo: str = "",
) -> FailureDiagnosis:
    evidence = extract_failure_evidence(error)
    return replace(
        _classify_failure(error, evidence, framework_repo=framework_repo, kernel_repo=kernel_repo),
        evidence=evidence,
    )


def _classify_failure(
    error: object,
    evidence: FailureEvidence,
    *,
    framework_repo: str,
    kernel_repo: str,
) -> FailureDiagnosis:
    text = evidence.full_error
    lower = text.lower()
    timed_out = bool(getattr(error, "timed_out", False))

    if getattr(error, "code", "") == "address_in_use" or "address already in use" in lower:
        return FailureDiagnosis(
            "resource_contention",
            "soft",
            "address_in_use",
        )
    if "modulenotfounderror" in lower or "importerror:" in lower:
        managed_context = _managed_source_context(
            evidence,
            framework_repo=framework_repo,
            kernel_repo=kernel_repo,
        )
        if managed_context is not None:
            _role, package_name = managed_context
            missing_module = _missing_module_root(evidence)
            if not missing_module or missing_module == package_name:
                managed_failure = _managed_source_failure(
                    evidence,
                    role=_role,
                )
                return managed_failure
        return FailureDiagnosis("dependency", "hard", "dependency_error")
    if is_missing_flydsl_backend_failure(text):
        return _repairable_failure(
            "kernel",
            "kernel_backend_unavailable",
            target_role="kernel",
            reason="known_signature",
            evidence_signature=evidence.signature,
        )
    if "quark_moe.py" in lower and "'nonetype' object has no attribute 'to'" in lower:
        return _repairable_failure(
            "framework",
            "quark_biasless_moe",
        )
    if (
        "_load_per_tensor_weight_scale" in text
        and "param_data[expert_id] = loaded_weight" in text
        and "expand(torch.bytetensor{" in lower
        and "size=[]" in lower
    ):
        return _repairable_failure(
            "framework",
            "quark_mxfp4_block_scale_as_scalar",
            target_role="framework",
            reason="known_signature",
            evidence_signature=evidence.signature,
        )
    if re.search(r"size of tensor .* must match|shape .* mismatch", lower):
        return _repairable_failure(
            "framework",
            "tensor_shape_mismatch",
        )
    if any(
        marker in lower
        for marker in (
            "unsupported architecture",
            "not implemented for",
            "no operator found",
            "unknown quantization method",
        )
    ):
        return _repairable_failure(
            "framework",
            "framework_compatibility",
        )
    if ("outofmemoryerror" in lower or "out of memory" in lower) and "_maybe_pad_weight" in text:
        return FailureDiagnosis(
            "resource_capacity",
            "soft",
            "rocm_moe_padding_oom",
            recovery={
                "action": "set_env",
                "key": "VLLM_ROCM_MOE_PADDING",
                "value": "0",
            },
        )
    if "no available memory for the cache blocks" in lower:
        return FailureDiagnosis(
            "resource_capacity",
            "soft",
            "kv_cache_no_memory",
            recovery={
                "action": "increase_gpu_memory_utilization",
                "step": 0.05,
                "maximum": 0.95,
            },
        )
    if "free memory on device" in lower and "desired gpu memory utilization" in lower:
        return FailureDiagnosis(
            "resource_contention",
            "soft",
            "gpu_memory_occupied",
        )
    if "hsa_status_error_out_of_resources" in lower:
        return FailureDiagnosis(
            "transient",
            "soft",
            "hsa_out_of_resources",
            recovery={"action": "retry_same"},
        )
    if timed_out or "timed out" in lower or "timeoutexpired" in lower:
        return FailureDiagnosis(
            "timeout",
            "soft",
            "timeout",
            recovery={"action": "retry_same"},
        )
    if any(
        marker in lower
        for marker in (
            "hiperrorstreamcaptureinvalidated",
            "operation failed due to a previous error during capture",
            "stream capture invalidated",
        )
    ):
        return FailureDiagnosis(
            "kernel",
            "soft",
            "cuda_graph_capture",
            recovery={"action": "retry_same"},
        )
    if "outofmemoryerror" in lower or "out of memory" in lower:
        return FailureDiagnosis(
            "resource_capacity",
            "soft",
            "capacity_oom",
        )
    if any(marker in lower for marker in ("kernel launch", "invalid device function")):
        return _repairable_failure(
            "kernel",
            "kernel_runtime_error",
            target_role="kernel",
            reason="known_signature",
            evidence_signature=evidence.signature,
        )

    managed_context = _managed_source_context(
        evidence,
        framework_repo=framework_repo,
        kernel_repo=kernel_repo,
    )
    if managed_context is not None:
        role, _package_name = managed_context
        return _managed_source_failure(evidence, role=role)
    return FailureDiagnosis(
        "unknown",
        "none",
        "unknown",
        evidence_signature=evidence.signature,
    )


_RUNTIME_ENV_KEYS = (
    "QUARK_QUANT_PERF_MXFP4_MOE_BACKEND",
    "QUARK_QUANT_PERF_MXFP4_GEMM_BACKEND",
    "QUARK_QUANT_PERF_W4A8_GEMM_BACKEND",
    "AITER_CONFIG_FMOE",
    "AITER_FLYDSL_FORCE",
    "AITER_KSPLIT",
    "VLLM_ROCM_MOE_PADDING",
    "VLLM_ROCM_USE_AITER",
    "VLLM_ROCM_USE_AITER_MOE",
    "VLLM_ROCM_USE_AITER_FLYDSL_MOE",
    "VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE",
    "VLLM_ROCM_MXFP4_GEMM_BACKEND",
    "VLLM_ROCM_W4A8_GEMM_BACKEND",
    "VLLM_ROCM_USE_AITER_FP4_ASM_GEMM",
)


def _runtime_env_fingerprint(env: dict[str, str]) -> dict[str, str]:
    return {
        key: str(env[key]) for key in sorted(env) if key in _RUNTIME_ENV_KEYS or key.startswith("AITER_CONFIG_GEMM_")
    }


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_hash(model_ref: str, filename: str) -> str:
    path = Path(model_ref) / filename
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_accuracy_fingerprint(
    spec: Spec,
    quant_ckpt_dir: str,
    *,
    framework_commit: str,
    kernel_commit: str,
    runtime_env: dict[str, str] | None = None,
) -> str:
    env = dict(runtime_env if runtime_env is not None else os.environ)
    configure_aiter_mxfp4_moe_ksplit(
        env,
        quant_ckpt_dir,
        spec.vllm_moe_backend or spec.mxfp4_moe_backend,
    )
    payload = {
        "quant_ckpt_dir": str(Path(quant_ckpt_dir).resolve()),
        "quant_config_hash": _file_hash(quant_ckpt_dir, "config.json"),
        "quant_index_hash": _file_hash(
            quant_ckpt_dir,
            "model.safetensors.index.json",
        ),
        "framework": spec.framework,
        "framework_commit": framework_commit,
        "kernel_commit": kernel_commit,
        "gpu_type": spec.gpu_type,
        "gpu_arch": spec.gpu_arch,
        "tp": spec.tp,
        "gsm8k_num_samples": spec.gsm8k_num_samples,
        "vllm_args": spec.inference_vllm_args,
        "eval_profile_hash": (spec.eval_profile.profile_hash if spec.eval_profile is not None else ""),
        "evaluation_runtime_version": EVALUATION_RUNTIME_VERSION,
        "runtime_env": _runtime_env_fingerprint(env),
    }
    if (kv_cache_dtype := spec.vllm_kv_cache_dtype) != "auto":
        payload["kv_cache_dtype"] = kv_cache_dtype
    return _payload_fingerprint(payload)


def build_runtime_fingerprint(
    spec: Spec,
    *,
    framework_commit: str,
    runtime_env: dict[str, str] | None = None,
    stage: str,
    effective_gpu_memory_utilization: float | None = None,
) -> str:
    env = runtime_env if runtime_env is not None else os.environ
    payload = {
        "model_ref": str(Path(spec.base_model).resolve()),
        "model_config_hash": _file_hash(spec.base_model, "config.json"),
        "model_index_hash": _file_hash(
            spec.base_model,
            "model.safetensors.index.json",
        ),
        "framework": spec.framework,
        "framework_commit": framework_commit,
        "gpu_type": spec.gpu_type,
        "gpu_arch": spec.gpu_arch,
        "tp": spec.tp,
        "gsm8k_num_samples": spec.gsm8k_num_samples,
        "vllm_args": spec.inference_vllm_args,
        "effective_gpu_memory_utilization": (effective_gpu_memory_utilization),
        "eval_profile_hash": (spec.eval_profile.profile_hash if spec.eval_profile is not None else ""),
        "evaluation_runtime_version": EVALUATION_RUNTIME_VERSION,
        "runtime_env": _runtime_env_fingerprint(env),
        "stage": stage,
    }
    if (kv_cache_dtype := spec.vllm_kv_cache_dtype) != "auto":
        payload["kv_cache_dtype"] = kv_cache_dtype
    return _payload_fingerprint(payload)
