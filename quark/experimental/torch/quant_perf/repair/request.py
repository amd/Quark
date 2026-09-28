#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from quark.experimental.torch.quant_perf.session.spec import Spec

from .evidence import extract_failure_evidence
from .types import RepairRequest

if TYPE_CHECKING:
    from quark.experimental.torch.quant_perf.runtime.recovery import FailureDiagnosis

_DEFAULT_ACCURACY_REPAIR_NUM_SAMPLES = 50


def build_repair_request(
    spec: Spec,
    *,
    failure_class: str,
    error: object,
    quant_ckpt_dir: str,
    verifier_profile: str,
    quant_signature: str = "",
    metrics: dict[str, Any] | None = None,
    verifier: Callable[[], tuple[bool, str]] | None = None,
    diagnosis: FailureDiagnosis | None = None,
) -> RepairRequest:
    return RepairRequest(
        failure_class=failure_class,
        error=str(error),
        evidence=getattr(diagnosis, "evidence", None) or extract_failure_evidence(error),
        model_dir=spec.model_dir,
        quant_ckpt_dir=quant_ckpt_dir,
        framework=spec.framework,
        framework_repo=(spec.active_framework_repo if spec.can_modify_framework else ""),
        kernel_repo=(spec.active_kernel_repo if spec.can_modify_kernel else ""),
        stack_fingerprint={
            "arch_fingerprint": spec.arch_fingerprint,
            "framework_version": spec.framework_version,
            "kernel_version": spec.kernel_version,
            "framework_source_kind": spec.framework_source_kind,
            "kernel_source_kind": spec.kernel_source_kind,
        },
        quant_signature=quant_signature,
        workload={
            "gpu_id": spec.gpu_id,
            "tp": spec.tp,
            "isl": spec.isl,
            "osl": spec.osl,
            "concurrency": spec.bench_concurrency,
        },
        immutable_constraints={
            "accuracy_gap": spec.accuracy_gap,
            "target_gain": spec.target_gain,
            "eval_profile": spec.eval_profile,
            "gpu_memory_utilization": spec.vllm_gpu_memory_utilization,
            "trust_remote_code": spec.vllm_trust_remote_code,
            "max_num_seqs": spec.vllm_max_num_seqs,
            "max_model_len": int(
                getattr(
                    spec.eval_profile,
                    "max_model_len",
                    4096,
                )
            ),
            "gsm8k_num_samples": spec.gsm8k_num_samples,
            "accuracy_repair_num_samples": min(
                _DEFAULT_ACCURACY_REPAIR_NUM_SAMPLES,
                spec.gsm8k_num_samples,
            ),
            "runtime_python": spec.runtime_python,
            "kv_cache_dtype": spec.vllm_kv_cache_dtype,
            "runtime_env": dict(spec.runtime_env),
            "original_enforce_eager": "--enforce-eager" in spec.expanded_vllm_args,
        },
        verifier_profile=verifier_profile,
        failure_code=str(getattr(diagnosis, "code", "") or ""),
        target_role=str(getattr(diagnosis, "target_role", "") or ""),
        evidence_signature=str(getattr(diagnosis, "evidence_signature", "") or ""),
        session_dir=spec.session_dir,
        metrics=dict(metrics or {}),
        verifier=verifier,
    )
