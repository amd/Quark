#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""AccuracyGate: shared by the quantization gate and the GEAK kernel-patch
gate.

Design ref: IMPL_SPEC §4.7.

Offline-first design: all real accuracy evaluation goes through the same frozen
EvalProfile and gsm8k_eval_offline() (lm_eval's in-process vLLM backend). No
HTTP server is needed at any point in the accuracy gate.
"""

from __future__ import annotations

from pathlib import Path

from quark.experimental.torch.quant_perf.session.spec import AccuracyResult, Spec


class AccuracyGate:
    def __init__(self, spec: Spec):
        self.spec = spec
        # The baseline (source) gsm8k score, measured once per run and
        # reused by eval_quantized().
        self._source_cache: float | None = None
        self._baseline_artifact: str = ""
        self._artifact_attempts: dict[str, int] = {}

    def _next_artifact_dir(self, role: str) -> Path:
        attempt = self._artifact_attempts.get(role, 0) + 1
        root = Path(self.spec.session_dir) / "evaluation" / role
        artifact_dir = root / f"attempt-{attempt}"
        while artifact_dir.exists():
            attempt += 1
            artifact_dir = root / f"attempt-{attempt}"
        self._artifact_attempts[role] = attempt
        return artifact_dir

    def reserve_artifact_dir(self, role: str) -> Path:
        """Reserve the next durable artifact path for an evaluation role."""
        return self._next_artifact_dir(role)

    def warm_baseline(
        self,
        *,
        output_dir: str | Path | None = None,
    ) -> None:
        """Measures the baseline (unquantized) model's GSM8K score offline.

        The result is cached for the lifetime of this AccuracyGate instance so
        it is only measured once per run regardless of how many times
        eval_quantized() is called.
        """
        if self._source_cache is not None:
            return
        from quark.experimental.torch.quant_perf.evaluation.gsm8k import gsm8k_eval_offline

        output_dir = Path(output_dir) if output_dir is not None else self._next_artifact_dir("baseline")
        self._baseline_artifact = str(output_dir)
        self._source_cache = gsm8k_eval_offline(
            model_dir=self.spec.base_model,
            gpu_id=self.spec.gpu_id,
            num_questions=self.spec.gsm8k_num_samples,
            tp=self.spec.tp,
            gpu_memory_utilization=self.spec.vllm_gpu_memory_utilization,
            moe_backend=self.spec.vllm_moe_backend,
            profile=self.spec.eval_profile,
            trust_remote_code=self.spec.vllm_trust_remote_code,
            max_num_seqs=self.spec.vllm_max_num_seqs,
            runtime_python=self.spec.runtime_python,
            runtime_env=self.spec.runtime_env,
            kv_cache_dtype=self.spec.vllm_kv_cache_dtype,
            output_dir=output_dir,
        )

    def check_baseline_health(self) -> tuple[bool, str, float]:
        """Verify the framework can run the BASE (unquantized) model before any
        quantized measurement. Returns (healthy, diagnosis, baseline_gsm8k).

        Preparation and generation use the actual evaluation runtime. The
        score must clear spec.baseline_floor to reject silent failures, and is
        cached for the accuracy gap. floor<=0 disables the floor check.
        """
        try:
            self.warm_baseline()
        except Exception as e:  # a crash in the full eval is itself a framework failure
            return False, f"baseline GSM8K eval failed: {e}", 0.0
        base = self._source_cache or 0.0
        if base < self.spec.baseline_floor:
            return (
                False,
                f"baseline GSM8K {base:.4f} is below the health floor "
                f"{self.spec.baseline_floor:.4f} -- the base model is broken in "
                f"{self.spec.framework}, not a quantization issue",
                base,
            )
        return True, "", base

    def eval_quantized(
        self,
        quant_ckpt_dir: str,
        *,
        gpu_memory_utilization: float | None = None,
        output_dir: str | Path | None = None,
    ) -> AccuracyResult:
        """Evaluates the quantized checkpoint offline and returns an
        AccuracyResult.  warm_baseline() is called automatically if it has
        not already been called. Repair and retry policy is owned by the
        Orchestrator; load/inference failures are surfaced unchanged.
        """
        if self._source_cache is None:
            self.warm_baseline()
        from quark.experimental.torch.quant_perf.evaluation.gsm8k import gsm8k_eval_offline

        output_dir = Path(output_dir) if output_dir is not None else self._next_artifact_dir("quantized")
        score = gsm8k_eval_offline(
            model_dir=quant_ckpt_dir,
            gpu_id=self.spec.gpu_id,
            num_questions=self.spec.gsm8k_num_samples,
            tp=self.spec.tp,
            gpu_memory_utilization=(
                self.spec.vllm_gpu_memory_utilization if gpu_memory_utilization is None else gpu_memory_utilization
            ),
            moe_backend=self.spec.vllm_moe_backend,
            profile=self.spec.eval_profile,
            trust_remote_code=self.spec.vllm_trust_remote_code,
            max_num_seqs=self.spec.vllm_max_num_seqs,
            runtime_python=self.spec.runtime_python,
            runtime_env=self.spec.runtime_env,
            kv_cache_dtype=self.spec.vllm_kv_cache_dtype,
            output_dir=output_dir,
        )

        source_score = self._source_cache
        if source_score is None:
            raise RuntimeError("baseline score is unavailable after evaluation")
        gap = max(0.0, (source_score - score) / max(source_score, 1e-9))
        return AccuracyResult(
            gap=gap,
            source_gsm8k=source_score,
            quantized_gsm8k=score,
            passed=gap <= self.spec.accuracy_gap,
            artifacts={
                **({"baseline": self._baseline_artifact} if self._baseline_artifact else {}),
                "quantized": str(output_dir),
            },
        )
