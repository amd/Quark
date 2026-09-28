#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import gc
import importlib
import importlib.metadata
import json
import math
import os
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch

from quark.torch.quantization.config.config import QConfig, QLayerConfig

from .._serialization import sha256_json
from ..candidates import Candidate
from ..data import TokenDataset
from ..decision_space import DecisionSpace
from ..errors import (
    CandidateEvaluationError,
    CandidateQuantizationError,
    EvaluatorStateError,
    EvaluatorUnavailableError,
    PlanSelectionError,
    SchemaValidationError,
)
from ..evaluator import (
    EvaluatorCandidateResult,
    EvaluatorDescriptor,
    RuntimeQConfigAudit,
)
from ..hardware_capability import get_scheme_config
from ..ppl import PplResult
from ..qconfig_builder import qconfig_semantic_hash

_WORKER_CLASS = "quark.experimental.torch.plugin.fakequant_worker.QuarkFakeQuantWorker"


@dataclass(frozen=True, slots=True)
class VllmRuntimeConfig:
    model: str
    expected_model_state_fingerprint: str
    expected_model_behavior_fingerprint: str
    model_revision: str | None = None
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.8
    max_model_len: int = 4096
    trust_remote_code: bool = False

    def __post_init__(self) -> None:
        if not self.model:
            raise SchemaValidationError("vLLM model source must not be empty.")
        if not self.expected_model_state_fingerprint.startswith("sha256:"):
            raise SchemaValidationError("vLLM expected model fingerprint must be a SHA-256 value.")
        if not self.expected_model_behavior_fingerprint.startswith("sha256:"):
            raise SchemaValidationError("vLLM expected model behavior fingerprint must be a SHA-256 value.")
        if self.model_revision == "":
            raise SchemaValidationError("vLLM model revision must not be empty.")
        if self.tensor_parallel_size != 1:
            raise SchemaValidationError("MVP vLLM tensor_parallel_size must be 1.")
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise SchemaValidationError("vLLM gpu_memory_utilization must be in (0, 1].")
        if self.max_model_len <= 1:
            raise SchemaValidationError("vLLM max_model_len must be greater than one.")


def _target_tail(name: str) -> str:
    marker = "layers."
    index = name.find(marker)
    return name[index:] if index >= 0 else name


def _resolve_runtime_targets(
    decision_space: DecisionSpace,
    runtime_module_names: tuple[str, ...],
) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for unit in decision_space.payload.decision_units:
        exact = [name for name in runtime_module_names if name == unit.unit_id]
        candidates = exact or [
            name for name in runtime_module_names if _target_tail(name) == _target_tail(unit.unit_id)
        ]
        if len(candidates) != 1:
            raise CandidateQuantizationError(
                f"Decision unit {unit.unit_id!r} resolved to runtime targets {sorted(candidates)}."
            )
        resolved[unit.unit_id] = candidates[0]
    if len(set(resolved.values())) != len(resolved):
        raise CandidateQuantizationError("Multiple Decision Units resolve to one vLLM runtime target.")
    return resolved


def _build_runtime_qconfig(
    decision_space: DecisionSpace,
    candidate: Candidate,
    runtime_targets: dict[str, str],
) -> tuple[QConfig, tuple[str, ...], tuple[str, ...]]:
    assignment = {entry.module_name: entry.scheme for entry in candidate.assignment}
    layer_quant_config: dict[str, QLayerConfig] = {}
    native_targets: list[str] = []
    quantized_targets: list[str] = []
    for unit in decision_space.payload.decision_units:
        schemes = {assignment[member] for member in unit.members}
        if len(schemes) != 1:
            raise CandidateQuantizationError(
                f"Decision unit {unit.unit_id!r} has incompatible member schemes {sorted(schemes)}."
            )
        scheme = next(iter(schemes))
        runtime_target = runtime_targets[unit.unit_id]
        if scheme == "native":
            native_targets.append(runtime_target)
        else:
            layer_quant_config[runtime_target] = get_scheme_config(decision_space.payload.model_type, scheme)
            quantized_targets.append(runtime_target)
    return (
        QConfig(
            global_quant_config=QLayerConfig(),
            layer_quant_config=layer_quant_config,
            layer_type_quant_config={},
            exclude=sorted(native_targets),
            kv_cache_quant_config={},
            kv_cache_group=[],
            shared_scale_groups=[],
        ),
        tuple(sorted(quantized_targets)),
        tuple(sorted(native_targets)),
    )


def _logprob_value(entry: Any) -> float:
    value = getattr(entry, "logprob", entry)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise CandidateEvaluationError("vLLM returned a non-numeric prompt logprob.")
    result = float(value)
    if not math.isfinite(result):
        raise CandidateEvaluationError("vLLM returned a non-finite prompt logprob.")
    return result


def evaluate_manifest_ppl(llm: Any, token_dataset: TokenDataset, sampling_params_cls: type[Any]) -> PplResult:
    """Compute the planner token-PPL protocol from exact token-id prompts."""
    sampling_params = sampling_params_cls(
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=1,
        detokenize=False,
    )
    prompts = [{"prompt_token_ids": list(sequence)} for sequence in token_dataset.sequences]
    outputs = llm.generate(prompts, sampling_params=sampling_params, use_tqdm=False)
    if len(outputs) != len(prompts):
        raise CandidateEvaluationError(f"vLLM returned {len(outputs)} outputs for {len(prompts)} token sequences.")

    nll_sum = 0.0
    token_count = 0
    for sequence, output in zip(token_dataset.sequences, outputs, strict=True):
        prompt_logprobs = getattr(output, "prompt_logprobs", None)
        if not isinstance(prompt_logprobs, list) or len(prompt_logprobs) < len(sequence):
            raise CandidateEvaluationError("vLLM returned incomplete prompt logprobs.")
        for position, token_id in enumerate(sequence[1:], start=1):
            values = prompt_logprobs[position]
            if values is None or token_id not in values:
                raise CandidateEvaluationError(
                    f"vLLM prompt logprobs omit target token {token_id} at position {position}."
                )
            nll_sum -= _logprob_value(values[token_id])
            token_count += 1

    expected_count = sum(len(sequence) - 1 for sequence in token_dataset.sequences)
    if token_count != expected_count:
        raise CandidateEvaluationError(f"vLLM scored {token_count} tokens, expected {expected_count}.")
    try:
        ppl = math.exp(nll_sum / token_count)
    except OverflowError as exc:
        raise CandidateEvaluationError("vLLM PPL overflowed to a non-finite value.") from exc
    if not math.isfinite(ppl):
        raise CandidateEvaluationError("vLLM PPL is non-finite.")
    return PplResult(nll_sum=nll_sum, token_count=token_count, ppl=ppl)


def validate_vllm_decision_space(decision_space: DecisionSpace) -> None:
    """Reject deployment assignments that the fused vLLM evaluator cannot preserve."""
    if decision_space.payload.deployment_backend != "vllm":
        raise EvaluatorUnavailableError(
            "The vLLM evaluator cannot faithfully bind a no_fusion Decision Space: vLLM uses one "
            "quantizer and scale domain for each fused QKV or gate/up module. Use the HF evaluator "
            "or build the Decision Space for the vLLM deployment backend."
        )


class VllmEvaluator:
    """Single-engine vLLM evaluator for all candidates in one selection run."""

    def __init__(
        self,
        runtime: VllmRuntimeConfig,
        decision_space: DecisionSpace,
        *,
        llm_factory: Callable[..., Any] | None = None,
        sampling_params_cls: type[Any] | None = None,
        runtime_version: str | None = None,
    ) -> None:
        self._runtime = runtime
        self._decision_space = decision_space
        validate_vllm_decision_space(decision_space)
        if llm_factory is None or sampling_params_cls is None:
            try:
                vllm = importlib.import_module("vllm")
            except ImportError as exc:
                raise EvaluatorUnavailableError(
                    "vLLM evaluator was selected, but vLLM is not installed. "
                    "Install a compatible vLLM build or explicitly select the HF evaluator."
                ) from exc
            llm_factory = llm_factory or vllm.LLM
            sampling_params_cls = sampling_params_cls or vllm.SamplingParams
            runtime_version = runtime_version or importlib.metadata.version("vllm")
        self._sampling_params_cls = sampling_params_cls
        self._descriptor = EvaluatorDescriptor(
            backend="vllm",
            protocol="token_ppl_v1",
            implementation_version=1,
            runtime_version=runtime_version or "test-double",
            options={
                "model": runtime.model,
                "model_revision": runtime.model_revision,
                "expected_model_state_fingerprint": runtime.expected_model_state_fingerprint,
                "expected_model_behavior_fingerprint": runtime.expected_model_behavior_fingerprint,
                "tensor_parallel_size": runtime.tensor_parallel_size,
                "gpu_memory_utilization": runtime.gpu_memory_utilization,
                "max_model_len": runtime.max_model_len,
                "trust_remote_code": runtime.trust_remote_code,
                "enforce_eager": True,
                "enable_prefix_caching": False,
            },
        )
        self._closed = False
        previous_quant_cfg = os.environ.pop("QUANT_CFG", None)
        try:
            try:
                self._llm = llm_factory(
                    model=runtime.model,
                    tensor_parallel_size=runtime.tensor_parallel_size,
                    gpu_memory_utilization=runtime.gpu_memory_utilization,
                    max_model_len=runtime.max_model_len,
                    trust_remote_code=runtime.trust_remote_code,
                    enforce_eager=True,
                    enable_prefix_caching=False,
                    worker_cls=_WORKER_CLASS,
                )
            finally:
                if previous_quant_cfg is not None:
                    os.environ["QUANT_CFG"] = previous_quant_cfg
            worker_names = self._llm.collective_rpc("quark_quantizable_module_names")
            if not worker_names or any(tuple(names) != tuple(worker_names[0]) for names in worker_names):
                raise EvaluatorUnavailableError("vLLM workers reported inconsistent quantizable module inventories.")
            self._runtime_module_names = tuple(worker_names[0])
            self._runtime_targets = _resolve_runtime_targets(decision_space, self._runtime_module_names)
        except Exception:
            self.close()
            raise

    @property
    def descriptor(self) -> EvaluatorDescriptor:
        return self._descriptor

    def _reset_caches(self) -> None:
        try:
            reset_prefix = getattr(self._llm, "reset_prefix_cache", None)
            if callable(reset_prefix):
                reset_prefix()
            reset_mm = getattr(self._llm, "reset_mm_cache", None)
            if callable(reset_mm):
                reset_mm()
        except Exception as exc:
            raise EvaluatorStateError(f"vLLM cache reset failed: {exc}") from exc

    def _reset_model(self) -> None:
        try:
            results = self._llm.collective_rpc("reset_to_original")
        except Exception as exc:
            raise EvaluatorStateError(f"vLLM model reset failed: {exc}") from exc
        if not results or not all(results):
            raise EvaluatorStateError(f"vLLM model reset failed: {results}.")

    def evaluate_baseline(self, ppl_tokens: TokenDataset) -> PplResult:
        try:
            self._reset_caches()
            self._reset_model()
            return evaluate_manifest_ppl(self._llm, ppl_tokens, self._sampling_params_cls)
        except CandidateEvaluationError:
            raise
        except Exception as exc:
            raise PlanSelectionError(f"vLLM baseline evaluation failed: {exc}") from exc
        finally:
            try:
                self._reset_caches()
            except EvaluatorStateError:
                self.close()
                raise

    def evaluate_candidate(
        self,
        decision_space: DecisionSpace,
        candidate: Candidate,
        qconfig: QConfig,
        calibration_tokens: TokenDataset,
        ppl_tokens: TokenDataset,
    ) -> EvaluatorCandidateResult:
        runtime_qconfig, expected_quantized, expected_native = _build_runtime_qconfig(
            decision_space,
            candidate,
            self._runtime_targets,
        )
        runtime_qconfig_hash = sha256_json(runtime_qconfig.to_dict())
        try:
            self._reset_caches()
            self._reset_model()
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w",
                    encoding="utf-8",
                    prefix="quark_mixed_precision_",
                    suffix=".json",
                ) as config_file:
                    json.dump(runtime_qconfig.to_dict(), config_file)
                    config_file.flush()
                    worker_results = self._llm.collective_rpc(
                        "requantize_with_config",
                        args=(
                            config_file.name,
                            None,
                            len(calibration_tokens.sequences),
                            len(calibration_tokens.sequences[0]),
                            calibration_tokens.sequences,
                            calibration_tokens.token_hash,
                            runtime_qconfig_hash,
                        ),
                    )
                if not worker_results or any(result != worker_results[0] for result in worker_results):
                    raise CandidateQuantizationError("vLLM workers returned inconsistent quantization audits.")
                worker_audit = worker_results[0]
                if (
                    worker_audit.get("calibration_token_hash") != calibration_tokens.token_hash
                    or worker_audit.get("runtime_qconfig_hash") != runtime_qconfig_hash
                ):
                    raise CandidateQuantizationError("vLLM worker audit hashes do not match candidate inputs.")
                actual_quantized = tuple(sorted(worker_audit.get("quantized_modules", ())))
                if actual_quantized != expected_quantized:
                    raise CandidateQuantizationError(
                        f"vLLM quantized targets differ; expected={expected_quantized}, actual={actual_quantized}."
                    )
            except CandidateQuantizationError:
                raise
            except Exception as exc:
                raise CandidateQuantizationError(str(exc)) from exc

            self._reset_caches()
            try:
                ppl = evaluate_manifest_ppl(self._llm, ppl_tokens, self._sampling_params_cls)
            except CandidateEvaluationError:
                raise
            except Exception as exc:
                raise CandidateEvaluationError(str(exc)) from exc

            runtime_audit = RuntimeQConfigAudit(
                backend="vllm",
                assignment_hash=candidate.candidate_id,
                runtime_qconfig_hash=runtime_qconfig_hash,
                resolved_quantized=expected_quantized,
                resolved_native=expected_native,
                calibration_token_hash=calibration_tokens.token_hash,
                calibrated_sequences=int(worker_audit["calibrated_sequences"]),
                calibrated_tokens=int(worker_audit["calibrated_tokens"]),
            )
            return EvaluatorCandidateResult(
                ppl=ppl,
                qconfig_hash=qconfig_semantic_hash(qconfig),
                runtime_audit=runtime_audit,
            )
        finally:
            try:
                self._reset_model()
                self._reset_caches()
            except EvaluatorStateError:
                self.close()
                raise

    def close(self) -> None:
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if hasattr(self, "_llm"):
            del self._llm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


__all__ = [
    "VllmEvaluator",
    "VllmRuntimeConfig",
    "evaluate_manifest_ppl",
    "validate_vllm_decision_space",
]
