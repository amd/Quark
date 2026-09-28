#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quant-Perf data contracts: Spec, Checkpoint, and the simple result types.

Design ref: IMPL_SPEC §5.1 / §5.1.1 / §4.1 (state.json schema + reliability trio).
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import signal
import uuid
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import UTC, datetime
from pathlib import Path
from subprocess import Popen
from typing import Any, Literal, TextIO, cast

from quark.experimental.torch.quant_perf.session.persistence import write_json_atomic
from quark.experimental.torch.quant_perf.session.state import SessionState

SCHEMA_VERSION = 9
DEFAULT_SERVER_HOST = "127.0.0.1"
PerformanceMode = Literal["off", "measure", "optimize"]

_RUNTIME_FIELD_NAMES = {
    "framework_source_kind",
    "kernel_source_kind",
    "framework_source_origin",
    "kernel_source_origin",
    "runtime_python",
    "runtime_env",
    "runtime_origins",
    "framework_branch",
    "framework_version",
    "kernel_branch",
    "kernel_version",
    "framework_worktree",
    "kernel_worktree",
}


@dataclass(frozen=True)
class EvalProfile:
    """Frozen settings shared by every real GSM8K measurement in a session."""

    profile_id: str
    profile_hash: str
    model_mode: str
    apply_chat_template: bool
    enable_thinking: bool | None
    detection_reason: str
    policy_version: str = "quark-quant-perf-gsm8k-profile-v1"
    task: str = "gsm8k_cot_zeroshot"
    metric: str = "exact_match,flexible-extract"
    max_model_len: int = 8192
    max_gen_toks: int = 1024
    batch_size: str = "auto"
    schema_version: int = 1
    benchmark: str = "gsm8k"
    request_type: str = "generate_until"
    num_fewshot: int = 0
    prompting_strategy: str = "cot"
    thinking_control: str = ""
    system_instruction: str = ""
    gen_kwargs: dict[str, object] = field(default_factory=lambda: {"temperature": 0, "top_p": 1})
    evaluation_purpose: str = "quality_gate"
    settings_source: str = "quant_perf_profile"
    source_reference: str = ""
    reference_score: float | None = None
    reference_setting: str = ""
    evidence_hash: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalProfile:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def with_computed_hash(self) -> EvalProfile:
        payload = self.to_dict()
        payload.pop("profile_hash", None)
        payload.pop("evidence_hash", None)
        payload.pop("detection_reason", None)
        payload.pop("settings_source", None)
        payload.pop("source_reference", None)
        payload.pop("reference_score", None)
        payload.pop("reference_setting", None)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return replace(self, profile_hash=digest)


@dataclass
class RuntimeContext:
    """Mutable execution details discovered after a run specification is fixed."""

    resolved_framework_repo: str = ""
    resolved_kernel_repo: str = ""
    framework_source_kind: str = ""
    kernel_source_kind: str = ""
    framework_source_origin: str = ""
    kernel_source_origin: str = ""
    runtime_python: str = ""
    runtime_env: dict[str, str] = field(default_factory=dict)
    runtime_origins: dict[str, str] = field(default_factory=dict)
    framework_branch: str = ""
    framework_version: str = ""
    kernel_branch: str = ""
    kernel_version: str = ""
    framework_worktree: str = ""
    kernel_worktree: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value not in ("", None, {}, [])}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> RuntimeContext:
        values = dict(data or {})
        known = {item.name for item in fields(cls)}
        return cls(**{key: value for key, value in values.items() if key in known})


@dataclass(frozen=True)
class Spec:
    """The complete run specification. Produced by intake.py (CLI parsing +
    arch fingerprinting), consumed by Orchestrator.run().
    """

    # -- Required ---------------------------------------------------------
    model_dir: str  # path to the model to quantize
    base_model: str  # baseline (original, unquantized) for accuracy/perf comparison
    framework: str  # "vllm" | "atom" (experimental)
    gpu_type: str  # "mi300x" | "mi325x" | "mi350x" | "mi355x"
    gpu_arch: str  # TraceLens arch name ("MI300X" etc.), corresponds to gpu_type
    isl: int  # input sequence length (affects roofline estimation and profiling)
    osl: int  # output sequence length
    quant_strategy: str | None  # None -> mix_precision_search; "fp8"/"mxfp4"/etc. -> direct_ptq

    # -- Quantization control ----------------------------------------------
    accuracy_gap: float = 0.02  # max allowed accuracy drop (relative to source)
    performance_mode: PerformanceMode | None = None
    # Compatibility inference for persisted and programmatic specs: a supplied
    # target means optimize, while no target means performance was not requested.
    target_gain: float | None = None  # desired throughput multiplier (relative to base_model)
    kv_cache_scheme: str | None = None  # None or "fp8" (Atom's flag only accepts bf16/fp8)
    exclude_layers: list[str] | None = None
    file2file_export: bool = False  # export mixed-precision candidates through the shard-wise public API
    layer_precision_candidates: list[str] | None = None  # None = auto: search every scheme the GPU
    # target supports EXCEPT mxfp6_e2m3 (arch-aware, resolved in search.py:
    # gfx942 -> native/fp8/ptpc_fp8; gfx950 -> + mxfp4/mxfp4_fp8). Explicit
    # CLI candidates restrict quantized modes or re-add mxfp6_e2m3; native is
    # always retained as the mixed-precision fallback.
    kv_cache_precision_candidates: list[str] = field(default_factory=lambda: ["native", "fp8"])
    max_search_candidates: int = 20  # mix_precision_search candidate limit; 0 exhausts
    # the generated search space and disables search-decision early stopping.
    search_timeout_s: float = 43200.0  # total isolated mixed-precision search/export wall-clock budget
    search_gpu_memory_utilization: float | None = None  # override shared vLLM memory for search only

    # -- Baseline health gate ----------------------------------------------
    baseline_floor: float = 0.03  # min baseline (unquantized) GSM8K for the run to
    # be meaningful. A base model that scores below this in the framework is
    # broken there (crash / silent fallback / garbage output), which makes the
    # accuracy gap and throughput gain undefined -- the pipeline aborts with
    # `base_unhealthy` instead of measuring the quantized model. Set 0 to disable.
    fix_base_framework: bool = False  # opt-in: on an unhealthy baseline, make ONE
    # bounded repair attempt for the BASE model before aborting.
    # Default off -- fixing base-model framework support is out of Quark Quant-Perf's core
    # scope (quant/perf co-design) and open-ended; the default is fail-fast.
    recheck_baseline: bool = False  # ignore an exact cached baseline failure
    # and run the real health gate again. --fix-base-framework implies this.
    retry_accuracy_gate: bool = False  # explicitly reopen a terminal failed
    # session at the quantized accuracy gate. If its saved quant checkpoint was
    # deleted, the persisted exhaustive-search winner is re-exported first.
    retry_perfopt: bool = False  # explicitly reopen a terminal perf_failed
    # session at bottleneck discovery using validated upstream artifacts.

    # -- Accuracy evaluation ------------------------------------------------
    gsm8k_num_samples: int = 1319  # authoritative baseline and exported-checkpoint gate
    search_gsm8k_num_samples: int | None = None  # per-candidate search budget;
    # None preserves the historical behavior by using gsm8k_num_samples.
    vllm_extra_args: list[str] = field(default_factory=list)
    # None preserves a legacy --moe-backend passthrough override. Otherwise
    # search delegates to mix_precision and inference keeps model-aware auto.
    search_moe_backend: str | None = None
    inference_moe_backend: str | None = None
    # MXFP4 MoE kernel backend: AITER/CK, AITER Triton, or AITER FlyDSL.
    mxfp4_moe_backend: str = "aiter"
    # Dense dynamic MXFP4 x MXFP4 GEMM backend.
    mxfp4_gemm_backend: str = "triton"
    # Dense static FP8-per-tensor x MXFP4 GEMM backend.
    w4a8_gemm_backend: str = "triton"
    aiter_config_fmoe: str = ""
    eval_profile: EvalProfile | None = None
    eval_discovery: str = "local"
    eval_allow_llm: bool = True
    eval_task: str | None = None
    eval_num_fewshot: int | None = None
    eval_prompting_strategy: str | None = None
    eval_thinking_mode: str = "auto"
    eval_max_gen_toks: int | None = None
    eval_runtime_drift: list[str] = field(default_factory=list)

    # -- Calibration parameters (decided by Quark's intake, no hard-coded defaults) --
    calib_dataset: str = "pileval"
    num_calib_data: int = 512
    calib_seqlen: int = 512

    # -- Performance optimization control -----------------------------------
    top_kernels: int = 5  # the number of top-N bottleneck kernels TraceLens takes
    bottleneck_mode: str = "differential"
    tracelens_gpu_arch_json: str = ""  # optional explicit TraceLens roofline architecture JSON
    keep_floor: float = 0.01  # the KEEP/REVERT stability floor (1%)
    framework_repo: str = ""  # the framework's source directory (kernel->source mapping)
    kernel_repo: str = ""  # optional dependency/kernel source repo
    workspace_source: str = "auto"
    bench_concurrency: int = 64  # concurrent sequences for the ORCHESTRATOR's
    # throughput benchmark (the final quant/gain number). 64 matches Hyperloom's
    # default CONC and is the realistic production operating point.

    # -- GEAK control ---------------------------------------------------------
    geak_model: str = "claude-opus-4-8"
    geak_direction_budget: int = 3  # kernel_workflow optimization-direction budget.
    geak_timeout_s: float = 5400.0  # per-kernel wall-clock budget: run
    # kernel_workflow to completion within this, else salvage the best result so
    # far (complete-or-salvage, not early-emit).
    # Also used by Landing (atom_adapter/vllm_adapter) as the starting physical
    # GPU index for ROCR_VISIBLE_DEVICES -- a real E2E run on a shared 8-GPU
    # host where GPU0-3 were busy showed a hardcoded range(tp) starting at 0
    # can't target a free non-zero-indexed GPU.
    gpu_id: int = 0
    # LLM models are selected uniformly by config.decision_model() and
    # config.codegen_model(); Spec intentionally has no second model selector.

    # -- Generated at runtime (filled in by Intake/Landing, not user input) ----
    model_arch: str = ""  # architecture family from config.json model_type
    arch_fingerprint: str = ""  # result of compute_arch_fingerprint(config.json)
    session_dir: str = ""  # this run's working directory
    invocation_argv: list[str] = field(default_factory=list)
    server_host: str = DEFAULT_SERVER_HOST
    server_port: int = 8080  # written with the real port when Landing starts
    # the profiling server for PerfOpt.
    runtime: RuntimeContext = field(default_factory=RuntimeContext, compare=False, repr=False)

    def __post_init__(self) -> None:
        from quark.experimental.torch.quant_perf.runtime.backends import extract_kv_cache_dtype

        extract_kv_cache_dtype(self.expanded_vllm_args, self.kv_cache_scheme)
        if self.search_gpu_memory_utilization is not None and not 0 < self.search_gpu_memory_utilization <= 1:
            raise ValueError("search_gpu_memory_utilization must be finite and in (0, 1]")
        if self.performance_mode not in {None, "off", "measure", "optimize"}:
            raise ValueError(f"unsupported performance mode: {self.performance_mode}")
        if self.target_gain is not None and self.target_gain <= 0:
            raise ValueError("target_gain must be greater than zero")
        if self.performance_mode in {"off", "measure"} and self.target_gain is not None:
            raise ValueError("target_gain requires performance_mode='optimize'")
        for phase in ("search", "inference"):
            self._resolve_moe_backend(phase)

    @property
    def effective_performance_mode(self) -> PerformanceMode:
        if self.performance_mode is not None:
            return self.performance_mode
        return "optimize" if self.target_gain is not None else "off"

    @property
    def quant_ckpt_dir(self) -> str:
        """Canonical path for the quantized model checkpoint within this session."""
        return f"{self.session_dir}/quant_ckpt"

    @property
    def framework_source_repo(self) -> str:
        return self.runtime.resolved_framework_repo or self.framework_repo

    @property
    def kernel_source_repo(self) -> str:
        return self.runtime.resolved_kernel_repo or self.kernel_repo

    @property
    def active_framework_repo(self) -> str:
        return self.runtime.framework_worktree or self.framework_source_repo

    @property
    def active_kernel_repo(self) -> str:
        return self.runtime.kernel_worktree or self.kernel_source_repo

    @property
    def can_modify_framework(self) -> bool:
        return self.workspace_source != "readonly" and bool(self.active_framework_repo)

    @property
    def can_modify_kernel(self) -> bool:
        return self.workspace_source != "readonly" and bool(self.active_kernel_repo)

    @property
    def framework_source_kind(self) -> str:
        return self.runtime.framework_source_kind

    @property
    def kernel_source_kind(self) -> str:
        return self.runtime.kernel_source_kind

    @property
    def framework_source_origin(self) -> str:
        return self.runtime.framework_source_origin

    @property
    def kernel_source_origin(self) -> str:
        return self.runtime.kernel_source_origin

    @property
    def runtime_python(self) -> str:
        return self.runtime.runtime_python

    @property
    def runtime_env(self) -> dict[str, str]:
        return self.runtime.runtime_env

    @property
    def runtime_origins(self) -> dict[str, str]:
        return self.runtime.runtime_origins

    @property
    def framework_branch(self) -> str:
        return self.runtime.framework_branch

    @property
    def framework_version(self) -> str:
        return self.runtime.framework_version

    @property
    def kernel_branch(self) -> str:
        return self.runtime.kernel_branch

    @property
    def kernel_version(self) -> str:
        return self.runtime.kernel_version

    @property
    def framework_worktree(self) -> str:
        return self.runtime.framework_worktree

    @property
    def kernel_worktree(self) -> str:
        return self.runtime.kernel_worktree

    @property
    def tp(self) -> int:
        """Parse tensor-parallel-size from vllm_extra_args.

        Handles both pre-split (['--tensor-parallel-size', '2']) and
        shell-quoted single-string ('--tensor-parallel-size 2') forms, plus
        argparse-style '--tensor-parallel-size=2'.
        """
        tokens = self.expanded_vllm_args
        for i, t in enumerate(tokens):
            if t in ("--tensor-parallel-size", "-tp") and i + 1 < len(tokens):
                try:
                    return int(tokens[i + 1])
                except ValueError:
                    pass
            if t.startswith("--tensor-parallel-size="):
                try:
                    return int(t.split("=", 1)[1])
                except ValueError:
                    pass
        return 1

    @property
    def vllm_gpu_memory_utilization(self) -> float:
        """Parse gpu-memory-utilization from vllm_extra_args."""
        tokens = self.expanded_vllm_args
        for i, token in enumerate(tokens):
            if token == "--gpu-memory-utilization" and i + 1 < len(tokens):
                try:
                    return float(tokens[i + 1])
                except ValueError:
                    pass
            if token.startswith("--gpu-memory-utilization="):
                try:
                    return float(token.split("=", 1)[1])
                except ValueError:
                    pass
        return 0.85

    @property
    def vllm_max_num_seqs(self) -> int | None:
        """Preserve an explicit scheduler limit across online and offline runs."""
        value = None
        tokens = self.expanded_vllm_args
        for index, token in enumerate(tokens):
            flag = token.replace("_", "-")
            if flag == "--max-num-seqs":
                value = tokens[index + 1] if index + 1 < len(tokens) else ""
            elif flag.startswith("--max-num-seqs="):
                value = token.split("=", 1)[1]
        if value is None:
            return None
        try:
            count = int(value)
        except ValueError as error:
            raise ValueError("--max-num-seqs must be a positive integer") from error
        if count <= 0:
            raise ValueError("--max-num-seqs must be a positive integer")
        return count

    @property
    def vllm_trust_remote_code(self) -> bool:
        """Honor vLLM's boolean flags, including the last explicit opt-out."""
        for token in reversed(self.expanded_vllm_args):
            flag = token.replace("_", "-")
            if flag == "--trust-remote-code":
                return True
            if flag == "--no-trust-remote-code":
                return False
        return False

    @property
    def effective_search_gsm8k_num_samples(self) -> int:
        """Return the mixed-precision candidate evaluation sample count."""
        if self.search_gsm8k_num_samples is not None:
            return self.search_gsm8k_num_samples
        return self.gsm8k_num_samples

    @property
    def vllm_moe_backend(self) -> str:
        """Return the inference override; an empty value keeps model-aware auto."""
        backend = self.effective_inference_moe_backend
        return "" if backend == "auto" else backend

    @property
    def vllm_kv_cache_dtype(self) -> str:
        """Effective storage dtype, independent of the native/FP8 search candidates."""
        from quark.experimental.torch.quant_perf.runtime.backends import extract_kv_cache_dtype, resolve_kv_cache_dtype

        _, requested = extract_kv_cache_dtype(self.expanded_vllm_args, self.kv_cache_scheme)
        return resolve_kv_cache_dtype(self.base_model, requested)

    def _with_kv_cache_dtype(self, args: list[str]) -> list[str]:
        from quark.experimental.torch.quant_perf.runtime.backends import extract_kv_cache_dtype

        remaining, _ = extract_kv_cache_dtype(args, self.kv_cache_scheme)
        dtype = self.vllm_kv_cache_dtype
        if dtype != "auto":
            remaining.append(f"--kv-cache-dtype={dtype}")
        return remaining

    def _moe_backend_args(self) -> tuple[list[str], str]:
        """Separate the legacy global MoE option from shared vLLM arguments."""
        tokens = self.expanded_vllm_args
        remaining: list[str] = []
        backend = ""
        index = 0
        while index < len(tokens):
            token = tokens[index]
            flag, separator, value = token.partition("=")
            if flag not in {"--moe-backend", "--moe_backend"}:
                remaining.append(token)
                index += 1
                continue
            if not separator:
                index += 1
                value = tokens[index] if index < len(tokens) else ""
            if not value or value.startswith("-"):
                raise ValueError(f"{flag} requires a backend value")
            value = value.lower().replace("-", "_")
            if backend and backend != value:
                raise ValueError("conflicting --moe-backend values in --vllm-extra-arg")
            backend = value
            index += 1
        return remaining, backend

    def _resolve_moe_backend(self, phase: str) -> str:
        _, legacy = self._moe_backend_args()
        requested = self.search_moe_backend if phase == "search" else self.inference_moe_backend
        if requested is not None and legacy and requested != legacy:
            raise ValueError(f"--{phase}-moe-backend conflicts with --moe-backend in --vllm-extra-arg")
        return requested or legacy or "auto"

    @property
    def effective_search_moe_backend(self) -> str:
        return self._resolve_moe_backend("search")

    @property
    def effective_inference_moe_backend(self) -> str:
        return self._resolve_moe_backend("inference")

    @property
    def search_vllm_args(self) -> list[str]:
        tokens, _ = self._moe_backend_args()
        backend = self.effective_search_moe_backend
        if backend != "auto":
            tokens.append(f"--moe-backend={backend}")
        return self._with_kv_cache_dtype(tokens)

    @property
    def inference_vllm_args(self) -> list[str]:
        tokens, _ = self._moe_backend_args()
        if self.vllm_moe_backend:
            tokens.append(f"--moe-backend={self.vllm_moe_backend}")
        return self._with_kv_cache_dtype(tokens)

    @property
    def expanded_vllm_args(self) -> list[str]:
        """Shell-expand repeated --vllm-extra-arg values once."""
        import shlex

        tokens: list[str] = []
        for arg in self.vllm_extra_args or []:
            tokens.extend(shlex.split(arg))
        return tokens

    @property
    def vllm_passthrough_args(self) -> list[str]:
        """Return vLLM args excluding server settings owned by Quant-Perf."""
        tokens = self.inference_vllm_args
        passthrough: list[str] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            if token in ("--tensor-parallel-size", "-tp", "--host", "--port"):
                i += 2
                continue
            if token.startswith(("--tensor-parallel-size=", "--host=", "--port=")):
                i += 1
                continue
            passthrough.append(token)
            i += 1
        return passthrough

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("runtime", None)
        return payload

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        runtime_context: dict[str, Any] | None = None,
    ) -> Spec:
        known = {item.name for item in fields(cls)} - {"runtime"}
        restored = {k: v for k, v in data.items() if k in known}
        if isinstance(restored.get("eval_profile"), dict):
            restored["eval_profile"] = EvalProfile.from_dict(restored["eval_profile"])
        if "performance_mode" not in restored:
            restored["performance_mode"] = "optimize" if restored.get("target_gain") is not None else "off"
        return cls(
            **restored,
            runtime=RuntimeContext.from_dict(runtime_context or data),
        )


class StageError(Exception):
    """Raised by a pipeline stage (quantize/land/throughput/perfopt). The Orchestrator
    converts this into a terminal result so cleanup and FINAL reporting run."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        code: str = "",
        diagnostic: str = "",
    ):
        super().__init__(f"[{stage}] {message}")
        self.stage = stage
        self.message = message
        self.code = code
        self.diagnostic = diagnostic or message


@dataclass
class ServerHandle:
    """Handle for a server process started with ``start_new_session=True``."""

    port: int
    proc: Popen[Any]
    process_group_id: int  # Equals proc.pid for a new session leader.

    def stop(self) -> None:
        # The group ID is captured when the process is spawned. Looking it up
        # here fails when the group leader has exited while workers remain.
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.process_group_id, signal.SIGTERM)
        # Reap the group leader if needed before a caller starts another server
        # on the same port.
        with contextlib.suppress(Exception):
            self.proc.wait(timeout=30)


@dataclass
class AccuracyResult:
    gap: float  # (source - quantized) / source, smaller is better
    source_gsm8k: float
    quantized_gsm8k: float
    passed: bool  # gap <= spec.accuracy_gap
    artifacts: dict[str, str] = field(default_factory=dict)


@dataclass
class PerfResult:
    patches: list[str]  # file paths of KEPT kernel_workflow patches (current_best.diff)
    gain: float  # result of aggregate_gain()
    dominant_bound: str = ""  # most frequent bound type across TraceLens results:
    # COMPUTE_BOUND / MEMORY_BOUND
    patch_srcs: list[str] = field(default_factory=list)
    patch_repos: list[str] = field(default_factory=list)
    runtime_candidates: list[dict[str, Any]] = field(default_factory=list)
    runtime_env: dict[str, str] = field(default_factory=dict)
    runtime_artifacts: dict[str, str] = field(default_factory=dict)


@dataclass
class DeployPackage:
    status: str  # success / accuracy_failed / performance_failed / perf_below_target / base_unhealthy
    quant_ckpt_dir: str = ""
    perf: PerfResult | None = None
    message: str = ""
    applied_patches: list[str] = field(default_factory=list)
    framework_branch: str = ""  # branch in framework_repo containing all changes
    original_branch: str = ""  # original branch before Quark Quant-Perf modified anything
    revert_command: str = ""  # command to undo all framework changes
    kernel_branch: str = ""
    kernel_original_branch: str = ""
    kernel_revert_command: str = ""
    report_status: str = ""
    report_paths: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeployPackage:
        restored = dict(data)
        perf = restored.get("perf")
        if isinstance(perf, dict):
            restored["perf"] = PerfResult(**perf)
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in restored.items() if k in known})


def _fresh_state(spec: Spec) -> SessionState:
    """The initial state.json content for a new run (IMPL_SPEC §4.1)."""
    eval_profile_resolver_version = None
    eval_profile_input_hash = None
    if spec.eval_profile is not None:
        from quark.experimental.torch.quant_perf.evaluation.profile import EVAL_PROFILE_RESOLVER_VERSION
        from quark.experimental.torch.quant_perf.evaluation.profile import (
            eval_profile_input_hash as build_eval_profile_input_hash,
        )

        thinking = {
            "enabled": True,
            "disabled": False,
        }.get(spec.eval_thinking_mode)
        eval_profile_resolver_version = EVAL_PROFILE_RESOLVER_VERSION
        eval_profile_input_hash = build_eval_profile_input_hash(
            spec.base_model,
            discovery=spec.eval_discovery,
            allow_llm=spec.eval_allow_llm,
            overrides={
                "task": spec.eval_task,
                "num_fewshot": spec.eval_num_fewshot,
                "prompting_strategy": spec.eval_prompting_strategy,
                "enable_thinking": thinking,
                "max_gen_toks": spec.eval_max_gen_toks,
            },
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": str(uuid.uuid4()),
        "stage": "quantize",
        "path": "direct_ptq" if spec.quant_strategy else "mix_precision_search",
        # -- fact fields (must be preserved across resume) --
        "quant_ckpt_dir": None,
        "best_candidate": None,
        "best_accuracy_gap": None,
        "mix_precision_search": {
            "schema_version": 1,
            "status": "pending",
            "api": "quark.experimental.torch.mix_precision",
            "config": {},
            "result": {},
            "candidate_queue": [],
            "candidate_cursor": 0,
            "candidate_order_source": ("quark_reverse_evaluation_order"),
        },
        "baseline_reference": None,
        "baseline_runtime_health": None,
        "accuracy_validation": None,
        "post_repair_rechecked_configs": [],
        "geak_patches": [],
        "vendor_gemm_tuning": {},
        "vendor_shape_evidence": {},
        "perfopt_history": [],
        "retained_runtime_env": {},
        "runtime_inventory": {},
        "retention_stack": [],
        "retention_base": {},
        "final_retention": {},
        "performance_runtime": None,
        "performance_status": ("not_requested" if spec.effective_performance_mode == "off" else "pending"),
        "performance_validation": {
            "policy_version": 1,
            "timed_samples": 3,
            "max_timed_samples": 4,
            "stability_relative_mad": 0.015,
            "strong_gain_floor": 0.05,
        },
        "recovery_attempts": [],
        "eval_profile": (spec.eval_profile.to_dict() if spec.eval_profile is not None else None),
        "eval_profile_hash": (spec.eval_profile.profile_hash if spec.eval_profile is not None else None),
        "eval_profile_resolver_version": eval_profile_resolver_version,
        "eval_profile_input_hash": eval_profile_input_hash,
        "eval_profile_history": [],
        "accuracy_attempts": [],
        "performance_measurements": [],
        "phase_timeline": [],
        "bottleneck_analysis": {
            "policy_version": 1,
            "requested_mode": spec.bottleneck_mode,
            "effective_mode": "",
            "status": "pending",
            "reason": "",
            "candidates": [],
            "quantized_trace": "",
            "baseline_trace": "",
            "trace_kind": "",
        },
        "kernel_journey": [],
        "kernel_provenance_artifact": None,
        "kernel_source_resolution_artifact": None,
        "retain_trials": [],
        "repair_journey": [],
        "change_ledger": [],
        "terminal_stage": None,
        "terminal_result": None,
        "reporting": {
            "status": "pending",
            "paths": {},
            "error": "",
        },
        "run_spec": spec.to_dict(),
        "runtime_context": spec.runtime.to_dict(),
        "invocation": {
            "argv": list(spec.invocation_argv),
            "spec": spec.to_dict(),
            "env": {
                key: os.environ[key]
                for key in (
                    "QUARK_QUANT_PERF_MXFP4_MOE_BACKEND",
                    "QUARK_QUANT_PERF_MXFP4_GEMM_BACKEND",
                    "QUARK_QUANT_PERF_W4A8_GEMM_BACKEND",
                    "AITER_CONFIG_FMOE",
                    "AITER_FLYDSL_FORCE",
                    "VLLM_ROCM_USE_AITER",
                    "VLLM_ROCM_USE_AITER_MOE",
                    "VLLM_ROCM_USE_AITER_FLYDSL_MOE",
                    "VLLM_ROCM_USE_AITER_TRITON_FUSED_MOE",
                    "VLLM_ROCM_MXFP4_GEMM_BACKEND",
                    "VLLM_ROCM_W4A8_GEMM_BACKEND",
                )
                if key in os.environ
            },
        },
        "kernel_repo": None,
        "kernel_branch": None,
        "kernel_original_branch": None,
        "repo_workspaces": {},
        "transient_resources": [],
        "cleanup": {
            "status": "pending",
            "removed": [],
            "errors": [],
        },
    }


def _migrate_v8_state(state: SessionState) -> SessionState:
    run_spec = dict(state.get("run_spec") or {})
    invocation_spec = dict((state.get("invocation") or {}).get("spec") or {})
    runtime = RuntimeContext.from_dict(run_spec)
    runtime.resolved_framework_repo = str(run_spec.get("framework_repo") or "")
    runtime.resolved_kernel_repo = str(run_spec.get("kernel_repo") or "")

    migrated_spec = {key: value for key, value in run_spec.items() if key not in _RUNTIME_FIELD_NAMES}
    for repo_field in ("framework_repo", "kernel_repo"):
        if repo_field in invocation_spec:
            migrated_spec[repo_field] = invocation_spec[repo_field]

    state["schema_version"] = SCHEMA_VERSION
    state["run_spec"] = migrated_spec
    state["runtime_context"] = runtime.to_dict()
    return state


class Checkpoint:
    """The run's persistent state, backed by session_dir/state.json.

    Reliability trio (IMPL_SPEC §4.1):
    - atomic write: tmp file + os.replace, never a partially-written file.
    - schema_version + a migration hook for future field changes.
    - a fact/cache field split -- callers should treat cache fields as
      recomputable and never rely on them surviving a resume.
    """

    def __init__(self, session_dir: Path, state: SessionState):
        self.session_dir = session_dir
        self.state = state

    @classmethod
    def fresh(cls, spec: Spec) -> Checkpoint:
        session_dir = Path(spec.session_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        return cls(session_dir, _fresh_state(spec))

    @classmethod
    def load(cls, session_dir: Path) -> Checkpoint | None:
        path = session_dir / "state.json"
        if not path.exists():
            return None
        state = cast(SessionState, json.loads(path.read_text()))
        version = state.get("schema_version")
        if version == 8:
            state = _migrate_v8_state(state)
            version = state.get("schema_version")
        if version != SCHEMA_VERSION:
            raise StageError(
                "checkpoint",
                f"state.json schema_version={version} is unsupported; current schema is {SCHEMA_VERSION}",
                code="unsupported_checkpoint_schema",
            )
        return cls(session_dir, state)

    def save(self) -> None:
        path = self.session_dir / "state.json"
        write_json_atomic(path, self.state)

    def record_phase_event(
        self,
        action: str,
        status: str,
        **details: Any,
    ) -> None:
        event = {
            "ts": datetime.now(UTC).isoformat(),
            "action": action,
            "status": status,
        }
        event.update(key_value for key_value in details.items() if key_value[1] is not None)
        self.state.setdefault("phase_timeline", []).append(event)


class SessionLock:
    """A single-orchestrator flock lock (IMPL_SPEC §4.1): acquired at
    startup, held for the entire run, released by the kernel on crash.
    """

    def __init__(self, session_dir: Path):
        self._path = session_dir / "orchestrator.lock"
        self._fh: TextIO | None = None

    def acquire(self) -> None:
        self._fh = open(self._path, "w")  # noqa: SIM115 - held for the lock lifetime
        try:
            fcntl.flock(self._fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            self._fh.close()
            self._fh = None
            raise StageError(
                "orchestrator",
                f"another Quark Quant-Perf orchestrator already holds the lock at {self._path}",
            ) from e
        self._fh.write(f"pid={os.getpid()} host={os.uname().nodename}\n")
        self._fh.flush()

    def release(self) -> None:
        if self._fh is not None:
            fcntl.flock(self._fh, fcntl.LOCK_UN)
            self._fh.close()
            self._fh = None

    def __enter__(self) -> SessionLock:
        self.acquire()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.release()
