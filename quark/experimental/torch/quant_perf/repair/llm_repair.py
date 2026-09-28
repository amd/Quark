#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""LLM-backed runtime repair strategies.

Three entry points:
- attempt_load_repair(): model crashes on load/inference — patch until it loads.
- classify_accuracy_failure() + attempt_accuracy_repair(): model loads fine but
  GSM8K accuracy is very poor — classify root cause (framework vs quantization
  quality), then patch the framework's quantization handling if needed.
- attempt_benchmark_repair(): a framework/kernel integration defect prevents
  the immutable throughput workload from completing.

Design ref: IMPL_SPEC §4.3.3.

Guarantees:
- Operates in an Quark Quant-Perf-owned candidate or integration worktree.
- Quark Quant-Perf independently verifies each round; agent self-report is not trusted.
- Each round feeds the new error/context back to the agent.
- Failed generations are discarded. After a verifier rejection, the next round
  starts from the candidate that produced that failure, preserving prerequisite
  changes. Unsuccessful repair restores the per-invocation baseline, including
  any base fixes Quark Quant-Perf already applied.
- On success, changes are committed to the working branch.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable
from typing import Any

from quark.experimental.torch.quant_perf import config
from quark.experimental.torch.quant_perf.llm.audit import append_llm_call
from quark.experimental.torch.quant_perf.llm.client import direct_api_call
from quark.experimental.torch.quant_perf.repair.agent_loop import (
    parse_summary as _parse_summary,
)
from quark.experimental.torch.quant_perf.repair.agent_loop import (
    run_agent_rounds as _run_agent_rounds,
)
from quark.experimental.torch.quant_perf.repair.evidence import extract_failure_evidence, render_failure_evidence
from quark.experimental.torch.quant_perf.repair.types import FailureEvidence, RepairKnowledgeQuery
from quark.experimental.torch.quant_perf.repair.verifiers import (
    read_quant_config as _read_config,
)
from quark.experimental.torch.quant_perf.repair.verifiers import (
    verify_load_and_inference as _verify_load_and_inference,
)
from quark.experimental.torch.quant_perf.session.spec import EvalProfile
from quark.experimental.torch.quant_perf.workspace.git import get_current_branch

logger = logging.getLogger(__name__)

# A GSM8K gap this large is not "proportional quantization loss" -- it means the
# quantized weights/scales aren't being applied at all (silent fp16 fallback,
# wrong scale load, broken dispatch), i.e. a framework bug. Above it, skip the
# LLM classifier and go straight to runtime repair.
_LARGE_GAP_FRAMEWORK_THRESHOLD = 0.5
_LOAD_REPAIR_TIMEOUT_S = 1800
_LOAD_REPAIR_AGENT_BUDGET_S = 5400

Verifier = Callable[[], tuple[bool, str | FailureEvidence]]
_MANAGED_REPAIR_DIFF_LIMIT = 12000


def _recent_managed_repair_context(
    framework_repo: str,
    *,
    limit: int = _MANAGED_REPAIR_DIFF_LIMIT,
) -> str:
    """Return the latest Quant-Perf repair diff that may explain a regression."""
    history = subprocess.run(
        ["git", "log", "--all", "--format=%H%x00%s", "-n", "12"],
        cwd=framework_repo,
        capture_output=True,
        text=True,
    )
    if history.returncode != 0:
        return "(no recent managed repair commit found)"
    repair_sha = ""
    for line in history.stdout.splitlines():
        sha, separator, subject = line.partition("\x00")
        if separator and subject.startswith("Quark Quant-Perf ") and subject.endswith(" repair"):
            repair_sha = sha
            break
    if not repair_sha:
        return "(no recent managed repair commit found)"
    repair = subprocess.run(
        [
            "git",
            "show",
            "--format=fuller",
            "--stat",
            "--patch",
            "--no-ext-diff",
            "--no-renames",
            repair_sha,
        ],
        cwd=framework_repo,
        capture_output=True,
        text=True,
    )
    if repair.returncode != 0 or not repair.stdout.strip():
        return f"(managed repair commit {repair_sha} could not be summarized)"
    return repair.stdout[:limit]


_SKILL_PROMPT = """\
You are fixing a quantized LLM model loading/inference failure in a managed
framework or kernel source repository.

Framework: {framework}
Target role: {role}
Target repo: {framework_repo}
Quantized checkpoint: {quant_ckpt_dir}
Tensor parallel size: {tp}
ROCR visible devices: {visible_devices}
Quantized checkpoint config.json:
{quant_config_json}

Current failure (authoritative, from round {round_num}):
{error}

Prior repair context (advisory only; never replace the current failure):
{context}

Quark Quant-Perf runs the authoritative TP={tp} load/inference verifier after you
return. Do not run the long multi-GPU reproduction inside the code-generation
budget. You may run focused unit tests that do not load the full model.

RULES:
1. Change only code owned by the target role. Framework repairs may change
   glue, config, quantization registration, or dispatch. Kernel repairs may
   change the failing backend implementation or dispatch, but must preserve
   the requested numerical semantics and immutable workload.
2. Read every file before editing it.
3. Keep the patch focused on the current root cause. Historical attempts and
   retrieved knowledge are hints, not substitutes for current evidence.
4. Common fixes:
   - "quant_method=quark not found" → find the quantization method registry
     (usually a dict or if/elif chain) and add a "quark" entry pointing to the
     Quark config class
   - kernel dispatch error (wrong dtype/format) → find the op dispatch table and
     add the missing case
   - ModuleNotFoundError → fix the import path or add a shim
   - shape/dtype mismatch on weight load → check weight loading assumptions
5. When the patch and focused tests are ready, output this JSON as the very last line:
   {{"changed_files": ["rel/path/a.py"], "summary": "what you fixed", "success": true}}
   If you cannot fix it, output:
   {{"changed_files": [], "summary": "reason", "success": false}}
"""

_ACCURACY_FIX_PROMPT = """\
A quantized LLM loads and runs without error, but GSM8K accuracy is terrible:
baseline={source_gsm8k:.4f}, quantized={quantized_gsm8k:.4f} (gap={gap:.1%}).
The framework is not correctly applying the quantized weights or scales.

Framework: {framework}
Framework repo: {framework_repo}
Quantized checkpoint: {quant_ckpt_dir}
Quantization config:
{quant_config_json}

Recent managed repair context:
{recent_managed_repair_context}

Round {round_num} — previous context:
{context}

Investigation order:
1. Inspect the recent managed repair first. A severe accuracy regression after
   a load fix is most likely caused by that source delta.
2. Then inspect the model-specific weight loader and the framework's
   quantization parameter loader for the affected fused modules.
3. Do not scan large checkpoint weight shards. Use config and tensor metadata
   only unless concrete evidence requires reading tensor values.

Common root causes:
1. Scale tensors are loaded but not applied in forward (dequant skipped)
2. Framework silently falls back to FP16 for unrecognized quantization scheme
3. Input/weight scales have wrong shape or are all-ones
4. KV cache dtype mismatch (written FP16, read as FP8 or vice versa)
5. Weight loading swaps input_scale / weight_scale
6. A load-time helper reconstructs derived weights by running a quantized
   linear on synthetic inputs such as identity matrices. Static activation
   scales describe real model inputs, so this can clip the synthetic input and
   silently corrupt the derived weight. Prefer an optional framework-level
   hook that can directly dequantize stored weights, with the identity-matrix
   path retained only as a fallback for methods that do not provide one.

RULES:
1. Only change framework Python code. Never modify weight files or model math.
2. Read every file before editing it.
3. Output as the very last line:
   {{"changed_files": ["rel/path.py"], "summary": "what you changed", "success": true}}
   or {{"changed_files": [], "summary": "reason", "success": false}}
"""

_PERFORMANCE_FIX_PROMPT = """\
You are fixing a framework or kernel integration bug that prevents a required
Quark Quant-Perf throughput benchmark from running.

Framework: {framework}
Framework repo: {framework_repo}
Quantized checkpoint: {quant_ckpt_dir}
Failure role: {role}
Immutable workload: TP={tp}, ISL={isl}, OSL={osl}, concurrency={concurrency}

Current error (round {round_num}):
{error}

The workload, model, quantization configuration, TP, ISL, OSL, concurrency,
accuracy threshold, and benchmark implementation must not change. Only fix the
framework/kernel integration defect responsible for the failure.

After every edit, run the supplied failing workload as closely as possible.
Quark Quant-Perf will independently rerun the accuracy gate and the complete
baseline/quantized throughput pair before accepting any patch.

Output as the final line:
{{"changed_files": ["rel/path.py"], "summary": "what you fixed", "success": true}}
or
{{"changed_files": [], "summary": "reason", "success": false}}
"""


def attempt_load_repair(
    error: str | FailureEvidence,
    framework: str,
    framework_repo: str,
    quant_ckpt_dir: str,
    role: str = "framework",
    gpu_id: int = 0,
    tp: int = 1,
    arch_fingerprint: str = "",
    framework_version: str = "",
    quant_signature: str = "",
    session_dir: str = "",
    verify: Verifier | None = None,
    knowledge_text: str = "",
    knowledge_ids: list[str] | None = None,
    runtime_python: str = "",
    runtime_env: dict[str, str] | None = None,
    attempts_out: list[dict[str, Any]] | None = None,
    phase_callback: Callable[[dict[str, Any]], None] | None = None,
    knowledge_for_round: RepairKnowledgeQuery | None = None,
    kv_cache_dtype: str | None = None,
) -> bool:
    """Attempt to fix a framework or kernel load/inference failure.

    Operates on the framework_repo worktree supplied by the orchestrator.
    Returns True only when Quark Quant-Perf's own verification passes.

    Retrieved experience is supplied through ``knowledge_text``.
    """
    config_json = _read_config(quant_ckpt_dir)
    logger.info("[RuntimeRepair] working on branch %s", get_current_branch(framework_repo))

    verify_callback = verify or (
        lambda: _verify_load_and_inference(
            quant_ckpt_dir,
            gpu_id,
            tp,
            runtime_python=runtime_python,
            runtime_env=runtime_env,
            kv_cache_dtype=kv_cache_dtype,
        )
    )

    initial_evidence = extract_failure_evidence(error)

    def make_prompt(r: int, ctx: str, evidence: FailureEvidence | None = None) -> str:
        visible = ",".join(str(gpu_id + i) for i in range(tp))
        prompt = _SKILL_PROMPT.format(
            framework=framework,
            framework_repo=framework_repo,
            role=role,
            quant_ckpt_dir=quant_ckpt_dir,
            quant_config_json=config_json,
            error=render_failure_evidence(evidence or initial_evidence),
            context=(ctx or "none")[:5000],
            round_num=r,
            tp=tp,
            visible_devices=visible,
        )
        return prompt

    result = _run_agent_rounds(
        framework_repo,
        make_prompt,
        verify_callback,
        commit_msg="Quark Quant-Perf runtime repair",
        label="RuntimeRepair",
        timeout_s=_LOAD_REPAIR_TIMEOUT_S,
        agent_round_start_budget_s=_LOAD_REPAIR_AGENT_BUDGET_S,
        initial_context="",
        session_dir=session_dir,
        knowledge_ids=list(knowledge_ids or []),
        knowledge_text=knowledge_text,
        knowledge_for_round=knowledge_for_round,
        initial_evidence=initial_evidence,
        phase_callback=phase_callback,
    )
    _publish_attempts(result, attempts_out)

    return result["success"]


def attempt_benchmark_repair(
    *,
    error: str | FailureEvidence,
    framework: str,
    framework_repo: str,
    quant_ckpt_dir: str,
    tp: int,
    isl: int,
    osl: int,
    concurrency: int,
    verify: Verifier,
    role: str = "",
    arch_fingerprint: str = "",
    framework_version: str = "",
    runtime_fingerprint: str = "",
    session_dir: str = "",
    knowledge_text: str = "",
    knowledge_ids: list[str] | None = None,
    attempts_out: list[dict[str, Any]] | None = None,
    phase_callback: Callable[[dict[str, Any]], None] | None = None,
    knowledge_for_round: RepairKnowledgeQuery | None = None,
) -> bool:
    """Patch a code-classified throughput failure and use caller verification."""

    initial_evidence = extract_failure_evidence(error)

    def make_prompt(round_num: int, context: str, evidence: FailureEvidence | None = None) -> str:
        prompt = _PERFORMANCE_FIX_PROMPT.format(
            framework=framework,
            framework_repo=framework_repo,
            quant_ckpt_dir=quant_ckpt_dir,
            role=role or "unknown",
            tp=tp,
            isl=isl,
            osl=osl,
            concurrency=concurrency,
            round_num=round_num,
            error=render_failure_evidence(evidence or initial_evidence) + "\n" + context[:5000],
        )
        return prompt

    result = _run_agent_rounds(
        framework_repo,
        make_prompt,
        verify,
        commit_msg="Quark Quant-Perf benchmark execution repair",
        label="BenchmarkExecutionRepair",
        session_dir=session_dir,
        knowledge_ids=list(knowledge_ids or []),
        knowledge_text=knowledge_text,
        knowledge_for_round=knowledge_for_round,
        initial_evidence=initial_evidence,
        phase_callback=phase_callback,
    )
    _publish_attempts(result, attempts_out)
    return result["success"]


def classify_accuracy_failure(
    source_gsm8k: float,
    quantized_gsm8k: float,
    gap: float,
    quant_ckpt_dir: str,
    quant_strategy: str,
    session_dir: str = "",
) -> str:
    """Classify whether poor accuracy is a framework bug or quantization quality issue.

    Direct claude CLI call, no tools — inputs are bounded, task is classification only.
    Returns 'framework' or 'quantization'. Defaults to 'quantization' on any error.
    """
    if gap >= _LARGE_GAP_FRAMEWORK_THRESHOLD:
        logger.info(
            "[FailureTriage] gap %.3f >= %.2f — treating as framework bug "
            "(skipping LLM classify), repair will be attempted",
            gap,
            _LARGE_GAP_FRAMEWORK_THRESHOLD,
        )
        return "framework"
    prompt = (
        f"Quantized LLM loads fine but GSM8K dropped: baseline={source_gsm8k:.4f}, "
        f"quantized={quantized_gsm8k:.4f} (gap {gap:.1%}). "
        f"Strategy: {quant_strategy or 'auto'}. Config: {_read_config(quant_ckpt_dir, 2000)}\n\n"
        f"Is this a FRAMEWORK bug (wrong scale loading, silent FP16 fallback, bad dispatch — "
        f"typically >30% gap for standard FP8/INT8) or QUANTIZATION QUALITY (proportional loss, "
        f"aggressive scheme, small calibration dataset)?\n"
        f'Reply with one JSON line: {{"root_cause":"framework"|"quantization","confidence":0.0-1.0}}'
    )
    output = direct_api_call(
        tag="accuracy_failure_classification",
        system=("Classify one quantized-model accuracy regression. Return one JSON object only."),
        user=prompt,
        model=config.decision_model(),
        max_tokens=256,
        default="",
    )
    if session_dir:
        append_llm_call(
            session_dir,
            call_type="accuracy_failure_classification",
            model=config.decision_model(),
            round_id=0,
            prompt=prompt,
            output=output,
            outcome="returned" if output else "fallback",
        )
    try:
        data = _parse_summary(output)
        if data.get("root_cause") == "framework" and float(data.get("confidence", 0)) >= 0.65:
            logger.info("[FailureTriage] framework bug (confidence=%.2f)", data["confidence"])
            return "framework"
    except Exception as e:
        logger.warning("[FailureTriage] failed: %s", e)
    return "quantization"


def attempt_accuracy_repair(
    source_gsm8k: float,
    quantized_gsm8k: float,
    gap: float,
    framework: str,
    framework_repo: str,
    quant_ckpt_dir: str,
    gpu_id: int,
    tp: int = 1,
    arch_fingerprint: str = "",
    framework_version: str = "",
    quant_signature: str = "",
    eval_profile: EvalProfile | None = None,
    gpu_memory_utilization: float = 0.85,
    session_dir: str = "",
    knowledge_text: str = "",
    knowledge_ids: list[str] | None = None,
    runtime_python: str = "",
    runtime_env: dict[str, str] | None = None,
    attempts_out: list[dict[str, Any]] | None = None,
    verify_callback: Verifier | None = None,
    phase_callback: Callable[[dict[str, Any]], None] | None = None,
    knowledge_for_round: RepairKnowledgeQuery | None = None,
    trust_remote_code: bool = False,
    max_num_seqs: int | None = None,
    kv_cache_dtype: str | None = None,
) -> bool:
    """Attempt to fix poor accuracy caused by a framework integration bug.

    Verification uses a 50-sample GSM8K subset; the orchestrator re-runs
    the full eval after this returns True. Gap must at least halve per round.

    Runtime experience is supplied through ``knowledge_text`` and never
    suppresses current verification.
    """
    from quark.experimental.torch.quant_perf.evaluation.gsm8k import gsm8k_eval_offline

    config_json = _read_config(quant_ckpt_dir)
    recent_managed_repair_context = _recent_managed_repair_context(framework_repo)
    threshold = gap * 0.5

    def verify_accuracy() -> tuple[bool, str | FailureEvidence]:
        if verify_callback is not None:
            return verify_callback()
        try:
            score = gsm8k_eval_offline(
                quant_ckpt_dir,
                gpu_id,
                num_questions=50,
                tp=tp,
                gpu_memory_utilization=gpu_memory_utilization,
                profile=eval_profile,
                trust_remote_code=trust_remote_code,
                max_num_seqs=max_num_seqs,
                runtime_python=runtime_python,
                runtime_env=runtime_env,
                kv_cache_dtype=kv_cache_dtype,
            )
            new_gap = max(0.0, (source_gsm8k - score) / max(source_gsm8k, 1e-9))
            logger.info("[AccuracyRepair] new_gap=%.4f threshold=%.4f", new_gap, threshold)
            if new_gap <= threshold:
                return True, ""
            return False, f"gap {gap:.4f}→{new_gap:.4f}, fix incomplete"
        except Exception as e:
            return False, extract_failure_evidence(e)

    initial_evidence = extract_failure_evidence(
        f"accuracy gap baseline={source_gsm8k:.4f} quantized={quantized_gsm8k:.4f} gap={gap:.4f}"
    )

    def make_prompt(r: int, ctx: str, evidence: FailureEvidence | None = None) -> str:
        prompt = _ACCURACY_FIX_PROMPT.format(
            source_gsm8k=source_gsm8k,
            quantized_gsm8k=quantized_gsm8k,
            gap=gap,
            framework=framework,
            framework_repo=framework_repo,
            quant_ckpt_dir=quant_ckpt_dir,
            quant_config_json=config_json,
            recent_managed_repair_context=recent_managed_repair_context,
            round_num=r,
            context=render_failure_evidence(evidence or initial_evidence) + "\n" + ctx,
        )
        return prompt

    result = _run_agent_rounds(
        framework_repo,
        make_prompt,
        verify_accuracy,
        commit_msg="Quark Quant-Perf accuracy repair",
        # A framework accuracy fix means investigating a large framework repo,
        # editing, and re-running load/inference -- 600s/round starved the agent
        # (every round hit the timeout with zero edits; the attempts trail showed
        # "agent round timed out" x4, which would otherwise read as "unfixable").
        # 30 min/round, 2 h total gives a round enough time to actually land a fix.
        label="AccuracyRepair",
        timeout_s=1800,
        agent_round_start_budget_s=7200,
        initial_context="",
        session_dir=session_dir,
        knowledge_ids=list(knowledge_ids or []),
        knowledge_text=knowledge_text,
        knowledge_for_round=knowledge_for_round,
        initial_evidence=initial_evidence,
        phase_callback=phase_callback,
    )
    _publish_attempts(result, attempts_out)

    return result["success"]


def _publish_attempts(
    result: dict[str, Any],
    attempts_out: list[dict[str, Any]] | None,
) -> None:
    if attempts_out is None:
        return
    attempts_out.extend(
        {
            "kind": "llm_repair",
            **dict(attempt),
        }
        for attempt in result.get("attempts", [])
    )
