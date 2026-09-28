#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""direct_ptq: user-specified PTQ, delegated to Quark's own quark-torch-ptq
workflow skill in YOLO mode.

Design ref: IMPL_SPEC §2.1.1. Quark Quant-Perf does not reimplement quantization
logic here -- it drives the skill via claude_agent_sdk and judges the result
by Quark Quant-Perf's own artifact criterion (§4.6), independent of the skill's
return format.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

from quark.experimental.torch.quant_perf import config
from quark.experimental.torch.quant_perf.knowledge import (
    KnowledgeContext,
    KnowledgeRenderer,
    build_knowledge_router,
)
from quark.experimental.torch.quant_perf.knowledge.audit import append_query_audit
from quark.experimental.torch.quant_perf.quantize.artifacts import validate_quant_checkpoint
from quark.experimental.torch.quant_perf.session.spec import Spec, StageError
from quark.experimental.torch.quant_perf.session.state import SessionState

if TYPE_CHECKING:
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore


def check_quant_artifacts(quant_dir: str) -> dict[str, Any]:
    """Quark Quant-Perf's own artifact judgement (IMPL_SPEC §4.6): does not trust the
    skill's self-reported status, only what actually landed on disk."""
    return validate_quant_checkpoint(quant_dir)


def _extract_text(msg: Any) -> str:
    text = ""
    for block in getattr(msg, "content", None) or []:
        block_text = getattr(block, "text", None)
        if block_text:
            text += block_text
    return text


def _load_agent_sdk() -> Any:
    """Load the Agent SDK required by the direct PTQ workflow.

    :return: Imported Agent SDK module.
    :raises StageError: If the Quant-Perf optional dependencies are missing.
    """
    try:
        import claude_agent_sdk as sdk
    except ImportError as error:
        raise StageError(
            "quantize",
            "Direct PTQ requires the Agent SDK from the Quant-Perf optional dependencies. "
            "Install it with `pip install 'amd-quark[quant_perf]'`.",
            code="missing_agent_sdk",
            diagnostic=f"{type(error).__name__}: {error}",
        ) from error
    return sdk


def _build_ptq_prompt(
    spec: Spec,
    session_dir: str,
    *,
    knowledge_text: str = "",
    quark_root: str = "",
) -> str:
    # Prompt structure mirrors Hyperloom's quantization-agent pattern
    # (driver/runner.py:build_attempt_prompt): structured "## Run context" block
    # followed by the verbatim user intent as a separate "## Quantization intent"
    # block.  This separation is intentional -- it lets Quark's quark-torch-ptq
    # skill see the raw user request without metadata noise, while still having
    # all the run-time facts it needs (output dir, eval gap, interactive mode).
    #
    # quant_strategy is treated as a free-form intent string normalised by the
    # Quark Quant-Perf request-intake skill before reaching here; it may be anything
    # from "fp8" to
    # "use FP8 for self_attn, MXFP4 for moe experts and mlp, exclude lm_head".
    # Anything not covered (algorithm, per-layer patterns) is left to Quark's
    # skill through the free-form intent -- Quark Quant-Perf never invents defaults.
    ctx_lines = [
        f"- Source model (READ-ONLY, never modify): {spec.model_dir}",
        f"- Output directory (write quantized model here): {spec.quant_ckpt_dir}",
        "- Quark execution workspace (session-owned; source checkout remains "
        f"unchanged): {quark_root or config.quark_root()}",
        f"- Acceptable eval gap: {spec.accuracy_gap:.4f} (max relative GSM8K accuracy drop)",
        "- Interactive mode: off (YOLO — pick sensible defaults, never pause to ask)",
        f"- Calibration dataset: {spec.calib_dataset}",
        f"- Number of calibration samples: {spec.num_calib_data} (use exactly this, do NOT use 512 as default)",
        f"- Calibration sequence length: {spec.calib_seqlen}",
    ]
    if spec.kv_cache_scheme:
        ctx_lines.append(f"- KV cache scheme: {spec.kv_cache_scheme}")
    if spec.exclude_layers:
        ctx_lines.append(f"- Exclude layers: {spec.exclude_layers}")

    # Attach pre-computed plan artifacts if they exist (e.g., from a prior interrupted run).
    # The skill can then reuse them and jump directly to Step 3/4 (Manifest → Execute).
    plan_dir = Path(session_dir)
    if (plan_dir / "model_analysis.json").exists():
        ctx_lines.append(
            f"- Pre-computed model analysis (REUSE THIS — skip Step 1 intake): {plan_dir / 'model_analysis.json'}"
        )
    if (plan_dir / "quant_plan.json").exists():
        ctx_lines.append(
            f"- Pre-computed quant plan (REUSE THIS — skip Step 2 planning, adapt "
            f"output_dir to the Output directory above): {plan_dir / 'quant_plan.json'}"
        )

    knowledge_block = ""
    if knowledge_text:
        knowledge_block = (
            "\n\n## Advisory knowledge\n"
            "Use this only to form candidates and checks; Quark Quant-Perf verifiers "
            "remain authoritative.\n"
            f"{knowledge_text}"
        )
    quant_strategy = spec.quant_strategy
    if quant_strategy is None:
        raise ValueError("direct PTQ requires quant_strategy")
    return (
        "Quantize the model with Quark in YOLO mode.\n\n"
        "## Run context\n"
        + "\n".join(ctx_lines)
        + "\n\n## Quantization intent (verbatim from user)\n"
        + quant_strategy
        + knowledge_block
        + "\n\nBegin now. YOLO rules:\n"
        "1. Run ALL workflow steps (Intake → Plan → Manifest → Execute) without stopping.\n"
        "2. NEVER pause for user confirmation at any checkpoint — proceed automatically.\n"
        "3. If pre-computed analysis/plan files are listed in Run context above, READ and "
        "REUSE them; skip the corresponding steps and jump ahead.\n"
        "4. The final step MUST execute the quantize_quark.py command. Do not stop until "
        ".safetensors files are written to the Output directory.\n"
        "5. Do not launch quantization in the background, detach it, or use a "
        "background wait-loop. Execute quantize_quark.py directly and wait for "
        "its foreground process to exit.\n"
        "6. After the command exits, verify the Output directory contains "
        "config.json and .safetensors files before reporting completion.\n"
        "7. Summarize the chosen config and final outcome after execution completes."
    )


async def run_ptq(
    spec: Spec,
    session_dir: str,
    *,
    experience_store: ExperienceStore | None = None,
    state: SessionState | None = None,
    quark_root: str = "",
) -> dict[str, Any]:
    """direct_ptq: drives the quark-torch-ptq workflow via
    claude_agent_sdk in YOLO mode, per the official prompt template at
    Quark/examples/agent_skills/prompts/torch_llm_ptq.md.

    Returns {status: success|failed, quantized_model_dir, agent_summary}.
    """
    sdk = _load_agent_sdk()

    context = KnowledgeContext(
        domain="quantization",
        stage="direct_ptq",
        model_arch=spec.model_arch,
        arch_fingerprint=spec.arch_fingerprint,
        framework=spec.framework,
        gpu_type=spec.gpu_type,
        quant_signature=spec.quant_strategy or "",
        workload={"tp": spec.tp, "isl": spec.isl, "osl": spec.osl},
    )
    bundle = build_knowledge_router(
        experience_store=experience_store,
        state=state,
    ).query(context)
    knowledge_text = KnowledgeRenderer(max_chars=6000).render(bundle)
    append_query_audit(
        session_dir,
        consumer="direct_ptq",
        context_hash=hashlib.sha256(repr(context).encode()).hexdigest(),
        bundle=bundle,
    )
    prompt = _build_ptq_prompt(
        spec,
        session_dir,
        knowledge_text=knowledge_text,
        quark_root=quark_root,
    )
    execution_root = quark_root or config.quark_root()
    options = sdk.ClaudeAgentOptions(
        cwd=execution_root,
        skills=["quark-torch-ptq"],  # explicit directory-based discovery
        allowed_tools=["Read", "Write", "Bash"],
        env=config.build_subprocess_env(),  # AMD gateway auth for the spawned `claude` CLI
    )

    summary_text = ""
    async for msg in sdk.query(prompt=prompt, options=options):
        summary_text += _extract_text(msg)  # accumulate the trailing summary for failure diagnosis

    result = check_quant_artifacts(spec.quant_ckpt_dir)
    result["agent_summary"] = summary_text[-2000:]  # kept truncated, used to locate the failing stage
    from quark.experimental.torch.quant_perf.llm.audit import append_llm_call

    append_llm_call(
        session_dir,
        call_type="direct_ptq",
        model="quark-torch-ptq-skill",
        round_id=0,
        prompt=prompt,
        output=summary_text,
        outcome=result["status"],
        knowledge_ids=bundle.source_ids,
    )
    return result
