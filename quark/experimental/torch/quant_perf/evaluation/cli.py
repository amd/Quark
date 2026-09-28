#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""CLI entrypoint for managed accuracy-only evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from quark.experimental.torch.quant_perf import config
from quark.experimental.torch.quant_perf.evaluation.service import EvaluationService
from quark.experimental.torch.quant_perf.runtime.backends import configure_runtime_env
from quark.experimental.torch.quant_perf.session.spec import RuntimeContext, Spec
from quark.experimental.torch.quant_perf.workspace.git import get_head_sha
from quark.experimental.torch.quant_perf.workspace.sources import (
    activate_runtime,
    verify_runtime_origins,
)


def add_eval_profile_args(
    parser: argparse.ArgumentParser,
    *,
    preserve_omitted: bool = False,
) -> None:
    parser.add_argument(
        "--eval-discovery",
        choices=["local", "online", "off"],
        default=None if preserve_omitted else "local",
        help="real accuracy profile discovery: local model metadata/card "
        "(default), bounded official online sources, or disabled",
    )
    parser.add_argument(
        "--eval-no-llm",
        action="store_true",
        default=None if preserve_omitted else False,
        help="disable constrained LLM extraction from unstructured eval docs",
    )
    parser.add_argument("--eval-task", default=None)
    parser.add_argument("--eval-num-fewshot", type=int, default=None)
    parser.add_argument(
        "--eval-prompting-strategy",
        choices=["cot", "direct", "task_default"],
        default=None,
    )
    parser.add_argument(
        "--eval-thinking-mode",
        choices=["auto", "enabled", "disabled"],
        default=None if preserve_omitted else "auto",
    )
    parser.add_argument("--eval-max-gen-toks", type=int, default=None)


def build_eval_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="quark-quant-perf eval",
        description=(
            "Run managed baseline and quantized real accuracy evaluation "
            "without quantization, benchmarking, or PerfOpt."
        ),
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", dest="model_dir")
    source.add_argument("--from-session")
    parser.add_argument("--base-model")
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--framework", choices=["vllm", "atom"], default=None)
    parser.add_argument(
        "--gpu-type",
        choices=["mi300x", "mi325x", "mi350x", "mi355x"],
        default=None,
    )
    parser.add_argument("--gpu-id", type=int, default=None)
    parser.add_argument("--framework-repo", default=None)
    parser.add_argument("--kernel-repo", default=None)
    parser.add_argument("--gsm8k-num-samples", type=int, default=None)
    parser.add_argument("--accuracy-gap", type=float, default=None)
    parser.add_argument(
        "--vllm-extra-arg",
        dest="vllm_extra_args",
        action="append",
        default=None,
    )
    parser.add_argument(
        "--inference-moe-backend",
        choices=["auto", "triton", "aiter", "aiter_mxfp4_bf16", "emulation"],
        default=None,
        help="MoE backend for real evaluation (default: session setting or model-aware auto)",
    )
    parser.add_argument(
        "--mxfp4-moe-backend",
        choices=["aiter", "triton", "flydsl"],
        default=None,
    )
    parser.add_argument(
        "--mxfp4-gemm-backend",
        choices=["triton", "flydsl", "asm"],
        default=None,
    )
    parser.add_argument(
        "--w4a8-gemm-backend",
        choices=["triton", "flydsl"],
        default=None,
    )
    add_eval_profile_args(parser, preserve_omitted=True)
    parser.add_argument("--allow-runtime-drift", action="store_true")
    return parser


def runtime_drift_errors(
    source_session: str | Path,
    current_heads: dict[str, str],
) -> list[str]:
    report = Path(source_session) / "reports" / "final.json"
    if not report.is_file():
        return []
    try:
        repositories = json.loads(report.read_text()).get("repositories") or {}
    except (OSError, json.JSONDecodeError):
        return []
    errors: list[str] = []
    for role in ("framework", "kernel"):
        expected = str((repositories.get(role) or {}).get("source_head") or "")
        current = str(current_heads.get(role) or "")
        if expected and current and expected != current:
            errors.append(f"{role} source HEAD changed: {expected} -> {current}")
    return errors


def current_eval_source_heads(spec: Spec) -> dict[str, str]:
    heads: dict[str, str] = {}
    for role, path in (
        ("framework", spec.active_framework_repo),
        ("kernel", spec.active_kernel_repo),
    ):
        if path and Path(path).is_dir():
            try:
                heads[role] = get_head_sha(path)
            except Exception:
                heads[role] = ""
    return heads


def eval_spec_from_args(args: argparse.Namespace) -> tuple[Spec, str]:
    from quark.experimental.torch.quant_perf.orchestration import intake
    from quark.experimental.torch.quant_perf.session.spec import Checkpoint

    if args.from_session:
        source_dir = Path(args.from_session).resolve()
        checkpoint = Checkpoint.load(source_dir)
        if checkpoint is None:
            raise SystemExit(f"no state.json found under {source_dir}")
        run_spec = checkpoint.state.get("run_spec")
        if not isinstance(run_spec, dict):
            raise SystemExit(f"session {source_dir} has no persisted run_spec")
        spec = Spec.from_dict(
            run_spec,
            runtime_context=dict(checkpoint.state.get("runtime_context") or {}),
        )
        quant_model = str(checkpoint.state.get("quant_ckpt_dir") or "")
        if not quant_model or not Path(quant_model).is_dir():
            raise SystemExit(f"session {source_dir} has no usable quantized checkpoint")
    else:
        if not args.base_model:
            raise SystemExit("--base-model is required with --model")
        base_config = intake.load_model_config(args.base_model)
        raw_text_config = base_config.get("text_config")
        text_config = raw_text_config if isinstance(raw_text_config, dict) else {}
        gpu_type = args.gpu_type or "mi355x"
        spec = Spec(
            model_dir=args.model_dir,
            base_model=args.base_model,
            framework=args.framework or "vllm",
            gpu_type=gpu_type,
            gpu_arch=("MI355X" if gpu_type == "mi350x" else gpu_type.upper()),
            isl=0,
            osl=0,
            quant_strategy=None,
            accuracy_gap=(0.02 if args.accuracy_gap is None else args.accuracy_gap),
            session_dir=str(Path(args.session_dir).resolve()),
            arch_fingerprint=intake.compute_arch_fingerprint(base_config),
            model_arch=str(base_config.get("model_type") or text_config.get("model_type") or "unknown"),
        )
        quant_model = args.model_dir

    source_framework_repo = spec.framework_source_repo
    source_kernel_repo = spec.kernel_source_repo
    updates: dict[str, Any] = {
        "model_dir": quant_model,
        "session_dir": str(Path(args.session_dir).resolve()),
        "workspace_source": "readonly",
    }
    if args.framework is not None:
        updates["framework"] = args.framework
    if args.gpu_type is not None:
        updates["gpu_type"] = args.gpu_type
        updates["gpu_arch"] = "MI355X" if args.gpu_type == "mi350x" else args.gpu_type.upper()
    if args.gpu_id is not None:
        updates["gpu_id"] = args.gpu_id
    if args.framework_repo is not None:
        updates["framework_repo"] = args.framework_repo
    elif args.from_session:
        updates["framework_repo"] = source_framework_repo
    if args.kernel_repo is not None:
        updates["kernel_repo"] = args.kernel_repo
    elif args.from_session:
        updates["kernel_repo"] = source_kernel_repo
    if args.gsm8k_num_samples is not None:
        updates["gsm8k_num_samples"] = args.gsm8k_num_samples
    if args.accuracy_gap is not None:
        updates["accuracy_gap"] = args.accuracy_gap
    if args.vllm_extra_args is not None:
        updates["vllm_extra_args"] = list(args.vllm_extra_args)
    if args.inference_moe_backend is not None:
        updates["inference_moe_backend"] = args.inference_moe_backend
    if args.mxfp4_moe_backend is not None:
        updates["mxfp4_moe_backend"] = args.mxfp4_moe_backend
    if args.mxfp4_gemm_backend is not None:
        updates["mxfp4_gemm_backend"] = args.mxfp4_gemm_backend
    if args.w4a8_gemm_backend is not None:
        updates["w4a8_gemm_backend"] = args.w4a8_gemm_backend
    profile_overridden = any(
        value is not None
        for value in (
            args.eval_discovery,
            args.eval_no_llm,
            args.eval_task,
            args.eval_num_fewshot,
            args.eval_prompting_strategy,
            args.eval_thinking_mode,
            args.eval_max_gen_toks,
        )
    )
    if not args.from_session or profile_overridden:
        updates["eval_profile"] = None
    if args.eval_discovery is not None:
        updates["eval_discovery"] = args.eval_discovery
    if args.eval_no_llm is not None:
        updates["eval_allow_llm"] = not args.eval_no_llm
    if args.eval_task is not None:
        updates["eval_task"] = args.eval_task
    if args.eval_num_fewshot is not None:
        updates["eval_num_fewshot"] = args.eval_num_fewshot
    if args.eval_prompting_strategy is not None:
        updates["eval_prompting_strategy"] = args.eval_prompting_strategy
    if args.eval_thinking_mode is not None:
        updates["eval_thinking_mode"] = args.eval_thinking_mode
    if args.eval_max_gen_toks is not None:
        updates["eval_max_gen_toks"] = args.eval_max_gen_toks
    spec = replace(
        spec,
        **updates,
        runtime=RuntimeContext(runtime_python=sys.executable),
    )
    return spec, quant_model


def run_eval_command(
    argv: list[str],
    *,
    preflight_framework_imports: Callable[[str], None],
) -> int:
    config.load_env()
    args = build_eval_parser().parse_args(argv)
    spec, quant_model = eval_spec_from_args(args)
    if args.from_session:
        drift = runtime_drift_errors(
            args.from_session,
            current_eval_source_heads(spec),
        )
        if drift and not args.allow_runtime_drift:
            raise SystemExit(
                "runtime source drift detected; pass --allow-runtime-drift to continue:\n- " + "\n- ".join(drift)
            )
        spec = replace(spec, eval_runtime_drift=drift)
    Path(spec.session_dir).mkdir(parents=True, exist_ok=True)
    configure_runtime_env(spec)
    runtime = activate_runtime(spec)
    runtime_origin_evidence = verify_runtime_origins(runtime)
    if spec.active_framework_repo:
        preflight_framework_imports(spec.active_framework_repo)
    result = EvaluationService().run(
        spec,
        quant_model,
        runtime_origin_evidence=runtime_origin_evidence,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "success" else 1
