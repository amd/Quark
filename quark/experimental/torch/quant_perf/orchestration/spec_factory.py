#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Construct the immutable run specification from parsed CLI arguments."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from quark.experimental.torch.quant_perf import config
from quark.experimental.torch.quant_perf.session.spec import Spec


def _resolve_performance_options(args: argparse.Namespace) -> tuple[str, float | None]:
    mode = args.performance_mode
    target_gain = args.target_gain

    if target_gain is not None and mode not in {None, "optimize"}:
        raise ValueError("--target-gain requires --performance-mode optimize")
    if args.retry_perfopt and mode in {"off", "measure"}:
        raise ValueError("--retry-perfopt requires --performance-mode optimize")
    if mode is None:
        mode = "optimize" if target_gain is not None or args.retry_perfopt else "off"
    if mode == "optimize" and target_gain is None:
        target_gain = 1.2
    return mode, target_gain


def build_spec_from_args(args: argparse.Namespace) -> Spec:
    gpu_arch = "MI355X" if args.gpu_type == "mi350x" else args.gpu_type.upper()
    session_dir = str(Path(args.session_dir or Path("runs") / time.strftime("%Y%m%d-%H%M%S")).resolve())
    performance_mode, target_gain = _resolve_performance_options(args)
    return Spec(
        model_dir=args.model_dir,
        base_model=args.base_model or args.model_dir,
        framework=args.framework,
        gpu_type=args.gpu_type,
        gpu_arch=gpu_arch,
        isl=args.isl,
        osl=args.osl,
        quant_strategy=args.quant_strategy,
        accuracy_gap=args.accuracy_gap,
        performance_mode=performance_mode,
        target_gain=target_gain,
        kv_cache_scheme=args.kv_cache_scheme,
        exclude_layers=args.exclude_layers,
        file2file_export=getattr(args, "file2file_export", False),
        max_search_candidates=args.max_search_candidates,
        search_timeout_s=args.search_timeout,
        search_gpu_memory_utilization=getattr(args, "search_gpu_memory_utilization", None),
        baseline_floor=args.baseline_floor,
        fix_base_framework=args.fix_base_framework,
        recheck_baseline=args.recheck_baseline,
        retry_accuracy_gate=args.retry_accuracy_gate,
        retry_perfopt=args.retry_perfopt,
        **({"layer_precision_candidates": args.layer_precision_candidates} if args.layer_precision_candidates else {}),
        **(
            {"kv_cache_precision_candidates": args.kv_cache_precision_candidates}
            if args.kv_cache_precision_candidates
            else {}
        ),
        search_gsm8k_num_samples=getattr(args, "search_gsm8k_num_samples", None),
        gsm8k_num_samples=args.gsm8k_num_samples,
        vllm_extra_args=args.vllm_extra_args,
        search_moe_backend=args.search_moe_backend,
        inference_moe_backend=args.inference_moe_backend,
        mxfp4_moe_backend=args.mxfp4_moe_backend,
        mxfp4_gemm_backend=args.mxfp4_gemm_backend,
        w4a8_gemm_backend=args.w4a8_gemm_backend,
        aiter_config_fmoe=args.aiter_config_fmoe,
        eval_discovery=args.eval_discovery,
        eval_allow_llm=not args.eval_no_llm,
        eval_task=args.eval_task,
        eval_num_fewshot=args.eval_num_fewshot,
        eval_prompting_strategy=args.eval_prompting_strategy,
        eval_thinking_mode=args.eval_thinking_mode,
        eval_max_gen_toks=args.eval_max_gen_toks,
        calib_dataset=args.calib_dataset,
        num_calib_data=args.num_calib_data,
        calib_seqlen=args.calib_seqlen,
        top_kernels=args.top_kernels,
        bottleneck_mode=args.bottleneck_mode,
        tracelens_gpu_arch_json=(
            str(Path(args.tracelens_gpu_arch_json).resolve()) if args.tracelens_gpu_arch_json else ""
        ),
        keep_floor=args.keep_floor,
        framework_repo=args.framework_repo,
        kernel_repo=args.kernel_repo,
        workspace_source=args.workspace_source,
        bench_concurrency=args.bench_concurrency,
        geak_direction_budget=args.geak_direction_budget,
        geak_timeout_s=args.geak_timeout,
        geak_model=args.geak_model or config.geak_model(),
        gpu_id=args.gpu_id,
        server_port=args.server_port,
        arch_fingerprint="",
        session_dir=session_dir,
    )
