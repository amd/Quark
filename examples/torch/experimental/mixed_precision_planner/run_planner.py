#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import torch
from torch import nn

from quark.experimental.torch.mixed_precision_planner import (
    EvaluationRuntimeContext,
    MixedPrecisionPlanner,
    MixedPrecisionStrategy,
    audit_export_binding,
    evaluate_ppl,
    materialize_calibration_tokens,
    materialize_ppl_tokens,
    resolve_model_source,
    validate_writable_paths,
    verify_export_roundtrip,
)
from quark.experimental.torch.mixed_precision_planner._serialization import atomic_write_json
from quark.experimental.torch.mixed_precision_planner.mixed_precision_strategy import EvaluatorBackend
from quark.experimental.torch.mixed_precision_planner.plan_selection import prepend_model
from quark.torch import ModelQuantizer, export_safetensors
from quark.torch.utils.llm.model_preparation import get_model, get_tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Quark Torch mixed-precision planner with Qwen3.")
    parser.add_argument("--model-dir", default="Qwen/Qwen3-0.6B-Base")
    parser.add_argument("--model-revision")
    parser.add_argument("--strategy", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-model-len", type=int, default=4096)
    return parser.parse_args()


def _clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    strategy = MixedPrecisionStrategy.load(args.strategy)
    model_source = resolve_model_source(args.model_dir, args.model_revision)
    output_dir = args.output_dir or Path(strategy.output.dir)
    validate_writable_paths(model_source, (output_dir,))
    output_dir.mkdir(parents=True, exist_ok=True)
    planner = MixedPrecisionPlanner(strategy)

    def model_factory() -> nn.Module:
        model, _ = get_model(
            ckpt_path=model_source.resolved_path,
            data_type="auto",
            device=args.device,
            multi_gpu=False,
            multi_device=False,
            attn_implementation="eager",
            trust_remote_code=args.trust_remote_code,
        )
        return model

    # Profiling and search: build the Decision Space, score sensitivity, and generate Top-K candidates.
    profile_model = model_factory()
    try:
        decision_space = planner.decision_space(profile_model)
        sensitivity_profile = planner.sensitivity_profile(profile_model, decision_space)
        candidates = planner.search(decision_space, sensitivity_profile)

        # Freeze calibration and evaluation inputs before comparing candidates.
        tokenizer = get_tokenizer(
            model_source.resolved_path,
            model_type=decision_space.payload.model_type,
            trust_remote_code=args.trust_remote_code,
        )
        calibration_tokens = materialize_calibration_tokens(tokenizer, model_source.requested, strategy)
        ppl_tokens = materialize_ppl_tokens(tokenizer, model_source.requested, strategy)
        selection_model_factory = (
            prepend_model(profile_model, model_factory)
            if strategy.plan_selection.evaluator.backend is EvaluatorBackend.HF
            else None
        )
    finally:
        del profile_model
        _clear_memory()

    # Plan selection: evaluate the baseline and candidates, then choose the best valid assignment.
    runtime = EvaluationRuntimeContext(
        model_source=model_source.resolved_path,
        device=args.device,
        model_revision=model_source.commit_hash,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        model_factory=selection_model_factory,
    )
    plan = planner.select_best_plan_with_runtime(
        runtime,
        decision_space,
        candidates,
        calibration_tokens,
        ppl_tokens,
    )

    # Rebuild the already-audited selected assignment without loading another model.
    qconfig = planner.build_qconfig_from_plan(decision_space, plan)

    # Persist outputs in workflow order.
    decision_space.save(output_dir / "decision_space.json")
    sensitivity_profile.save(output_dir / "sensitivity_profile.json")
    candidates.save(output_dir / "candidates.json")
    calibration_tokens.save(output_dir / "calibration_tokens.json")
    ppl_tokens.save(output_dir / "ppl_tokens.json")
    plan.save(output_dir / "mixed_precision_plan.json")
    atomic_write_json(output_dir / "qconfig.json", qconfig.to_dict())

    # Quantize and export the selected assignment with a fresh source model.
    if args.export:
        export_dir = output_dir / "exported_model"
        export_model = model_factory()
        try:
            qconfig_audit = audit_export_binding(
                export_model,
                qconfig,
                decision_space,
                plan,
                calibration_tokens,
                ppl_tokens,
            )
            source_hf_ppl = evaluate_ppl(export_model, ppl_tokens, args.device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            quantizer = ModelQuantizer(qconfig)
            export_model = quantizer.quantize_model(
                export_model,
                calibration_tokens.to_dataloader(args.device),
            )
            export_model = quantizer.freeze(export_model)
            export_safetensors(
                model=export_model,
                output_dir=export_dir,
                custom_mode="quark",
                weight_format="real_quantized",
                pack_method="reorder",
            )
        finally:
            del export_model
            _clear_memory()

        atomic_write_json(export_dir / "qconfig.json", qconfig.to_dict())
        atomic_write_json(export_dir / "qconfig_audit.json", qconfig_audit.to_dict())
        # Verify the serialized model by reloading it in a fresh process.
        verify_export_roundtrip(
            model_dir=export_dir,
            ppl_tokens=ppl_tokens,
            source_hf_ppl=source_hf_ppl,
            expected_qconfig_hash=qconfig_audit.qconfig_hash,
            max_degradation=plan.payload.quality_gate_max_degradation,
            device=args.device,
            trust_remote_code=args.trust_remote_code,
        )


if __name__ == "__main__":
    main()
