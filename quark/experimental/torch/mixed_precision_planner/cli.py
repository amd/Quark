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

from quark.experimental.cli.base_cli import BaseQuarkCLICommand
from quark.torch import ModelQuantizer, export_safetensors
from quark.torch.quantization.config.config import QConfig
from quark.torch.utils.llm.model_preparation import get_model, get_tokenizer

from ._serialization import atomic_write_json
from .api import MixedPrecisionPlanner
from .candidates import Candidates
from .data import (
    TokenDataset,
    fingerprint_tokenizer,
    materialize_calibration_tokens,
    materialize_ppl_tokens,
    validate_token_datasets,
)
from .decision_space import DecisionSpace
from .export import ReloadValidationResult, audit_export_binding, verify_export_roundtrip
from .mixed_precision_plan import MixedPrecisionPlan
from .mixed_precision_strategy import EvaluatorBackend, MixedPrecisionStrategy
from .model_source import ResolvedModelSource, resolve_model_source, validate_writable_paths
from .plan_selection import ModelFactory, prepend_model
from .ppl import evaluate_ppl
from .runtime import EvaluationRuntimeContext
from .sensitivity_profile import SensitivityProfile

_STAGES = (
    "decision-space",
    "sensitivity-profile",
    "search",
    "select-best-plan",
    "build-qconfig",
    "run",
)


def _export_selected_plan(
    model_factory: ModelFactory,
    qconfig: QConfig,
    decision_space: DecisionSpace,
    plan: MixedPrecisionPlan,
    calibration_tokens: TokenDataset,
    ppl_tokens: TokenDataset,
    *,
    device: str,
    output_dir: Path,
    trust_remote_code: bool,
) -> ReloadValidationResult:
    model = model_factory()
    try:
        qconfig_audit = audit_export_binding(
            model,
            qconfig,
            decision_space,
            plan,
            calibration_tokens,
            ppl_tokens,
        )
        source_hf_ppl = evaluate_ppl(model, ppl_tokens, device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        quantizer = ModelQuantizer(qconfig)
        model = quantizer.quantize_model(model, calibration_tokens.to_dataloader(device))
        model = quantizer.freeze(model)
        export_safetensors(
            model=model,
            output_dir=output_dir,
            custom_mode="quark",
            weight_format="real_quantized",
            pack_method="reorder",
        )
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atomic_write_json(output_dir / "qconfig.json", qconfig.to_dict())
    atomic_write_json(output_dir / "qconfig_audit.json", qconfig_audit.to_dict())
    return verify_export_roundtrip(
        model_dir=output_dir,
        ppl_tokens=ppl_tokens,
        source_hf_ppl=source_hf_ppl,
        expected_qconfig_hash=qconfig_audit.qconfig_hash,
        max_degradation=plan.payload.quality_gate_max_degradation,
        device=device,
        trust_remote_code=trust_remote_code,
    )


class MixedPrecisionPlannerCLI(BaseQuarkCLICommand):
    """Quark CLI adapter for the experimental mixed-precision planner."""

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("stage", choices=_STAGES)
        parser.add_argument("--strategy", type=Path, required=True)
        parser.add_argument("--model-dir")
        parser.add_argument("--model-revision")
        parser.add_argument("--device", default="cuda")
        parser.add_argument("--trust-remote-code", action="store_true")
        parser.add_argument("--output-dir", type=Path)
        parser.add_argument("--decision-space", type=Path)
        parser.add_argument("--sensitivity-profile", type=Path)
        parser.add_argument("--candidates", type=Path)
        parser.add_argument("--calibration-tokens", type=Path)
        parser.add_argument("--ppl-tokens", type=Path)
        parser.add_argument("--plan", type=Path)
        parser.add_argument("--qconfig", type=Path)
        parser.add_argument("--export-dir", type=Path)
        parser.add_argument("--tensor-parallel-size", type=int, default=1)
        parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
        parser.add_argument("--max-model-len", type=int, default=4096)

    def _require_model_dir(self) -> str:
        model_dir = self.args.model_dir
        if not isinstance(model_dir, str) or not model_dir:
            raise ValueError(f"--model-dir is required for stage {self.args.stage!r}.")
        return model_dir

    def _load_model(self) -> nn.Module:
        model, _ = get_model(
            ckpt_path=self._resolve_model_source().resolved_path,
            data_type="auto",
            device=self.args.device,
            multi_gpu=False,
            multi_device=False,
            attn_implementation="eager",
            trust_remote_code=self.args.trust_remote_code,
        )
        return model

    def _resolve_model_source(self) -> ResolvedModelSource:
        resolved = getattr(self, "_resolved_source", None)
        if resolved is None:
            resolved = resolve_model_source(self._require_model_dir(), self.args.model_revision)
            self._resolved_source = resolved
        return resolved

    def run(self) -> None:
        strategy = MixedPrecisionStrategy.load(self.args.strategy)
        planner = MixedPrecisionPlanner(strategy)
        model_source = self._resolve_model_source() if self.args.stage != "search" else None
        output_dir = self.args.output_dir or Path(strategy.output.dir)
        paths = {
            "decision_space": self.args.decision_space or output_dir / "decision_space.json",
            "sensitivity_profile": self.args.sensitivity_profile or output_dir / "sensitivity_profile.json",
            "candidates": self.args.candidates or output_dir / "candidates.json",
            "calibration_tokens": self.args.calibration_tokens or output_dir / "calibration_tokens.json",
            "ppl_tokens": self.args.ppl_tokens or output_dir / "ppl_tokens.json",
            "plan": self.args.plan or output_dir / "mixed_precision_plan.json",
            "qconfig": self.args.qconfig or output_dir / "qconfig.json",
        }
        writable_keys = {
            "decision-space": ("decision_space",),
            "sensitivity-profile": ("sensitivity_profile",),
            "search": ("candidates",),
            "select-best-plan": ("calibration_tokens", "ppl_tokens", "plan"),
            "build-qconfig": ("qconfig",),
            "run": tuple(paths),
        }
        if model_source is not None:
            writable_paths = [output_dir, *(paths[key] for key in writable_keys[self.args.stage])]
            if self.args.stage == "run":
                writable_paths.append(self.args.export_dir or output_dir / "exported_model")
            validate_writable_paths(model_source, writable_paths)
        output_dir.mkdir(parents=True, exist_ok=True)

        if self.args.stage == "decision-space":
            planner.decision_space(self._load_model()).save(paths["decision_space"])
            return
        if self.args.stage == "sensitivity-profile":
            space = DecisionSpace.load(paths["decision_space"])
            planner.sensitivity_profile(self._load_model(), space).save(paths["sensitivity_profile"])
            return
        if self.args.stage == "search":
            space = DecisionSpace.load(paths["decision_space"])
            profile = SensitivityProfile.load(paths["sensitivity_profile"])
            planner.search(space, profile).save(paths["candidates"])
            return

        assert model_source is not None
        model_dir = model_source.resolved_path
        run_model = None
        if self.args.stage == "run":
            run_model = self._load_model()
            space = planner.decision_space(run_model)
            space.save(paths["decision_space"])
        else:
            space = DecisionSpace.load(paths["decision_space"])
        if self.args.stage == "build-qconfig":
            plan = MixedPrecisionPlan.load(paths["plan"])
            qconfig = planner.build_qconfig(self._load_model(), space, plan)
            atomic_write_json(paths["qconfig"], qconfig.to_dict())
            return
        tokenizer = get_tokenizer(
            model_dir,
            model_type=space.payload.model_type,
            trust_remote_code=self.args.trust_remote_code,
        )

        if paths["calibration_tokens"].exists():
            calibration_tokens = TokenDataset.load(paths["calibration_tokens"])
        else:
            calibration_tokens = materialize_calibration_tokens(tokenizer, model_source.requested, strategy)
            calibration_tokens.save(paths["calibration_tokens"])
        if paths["ppl_tokens"].exists():
            ppl_tokens = TokenDataset.load(paths["ppl_tokens"])
        else:
            ppl_tokens = materialize_ppl_tokens(tokenizer, model_source.requested, strategy)
            ppl_tokens.save(paths["ppl_tokens"])
        current_tokenizer_fingerprint = fingerprint_tokenizer(tokenizer)
        validate_token_datasets(
            calibration_tokens,
            ppl_tokens,
            strategy,
            tokenizer_fingerprint=current_tokenizer_fingerprint,
        )

        model_factory = self._load_model
        if self.args.stage == "select-best-plan":
            candidates = Candidates.load(paths["candidates"])
            runtime = EvaluationRuntimeContext(
                model_source=model_dir,
                device=self.args.device,
                model_revision=model_source.commit_hash,
                trust_remote_code=self.args.trust_remote_code,
                tensor_parallel_size=self.args.tensor_parallel_size,
                gpu_memory_utilization=self.args.gpu_memory_utilization,
                max_model_len=self.args.max_model_len,
                model_factory=(
                    model_factory if strategy.plan_selection.evaluator.backend is EvaluatorBackend.HF else None
                ),
            )
            planner.select_best_plan_with_runtime(
                runtime,
                space,
                candidates,
                calibration_tokens,
                ppl_tokens,
            ).save(paths["plan"])
            return
        assert run_model is not None
        profile = planner.sensitivity_profile(run_model, space)
        profile.save(paths["sensitivity_profile"])
        candidates = planner.search(space, profile)
        candidates.save(paths["candidates"])
        selection_model_factory = (
            prepend_model(run_model, model_factory)
            if strategy.plan_selection.evaluator.backend is EvaluatorBackend.HF
            else None
        )
        run_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        runtime = EvaluationRuntimeContext(
            model_source=model_dir,
            device=self.args.device,
            model_revision=model_source.commit_hash,
            trust_remote_code=self.args.trust_remote_code,
            tensor_parallel_size=self.args.tensor_parallel_size,
            gpu_memory_utilization=self.args.gpu_memory_utilization,
            max_model_len=self.args.max_model_len,
            model_factory=selection_model_factory,
        )
        plan = planner.select_best_plan_with_runtime(
            runtime,
            space,
            candidates,
            calibration_tokens,
            ppl_tokens,
        )
        plan.save(paths["plan"])
        qconfig = planner.build_qconfig_from_plan(space, plan)
        atomic_write_json(paths["qconfig"], qconfig.to_dict())
        _export_selected_plan(
            model_factory,
            qconfig,
            space,
            plan,
            calibration_tokens,
            ppl_tokens,
            device=self.args.device,
            output_dir=self.args.export_dir or output_dir / "exported_model",
            trust_remote_code=self.args.trust_remote_code,
        )


__all__ = ["MixedPrecisionPlannerCLI"]
