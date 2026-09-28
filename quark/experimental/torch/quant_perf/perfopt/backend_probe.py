#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch

from quark.experimental.torch.quant_perf.evaluation.throughput import ThroughputMeasurement
from quark.experimental.torch.quant_perf.perfopt.backend_probe_inputs import BackendProbeInputs
from quark.experimental.torch.quant_perf.pipeline.performance_policy import (
    RetestDisposition,
    abba_gain,
    decide_confirmed_retest,
    effective_keep_floor,
)


@dataclass(frozen=True)
class BackendMicroCase:
    name: str
    flydsl_us: float
    asm_us: float
    correct: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class MicroScreenDecision:
    passed: bool
    speedup: float
    reason: str


@dataclass(frozen=True)
class BackendProbeDecision:
    selected_backend: str
    multiplier: float
    effective_floor: float
    reason: str


def basic_output_valid(
    flydsl_output: torch.Tensor,
    asm_output: torch.Tensor,
) -> bool:
    return bool(
        flydsl_output.shape == asm_output.shape
        and torch.isfinite(flydsl_output).all()
        and torch.isfinite(asm_output).all()
    )


def _prepare_asm_weight(
    weight: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    from aiter.ops.shuffle import shuffle_weight  # type: ignore[import-untyped]

    shuffled_weight = shuffle_weight(weight, layout=(16, 16))
    sm, sn = scale.shape
    shuffled_scale = (
        scale.view(sm // 32, 2, 16, sn // 8, 2, 4, 1).permute(0, 3, 5, 2, 4, 1, 6).contiguous().view(sm, sn)
    )
    return shuffled_weight, shuffled_scale


def run_micro_probe(
    inputs: BackendProbeInputs,
) -> list[BackendMicroCase]:
    import aiter  # type: ignore[import-untyped]
    from aiter.test_common import run_perftest  # type: ignore[import-untyped]

    if not (
        hasattr(aiter, "gemm_a4w4")
        and hasattr(aiter, "per_1x32_f4_quant_hip")
        and hasattr(aiter, "flydsl_gemm_a4w4_dynamic")
        and hasattr(aiter, "prepare_flydsl_gemm_mxfp4_weight")
    ):
        raise RuntimeError("required ASM/FlyDSL A4W4 APIs are unavailable")

    weights = inputs.projections
    cases: list[BackendMicroCase] = []
    torch.manual_seed(1234)
    for projection in ("gate_up", "down"):
        raw_weight = weights[projection]["weight"].cuda()
        raw_scale = weights[projection]["scale"].cuda()
        n, packed_k = raw_weight.shape
        logical_k = packed_k * 2
        fly_weight, fly_scale = aiter.prepare_flydsl_gemm_mxfp4_weight(
            raw_weight,
            raw_scale,
        )
        asm_weight, asm_scale = _prepare_asm_weight(
            raw_weight,
            raw_scale,
        )
        for m in (32, 64):
            x = torch.randn(
                (m, logical_k),
                dtype=torch.bfloat16,
                device="cuda",
            )

            def _flydsl(
                x: torch.Tensor = x,
                fly_weight: torch.Tensor = fly_weight,
                fly_scale: torch.Tensor = fly_scale,
                n: int = n,
            ) -> torch.Tensor:
                return aiter.flydsl_gemm_a4w4_dynamic(
                    x,
                    fly_weight,
                    fly_scale,
                    n,
                    torch.bfloat16,
                )

            def _asm(
                x: torch.Tensor = x,
                asm_weight: torch.Tensor = asm_weight,
                asm_scale: torch.Tensor = asm_scale,
            ) -> torch.Tensor:
                x_q, x_s = aiter.per_1x32_f4_quant_hip(
                    x,
                    shuffle=True,
                )
                return aiter.gemm_a4w4(
                    x_q,
                    asm_weight.view(x_q.dtype),
                    x_s,
                    asm_scale.view(x_s.dtype),
                    dtype=torch.bfloat16,
                    bpreshuffle=True,
                )

            fly_out = _flydsl()
            asm_out = _asm()
            correct = basic_output_valid(fly_out, asm_out)
            _, flydsl_us = run_perftest(
                _flydsl,
                num_iters=101,
                num_warmup=5,
                use_cuda_event=True,
            )
            _, asm_us = run_perftest(
                _asm,
                num_iters=101,
                num_warmup=5,
                use_cuda_event=True,
            )
            cases.append(
                BackendMicroCase(
                    f"{projection}_m{m}",
                    float(flydsl_us),
                    float(asm_us),
                    correct,
                )
            )
    return cases


def decide_micro_screen(
    cases: list[BackendMicroCase],
    *,
    min_gain: float = 0.01,
    max_shape_regression: float = 0.05,
) -> MicroScreenDecision:
    if not cases or not all(case.correct for case in cases):
        return MicroScreenDecision(False, 0.0, "correctness_failed")
    if any(case.asm_us > case.flydsl_us * (1.0 + max_shape_regression) for case in cases):
        return MicroScreenDecision(False, 0.0, "shape_regression")
    flydsl_total = sum(case.flydsl_us for case in cases)
    asm_total = sum(case.asm_us for case in cases)
    speedup = flydsl_total / asm_total if asm_total > 0 else 0.0
    if speedup < 1.0 + min_gain:
        return MicroScreenDecision(False, speedup, "insufficient_micro_gain")
    return MicroScreenDecision(True, speedup, "micro_gain")


def decide_probe_backend(
    *,
    micro_passed: bool,
    accuracy_passed: bool,
    anchor_first: ThroughputMeasurement,
    candidate_first: ThroughputMeasurement,
    candidate_second: ThroughputMeasurement,
    anchor_second: ThroughputMeasurement,
    keep_floor: float,
) -> BackendProbeDecision:
    if not micro_passed:
        return BackendProbeDecision("flydsl", 1.0, keep_floor, "micro_screen_failed")
    if not accuracy_passed:
        return BackendProbeDecision("flydsl", 1.0, keep_floor, "accuracy_failed")
    measurements = (
        anchor_first,
        candidate_first,
        candidate_second,
        anchor_second,
    )
    if not all(measurement.stable for measurement in measurements):
        return BackendProbeDecision("flydsl", 1.0, keep_floor, "unstable_measurement")
    floor = effective_keep_floor(keep_floor, *measurements)
    multiplier = abba_gain(
        anchor_first.median_tps,
        candidate_first.median_tps,
        candidate_second.median_tps,
        anchor_second.median_tps,
    )
    disposition = decide_confirmed_retest(multiplier, floor)
    if disposition is RetestDisposition.KEEP:
        return BackendProbeDecision("asm", multiplier, floor, "confirmed_e2e_gain")
    return BackendProbeDecision(
        "flydsl",
        multiplier,
        floor,
        ("confirmed_no_gain" if disposition is RetestDisposition.DROP_CONFIRMED else "within_measurement_noise"),
    )
