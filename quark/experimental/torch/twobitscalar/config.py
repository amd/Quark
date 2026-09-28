#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from dataclasses import dataclass

from quark.common.config import BaseAlgoConfig


@dataclass
class TwoBitScalarConfig(BaseAlgoConfig):
    """
    TwoBitScalar config used by TwoBitScalarProcessor.

    Inherits ``BaseAlgoConfig`` (the shared base that ``AlgoConfig`` also derives
    from), so ``from_dict`` is the strict ``cls(**data)`` that raises on unknown
    keys instead of silently dropping them. (``BaseAlgoConfig`` is used rather than
    the torch ``AlgoConfig`` to avoid a circular import via the algorithm API.)

    NOTE:
    - This implementation is OFFLINE-only (no runtime wrappers).
    - It applies transforms for quantization, then maps back so model forward is unchanged.
    """

    # Key used in PROCESSOR_MAP and CLI: --quant_algo twobitscalar
    name: str = "twobitscalar"

    # Weight-only low-bit settings
    bits: int = 2  # TwoBitScalar is a 4-level (2-bit) quantizer; only bits=2 is supported
    group_size: int = 64
    # Sub-group quantization: apply per-row scale at a finer granularity than the
    # SRHT rotation group.  E.g. group_size=32 with sub_group_size=4 gives the
    # SRHT decorrelation of g=32 but the quantization precision of g=4.
    # 0 means same as group_size (backward compat).
    sub_group_size: int = 0

    # Offline AWQ-like scaling (used only during quantization, then undone)
    act_scale_alpha: float = 0.5
    scale_clip_min: float = 0.25
    scale_clip_max: float = 4.0

    # Calibration control
    calib_batches: int | None = None  # None => use full loader
    max_samples_per_layer: int = 2048  # cap stored rows X per layer
    max_rows_per_batch: int = 256  # cap rows taken from each batch
    min_in_features: int = 16

    # Include / exclude (wildcards supported)
    include_layers: list[str] | None = None
    exclude_layers: list[str] | None = None

    # Incoherence rotation (SRHT within each group)
    enable_incoherence: bool = False
    incoherence_seed: int = 1234
    num_hadamard_passes: int = 1  # multi-pass Hadamard; 1 is optimal when sub_g divides block
    enable_output_rotation: bool = False  # also rotate output (row) dimension for full incoherence
    use_lloyd_max_levels: bool = False  # Gaussian-optimal inner levels (0.30 vs 0.33)
    use_hessian_weighted_scale: bool = False  # weight MSE by Hessian diagonal in scale search
    fine_scale_grid: bool = False  # 41-point scale grid [0.50..1.0] instead of 9-point [0.6..1.0]
    adaptive_grid_levels: bool = False  # 2D grid search over (scale, inner_level) per row

    # Blockwise Hessian whitening per group (offline) — legacy, weak
    enable_blockwise_hessian: bool = False
    hess_damp: float = 0.1
    hess_max_rows: int = 2048

    # NEW: speed knob (apply Hessian only every N groups; 1 = every group)
    hess_every_n_groups: int = 1

    # NEW: safety knob (skip Hessian if group size is too small/too unstable)
    # For g=4, Hessian can easily hurt unless heavily damped; default: only allow if g>=8
    hess_min_group_size: int = 8  # cap X rows used to build covariance

    # Quantization behavior
    do_quantize: bool = True
    preserve_weight_dtype: bool = True  # keep bf16/fp16 in weight tensor
    verbose: bool = False

    # Debugging / instrumentation
    debug_dump_per_layer: bool = False  # prints per-layer RMS error summaries

    # Dump per-layer SRHT/AWQ sidecar parameters to this directory during quantization.
    # When set, writes {awq_scale, srht_perm, srht_signs}.safetensors (each a dict
    # keyed by layer name). These reconstruct the input-side transform
    #   x_rot = SRHT(x / awq_scale)
    # used by downstream "factored" / shared-rotation exports, so the rotation does
    # not need to be undone into the weights. Requires enable_incoherence=True and
    # num_hadamard_passes=1 (single-pass rotation has one perm/sign per layer).
    dump_sidecar_dir: str | None = None

    def __post_init__(self) -> None:
        # `bits` is a fixed property of the algorithm (4 levels), not a tunable
        # knob — the quantizer always emits 2-bit. Fail loudly rather than
        # silently ignoring e.g. bits=4.
        if self.bits != 2:
            raise ValueError(f"TwoBitScalar only supports bits=2 (4 levels), got bits={self.bits}.")
