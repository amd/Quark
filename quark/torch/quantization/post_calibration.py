#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
from tqdm import tqdm

from quark.common.utils.log import ScreenLogger
from quark.torch.quantization.config.type import Dtype
from quark.torch.quantization.nn.modules.mixin import QuantMixin
from quark.torch.quantization.tensor_quantize import SequentialQuantize

logger = ScreenLogger(__name__)


def realign_first_stage_block_scales_after_calibration(model: torch.nn.Module) -> None:
    """Realign block scales for SequentialQuantize paths after calibration.

    For FP4+FP8-style sequential quantization, compute the effective tensor-stage
    scale using a single float32 division and cache the packed per-block scale
    for export. This makes calibrated scales independent from subsequent freeze
    behavior while keeping the effective scale in float32.
    """
    realigned_quantizer_count = 0
    quant_modules = [module for _module_name, module in model.named_modules() if isinstance(module, QuantMixin)]

    # The per-block scale tensors handled here are small (a few MB). PyTorch's
    # default intra-op parallelism opens one wide parallel region per tiny op, and
    # at high thread counts the thread launch/sync overhead dominates: each
    # quantizer costs ~80 ms at 32 threads vs <1 ms at a handful of threads. Cap
    # the intra-op thread count for this loop and restore it afterwards. This is a
    # pure performance change -- the computation (and its results) are unchanged.
    saved_num_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(min(8, saved_num_threads))
        for module in tqdm(
            quant_modules,
            desc="Realigning block scales",
            total=len(quant_modules),
        ):
            weight_quantizer = module._weight_quantizer
            if isinstance(weight_quantizer, SequentialQuantize):
                if realign_single_sequential_quantizer(weight_quantizer):
                    realigned_quantizer_count += 1
    finally:
        torch.set_num_threads(saved_num_threads)

    if realigned_quantizer_count > 0:
        logger.info(
            "Realigned first-stage block scales for %d SequentialQuantize modules after calibration.",
            realigned_quantizer_count,
        )


def realign_single_sequential_quantizer(sequential_quantizer: SequentialQuantize) -> bool:
    """Realign an FP4 tensor stage with its adjacent FP8 scale stage."""
    has_realign_action = False
    for quantizer_index, quantizer in enumerate(sequential_quantizer):
        if quantizer.is_scale_quant:
            continue

        if quantizer_index >= len(sequential_quantizer) - 1:
            continue

        next_quantizer = sequential_quantizer[quantizer_index + 1]
        if not next_quantizer.is_scale_quant:
            continue

        is_fp8_scale_quantizer = next_quantizer.quant_spec.dtype in (Dtype.fp8_e4m3, Dtype.fp8_e5m2, Dtype.fp8_e5m3)
        if not is_fp8_scale_quantizer:
            continue

        if not (
            hasattr(quantizer, "observer")
            and hasattr(quantizer.observer, "amax")
            and hasattr(next_quantizer, "observer")
            and getattr(next_quantizer.observer, "quant_max_first_level", None) is not None
        ):
            continue

        element_format_max = next_quantizer.observer.quant_max_first_level
        tensor_stage_amax = quantizer.observer.amax.detach().float().cpu()
        global_scale = next_quantizer.scale.cpu()

        # Match export-time combined-division semantics in float32:
        # per_block_scale = amax / (global_scale * element_format_max)
        scale_dtype = next_quantizer.quant_spec.dtype.to_torch_packed_dtype()
        tensor_stage_amax.div_(global_scale * element_format_max)
        quantized_per_block_scale = tensor_stage_amax.to(scale_dtype)
        # Protect zeros created by FP8 underflow after cast.
        quantized_per_block_scale = quantized_per_block_scale.to(torch.float32)
        quantized_per_block_scale[quantized_per_block_scale == 0] = 1.0
        quantized_per_block_scale = quantized_per_block_scale.to(scale_dtype)

        quantizer_scale_device = quantizer.scale.device
        quantizer.scale = (quantized_per_block_scale.to(torch.float32) * global_scale).to(quantizer_scale_device)
        quantizer._quantized_block_scale = quantized_per_block_scale.detach()
        has_realign_action = True

        # Release temporary tensors as soon as persistent buffers are updated.
        # Reclaim device memory per quantizer to keep the peak low on huge
        # multi-GPU MoE models (avoids the cache pool holding freed blocks).
        del tensor_stage_amax
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return has_realign_action
