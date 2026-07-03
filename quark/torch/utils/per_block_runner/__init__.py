#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
per_block_runner — memory-efficient per-block forward execution for large PyTorch LLMs.

Two complementary strategies are provided:

* ``runner`` (safetensors-based): streams weights from a sharded checkpoint on demand.
  Use when host RAM is not large enough to hold the full model on CPU.
* ``lazy_loader`` (to-hook-based): hooks ``_apply`` so ``model.to("cuda")`` keeps
  block weights on CPU, then moves each block to GPU only during its forward pass.
  Use when weights are already loaded on CPU (the normal ``from_pretrained`` path).

Safetensors-based API::

    from quark.torch.utils.per_block_runner import prepare, finalize

Lazy-loader API::

    from quark.torch.utils.per_block_runner.lazy_loader import prepare, finalize
"""

from .runner import (
    finalize,
    get_module_weight_by_name,
    prepare,
)
from .utils import infer_decoder_layers_path

__all__ = [
    "prepare",
    "finalize",
    "get_module_weight_by_name",
    "infer_decoder_layers_path",
]
