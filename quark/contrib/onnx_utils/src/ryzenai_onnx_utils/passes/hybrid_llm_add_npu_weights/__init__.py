# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
These passes add packed NPU weights to MatMulNBits in standalone custom ops, or
as part of other custom ops like SSMLP or GQO. Depending on the attributes,
these passes may remove the GPU weights, keeping only the packed NPU weights e.g.
for NPU only modes.
"""

import ryzenai_onnx_utils.passes

PATTERN, REPLACEMENT, __all__ = ryzenai_onnx_utils.passes.pass_loader(__file__)
