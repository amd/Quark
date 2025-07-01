#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch

from quark.torch.kernel.hw_emulation.extensions import kernel_ext


def dq_mxfp4_hip(x: torch.Tensor, scale: torch.Tensor, float_dtype: torch.dtype) -> torch.Tensor:
    dequant_weight_shape = (*x.shape[:-1], x.shape[-1] * 2)

    dq_w = torch.empty(dequant_weight_shape, device=x.device, dtype=float_dtype)
    kernel_ext.dq_uint8_mxfp4_to_half(x, scale, dq_w, 32)

    return dq_w


def qdq_mxfp4_hip(x: torch.Tensor, scale_calculation_mode: str = "even") -> torch.Tensor:
    if scale_calculation_mode != "even":
        raise NotImplementedError(f"MXScaleCalculationMode {scale_calculation_mode} is not supported on HIP")
    return kernel_ext.qdq_mxfp4(x, 32)  # type: ignore[no-any-return]
