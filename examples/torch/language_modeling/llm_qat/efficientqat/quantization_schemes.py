#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch.nn as nn

from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import Int2PerGroupSpec, QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver

INT4_QUANT_SCHEMES = [
    "w_uint4_asym",
    "w_int4_sym",
    "w_int4_asym",
]
INT2_QUANT_SCHEMES = [
    "w_int2_asym",
]

SUPPORTED_QUANT_SCHEMES = [
    *INT4_QUANT_SCHEMES,
    *INT2_QUANT_SCHEMES,
]


def _build_int2_weight_spec(group_size: int) -> QTensorConfig:
    if group_size <= 0:
        raise ValueError(f"group_size must be a positive integer for INT2 quantization, got {group_size}")
    return Int2PerGroupSpec(
        ch_axis=-1,
        group_size=group_size,
        symmetric=False,
        scale_type="float",
        round_method="half_even",
        is_dynamic=False,
    ).to_quantization_spec()


def _build_int4_or_uint4_weight_spec(quant_scheme: str, group_size: int) -> QTensorConfig:
    if group_size <= 0:
        raise ValueError(f"group_size must be a positive integer for PTQ, got {group_size}")
    if quant_scheme == "w_uint4_asym":
        dtype = Dtype.uint4
        symmetric = False
    elif quant_scheme == "w_int4_sym":
        dtype = Dtype.int4
        symmetric = True
    elif quant_scheme == "w_int4_asym":
        dtype = Dtype.int4
        symmetric = False
    else:
        raise ValueError(f"Unsupported quant_scheme for PTQ: {quant_scheme}")
    return QTensorConfig(
        dtype=dtype,
        observer_cls=PerGroupMinMaxObserver,
        symmetric=symmetric,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        qscheme=QSchemeType.per_group,
        ch_axis=1,
        is_dynamic=False,
        group_size=group_size,
    )


def weight_only_quantize(model: nn.Module, quant_scheme: str, group_size: int) -> nn.Module:
    """Apply weight-only quantization to a PyTorch model.

    Args:
        model: Model to quantize.
        quant_scheme: Quantization scheme name, e.g. INT2 or INT4/UINT4 variants.
        group_size: Per-group quantization size. Must be a positive integer.

    Returns:
        Quantized model. ``lm_head`` is excluded to reduce eval/PPL drift.
    """
    if quant_scheme in INT2_QUANT_SCHEMES:
        weight_spec = _build_int2_weight_spec(group_size)
    else:
        weight_spec = _build_int4_or_uint4_weight_spec(quant_scheme, group_size)

    quant_config = QConfig(
        global_quant_config=QLayerConfig(weight=weight_spec),
        # Keep lm_head in full precision to avoid eval/PPL drift.
        exclude=["lm_head", "*lm_head"],
    )
    quantizer = ModelQuantizer(quant_config)
    model = quantizer.quantize_model(model, None)
    return model
