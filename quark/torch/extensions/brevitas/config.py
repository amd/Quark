#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization Config API for Brevitas."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum, auto

import quark.torch.quantization.config.type as quark_config_type
import quark.torch.extensions.brevitas.algos as brevitas_algos


class Backend(Enum):
    """
    The backend target for quantization:
    - `layerwise`: Only quantizes inputs and weights of compute-heavy layers.
    """

    layerwise = auto()


@dataclass(eq=True)
class Config:
    """
    A class that encapsulates comprehensive quantization configurations for a machine learning model, allowing for detailed and hierarchical control over quantization parameters across different model components.

    - `global_quant_config`: The quantization configuration to be applied to the entire model.
    - `pre_quant_opt_config`: Optional optimization and pre-processing algorithms to apply to the model before quantization.
    - `algo_config`: optional algorithms to apply to the model after quantization to improve accuracy.
    - `backend`: The quantization backend to use.
    """

    # Global quantization configuration applied to the entire model.
    global_quant_config: QuantizationConfig

    # Optional pre-processing optimization - these will be applied in the same order as their position in the list.
    pre_quant_opt_config: List[brevitas_algos.PreQuantOptConfig] = field(
        default_factory=lambda: [brevitas_algos.Preprocess()])

    # Optional configuration for the quantization algorithm  - these will be applied in the same order as their position in the list.
    algo_config: List[brevitas_algos.AlgoConfig] = field(default_factory=list)

    backend: Backend = Backend.layerwise


@dataclass(eq=True)
class QuantizationConfig:
    """
    A data class that specifies quantization configurations for different components of a module, allowing hierarchical control over how each tensor type is quantized.

    - `input_tensors`: The quantization parameters (if any) to apply to activation inputs.
    - `output_tensors`: The quantization parameters (if any) to apply to activation outputs.
    - `weight`: The quantization parameters (if any) to apply to the model weights.
    - `bias`: The quantization parameters (if any) to apply to the model biases.
    """

    input_tensors: Optional[QuantizationSpec] = None
    output_tensors: Optional[QuantizationSpec] = None
    weight: Optional[QuantizationSpec] = None
    bias: Optional[QuantizationSpec] = None


class QuantType(Enum):
    """
    The fundamental data type of the quantized values:

    - `int_quant`: Values quantized to integers.
    - `float_quant`: Values quantized to floating point.

    """
    int_quant = auto()
    float_quant = auto()


class ParamType(Enum):
    """
    Method for determining scale and zero point:

    - `stats`: Statistics
    - `mse`: Mean Squared Error
    """
    stats = auto()
    mse = auto()


@dataclass(eq=True, frozen=True)
class QuantizationSpec:
    """
    A data class that defines the specifications for quantizing tensors within a model.
    It has some reasonable defaults so it can be used as is if desired.

    - `qscheme`: The granularity of quantization e.g. if applied to the whole tensor or to each channel.
    - `symmetric`: If true, the zero point is in the middle of the range of representable numbers, if false the quantized value will be mapped to between the minimum and maximum observed values. Asymmetric quantization is more expensive but may be better for ranges that aren't expected to be negative.
    - `scale_type`: Whether the scales use floating point or power of two values. Power of two allows lower bit widths and may be required by some embedded devices.
    - `quant_type`: The type of quantization we want: integer or floating point. If float, we also need to specify the exponent and mantissa bit widths.
    - `param_type`: Method for determing scale and zero point.
    - `bit_width`: Level of precision we want the quantization to be.
    - `exponent_bit_width`: The level of precision we want for the exponent when using the float quant_type.
    - `mantissa_bit_width`: The level of precision we want for the mantissa when using the float quant_type.
    """

    qscheme: quark_config_type.QSchemeType = quark_config_type.QSchemeType.per_tensor
    symmetric: bool = True
    scale_type: quark_config_type.ScaleType = quark_config_type.ScaleType.float
    quant_type: QuantType = QuantType.int_quant
    param_type: ParamType = ParamType.stats

    bit_width: int = 8
    exponent_bit_width: Optional[int] = None
    mantissa_bit_width: Optional[int] = None
