#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization Config API for ONNX"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any, List, Union, Tuple

from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantType, QuantFormat

from quark.onnx.quant_utils import (PowerOfTwoMethod, ExtendedQuantType, ExtendedQuantFormat)


@dataclass(eq=True)
class Config:
    """
    A class that encapsulates comprehensive quantization configurations for a machine learning model, allowing for detailed and hierarchical control over quantization parameters across different model components.

    :param QuantizationConfig global_quant_config: Global quantization configuration applied to the entire model unless overridden at the layer level.
    """

    # Global quantization configuration applied to the entire model unless overridden at the layer level.
    global_quant_config: QuantizationConfig


@dataclass(eq=True)
class QuantizationConfig:
    """
    A data class that specifies quantization configurations for different components of a module, allowing hierarchical control over how each tensor type is quantized.

    :param Union[CalibrationMethod, PowerOfTwoMethod] calibrate_method: Method used for calibration. Default is ``CalibrationMethod.MinMax``.
    :param Union[QuantFormat, ExtendedQuantType] quant_format: Format of quantization. Default is ``QuantFormat.QDQ``.
    :param Union[QuantType, ExtendedQuantType] activation_type: Type of quantization for activations. Default is ``QuantType.QInt8``.
    :param Union[QuantFormat, ExtendedQuantType] weight_type: Type of quantization for weights. Default is ``QuantType.QInt8``.
    :param List[str] input_nodes: List of input nodes to be quantized. Default is ``[]``.
    :param List[str] output_nodes: List of output nodes to be quantized. Default is ``[]``.
    :param List[str] op_types_to_quantize: List of operation types to be quantized. Default is ``[]``.
    :param List[str] extra_op_types_to_quantize: List of additional operation types to be quantized. Default is ``[]``.
    :param List[str] nodes_to_quantize: List of node names to be quantized. Default is ``[]``.
    :param List[str] nodes_to_exclude: List of node names to be excluded from quantization. Default is ``[]``.
    :param List[Tuple[List[str]] subgraphs_to_exclude: List of start and end node names of subgraphs to be excluded from quantization. Default is ``[]``.
    :param bool specific_tensor_precision: Flag to enable specific tensor precision. Default is ``False``.
    :param List[str] execution_providers: List of execution providers. Default is ``['CPUExecutionProvider']``.
    :param bool per_channel: Flag to enable per-channel quantization. Default is ``False``.
    :param bool reduce_range: Flag to reduce quantization range. Default is ``False``.
    :param bool optimize_model: Flag to optimize the model. Default is ``True``.
    :param bool use_dynamic_quant: Flag to use dynamic quantization. Default is ``False``.
    :param bool use_external_data_format: Flag to use external data format. Default is ``False``.
    :param bool convert_fp16_to_fp32: Flag to convert FP16 to FP32. Default is ``False``.
    :param bool convert_nchw_to_nhwc: Flag to convert NCHW to NHWC. Default is ``False``.
    :param bool include_sq: Flag to include square root in quantization. Default is ``False``.
    :param bool include_cle: Flag to include CLE in quantization. Default is ``True``.
    :param bool include_auto_mp: Flag to include automatic mixed precision. Default is ``False``.
    :param bool include_fast_ft: Flag to include fast fine-tuning. Default is ``False``.
    :param bool enable_npu_cnn: Flag to enable NPU CNN. Default is ``False``.
    :param bool enable_npu_transformer: Flag to enable NPU Transformer. Default is ``False``.
    :param bool debug_mode: Flag to enable debug mode. Default is ``False``.
    :param bool print_summary: Flag to print summary of quantization. Default is ``True``.
    :param bool ignore_warnings:: Flag to suppress the warnings globally. Default is ``True``.
    :param int log_severity_level: 0:DEBUG, 1:INFO, 2:WARNING. 3:ERROR, 4:CRITICAL/FATAL. Default is ``1``.
    :param Dict[str, Any] extra_options: Dictionary for additional options. Default is ``{}``.
    :param bool crypto_mode: Flag to enable crypto mode (the model information will be encrypted or hidden). Default is ``False``.
    """
    calibrate_method: Union[CalibrationMethod, PowerOfTwoMethod] = CalibrationMethod.MinMax
    quant_format: Union[QuantFormat, ExtendedQuantFormat] = QuantFormat.QDQ
    activation_type: Union[QuantType, ExtendedQuantType] = QuantType.QInt8
    weight_type: Union[QuantType, ExtendedQuantType] = QuantType.QInt8

    input_nodes: List[str] = field(default_factory=list)
    output_nodes: List[str] = field(default_factory=list)
    op_types_to_quantize: List[str] = field(default_factory=list)
    nodes_to_quantize: List[str] = field(default_factory=list)
    extra_op_types_to_quantize: List[str] = field(default_factory=list)
    nodes_to_exclude: List[str] = field(default_factory=list)
    subgraphs_to_exclude: List[Tuple[List[str]]] = field(default_factory=list)

    specific_tensor_precision: bool = False
    execution_providers: List[str] = field(default_factory=lambda: ['CPUExecutionProvider'])

    per_channel: bool = False
    reduce_range: bool = False
    optimize_model: bool = True
    use_dynamic_quant: bool = False
    use_external_data_format: bool = False

    convert_fp16_to_fp32: bool = False
    convert_nchw_to_nhwc: bool = False

    include_sq: bool = False
    include_rotation: bool = False
    include_cle: bool = True
    include_auto_mp: bool = False
    include_fast_ft: bool = False

    enable_npu_cnn: bool = False
    enable_npu_transformer: bool = False

    debug_mode: bool = False
    crypto_mode: bool = False
    print_summary: bool = True
    ignore_warnings: bool = True
    log_severity_level: int = 1

    extra_options: Dict[str, Any] = field(default_factory=dict)
