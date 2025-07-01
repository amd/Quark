#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization API for ONNX."""

import os
import logging
import warnings
from quark.shares.utils.log import ScreenLogger, log_errors

from pathlib import Path
from typing import Union, Optional

import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader

from quark.onnx.quantization.config.config import Config
from quark.onnx.quantize import quantize_static, quantize_dynamic

__all__ = ["ModelQuantizer"]

logger = ScreenLogger(__name__)


class ModelQuantizer:
    """
    Provides an API for quantizing deep learning models using ONNX. This class handles the
    configuration and processing of the model for quantization based on user-defined parameters.

    Args:
        config (Config): Configuration object containing settings for quantization.

    Note:
        It is essential to ensure that the 'config' provided has all necessary quantization parameters defined.
        This class assumes that the model is compatible with the quantization settings specified in 'config'.
    """

    def __init__(self, config: Config) -> None:
        """
        Initializes the ModelQuantizer with the provided configuration.

        Args:
            config (Config): Configuration object containing global quantization settings.
        """
        self.config = config.global_quant_config

        self.set_logging_level()

        if self.config.ignore_warnings:
            warnings.simplefilter("ignore", ResourceWarning)
            warnings.simplefilter("ignore", UserWarning)

    def set_logging_level(self) -> None:
        if self.config.debug_mode:
            ScreenLogger.set_shared_level(logging.DEBUG)
        elif self.config.crypto_mode:
            ScreenLogger.set_shared_level(logging.CRITICAL)
        elif self.config.log_severity_level == 0:
            ScreenLogger.set_shared_level(logging.DEBUG)
        elif self.config.log_severity_level == 1:
            ScreenLogger.set_shared_level(logging.INFO)
        elif self.config.log_severity_level == 2:
            ScreenLogger.set_shared_level(logging.WARNING)
        elif self.config.log_severity_level == 3:
            ScreenLogger.set_shared_level(logging.ERROR)
        else:
            ScreenLogger.set_shared_level(logging.CRITICAL)

    @log_errors
    def quantize_model(self,
                       model_input: Union[str, Path, onnx.ModelProto],
                       model_output: Optional[Union[str, Path]] = None,
                       calibration_data_reader: Optional[CalibrationDataReader] = None,
                       calibration_data_path: Optional[str] = None) -> Optional[onnx.ModelProto]:
        """
        Quantizes the given ONNX model and saves the output to the specified path or returns a ModelProto.

        :param Union[str, Path, onnx.ModelProto] model_input: Path to the input ONNX model file or a ModelProto.
        :param Optional[Union[str, Path]] model_output: Path where the quantized ONNX model will be saved. Defaults to ``None``, in which case the model is not saved but the function returns a ModelProto.
        :param Union[CalibrationDataReader, None] calibration_data_reader: Data reader for model calibration. Defaults to ``None``.

        :return: ``None``
        """
        if isinstance(model_input, (str, Path)) and not os.path.exists(model_input):
            raise FileNotFoundError(f"Input model file {model_input} does not exist.")

        if not self.config.use_dynamic_quant:
            return quantize_static(model_input=model_input,
                                   model_output=model_output,
                                   calibration_data_reader=calibration_data_reader,
                                   calibration_data_path=calibration_data_path,
                                   calibrate_method=self.config.calibrate_method,
                                   quant_format=self.config.quant_format,
                                   activation_type=self.config.activation_type,
                                   weight_type=self.config.weight_type,
                                   input_nodes=self.config.input_nodes,
                                   output_nodes=self.config.output_nodes,
                                   op_types_to_quantize=self.config.op_types_to_quantize,
                                   nodes_to_quantize=self.config.nodes_to_quantize,
                                   extra_op_types_to_quantize=self.config.extra_op_types_to_quantize,
                                   nodes_to_exclude=self.config.nodes_to_exclude,
                                   subgraphs_to_exclude=self.config.subgraphs_to_exclude,
                                   specific_tensor_precision=self.config.specific_tensor_precision,
                                   execution_providers=self.config.execution_providers,
                                   per_channel=self.config.per_channel,
                                   reduce_range=self.config.reduce_range,
                                   optimize_model=self.config.optimize_model,
                                   use_external_data_format=self.config.use_external_data_format,
                                   convert_fp16_to_fp32=self.config.convert_fp16_to_fp32,
                                   convert_nchw_to_nhwc=self.config.convert_nchw_to_nhwc,
                                   include_sq=self.config.include_sq,
                                   include_rotation=self.config.include_rotation,
                                   include_cle=self.config.include_cle,
                                   include_auto_mp=self.config.include_auto_mp,
                                   include_fast_ft=self.config.include_fast_ft,
                                   enable_npu_cnn=self.config.enable_npu_cnn,
                                   enable_npu_transformer=self.config.enable_npu_transformer,
                                   debug_mode=self.config.debug_mode,
                                   crypto_mode=self.config.crypto_mode,
                                   print_summary=self.config.print_summary,
                                   extra_options=self.config.extra_options)
        else:
            return quantize_dynamic(model_input=model_input,
                                    model_output=model_output,
                                    op_types_to_quantize=self.config.op_types_to_quantize,
                                    per_channel=self.config.per_channel,
                                    reduce_range=self.config.reduce_range,
                                    weight_type=self.config.weight_type,
                                    nodes_to_quantize=self.config.nodes_to_quantize,
                                    nodes_to_exclude=self.config.nodes_to_exclude,
                                    subgraphs_to_exclude=self.config.subgraphs_to_exclude,
                                    use_external_data_format=self.config.use_external_data_format,
                                    debug_mode=self.config.debug_mode,
                                    crypto_mode=self.config.crypto_mode,
                                    extra_options=self.config.extra_options)
