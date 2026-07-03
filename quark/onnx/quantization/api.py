#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization API for ONNX."""

import copy
import os
import warnings
from pathlib import Path

import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader

from quark.common.utils.log import ScreenLogger, log_errors

from ..utils.print_utils import print_user_supplied_configuration
from .config.algorithm import (
    AdaQuantConfig,
    AdaRoundConfig,
    AlgoConfig,
    AutoMixprecisionConfig,
    CLEConfig,
    QuarotConfig,
    SmoothQuantConfig,
    _algo_flag,
    _resolove_algo_conflict,
)
from .config.config import Config, QConfig
from .config.legacy import QuantizationConfig
from .config.maps import _check_q_config, _map_mixed_precision_tensors, _map_q_config
from .quant_utils import recursive_update
from .quantize import quantize_dynamic, quantize_static

__all__ = ["ModelQuantizer"]

logger = ScreenLogger(__name__)


class ModelQuantizer:
    """Provides an API for quantizing deep learning models using ONNX.

    This class handles the configuration and processing of the model for quantization based on user-defined parameters.

    :param Config config: Configuration object containing settings for quantization.

    Note:
        - It is essential to ensure that the 'config' provided has all necessary quantization parameters defined.
        - This class assumes that the model is compatible with the quantization settings specified in 'config'.
    """

    def __init__(self, config: Config | QConfig) -> None:
        """Initializes the ModelQuantizer with the provided configuration.

        :param Config config: Configuration object containing global quantization settings.
        """
        if isinstance(config, Config):
            logger.warning("Config has been replaced by QConfig. The old API will be removed in the next release.")
            self.config = config.global_quant_config

            # Do not call ScreenLogger.set_shared_level here: it mutates global log level for the
            # entire process and is never restored, flooding CI with [QUARK-DEBUG] after any test
            # that uses debug_mode. Use QUARK_LOG_LEVEL=debug when you need verbose Quark logs.

            if self.config.ignore_warnings:
                warnings.simplefilter("ignore", ResourceWarning)
                warnings.simplefilter("ignore", UserWarning)
        elif isinstance(config, QConfig):
            self.config = config  # type: ignore

            if self.config.extra_options.get("IgnoreWarnings", True):
                warnings.simplefilter("ignore", ResourceWarning)
                warnings.simplefilter("ignore", UserWarning)
        else:
            raise ValueError("quantization config must be one of Config and QConfig.")

    @log_errors
    def quantize_model(
        self,
        model_input: str | Path | onnx.ModelProto,
        model_output: str | Path | None = None,
        calibration_data_reader: CalibrationDataReader | None = None,
        calibration_data_path: str | None = None,
        algorithms: list[AlgoConfig] | None = None,
    ) -> onnx.ModelProto | None:
        """Quantizes the given ONNX model and saves the output to the specified path or returns a ModelProto.

        :param Union[str, Path, onnx.ModelProto] model_input: Path to the input ONNX model file or a ModelProto.
        :param Optional[Union[str, Path]] model_output: Path where the quantized ONNX model will be saved. Defaults to ``None``, in which case the model is not saved but the function returns a ModelProto.
        :param Union[CalibrationDataReader, None] calibration_data_reader: Data reader for model calibration. Defaults to ``None``.
        :param List[AlgoConfig] algorithms: List of algorithms like CLE, SmoothQuant and AdaRound. Defaults to ``None``.

        :return: None
        """
        if isinstance(self.config, QuantizationConfig):
            algorithms = algorithms or []
            logger.warning(
                "The algorithm API is algo_config in QConfig. The old API will be removed in the next release."
            )
        if isinstance(self.config, QConfig):
            _check_q_config(self.config)
            algorithms = self.config.algo_config

        if isinstance(model_input, str | Path) and not os.path.exists(model_input):
            raise FileNotFoundError(f"Input model file {model_input} does not exist.")

        if not (isinstance(self.config, QuantizationConfig) and self.config.use_dynamic_quant):
            algorithms = _resolove_algo_conflict(algorithms)
            # Snapshot the raw user input before algorithm configs mutate it, so the
            # effective-config summary (QConfig path only) can show both "what the
            # user passed" and "what quantize_static will actually run with".
            user_extra_snapshot = copy.deepcopy(self.config.extra_options) if isinstance(self.config, QConfig) else None
            for algo in algorithms:
                recursive_update(self.config.extra_options, algo._get_config(self.config.extra_options))
            if isinstance(self.config, QuantizationConfig):
                if self.config.specific_tensor_precision and "MixedPrecisionTensor" in self.config.extra_options:
                    self.config.extra_options["TensorQuantOverrides"] = _map_mixed_precision_tensors(
                        self.config.extra_options["MixedPrecisionTensor"]
                    )

                return quantize_static(
                    model_input=model_input,
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
                    include_sq=(self.config.include_sq or _algo_flag(algorithms, SmoothQuantConfig)),
                    include_rotation=(self.config.include_rotation or _algo_flag(algorithms, QuarotConfig)),
                    include_cle=(self.config.include_cle or _algo_flag(algorithms, CLEConfig)),
                    include_auto_mp=(self.config.include_auto_mp or _algo_flag(algorithms, AutoMixprecisionConfig)),
                    include_fast_ft=(
                        self.config.include_fast_ft
                        or _algo_flag(algorithms, AdaRoundConfig)
                        or _algo_flag(algorithms, AdaQuantConfig)
                    ),
                    enable_npu_cnn=self.config.enable_npu_cnn,
                    enable_npu_transformer=self.config.enable_npu_transformer,
                    debug_mode=self.config.debug_mode,
                    crypto_mode=self.config.crypto_mode,
                    print_summary=self.config.print_summary,
                    # Legacy Config / QuantizationConfig keeps main's flat-dump
                    # summary; only QConfig opts into the categorized one below.
                    _print_effective_summary=False,
                    extra_options=self.config.extra_options,
                )
            if isinstance(self.config, QConfig):
                mapping = _map_q_config(self.config, model_input)

                # Pre-quantization: echo user input as supplied to quantize_static.
                # Values shown here may still be normalized downstream (FP16 detect,
                # TensorQuantOverrides upgrade, Int16Scale flip). The final
                # effective extra_options is printed after run_static_quantization.
                print_user_supplied_configuration(
                    q_config_source=self.config,
                    user_extra=user_extra_snapshot if user_extra_snapshot is not None else {},
                    quant_format=mapping["quant_format"],
                    activation_type=mapping["activation_type"].map_onnx_format,
                    weight_type=mapping["weight_type"].map_onnx_format,
                    calibrate_method=mapping["calibrate_method"],
                    optimize_model=mapping["extra_options"]["OptimizeModel"],
                    model_input=model_input,
                    model_output=model_output,
                    calibration_data_reader=calibration_data_reader,
                )

                return quantize_static(
                    model_input=model_input,
                    model_output=model_output,
                    calibration_data_reader=calibration_data_reader,
                    calibration_data_path=calibration_data_path,
                    calibrate_method=mapping["calibrate_method"],
                    quant_format=mapping["quant_format"],
                    activation_type=mapping["activation_type"].map_onnx_format,
                    weight_type=mapping["weight_type"].map_onnx_format,
                    input_nodes=mapping["extra_options"]["InputNodes"],
                    output_nodes=mapping["extra_options"]["OutputNodes"],
                    op_types_to_quantize=mapping["extra_options"]["OpTypesToQuantize"],
                    nodes_to_quantize=mapping["extra_options"]["NodesToQuantize"],
                    specific_tensor_precision=mapping["extra_options"]["SpecificTensorPrecision"],
                    extra_op_types_to_quantize=mapping["extra_options"]["ExtraOpTypesToQuantize"],
                    execution_providers=mapping["extra_options"]["ExecutionProviders"],
                    optimize_model=mapping["extra_options"]["OptimizeModel"],
                    convert_fp16_to_fp32=mapping["extra_options"]["ConvertFP16ToFP32"],
                    convert_nchw_to_nhwc=mapping["extra_options"]["ConvertNCHWToNHWC"],
                    enable_npu_cnn=mapping["extra_options"]["EnableNPUCnn"],
                    debug_mode=mapping["extra_options"]["DebugMode"],
                    crypto_mode=mapping["extra_options"]["CryptoMode"],
                    print_summary=mapping["extra_options"]["PrintSummary"],
                    _print_effective_summary=True,
                    _user_extra_snapshot=user_extra_snapshot,
                    nodes_to_exclude=mapping["nodes_to_exclude"],
                    subgraphs_to_exclude=mapping["subgraphs_to_exclude"],
                    per_channel=mapping["per_channel"],
                    use_external_data_format=mapping["use_external_data_format"],
                    include_sq=_algo_flag(algorithms, SmoothQuantConfig),
                    include_rotation=_algo_flag(algorithms, QuarotConfig),
                    include_cle=_algo_flag(algorithms, CLEConfig),
                    include_auto_mp=_algo_flag(algorithms, AutoMixprecisionConfig),
                    include_fast_ft=_algo_flag(algorithms, AdaRoundConfig) or _algo_flag(algorithms, AdaQuantConfig),
                    extra_options=mapping["extra_options"],
                )
        else:
            return quantize_dynamic(
                model_input=model_input,
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
                extra_options=self.config.extra_options,
            )
