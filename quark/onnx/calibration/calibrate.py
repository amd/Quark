#!/usr/bin/env python
#
# Modifications copyright(c) 2023 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

from pathlib import Path
from typing import Any, Sequence

import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod, TensorsData
from onnxruntime.quantization.quant_utils import QuantType

from quark.onnx.quantization.quant_utils import ExtendedQuantType
from quark.onnx.utils.system_utils import Profiler, create_tmp_dir
from quark.shares.utils.log import ScreenLogger, log_errors

from .calibrators import create_calibrator_float_scale, create_calibrator_power_of_two
from .methods import LayerWiseMethod, PowerOfTwoMethod

logger = ScreenLogger(__name__)


@log_errors
def calibrate_model(
    model_input: str | Path | onnx.ModelProto,
    calib_data_reader: CalibrationDataReader,
    op_types_to_calibrate: Sequence[str] | None = None,
    activation_type: QuantType | ExtendedQuantType = QuantType.QInt8,
    calibrate_method: CalibrationMethod | LayerWiseMethod | PowerOfTwoMethod = CalibrationMethod.MinMax,
    use_external_data_format: bool = False,
    execution_providers: list[str] | None = ["CPUExecutionProvider"],
    quantized_tensor_type: dict[Any, Any] = {},
    calib_extra_options: dict[str, Any] = {},
) -> TensorsData:
    """
    Calling the calibrator to calibrate activation tensors.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.
    :param CalibrationDataReader calib_data_reader: Data reader for model calibration that needs to implement the ``__len__`` method.
    :param Optional[Sequence[str]] op_types_to_calibrate: List of operator types to calibrate. Defaults to ``None``, which indicates that all float32/float16 tensors are calibrated.
    :param Union[QuantType, ExtendedQuantType] activation_type: The quantization type of activation. Default is QuantType.QInt8.
    :param Union[CalibrationMethod, LayerWiseMethod, PowerOfTwoMethod] calibrate_method: Calibration method to use (MinMax, Entropy, Percentile, Distribution, NonOverflow or MinMSE).
    :param bool use_external_data_format: Whether to use external data format for large models.
    :param Union[List[str], None] execution_providers: List of execution providers for ONNX Runtime.
    :param Dict[str, Any] calib_extra_options: Additional options for calibrator configuration.

    :return: Data range for each quantizing tensor.
    """

    with create_tmp_dir("quark_onnx.calib.") as quant_tmp_dir:
        if isinstance(calibrate_method, PowerOfTwoMethod):
            calibrator = create_calibrator_power_of_two(
                model_input,
                op_types_to_calibrate,
                augmented_model_path=Path(quant_tmp_dir).joinpath("augmented_model.onnx").as_posix(),
                activation_type=activation_type,
                calibrate_method=calibrate_method,
                use_external_data_format=use_external_data_format,
                execution_providers=execution_providers,
                quantized_tensor_type=quantized_tensor_type,
                extra_options=calib_extra_options,
            )
        else:
            calibrator = create_calibrator_float_scale(
                model_input,
                op_types_to_calibrate,
                augmented_model_path=Path(quant_tmp_dir).joinpath("augmented_model.onnx").as_posix(),
                calibrate_method=calibrate_method,
                use_external_data_format=use_external_data_format,
                execution_providers=execution_providers,
                extra_options=calib_extra_options,
            )
        logger.info(
            f"Data collection of {calibrate_method} in progress. Runtime will depend on your model and data size."
        )
        with Profiler(msg=[["", "", "calibration: collect data (onnx inference + numpy statistics)"]]):
            calibrator.collect_data(calib_data_reader)
        with Profiler(msg=[["", "", "calibration: compute data"]]):
            tensors_range = calibrator.compute_data()
        del calibrator

        return tensors_range
