#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod, TensorsData
from onnxruntime.quantization.quant_utils import QuantType

from quark.onnx.quant_utils import get_memory_usage
from quark.shares.utils.log import ScreenLogger, log_errors

from .calibrators import calibrate_model
from .data_readers import CachedDataReader
from .methods import LayerWiseMethod, PowerOfTwoMethod

logger = ScreenLogger(__name__)

extra_options_keys_mapping = [
    ("CalibDataSize", "data_size"),
    ("CalibTensorRangeSymmetric", "symmetric"),
    ("CalibMovingAverage", "moving_average"),
    ("CalibMovingAverageConstant", "averaging_constant"),
    ("NumBins", "num_bins"),
    ("NumQuantizedBins", "num_quantized_bins"),
    ("Percentile", "percentile"),
    ("Scenario", "scenario"),
    ("LWPMetric", "lwp_metric"),
    ("ActivationBitWidth", "activation_bitwidth"),
    ("PercentileCandidates", "percentile_candidates"),
    ("MinMSEModePof2Scale", "minmse_mode"),
    ("CalibOptimizeMem", "optimize_mem"),
    ("CalibWorkerNum", "worker_num"),
]


@log_errors
def run_calibration(
    model_input: Union[str, Path, onnx.ModelProto],
    data_reader: CalibrationDataReader,
    op_types_to_calibrate: Sequence[str] | None = None,
    activation_type: QuantType = QuantType.QInt8,
    calibrate_method: Union[CalibrationMethod, LayerWiseMethod, PowerOfTwoMethod] = CalibrationMethod.MinMax,
    use_external_data_format: bool = False,
    execution_providers: Union[list[str], None] = ["CPUExecutionProvider"],
    quantized_tensor_type: dict[Any, Any] = {},
    extra_options: dict[str, Any] = {},
) -> TensorsData:
    """
    This is an interface function used for calibration.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.
    :param CalibrationDataReader data_reader: Data reader for model calibration.
    :param Optional[Sequence[str]] op_types_to_calibrate: List of operator types to calibrate. Defaults to ``None``, which indicates that all float32/float16 tensors are calibrated.
    :param QuantType activation_type: The quantization type of activation. Default is QuantType.QInt8.
    :param Union[CalibrationMethod, LayerWiseMethod, PowerOfTwoMethod] calibrate_method: Calibration method to use (MinMax, Entropy, Percentile, Distribution, NonOverflow or MinMSE).
    :param bool use_external_data_format: Whether to use external data format for large models.
    :param Union[List[str], None] execution_providers: List of execution providers for ONNX Runtime.
    :param Dict[str, Any] extra_options: Extra options for quantization, which contains additional options for calibrator configuration.

    :return: Data range for each quantizing tensor.
    """

    # Mapping calibration extra options from the generic dict
    calib_extra_options = {
        key: extra_options.get(name) for (name, key) in extra_options_keys_mapping if name in extra_options
    }

    calib_data_size = calib_extra_options.get("data_size")
    calib_data_reader = CachedDataReader(data_reader, calib_data_size)

    logger.info(
        f"Start running calibration on {len(calib_data_reader)} samples with extra options {calib_extra_options}..."
    )
    start_time = time.perf_counter()

    tensors_range: TensorsData | None = None
    try:
        tensors_range = calibrate_model(
            model_input,
            calib_data_reader,
            op_types_to_calibrate,
            activation_type,
            calibrate_method,
            use_external_data_format,
            execution_providers,
            quantized_tensor_type,
            calib_extra_options,
        )

    except OSError as e:
        logger.error(f"Encountered an error (commonly due to insufficient disk space for the temporary directory): {e}")

        if calibrate_method == PowerOfTwoMethod.MinMSE and calib_extra_options.get("optimize_mem", True):
            logger.warning("Will automatically set the 'CalibOptimizeMem' to False and retry.")
            calib_extra_options["optimize_mem"] = False
            calib_data_reader.reset_iter()
            tensors_range = calibrate_model(
                model_input,
                calib_data_reader,
                op_types_to_calibrate,
                activation_type,
                calibrate_method,
                use_external_data_format,
                execution_providers,
                quantized_tensor_type,
                calib_extra_options,
            )
        else:
            logger.warning(
                "Please provide another temporary directory with sufficient disk space via the option 'TmpDir'."
            )

    except onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException as e:
        logger.error(f"Encountered an error (commonly occurs when initializing an inference session): {e}")

        if isinstance(execution_providers, list) and "CPUExecutionProvider" not in execution_providers:
            logger.warning("Will automatically set the executiion provider to CPU and retry.")
            calib_data_reader.reset_iter()
            tensors_range = calibrate_model(
                model_input,
                calib_data_reader,
                op_types_to_calibrate,
                activation_type,
                calibrate_method,
                use_external_data_format,
                ["CPUExecutionProvider"],
                quantized_tensor_type,
                calib_extra_options,
            )
        else:
            logger.warning("Please switch to another execution provider via the argument 'execution_providers'.")

    except Exception as e:
        memory_usage = get_memory_usage()
        logger.warning(f"Currently the host memory usage is {memory_usage:.1f}%.")

        if memory_usage < 95:
            logger.error(f"Encountered an unexpected error: {e}")

            if calibrate_method == PowerOfTwoMethod.MinMSE:
                logger.warning(
                    "Will automatically set the 'calibrate_method' to the simple and faster method 'NonOverflow' and retry."
                )
                calib_data_reader.reset_iter()
                tensors_range = calibrate_model(
                    model_input,
                    calib_data_reader,
                    op_types_to_calibrate,
                    activation_type,
                    PowerOfTwoMethod.NonOverflow,
                    use_external_data_format,
                    execution_providers,
                    quantized_tensor_type,
                    calib_extra_options,
                )
            elif calibrate_method in (
                CalibrationMethod.Percentile,
                CalibrationMethod.Entropy,
                CalibrationMethod.Distribution,
                LayerWiseMethod.LayerWisePercentile,
            ):
                logger.warning(
                    "Will automatically set the 'calibrate_method' to the simple and faster method 'MinMax' and retry."
                )
                calib_data_reader.reset_iter()
                tensors_range = calibrate_model(
                    model_input,
                    calib_data_reader,
                    op_types_to_calibrate,
                    activation_type,
                    CalibrationMethod.MinMax,
                    use_external_data_format,
                    execution_providers,
                    quantized_tensor_type,
                    calib_extra_options,
                )
        else:
            logger.error(f"Encountered an error (commonly due to host OOM while caching data): {e}")

            logger.warning("Please reduce the size of calibration dataset via the option 'CalibDataSize'.")

    finally:
        if tensors_range is None:
            raise RuntimeError(
                "The calibration failed, please follow the hints to take action or check your model and data reader."
            )

    end_time = time.perf_counter()
    calib_time = end_time - start_time
    logger.info(f"The calibration has been finished. It took {calib_time:.1f}s to complete.")

    return tensors_range


@log_errors
def fake_calibration(model_input: Union[str, Path, onnx.ModelProto]) -> TensorsData:
    """A calibration function that produces fake tensor range of [0,1],
    intended for scenarios that don't need actual calibration, such as
    block FP quantization, to accelerate the entire process.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.

    :return: Data range for each quantizing tensor.
    """

    def _get_fake_tensor_range(model: onnx.ModelProto) -> dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
        fake_tensor_range = {}
        for node in model.graph.node:
            for input_ in node.input:
                fake_tensor_range[input_] = (np.array([0.0]).astype(np.float32), np.array([1.0]).astype(np.float32))
            for output_ in node.output:
                fake_tensor_range[output_] = (np.array([0.0]).astype(np.float32), np.array([1.0]).astype(np.float32))
        return fake_tensor_range

    model = model_input if isinstance(model_input, onnx.ModelProto) else onnx.load(model_input)
    fake_tensor_range = _get_fake_tensor_range(model)
    tensors_range = TensorsData(CalibrationMethod.MinMax, fake_tensor_range)

    return tensors_range
