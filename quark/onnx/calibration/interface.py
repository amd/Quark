#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod, TensorsData
from onnxruntime.quantization.quant_utils import QuantType

from quark.common.profiler import ProfileStep, profile_scope
from quark.common.utils.log import ScreenLogger, log_errors
from quark.onnx.utils.system_utils import get_memory_usage

from .calib_utils import save_tensor_hist_fig
from .calibrate import calibrate_model
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
    ("PercentileCandidates", "percentile_candidates"),
    ("MinMSEModePof2Scale", "minmse_mode"),
    ("CalibOptimizeMem", "optimize_mem"),
    ("CalibOptimizeDisk", "optimize_disk"),
    ("CalibWorkerNum", "worker_num"),
]


@log_errors
@profile_scope(ProfileStep.CALIBRATION)
def run_calibration(
    model_input: str | Path | onnx.ModelProto,
    data_reader: CalibrationDataReader,
    op_types_to_calibrate: Sequence[str] | None = None,
    activation_type: QuantType = QuantType.QInt8,
    calibrate_method: CalibrationMethod | LayerWiseMethod | PowerOfTwoMethod = CalibrationMethod.MinMax,
    use_external_data_format: bool = False,
    execution_providers: list[str] | None = ["CPUExecutionProvider"],
    extra_options: dict[str, Any] = {},
) -> TensorsData:
    """
    This is an interface function used for the calibration.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.
    :param CalibrationDataReader data_reader: Data reader for model calibration.
    :param Optional[Sequence[str]] op_types_to_calibrate: List of operator types to calibrate. Defaults to ``None``, which indicates that all float32/float16 tensors are calibrated.
    :param QuantType activation_type: The quantization type of activation. Default is QuantType.QInt8.
    :param Union[CalibrationMethod, LayerWiseMethod, PowerOfTwoMethod] calibrate_method: Calibration method to use (MinMax, Entropy, Percentile, Distribution, NonOverflow or MinMSE).
    :param bool use_external_data_format: Whether to use external data format for large models.
    :param Union[List[str], None] execution_providers: List of execution providers for ONNX Runtime.
    :param Dict[str, Any] extra_options: Extra options for quantization, which contains additional options for calibrator configuration.

    :return: Data range for each quantizing activation tensor.
    """
    tensors_range: TensorsData | None = None

    quantized_tensor_type: dict[str, QuantType] = {}
    if len(extra_options.get("TensorQuantOverrides", {})):
        for tensor_name, quant_overrides in extra_options["TensorQuantOverrides"].items():
            if not isinstance(quant_overrides, list):
                logger.warning(f"Invalid quant overrides {quant_overrides} for tensor {tensor_name}.")
                continue

            for override in quant_overrides:
                if not isinstance(override, dict):
                    logger.warning(f"The quant override {override} should be a dict.")
                    continue

                if "quant_type" in override:
                    quantized_tensor_type[tensor_name] = override["quant_type"]

    # Mapping calibration extra options from the generic dict
    calib_extra_options = {
        key: extra_options.get(name) for (name, key) in extra_options_keys_mapping if name in extra_options
    }

    calib_data_size = calib_extra_options.get("data_size")
    calib_data_reader = data_reader
    if not isinstance(data_reader, CachedDataReader) or calib_data_size and calib_data_size < len(data_reader):
        calib_data_reader = CachedDataReader(data_reader, calib_data_size)
    calib_data_reader.reset_iter()

    logger.info(
        f"Start running calibration on {len(calib_data_reader)} samples with extra options {calib_extra_options}..."
    )
    start_time = time.perf_counter()

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

        if calibrate_method in [
            PowerOfTwoMethod.MinMSE,
            LayerWiseMethod.LayerWisePercentile,
        ] and calib_extra_options.get("optimize_mem", True):
            logger.warning(
                "Will automatically set the 'CalibOptimizeMem' to False and retry. "
                "If it still doesn't work, please provide another temporary directory "
                "with sufficient disk space via the option 'TmpDir' and try again."
            )
            calib_extra_options["optimize_mem"] = False

            if calibrate_method == PowerOfTwoMethod.MinMSE:
                if calib_extra_options.get("minmse_mode", "All") != "MostCommon":
                    logger.debug("The minmse_mode parameter changed to 'MostCommon' to avoid caching data")
                    calib_extra_options["minmse_mode"] = "MostCommon"

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
            logger.warning("Will automatically set the execution provider to CPU and retry.")
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

    if extra_options.get("SaveTensorHistFig", False):
        calib_data_reader.reset_iter()
        save_tensor_hist_fig(
            model_input,
            calib_data_reader,
            op_types_to_calibrate,
            activation_type,
            calibrate_method,
            use_external_data_format,
            execution_providers,
            calib_extra_options,
        )

    return tensors_range


@log_errors
def fake_calibration(model_input: str | Path | onnx.ModelProto) -> TensorsData:
    """
    A calibration function that produces fake tensor range of [0,1],
    intended for scenarios that don't need actual calibration, such as
    block FP quantization, to accelerate the entire process.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.

    :return: Data range for each quantizing tensor.
    """

    def _get_fake_tensor_range(model: onnx.ModelProto) -> dict[str, tuple[np.ndarray[Any, Any], np.ndarray[Any, Any]]]:
        initializer_names = {init.name for init in model.graph.initializer}

        fake_tensor_range = {}
        for node in model.graph.node:
            for inp in node.input:
                if inp not in initializer_names and inp not in fake_tensor_range:
                    fake_tensor_range[inp] = (np.array([0.0]).astype(np.float32), np.array([1.0]).astype(np.float32))
            for out in node.output:
                if out not in initializer_names and out not in fake_tensor_range:
                    fake_tensor_range[out] = (np.array([0.0]).astype(np.float32), np.array([1.0]).astype(np.float32))
        return fake_tensor_range

    model = model_input if isinstance(model_input, onnx.ModelProto) else onnx.load(model_input)
    fake_tensor_range = _get_fake_tensor_range(model)
    tensors_range = TensorsData(CalibrationMethod.MinMax, fake_tensor_range)

    return tensors_range
