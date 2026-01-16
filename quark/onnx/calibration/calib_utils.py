#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json
import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod, HistogramCollector, TensorsData
from onnxruntime.quantization.quant_utils import QuantType

from quark.onnx.utils.system_utils import check_and_create_path, create_tmp_dir
from quark.shares.utils.import_utils import _is_package_available
from quark.shares.utils.log import ScreenLogger

from .calibrators import create_calibrator_float_scale
from .methods import LayerWiseMethod

logger = ScreenLogger(__name__)


def save_tensor_histogram(calibrator: HistogramCollector) -> str:
    import matplotlib.pyplot as plt

    hist_tmp_dir = "./tensor_hist"
    check_and_create_path(hist_tmp_dir)
    hist_tmp_dir = os.path.abspath(hist_tmp_dir)

    percentile_dict = calibrator.collector.compute_percentile()
    for tensor_name, tensor_value in calibrator.collector.histogram_dict.items():
        percentile_min = percentile_dict[tensor_name][0].item()
        percentile_max = percentile_dict[tensor_name][1].item()
        tensor_name = tensor_name.replace("/", "_")
        tensor_name = tensor_name.replace(".", "_")
        tensor_name = tensor_name.replace(":", "_")
        tensor_bins = tensor_value[1]
        tensor_freq = tensor_value[0]
        bar_width = tensor_bins[1] - tensor_bins[0]
        plt.bar(tensor_bins[:-1], tensor_freq, width=bar_width)

        model_hist_path = Path(hist_tmp_dir).joinpath(tensor_name).as_posix()
        min_value = tensor_value[2]
        max_value = tensor_value[3]
        plt.title(tensor_name)
        plt.axvline(x=max_value, color="r", linestyle="--", linewidth=2)
        plt.axvline(x=percentile_max, color="r", linestyle="--", linewidth=2)
        plt.axvline(x=min_value, color="r", linestyle="--", linewidth=2)
        plt.axvline(x=percentile_min, color="r", linestyle="--", linewidth=2)
        plt.xlabel(
            f"Value Max:{max_value:.4f}; PerMax:{percentile_max:.4f} Min:{min_value:.4f}; PerMin:{percentile_min:.4f}"
        )
        plt.ylabel("Frequency")
        plt.savefig(model_hist_path)

        plt.close()

    return hist_tmp_dir


def save_tensor_hist_fig(
    model_input: str | Path | onnx.ModelProto,
    calib_data_reader: CalibrationDataReader,
    op_types_to_calibrate: Sequence[str] | None = None,
    activation_type: QuantType = QuantType.QInt8,
    calibrate_method: CalibrationMethod | LayerWiseMethod = CalibrationMethod.Percentile,
    use_external_data_format: bool = False,
    execution_providers: list[str] | None = ["CPUExecutionProvider"],
    calib_extra_options: dict[str, Any] = {},
) -> None:
    """
    Save the histogram of tensors to files.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.
    :param CalibrationDataReader calib_data_reader: Data reader for model calibration that needs to implement the ``__len__`` method.
    :param Optional[Sequence[str]] op_types_to_calibrate: List of operator types to calibrate. Defaults to ``None``, which indicates that all float32/float16 tensors are calibrated.
    :param QuantType activation_type: The quantization type of activation. Default is QuantType.QInt8.
    :param Union[CalibrationMethod, LayerWiseMethod, PowerOfTwoMethod] calibrate_method: Calibration method to use (MinMax, Entropy, Percentile, Distribution, NonOverflow or MinMSE).
    :param bool use_external_data_format: Whether to use external data format for large models.
    :param Union[List[str], None] execution_providers: List of execution providers for ONNX Runtime.
    :param Dict[str, Any] calib_extra_options: Additional options for calibrator configuration.
    """

    if not _is_package_available("matplotlib")[0]:
        raise ImportError(
            "The 'matplotlib' is required but not installed. Please install it via 'pip install matplotlib'."
        )

    with create_tmp_dir("quark_onnx.hist.") as quant_tmp_dir:
        calibrator = create_calibrator_float_scale(
            model_input,
            op_types_to_calibrate,
            augmented_model_path=Path(quant_tmp_dir).joinpath("augmented_model.onnx").as_posix(),
            calibrate_method=calibrate_method,
            use_external_data_format=use_external_data_format,
            execution_providers=execution_providers,
            extra_options=calib_extra_options,
        )

        calibrator.collect_data(calib_data_reader)

        if not hasattr(calibrator, "collector") or not calibrator.collector or not calibrator.collector.histogram_dict:
            logger.warning("This calibrator is not histogram-based, we can not save histogram with that.")
        else:
            hist_tmp_dir = save_tensor_histogram(calibrator)
            logger.info(f"Saved the histogram of tensors using {calibrate_method} to {hist_tmp_dir}.")

        del calibrator


def save_tensors_range(tensors_range: Any, tensors_range_file: None | str) -> None:
    if tensors_range_file is not None:
        tensors_range_dict = {}
        for key in tensors_range.data:
            temp_value = tensors_range.data[key].range_value
            tensors_range_dict[key] = (temp_value[0].tolist(), temp_value[1].tolist())
        with open(tensors_range_file, "w") as json_file:
            json.dump(tensors_range_dict, json_file, indent=2)
    else:
        logger.error("Save the tensors range failed, because no file path provided.")


def load_tensors_range(tensors_range_file: None | str) -> TensorsData | None:
    tensors_range: TensorsData | None = None
    if tensors_range_file is not None:
        assert os.path.exists(tensors_range_file), "The tensors range file does not exist."
        with open(tensors_range_file) as json_file:
            loaded_dict = json.load(json_file)
        tensors_range_dict = {}
        for key in loaded_dict:
            temp_value = loaded_dict[key]
            tensors_range_dict[key] = (
                np.array(temp_value[0], dtype=np.float32),
                np.array(temp_value[1], dtype=np.float32),
            )
        tensors_range = TensorsData(CalibrationMethod.MinMax, tensors_range_dict)
    else:
        logger.error("Load the tensors range failed, because no file path provided.")
    return tensors_range
