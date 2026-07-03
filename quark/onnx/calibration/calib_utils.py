#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod, HistogramCollector, TensorsData
from onnxruntime.quantization.quant_utils import QuantType

from quark.common.utils.import_utils import _is_package_available
from quark.common.utils.log import ScreenLogger
from quark.onnx.utils.system_utils import check_and_create_path, create_tmp_dir

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


def build_tensor_producer_map(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    """
    Build a mapping from tensor names to the ONNX nodes that produce them.

    This function iterates over all nodes in the model graph and records
    which node is responsible for producing each output tensor. The resulting
    mapping enables efficient upstream traversal of the computation graph
    starting from any tensor.

    :param onnx.ModelProto model: The ONNX model whose graph will be analyzed.
    :return: Dict[str, onnx.NodeProto]: A dictionary mapping output tensor names
        to the `NodeProto` that produces each tensor.
    """
    tensor_producer = {}
    for node in model.graph.node:
        for output in node.output:
            tensor_producer[output] = node
    return tensor_producer


def find_nearest_non_passthrough_output(
    tensor_name: str, tensor_producer: dict[str, onnx.NodeProto], passthrough_types: set[str]
) -> str | None:
    """
    Find the nearest upstream tensor produced by a non-passthrough node.

    Starting from the given tensor name, this function walks upstream through
    the computation graph. If the tensor is produced by a passthrough node
    (e.g., Reshape, Transpose), the search continues recursively through
    that node's inputs until a non-passthrough node is found.

    If the tensor has no recorded producer, it is assumed to be a graph input
    or initializer and is returned as-is.

    :param str tensor_name: The name of the tensor to trace upstream from.
    :param Dict[str, onnx.NodeProto] tensor_producer: Mapping from tensor names to the nodes that produce them.
    :param set passthrough_types: Set of ONNX op_type strings that are considered passthrough operations.
    :return: Optional[str]: The name of the nearest upstream tensor produced by a non-passthrough node, or None if no such ancestor can be found.
    """
    if tensor_name not in tensor_producer:
        # Graph input or initializer
        return tensor_name

    node = tensor_producer[tensor_name]

    if node.op_type not in passthrough_types:
        # Found non-passthrough ancestor
        return tensor_name

    # Node is passthrough, continue searching through its inputs
    for input_tensor in node.input:
        ancestor = find_nearest_non_passthrough_output(input_tensor, tensor_producer, passthrough_types)
        if ancestor is not None:
            return ancestor

    return None


def nearest_non_passthrough_ancestor_mapping(
    model: onnx.ModelProto, passthrough_node_types: list[str]
) -> tuple[dict[str, str], list[str]]:
    """
    Compute a mapping from passthrough node outputs to their nearest
    non-passthrough ancestor outputs.

    This function implements the Nearest Non-Passthrough Ancestor Mapping (NNPAM)
    algorithm. For each passthrough node in the model, it determines the closest
    upstream tensor that originates from a non-passthrough operation.

    The resulting mapping can be used for graph simplification, optimization,
    or dependency analysis where passthrough operations should be ignored.

    :param onnx.ModelProto model: The ONNX model to analyze.
    :param List[str] passthrough_node_types: List of ONNX op_type strings that should be treated as passthrough nodes.
    :param set passthrough_types: Set of ONNX op_type strings that are considered passthrough operations.
    :return: Dict[str, str]: A dictionary mapping each passthrough node output tensor name to the tensor name of its nearest non-passthrough ancestor.
    """
    passthrough_types = set(passthrough_node_types)
    tensor_producer = build_tensor_producer_map(model)

    result = {}
    missing_types = []

    for node in model.graph.node:
        if node.op_type not in passthrough_types:
            continue

        for output in node.output:
            ancestor = find_nearest_non_passthrough_output(output, tensor_producer, passthrough_types)
            if ancestor is not None:
                result[output] = ancestor
            if ancestor is None and node.op_type not in missing_types:
                missing_types.append(node.op_type)

    return result, missing_types


def update_tensors_range_with_dependencies(tensors_range: Any, dependencies: dict[str, str]) -> Any:
    """
    Update tensor ranges based on dependency mappings.

    This function propagates tensor range values according to a dependency mapping. If a tensor depends on
    another tensor, its range is replaced with the range of its dependency.

    The function assumes `tensors_range.data` is a mapping from tensor names to objects that expose a
    `range_value` attribute, where `range_value` contains a pair of numeric arrays (e.g., min/max).

    :param Any tensors_range: An object containing tensor range data. It must have a `.data` attribute structured as: Dict[str, TensorRange] where `TensorRange.range_value` is a tuple/list of arrays.
    :param Dict[str, str] dependencies: A mapping from tensor name to dependent tensor name. Example: {"output_tensor": "input_tensor"}
    :return: Any: A new `TensorsData` object with updated tensor ranges, where dependent tensors inherit the range of their source tensors.
    """
    # Convert tensors_range into a simple dict: tensor_name -> (min, max)
    tensors_range_dict: dict[str, Any] = {}

    for tensor_name, tensor_data in tensors_range.data.items():
        min_val, max_val = tensor_data.range_value
        tensors_range_dict[tensor_name] = (
            min_val.tolist(),
            max_val.tolist(),
        )

    # Copy original ranges so only dependent tensors are overridden
    new_tensors_range_dict = copy.deepcopy(tensors_range_dict)

    # Apply dependency-based updates
    for target_tensor, source_tensor in dependencies.items():
        if source_tensor in tensors_range_dict:
            new_tensors_range_dict[target_tensor] = tensors_range_dict[source_tensor]

    for key in new_tensors_range_dict:
        temp_value = new_tensors_range_dict[key]
        new_tensors_range_dict[key] = (
            np.array(temp_value[0], dtype=np.float32),
            np.array(temp_value[1], dtype=np.float32),
        )

    # Reconstruct TensorsData object
    new_tensors_range = TensorsData(
        CalibrationMethod.MinMax,
        new_tensors_range_dict,
    )

    return new_tensors_range
