#!/usr/bin/env python
#
# Modifications copyright(c) 2025 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft, Intel Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------

import copy
import os
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
from joblib import Parallel, delayed  # type: ignore
from onnx import numpy_helper
from onnxruntime.quantization.calibrate import CalibraterBase, CalibrationDataReader, CalibrationMethod, TensorsData
from onnxruntime.quantization.calibrate import HistogramCalibrater as OrtHistogramCalibrater
from onnxruntime.quantization.calibrate import MinMaxCalibrater as OrtMinMaxCalibrater
from onnxruntime.quantization.quant_utils import QuantType
from tqdm import tqdm

from quark.common.utils.log import ScreenLogger, log_errors
from quark.onnx.quantization.quant_utils import ExtendedQuantType, get_qmin_qmax_for_qType, get_tensor_type_from_qType
from quark.onnx.utils.file_utils import save_quantized_info
from quark.onnx.utils.model_utils import create_infer_session_for_onnx_model, sanitize_model_outputs

from .collectors import (
    OverridedHistogramCollector,
    PowOfTwoCollector,
    compute_lwp_metric_from_histogram,
    compute_lwp_scale_zp,
    loading_data_from_disk,
)
from .methods import LayerWiseMethod, PowerOfTwoMethod

logger = ScreenLogger(__name__)


# Per-method calibrator-internal default keys (lowercase, matching what
# create_calibrator_* constructors expect). These mirror the inline `x = default
# if "x" not in extra_options else extra_options["x"]` blocks that used to live
# in create_calibrator_power_of_two / create_calibrator_float_scale.
_CALIB_METHOD_DEFAULTS: dict[Any, dict[str, Any]] = {
    PowerOfTwoMethod.NonOverflow: {
        "symmetric": True,
        "moving_average": False,
        "averaging_constant": 0.01,
        "optimize_mem": True,
    },
    PowerOfTwoMethod.MinMSE: {
        "symmetric": True,
        "minmse_mode": "All",
        "num_bins": 2048,
        "percentile": 99.999,
        "optimize_mem": True,
        "worker_num": 1,
    },
    CalibrationMethod.MinMax: {
        "symmetric": False,
        "moving_average": False,
        "averaging_constant": 0.01,
        "optimize_mem": True,
    },
    CalibrationMethod.Entropy: {
        "num_bins": 128,
        "num_quantized_bins": 128,
        "symmetric": False,
        "optimize_mem": True,
        "worker_num": 1,
    },
    CalibrationMethod.Percentile: {
        "num_bins": 2048,
        "percentile": 99.999,
        "symmetric": True,
        "optimize_mem": True,
        "worker_num": 1,
    },
    CalibrationMethod.Distribution: {
        "num_bins": 2048,
        "scenario": "same",
        "optimize_mem": True,
        "worker_num": 1,
    },
    LayerWiseMethod.LayerWisePercentile: {
        "num_bins": 2048,
        "percentile": 99.999,
        "symmetric": True,
        "worker_num": 1,
        "lwp_metric": "mae",
        "percentile_candidates": [99.99, 99.999, 99.99999],
        "lwp_use_histogram": True,
        # In-memory scoring is the best-performing default for LayerWisePercentile.
        "optimize_mem": False,
    },
}


def resolve_calibrator_extra_defaults(
    calibrate_method: Any,
    extra_options: dict[str, Any],
) -> dict[str, Any]:
    """Return the effective ``{lowercase_key: value}`` overlay for the given
    ``calibrate_method`` based on user-provided ``extra_options`` and built-in
    per-method defaults.

    Single source of truth for calibrator-internal defaults. Used by both the
    calibrator factories in this module and the effective-config summary
    printer in :mod:`quark.onnx.utils.print_utils` so the two never drift.
    """
    defaults = _CALIB_METHOD_DEFAULTS.get(calibrate_method)
    if defaults is None:
        return {}
    return {k: extra_options.get(k, v) for k, v in defaults.items()}


def generate_an_empty_onnx_model(model_path: str) -> None:
    """
    This function is used to generate an empty onnx model based on
    the provided path for the initialization of calibrators.
    """
    graph = onnx.helper.make_graph(name="EmptyGraph", inputs=[], outputs=[], nodes=[])
    model = onnx.helper.make_model(graph, producer_name="empty-model")
    onnx.save(model, model_path)


def caching_data_on_disk(
    session: onnxruntime.InferenceSession, inputs: dict[str, Any], cache_dir: str, append_mode: bool
) -> tuple[list[str], int]:
    """
    Execute a session run and save the outputs to the caching directory.
    This function encapsulates the session run within a function and save the outputs
    to local files, ensuring that memory of output arrays is released.

    :param onnxruntime.InferenceSession session: the session to run
    :param dict[str, Any] inputs: the input data for the run
    :param str cache_dir: the caching directory to store the files
    :param bool append_mode: append the data to existing file or not
    :return: The list of saved file paths and the number of bytes it saved
    """
    outputs = session.run(None, inputs)
    sanitize_model_outputs(outputs)

    file_list: list[str] = []
    cached_nbytes = 0

    for output_index, output in enumerate(outputs):
        dtype = output.dtype.name
        if append_mode:
            # For append mode, the data is stored in float16 to save disk space
            output_fp16 = output.astype(np.float16)
            nbytes = output_fp16.nbytes
            file_path = os.path.join(cache_dir, f"output{output_index}_data_{dtype}.npystream")
            with open(file_path, "ab") as f:
                np.save(f, output_fp16)
        else:
            nbytes = output.nbytes
            file_path = os.path.join(cache_dir, f"output{output_index}_data_{dtype}.npy")
            with open(file_path, "wb") as f:
                np.save(f, output)

        file_list.append(file_path)
        cached_nbytes += nbytes

    return file_list, cached_nbytes


def get_clean_merged_dict(
    intermediate_outputs: list[list[Any]], output_names: list[str], tensors_to_calibrate: list[str] | None
) -> dict[str, list[list[Any]]]:
    """
    Based on the tensor_to_calibrate list, filter out tensors from output_names
    that do not require calibration to form the clean dictionary. In this dict,
    the keys are tensor names, and the values come from intermediate_outputs.

    :param list[Any] intermediate_outputs: the intermediate outputs
    :param list[str] output_names: the list of output names
    :param list[str] | None tensors_to_calibrate: the list of tensors to calibrate
    :return: the cleaned and merged dictionary
    """

    output_dicts_list = [
        dict(zip(output_names, intermediate_output, strict=False)) for intermediate_output in intermediate_outputs
    ]

    merged_dict: dict[str, Any] = {}
    for d in output_dicts_list:
        for k, v in d.items():
            merged_dict.setdefault(k, []).append(v)

    clean_merged_dict: dict[str, Any] = merged_dict
    if tensors_to_calibrate is not None:
        clean_merged_dict = {i: merged_dict[i] for i in merged_dict if i in tensors_to_calibrate}

    return clean_merged_dict


class OverridedMinMaxCalibrater(OrtMinMaxCalibrater):  # type: ignore
    """
    This class is used to override ORT official Calibrater to prevent
    saving the augmented model to disk if the model size is less than 2GB.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.
    :param Optional[Sequence[str]] op_types_to_calibrate: List of operator types to calibrate. Defaults to ``None``, which indicates that all float32/float16 tensors are calibrated.
    :param str augmented_model_path: save augmented model to this path.
    :param bool symmetric: make range of tensor symmetric (central point is 0).
    :param bool use_external_data_format: use external data format to store model which size is >= 2Gb
    :param bool moving_average: compute the moving average of the minimum and maximum values instead of the global minimum and maximum.
    :param float averaging_constant: constant smoothing factor to use when computing the moving average.
    :param Optional[int] max_intermediate_outputs: maximum number of intermediate outputs before an intermediate range is computed.
    """

    def __init__(
        self,
        model_input: str | Path | onnx.ModelProto,
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path: str = "augmented_model.onnx",
        symmetric: bool = False,
        use_external_data_format: bool = False,
        moving_average: bool = False,
        averaging_constant: float = 0.01,
        max_intermediate_outputs: int | None = None,
        optimize_mem: bool = True,
    ):
        if isinstance(model_input, onnx.ModelProto):
            generate_an_empty_onnx_model(augmented_model_path)
            model_path = augmented_model_path  # Generate an empty model for the base class to load
        else:
            model_path = model_input.as_posix() if isinstance(model_input, Path) else model_input

        super().__init__(
            model_path,
            op_types_to_calibrate=op_types_to_calibrate,
            augmented_model_path=augmented_model_path,
            symmetric=symmetric,
            use_external_data_format=use_external_data_format,
            moving_average=moving_average,
            averaging_constant=averaging_constant,
            max_intermediate_outputs=max_intermediate_outputs,
        )

        if isinstance(model_input, onnx.ModelProto):
            # Replace the empty model with the real input model.
            # The copy is to avoid modifying the input model.
            self.model = copy.deepcopy(model_input)

        self.optimize_mem = optimize_mem

    def augment_graph(self) -> None:
        """
        Adds ReduceMin and ReduceMax nodes to all quantization_candidates op type nodes in
        model and ensures their outputs are stored as part of the graph output

        :return: augmented ONNX model
        """
        tensors, _ = self.select_tensors_to_calibrate(self.model)
        reshape_shape_name = str(uuid.uuid4())
        reshape_shape = numpy_helper.from_array(np.array([1], dtype=np.int64), reshape_shape_name)
        self.model.graph.initializer.append(reshape_shape)

        def add_reduce_min_max(tensor_name: str, reduce_op_name: str) -> None:
            # When doing ReduceMax/ReduceMin, ORT can't reduce on dim with value of 0 if 'keepdims' is false.
            # To make the code simple, we always let keepdims to be 1.
            keepdims = 1

            # Adding ReduceMin/ReduceMax nodes: ReduceMin/ReduceMax -> Reshape-> (output)
            reduce_output = tensor_name + "_" + reduce_op_name
            intermediate_output = reduce_output + "_Reshape"
            reduce_node = onnx.helper.make_node(
                reduce_op_name, [tensor_name], [intermediate_output], keepdims=keepdims, name=reduce_output
            )

            reshape_node = onnx.helper.make_node(
                "Reshape",
                inputs=[intermediate_output, reshape_shape_name],
                outputs=[reduce_output],
                name=intermediate_output,
            )

            self.model.graph.node.extend([reduce_node, reshape_node])
            value_infos = {vi.name: vi for vi in self.model.graph.value_info}
            value_infos.update({o.name: o for o in self.model.graph.output})
            value_infos.update({i.name: i for i in self.model.graph.input})
            if tensor_name in value_infos:
                onnx_type = value_infos[tensor_name].type.tensor_type.elem_type
            else:
                raise ValueError(
                    f"Unable to guess tensor type for tensor {tensor_name!r}, "
                    f"running shape inference before quantization may resolve this issue."
                )
            self.model.graph.output.append(onnx.helper.make_tensor_value_info(reduce_output, onnx_type, [1]))

        for tensor in tensors:
            add_reduce_min_max(tensor, "ReduceMin")
            add_reduce_min_max(tensor, "ReduceMax")

        if self.use_external_data_format:
            onnx.save(
                self.model,
                self.augmented_model_path,
                save_as_external_data=self.use_external_data_format,
            )

    def collect_data(self, data_reader: CalibrationDataReader) -> None:
        """
        Collect intermediate outputs from the model using the provided data_reader,
        and prepare data for calibration.
        """
        try:
            data_size = len(data_reader)
        except NotImplementedError as e:  # pragma: no cover
            raise ValueError("The data reader should implement the '__len__' method to provide the data size.") from e

        for _ in tqdm(range(data_size)):
            inputs = data_reader.get_next()
            if not inputs:
                break
            self.intermediate_outputs.append(self.infer_session.run(None, inputs))
            if (
                self.max_intermediate_outputs is not None
                and len(self.intermediate_outputs) == self.max_intermediate_outputs
            ):
                self.clear_collected_data()

        if not self.intermediate_outputs and self.calibrate_tensors_range is None:
            raise ValueError("No data is collected.")

        sanitize_model_outputs(self.intermediate_outputs[0])

        t = self.compute_data()
        if not isinstance(t, TensorsData):
            raise TypeError(f"compute_data must return a TensorsData not {type(t)}.")
        self.clear_collected_data()

    def create_inference_session(self) -> None:
        """
        create an OnnxRuntime InferenceSession.
        """
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        if self.optimize_mem and self.execution_providers == ["CPUExecutionProvider"]:
            sess_options.enable_cpu_mem_arena = False

        if self.use_external_data_format:
            self.infer_session = create_infer_session_for_onnx_model(
                self.augmented_model_path,
                sess_options=sess_options,
                providers=self.execution_providers,
                use_external_data_format=self.use_external_data_format,
            )
        else:
            self.infer_session = create_infer_session_for_onnx_model(
                self.model,
                sess_options=sess_options,
                providers=self.execution_providers,
            )


class OverridedHistogramCalibrater(OrtHistogramCalibrater):  # type: ignore
    """
    This class is used to override ORT official Calibrater to optimize memory usage,
    it also has a processing bar when collecting data.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.
    :param Optional[Sequence[str]] op_types_to_calibrate: List of operator types to calibrate. Defaults to ``None``, which indicates that all float32/float16 tensors are calibrated.
    :param str augmented_model_path: save augmented model to this path.
    :param bool use_external_data_format: use external data format to store model which size is >= 2Gb
    :param str method: A string. One of ['entropy', 'percentile', 'distribution'].
    :param bool symmetric: make range of tensor symmetric (central point is 0).
    :param int num_bins: number of bins to create a new histogram for collecting tensor values.
    :param int num_quantized_bins: number of quantized bins. Default 128.
    :param float percentile: A float number between [0, 100]. Default 99.99.
    :param str scenario: for float 8 only, if ``scenario="same"``,
        the algorithm weights and float 8 follow the same distribution,
        if ``scenario="p3"``, it assumes the weights follow
        a gaussian law and float 8 ~ X^3 where X is a gaussian law. Defaults to ``"same"``.
    :param bool optimize_mem: Whether to optimize memory consumption. Default is True.
    :param int worker_num: Number of workers to do the data collection. Default is 1.
    """

    def __init__(
        self,
        model_input: str | Path | onnx.ModelProto,
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path: str = "augmented_model.onnx",
        use_external_data_format: bool = False,
        method: str = "percentile",
        symmetric: bool = False,
        num_bins: int = 128,
        num_quantized_bins: int = 2048,
        percentile: float = 99.999,
        scenario: str = "same",
        optimize_mem: bool = True,
        worker_num: int = 1,
    ):
        if isinstance(model_input, onnx.ModelProto):
            generate_an_empty_onnx_model(augmented_model_path)
            model_path = augmented_model_path  # Generate an empty model for the base class to load
        else:
            model_path = model_input.as_posix() if isinstance(model_input, Path) else model_input

        super().__init__(
            model_path,
            op_types_to_calibrate=op_types_to_calibrate,
            augmented_model_path=augmented_model_path,
            use_external_data_format=use_external_data_format,
            method=method,
            symmetric=symmetric,
            num_bins=num_bins,
            num_quantized_bins=num_quantized_bins,
            percentile=percentile,
            scenario=scenario,
        )

        if isinstance(model_input, onnx.ModelProto):
            # Replace the empty model with the real input model.
            # The copy is to avoid modifying the input model.
            self.model = copy.deepcopy(model_input)

        self.clean_merged_dict: dict[str, list[list[Any]]] = {}  # Only for layerwise percentile

        self.optimize_mem = optimize_mem
        self.worker_num = worker_num

    def augment_graph(self) -> None:
        """
        make all quantization_candidates op type nodes as part of the graph output.

        :return: augmented ONNX model
        """
        self.tensors_to_calibrate, value_infos = self.select_tensors_to_calibrate(self.model)
        for tensor in self.tensors_to_calibrate:
            if tensor not in self.model_original_outputs:
                self.model.graph.output.append(value_infos[tensor])

        if self.use_external_data_format:
            onnx.save(
                self.model,
                self.augmented_model_path,
                save_as_external_data=self.use_external_data_format,
            )

    def create_inference_session(self) -> None:
        """
        create an OnnxRuntime InferenceSession.
        """
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        if self.use_external_data_format:
            self.infer_session = create_infer_session_for_onnx_model(
                self.augmented_model_path,
                sess_options=sess_options,
                providers=self.execution_providers,
                use_external_data_format=self.use_external_data_format,
            )
        else:
            self.infer_session = create_infer_session_for_onnx_model(
                self.model,
                sess_options=sess_options,
                providers=self.execution_providers,
            )

    def collect_data(self, data_reader: CalibrationDataReader, layer_wise: bool = False) -> None:
        """
        Calibrator collects activation tensors.
        """

        # Initialize the collector
        if not self.collector:  # type: ignore
            self.collector = OverridedHistogramCollector(
                method=self.method,
                symmetric=self.symmetric,
                num_bins=self.num_bins,
                num_quantized_bins=self.num_quantized_bins,
                percentile=self.percentile,
                scenario=self.scenario,
                optimize_mem=self.optimize_mem,
                worker_num=self.worker_num,
                layer_wise=layer_wise,
            )

        input_names_set = {node_arg.name for node_arg in self.infer_session.get_inputs()}
        output_names = [node_arg.name for node_arg in self.infer_session.get_outputs()]

        try:
            data_size = len(data_reader)
        except NotImplementedError as e:  # pragma: no cover
            raise ValueError("The data reader should implement the '__len__' method to provide the data size.") from e

        cache_dir = os.path.dirname(self.augmented_model_path)  # For caching tensors
        cache_capacity = 0

        onnx_infer_time = []
        numpy_stat_time = []

        pbar = tqdm(range(data_size))
        for _ in pbar:
            collect_data_start_time = time.perf_counter()
            inputs = data_reader.get_next()
            if not inputs:
                break

            fixed_outputs: list[str | np.ndarray[Any, Any]] = []

            if self.optimize_mem:
                append_mode = self.collector.layerwise_percentile
                fixed_outputs, cached_nbytes = caching_data_on_disk(self.infer_session, inputs, cache_dir, append_mode)  # type: ignore
            else:
                outputs = self.infer_session.run(None, inputs)
                sanitize_model_outputs(outputs)

                cached_nbytes = 0
                for output_index, output in enumerate(outputs):
                    # Copy np.ndarray only for graph outputs that are also graph inputs to workaround bug:
                    # https://github.com/microsoft/onnxruntime/issues/21922
                    if output_names[output_index] in input_names_set:
                        fixed_outputs.append(copy.copy(output))
                    else:
                        fixed_outputs.append(output)
                    cached_nbytes += output.nbytes

            collect_data_onnx_infer_time = time.perf_counter()

            if layer_wise:
                self.intermediate_outputs.append(fixed_outputs)
                cache_capacity += cached_nbytes
            else:
                cache_capacity = cached_nbytes

            collect_data_acquired_data_time = time.perf_counter()
            if len(self.intermediate_outputs) and self.optimize_mem:
                # The purpose of using self.intermediate_outputs to get clean merged dict is
                # to enable layerwise percentile to load the last array from caching files
                clean_merged_dict = get_clean_merged_dict(
                    self.intermediate_outputs, output_names, self.tensors_to_calibrate
                )
            else:
                clean_merged_dict = get_clean_merged_dict([fixed_outputs], output_names, self.tensors_to_calibrate)
            self.collector.collect(clean_merged_dict)
            # Free this batch's activations early to lower peak memory.
            if not self.optimize_mem and not layer_wise:
                del clean_merged_dict, fixed_outputs, outputs
            collect_data_end_time = time.perf_counter()

            onnx_infer_time.append(collect_data_onnx_infer_time - collect_data_start_time)
            numpy_stat_time.append(collect_data_end_time - collect_data_acquired_data_time)

            pbar.set_description(
                f"Cached {cache_capacity / (1024**3):.2f}GB on {'disk' if self.optimize_mem else 'memory'}"
            )

        onnx_infer_time_sum = np.sum(onnx_infer_time)
        numpy_stat_time_sum = np.sum(numpy_stat_time)
        save_quantized_info(
            [
                ["", "", "calibration collect data (onnx inference)", onnx_infer_time_sum],
                ["", "", "calibration collect data (numpy statistics)", numpy_stat_time_sum],
            ]
        )
        logger.info(
            f"Quark_latency_profiler: calibration collect data (onnx inference) time consumed: {onnx_infer_time_sum:1f}s"
        )
        logger.info(
            f"Quark_latency_profiler: calibration collect data (numpy statistics) time consumed: {numpy_stat_time_sum:1f}"
        )

        if self.collector.layerwise_percentile:
            if len(self.intermediate_outputs) == 0:
                raise ValueError("No data is collected.")

            # This clean merged dict is used for subsequent choosing optimal percentile
            self.clean_merged_dict = get_clean_merged_dict(
                self.intermediate_outputs, output_names, self.tensors_to_calibrate
            )

        self.clear_collected_data()

    def compute_data(self) -> TensorsData:
        """
        Compute the min-max range of tensor

        :return: dictionary mapping: {tensor name: (min value, max value)}
        """
        if not self.collector:
            raise ValueError("No collector created and can't generate calibration data.")

        if isinstance(self, EntropyCalibrater):
            cal = CalibrationMethod.Entropy
        elif isinstance(self, PercentileCalibrater):
            cal = CalibrationMethod.Percentile
        elif isinstance(self, DistributionCalibrater):
            cal = CalibrationMethod.Distribution
        else:
            raise TypeError(f"Unknown calibrater {type(self)}. This method must be overwritten.")
        return TensorsData(cal, self.collector.compute_collection_result())


class MinMaxCalibrater(OverridedMinMaxCalibrater):
    """
    This method obtains the quantization parameters based on the minimum and maximum values of each tensor.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.
    :param Optional[Sequence[str]] op_types_to_calibrate: List of operator types to calibrate. Defaults to ``None``, which indicates that all float32/float16 tensors are calibrated.
    :param str augmented_model_path: Path to save the augmented model. Default is ``"augmented_model.onnx"``.
    :param bool symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is ``False``.
    :param bool use_external_data_format: Whether to use external data format to store model which size is >= 2GB. Default is ``False``.
    :param bool moving_average: Whether to compute the moving average of the minimum and maximum values instead of the global minimum and maximum. Default is ``False``.
    :param float averaging_constant: Constant smoothing factor to use when computing the moving average. Default is ``0.01``. Should be between 0 and 1.
    :raises ValueError: If averaging_constant is not between 0 and 1 when moving_average is True.
    """

    def __init__(
        self,
        model_input: str | Path | onnx.ModelProto,
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path: str = "augmented_model.onnx",
        symmetric: bool = False,
        use_external_data_format: bool = False,
        moving_average: bool = False,
        averaging_constant: float = 0.01,
        optimize_mem: bool = True,
    ) -> None:
        super().__init__(
            model_input,
            op_types_to_calibrate=op_types_to_calibrate,
            augmented_model_path=augmented_model_path,
            symmetric=symmetric,
            use_external_data_format=use_external_data_format,
            moving_average=moving_average,
            averaging_constant=averaging_constant,
            optimize_mem=optimize_mem,
        )
        self.intermediate_outputs: list[str] = []
        self.calibrate_tensors_range = None
        self.num_model_outputs = len(self.model.graph.output)
        self.model_original_outputs = {output.name for output in self.model.graph.output}
        self.moving_average = moving_average
        if moving_average and (averaging_constant < 0 or averaging_constant > 1):
            raise ValueError("Invalid averaging constant, which should not be < 0 or > 1.")
        self.averaging_constant = averaging_constant


class EntropyCalibrater(OverridedHistogramCalibrater):
    """
    This method determines the quantization parameters by considering the entropy algorithm of each tensor's distribution.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.
    :param Optional[Sequence[str]] op_types_to_calibrate: List of operator types to calibrate. Defaults to ``None``, which indicates that all float32/float16 tensors are calibrated.
    :param str augmented_model_path: Path to save the augmented model. Default is ``"augmented_model.onnx"``.
    :param bool use_external_data_format: Whether to use external data format to store model which size is >= 2GB. Default is ``False``.
    :param str method: Method for calibration. One of ['entropy', 'percentile', 'distribution']. Default is ``"entropy"``.
    :param bool symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is ``False``.
    :param int num_bins: Number of bins to create a new histogram for collecting tensor values. Default is ``128``.
    :param int num_quantized_bins: Number of quantized bins. Default is ``128``.
    :param bool optimize_mem: Whether to optimize memory consumption. Default is True.
    :param int worker_num: Number of workers to do the data collection. Default is 1.
    """

    def __init__(
        self,
        model_input: str | Path | onnx.ModelProto,
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path: str = "augmented_model.onnx",
        use_external_data_format: bool = False,
        method: str = "entropy",
        symmetric: bool = False,
        num_bins: int = 128,
        num_quantized_bins: int = 128,
        optimize_mem: bool = True,
        worker_num: int = 1,
    ) -> None:
        super().__init__(
            model_input,
            op_types_to_calibrate=op_types_to_calibrate,
            augmented_model_path=augmented_model_path,
            use_external_data_format=use_external_data_format,
            method=method,
            symmetric=symmetric,
            num_bins=num_bins,
            num_quantized_bins=num_quantized_bins,
            optimize_mem=optimize_mem,
            worker_num=worker_num,
        )
        self.collector: Any = None


class PercentileCalibrater(OverridedHistogramCalibrater):
    """
    This method calculates quantization parameters using percentiles of the tensor values.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.
    :param Optional[Sequence[str]] op_types_to_calibrate: List of operator types to calibrate. Defaults to ``None``, which indicates that all float32/float16 tensors are calibrated.
    :param str augmented_model_path: Path to save the augmented model. Default is ``"augmented_model.onnx"``.
    :param bool use_external_data_format: Whether to use external data format to store model which size is >= 2GB. Default is ``False``.
    :param str method: Method for calibration. One of ``"entropy"``, ``"percentile"`` or ``"distribution"``. Default is ``"percentile"``.
    :param bool symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is ``False``.
    :param int num_bins: Number of bins to create a new histogram for collecting tensor values. Default is ``2048``.
    :param float percentile: Percentile value for calibration, a float between [0, 100]. Default is ``99.999``.
    :param bool optimize_mem: Whether to optimize memory consumption. Default is True.
    :param int worker_num: Number of workers to do the data collection. Default is 1.
    """

    def __init__(
        self,
        model_input: str | Path | onnx.ModelProto,
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path: str = "augmented_model.onnx",
        use_external_data_format: bool = False,
        method: str = "percentile",
        symmetric: bool = False,
        num_bins: int = 2048,
        percentile: float = 99.999,
        optimize_mem: bool = True,
        worker_num: int = 1,
    ):
        super().__init__(
            model_input,
            op_types_to_calibrate=op_types_to_calibrate,
            augmented_model_path=augmented_model_path,
            use_external_data_format=use_external_data_format,
            method=method,
            symmetric=symmetric,
            num_bins=num_bins,
            percentile=percentile,
            optimize_mem=optimize_mem,
            worker_num=worker_num,
        )
        self.collector: Any = None


class DistributionCalibrater(OverridedHistogramCalibrater):
    """
    This method calculates quantization parameters according to distribution of the tensor values.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.
    :param Optional[Sequence[str]] op_types_to_calibrate: List of operator types to calibrate. Defaults to ``None``, which indicates that all float32/float16 tensors are calibrated.
    :param str augmented_model_path: save augmented model to this path. Defaults to ``"augmented_model.onnx"``.
    :param bool use_external_data_format: use external data format to store model which size is >= 2Gb. Defaults to ``False``.
    :param str method: One of ['entropy', 'percentile', 'distribution']. Defaults to ``"distribution"``.
    :param int num_bins: number of bins to create a new histogram for collecting tensor values. Defaults to ``128``.
    :param str scenario: for float 8 only, if ``scenario="same"``,
        the algorithm weights and float 8 follow the same distribution,
        if ``scenario="p3"``, it assumes the weights follow
        a gaussian law and float 8 ~ X^3 where X is a gaussian law. Defaults to ``"same"``.
    :param bool optimize_mem: Whether to optimize memory consumption. Default is True.
    :param int worker_num: Number of workers to do the data collection. Default is 1.
    """

    def __init__(
        self,
        model_input: str | Path | onnx.ModelProto,
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path: str = "augmented_model.onnx",
        use_external_data_format: bool = False,
        method: str = "distribution",
        num_bins: int = 128,
        scenario: str = "same",
        optimize_mem: bool = True,
        worker_num: int = 1,
    ):
        super().__init__(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format,
            method=method,
            num_bins=num_bins,
            scenario=scenario,
            optimize_mem=optimize_mem,
            worker_num=worker_num,
        )
        self.collector: Any = None


class PowOfTwoCalibrater(CalibraterBase):  # type: ignore
    """
    This method get the power-of-two quantize parameters for each tensor to minimize the mean-square-loss of quantized values and float values.
    This takes longer time but usually gets better accuracy.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.
    :param Optional[Sequence[str]] op_types_to_calibrate: List of operator types to calibrate. Defaults to ``None``, which indicates that all float32/float16 tensors are calibrated.
    :param str augmented_model_path: Path to save the augmented model. Default is ``"augmented_model.onnx"``.
    :param bool use_external_data_format: Whether to use external data format to store model which size is >= 2GB. Default is ``False``.
    :param Union[QuantType, ExtendedQuantType] activation_type: Type of quantization for activations. Default is ``QuantType.QInt8``.
    :param PowerOfTwoMethod method: Calibration method. Default is ``PowerOfTwoMethod.MinMSE``.
    :param bool symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is ``True``.
    :param str minmse_mode: Mode for the MinMSE method. Default is ``"All"``.
    :param int num_bins: Number of histogram bins for histogram-based MSE computation. Default is 2048.
    :param float percentile: Percentile value for calibration, a float between 0 and 100. Default is ``99.999``.
    :param bool optimize_mem: Whether to optimize memory consumption. Default is True.
    :param int worker_num: Number of workers to do the data collection. Default is 1.
    :param Dict[Any, Any] quantized_tensor_type: Dictionary specifying the quantized tensor type. Default is ``{}``.
    """

    def __init__(
        self,
        model_input: str | Path | onnx.ModelProto,
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path: str = "augmented_model.onnx",
        use_external_data_format: bool = False,
        activation_type: QuantType | ExtendedQuantType = QuantType.QInt8,
        method: PowerOfTwoMethod = PowerOfTwoMethod.MinMSE,
        symmetric: bool = True,
        minmse_mode: str = "All",
        num_bins: int = 2048,
        percentile: float = 99.999,
        optimize_mem: bool = True,
        worker_num: int = 1,
        quantized_tensor_type: dict[Any, Any] = {},
    ) -> None:
        if isinstance(model_input, onnx.ModelProto):
            generate_an_empty_onnx_model(augmented_model_path)
            model_path = augmented_model_path  # Generate an empty model for the base class to load
        else:
            model_path = model_input.as_posix() if isinstance(model_input, Path) else model_input

        super().__init__(model_path, op_types_to_calibrate, augmented_model_path, symmetric, use_external_data_format)

        if isinstance(model_input, onnx.ModelProto):
            # Replace the empty model with the real input model.
            # The copy is to avoid modifying the input model.
            self.model = copy.deepcopy(model_input)

        self.intermediate_outputs: list[Any] = []
        self.calibrate_tensors_range = None
        self.num_model_outputs = len(self.model.graph.output)
        self.model_original_outputs = {output.name for output in self.model.graph.output}
        self.collector: PowOfTwoCollector | None = None
        self.method = method
        self.symmetric = symmetric
        self.tensors_to_calibrate = None
        self.activation_type = activation_type
        self.use_external_data_format = use_external_data_format
        self.minmse_mode = minmse_mode
        self.num_bins = num_bins
        self.percentile = percentile
        self.optimize_mem = optimize_mem
        self.worker_num = worker_num
        self.quantized_tensor_type = quantized_tensor_type

    def augment_graph(self) -> None:
        """
        make all quantization_candidates op type nodes as part of the graph output.

        :return: augmented ONNX model
        """
        self.tensors_to_calibrate, value_infos = self.select_tensors_to_calibrate(self.model)
        if self.tensors_to_calibrate is not None:
            for tensor in self.tensors_to_calibrate:
                if tensor not in self.model_original_outputs:
                    self.model.graph.output.append(value_infos[tensor])

        if self.use_external_data_format:
            onnx.save(
                self.model,
                self.augmented_model_path,
                save_as_external_data=self.use_external_data_format,
            )

    def clear_collected_data(self) -> None:
        self.intermediate_outputs = []

    def collect_data(self, data_reader: CalibrationDataReader) -> None:
        """
        MinMSE Calibrator collects operators' tensors.
        """

        if not self.collector:
            self.collector = PowOfTwoCollector(
                activation_type=self.activation_type,
                method=self.method,
                symmetric=self.symmetric,
                minmse_mode=self.minmse_mode,
                num_bins=self.num_bins,
                percentile=self.percentile,
                optimize_mem=self.optimize_mem,
                worker_num=self.worker_num,
                quantized_tensor_type=self.quantized_tensor_type,
            )

        input_names_set = {node_arg.name for node_arg in self.infer_session.get_inputs()}
        output_names = [node_arg.name for node_arg in self.infer_session.get_outputs()]

        try:
            data_size = len(data_reader)
        except NotImplementedError as e:  # pragma: no cover
            raise ValueError("The data reader should implement the '__len__' method to provide the data size.") from e

        cache_dir = os.path.dirname(self.augmented_model_path)  # For caching tensors
        cache_capacity = 0

        per_batch_process = self.collector.all_with_histogram or self.collector.mostcommon_minmse

        pbar = tqdm(range(data_size))
        for _ in pbar:
            inputs = data_reader.get_next()
            if not inputs:
                break

            fixed_outputs: list[str | np.ndarray[Any, Any]] = []

            if self.optimize_mem:
                fixed_outputs, cached_nbytes = caching_data_on_disk(  # type: ignore
                    self.infer_session, inputs, cache_dir, not per_batch_process
                )
            else:
                outputs = self.infer_session.run(None, inputs)
                sanitize_model_outputs(outputs)
                cached_nbytes = 0
                for output_index, output in enumerate(outputs):
                    # Copy np.ndarray only for graph outputs that are also graph inputs to workaround bug:
                    # https://github.com/microsoft/onnxruntime/issues/21922
                    if output_names[output_index] in input_names_set:
                        fixed_outputs.append(copy.copy(output))
                    else:
                        fixed_outputs.append(output)
                    cached_nbytes += output.nbytes

            if not per_batch_process:
                self.intermediate_outputs.append(fixed_outputs)
                cache_capacity += cached_nbytes
            else:
                clean_merged_dict = get_clean_merged_dict([fixed_outputs], output_names, self.tensors_to_calibrate)
                self.collector.collect(clean_merged_dict)
                cache_capacity = cached_nbytes

            pbar.set_description(
                f"Cached {cache_capacity / (1024**3):.2f}GB on {'disk' if self.optimize_mem else 'memory'}"
            )

        # To prevent the accumulation of RSS during subsequent computations,
        # explicitly release the session that will no longer be used
        del self.infer_session

        if per_batch_process:
            # For most common and histogram modes, data is processed per-batch;
            # no need to call self.clear_collected_data() and just return directly
            return None

        if len(self.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        clean_merged_dict = get_clean_merged_dict(self.intermediate_outputs, output_names, self.tensors_to_calibrate)

        self.collector.collect(clean_merged_dict)

        self.clear_collected_data()

    def compute_data(self) -> TensorsData:
        """
        Compute the min-max range of tensor

        :return: dictionary mapping: {tensor name: (min value, max value)}
        """
        if not self.collector:
            raise ValueError("No collector created and can't generate calibration data.")

        cal = CalibrationMethod.MinMax
        return TensorsData(cal, self.collector.compute_collection_result())

    def create_inference_session(self) -> None:
        """
        create an OnnxRuntime InferenceSession.
        """
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        if self.use_external_data_format:
            self.infer_session = create_infer_session_for_onnx_model(
                self.augmented_model_path,
                sess_options=sess_options,
                providers=self.execution_providers,
                use_external_data_format=self.use_external_data_format,
            )
        else:
            self.infer_session = create_infer_session_for_onnx_model(
                self.model,
                sess_options=sess_options,
                providers=self.execution_providers,
            )


class LayerWisePercentileCalibrater(PercentileCalibrater):
    """
    This class extends the PercentileCalibrater to support layerwise calibration,
    which typically improves accuracy.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.
    :param Optional[Sequence[str]] op_types_to_calibrate: List of operator types to calibrate. Defaults to ``None``, which indicates that all float32/float16 tensors are calibrated.
    :param str augmented_model_path: save augmented model to this path.
    :param bool use_external_data_format: use external data format to store model which size is >= 2Gb
    :param str method: A string. One of ['entropy', 'percentile', 'distribution'].
    :param bool symmetric: make range of tensor symmetric (central point is 0).
    :param int num_bins: number of quantized bins. Default 128.
    :param float percentile: A float number between [0, 100]. Default 99.99.
    :param int worker_num: Number of workers to do the data collection. Default is 1.
    :param str lwp_mtric: A str value which is use to judge the percentile's metric. One of ['mae', 'mse']. Defaults to ``"mae"``.
    :param int activation_type: Bitwidth setting for activations.QuantType.QInt8.
    :param List[float] percentile_candidates: Percentile candidates. Defaults to ``[99.99, 99.999, 99.99999]``.
    :param bool lwp_use_histogram: Score the optimal percentile from each tensor's histogram. Default is ``True``.
    :param bool optimize_mem: Keep activations in memory (``False``) or cache to disk (``True``) during scoring. Default is ``False``.
    """

    def __init__(
        self,
        model_input: str | Path | onnx.ModelProto,
        op_types_to_calibrate: Sequence[str] | None = None,
        augmented_model_path: str = "augmented_model.onnx",
        use_external_data_format: bool = False,
        method: str = "percentile",
        symmetric: bool = False,
        num_bins: int = 2048,
        percentile: float = 99.999,
        worker_num: int = 1,
        lwp_metric: str = "mae",
        activation_type: QuantType | ExtendedQuantType = QuantType.QInt8,
        percentile_candidates: list[float] = [99.99, 99.999, 99.99999],
        lwp_use_histogram: bool = True,
        optimize_mem: bool = True,
    ):
        super().__init__(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format,
            method=method,
            symmetric=symmetric,
            num_bins=num_bins,
            percentile=percentile,
            optimize_mem=optimize_mem,
            worker_num=worker_num,
        )
        self.minmax_dict: dict[str, float] = {}
        self.lwp_metric = lwp_metric
        self.activation_qType = get_tensor_type_from_qType(activation_type)
        self.q_min, self.q_max = get_qmin_qmax_for_qType(self.activation_qType, reduce_range=False)
        self.percentile_candidates = percentile_candidates
        self.lwp_use_histogram = lwp_use_histogram

    def cal_one_layer_metric(self, input_tensor: list[Any], temp_scale: float, temp_zp: int) -> float:
        """
        Compute the quantization error metric for a single layer tensor.

        The procedure is:
            1. Quantize:
                q = round(input_tensor / temp_scale - temp_zp)
                q = clip(q, self.q_min, self.q_max)
            2. Dequantize:
                dq = (q + temp_zp) * temp_scale
            3. Compute error between input_tensor and dq.

        :param list[Any] input_tensor: Input tensor values to evaluate. Must be broadcast-compatible with NumPy operations.
        :param float temp_scale: Quantization scale factor.
        :param int temp_zp: Quantization zero-point.

        :return float: The computed quantization error for the tensor.
        """

        buffer = np.empty_like(input_tensor)

        # quantize tensor
        np.divide(input_tensor, temp_scale, out=buffer)
        buffer -= temp_zp
        np.round(buffer, out=buffer)
        np.clip(buffer, self.q_min, self.q_max, out=buffer)

        # dequantize
        buffer += temp_zp
        buffer *= temp_scale

        chunk_diff = input_tensor - buffer
        if self.lwp_metric == "mse":
            temp_metric = float(np.mean(chunk_diff * chunk_diff))
        elif self.lwp_metric == "mae":
            temp_metric = float(np.mean(np.abs(chunk_diff)))
        else:
            raise ValueError(f"Unknown lwp_metric {self.lwp_metric!r}. Expected 'mae' or 'mse'.")

        return temp_metric

    def collect_data(
        self,
        data_reader: CalibrationDataReader,
        layer_wise: bool = False,
    ) -> None:
        use_raw_data = not self.lwp_use_histogram
        super().collect_data(data_reader, layer_wise=use_raw_data)
        del self.infer_session
        if use_raw_data:
            # clean_merged_dict is only populated on the raw path.
            assert self.clean_merged_dict, "No data for the layerwise percentile"

        percentile_tensor_minmax = []
        for temp_percentile in self.percentile_candidates:
            self.collector.percentile = temp_percentile
            percentile_tensor_minmax.append(self.collector.compute_percentile())

        if use_raw_data:
            self.compute_data_from_raw(percentile_tensor_minmax)
        else:
            self.compute_data_from_histogram(percentile_tensor_minmax)

    def compute_data_from_raw(self, percentile_tensor_minmax: list[Any]) -> None:
        def cal_layers_minmax(tensor_name: str) -> None:
            data_arr = self.clean_merged_dict[tensor_name]
            assert isinstance(data_arr, list)
            metrics = np.zeros(len(percentile_tensor_minmax))

            if self.worker_num <= 1:
                for chunk_idx in range(len(data_arr)):
                    data_list = loading_data_from_disk(data_arr, chunk_idx, chunk_idx + 1)
                    temp_tensor = np.asarray(data_list, dtype=np.float32).reshape(-1)
                    for percentile_idx in range(len(percentile_tensor_minmax)):
                        candidate = percentile_tensor_minmax[percentile_idx][tensor_name]
                        temp_scale, temp_zp = compute_lwp_scale_zp(candidate[0], candidate[1], self.q_min, self.q_max)
                        metrics[percentile_idx] += self.cal_one_layer_metric(temp_tensor, temp_scale, temp_zp)
            else:
                data_list = loading_data_from_disk(data_arr, 0, len(data_arr))
                temp_tensor = np.asarray(data_list, dtype=np.float32).reshape(-1)
                for percentile_idx in range(len(percentile_tensor_minmax)):
                    candidate = percentile_tensor_minmax[percentile_idx][tensor_name]
                    temp_scale, temp_zp = compute_lwp_scale_zp(candidate[0], candidate[1], self.q_min, self.q_max)
                    metrics[percentile_idx] += self.cal_one_layer_metric(temp_tensor, temp_scale, temp_zp)

            min_metric_index = np.argmin(metrics)
            self.minmax_dict[tensor_name] = percentile_tensor_minmax[min_metric_index][tensor_name]

        if self.worker_num > 1:
            Parallel(n_jobs=self.worker_num, backend="threading")(
                delayed(cal_layers_minmax)(tensor_name) for tensor_name in tqdm(percentile_tensor_minmax[0])
            )
        else:
            for tensor_name in tqdm(percentile_tensor_minmax[0]):
                cal_layers_minmax(tensor_name)

    def compute_data_from_histogram(
        self,
        percentile_tensor_minmax: list[Any],
    ) -> None:
        """
        Select the optimal percentile per tensor from each tensor's histogram, scoring
        every candidate via :func:`compute_lwp_metric_from_histogram` and keeping the one
        with the minimum error metric. Updates ``self.minmax_dict`` in place.

        :param list[Any] percentile_tensor_minmax: One dict per percentile candidate,
            each mapping tensor name to its ``(min, max)`` range. Every dict shares the
            same keys, so the first one enumerates the tensors to calibrate.
        """
        histogram_dict = self.collector.histogram_dict

        def select_from_histogram(tensor_name: str) -> None:
            candidate_minmax = [percentile_minmax[tensor_name] for percentile_minmax in percentile_tensor_minmax]
            candidate_metrics = compute_lwp_metric_from_histogram(
                histogram_dict[tensor_name],
                candidate_minmax,
                self.q_min,
                self.q_max,
                self.lwp_metric,
            )
            min_metric_index = int(np.argmin(candidate_metrics))
            self.minmax_dict[tensor_name] = candidate_minmax[min_metric_index]

        if self.worker_num > 1:
            Parallel(n_jobs=self.worker_num, backend="threading")(
                delayed(select_from_histogram)(tensor_name) for tensor_name in tqdm(percentile_tensor_minmax[0])
            )
        else:
            for tensor_name in tqdm(percentile_tensor_minmax[0]):
                select_from_histogram(tensor_name)

    def compute_data(self) -> TensorsData:
        """
        Compute the min-max range of tensor

        :return: dictionary mapping: {tensor name: (min value, max value)}
        """
        if not self.collector:
            raise ValueError("No collector created and can't generate calibration data.")

        cal = LayerWiseMethod.LayerWisePercentile
        return TensorsData(cal, self.minmax_dict)


@log_errors
def create_calibrator_power_of_two(
    model_input: str | Path | onnx.ModelProto,
    op_types_to_calibrate: Sequence[str] | None = None,
    augmented_model_path: str = "augmented_model.onnx",
    activation_type: QuantType | ExtendedQuantType = QuantType.QInt8,
    calibrate_method: PowerOfTwoMethod = PowerOfTwoMethod.NonOverflow,
    use_external_data_format: bool = False,
    execution_providers: list[str] | None = ["CPUExecutionProvider"],
    quantized_tensor_type: dict[Any, Any] = {},
    extra_options: dict[str, Any] = {},
) -> Any:
    """
    Create a calibrator for power-of-two quantization.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.
    :param Optional[Sequence[str]] op_types_to_calibrate: List of operator types to calibrate. Defaults to ``None``, which indicates that all float32/float16 tensors are calibrated.
    :param str augmented_model_path: Path to save the augmented ONNX model.
    :param Union[QuantType, ExtendedQuantType] activation_type: Type of quantization for activations.
    :param PowerOfTwoMethod calibrate_method: Calibration method to use.
    :param bool use_external_data_format: Whether to use external data format for large models.
    :param Union[List[str], None] execution_providers: List of execution providers for ONNX Runtime.
    :param Dict[Any, Any] quantized_tensor_type: Dictionary specifying the quantized tensor type.
    :param Dict[str, Any] extra_options: Additional options for calibrator configuration.

    :return: Initialized calibrator object.
    """
    calibrator = None

    overlay = resolve_calibrator_extra_defaults(calibrate_method, extra_options)
    if calibrate_method == PowerOfTwoMethod.NonOverflow:
        calibrator = MinMaxCalibrater(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=overlay["symmetric"],
            moving_average=overlay["moving_average"],
            averaging_constant=overlay["averaging_constant"],
            optimize_mem=overlay["optimize_mem"],
        )
    elif calibrate_method == PowerOfTwoMethod.MinMSE:
        calibrator = PowOfTwoCalibrater(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            activation_type=activation_type,
            method=calibrate_method,
            symmetric=overlay["symmetric"],
            minmse_mode=overlay["minmse_mode"],
            num_bins=overlay["num_bins"],
            percentile=overlay["percentile"],
            optimize_mem=overlay["optimize_mem"],
            worker_num=overlay["worker_num"],
            quantized_tensor_type=quantized_tensor_type,
        )

    if calibrator:
        calibrator.augment_graph()
        calibrator.execution_providers = execution_providers
        calibrator.create_inference_session()
        return calibrator

    raise ValueError(f"Unsupported calibration method {calibrate_method}")


@log_errors
def create_calibrator_float_scale(
    model_input: str | Path | onnx.ModelProto,
    op_types_to_calibrate: Sequence[str] | None = None,
    augmented_model_path: str = "augmented_model.onnx",
    activation_type: QuantType | ExtendedQuantType = QuantType.QInt8,
    calibrate_method: CalibrationMethod | LayerWiseMethod = CalibrationMethod.MinMax,
    use_external_data_format: bool = False,
    execution_providers: list[str] | None = ["CPUExecutionProvider"],
    extra_options: dict[str, Any] = {},  # noqa: B006
) -> Any:
    """
    Create a calibrator for floating-point scale quantization.

    :param Union[str, Path, onnx.ModelProto] model_input: ONNX model to calibrate.
    :param Optional[Sequence[str]] op_types_to_calibrate: List of operator types to calibrate. Defaults to ``None``, which indicates that all float32/float16 tensors are calibrated.
    :param str augmented_model_path: Path to save the augmented ONNX model.
    :param Union[CalibrationMethod, LayerWiseMethod] calibrate_method: Calibration method to use (MinMax, Entropy, Percentile, or Distribution).
    :param bool use_external_data_format: Whether to use external data format for large models.
    :param Union[List[str], None] execution_providers: List of execution providers for ONNX Runtime.
    :param Dict[str, Any] extra_options: Additional options for calibrator configuration.

    :return: Initialized calibrator object.
    """
    calibrator = None
    overlay = resolve_calibrator_extra_defaults(calibrate_method, extra_options)

    if calibrate_method == CalibrationMethod.MinMax:
        calibrator = MinMaxCalibrater(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=overlay["symmetric"],
            moving_average=overlay["moving_average"],
            averaging_constant=overlay["averaging_constant"],
            optimize_mem=overlay["optimize_mem"],
        )
    elif calibrate_method == CalibrationMethod.Entropy:
        calibrator = EntropyCalibrater(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=overlay["symmetric"],
            num_bins=overlay["num_bins"],
            num_quantized_bins=overlay["num_quantized_bins"],
            optimize_mem=overlay["optimize_mem"],
            worker_num=overlay["worker_num"],
        )
    elif calibrate_method == CalibrationMethod.Percentile:
        calibrator = PercentileCalibrater(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=overlay["symmetric"],
            num_bins=overlay["num_bins"],
            percentile=overlay["percentile"],
            optimize_mem=overlay["optimize_mem"],
            worker_num=overlay["worker_num"],
        )
    elif calibrate_method == CalibrationMethod.Distribution:
        calibrator = DistributionCalibrater(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            num_bins=overlay["num_bins"],
            scenario=overlay["scenario"],
            optimize_mem=overlay["optimize_mem"],
            worker_num=overlay["worker_num"],
        )
    elif calibrate_method == LayerWiseMethod.LayerWisePercentile:
        calibrator = LayerWisePercentileCalibrater(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=overlay["symmetric"],
            num_bins=overlay["num_bins"],
            percentile=overlay["percentile"],
            worker_num=overlay["worker_num"],
            lwp_metric=overlay["lwp_metric"],
            activation_type=activation_type,
            percentile_candidates=overlay["percentile_candidates"],
            lwp_use_histogram=overlay["lwp_use_histogram"],
            optimize_mem=overlay["optimize_mem"],
        )

    if calibrator:
        calibrator.augment_graph()
        calibrator.execution_providers = execution_providers
        calibrator.create_inference_session()
        return calibrator

    raise ValueError(f"Unsupported calibration method {calibrate_method}")
