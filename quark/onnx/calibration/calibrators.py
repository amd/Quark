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
from pathlib import Path
from typing import Any, Sequence

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

from quark.onnx.quantization.quant_utils import ExtendedQuantType
from quark.onnx.utils.file_utils import save_quantized_info
from quark.onnx.utils.model_utils import create_infer_session_for_onnx_model, sanitize_model_outputs
from quark.shares.utils.log import ScreenLogger, log_errors

from .collectors import LoadingDataFromDisk, OverridedHistogramCollector, PowOfTwoCollector
from .methods import LayerWiseMethod, PowerOfTwoMethod

logger = ScreenLogger(__name__)


def GenerateAnEmptyOnnxModel(model_path: str) -> None:
    """
    This function is used to generate an empty onnx model based on
    the provided path for the initialization of calibrators.
    """
    graph = onnx.helper.make_graph(name="EmptyGraph", inputs=[], outputs=[], nodes=[])
    model = onnx.helper.make_model(graph, producer_name="empty-model")
    onnx.save(model, model_path)


def CachingDataOnDisk(
    session: onnxruntime.InferenceSession, inputs: dict[str, Any], cache_dir: str, to_append: bool
) -> tuple[list[str], int]:
    """
    Execute a session run and save the outputs to the caching directory.
    This function encapsulates the session run within a function and save the outputs
    to local files, ensuring that memory of output arrays is released.

    :param onnxruntime.InferenceSession session: the session to run
    :param dict[str, Any] inputs: the input data for the run
    :param str cache_dir: the caching directory to store the files
    :param str to_append: to append the data to existing file or not
    :return: The list of saved file paths and the number of bytes it saved
    """
    outputs = session.run(None, inputs)
    sanitize_model_outputs(outputs)

    file_list: list[str] = []
    cached_nbytes = 0

    for output_index, output in enumerate(outputs):
        if to_append:
            # For append mode, the data is stored in float16 to save disk space
            nbytes = output.nbytes / 2 if output.dtype == np.float32 else output.nbytes
            file_path = os.path.join(cache_dir, f"output{output_index}_data.npz")
            with open(file_path, "ab") as f:
                np.save(f, output.astype(np.float16))
        else:
            nbytes = output.nbytes
            file_path = os.path.join(cache_dir, f"output{output_index}_data.npy")
            with open(file_path, "wb") as f:
                np.save(f, output)

        file_list.append(file_path)
        cached_nbytes += nbytes

    return file_list, cached_nbytes


def GetCleanMergedDict(
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
            GenerateAnEmptyOnnxModel(augmented_model_path)
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
        except NotImplementedError:
            raise ValueError("The data reader should implement the '__len__' method to provide the data size.")

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
    :param bool layer_wise: Whether to work on layerwise percentile mode. Default is False.
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
        layer_wise: bool = False,
        optimize_mem: bool = True,
        worker_num: int = 1,
    ):
        if isinstance(model_input, onnx.ModelProto):
            GenerateAnEmptyOnnxModel(augmented_model_path)
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

        self.layer_wise = layer_wise
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

    def collect_data(self, data_reader: CalibrationDataReader) -> None:
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
            )

        input_names_set = {node_arg.name for node_arg in self.infer_session.get_inputs()}
        output_names = [node_arg.name for node_arg in self.infer_session.get_outputs()]

        try:
            data_size = len(data_reader)
        except NotImplementedError:
            raise ValueError("The data reader should implement the '__len__' method to provide the data size.")

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
                to_append = self.method == "percentile" and self.layer_wise
                fixed_outputs, cached_nbytes = CachingDataOnDisk(self.infer_session, inputs, cache_dir, to_append)  # type: ignore
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

            clean_merged_dict = GetCleanMergedDict([fixed_outputs], output_names, self.tensors_to_calibrate)

            if self.method == "percentile" and self.layer_wise:
                self.intermediate_outputs.append(fixed_outputs)
                cache_capacity += cached_nbytes
            else:
                cache_capacity = cached_nbytes

            collect_data_acquired_data_time = time.perf_counter()
            self.collector.collect(clean_merged_dict)
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

        if len(self.intermediate_outputs):
            # For layer-wise percentile, all data may be used
            self.clean_merged_dict = GetCleanMergedDict(
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
    :param bool layer_wise: Whether to work on layerwise percentile mode. Default is False.
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
        layer_wise: bool = False,
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
            layer_wise=layer_wise,
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
        percentile: float = 99.999,
        optimize_mem: bool = True,
        worker_num: int = 1,
        quantized_tensor_type: dict[Any, Any] = {},
    ) -> None:
        if isinstance(model_input, onnx.ModelProto):
            GenerateAnEmptyOnnxModel(augmented_model_path)
            model_path = augmented_model_path  # Generate an empty model for the base class to load
        else:
            model_path = model_input.as_posix() if isinstance(model_input, Path) else model_input

        super(PowOfTwoCalibrater, self).__init__(
            model_path, op_types_to_calibrate, augmented_model_path, symmetric, use_external_data_format
        )

        if isinstance(model_input, onnx.ModelProto):
            # Replace the empty model with the real input model.
            # The copy is to avoid modifying the input model.
            self.model = copy.deepcopy(model_input)

        self.intermediate_outputs: list[Any] = []
        self.calibrate_tensors_range = None
        self.num_model_outputs = len(self.model.graph.output)
        self.model_original_outputs = set(output.name for output in self.model.graph.output)
        self.collector: PowOfTwoCollector | None = None
        self.method = method
        self.symmetric = symmetric
        self.tensors_to_calibrate = None
        self.activation_type = activation_type
        self.use_external_data_format = use_external_data_format
        self.minmse_mode = minmse_mode
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
                percentile=self.percentile,
                optimize_mem=self.optimize_mem,
                worker_num=self.worker_num,
                quantized_tensor_type=self.quantized_tensor_type,
            )

        input_names_set = {node_arg.name for node_arg in self.infer_session.get_inputs()}
        output_names = [node_arg.name for node_arg in self.infer_session.get_outputs()]

        try:
            data_size = len(data_reader)
        except NotImplementedError:
            raise ValueError("The data reader should implement the '__len__' method to provide the data size.")

        cache_dir = os.path.dirname(self.augmented_model_path)  # For caching tensors
        cache_capacity = 0

        pbar = tqdm(range(data_size))
        for _ in pbar:
            inputs = data_reader.get_next()
            if not inputs:
                break

            fixed_outputs: list[str | np.ndarray[Any, Any]] = []

            if self.optimize_mem:
                to_append = not (self.collector and self.collector.optimized_mostcommon)
                fixed_outputs, cached_nbytes = CachingDataOnDisk(self.infer_session, inputs, cache_dir, to_append)  # type: ignore
                cache_capacity = cache_capacity + cached_nbytes if to_append else cached_nbytes
            else:
                outputs = self.infer_session.run(None, inputs)
                sanitize_model_outputs(outputs)

                for output_index, output in enumerate(outputs):
                    # Copy np.ndarray only for graph outputs that are also graph inputs to workaround bug:
                    # https://github.com/microsoft/onnxruntime/issues/21922
                    if output_names[output_index] in input_names_set:
                        fixed_outputs.append(copy.copy(output))
                    else:
                        fixed_outputs.append(output)
                    cache_capacity += output.nbytes

            if self.collector and self.collector.optimized_mostcommon:
                clean_merged_dict = GetCleanMergedDict([fixed_outputs], output_names, self.tensors_to_calibrate)
                self.collector.collect(clean_merged_dict)
            else:
                self.intermediate_outputs.append(fixed_outputs)

            pbar.set_description(
                f"Cached {cache_capacity / (1024**3):.2f}GB on {'disk' if self.optimize_mem else 'memory'}"
            )

        if self.collector.optimized_mostcommon:
            return None

        if len(self.intermediate_outputs) == 0:
            raise ValueError("No data is collected.")

        clean_merged_dict = GetCleanMergedDict(self.intermediate_outputs, output_names, self.tensors_to_calibrate)

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
    :param int activation_bitwidth: Bitwidth for activations. Defaults to ``8``.
    :param List[float] percentile_candidates: Percentile candidates. Defaults to ``[99.99, 99.999, 99.9999]``.
    :param bool optimize_mem: Whether to optimize memory consumption. Default is True.
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
        lwp_metric: str = "mae",
        activation_bitwidth: int = 8,
        percentile_candidates: list[float] = [99.99, 99.999, 99.9999],
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
            layer_wise=True,
            optimize_mem=optimize_mem,
            worker_num=worker_num,
        )
        self.minmax_dict: dict[str, float] = {}
        self.percentile_dict: dict[str, float] = {}
        self.lwp_metric = lwp_metric
        self.activation_bitwidth = activation_bitwidth
        self.q_min = 0
        self.q_max = 2**self.activation_bitwidth - 1
        self.percentile_candidates = percentile_candidates

    def collect_data(self, data_reader: CalibrationDataReader) -> None:
        # Call the parent class method to calculate the histogram
        super().collect_data(data_reader)
        assert self.clean_merged_dict, "No data for the layerwise percentile"

        # Assign different percentiles to compute the tensors range. Note that the list
        # stores dictionaries that the key is tensor name and the value is the range
        tensors_ranges_percentiles = []

        for temp_percentile in self.percentile_candidates:
            self.collector.percentile = temp_percentile
            temp_ranges = self.collector.compute_percentile()
            tensors_ranges_percentiles.append(temp_ranges)

        baseline_tensors_range = tensors_ranges_percentiles[0]

        def cal_layers_minmax(key: str) -> None:
            """
            Compute layer min-max values by finding the optimal percentile for a given tensor.
            This function evaluates different percentile candidates for a tensor and selects
            the one that minimizes the quantization error (MSE or MAE). It updates the
            minmax_dict and percentile_dict with the best values found.
            :param key: The tensor name/key to compute min-max values for.
            """
            min_metric_value = 1000000.0
            for idx in range(len(tensors_ranges_percentiles)):
                temp_value = tensors_ranges_percentiles[idx][key]
                q_min, q_max = self.q_min, self.q_max

                data_arr = self.clean_merged_dict[key]
                assert isinstance(data_arr, list), "The value should be a list"
                data_list = LoadingDataFromDisk(data_arr)
                temp_tensor = np.asarray(data_list).reshape(-1)

                temp_scale = (temp_value[1] - temp_value[0]) / (q_max - q_min)
                # Preventing spills of scale value
                temp_scale = temp_scale + 1e-6
                temp_zp = np.round(temp_value[0] / temp_scale - q_min)
                q_temp_tensor = np.clip(np.round(temp_tensor / temp_scale - temp_zp), q_min, q_max)
                qdq_temp_tensor = (q_temp_tensor + temp_zp) * temp_scale
                if self.lwp_metric == "mse":
                    temp_metric_value = np.mean((temp_tensor - qdq_temp_tensor) ** 2)
                else:
                    temp_metric_value = np.mean(np.abs(temp_tensor - qdq_temp_tensor))

                if temp_metric_value < min_metric_value:
                    min_metric_value = temp_metric_value
                    self.minmax_dict[key] = temp_value
                    self.percentile_dict[key] = self.percentile_candidates[idx]

        if self.worker_num > 1:
            Parallel(n_jobs=self.worker_num, backend="threading")(
                delayed(cal_layers_minmax)(key) for key in tqdm(baseline_tensors_range)
            )
        else:
            for key in tqdm(baseline_tensors_range):
                cal_layers_minmax(key)

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

    if calibrate_method == PowerOfTwoMethod.NonOverflow:
        symmetric = True if "symmetric" not in extra_options else extra_options["symmetric"]
        moving_average = False if "moving_average" not in extra_options else extra_options["moving_average"]
        averaging_constant = 0.01 if "averaging_constant" not in extra_options else extra_options["averaging_constant"]
        optimize_mem = True if "optimize_mem" not in extra_options else extra_options["optimize_mem"]
        calibrator = MinMaxCalibrater(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            moving_average=moving_average,
            averaging_constant=averaging_constant,
            optimize_mem=optimize_mem,
        )
    elif calibrate_method == PowerOfTwoMethod.MinMSE:
        symmetric = True if "symmetric" not in extra_options else extra_options["symmetric"]
        minmse_mode = "All" if "minmse_mode" not in extra_options else extra_options["minmse_mode"]
        percentile = 99.999 if "percentile" not in extra_options else extra_options["percentile"]
        optimize_mem = True if "optimize_mem" not in extra_options else extra_options["optimize_mem"]
        worker_num = 1 if "worker_num" not in extra_options else extra_options["worker_num"]
        calibrator = PowOfTwoCalibrater(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            activation_type=activation_type,
            method=calibrate_method,
            symmetric=symmetric,
            minmse_mode=minmse_mode,
            percentile=percentile,
            optimize_mem=optimize_mem,
            worker_num=worker_num,
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

    if calibrate_method == CalibrationMethod.MinMax:
        # default settings for min-max algorithm
        symmetric = False if "symmetric" not in extra_options else extra_options["symmetric"]
        moving_average = False if "moving_average" not in extra_options else extra_options["moving_average"]
        averaging_constant = 0.01 if "averaging_constant" not in extra_options else extra_options["averaging_constant"]
        optimize_mem = True if "optimize_mem" not in extra_options else extra_options["optimize_mem"]
        calibrator = MinMaxCalibrater(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            moving_average=moving_average,
            averaging_constant=averaging_constant,
            optimize_mem=optimize_mem,
        )
    elif calibrate_method == CalibrationMethod.Entropy:
        # default settings for entropy algorithm
        num_bins = 128 if "num_bins" not in extra_options else extra_options["num_bins"]
        num_quantized_bins = 128 if "num_quantized_bins" not in extra_options else extra_options["num_quantized_bins"]
        symmetric = False if "symmetric" not in extra_options else extra_options["symmetric"]
        optimize_mem = True if "optimize_mem" not in extra_options else extra_options["optimize_mem"]
        worker_num = 1 if "worker_num" not in extra_options else extra_options["worker_num"]
        calibrator = EntropyCalibrater(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            num_bins=num_bins,
            num_quantized_bins=num_quantized_bins,
            optimize_mem=optimize_mem,
            worker_num=worker_num,
        )
    elif calibrate_method == CalibrationMethod.Percentile:
        # default settings for percentile algorithm
        num_bins = 2048 if "num_bins" not in extra_options else extra_options["num_bins"]
        percentile = 99.999 if "percentile" not in extra_options else extra_options["percentile"]
        symmetric = True if "symmetric" not in extra_options else extra_options["symmetric"]
        optimize_mem = True if "optimize_mem" not in extra_options else extra_options["optimize_mem"]
        worker_num = 1 if "worker_num" not in extra_options else extra_options["worker_num"]
        calibrator = PercentileCalibrater(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            num_bins=num_bins,
            percentile=percentile,
            optimize_mem=optimize_mem,
            worker_num=worker_num,
        )
    elif calibrate_method == CalibrationMethod.Distribution:
        # default settings for distribution algorithm
        num_bins = 2048 if "num_bins" not in extra_options else extra_options["num_bins"]
        scenario = "same" if "scenario" not in extra_options else extra_options["scenario"]
        optimize_mem = True if "optimize_mem" not in extra_options else extra_options["optimize_mem"]
        worker_num = 1 if "worker_num" not in extra_options else extra_options["worker_num"]
        calibrator = DistributionCalibrater(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            num_bins=num_bins,
            scenario=scenario,
            optimize_mem=optimize_mem,
            worker_num=worker_num,
        )
    elif calibrate_method == LayerWiseMethod.LayerWisePercentile:
        # default settings for layerwise percentile algorithm
        num_bins = 2048 if "num_bins" not in extra_options else extra_options["num_bins"]
        percentile = 99.999 if "percentile" not in extra_options else extra_options["percentile"]
        symmetric = True if "symmetric" not in extra_options else extra_options["symmetric"]
        optimize_mem = True if "optimize_mem" not in extra_options else extra_options["optimize_mem"]
        worker_num = 1 if "worker_num" not in extra_options else extra_options["worker_num"]
        lwp_metric = "mae" if "lwp_metric" not in extra_options else extra_options["lwp_metric"]
        activation_bitwidth = 8 if "activation_bitwidth" not in extra_options else extra_options["activation_bitwidth"]
        percentile_candidates = (
            [99.99, 99.999, 99.99999]
            if "percentile_candidates" not in extra_options
            else extra_options["percentile_candidates"]
        )
        optimize_mem = True if "optimize_mem" not in extra_options else extra_options["optimize_mem"]
        calibrator = LayerWisePercentileCalibrater(
            model_input,
            op_types_to_calibrate,
            augmented_model_path,
            use_external_data_format=use_external_data_format,
            symmetric=symmetric,
            num_bins=num_bins,
            percentile=percentile,
            optimize_mem=optimize_mem,
            worker_num=worker_num,
            lwp_metric=lwp_metric,
            activation_bitwidth=activation_bitwidth,
            percentile_candidates=percentile_candidates,
        )

    if calibrator:
        calibrator.augment_graph()
        calibrator.execution_providers = execution_providers
        calibrator.create_inference_session()
        return calibrator

    raise ValueError(f"Unsupported calibration method {calibrate_method}")
