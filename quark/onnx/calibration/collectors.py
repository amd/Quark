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

import multiprocessing
from typing import Any

import numpy as np
from joblib import Parallel, delayed  # type: ignore
from numpy.typing import NDArray
from onnxruntime.quantization.calibrate import CalibrationDataCollector, HistogramCollector
from onnxruntime.quantization.quant_utils import QuantType
from tqdm import tqdm

from quark.onnx.quantization.quant_utils import ExtendedQuantType, get_tensor_type_from_qType, quantize_data
from quark.shares.utils.log import ScreenLogger, log_errors

from .methods import PowerOfTwoMethod

logger = ScreenLogger(__name__)


class OverridedHistogramCollector(HistogramCollector):  # type: ignore
    """
    Collecting histogram for each tensor. Distribution, Percentile and Entropy method are supported.
    This overrided collector is used to accelerate collecting data using multiple processes.

    :param str method: A string. One of ['entropy', 'percentile', 'distribution'].
    :param bool symmetric: make range of tensor symmetric (central point is 0).
    :param int num_bins: number of bins to create a new histogram for collecting tensor values.
    :param int num_quantized_bins: number of quantized bins.
    :param float percentile: A float number between [0, 100].
    :param str scenario: scenario string for Distribution method.
    """

    def __init__(
        self,
        method: str,
        symmetric: bool,
        num_bins: int,
        num_quantized_bins: int,
        percentile: float,
        scenario: str = "same",
        worker_num: int = 1,
    ) -> None:
        super().__init__(method, symmetric, num_bins, num_quantized_bins, percentile, scenario)

        if worker_num > multiprocessing.cpu_count():
            logger.warning(
                f"The number of workers {worker_num} can not larger than cpu cores {multiprocessing.cpu_count()}"
            )
            self.worker_num = multiprocessing.cpu_count()
        else:
            self.worker_num = max(worker_num, 1)

    def compute_percentile(self) -> dict[str, tuple[NDArray[Any], NDArray[Any]]]:
        """
        Compute percentile-based thresholds for each tensor in the histogram.

        :return: Dictionary mapping tensor names to threshold tuples.
        :rtype: dict[str, tuple[NDArray]]
        """
        if self.percentile < 0 or self.percentile > 100:
            raise ValueError("Invalid percentile. Must be in range 0 <= percentile <= 100.")

        histogram_dict = self.histogram_dict
        percentile = self.percentile

        thresholds_dict = {}  # per tensor thresholds

        logger.info(f"Number of tensors : {len(histogram_dict)}")
        logger.info(f"Number of histogram bins : {self.num_bins}")
        logger.info(f"Percentile : ({100.0 - percentile},{percentile})")

        def compute_layer_wise_percentile(tensor: str, histogram: tuple[Any]) -> None:
            hist = histogram[0]
            hist_edges = histogram[1]
            total = hist.sum()
            cdf = np.cumsum(hist / total)
            if self.symmetric:
                idx_right = np.searchsorted(cdf, percentile / 100.0)

                thresholds_dict[tensor] = (
                    -np.array(hist_edges[idx_right], dtype=hist_edges.dtype),
                    np.array(hist_edges[idx_right], dtype=hist_edges.dtype),
                )
            else:
                percent_to_cut_one_side = (100.0 - percentile) / 200.0
                idx_right = np.searchsorted(cdf, 1.0 - percent_to_cut_one_side)
                idx_left = np.searchsorted(cdf, percent_to_cut_one_side)
                thresholds_dict[tensor] = (
                    np.array(hist_edges[idx_left], dtype=hist_edges.dtype),
                    np.array(hist_edges[idx_right], dtype=hist_edges.dtype),
                )
            min_value = histogram[2]
            max_value = histogram[3]
            if thresholds_dict[tensor][0] < min_value:
                thresholds_dict[tensor] = (min_value, thresholds_dict[tensor][1])
            if thresholds_dict[tensor][1] > max_value:
                thresholds_dict[tensor] = (thresholds_dict[tensor][0], max_value)
            thresholds_dict[tensor] = (*thresholds_dict[tensor], *hist[:2])

        if self.worker_num > 1:
            Parallel(n_jobs=self.worker_num, backend="threading")(
                delayed(compute_layer_wise_percentile)(tensor, histogram)
                for tensor, histogram in histogram_dict.items()
            )
        else:
            for tensor, histogram in histogram_dict.items():
                compute_layer_wise_percentile(tensor, histogram)

        return thresholds_dict

    def collect_absolute_value(self, name_to_arr: dict[str, list[NDArray[Any]]]) -> None:
        """
        Collect histogram on absolute value
        """

        def collect_layer_wise_absolute_value(tensor: str, data_arr: list[NDArray[Any]]) -> None:
            if isinstance(data_arr, list):
                for arr in data_arr:
                    assert isinstance(arr, np.ndarray), f"Unexpected type {type(arr)} for tensor={tensor!r}"
                dtypes = {a.dtype for a in data_arr}
                assert len(dtypes) == 1, (
                    f"The calibration expects only one element type but got {dtypes} for tensor={tensor!r}"
                )
                data_arr_np = np.asarray(data_arr)
            elif not isinstance(data_arr, np.ndarray):
                raise ValueError(f"Unexpected type {type(data_arr)} for tensor={tensor!r}")
            else:
                data_arr_np = data_arr
            data_arr_np = data_arr_np.flatten()
            if data_arr_np.size > 0:
                min_value = np.nanmin(data_arr_np)
                max_value = np.nanmax(data_arr_np)
            else:
                min_value = np.array(0, dtype=data_arr_np.dtype)
                max_value = np.array(0, dtype=data_arr_np.dtype)

            data_arr_np = np.absolute(data_arr_np)  # only consider absolute value

            if tensor not in self.histogram_dict:
                # first time it uses num_bins to compute histogram.
                hist, hist_edges = np.histogram(data_arr_np, bins=self.num_bins)
                hist_edges = hist_edges.astype(data_arr_np.dtype)
                assert data_arr_np.dtype != np.float64, (
                    "only float32 or float16 is supported, every constant must be explicitly typed"
                )
                self.histogram_dict[tensor] = (hist, hist_edges, min_value, max_value)
            else:
                old_histogram = self.histogram_dict[tensor]
                old_min = old_histogram[2]
                old_max = old_histogram[3]
                assert hasattr(old_min, "dtype"), f"old_min should be a numpy array but is {type(old_min)}"
                assert hasattr(old_max, "dtype"), f"old_min should be a numpy array but is {type(old_max)}"
                old_hist = old_histogram[0]
                old_hist_edges = old_histogram[1]
                temp_amax = np.nanmax(data_arr_np)
                if temp_amax > old_hist_edges[-1]:
                    # increase the number of bins
                    width = old_hist_edges[1] - old_hist_edges[0]
                    # NOTE: np.arange may create an extra bin after the one containing temp_amax
                    new_bin_edges = np.arange(old_hist_edges[-1] + width, temp_amax + width, width)
                    old_hist_edges = np.hstack((old_hist_edges, new_bin_edges))
                hist, hist_edges = np.histogram(data_arr_np, bins=old_hist_edges)
                hist_edges = hist_edges.astype(data_arr_np.dtype)
                hist[: len(old_hist)] += old_hist
                assert data_arr_np.dtype != np.float64, (
                    "only float32 or float16 is supported, every constant must be explicitly typed"
                )
                self.histogram_dict[tensor] = (hist, hist_edges, min(old_min, min_value), max(old_max, max_value))

        if self.worker_num > 1:
            Parallel(n_jobs=self.worker_num, backend="threading")(
                delayed(collect_layer_wise_absolute_value)(tensor, data_arr) for tensor, data_arr in name_to_arr.items()
            )
        else:
            for tensor, data_arr in name_to_arr.items():
                collect_layer_wise_absolute_value(tensor, data_arr)

    def collect_value(self, name_to_arr: dict[str, list[NDArray[Any]]]) -> None:
        """
        Collect histogram on real value
        """

        def collect_layer_wise_value(tensor: str, data_arr: list[NDArray[Any]]) -> None:
            data_arr = np.asarray(data_arr)  # noqa: PLW2901
            data_arr = data_arr.flatten()  # noqa: PLW2901

            if data_arr.size > 0:
                min_value = np.nanmin(data_arr)
                max_value = np.nanmax(data_arr)
            else:
                min_value = np.array(0, dtype=data_arr.dtype)
                max_value = np.array(0, dtype=data_arr.dtype)

            threshold = np.array(max(abs(min_value), abs(max_value)), dtype=data_arr.dtype)

            if tensor in self.histogram_dict:
                old_histogram = self.histogram_dict[tensor]
                self.histogram_dict[tensor] = self.merge_histogram(
                    old_histogram, data_arr, min_value, max_value, threshold
                )
            else:
                hist, hist_edges = np.histogram(data_arr, self.num_bins, range=(-threshold, threshold))
                self.histogram_dict[tensor] = (
                    hist,
                    hist_edges,
                    min_value,
                    max_value,
                    threshold,
                )

        if self.worker_num > 1:
            Parallel(n_jobs=self.worker_num, backend="threading")(
                delayed(collect_layer_wise_value)(tensor, data_arr) for tensor, data_arr in name_to_arr.items()
            )
        else:
            for tensor, data_arr in name_to_arr.items():
                collect_layer_wise_value(tensor, data_arr)


calib_quant_types = [
    QuantType.QInt8,
    QuantType.QUInt8,
    QuantType.QInt16,
    QuantType.QUInt16,
    ExtendedQuantType.QInt8,
    ExtendedQuantType.QUInt8,
    ExtendedQuantType.QInt16,
    ExtendedQuantType.QUInt16,
    ExtendedQuantType.QInt32,
    ExtendedQuantType.QUInt32,
]


def compute_minmse_worker(
    tensor_name: str,
    tensor_data: list[Any],
    quantized_tensor_type: dict[Any, Any],
    minmse_mode: str,
    optimize_mem: bool,
    activation_qType: Any,
    symmetric: Any,
    method: Any,
    percentile: Any,
) -> tuple[str, tuple[Any, Any]]:
    """This is the worker function for collecting MinMSE data.
    In order to enable multiple processing, we separate the
    code from the collector.
    """

    def _all_dims_equal(data_arr: list[Any]) -> bool:
        if isinstance(data_arr, list) and len(data_arr) > 1:
            ref_shape = data_arr[0].shape
            for arr in data_arr[1:]:
                if arr.shape != ref_shape:
                    return False
        return True

    def _nonbatch_dims_equal(data_arr: list[Any]) -> bool:
        if isinstance(data_arr, list) and len(data_arr) > 1 and len(data_arr[0].shape) > 1:
            ref_shape = data_arr[0].shape[1:]
            for arr in data_arr[1:]:
                if arr.shape[1:] != ref_shape:
                    return False
        return True

    def _mostcommon_mode(data_arr: list[Any], act_type: Any, symmetric: Any, method: Any) -> tuple[Any, Any]:
        scale2threshold: dict[float, tuple[Any, Any]] = {}

        scale_list = []
        for d in data_arr:
            if isinstance(d, list):
                # We have calculated the quantization parameters already in collecting data process
                rmin_mse, rmax_mse, scale_mse = d[0], d[1], d[2]
            else:
                # This needs caching data in memory, abandoned
                rmin_mse, rmax_mse, _, scale_mse, _ = quantize_data(
                    data=d.astype(np.float32, copy=False), qType=act_type, symmetric=symmetric, method=method
                )
            scale2threshold[float(scale_mse)] = (rmin_mse, rmax_mse)
            scale_list.append(scale_mse)
        u, indices = np.unique(scale_list, return_inverse=True)
        scale = u[np.argmax(np.bincount(indices))]

        return scale2threshold[scale]

    def _percentile_mode(
        data_arr: list[Any], act_type: Any, symmetric: Any, method: Any, percentile: Any
    ) -> tuple[Any, Any]:
        if _all_dims_equal(data_arr):
            # The np.array() requires all dims to be exactly the same
            d = np.array(data_arr).flatten()
        elif _nonbatch_dims_equal(data_arr):
            # The np.concatenate() requires array dims except for the concat axis must match exactly
            d = np.concatenate(data_arr, axis=0).flatten()
        else:
            raise ValueError("The dims of samples do not match exactly!")

        if symmetric:
            lower_limit = -np.percentile(np.abs(d), percentile)
            upper_limit = np.percentile(np.abs(d), percentile)
        else:
            lower_limit = np.percentile(d, (100 - percentile) / 2)
            upper_limit = np.percentile(d, 100 - (100 - percentile) / 2)
        d = d[(d >= lower_limit) & (d <= upper_limit)]

        rmin_mse, rmax_mse, *_ = quantize_data(
            data=d.astype(np.float32, copy=False), qType=act_type, symmetric=symmetric, method=method
        )
        return (rmin_mse, rmax_mse)

    def _all_arrays_mode(data_arr: list[Any], act_type: Any, symmetric: Any, method: Any) -> tuple[Any, Any]:
        if _all_dims_equal(data_arr):
            # The np.array() requires all dims to be exactly the same
            d = np.array(data_arr).flatten()
        elif _nonbatch_dims_equal(data_arr):
            # The np.concatenate() requires array dims except for the concat axis must match exactly
            d = np.concatenate(data_arr, axis=0).flatten()
        else:
            raise ValueError("The dims of samples do not match exactly!")

        rmin_mse, rmax_mse, *_ = quantize_data(
            data=d.astype(np.float32, copy=False), qType=act_type, symmetric=symmetric, method=method
        )
        return (rmin_mse, rmax_mse)

    if not optimize_mem:
        data_arr = tensor_data
    else:
        # In this case, the content in the list is the caching file name,
        # need to load data to memory
        data_arr = []
        with open(tensor_data[0], "rb") as f:
            for _ in range(len(tensor_data)):
                data_arr.append(np.load(f))

    if len(data_arr) == 0:
        raise ValueError(f"Missed data for the tensor {tensor_name}, please check.")

    act_type = activation_qType
    if tensor_name in quantized_tensor_type and quantized_tensor_type[tensor_name] in calib_quant_types:
        logger.info(
            f"The type of tensor {tensor_name} is {quantized_tensor_type[tensor_name]}, using specific tensor precision"
        )
        act_type = get_tensor_type_from_qType(quantized_tensor_type[tensor_name])

    if minmse_mode == "MostCommon" and symmetric:
        threshold = _mostcommon_mode(data_arr, act_type, symmetric, method)
    elif minmse_mode == "Percentile":
        threshold = _percentile_mode(data_arr, act_type, symmetric, method, percentile)
    else:
        threshold = _all_arrays_mode(data_arr, act_type, symmetric, method)

    return tensor_name, threshold


def compute_minmse_worker_unpack(args: Any) -> tuple[str, tuple[Any, Any]]:
    """This is a helper function to unpack the arguments"""
    return compute_minmse_worker(*args)


class PowOfTwoCollector(CalibrationDataCollector):  # type: ignore
    """
    Collecting PowOfTwoCollector quantize for each tensor. Support MinMSE method.

    :param Union[QuantType, ExtendedQuantType] activation_type: Type of quantization for activations. Default is QuantType.QInt8.
    :param PowerOfTwoMethod method: Calibration method. Default is PowerOfTwoMethod.MinMSE.
    :param bool symmetric: Whether to make the range of tensor symmetric (central point is 0). Default is True.
    :param str minmse_mode: Mode for the MinMSE method. Default is "All".
    :param float percentile: Percentile value for calibration, a float between 0 and 100. Default is 99.999.
    :param bool optimize_mem: Whether to optimize memory consumption. Default is True.
    :param int worker_num: Number of workers to do the data collection. Default is 1.
    :param Dict[Any, Any] quantized_tensor_type: Dictionary specifying the quantized tensor type. Default is an empty dictionary.
    """

    def __init__(
        self,
        activation_type: QuantType | ExtendedQuantType = QuantType.QInt8,
        method: PowerOfTwoMethod = PowerOfTwoMethod.MinMSE,
        symmetric: bool = True,
        minmse_mode: str = "All",
        percentile: float = 99.999,
        optimize_mem: bool = True,
        worker_num: int = 1,
        quantized_tensor_type: dict[Any, Any] = {},
    ):
        if activation_type not in calib_quant_types:
            logger.warning(f"Unsupported activation type {activation_type} for MinMSE, applying Int8 instead.")
            self.activation_qType = get_tensor_type_from_qType(QuantType.QInt8)
        else:
            self.activation_qType = get_tensor_type_from_qType(activation_type)
        self.method = method
        self.symmetric = symmetric
        self.minmse_mode = minmse_mode
        self.percentile = percentile
        self.optimize_mem = optimize_mem
        self.worker_num = worker_num
        self.quantized_tensor_type = quantized_tensor_type

        self.is_mostcommon = True if self.minmse_mode == "MostCommon" and self.symmetric else False
        self.name_to_arr: dict[Any, Any] = {}  # For MostCommon, the value is a 2D list storing the quant params

    def collect(self, name_to_arr: dict[Any, Any]) -> None:
        if self.is_mostcommon:
            return self.collect_mostcommon_value(name_to_arr)
        else:
            return self.collect_value(name_to_arr)

    def collect_mostcommon_value(self, name_to_arr: dict[Any, Any]) -> None:
        """Collect data for the most common mode"""
        for tensor_name, data_arr in name_to_arr.items():
            act_type = self.activation_qType
            if (
                tensor_name in self.quantized_tensor_type
                and self.quantized_tensor_type[tensor_name] in calib_quant_types
            ):
                act_type = get_tensor_type_from_qType(self.quantized_tensor_type[tensor_name])

            assert len(data_arr) == 1, "Each time it only supports collect a sample"
            rmin_mse, rmax_mse, _, scale_mse, _ = quantize_data(
                data=data_arr[0].astype(np.float32, copy=False),
                qType=act_type,
                symmetric=self.symmetric,
                method=self.method,
            )

            if tensor_name not in self.name_to_arr:
                self.name_to_arr[tensor_name] = [[rmin_mse, rmax_mse, scale_mse]]
            else:
                self.name_to_arr[tensor_name].append([rmin_mse, rmax_mse, scale_mse])

    def collect_value(self, name_to_arr: dict[Any, Any]) -> None:
        """Collect data for the percentile and all mode"""
        self.name_to_arr = name_to_arr

    def compute_collection_result(self) -> Any:
        if not self.name_to_arr or len(self.name_to_arr) == 0:
            raise ValueError("Data has not been collected. Please run collect() first.")
        logger.info(
            f"Finding optimal threshold for each tensor using {self.method} algorithm in '{self.minmse_mode}' mode ..."
        )

        if self.method == PowerOfTwoMethod.MinMSE:
            return self.compute_minmse()
        else:
            raise ValueError("Only 'MinMSE' method is supported")

    @log_errors
    def compute_minmse(self) -> dict[Any, Any]:
        """Compute the data range for each tensor. It supports working in three modes:
        'MostCommon': Calculate by batch and use the one with the highest number of occurrences
        'Percentile': Calculate only a portion of representative data
        'All': Calculate all data, this is the default mode
        """
        if self.minmse_mode == "MostCommon" and not self.symmetric:
            logger.warning(
                f"The {self.minmse_mode} mode does not support asymmetric activations, will use the default mode instead."
            )
        elif self.minmse_mode == "Percentile":
            logger.debug(
                f"The {self.minmse_mode} mode has CalibTensorRangeSymmetric {self.symmetric} and Percentile {self.percentile}"
            )
        if self.worker_num > multiprocessing.cpu_count():
            logger.warning(
                f"The number of workers {self.worker_num} can not larger than cpu cores {multiprocessing.cpu_count()}"
            )
            self.worker_num = multiprocessing.cpu_count()

        thresholds_dict: dict[str, tuple[Any, Any]] = {}  # Per tensor thresholds

        if self.worker_num <= 1:
            for tensor, data_arr in tqdm(self.name_to_arr.items()):
                name, threshold = compute_minmse_worker(
                    tensor,
                    data_arr,
                    self.quantized_tensor_type,
                    self.minmse_mode,
                    self.optimize_mem,
                    self.activation_qType,
                    self.symmetric,
                    self.method,
                    self.percentile,
                )
                thresholds_dict[name] = threshold  # The name is the tensor
        else:
            results = Parallel(n_jobs=self.worker_num, backend="threading")(
                delayed(compute_minmse_worker)(
                    tensor,
                    data_arr,
                    self.quantized_tensor_type,
                    self.minmse_mode,
                    self.optimize_mem,
                    self.activation_qType,
                    self.symmetric,
                    self.method,
                    self.percentile,
                )
                for tensor, data_arr in tqdm(self.name_to_arr.items())
            )

            for name, threshold in results:
                thresholds_dict[name] = threshold  # The name is the tensor

        return thresholds_dict
