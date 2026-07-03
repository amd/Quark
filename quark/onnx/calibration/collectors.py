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

import math
import multiprocessing
import os
from typing import Any

import numpy as np
from joblib import Parallel, delayed  # type: ignore
from numpy.typing import NDArray
from onnxruntime.quantization.calibrate import CalibrationDataCollector, HistogramCollector
from onnxruntime.quantization.quant_utils import QuantType
from tqdm import tqdm

from quark.common.utils.log import ScreenLogger, log_errors
from quark.onnx.quantization.quant_utils import (
    ExtendedQuantType,
    compute_scale_zp,
    get_qmin_qmax_for_qType,
    get_tensor_type_from_qType,
    pos2scale,
    quantize_data,
    scale2pos,
)

from .methods import PowerOfTwoMethod

logger = ScreenLogger(__name__)


def loading_data_from_disk(
    data_arr: list[str | NDArray[Any] | list[NDArray[Any]]],
    start_index: int | None = None,
    end_index: int | None = None,
) -> list[NDArray[Any] | list[NDArray[Any]]]:
    """
    Each element of the list data_arr corresponds to a sample from the data reader.

    If the list contains file paths, read the files and store their contents to a new list.
    If the elements are already Numpy arrays, return them directly. There is a special case
    for 'mostcommon' of MinMSE method, where the data is stored as a list containing rmin,
    rmax, and scale. It can be returned as is.

    Additionally, this function supports extracting a sublist from the original list by
    specifying the start index and end index of the elements.

    :param list[NDArray[Any] | str] data_arr: the list contains file path or numpy arrays
    :param int | None start_index: the start index. if None, start from 0
    :param int | None end_index: the end index (not included). If None, it ends at the end
    :return: The extracted list of numpy arrays
    """
    data_size = len(data_arr)
    assert data_size, "The list should be non empty"

    start = 0 if start_index is None else start_index
    end = data_size if end_index is None else end_index

    assert isinstance(start, int) and isinstance(end, int)
    assert 0 <= start < data_size, f"start_index {start} out of range [0, {data_size - 1}]"
    assert 0 < end <= data_size, f"end_index {end} out of range [1, {data_size}]"
    assert start < end, f"start_index {start} should be lower than end_index {end}"

    data_list: list[NDArray[Any]] = []
    if isinstance(data_arr[0], str):
        assert all(isinstance(item, str) for item in data_arr), "Not all elements are string"
        # Retrieve the raw data type of the tensor from the file name, which typically
        # follows the format of "output{output_index}_data_{dtype}.npystream” (contains multiple
        # arrays) or "output{output_index}_data_{dtype}.npy” (contains a single array)
        filename = os.path.basename(data_arr[0])
        dtype = filename.split("_data_")[-1].split(".")[0]
        with open(data_arr[0], "rb") as f:
            for index in range(data_size):
                d = np.load(f)
                if index < start or index >= end:
                    continue
                data_list.append(d.astype(np.dtype(dtype), copy=False))
    elif isinstance(data_arr[0], np.ndarray):
        assert all(isinstance(item, np.ndarray) for item in data_arr), "Not all elements are np.ndarray"
        data_list = data_arr[start:end]  # type: ignore
    else:
        # This is a special case for 'mostcommon' of MinMSE method
        return data_arr[start:end]  # type: ignore

    dtypes = {a.dtype for a in data_list}
    assert len(dtypes) == 1, f"The calibration expects only one element type but got {dtypes}"

    return data_list  # type: ignore


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
        optimize_mem: bool = True,
        worker_num: int = 1,
        layer_wise: bool = False,
    ) -> None:
        super().__init__(method, symmetric, num_bins, num_quantized_bins, percentile, scenario)

        self.optimize_mem = optimize_mem

        if worker_num > multiprocessing.cpu_count():
            logger.warning(
                f"The number of workers {worker_num} can not larger than cpu cores {multiprocessing.cpu_count()}"
            )
            self.worker_num = multiprocessing.cpu_count()
        else:
            self.worker_num = max(worker_num, 1)

        self.layerwise_percentile = True if self.method == "percentile" and layer_wise else False

    def collect(self, name_to_arr: dict[str, list[tuple[NDArray[Any], str]]]) -> None:
        # TODO: Currently we have different collect() for entropy and percentile method respectively.
        #       Need unified collect in the future.
        if self.method in {"distribution", "entropy"}:
            return self.collect_value(name_to_arr)
        elif self.method == "percentile":
            if self.symmetric:
                return self.collect_absolute_value(name_to_arr)
            else:
                return self.collect_value(name_to_arr)
        else:
            raise ValueError("Only 'entropy', 'percentile' or 'distribution' methods are supported")

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

        def compute_percentile_worker(tensor: str, histogram: tuple[Any]) -> None:
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
                delayed(compute_percentile_worker)(tensor, histogram) for tensor, histogram in histogram_dict.items()
            )
        else:
            for tensor, histogram in histogram_dict.items():
                compute_percentile_worker(tensor, histogram)

        return thresholds_dict

    def collect_absolute_value(self, name_to_arr: dict[str, list[tuple[NDArray[Any], str]]]) -> None:
        """
        Collect histogram on absolute value
        """

        def collect_absolute_value_worker(tensor: str, data_arr: list[tuple[NDArray[Any], str]]) -> None:
            if isinstance(data_arr, list):
                start_index = len(data_arr) - 1 if self.layerwise_percentile else None
                data_list = loading_data_from_disk(data_arr, start_index)
                data_arr_np = np.asarray(data_list)
            elif isinstance(data_arr, np.ndarray):
                data_arr_np = data_arr
            else:
                raise ValueError(f"Unexpected type {type(data_arr)} for tensor={tensor!r}")

            data_arr_np = data_arr_np.flatten()
            if data_arr_np.size > 0:
                min_value = np.nanmin(data_arr_np)
                max_value = np.nanmax(data_arr_np)
            else:
                min_value = np.array(0, dtype=data_arr_np.dtype)
                max_value = np.array(0, dtype=data_arr_np.dtype)

            data_arr_np = np.absolute(data_arr_np)  # only consider absolute value

            # Convert to float32 for histogram computation (float16 lacks precision for many bins)
            orig_dtype = data_arr_np.dtype
            if orig_dtype == np.float16:
                data_arr_np = data_arr_np.astype(np.float32)

            if tensor not in self.histogram_dict:
                # first time it uses num_bins to compute histogram.
                hist, hist_edges = np.histogram(data_arr_np, bins=self.num_bins)
                hist_edges = hist_edges.astype(orig_dtype)
                assert orig_dtype != np.float64, (
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
                hist_edges = hist_edges.astype(orig_dtype)
                hist[: len(old_hist)] += old_hist
                assert orig_dtype != np.float64, (
                    "only float32 or float16 is supported, every constant must be explicitly typed"
                )
                self.histogram_dict[tensor] = (hist, hist_edges, min(old_min, min_value), max(old_max, max_value))

        if self.worker_num > 1:
            Parallel(n_jobs=self.worker_num, backend="threading")(
                delayed(collect_absolute_value_worker)(tensor, data_arr) for tensor, data_arr in name_to_arr.items()
            )
        else:
            for tensor, data_arr in name_to_arr.items():
                collect_absolute_value_worker(tensor, data_arr)

    def collect_value(self, name_to_arr: dict[str, list[tuple[NDArray[Any], str]]]) -> None:
        """
        Collect histogram on real value
        """

        def collect_value_worker(tensor: str, data_arr: list[tuple[NDArray[Any], str]]) -> None:
            start_index = len(data_arr) - 1 if self.layerwise_percentile else None
            data_list = loading_data_from_disk(data_arr, start_index)
            data_arr_np = np.asarray(data_list)  # noqa: PLW2901
            data_arr_np = data_arr_np.flatten()  # noqa: PLW2901

            # Convert to float32 for histogram computation (float16 lacks precision for many bins)
            orig_dtype = data_arr_np.dtype
            if orig_dtype == np.float16:
                data_arr_np = data_arr_np.astype(np.float32)

            if data_arr_np.size > 0:
                min_value = np.nanmin(data_arr_np)
                max_value = np.nanmax(data_arr_np)
            else:
                min_value = np.array(0, dtype=orig_dtype)
                max_value = np.array(0, dtype=orig_dtype)

            threshold = np.array(max(abs(min_value), abs(max_value)), dtype=orig_dtype)

            if tensor in self.histogram_dict:
                old_histogram = self.histogram_dict[tensor]
                self.histogram_dict[tensor] = self.merge_histogram(
                    old_histogram, data_arr_np, min_value, max_value, threshold
                )
            else:
                hist, hist_edges = np.histogram(data_arr_np, self.num_bins, range=(-threshold, threshold))
                self.histogram_dict[tensor] = (
                    hist,
                    hist_edges,
                    min_value,
                    max_value,
                    threshold,
                )

        if self.worker_num > 1:
            Parallel(n_jobs=self.worker_num, backend="threading")(
                delayed(collect_value_worker)(tensor, data_arr) for tensor, data_arr in name_to_arr.items()
            )
        else:
            for tensor, data_arr in name_to_arr.items():
                collect_value_worker(tensor, data_arr)


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
                # This needs to calculate the quantization parameters using the original data
                rmin_mse, rmax_mse, _, scale_mse, _ = quantize_data(
                    data=d, qType=act_type, symmetric=symmetric, method=method
                )
            scale2threshold[float(scale_mse)] = (rmin_mse, rmax_mse)
            scale_list.append(scale_mse)
        u, indices = np.unique(scale_list, return_inverse=True)
        scale = u[np.argmax(np.bincount(indices))]

        return scale2threshold[scale]

    def _percentile_mode(
        data_arr: list[Any], act_type: Any, symmetric: Any, method: Any, percentile: Any
    ) -> tuple[Any, Any]:
        assert all(isinstance(item, np.ndarray) for item in data_arr), "Not all elements are np.ndarray"

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

        rmin_mse, rmax_mse, *_ = quantize_data(data=d, qType=act_type, symmetric=symmetric, method=method)
        return (rmin_mse, rmax_mse)

    def _all_arrays_mode(
        data_arr: list[Any],
        act_type: Any,
        symmetric: Any,
        method: Any,
        rmin: np.ndarray[Any, Any] | None = None,
        rmax: np.ndarray[Any, Any] | None = None,
        is_partial_cal: bool = True,
    ) -> tuple[Any, Any, Any, Any]:
        """
        Process all arrays mode for quantization data collection.
        Flattens the input data arrays and performs quantization data calculation
        using the specified quantization parameters.

        :param list[Any] data_arr: List of numpy arrays to be processed.
        :param Any act_type: Activation quantization type.
        :param Any symmetric: Whether to use symmetric quantization.
        :param Any method: Quantization method to use.
        :param float | None rmin: Optional minimum range value for quantization.
        :param float | None rmax: Optional maximum range value for quantization.
        :return: Tuple containing quantization data results.
        :rtype: tuple[Any, Any, Any, Any]
        """
        assert all(isinstance(item, np.ndarray) for item in data_arr), "Not all elements are np.ndarray"

        if _all_dims_equal(data_arr):
            # The np.array() requires all dims to be exactly the same
            d = np.array(data_arr).flatten()
        elif _nonbatch_dims_equal(data_arr):
            # The np.concatenate() requires array dims except for the concat axis must match exactly
            d = np.concatenate(data_arr, axis=0).flatten()
        else:
            raise ValueError("The dims of samples do not match exactly!")

        return quantize_data(
            data=d,
            qType=act_type,
            symmetric=symmetric,
            method=method,
            is_partial_cal=is_partial_cal,
            rmin_override=rmin,
            rmax_override=rmax,
        )

    if not tensor_data:
        raise ValueError(f"Missed data for the tensor {tensor_name}, please check.")

    act_type = activation_qType
    if tensor_name in quantized_tensor_type and quantized_tensor_type[tensor_name] in calib_quant_types:
        logger.info(
            f"The type of tensor {tensor_name} is {quantized_tensor_type[tensor_name]}, using specific tensor precision"
        )
        act_type = get_tensor_type_from_qType(quantized_tensor_type[tensor_name])

    if minmse_mode == "MostCommon" and symmetric:
        data_arr = loading_data_from_disk(tensor_data)
        threshold = _mostcommon_mode(data_arr, act_type, symmetric, method)
    elif minmse_mode == "Percentile":
        # This is an experimental parameter, not optimized yet
        data_arr = loading_data_from_disk(tensor_data)
        threshold = _percentile_mode(data_arr, act_type, symmetric, method, percentile)
    else:
        chunk_size = 1
        is_partial_cal = True
        chunk_diffs = []
        rmins = []
        rmaxs = []
        for i in range(math.ceil(len(tensor_data) / chunk_size)):
            chunk_data_arr = loading_data_from_disk(
                tensor_data, start_index=i * chunk_size, end_index=(i + 1) * chunk_size
            )
            rmin_temp = np.min(chunk_data_arr)
            rmax_temp = np.max(chunk_data_arr)
            rmins.append(rmin_temp)
            rmaxs.append(rmax_temp)
        rmin = np.array([np.min(rmins)])
        rmax = np.array([np.max(rmaxs)])
        for i in range(math.ceil(len(tensor_data) / chunk_size)):
            chunk_data_arr = loading_data_from_disk(
                tensor_data, start_index=i * chunk_size, end_index=(i + 1) * chunk_size
            )
            minmse_diffs, minmse_scales, minmse_zps, qmin, qmax = _all_arrays_mode(
                chunk_data_arr,
                act_type,
                symmetric,
                method,
                rmin,
                rmax,
                is_partial_cal=is_partial_cal,
            )
            chunk_diffs.append(minmse_diffs)
        merge_chunk_diffs = np.array(chunk_diffs).sum(axis=0)
        minmse_idx = np.argmin(merge_chunk_diffs)
        scale_mse = minmse_scales[minmse_idx]
        zp_mse = minmse_zps[minmse_idx]
        rmin_mse = (qmin.astype(np.float32) - zp_mse.astype(np.float32)) * scale_mse
        rmax_mse = (qmax.astype(np.float32) - zp_mse.astype(np.float32)) * scale_mse
        threshold = (np.array(rmin_mse, dtype=scale_mse.dtype), np.array(rmax_mse, dtype=scale_mse.dtype))

    return tensor_name, threshold


def compute_minmse_worker_unpack(args: Any) -> tuple[str, tuple[Any, Any]]:
    """This is a helper function to unpack the arguments"""
    return compute_minmse_worker(*args)


def compute_minmse_from_histogram(
    tensor_name: str,
    histogram: tuple[Any, Any, Any, Any],
    quantized_tensor_type: dict[Any, Any],
    activation_qType: Any,
    symmetric: Any,
    pos_range: int = 5,
) -> tuple[str, tuple[Any, Any]]:
    """Compute MinMSE calibration thresholds from a pre-built histogram.

    :param str tensor_name: Name of the tensor being calibrated.
    :param tuple histogram: A tuple ``(hist, hist_edges, rmin, rmax)`` where
        ``hist`` is an int64 array of bin counts, ``hist_edges`` is a float64
        array of bin edges (float64 avoids overflow for large float32 ranges),
        and ``rmin``/``rmax`` are float32 scalars giving the global observed min/max.
    :param dict quantized_tensor_type: Optional per-tensor type overrides.
    :param activation_qType: Default quantization type for activations.
    :param symmetric: Whether to use symmetric quantization.
    :param int pos_range: Number of power-of-two scale candidates to search.
    :return: ``(tensor_name, (rmin_mse, rmax_mse))``
    :rtype: tuple[str, tuple[Any, Any]]
    """
    if not symmetric:
        raise ValueError(
            f"compute_minmse_from_histogram only supports symmetric quantization; got symmetric={symmetric!r} for tensor {tensor_name!r}."
        )

    hist, hist_edges, rmin, rmax = histogram

    if hist.sum() == 0:
        raise ValueError(f"Empty histogram for tensor {tensor_name!r}, no data was collected.")

    act_type = activation_qType
    if tensor_name in quantized_tensor_type and quantized_tensor_type[tensor_name] in calib_quant_types:
        logger.info(
            f"The type of tensor {tensor_name} is {quantized_tensor_type[tensor_name]}, using specific tensor precision"
        )
        act_type = get_tensor_type_from_qType(quantized_tensor_type[tensor_name])

    qmin, qmax = get_qmin_qmax_for_qType(act_type, symmetric=symmetric)
    zero_point, scale = compute_scale_zp(rmin, rmax, qmin, qmax, act_type, PowerOfTwoMethod.MinMSE, symmetric=symmetric)

    bin_centres_f32 = ((hist_edges[:-1] + hist_edges[1:]) / 2).astype(np.float32)
    bin_centres_f64 = bin_centres_f32.astype(np.float64)
    hist_f64 = hist.astype(np.float64)

    best_diff = float("inf")
    scale_mse = scale
    for i in range(pos_range):
        s_i = np.array(pos2scale(scale2pos(float(scale)) + i - 1), dtype=np.float32)
        q_i = np.clip(
            np.round(bin_centres_f32 / s_i).astype(np.int32) + int(zero_point),
            int(qmin),
            int(qmax),
        )
        dq_i = (q_i.astype(np.float64) - float(zero_point)) * float(s_i)
        diff = float(np.sum(hist_f64 * (bin_centres_f64 - dq_i) ** 2))
        if diff < best_diff:
            best_diff = diff
            scale_mse = s_i

    rmin_mse = np.array((float(qmin) - float(zero_point)) * float(scale_mse), dtype=rmin.dtype)
    rmax_mse = np.array((float(qmax) - float(zero_point)) * float(scale_mse), dtype=rmax.dtype)
    return tensor_name, (rmin_mse, rmax_mse)


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
        num_bins: int = 2048,
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
        self.num_bins = num_bins
        self.percentile = percentile
        self.optimize_mem = optimize_mem
        self.worker_num = worker_num
        self.quantized_tensor_type = quantized_tensor_type

        self.all_with_histogram = True if self.minmse_mode == "All" and self.num_bins > 0 else False
        self.mostcommon_minmse = True if self.minmse_mode == "MostCommon" and self.symmetric else False

        self.histogram_dict: dict[str, tuple[Any, Any, Any, Any]] = {}  # For All mode with histogram-based mse
        self.name_to_arr: dict[Any, Any] = {}  # For MostCommon, the value is a 2D list storing the quant params

    def collect(self, name_to_arr: dict[Any, Any]) -> None:
        if self.all_with_histogram:
            return self.collect_histogram_value(name_to_arr)
        elif self.mostcommon_minmse:
            return self.collect_mostcommon_value(name_to_arr)
        else:
            return self.collect_value(name_to_arr)

    def collect_histogram_value(self, name_to_arr: dict[str, list[Any]]) -> None:
        """Collect one batch of tensor data into per-tensor histograms.

        For each tensor, the batch data is flattened and merged into an existing
        histogram using an expand-and-merge strategy: if the new data fits within the
        current histogram range, its counts are added directly; if the new data
        extends beyond the current range, the histogram is expanded with uniform-width
        bins on each side before merging.

        :param dict name_to_arr: Mapping from tensor name to a list of arrays (one per sample).
        """
        for tensor, data_arr in name_to_arr.items():
            data_list = loading_data_from_disk(data_arr)
            raw = np.concatenate([np.asarray(d).flatten() for d in data_list])
            if raw.dtype == np.float64:
                raise TypeError(
                    f"collect_histogram received float64 data for tensor {tensor!r}; "
                    "only float16/float32 inputs are supported."
                )
            orig_dtype = raw.dtype
            data = raw.astype(np.float32)
            # Replace NaN with 0.0 and clip ±inf to float32 extremes, matching sanitize_model_outputs.
            data = np.nan_to_num(data, nan=0.0)
            new_rmin = np.array(data.min(), dtype=orig_dtype)
            new_rmax = np.array(data.max(), dtype=orig_dtype)

            if tensor not in self.histogram_dict:
                # Use float64 for edges to avoid overflow when data spans the full float32 range.
                hist, edges = np.histogram(data.astype(np.float64), bins=self.num_bins)
                self.histogram_dict[tensor] = (hist, edges, new_rmin, new_rmax)
                continue

            old_hist, old_edges, old_rmin, old_rmax = self.histogram_dict[tensor]
            lo, hi = old_edges[0], old_edges[-1]
            width = float(old_edges[1] - old_edges[0])

            if float(new_rmin) >= lo and float(new_rmax) <= hi:
                new_hist, _ = np.histogram(data.astype(np.float64), bins=old_edges)
                self.histogram_dict[tensor] = (
                    old_hist + new_hist,
                    old_edges,
                    np.minimum(old_rmin, new_rmin),
                    np.maximum(old_rmax, new_rmax),
                )
            else:
                left_extra = max(0, int(np.ceil((lo - float(new_rmin)) / width)))
                right_extra = max(0, int(np.ceil((float(new_rmax) - hi) / width)))
                new_lo = lo - left_extra * width
                new_hi = hi + right_extra * width
                new_num_bins = len(old_hist) + left_extra + right_extra
                new_edges = np.linspace(new_lo, new_hi, new_num_bins + 1)
                expanded_hist = np.zeros(new_num_bins, dtype=old_hist.dtype)
                expanded_hist[left_extra : left_extra + len(old_hist)] = old_hist
                new_hist, _ = np.histogram(data.astype(np.float64), bins=new_edges)
                self.histogram_dict[tensor] = (
                    expanded_hist + new_hist,
                    new_edges,
                    np.minimum(old_rmin, new_rmin),
                    np.maximum(old_rmax, new_rmax),
                )

    def collect_mostcommon_value(self, name_to_arr: dict[Any, Any]) -> None:
        """Collect data for the most common mode"""
        for tensor_name, data_arr in name_to_arr.items():
            act_type = self.activation_qType
            if (
                tensor_name in self.quantized_tensor_type
                and self.quantized_tensor_type[tensor_name] in calib_quant_types
            ):
                act_type = get_tensor_type_from_qType(self.quantized_tensor_type[tensor_name])

            assert len(data_arr) == 1, "Each time it only supports collecting a sample"

            data_list = loading_data_from_disk(data_arr)

            rmin_mse, rmax_mse, _, scale_mse, _ = quantize_data(
                data=data_list[0],
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
        if not self.name_to_arr and not self.histogram_dict:
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
        if self.histogram_dict:
            return self._compute_minmse_from_histograms()
        else:
            return self._compute_minmse_from_raw()

    def _compute_minmse_from_histograms(self) -> dict[Any, Any]:
        """Compute MinMSE thresholds for all tensors using the accumulated histograms.

        Called when ``histogram_dict`` is populated (i.e. "All" mode with histogram
        collection). Iterates over each tensor's histogram and delegates to
        ``compute_minmse_from_histogram`` to find the power-of-two scale that
        minimises MSE. Supports parallel execution via ``self.worker_num``.

        :return: Mapping from tensor name to ``(rmin, rmax)`` threshold tuple.
        :rtype: dict[str, tuple[Any, Any]]
        """
        thresholds_dict: dict[str, tuple[Any, Any]] = {}
        if self.worker_num <= 1:
            for tensor, histogram in tqdm(self.histogram_dict.items()):
                name, threshold = compute_minmse_from_histogram(
                    tensor,
                    histogram,
                    self.quantized_tensor_type,
                    self.activation_qType,
                    self.symmetric,
                )
                thresholds_dict[name] = threshold
        else:
            results = Parallel(n_jobs=self.worker_num, backend="threading")(
                delayed(compute_minmse_from_histogram)(
                    tensor,
                    histogram,
                    self.quantized_tensor_type,
                    self.activation_qType,
                    self.symmetric,
                )
                for tensor, histogram in tqdm(self.histogram_dict.items())
            )
            for name, threshold in results:
                thresholds_dict[name] = threshold
        return thresholds_dict

    def _compute_minmse_from_raw(self) -> dict[Any, Any]:
        """Compute MinMSE thresholds for all tensors using the raw accumulated data.

        Called when ``histogram_dict`` is empty, i.e. for "MostCommon" and "Percentile"
        modes, or for "All" mode when histogram collection is disabled. Processes
        ``self.name_to_arr`` directly via ``compute_minmse_worker``.

        :return: Mapping from tensor name to ``(rmin, rmax)`` threshold tuple.
        :rtype: dict[str, tuple[Any, Any]]
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
