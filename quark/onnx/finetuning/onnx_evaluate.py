#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from quark.shares.utils.log import ScreenLogger, log_errors
import numpy as np
import onnx
import onnxruntime

from pathlib import Path
from typing import Union, List, Any
from quark.onnx.quant_utils import (register_custom_ops_library, CachedDataReader, create_infer_session_for_onnx_model)

logger = ScreenLogger(__name__)


def create_session(onnx_model: Union[str, Path, onnx.ModelProto]) -> onnxruntime.InferenceSession:
    """
    Create a inference session for the onnx model and register libraries for it.
    :param onnx_model: the proto or the path of the onnx model
    :return: the created inference session
    """
    so = onnxruntime.SessionOptions()

    providers: List[str] = []
    if 'ROCMExecutionProvider' in onnxruntime.get_available_providers():
        providers.append('ROCMExecutionProvider')
        register_custom_ops_library(so, device='ROCM')
    elif 'CUDAExecutionProvider' in onnxruntime.get_available_providers():
        providers.append('CUDAExecutionProvider')
        register_custom_ops_library(so, device='CUDA')
    else:
        register_custom_ops_library(so, device='CPU')
    providers.append('CPUExecutionProvider')

    # Note that we disabled all the graph optimizations because we found that ort
    # turns the QDQ into QOP to accelerate the inference, but this process leads
    # to a loss of precision both for Int8 and Int16 QDQs (especially the latter).
    so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL

    return create_infer_session_for_onnx_model(onnx_model, sess_options=so, providers=providers)


def inference_model(onnx_model: Union[str, Path, onnx.ModelProto],
                    data_reader: CachedDataReader,
                    data_num: Union[int, None] = None,
                    output_index: Union[int, None] = None) -> List[List[np.ndarray[Any, Any]]]:
    """
    Run the onnx model and feeding it with the data from the cached data reader.
    :param onnx_model: the proto or the path of the onnx model
    :param data_reader: the cached data reader
    :param data_num: how many samples will be used in the data reader
    :param output_index: which output will be chosen to calculate L2
    :return: the results after inference
    """
    session = create_session(onnx_model)
    data_reader.reset_iter()

    results: List[Any] = []

    while True:
        input_dict = data_reader.get_next()
        if not input_dict or (data_num is not None and len(results) > data_num):
            break

        result = session.run(None, input_dict)
        if output_index is not None and output_index >= 0 and output_index < len(session.get_outputs()):
            results.append([result[output_index]])
        else:
            results.append(result)

    return results


@log_errors
def average_L2(float_results: List[List[np.ndarray[Any, Any]]], quant_results: List[List[np.ndarray[Any, Any]]]) -> Any:
    """
    Calculate the average L2 distance between the float model and the quantized model.
    :param float_results: the result of the float model
    :param quant_results: the result of the quant model
    :return: the average L2 distance
    """
    data_num = len(float_results)
    if data_num <= 0 or len(quant_results) != data_num:
        raise ValueError("The number of results for calculating L2 does not match.")
        return None

    out_num = len(float_results[0])
    if out_num <= 0 or len(quant_results[0]) != out_num:
        raise ValueError("The number of outputs for calculating L2 does not match.")
        return None

    l2_distances = []

    for i in range(0, data_num):
        for j in range(0, out_num):
            l2_distance = np.linalg.norm(
                np.array(float_results[i][j]).astype(np.float32) - np.array(quant_results[i][j]).astype(np.float32))
            l2_distances.append(l2_distance)

    return np.mean(l2_distances).item()
