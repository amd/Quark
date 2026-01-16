#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import operator
import os
from functools import reduce
from typing import Any

import numpy as np
import onnxruntime as ort

from quark.onnx.operators.custom_ops import get_library_path
from quark.onnx.utils.model_utils import create_infer_session_for_onnx_model


def validate_search_space(search_space: dict[str, Any]) -> dict[str, Any]:
    def is_continuous(param: Any) -> bool:
        return isinstance(param, dict) and param.get("type") in {"float", "int"}

    def get_effective_params(value: Any, parent_keys: dict[str, Any] | None = None) -> Any:
        if parent_keys is None:
            parent_keys = {}

        if isinstance(value, dict):
            sizes = []
            contains_cont = False
            for k, v in value.items():
                child_sizes, has_cont = get_effective_params(v, parent_keys)
                sizes.extend(child_sizes)
                contains_cont = contains_cont or has_cont
            return sizes, contains_cont

        elif isinstance(value, list):
            return ([len(value)], any(is_continuous(v) for v in value))

        elif isinstance(value, dict) and is_continuous(value):
            return ([], True)

        return ([], False)

    total_sizes = []
    contains_continuous = False

    for key, value in search_space.items():
        sizes, has_cont = get_effective_params(value)
        total_sizes.extend(sizes)
        contains_continuous = contains_continuous or has_cont

    # Compute the product of all sizes
    discrete_space_size = reduce(operator.mul, total_sizes, 1)

    return {"discrete_space_size": discrete_space_size, "contains_continuous": contains_continuous}


def buildin_eval_func(onnx_path: str, data_loader: Any, save_path: str = "", save_prefix: str = "iter_x_") -> str:
    """
    Buildin evalation function using data_loader

    Args:
        onnx_path: onnx model path that will excute evalution, it can be  either float porint or quantized onnx model
        data_loader: user defined data_loader
        save_path: path used to save the output result
        save_prefix: prefix string used to name the saved output

    Note: Data reader here should be defined as dataloader.Because the raw data reader is iterator, it's
           not convient for evaluation.
    """
    # TODO check the onnx
    if "ROCMExecutionProvider" in ort.get_available_providers():
        device = "ROCM"
        providers = ["ROCMExecutionProvider"]
    elif "CUDAExecutionProvider" in ort.get_available_providers():
        device = "CUDA"
        providers = ["CUDAExecutionProvider"]
    else:
        device = "CPU"
        providers = ["CPUExecutionProvider"]
    so = ort.SessionOptions()
    so.register_custom_ops_library(get_library_path(device))
    ort_session = create_infer_session_for_onnx_model(onnx_path, so, providers=providers)

    for excute_idx, data in enumerate(data_loader):
        output = ort_session.run(None, data)
        if save_path is not None:
            temp_save_path = os.path.join(save_path, str(save_prefix) + str(excute_idx) + ".npy")
            output = np.concatenate([item.reshape(-1) for item in output])
            np.save(temp_save_path, output)

    return save_path
