# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
Common matmul_noqdq functions
"""

from collections.abc import Sequence
from typing import cast

import numpy as np
import onnx
import onnx_tool

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import is_static_shape


def _get_index(shape: Sequence[int]) -> int:
    if len(shape) == 2:
        input_index = 0
    elif len(shape) == 3:
        assert shape[0] == 1, shape[0]
        input_index = 1
    elif len(shape) == 4:
        assert shape[0] == 1, shape[0]
        assert shape[1] == 1, shape[1]
        input_index = 2
    else:
        raise ValueError(f"Unsupported input shape of size {len(shape)}")
    return input_index


def _get_matmul_params(
    input_shape: tuple[int, ...], weight_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> tuple[int, int, int]:
    input_index = _get_index(input_shape)
    output_index = _get_index(output_shape)
    weight_index = _get_index(weight_shape)
    m = input_shape[input_index]
    assert m == output_shape[output_index]
    k = input_shape[input_index + 1]
    assert k == weight_shape[weight_index]
    n = output_shape[output_index + 1]
    assert n == weight_shape[weight_index + 1]
    return (m, k, n)


def get_matmul_params(matmul: onnx.NodeProto, extractor: onnx.utils.Extractor) -> tuple[int, int, int]:
    input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[1], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.output[0], extractor)

    assert is_static_shape(input_shape) and is_static_shape(weight_shape) and is_static_shape(output_shape)

    if "(MatMul_transpose)" in matmul.name:
        m, k, n = output_shape
        output_shape = (m, n, k)

    return _get_matmul_params(input_shape, weight_shape, output_shape)


def is_mm_supported(matmul: onnx.NodeProto, op_namespace: str, extractor: onnx.utils.Extractor) -> bool:
    supported_shapes = {
        "sdxlt": {
            (4096, 640, 640),
            (4096, 640, 5120),
            (4096, 2560, 640),
            (1024, 1280, 1280),
            (1024, 1280, 10240),
            (1024, 5120, 1280),
            (77, 2048, 640),
            (77, 2048, 1280),
            (77, 1280, 1280),
            (77, 5120, 1280),
            (1, 1280, 1280),
            (256, 1280, 1280),
            (256, 5120, 1280),
            (256, 1280, 10240),
            (1024, 640, 640),
            (1024, 640, 5120),
            (1024, 2560, 640),
            (4096, 512, 512),
        }
    }

    m, k, n = get_matmul_params(matmul, extractor)
    if (m, k, n) not in supported_shapes[op_namespace]:
        return False

    initializers = ryzenai_onnx_utils.matcher.get_initializers(matmul.input[1], extractor, False)
    return len(initializers) == 1


def get_empty_bias(shape: Sequence[int], extractor: onnx.utils.Extractor) -> str | None:
    for name, initializer in extractor.wmap.items():
        name = cast(str, name)
        if name.endswith(".empty_bias") and ryzenai_onnx_utils.matcher.get_shape(initializer) == shape:
            dtype = ryzenai_onnx_utils.matcher.get_dtype(name, extractor)
            np_dtype = onnx_tool.tensor.onnxdtype2npdtype(dtype)
            assert np_dtype is not None
            buffer = np.frombuffer(initializer.raw_data, np_dtype)
            # if all zeros
            if not np.any(buffer):
                return name
    return None
