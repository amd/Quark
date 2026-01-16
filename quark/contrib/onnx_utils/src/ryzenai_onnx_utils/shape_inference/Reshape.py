# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import is_static_shape


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    input_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    assert is_static_shape(input_shape)
    shape = ryzenai_onnx_utils.matcher.get_initializer_or_const(node.input[1], extractor)

    input_prod = np.prod(input_shape)
    output_shape = []
    inferred_index = None

    for i, dim_i in enumerate(shape):
        if dim_i == -1:
            if inferred_index is not None:
                raise ValueError("Only one -1 is allowed in the target shape.")
            inferred_index = i
            output_shape.append(-1)  # Placeholder for now
        elif dim_i == 0:
            if i >= len(input_shape):
                raise ValueError(f"Cannot inherit dimension at index {i} from input shape.")
            output_shape.append(input_shape[i])  # Inherit from input
        elif dim_i > 0:
            output_shape.append(dim_i.item())
        else:
            raise ValueError(f"Invalid target dimension: {dim_i}.")

    # Replace -1 with the inferred dimension
    if inferred_index is not None:
        inferred_dim = input_prod // np.prod([dim for dim in output_shape if dim != -1])
        if input_prod % np.prod([dim for dim in output_shape if dim != -1]) != 0:
            raise ValueError("Cannot infer -1: input size not divisible by known target size.")
        output_shape[inferred_index] = inferred_dim.item()

    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)
    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)
    extractor.vimap[node.output[0]] = tvi
