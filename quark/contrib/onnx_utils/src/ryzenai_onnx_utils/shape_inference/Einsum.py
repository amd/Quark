# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from collections import defaultdict

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import are_static_shapes


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    equation: str = onnx.helper.get_node_attr_value(node, "equation").decode()
    input_shapes = ryzenai_onnx_utils.matcher.get_shapes(node.input, extractor)
    assert are_static_shapes(input_shapes)
    input_eq, output_eq = equation.split("->")
    input_labels = input_eq.split(",")

    dim_sizes: defaultdict[str, int] = defaultdict(int)
    for labels, shape in zip(input_labels, input_shapes, strict=True):
        for label, size in zip(labels, shape, strict=True):
            if dim_sizes[label] == 0:
                dim_sizes[label] = size

    output_shape = tuple(dim_sizes[label] for label in output_eq)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)
    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)

    extractor.vimap[node.output[0]] = tvi
