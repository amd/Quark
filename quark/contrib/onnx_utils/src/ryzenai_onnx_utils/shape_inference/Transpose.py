# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    activation_shape = np.array(ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor))
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)

    perm = onnx.helper.get_node_attr_value(node, "perm")
    transposed_shape = activation_shape[perm].tolist()

    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, transposed_shape)

    extractor.vimap[node.output[0]] = tvi
