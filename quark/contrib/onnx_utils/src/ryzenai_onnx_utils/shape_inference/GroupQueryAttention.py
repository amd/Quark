# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    activation_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)

    # assuming that node is not using packed QKV!

    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, activation_shape)

    extractor.vimap[node.output[0]] = tvi
