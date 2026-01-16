# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    activation_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[1], extractor)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)

    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, [*activation_shape[:-1], weight_shape[0]])

    extractor.vimap[node.output[0]] = tvi
