# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    input_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    sizes = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(node.input[2], extractor).tolist()

    assert len(input_shape) == len(sizes)

    output_shape = [int(x * y) for x, y in zip(input_shape, sizes, strict=True)]

    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)

    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)

    extractor.vimap[node.output[0]] = tvi
