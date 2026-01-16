# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    activation_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    if not ryzenai_onnx_utils.matcher.is_initializer(node.input[1], extractor):
        return
    pads = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(node.input[1], extractor)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)
    output_shape = []
    for i in range(len(activation_shape)):
        output_shape.append(activation_shape[i] + pads[i].item() + pads[i + len(activation_shape)].item())
    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)

    extractor.vimap[node.output[0]] = tvi
