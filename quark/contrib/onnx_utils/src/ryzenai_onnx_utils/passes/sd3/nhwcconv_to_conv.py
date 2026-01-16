# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform.transpose import add_transpose
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (conv,) = subgraph
    it_nchw = conv.input[0] + "_nchw"
    dtype = ryzenai_onnx_utils.matcher.get_dtype(conv.input[0], extractor)
    in_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[0], extractor)
    wts_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[1], extractor)
    out_shape = ryzenai_onnx_utils.matcher.get_shape(conv.output[0], extractor)
    tvis = []
    transpose, transpose_tvis = add_transpose(
        conv.name + "in_transpose",
        conv.input[0],
        it_nchw,
        dtype,
        in_shape,
        [in_shape[0], in_shape[3], in_shape[1], in_shape[2]],
        [0, 3, 1, 2],
    )
    tvis.extend(transpose_tvis)

    wts_tvi = onnx.helper.make_tensor_value_info(
        conv.input[1] + "_nchw",
        dtype,
        [wts_shape[0], wts_shape[3], wts_shape[1], wts_shape[2]],
    )
    tvis.append(wts_tvi)
    wts_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv.input[1], extractor)
    wts_tensor = onnx.helper.make_tensor(
        wts_tvi.name,
        dtype,
        [wts_shape[0], wts_shape[3], wts_shape[1], wts_shape[2]],
        np.transpose(wts_data, (0, 3, 1, 2)).tobytes(),
        True,
    )

    conv_out_tvi = onnx.helper.make_tensor_value_info(
        conv.name + "out_transpose",
        dtype,
        [out_shape[0], out_shape[3], out_shape[1], out_shape[2]],
    )
    new_conv = onnx.helper.make_node(
        "Conv",
        [it_nchw, wts_tvi.name, conv.input[2]],
        [conv_out_tvi.name],
        name=conv.name,
        domain="ai.onnx",
    )
    ryzenai_onnx_utils.matcher.copy_attributes(conv, new_conv)

    out_transpose, out_transpose_tvis = add_transpose(
        conv.name + "out_transpose",
        conv_out_tvi.name,
        conv.output[0],
        dtype,
        [out_shape[0], out_shape[3], out_shape[1], out_shape[2]],
        out_shape,
        [0, 2, 3, 1],
    )
    tvis.extend(out_transpose_tvis)
    return [transpose, new_conv, out_transpose], [wts_tensor], tvis


PATTERN = ["NhwcConv([?,?,?],?)"]
REPLACEMENT = replacement
