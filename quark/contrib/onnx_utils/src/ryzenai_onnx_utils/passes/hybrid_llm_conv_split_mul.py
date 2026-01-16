# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces the Conv+Split+Mul pattern with a custom operator version.
"""

import numpy as np
import onnx
from onnx import TensorProto

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    in_cast0 = subgraph[0]
    in_cast1 = subgraph[1]
    split = subgraph[2]
    concat = subgraph[4]
    conv = subgraph[5]
    out_cast0 = subgraph[7]
    slice = subgraph[8]
    pad = subgraph[9]
    out_cast1 = subgraph[10]

    new_nodes = []
    new_initializers = []
    new_tvis = []

    wts_fp = (ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv.input[1], extractor)).astype(np.float32)
    wts_bf16 = (wts_fp.view(np.uint32) >> 16).astype(np.uint16)
    conv_wts_name = conv.input[1]  # + ".t"
    wts_tensor = onnx.numpy_helper.from_array(wts_bf16, conv_wts_name)
    wts_tensor.data_type = TensorProto.BFLOAT16
    new_initializers.append(wts_tensor)

    # add conv+slice
    new_conv = onnx.helper.make_node(
        "ConvSplitMul",
        inputs=[
            in_cast0.input[0],
            in_cast1.input[0],
            split.input[1],
            conv_wts_name,
            slice.input[1],
            slice.input[2],
            slice.input[3],
            pad.input[1],
        ],
        outputs=[out_cast0.output[0], out_cast1.output[0]],
        name=conv.name,  # + f"_{pass_id}",
        domain=params.get_domain(conv.op_type),
    )
    ryzenai_onnx_utils.matcher.copy_attributes(conv, new_conv)
    ryzenai_onnx_utils.matcher.set_attribute(
        new_conv, "concat_axis", int(ryzenai_onnx_utils.matcher.get_attribute(concat, "axis"))
    )
    ryzenai_onnx_utils.matcher.set_attribute(
        new_conv, "split_axis", int(ryzenai_onnx_utils.matcher.get_attribute(split, "axis"))
    )
    new_nodes.append(new_conv)

    return new_nodes, new_initializers, new_tvis


REPLACEMENT = replacement
PATTERN = [
    "CastAvx(?, cast1_out)",
    "CastAvx(?, cast2_out)",
    "Split(cast2_out, [split0, split1, split2])",
    "Mul([split0, split2], mul1_out)",
    "Concat([cast1_out, mul1_out], concat_out)",
    "Conv([concat_out, ?], conv_out)",
    "Mul([split1, conv_out], mul2_out)",
    "CastAvx(mul2_out, ?)",
    "Slice([concat_out, ?, ?, ?], slice_out)",
    "Pad([slice_out, ?], pad_out)",
    "CastAvx(pad_out, ?)",
]
