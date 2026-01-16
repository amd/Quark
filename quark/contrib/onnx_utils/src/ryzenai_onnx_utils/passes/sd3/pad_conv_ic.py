# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.pattern_generator
from ryzenai_onnx_utils.matcher import copy_attributes
from ryzenai_onnx_utils.transform.transpose import add_transpose
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_ic_needs_padding(transpose: onnx.NodeProto, conv: onnx.NodeProto, extractor: onnx.utils.Extractor) -> bool:
    conv_ifm_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[0], extractor)
    permute = onnx.helper.get_node_attr_value(transpose, "perm")
    if permute != [0, 2, 3, 1]:
        return False
    needs_pading_ic = [3, 17]
    return conv_ifm_shape[-1] in needs_pading_ic


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (transpose, conv) = subgraph
    # assert len(conv.input) == 3
    if not is_ic_needs_padding(transpose, conv, extractor):
        return subgraph, [], None
    N, C, H, W = ryzenai_onnx_utils.matcher.get_shape(transpose.input[0], extractor)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(transpose.input[0], extractor)
    assert (C == 3) or (C == 17)
    after_padding = {
        "3": 4,
        "17": 24,
    }
    padded_ic = after_padding[f"{C}"]
    tvis = []
    input_padded_ic_tvi = onnx.helper.make_tensor_value_info(
        transpose.input[0] + f"_ic{padded_ic}", dtype, [N, padded_ic, H, W]
    )
    tvis.append(input_padded_ic_tvi)
    pad_pads_tensor = onnx.helper.make_tensor(
        transpose.input[0] + "_pads",
        onnx.TensorProto.INT64,
        [8],
        [0, 0, 0, 0, 0, padded_ic - C, 0, 0],
    )
    pad_value_tensor = onnx.helper.make_tensor(
        transpose.input[0] + "_value",
        dtype,
        [],
        [0.0],
    )
    pad_node = onnx.helper.make_node(
        "Pad",
        inputs=[transpose.input[0], pad_pads_tensor.name, pad_value_tensor.name],
        outputs=[input_padded_ic_tvi.name],
        mode="constant",
    )

    padded_ic_transpose_tvi = onnx.helper.make_tensor_value_info(transpose.output[0], dtype, [N, H, W, padded_ic])
    transpose_padded_ic_node, transpose_tvis = add_transpose(
        transpose.name + f"_ic{padded_ic}",
        pad_node.output[0],
        padded_ic_transpose_tvi.name,
        dtype,
        [N, padded_ic, H, W],
        [N, H, W, padded_ic],
        [0, 2, 3, 1],
    )
    tvis.extend(transpose_tvis)

    wts_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(conv.input[1], extractor)
    wts_data_padded_ic = np.pad(
        wts_data,
        ((0, 0), (0, 0), (0, 0), (0, padded_ic - C)),
        mode="constant",
        constant_values=0,
    )
    wts_shape = wts_data.shape
    wts_padded_ic_shape = list(wts_shape)
    wts_padded_ic_shape[-1] = padded_ic
    wts_padded_ic_tensor = onnx.helper.make_tensor(
        conv.input[1],
        ryzenai_onnx_utils.matcher.get_dtype(conv.input[1], extractor),
        wts_data_padded_ic.shape,
        wts_data_padded_ic.tobytes(),
        True,
    )
    conv_input = [transpose_padded_ic_node.output[0], wts_padded_ic_tensor.name]
    if len(conv.input) == 3:
        conv_input.append(conv.input[2])
    conv_node = onnx.helper.make_node(
        conv.op_type,
        inputs=conv_input,
        outputs=conv.output,
        name=conv.name,
        domain="com.microsoft",
    )
    copy_attributes(conv, conv_node)
    return (
        [pad_node, transpose_padded_ic_node, conv_node],
        [pad_pads_tensor, pad_value_tensor, wts_padded_ic_tensor],
        transpose_tvis,
    )


PATTERN = ["Transpose([?], a0)", "NhwcConv([a0,?,?], ?)"]
REPLACEMENT = replacement
