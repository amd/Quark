# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import copy_attributes
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported(extractor, pad, conv) -> bool:
    if ryzenai_onnx_utils.matcher.has_multiple_successors(pad.output[0], extractor.graph):
        return False
    pad_mode = onnx.helper.get_node_attr_value(pad, "mode").decode()
    if pad_mode != "constant":
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(pad.input[2], extractor):
        return False
    pad_constant_value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(pad.input[2], extractor)
    if pad_constant_value != 0:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(pad.input[1], extractor):
        return False
    pad_pads = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(pad.input[1], extractor)
    if not all(pad_value >= 0 for pad_value in pad_pads):
        return False
    pad_input_shape = ryzenai_onnx_utils.matcher.get_shape(pad.input[0], extractor)
    if len(pad_pads) != len(pad_input_shape) * 2:
        return False
    pads_pixel_start = pad_pads[: len(pad_input_shape)]
    pads_pixel_end = pad_pads[len(pad_input_shape) :]
    return not (
        pads_pixel_start[0] != 0 or pads_pixel_start[-1] != 0 or pads_pixel_end[0] != 0 or pads_pixel_end[-1] != 0
    )


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    (
        pad,
        conv,
    ) = subgraph
    if not is_supported(extractor, pad, conv):
        return subgraph, [], None
    assert len(pad.input) == 3
    assert len(conv.input) == 3
    pad_input_shape = ryzenai_onnx_utils.matcher.get_shape(pad.input[0], extractor)
    pad_pads = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(pad.input[1], extractor)
    pad_pads_left = pad_pads[: len(pad_input_shape)]
    pad_pads_right = pad_pads[len(pad_input_shape) :]

    pad_pads = np.hstack(
        (
            pad_pads_left[1 : len(pad_input_shape) - 1],
            pad_pads_right[1 : len(pad_input_shape) - 1],
        )
    )
    conv_pads = onnx.helper.get_node_attr_value(conv, "pads")
    conv_pads = np.add(pad_pads, conv_pads)

    conv_node = onnx.helper.make_node(
        conv.op_type,
        inputs=[pad.input[0], conv.input[1], conv.input[2]],
        outputs=conv.output,
        name=conv.name,
        domain="com.microsoft",
    )
    copy_attributes(conv, conv_node)
    ryzenai_onnx_utils.matcher.set_attribute(conv_node, "pads", conv_pads)

    return [conv_node], [], None


PATTERN = ["Pad([?,?,?], a0)", "NhwcConv([a0,?,?], ?)"]
REPLACEMENT = replacement
