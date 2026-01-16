# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    get_attribute,
)
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    WhiteboxBasePass,
    register_whitebox_pass,
)
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype,
    add_cast_dtype_to_bfloat16,
)
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDSlice")
class SDSlicePass(WhiteboxBasePass):
    whitebox_flow_op_type = "Slice"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd3": {
                # mmdit 512
                ((2, 1178, 1536), (2, 1024, 1536)),
                ((2, 1178, 1536), (2, 154, 1536)),
                # mmdit 1024
                ((2, 4250, 1536), (2, 4096, 1536)),
                ((2, 4250, 1536), (2, 154, 1536)),
                # mmdit seq-160
                ((2, 1184, 1536), (2, 1024, 1536)),  # 512x512
                ((2, 1184, 1536), (2, 160, 1536)),  # 512x512
                ((2, 4256, 1536), (2, 4096, 1536)),  # 1024x1024
                ((2, 4256, 1536), (2, 160, 1536)),  # 1024x1024
            }
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        output_shape = tuple(check_shapes["output_shape"][0])
        return (input_shape, output_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                get_attribute(node, "input_shape"),
            ],
            "output_shape": [
                get_attribute(node, "output_shape"),
            ],
        }
        return shape_lists


def is_slice_supported_pattern(extractor, slice):
    input_shape = ryzenai_onnx_utils.matcher.get_shape(slice.input[0], extractor)
    return len(input_shape) == 3


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDSlice")

    slice = subgraph[0]
    if not is_slice_supported_pattern(extractor, slice):
        return subgraph, [], None
    input_shape = ryzenai_onnx_utils.matcher.get_shape(slice.input[0], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(slice.output[0], extractor)
    if len(input_shape) != 3:
        return subgraph, [], None

    tvis = []

    pre_cast_output = slice.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
        slice.input[0],
        pre_cast_output,
        input_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(slice.input[0], extractor),
    )
    tvis.extend(pre_cast_tvi)

    new_inputs = [pre_cast_output]
    # slice kernel requirement: additional weights buffer which lengths is 128
    wts_data = np.zeros(128, dtype=np.uint8)
    wts_name = slice.name + f".wts{pass_id}"
    # weights buffer size is 128, so we need to create weights buffer tensor with shape (128,)
    wts_tvi = onnx.helper.make_tensor_value_info(wts_name, onnx.TensorProto.UINT8, (128,))
    tvis.append(wts_tvi)
    wts_init = onnx.helper.make_tensor(wts_name, onnx.TensorProto.UINT8, (128,), wts_data.tobytes(), True)
    new_inputs.append(wts_name)

    slice_out = slice.output[0] + f".out{pass_id}"
    sd_slice = onnx.helper.make_node(
        "SDSlice",
        inputs=new_inputs,
        outputs=[slice_out],
        domain=domain,
        name=slice.name + "_bf16",
    )
    add_attribute(sd_slice, "input_shape", input_shape)
    add_attribute(sd_slice, "output_shape", output_shape)
    add_attribute(sd_slice, "weight_shape", [128])
    add_attribute(sd_slice, "in_dtypes", ["bfloat16"])
    add_attribute(sd_slice, "out_dtypes", ["bfloat16"])
    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        slice_out,
        slice.output[0],
        ryzenai_onnx_utils.matcher.get_shape(subgraph[-1].output[0], extractor),
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(slice.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast, sd_slice, *post_cast], [wts_init], tvis


PATTERN = ["Slice([?,?,?,?,?],[?])"]
REPLACEMENT = replacement
