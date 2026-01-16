# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.utils
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
from ryzenai_onnx_utils.utils import float_numpy_to_bfloat_tensor


@register_whitebox_pass("SDGelu")
class SDGeluPass(WhiteboxBasePass):
    whitebox_flow_op_type = "Gelu"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        #   M, K, N
        supported_shapes = {
            "sd15": {
                # unet
                (2, 4096, 1280),  # count 5
                (2, 1024, 2560),  # count 5
                (2, 256, 5120),  # count 5
                (2, 64, 5120),  # count 1
            },
            "sd3": {
                # mmdit 512
                (2, 1024, 6144),
                (2, 154, 6144),
                # mmdit  1024 new
                (2, 4096, 6144),
                # mmdit seq-160
                (2, 160, 6144),
            },
            "phi3.5": {
                (1, 577, 4096),  # count 23
            },
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        return input_shape in supported_shapes[op_namespace]

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


def get_gelu_params(gelu, extractor):
    input_shape = ryzenai_onnx_utils.matcher.get_shape(gelu.input[0], extractor)
    B, M, N = input_shape
    return B, M, N


def is_gelu_supported_pattern(extractor: onnx.utils.Extractor, gelu_node: onnx.NodeProto) -> bool:
    in_shape = ryzenai_onnx_utils.matcher.get_shape(gelu_node.input[0], extractor)
    return len(in_shape) == 3


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDGelu")
    (gelu,) = subgraph

    assert len(gelu.input) == 1
    assert len(gelu.output) == 1

    if not is_gelu_supported_pattern(extractor, gelu):
        return subgraph, [], None

    B, M, N = get_gelu_params(gelu, extractor)
    tvis = []

    pre_cast_output = gelu.input[0] + f".out{pass_id}"
    gelu_input_shape = ryzenai_onnx_utils.matcher.get_shape(gelu.input[0], extractor)
    gelu_output_shape = ryzenai_onnx_utils.matcher.get_shape(gelu.output[0], extractor)
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
        gelu.input[0],
        pre_cast_output,
        gelu_input_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(gelu.input[0], extractor),
    )
    tvis.extend(pre_cast_tvi)
    # add addition wts tensor for xrt
    wts_name = gelu.name + f".weights{pass_id}"
    wts_type = onnx.TensorProto.BFLOAT16
    wts = float_numpy_to_bfloat_tensor(np.zeros(64), wts_name, True)
    wts_shape = [64]
    wts_tvi = onnx.helper.make_tensor_value_info(wts_name, wts_type, wts_shape)
    tvis.append(wts_tvi)
    gelu_output = gelu.output[0] + f".out{pass_id}"
    op_type = "SDGelu"
    gelu_node = onnx.helper.make_node(
        op_type,
        inputs=[pre_cast_output, wts_name],
        outputs=[gelu_output],
        domain=domain,
        name=gelu.name,
    )
    add_attribute(gelu_node, "input_shape", [B, M, N])
    add_attribute(gelu_node, "in_dtypes", ["bfloat16", "bfloat16"])
    add_attribute(gelu_node, "out_dtypes", ["bfloat16"])
    add_attribute(gelu_node, "output_shape", [B, M, N])

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        gelu.output[0] + f".out{pass_id}",
        gelu.output[0],
        gelu_output_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(gelu.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast, gelu_node, *post_cast], [wts], tvis


PATTERN = ["Gelu([?], ?)"]
REPLACEMENT = replacement
