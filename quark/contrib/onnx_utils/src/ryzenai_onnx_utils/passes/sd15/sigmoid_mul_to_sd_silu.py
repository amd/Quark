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
from ryzenai_onnx_utils.utils import float_numpy_to_bfloat_tensor


@register_whitebox_pass("SDSilu")
class SDSiluPass(WhiteboxBasePass):
    whitebox_flow_op_type = "Silu"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                # unet
                (2, 64, 64, 320),  # count 8
                (2, 1280),  # count 2
                (2, 32, 32, 320),  # count 1
                (2, 32, 32, 640),  # count 6
                (2, 16, 16, 640),  # count 1
                (2, 16, 16, 1280),  # count 6
                (2, 8, 8, 1280),  # count 11
                (2, 8, 8, 2560),  # count 3
                (2, 16, 16, 2560),  # count 2
                (2, 16, 16, 1920),  # count 1
                (2, 32, 32, 1920),  # count 1
                (2, 32, 32, 1280),  # count 1
                (2, 32, 32, 960),  # count 1
                (2, 64, 64, 960),  # count 1
                (2, 64, 64, 640),  # count 2
                # sd2.1-v unet
                (2, 48, 48, 320),
                (2, 96, 96, 320),
                (2, 48, 48, 640),
                (2, 96, 96, 640),
                (2, 24, 24, 640),
                (2, 48, 48, 960),
                (2, 96, 96, 960),
                (2, 12, 12, 1280),
                (2, 1, 1, 1280),
                (2, 24, 24, 1280),
                (2, 48, 48, 1280),
                (2, 24, 24, 1920),
                (2, 48, 48, 1920),
                (2, 12, 12, 2560),
                (2, 24, 24, 2560),
                # vae_decoder
                (1, 64, 64, 512),  # count 10
                (1, 128, 128, 512),  # count 6
                (1, 256, 256, 512),  # count 1
                (1, 256, 256, 256),  # count 5
                (1, 512, 512, 256),  # count 1
                (1, 512, 512, 128),  # count 6
                # sd2.1 vae_decoder
                (1, 768, 768, 128),
                (1, 768, 768, 256),
                (1, 384, 384, 256),
                (1, 192, 192, 512),
                (1, 384, 384, 512),
                (1, 96, 96, 512),
                # controlnet
                # (2, 512, 512, 16),  # count 2
                # (2, 256, 256, 32),  # count 2 #4
                (2, 128, 128, 96),  # count 2 #2
                (2, 64, 64, 256),  # count 1 #1
                # sd(xl)-turbo bs1
                (1, 64, 64, 320),
                (1, 1280),
                (1, 32, 32, 320),
                (1, 32, 32, 640),
                (1, 16, 16, 640),
                (1, 16, 16, 1280),
                (1, 8, 8, 1280),
                (1, 8, 8, 2560),
                (1, 16, 16, 2560),
                (1, 16, 16, 1920),
                (1, 32, 32, 1920),
                (1, 32, 32, 1280),
                (1, 32, 32, 960),
                (1, 64, 64, 960),
                (1, 64, 64, 640),
                # sdxl-base vae_decoder
                (1, 512, 512, 512),
                (1, 1024, 1024, 256),
                (1, 1024, 1024, 128),
                # sdxl-base unet
                (2, 32, 32, 2560),
                (2, 64, 64, 1280),
                (2, 64, 64, 1920),
                (2, 128, 128, 320),
                (2, 128, 128, 640),
                (2, 128, 128, 960),
            },
            "sd3": {
                # mmdit
                (1, 1536),  # count 3
                (2, 1536),  # count 3
                # vae decoder 512
                (1, 64, 64, 512),  # count 10
                (1, 128, 128, 512),  # count 6
                (1, 256, 256, 512),  # count 1
                (1, 256, 256, 256),  # count 5
                (1, 512, 512, 256),  # count 1
                (1, 512, 512, 128),  # count 6
                # new in vae decoder 1024
                (1, 512, 512, 512),  # count 1
                (1, 1024, 1024, 256),  # count 1
                (1, 1024, 1024, 128),  # count 6
                # vae encoder 512
                (1, 256, 256, 128),  # count 1
                (1, 128, 128, 256),  # count 1
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


def is_silu_supported_pattern(extractor, sigmoid, mul):
    return len(sigmoid.input) == 1 and len(sigmoid.output) == 1


# This `blacklist` filters out operators with specific shapes
# that negatively impact the accuracy of SD15, and offloads them to the CPU.
def in_silu_black_list(op_namespace, input_shape) -> bool:
    if op_namespace != "sd15":
        return False
    black_list = {
        "sd15": {
            # controlnet
            (2, 512, 512, 16),  # count 2
            (2, 256, 256, 32),  # count 2 #4
        }
    }
    return input_shape in black_list[op_namespace]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDSilu")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    (sigmoid, mul) = subgraph

    if not is_silu_supported_pattern(extractor, sigmoid, mul):
        return subgraph, [], None

    input_shape = ryzenai_onnx_utils.matcher.get_shape(sigmoid.input[0], extractor)

    if in_silu_black_list(op_namespace, input_shape):
        return subgraph, [], None

    tvis = []

    pre_cast_output = sigmoid.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
        sigmoid.input[0],
        pre_cast_output,
        input_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(sigmoid.input[0], extractor),
    )
    tvis.extend(pre_cast_tvi)
    # add addition wts tensor for xrt
    wts_name = sigmoid.name + f".weights{pass_id}"
    wts_type = onnx.TensorProto.BFLOAT16
    # runtime(xrt) run kernel needs extra weights buffer for Silu, the size
    # of weights buffer at least 128, which is all 0, If there is no weights
    # buffer, resulting in may be precision errors
    wts = float_numpy_to_bfloat_tensor(np.zeros(128), wts_name, True)
    wts_shape = [128]
    wts_tvi = onnx.helper.make_tensor_value_info(wts_name, wts_type, wts_shape)
    tvis.append(wts_tvi)

    new_inputs = [pre_cast_output, wts_name]
    add_node_output = sigmoid.output[0] + f".out{pass_id}"
    name = sigmoid.name if sigmoid.name else f"silu_{pass_id}"
    dd_silu_node = onnx.helper.make_node(
        "SDSilu",
        inputs=new_inputs,
        outputs=[add_node_output],
        domain=domain,
        name=name,
    )
    add_attribute(dd_silu_node, "input_shape", input_shape)
    add_attribute(dd_silu_node, "output_shape", input_shape)
    add_attribute(dd_silu_node, "in_dtypes", ["bfloat16"])
    add_attribute(dd_silu_node, "out_dtypes", ["bfloat16"])
    add_attribute(dd_silu_node, "weight_shape", [128])

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        add_node_output,
        mul.output[0],
        ryzenai_onnx_utils.matcher.get_shape(subgraph[-1].output[0], extractor),
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(mul.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast, dd_silu_node, *post_cast], [wts], tvis


PATTERN = ["Sigmoid([?],a)", "Mul([?,a], ?)"]
REPLACEMENT = replacement
