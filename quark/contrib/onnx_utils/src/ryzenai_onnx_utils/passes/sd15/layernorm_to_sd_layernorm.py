# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    copy_attributes,
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


@register_whitebox_pass("SDLayerNorm")
class SDLayerNormPass(WhiteboxBasePass):
    whitebox_flow_op_type = "LayerNormalization"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                # unet
                ((2, 4096, 320), (320,), (320,)),  # count 15
                ((2, 1024, 640), (640,), (640,)),  # count 15
                ((2, 256, 1280), (1280,), (1280,)),  # count 15
                ((2, 64, 1280), (1280,), (1280,)),  # count 3
                # sd2.1-v unet
                ((2, 144, 1280), (1280,), (1280,)),
                ((2, 2304, 640), (640,), (640,)),
                ((2, 576, 1280), (1280,), (1280,)),
                ((2, 9216, 320), (320,), (320,)),
                # sd(xl)-turbo bs1
                ((1, 4096, 320), (320,), (320,)),  # count 15
                ((1, 1024, 640), (640,), (640,)),  # count 15
                ((1, 256, 1280), (1280,), (1280,)),  # count 15
                ((1, 64, 1280), (1280,), (1280,)),  # count 3
                # sdxl-base unet
                ((2, 1024, 1280), (1280,), (1280,)),
                ((2, 4096, 640), (640,), (640,)),
            },
            "sd3": {
                # mmdit
                ((2, 1024, 1536), (1536,), (1536,)),
                ((2, 154, 1536), (1536,), (1536,)),
                # mmdit 1024
                ((2, 4096, 1536), (1536,), (1536,)),
                # mmdit seq-160
                ((2, 160, 1536), (1536,), (1536,)),
            },
            "phi3.5": {
                ((1, 577, 1024), (1024,), (1024,)),  # count 47
            },
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        gamma_shape = tuple(check_shapes["input_shape"][1])
        beta_shape = tuple(check_shapes["input_shape"][2])
        return (input_shape, gamma_shape, beta_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                get_attribute(node, "input_shape"),
                get_attribute(node, "gamma_shape"),
                get_attribute(node, "beta_shape"),
            ],
            "output_shape": [
                get_attribute(node, "output_shape"),
            ],
        }
        return shape_lists


def is_layernorm_supported_pattern(extractor: onnx.utils.Extractor, layernorm_node: onnx.NodeProto) -> bool:
    return len(layernorm_node.input) == 3 and len(layernorm_node.output) == 1


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDLayerNorm")
    layernorm = subgraph[0]

    if not is_layernorm_supported_pattern(extractor, layernorm):
        return subgraph, [], None

    # convert gamma and beta float to bfloat16
    input_shape = ryzenai_onnx_utils.matcher.get_shape(layernorm.input[0], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(layernorm.output[0], extractor)
    gamma_shape = ryzenai_onnx_utils.matcher.get_shape(layernorm.input[1], extractor)
    beta_shape = ryzenai_onnx_utils.matcher.get_shape(layernorm.input[2], extractor)
    flatten = True
    gamma_name = layernorm.input[1]
    gamma_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gamma_name, extractor)
    gamma = float_numpy_to_bfloat_tensor(gamma_f, gamma_name, flatten)
    beta_name = layernorm.input[2]
    beta_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(beta_name, extractor)
    beta = float_numpy_to_bfloat_tensor(beta_f, beta_name, flatten)

    new_nodes = []
    new_inputs = []
    initializers = [gamma, beta]
    tvis = []

    pre_cast_output = layernorm.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
        layernorm.input[0],
        pre_cast_output,
        input_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(layernorm.input[0], extractor),
    )
    new_nodes.extend(pre_cast)
    tvis.extend(pre_cast_tvi)
    new_inputs.extend([pre_cast_output, gamma.name, beta.name])
    layernorm_output = layernorm.output[0] + f".out{pass_id}"
    sd_layernorm_node = onnx.helper.make_node(
        "SDLayerNorm",
        inputs=new_inputs,
        outputs=[layernorm_output],
        domain=domain,
        name=layernorm.name,
    )
    copy_attributes(layernorm, sd_layernorm_node)
    add_attribute(sd_layernorm_node, "input_shape", input_shape)
    add_attribute(sd_layernorm_node, "in_dtypes", ["bfloat16", "bfloat16"])
    add_attribute(sd_layernorm_node, "out_dtypes", ["bfloat16"])
    add_attribute(sd_layernorm_node, "output_shape", output_shape)
    add_attribute(sd_layernorm_node, "gamma_shape", gamma_shape)
    add_attribute(sd_layernorm_node, "beta_shape", beta_shape)

    new_nodes.append(sd_layernorm_node)

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        layernorm_output,
        layernorm.output[0],
        output_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(layernorm.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)
    new_nodes.extend(post_cast)

    return new_nodes, initializers, tvis


PATTERN = ["LayerNormalization([?,?,?],?)"]
REPLACEMENT = replacement
