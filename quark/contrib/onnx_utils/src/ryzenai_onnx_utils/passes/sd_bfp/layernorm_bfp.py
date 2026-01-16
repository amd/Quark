# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    copy_attributes,
    get_shape,
    is_initializer,
    set_attribute,
)
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    register_whitebox_pass,
)
from ryzenai_onnx_utils.passes.sd15.layernorm_to_sd_layernorm import SDLayerNormPass
from ryzenai_onnx_utils.passes.sd_bfp.bfp_cast import add_sd_cast_bf16_to_bfp, add_sd_cast_bfp_to_bf16
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDLayerNorm_bfp")
class SDSLayerNormBfpPass(SDLayerNormPass):
    whitebox_flow_op_type: str = SDLayerNormPass.whitebox_flow_op_type

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                ((2, 64, 1280), (1280,), (1280,)),
                ((2, 256, 1280), (1280,), (1280,)),
                ((2, 1024, 640), (640,), (640,)),
                ((2, 4096, 320), (320,), (320,)),
            },
            "sd3": {},
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        gamma_shape = tuple(check_shapes["input_shape"][1])
        beta_shape = tuple(check_shapes["input_shape"][2])
        return (input_shape, gamma_shape, beta_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDLayerNormPass.get_input_output_shapes(node, extractor)


@register_whitebox_pass("SDLayerNorm_bfbfp")
class SDLayerNormBfBfpPass(SDLayerNormPass):
    whitebox_flow_op_type: str = SDLayerNormPass.whitebox_flow_op_type

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {},
            "sd3": {
                ((2, 1024, 1536), (1536,), (1536,)),
                ((2, 160, 1536), (1536,), (1536,)),
                ((2, 4096, 1536), (1536,), (1536,)),
                ((1, 1024, 1536), (1536,), (1536,)),
                ((1, 160, 1536), (1536,), (1536,)),
                ((1, 4096, 1536), (1536,), (1536,)),
            },
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        gamma_shape = tuple(check_shapes["input_shape"][1])
        beta_shape = tuple(check_shapes["input_shape"][2])
        return (input_shape, gamma_shape, beta_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDLayerNormPass.get_input_output_shapes(node, extractor)


def is_broadcast_shape(input_shape: tuple[int | str, ...], output_shape: tuple[int | str, ...]) -> bool:
    return input_shape != output_shape


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # Disable for dynamic shape models handled elsewhere
    layernorm_node = subgraph[0]

    domain = params.get_domain("SDLayerNorm_bfp")
    op_namespace = params.get_subgraph_op_namespace(subgraph)

    tvis: list[onnx.ValueInfoProto] = []
    initializers: list[onnx.TensorProto] = []
    new_nodes: list[onnx.NodeProto] = []

    bfp_inputs: list[str] = []

    for input_name in layernorm_node.input:
        if op_namespace == "sd3" or is_initializer(input_name, extractor):
            bfp_inputs.append(input_name)
        else:
            input_shape = get_shape(layernorm_node.input[0], extractor)
            bfp_output_name = input_name + f"_bfp.out{pass_id}"
            pre_cast_bf2bfp_nodes, pre_cast_bf2bfp_inits, pre_cast_bf2bfp_tvis = add_sd_cast_bf16_to_bfp(
                input_name, bfp_output_name, input_shape, domain
            )
            new_nodes.extend(pre_cast_bf2bfp_nodes)
            initializers.extend(pre_cast_bf2bfp_inits)
            tvis.extend(pre_cast_bf2bfp_tvis)
            bfp_inputs.append(bfp_output_name)

    output_shape = get_shape(layernorm_node.output[0], extractor)
    sd_output_name = f"{layernorm_node.output[0]}_bfp.out{pass_id}"
    sd_output_tvi = onnx.helper.make_tensor_value_info(sd_output_name, onnx.TensorProto.UINT8, output_shape)
    tvis.append(sd_output_tvi)

    op_type = "SDLayerNorm_bfp" if op_namespace == "sd15" else "SDLayerNorm_bfbfp"

    bfp_node = onnx.helper.make_node(
        op_type,
        inputs=bfp_inputs,
        outputs=[sd_output_name],
        domain=domain,
        name=layernorm_node.name,
    )
    copy_attributes(layernorm_node, bfp_node)
    if op_namespace == "sd3":
        set_attribute(bfp_node, "in_dtypes", ["bfloat16", "bfloat16"])
    elif op_namespace == "sd15":
        set_attribute(bfp_node, "in_dtypes", ["bfp16ebs8", "bfloat16"])
    set_attribute(bfp_node, "out_dtypes", ["bfp16ebs8"])
    new_nodes.append(bfp_node)

    post_cast_bfp2bf_nodes, post_cast_bfp2bf_inits, post_cast_bfp2bf_tvis = add_sd_cast_bfp_to_bf16(
        sd_output_name, layernorm_node.output[0], output_shape, domain
    )
    new_nodes.extend(post_cast_bfp2bf_nodes)
    initializers.extend(post_cast_bfp2bf_inits)
    tvis.extend(post_cast_bfp2bf_tvis)

    return new_nodes, initializers, tvis


PATTERN = ["SDLayerNorm([?,?,?], ?)"]
REPLACEMENT = replacement
