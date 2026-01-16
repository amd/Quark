# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    copy_attributes,
    get_shape,
    get_shapes,
    set_attribute,
)
from ryzenai_onnx_utils.passes.sd3.mul_add_to_sd_mul_add import SDMulAddPass
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    register_whitebox_pass,
)
from ryzenai_onnx_utils.passes.sd_bfp.bfp_cast import add_sd_cast_bf16_to_bfp, add_sd_cast_bfp_to_bf16
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDMulAdd_bfp")
class SDBfpMulAddPass(SDMulAddPass):
    whitebox_flow_op_type: str = "MulAdd"
    force_whitelist: bool = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd3": {
                # mmdit 512
                ((2, 1024, 1536), (2, 1, 1536), (2, 1, 1536)),
                # mmdit 1024
                ((2, 4096, 1536), (2, 1, 1536), (2, 1, 1536)),
            },
        }
        a_shape = tuple(check_shapes["input_shape"][0])
        b_shape = tuple(check_shapes["input_shape"][1])
        c_shape = tuple(check_shapes["input_shape"][2])
        return (a_shape, b_shape, c_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDMulAddPass.get_input_output_shapes(node, extractor)


@register_whitebox_pass("SDMulAdd_bfpbfbf")
class SDBfpMulAddBfpBfpPass(SDMulAddPass):
    whitebox_flow_op_type: str = "MulAdd"
    force_whitelist: bool = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd3": {
                # mmdit 512
                ((2, 1024, 1536), (2, 1, 1536), (2, 1024, 1536)),
                # mmdit 1024
                ((2, 4096, 1536), (2, 1, 1536), (2, 4096, 1536)),
            },
        }
        a_shape = tuple(check_shapes["input_shape"][0])
        b_shape = tuple(check_shapes["input_shape"][1])
        c_shape = tuple(check_shapes["input_shape"][2])
        return (a_shape, b_shape, c_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        return SDMulAddPass.get_input_output_shapes(node, extractor)


def get_broadcast_type(
    extractor: onnx.utils.Extractor, node: onnx.NodeProto, output_shape: tuple[int | str, ...]
) -> bool:
    input_shapes = get_shapes(node.input, extractor)
    output_shape = get_shape(node.output[0], extractor)
    broadcast_num = 0
    for input_shape in input_shapes:
        if input_shape != output_shape:
            broadcast_num += 1
    return broadcast_num


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # Disable for dynamic shape models handled elsewhere
    mul_add_node = subgraph[0]

    domain = params.get_domain("SDMulAdd")
    output_shape = get_shape(mul_add_node.output[0], extractor)

    tvis: list[onnx.ValueInfoProto] = []
    initializers: list[onnx.TensorProto] = []
    new_nodes: list[onnx.NodeProto] = []

    # 1) Cast BF16 -> BFP16 for each input using SDCastBf2Bfp (with per-input weights)
    bfp_inputs: list[str] = []
    bfp_input_dtypes: list[str] = []
    for index, input_name in enumerate(mul_add_node.input):
        if index != 0:
            bfp_inputs.append(input_name)
            bfp_input_dtypes.append("bfloat16")
        else:
            input_shape = get_shape(mul_add_node.input[0], extractor)
            bfp_output_name = input_name + f"_bfp.out{pass_id}"
            pre_cast_bf2bfp_nodes, pre_cast_bf2bfp_inits, pre_cast_bf2bfp_tvis = add_sd_cast_bf16_to_bfp(
                input_name, bfp_output_name, input_shape, domain
            )
            new_nodes.extend(pre_cast_bf2bfp_nodes)
            initializers.extend(pre_cast_bf2bfp_inits)
            tvis.extend(pre_cast_bf2bfp_tvis)

            bfp_inputs.append(bfp_output_name)
            bfp_input_dtypes.append("bfp16ebs8")

    # 2) BFP16 SDMulAdd
    broadcast_type = get_broadcast_type(extractor, mul_add_node, output_shape)
    sd_output_name = mul_add_node.output[0] + f"_bfp.out{pass_id}" if broadcast_type == 2 else mul_add_node.output[0]
    sd_output_dtype = onnx.TensorProto.UINT8 if broadcast_type == 2 else onnx.TensorProto.BFLOAT16
    sd_output_tvi = onnx.helper.make_tensor_value_info(sd_output_name, sd_output_dtype, output_shape)
    tvis.append(sd_output_tvi)
    if broadcast_type == 1:
        sd_mul_add_node = onnx.helper.make_node(
            "SDMulAdd_bfpbfbf",
            inputs=bfp_inputs,
            outputs=[sd_output_name],
            domain=domain,
            name=mul_add_node.name,
        )
    else:
        sd_mul_add_node = onnx.helper.make_node(
            "SDMulAdd_bfp",
            inputs=bfp_inputs,
            outputs=[sd_output_name],
            domain=domain,
            name=mul_add_node.name,
        )
    # Shape attributes (consistent with SDMulAdd)
    # DType attributes for BFP path
    copy_attributes(mul_add_node, sd_mul_add_node)
    set_attribute(sd_mul_add_node, "in_dtypes", bfp_input_dtypes)
    if broadcast_type == 2:
        set_attribute(sd_mul_add_node, "out_dtypes", ["bfp16ebs8"])
    else:
        set_attribute(sd_mul_add_node, "out_dtypes", ["bfloat16"])
    new_nodes.append(sd_mul_add_node)

    # 3) Cast BFP16 -> BF16 using SDCastBfp2Bf, reuse first cast weight tensor
    if broadcast_type == 2:
        post_cast_bfp2bf_nodes, post_cast_bfp2bf_inits, post_cast_bfp2bf_tvis = add_sd_cast_bfp_to_bf16(
            sd_output_name, mul_add_node.output[0], output_shape, domain
        )
        new_nodes.extend(post_cast_bfp2bf_nodes)
        initializers.extend(post_cast_bfp2bf_inits)
        tvis.extend(post_cast_bfp2bf_tvis)

    return new_nodes, initializers, tvis


PATTERN = ["SDMulAdd([?,?,?], ?)"]
REPLACEMENT = replacement
