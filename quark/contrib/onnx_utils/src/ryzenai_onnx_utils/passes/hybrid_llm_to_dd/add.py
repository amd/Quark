# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast as cast
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("MLADFADD")
    add = subgraph[0]
    input_shape = ryzenai_onnx_utils.matcher.get_shape(add.input[0], extractor)
    input_shape2 = ryzenai_onnx_utils.matcher.get_shape(add.input[1], extractor)

    if len(input_shape) == 1 or len(input_shape2) == 1 or (len(input_shape) != len(input_shape2)):
        return subgraph, [], []

    new_nodes = []
    new_tvis = []
    input_cast, input_cast_tvis = cast.add_cast_dtype_to_bfloat16_auto(add.input[0], pass_id, domain, extractor)
    new_nodes.extend(input_cast)
    new_tvis.extend(input_cast_tvis)

    input_cast_1, input_cast_1_tvis = cast.add_cast_dtype_to_bfloat16_auto(add.input[1], pass_id, domain, extractor)
    new_nodes.extend(input_cast_1)
    new_tvis.extend(input_cast_1_tvis)

    output_cast, output_cast_tvis = cast.add_cast_bfloat16_to_dtype_auto(add.output[0], pass_id, domain, extractor)
    new_nodes.extend(output_cast)
    new_tvis.extend(output_cast_tvis)

    elw_add = onnx.helper.make_node(
        "MLADFADD",
        inputs=[input_cast[0].output[0], input_cast_1[0].output[0]],
        outputs=[output_cast[0].input[0]],
        name=f"add_{pass_id}",
        domain=domain,
    )
    lora = params.get_bool_attr("lora", False)
    if lora:
        ryzenai_onnx_utils.matcher.add_attribute(elw_add, "lora", lora)
    pdi_id = int(params.attributes.get("pdi_id", 0))
    if pdi_id != 0:
        ryzenai_onnx_utils.matcher.add_attribute(elw_add, "pdi_id", int(pdi_id))
    ryzenai_onnx_utils.matcher.add_attribute(elw_add, "op_version", params.attributes["mladf_version"])
    enable_ctrl_pkt = params.get_bool_attr("enable_ctrl_pkt", False)
    if enable_ctrl_pkt:
        ryzenai_onnx_utils.matcher.add_attribute(elw_add, "enable_ctrl_pkt", enable_ctrl_pkt)
    new_nodes.append(elw_add)

    return new_nodes, [], new_tvis


PATTERN = ["Add([?,?], ?)"]
REPLACEMENT = replacement
