# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast as cast
from ryzenai_onnx_utils.strategy_builder import MladfVersion
from ryzenai_onnx_utils.typing import PassOutputArgs

from ..hybrid_llm_prune_logits import prune_config


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SILU")
    sigmoid = subgraph[0]

    new_nodes = []
    new_tvis = []

    input_cast, input_cast_tvis = cast.add_cast_dtype_to_bfloat16_auto(sigmoid.input[0], pass_id, domain, extractor)
    new_nodes.extend(input_cast)
    new_tvis.extend(input_cast_tvis)

    output_cast, output_cast_tvis = cast.add_cast_bfloat16_to_dtype_auto(sigmoid.output[0], pass_id, domain, extractor)
    new_nodes.extend(output_cast)
    new_tvis.extend(output_cast_tvis)
    silu = onnx.helper.make_node(
        "SILU",
        inputs=[input_cast[0].output[0]],
        outputs=[output_cast[0].input[0]],
        name=f"silu_{pass_id}",
        domain=domain,
    )
    prune_config(sigmoid, silu, params)
    op_version = MladfVersion.AIE2_V2
    lora = params.get_bool_attr("lora", False)
    ryzenai_onnx_utils.matcher.add_attribute(silu, "op_name", "gelu")

    if lora:
        op_version = MladfVersion.FLAT
        ryzenai_onnx_utils.matcher.add_attribute(silu, "lora", lora)
    ryzenai_onnx_utils.matcher.add_attribute(silu, "op_version", str(op_version))
    enable_ctrl_pkt = params.get_bool_attr("enable_ctrl_pkt", False)
    if enable_ctrl_pkt:
        ryzenai_onnx_utils.matcher.add_attribute(silu, "enable_ctrl_pkt", enable_ctrl_pkt)
    new_nodes.append(silu)

    return new_nodes, [], new_tvis


PATTERN = ["Gelu([?],a)"]
REPLACEMENT = replacement
