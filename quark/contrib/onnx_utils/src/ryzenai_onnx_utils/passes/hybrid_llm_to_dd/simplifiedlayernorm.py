# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast as cast
import ryzenai_onnx_utils.utils
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # assuming all in same domain for now
    domain = params.get_domain("MLADFRMSNORM")
    first_in = subgraph[0]
    for node in subgraph:
        if node.op_type == "SimplifiedLayerNormalization":
            sln_0 = node
    last_node = subgraph[-1]
    op_version = params.attributes["mladf_version"]

    new_nodes = []
    new_tvis = []
    eps = onnx.helper.get_node_attr_value(sln_0, "epsilon")
    eps_tensor = ryzenai_onnx_utils.utils.float_numpy_to_bfloat_tensor(np.array([eps]), f"eps_{pass_id}")
    input_cast_0, input_cast_0_tvis = cast.add_cast_dtype_to_bfloat16_auto(
        first_in.input[0], pass_id, domain, extractor
    )
    new_nodes.extend(input_cast_0)
    new_tvis.extend(input_cast_0_tvis)

    output_cast_0, output_cast_0_tvis = cast.add_cast_bfloat16_to_dtype_auto(
        last_node.output[0], pass_id, domain, extractor
    )

    new_tvis.extend(output_cast_0_tvis)

    gamma = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(sln_0.input[1], extractor)
    gamma_bf = ryzenai_onnx_utils.utils.float_numpy_to_bfloat_tensor(gamma, sln_0.input[1])

    inputs_list = [input_cast_0[0].output[0], sln_0.input[1], eps_tensor.name]
    rms_norm = onnx.helper.make_node(
        "MLADFRMSNORM",
        inputs=inputs_list,
        outputs=[output_cast_0[0].input[0]],
        name=f"rms_norm_{pass_id}",
        domain=domain,
    )
    pdi_id = int(params.attributes.get("pdi_id", 0))
    if pdi_id != 0:
        ryzenai_onnx_utils.matcher.add_attribute(rms_norm, "pdi_id", int(pdi_id))
    enable_ctrl_pkt = params.get_bool_attr("enable_ctrl_pkt", False)
    if enable_ctrl_pkt:
        ryzenai_onnx_utils.matcher.add_attribute(rms_norm, "enable_ctrl_pkt", True)
    ryzenai_onnx_utils.matcher.add_attribute(rms_norm, "op_version", op_version)
    new_nodes.append(rms_norm)
    new_nodes.extend(output_cast_0)

    return new_nodes, [gamma_bf, eps_tensor], new_tvis


PATTERN = [
    ["Cast(?, a1)", "SimplifiedLayerNormalization([a1, ?], [a2])", "Cast([a2],?)"],
    # Gemma3-4B has casts around SLN in the MLP block
    [
        "SimplifiedLayerNormalization([?, ?], [a1])",
        "Cast([a1], ?)",
    ],
    ["Cast(?, a1)", "SimplifiedLayerNormalization([a1, ?], [?])"],
    ["SimplifiedLayerNormalization([?, ?], [?])"],
]
REPLACEMENT = [replacement] * 4
