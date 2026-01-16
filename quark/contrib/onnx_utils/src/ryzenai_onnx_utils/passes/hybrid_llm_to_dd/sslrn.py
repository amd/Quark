# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast as cast
import ryzenai_onnx_utils.utils
from ryzenai_onnx_utils.typing import PassOutputArgs

from ..hybrid_llm_prune_logits import prune_config


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # assuming all in same domain for now
    domain = params.get_domain("MLADFADD")

    ssln_0 = subgraph[0]
    if ssln_0.input[0] == ssln_0.input[1]:
        return subgraph, [], None

    new_nodes = []
    new_tvis = []
    eps = onnx.helper.get_node_attr_value(ssln_0, "epsilon")
    eps_tensor = ryzenai_onnx_utils.utils.float_numpy_to_bfloat_tensor(np.array([eps]), f"eps_{pass_id}")
    input_cast_0, input_cast_0_tvis = cast.add_cast_dtype_to_bfloat16_auto(ssln_0.input[0], pass_id, domain, extractor)
    new_nodes.extend(input_cast_0)
    new_tvis.extend(input_cast_0_tvis)

    input_cast_1, input_cast_1_tvis = cast.add_cast_dtype_to_bfloat16_auto(ssln_0.input[1], pass_id, domain, extractor)
    new_nodes.extend(input_cast_1)
    new_tvis.extend(input_cast_1_tvis)

    if len(ssln_0.output) == 4:
        output_cast_0, output_cast_0_tvis = cast.add_cast_bfloat16_to_dtype_auto(
            ssln_0.output[3], pass_id, domain, extractor
        )
        new_nodes.extend(output_cast_0)
        new_tvis.extend(output_cast_0_tvis)
        add_output = output_cast_0[0].input[0]
    else:
        add_output = f"dummy_{pass_id}"
        output_shape = ryzenai_onnx_utils.matcher.get_shape(ssln_0.output[0], extractor)
        new_tvis.append(onnx.helper.make_tensor_value_info(add_output, onnx.TensorProto.BFLOAT16, output_shape))

    add = onnx.helper.make_node(
        "MLADFADD",
        inputs=[input_cast_0[0].output[0], input_cast_1[0].output[0]],
        outputs=[add_output],
        name=f"add_{pass_id}",
        domain=domain,
    )
    prune_config(ssln_0, add, params)

    ryzenai_onnx_utils.matcher.add_attribute(add, "op_version", "v2")
    new_nodes.append(add)

    output_cast_1, output_cast_1_tvis = cast.add_cast_bfloat16_to_dtype_auto(
        ssln_0.output[0], pass_id, domain, extractor
    )
    new_nodes.extend(output_cast_1)
    new_tvis.extend(output_cast_1_tvis)

    gamma = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(ssln_0.input[2], extractor)
    gamma_bf = ryzenai_onnx_utils.utils.float_numpy_to_bfloat_tensor(gamma, ssln_0.input[2])

    rms_norm = onnx.helper.make_node(
        "MLADFRMSNORM",
        inputs=[add_output, ssln_0.input[2], eps_tensor.name],
        outputs=[output_cast_1[0].input[0]],
        name=f"rms_norm_{pass_id}",
        domain=domain,
    )
    prune_config(ssln_0, rms_norm, params)
    enable_ctrl_pkt = params.get_bool_attr("enable_ctrl_pkt", False)
    if enable_ctrl_pkt:
        ryzenai_onnx_utils.matcher.add_attribute(rms_norm, "enable_ctrl_pkt", True)

    ryzenai_onnx_utils.matcher.add_attribute(rms_norm, "op_version", "v2")
    new_nodes.append(rms_norm)

    return new_nodes, [gamma_bf, eps_tensor], new_tvis


PATTERN = [
    "SkipSimplifiedLayerNormalization([a1,a7,?], [?,?,?,?])",
]
REPLACEMENT = replacement
