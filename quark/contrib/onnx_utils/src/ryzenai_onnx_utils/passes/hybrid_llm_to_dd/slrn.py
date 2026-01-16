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
    domain = params.get_domain("MLADFRMSNORM")

    reshape_0 = subgraph[0]
    ssln = subgraph[1]
    reshape_1 = subgraph[2]

    new_nodes = []
    new_tvis = []
    eps = onnx.helper.get_node_attr_value(ssln, "epsilon")
    eps_tensor = ryzenai_onnx_utils.utils.float_numpy_to_bfloat_tensor(np.array([eps]), f"eps_{pass_id}")
    reshaped_shape = ryzenai_onnx_utils.matcher.get_shape(reshape_0.output[0], extractor)
    input_cast_0, input_cast_0_tvis = cast.add_cast_dtype_to_bfloat16_auto(
        reshape_0.input[0], pass_id, domain, extractor, shape=reshaped_shape
    )
    new_nodes.extend(input_cast_0)
    new_tvis.extend(input_cast_0_tvis)

    output_cast_1, output_cast_1_tvis = cast.add_cast_bfloat16_to_dtype_auto(
        reshape_1.output[0], pass_id, domain, extractor
    )
    new_nodes.extend(output_cast_1)
    new_tvis.extend(output_cast_1_tvis)

    gamma = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(ssln.input[1], extractor)
    gamma_bf = ryzenai_onnx_utils.utils.float_numpy_to_bfloat_tensor(gamma, ssln.input[1])

    rms_norm = onnx.helper.make_node(
        "MLADFRMSNORM",
        inputs=[input_cast_0[0].output[0], gamma_bf.name, eps_tensor.name],
        outputs=[output_cast_1[0].input[0]],
        name=f"rms_norm_{pass_id}",
        domain=domain,
    )

    bfp16 = (
        params.attributes["is_bfp16"]
        if "is_bfp16" in params.attributes and params.attributes["is_bfp16"]
        else "weights"
    )
    op_version = "v2" if bfp16 == "weights" else "flat"
    ryzenai_onnx_utils.matcher.add_attribute(rms_norm, "op_version", op_version)
    enable_ctrl_pkt = params.get_bool_attr("enable_ctrl_pkt", False)
    if enable_ctrl_pkt:
        ryzenai_onnx_utils.matcher.add_attribute(rms_norm, "enable_ctrl_pkt", True)
    new_nodes.append(rms_norm)

    return new_nodes, [eps_tensor, gamma_bf], new_tvis


PATTERN = [
    "Reshape([?, ?], [a1])",
    "SimplifiedLayerNormalization([a1,?], a2)",
    "Reshape([a2, ?], [?])",
]
REPLACEMENT = replacement
