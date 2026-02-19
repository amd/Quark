# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces an SkipSimplifiedLayerNorm with a DD FlatRMSAdd operator for
DD fusion.
"""

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.utils
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype_auto,
    add_cast_dtype_to_bfloat16_auto,
)
from ryzenai_onnx_utils.typing import PassOutputArgs


def get_gate_up_attributes(
    gate_proj: onnx.NodeProto, up_proj: onnx.NodeProto, extractor: onnx.utils.Extractor
) -> tuple[int, int, int]:
    m_0 = ryzenai_onnx_utils.matcher.get_shape(gate_proj.input[0], extractor)[1]
    m_1 = ryzenai_onnx_utils.matcher.get_shape(up_proj.input[0], extractor)[1]
    assert m_0 == m_1 and isinstance(m_0, int)
    k_0 = onnx.helper.get_node_attr_value(gate_proj, "K")
    k_1 = onnx.helper.get_node_attr_value(up_proj, "K")
    assert k_0 == k_1
    n_0 = onnx.helper.get_node_attr_value(gate_proj, "N")
    n_1 = onnx.helper.get_node_attr_value(up_proj, "N")
    assert n_0 == n_1
    return (m_0, k_0, n_0)


def convert_matmul_tensors(
    matmul: onnx.NodeProto, extractor: onnx.utils.Extractor
) -> tuple[onnx.TensorProto, onnx.TensorProto, onnx.TensorProto, onnx.TensorProto]:
    weights = ryzenai_onnx_utils.matcher.get_initializer(matmul.input[1], extractor, False)

    scales_np = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul.input[2], extractor)
    scales_np = scales_np.astype(np.float32)
    scales = onnx.numpy_helper.from_array(scales_np, matmul.input[2] + ".f")

    zp = ryzenai_onnx_utils.matcher.get_initializer(matmul.input[3], extractor, False)

    bias_np = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul.input[5], extractor)
    bias_np = bias_np.astype(np.float32)
    bias = onnx.numpy_helper.from_array(bias_np, matmul.input[5] + ".f")

    return (weights, scales, zp, bias)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("FlatRMSAdd")
    ssln = subgraph[0]
    output = subgraph[-1]
    new_nodes = []
    new_initializers = []
    new_tvis = []

    lora = params.get_bool_attr("lora", False)

    input_cast_0, input_tvi_0 = add_cast_dtype_to_bfloat16_auto(ssln.input[0], pass_id, domain, extractor)
    new_nodes.extend(input_cast_0)
    new_tvis.extend(input_tvi_0)

    # TODO(varunsh): this needs to be handled generically
    if ssln.input[1] != ssln.input[0]:
        input_cast_1, input_tvi_1 = add_cast_dtype_to_bfloat16_auto(ssln.input[1], pass_id, domain, extractor)
        new_nodes.extend(input_cast_1)
        new_tvis.extend(input_tvi_1)
    else:
        input_cast_1 = input_cast_0
        # return subgraph, [], None
    eps = onnx.helper.get_node_attr_value(ssln, "epsilon")
    eps_tensor = ryzenai_onnx_utils.utils.float_numpy_to_bfloat_tensor(np.array([eps]), f"eps_{pass_id}")
    # input_cast_1, input_tvi_1 = add_cast_dtype_to_bfloat16_auto(ssln.input[1], pass_id, domain, extractor)
    # new_nodes.extend(input_cast_1)
    # new_tvis.extend(input_tvi_1)

    input_shape_0 = ryzenai_onnx_utils.matcher.get_shape(ssln.input[0], extractor)
    input_shape_1 = ryzenai_onnx_utils.matcher.get_shape(ssln.input[1], extractor)
    assert input_shape_0 == input_shape_1
    assert all(isinstance(x, int) for x in input_shape_0)

    gamma = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(ssln.input[2], extractor)
    gamma_bf = ryzenai_onnx_utils.utils.float_numpy_to_bfloat_tensor(gamma, ssln.input[2] + ".bf")
    new_initializers.append(gamma_bf)
    new_initializers.append(eps_tensor)
    output_cast_0, output_tvi_0 = add_cast_bfloat16_to_dtype_auto(output.output[0], pass_id, domain, extractor)
    new_nodes.extend(output_cast_0)
    new_tvis.extend(output_tvi_0)

    new_outputs = []
    if len(ssln.output) > 1:
        output_cast_1, output_tvi_1 = add_cast_bfloat16_to_dtype_auto(ssln.output[3], pass_id, domain, extractor)
        new_nodes.extend(output_cast_1)
        new_tvis.extend(output_tvi_1)
        new_outputs.append(output_cast_1[0].input[0])
    else:
        # in this case, add a dummy output for fusion compatibility
        new_outputs.append(ssln.output[0] + ".dummy")
        output_0_shape = ryzenai_onnx_utils.matcher.get_shape(output_cast_0[0].output[0], extractor)
        new_tvis.append(
            onnx.helper.make_tensor_value_info(ssln.output[0] + ".dummy", onnx.TensorProto.BFLOAT16, output_0_shape)
        )
    new_outputs.append(output_cast_0[0].input[0])

    rmsadd = onnx.helper.make_node(
        "FlatRMSAdd",
        inputs=[input_cast_0[0].output[0], input_cast_1[0].output[0], gamma_bf.name, eps_tensor.name],
        outputs=new_outputs,
        name=f"FlatRMSAdd_{pass_id}",
        domain=domain,
    )
    new_nodes.append(rmsadd)

    ryzenai_onnx_utils.matcher.add_attribute(rmsadd, "a_shape", input_shape_0)
    ryzenai_onnx_utils.matcher.add_attribute(rmsadd, "in_dtypes", ["bfloat16", "bfloat16"])
    ryzenai_onnx_utils.matcher.add_attribute(rmsadd, "out_dtypes", ["bfloat16", "bfloat16"])
    ryzenai_onnx_utils.matcher.add_attribute(rmsadd, "c_shape", input_shape_0)
    ryzenai_onnx_utils.matcher.add_attribute(rmsadd, "b_shape", input_shape_0)
    ryzenai_onnx_utils.matcher.add_attribute(rmsadd, "is_gamma_ifm", [1])
    if lora:
        ryzenai_onnx_utils.matcher.add_attribute(rmsadd, "lora", lora)
    enable_ctrl_pkt = params.get_bool_attr("enable_ctrl_pkt", False)
    if enable_ctrl_pkt:
        ryzenai_onnx_utils.matcher.add_attribute(rmsadd, "enable_ctrl_pkt", True)
    pdi_id = int(params.attributes.get("pdi_id", 0))
    if pdi_id != 0:
        ryzenai_onnx_utils.matcher.add_attribute(rmsadd, "pdi_id", int(pdi_id))
    return new_nodes, new_initializers, new_tvis


PATTERN = [
    ["SkipSimplifiedLayerNormalization([?,?,?], [a1,?,?,?])", "Cast([a1], ?)"],
    ["SkipSimplifiedLayerNormalization([?,?,?], [?,?,?,?])"],
]
REPLACEMENT = [replacement] * 2
