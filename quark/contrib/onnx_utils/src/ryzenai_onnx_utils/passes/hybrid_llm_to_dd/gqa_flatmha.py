# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform
import ryzenai_onnx_utils.transform.cast as cast
import ryzenai_onnx_utils.transform.reshape
from ryzenai_onnx_utils.typing import PassOutputArgs, StrictPassOutputArgs, is_static_shape

from .rotary_embedding import create_rope


def pad_v_matmul(
    gqa: onnx.NodeProto,
    v_matmul_output: str,
    v_matmul_shape: list[int],
    pass_id: str,
    domain: str,
    extractor: onnx.utils.Extractor,
) -> StrictPassOutputArgs:
    new_nodes = []
    new_tvis = []
    new_initializers = []

    present_v_shape = ryzenai_onnx_utils.matcher.get_shape(gqa.output[2], extractor)
    reshaped_shape = [1, *v_matmul_shape]
    reshaped_output = v_matmul_output + ".reshaped"
    v_matmul_dtype = onnx.TensorProto.FLOAT16
    reshape, reshape_tvis, reshape_tensors = ryzenai_onnx_utils.transform.reshape.add_reshape(
        v_matmul_output,
        "present_v_reshape",
        reshaped_output,
        v_matmul_dtype,
        v_matmul_shape,
        reshaped_shape,
    )
    new_nodes.append(reshape)
    new_tvis.extend(reshape_tvis)
    new_initializers.append(reshape_tensors)

    # if the total_seq_len is dynamic, use an arbitrary placeholder value
    # to move forward. This pad now MUST be replaced by a strided matmulnbits
    # or the model will likely give an exception at runtime
    pad_value = present_v_shape[2] - v_matmul_shape[1] if isinstance(present_v_shape[2], int) else 511

    pad_tensor = onnx.helper.make_tensor(
        f"pad_tensor_{pass_id}",
        onnx.TensorProto.INT64,
        [8],
        [0, 0, 0, 0, 0, 0, pad_value, 0],
    )
    new_initializers.append(pad_tensor)
    pad = onnx.helper.make_node(
        "Pad",
        inputs=[reshaped_output, f"pad_tensor_{pass_id}"],
        outputs=[gqa.output[2]],
        name=gqa.name + ".pad",
    )
    new_nodes.append(pad)

    return new_nodes, new_initializers, new_tvis


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("FLATMHATTFT")
    q_matmul = subgraph[0]
    if len(subgraph) == 6:
        v_matmul = subgraph[2]
        q_rotary = subgraph[3]
        k_rotary = subgraph[4]
        gqa = subgraph[5]
    else:
        v_matmul = subgraph[12]
        q_rotary = subgraph[13]
        k_rotary = subgraph[14]
        gqa = subgraph[15]

    new_nodes: list[onnx.NodeProto] = []
    new_initializers: list[onnx.TensorProto] = []
    new_tvis: list[onnx.ValueInfoProto] = []

    q_rotary_nodes, q_rotary_initializers, q_rotary_tvis = create_rope(extractor, pass_id + "_1", q_rotary, params)
    new_nodes.extend(q_rotary_nodes)
    new_initializers.extend(q_rotary_initializers)
    new_tvis.extend(q_rotary_tvis)

    k_rotary_nodes_2, k_rotary_initializers_2, k_rotary_tvis_2 = create_rope(
        extractor, pass_id, k_rotary, params, gqa.output[1], False, True
    )
    new_nodes.extend(k_rotary_nodes_2)
    new_initializers.extend(k_rotary_initializers_2)
    new_tvis.extend(k_rotary_tvis_2)

    input_cast, input_cast_tvis = cast.add_cast_dtype_to_bfloat16_auto(gqa.input[0], pass_id + "_0", domain, extractor)
    new_nodes.extend(input_cast)
    new_tvis.extend(input_cast_tvis)

    mha_input_0 = gqa.output[1]
    mha_input_1 = gqa.output[2]
    output_cast, output_cast_tvis = cast.add_cast_bfloat16_to_dtype_auto(gqa.output[0], pass_id, domain, extractor)
    new_nodes.extend(output_cast)
    new_tvis.extend(output_cast_tvis)

    past_k_shape = ryzenai_onnx_utils.matcher.get_shape(gqa.input[3], extractor)
    assert is_static_shape(past_k_shape)

    v_matmul_output = v_matmul.output[0]
    head_size = past_k_shape[3]

    v_matmul_shape = ryzenai_onnx_utils.matcher.get_shape(gqa.input[2], extractor)
    assert isinstance(v_matmul_shape[2], int)
    total_seq_len = int(params.attributes["total_seq_len"])
    new_v_matmul_shape = [
        int(v_matmul_shape[2] / head_size),
        total_seq_len,
        head_size,
    ]

    pad_nodes, pad_initializers, pad_tvis = pad_v_matmul(
        gqa, v_matmul_output, new_v_matmul_shape, pass_id, domain, extractor
    )
    new_nodes.extend(pad_nodes)
    new_initializers.extend(pad_initializers)
    new_tvis.extend(pad_tvis)

    mha = onnx.helper.make_node(
        "FLATMHATTFT",
        inputs=[
            input_cast[0].output[0],
            mha_input_0,
            mha_input_1,
        ],
        outputs=[output_cast[0].input[0]],
        name=f"mha_{pass_id}",
        domain=domain,
    )
    output_shape = list(ryzenai_onnx_utils.matcher.get_shape(q_matmul.output[0], extractor))
    kv_num_heads = past_k_shape[1]
    num_heads = onnx.helper.get_node_attr_value(gqa, "num_heads")
    seq_len = output_shape[1]
    total_seq_len = past_k_shape[2]
    ryzenai_onnx_utils.matcher.add_attribute(
        mha, "input_shape", [num_heads, kv_num_heads, seq_len, total_seq_len, head_size]
    )
    ryzenai_onnx_utils.matcher.add_attribute(mha, "use_gm", "v2")
    match_index = int(pass_id.split("_")[-1])
    pdi_id = int(params.attributes.get("pdi_id", 0))
    if pdi_id != 0:
        ryzenai_onnx_utils.matcher.add_attribute(mha, "pdi_id", int(pdi_id))
    # controls how many outputs are being converted to external. 4 if past k/v
    # and present k/v; 2 if just present k/v
    buffer_factor = params.attributes.get("buffer_factor", 4)

    buffer_offset = (match_index) * buffer_factor
    alias_offset = (match_index) * 2

    external_buffers = []
    # present k
    external_buffers.extend([1, 0, buffer_offset, alias_offset])
    # present v
    external_buffers.extend([2, 0, buffer_offset + 1, alias_offset + 1])
    ryzenai_onnx_utils.matcher.add_attribute(mha, "external_buffers", external_buffers)
    new_nodes.append(mha)
    if len(subgraph) == 6:
        new_nodes.append(subgraph[0])
        new_nodes.append(subgraph[1])
        new_nodes.append(subgraph[2])
    else:
        new_nodes.append(subgraph[0])
        new_nodes.append(subgraph[1])
        new_nodes.append(subgraph[2])
        new_nodes.append(subgraph[3])
        new_nodes.append(subgraph[4])
        new_nodes.append(subgraph[5])
        new_nodes.append(subgraph[6])
        new_nodes.append(subgraph[7])
        new_nodes.append(subgraph[8])
        new_nodes.append(subgraph[9])
        new_nodes.append(subgraph[10])
        new_nodes.append(subgraph[11])
        new_nodes.append(subgraph[12])
    return new_nodes, new_initializers, new_tvis


PATTERN = [
    [
        "MatMulNBits([?,?,?,?,?], [a0])",  # q
        "MatMulNBits([?,?,?,?,?], [a1])",  # k
        "MatMulNBits([?,?,?,?,?], [a2])",  # v
        "RotaryEmbedding([a0,?,?,?], a3)",
        "RotaryEmbedding([a1,?,?,?], a4)",
        "GroupQueryAttention([a3,a4,a2,?,?,?,?,?,?], [?,?,?])",
    ],
    [
        "MatMulNBits([?,?,?,?,?], [a0])",  # q
        "Reshape([a0, ?], [c00])",
        "Cast(c00, c0)",
        "SimplifiedLayerNormalization([c0,?], b00)",
        "Cast(b00, b0)",
        "Reshape([b0, ?], [d0])",
        "MatMulNBits([?,?,?,?,?], [a1])",  # k
        "Reshape([a1, ?], [c01])",
        "Cast(c01, c1)",
        "SimplifiedLayerNormalization([c1,?], b01)",
        "Cast(b01, b1)",
        "Reshape([b1, ?], [d1])",
        "MatMulNBits([?,?,?,?,?], [a2])",  # v
        "RotaryEmbedding([d0,?,?,?], a3)",
        "RotaryEmbedding([d1,?,?,?], a4)",
        "GroupQueryAttention([a3,a4,a2,?,?,?,?,?,?], [?,?,?])",
    ],
]
REPLACEMENT = [replacement] * 2
