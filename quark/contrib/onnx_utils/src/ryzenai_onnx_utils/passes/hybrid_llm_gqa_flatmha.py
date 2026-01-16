# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces the q, k, v MatMulNBits and GQA pattern with a DD FlatMHA
operator for DD fusion. It can optionally fuse the qk matmulnbits into one as
well or keep them separate depending on the inputs of FLATMHA.
"""

import contextlib
from collections.abc import Sequence

import ml_dtypes
import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform
import ryzenai_onnx_utils.transform.reshape
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype_auto,
    add_cast_dtype_to_bfloat16_auto,
)
from ryzenai_onnx_utils.typing import PassOutputArgs, ShapeType, SubPass


def concatenate_initializers(
    q_matmul: onnx.NodeProto, k_matmul: onnx.NodeProto, index: int, extractor: onnx.utils.Extractor
) -> tuple[onnx.TensorProto, onnx.ValueInfoProto]:
    q_np = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(q_matmul.input[index], extractor)
    k_np = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(k_matmul.input[index], extractor)

    qk_np = np.concatenate((q_np, k_np), 0)
    qk_np_name: str = q_matmul.input[index].replace("q_proj", "qk_proj")
    qk_np_name = qk_np_name.replace("Add", "MatMulNBits")
    qk_tensor = onnx.numpy_helper.from_array(qk_np, qk_np_name)
    tvi = onnx.helper.make_tensor_value_info(qk_np_name, onnx.helper.np_dtype_to_tensor_dtype(qk_np.dtype), qk_np.shape)
    return qk_tensor, tvi


def get_qk_attribute(matmuls: Sequence[onnx.NodeProto], attr_name: str, check_equal: bool) -> onnx.AttributeProto:
    """
    Go through the qk matmuls and extract the attribute values. Check if they're
    equal, except for the case of N where we add them together

    Args:
        matmuls (Sequence[NodeProto]): q, k matmuls
        attr_name (str): Name of the attribute to get
        check_equal (bool): Check if attribute value across matmuls are equal
    """
    attr_value = None
    for matmul in matmuls:
        if attr_value is None:
            attr_value = onnx.helper.get_node_attr_value(matmul, attr_name)
        else:
            new_value = onnx.helper.get_node_attr_value(matmul, attr_name)
            if check_equal:
                assert attr_value == new_value
            else:
                # for N
                attr_value += new_value
    return onnx.helper.make_attribute(attr_name, attr_value)


def replace_output_shape(graph: onnx.GraphProto, output_name: str, dtype: int, new_shape: ShapeType) -> None:
    index = next(i for i, v in enumerate(graph.output) if v.name == output_name)
    graph.output.pop(index)
    new_tvi = onnx.helper.make_tensor_value_info(output_name, dtype, new_shape)
    graph.output.insert(index, new_tvi)


def create_sin_cos_cache(extractor: onnx.utils.Extractor) -> tuple[list[onnx.NodeProto], list[onnx.ValueInfoProto]]:
    new_nodes = []
    new_tvis = []

    sin_cache = ryzenai_onnx_utils.matcher.get_initializer_as_numpy("sin_cache", extractor)
    cos_cache = ryzenai_onnx_utils.matcher.get_initializer_as_numpy("cos_cache", extractor)
    sin_cos_cache = np.concatenate((cos_cache, sin_cache), 1)

    if sin_cos_cache.dtype != ml_dtypes.bfloat16:
        sin_cos_cache = sin_cos_cache.astype(ml_dtypes.bfloat16)
    sin_cos_cache_tensor = onnx.helper.make_tensor(
        "sin_cos_cache_token",
        onnx.TensorProto.BFLOAT16,
        sin_cos_cache.shape,
        sin_cos_cache.tobytes(),
        True,
    )

    sin_cos_cache_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["sin_cos_cache_token"],
        value=sin_cos_cache_tensor,
        name="sin_cos_cache",
    )
    new_tvi = onnx.helper.make_tensor_value_info(
        "sin_cos_cache_token",
        onnx.TensorProto.BFLOAT16,
        sin_cos_cache.shape,
    )
    new_nodes.append(sin_cos_cache_node)
    new_tvis.append(new_tvi)

    return new_nodes, new_tvis


def add_attention_mask() -> tuple[list[onnx.NodeProto], list[onnx.ValueInfoProto]]:
    new_nodes = []
    new_tvis = []

    attn_mask_node = onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=["attention_mask_const"],
        value_ints=[1],
        name="attn_mask_node",
    )
    new_nodes.append(attn_mask_node)
    new_tvis.append(
        onnx.helper.make_tensor_value_info(
            "attention_mask_const",
            onnx.TensorProto.INT64,
            [1],
        )
    )

    cast_node = onnx.helper.make_node(
        "Cast",
        # before sub
        inputs=["/model/attn_mask_reformat/attn_mask_subgraph/ReduceSum/output_0"],
        # after sub
        # inputs=["/model/attn_mask_reformat/attn_mask_subgraph/Sub/output_0"],
        outputs=["attention_mask_casted"],
        name="attn_cast",
        to=onnx.TensorProto.UINT32,
    )
    new_nodes.append(cast_node)
    new_tvis.append(
        onnx.helper.make_tensor_value_info(
            "attention_mask_casted",
            onnx.TensorProto.UINT32,
            [1, 1],
        )
    )

    reshape_node = onnx.helper.make_node(
        "Reshape",
        name="attn_mask_reshape",
        inputs=["attention_mask_casted", "attention_mask_const"],
        outputs=["attention_mask_const_uint"],
    )
    new_tvis.append(
        onnx.helper.make_tensor_value_info(
            "attention_mask_const_uint",
            onnx.TensorProto.UINT32,
            [1],
        )
    )
    new_nodes.append(reshape_node)

    return new_nodes, new_tvis


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    fuse_qk_mha = params.get_bool_attr("fuse_qk_mha", True)
    domain = params.get_domain("FLATMHA")

    new_nodes = []
    if len(subgraph) == 4:
        q_matmul = subgraph[0]
        k_matmul = subgraph[1]
        v_matmul = subgraph[2]
        gqa = subgraph[3]
    else:
        q_matmul = subgraph[0]
        k_matmul = subgraph[4]
        v_matmul = subgraph[8]
        gqa = subgraph[9]
        assert not fuse_qk_mha, "Cannot fuse qk matmulnbits when SLN is present"
        # add back the SLNs and reshapes
        new_nodes.extend(subgraph[1:4])
        new_nodes.extend(subgraph[5:8])

    # assuming matmulnbits have weights, scales, zero points, g_idx, and bias
    assert len(q_matmul.input) == len(k_matmul.input) == len(v_matmul.input)
    assert len(q_matmul.input) == 6, len(q_matmul.input)
    assert q_matmul.input[0] == k_matmul.input[0] == v_matmul.input[0]

    new_tvis = []
    new_initializers = []
    output_shape = list(ryzenai_onnx_utils.matcher.get_shape(q_matmul.output[0], extractor))
    if fuse_qk_mha:
        qk_weights_tensor, weights_tvi = concatenate_initializers(q_matmul, k_matmul, 1, extractor)
        qk_scales_tensor, scales_tvi = concatenate_initializers(q_matmul, k_matmul, 2, extractor)
        qk_zp_tensor, zp_tvi = concatenate_initializers(q_matmul, k_matmul, 3, extractor)

        qk_bias_tensor, bias_tvi = concatenate_initializers(q_matmul, k_matmul, 5, extractor)
        new_tvis.extend([weights_tvi, scales_tvi, zp_tvi, bias_tvi])
        qk_inputs = [
            q_matmul.input[0],
            qk_weights_tensor.name,
            qk_scales_tensor.name,
            qk_zp_tensor.name,
            "",  # assuming g_idx is not used
            qk_bias_tensor.name,
        ]
        new_initializers.extend([qk_weights_tensor, qk_scales_tensor, qk_zp_tensor, qk_bias_tensor])

        qk_output_name = q_matmul.output[0].replace("q_proj", "qk_proj")

        qk_matmul = onnx.helper.make_node(
            "MatMulNBits",
            inputs=qk_inputs,
            outputs=[qk_output_name],
            domain=q_matmul.domain,
            name=f"MatMulNBits_{pass_id}",
        )

        with contextlib.suppress(ValueError):
            qk_matmul.attribute.append(get_qk_attribute((q_matmul, k_matmul), "accuracy_level", True))
        qk_matmul.attribute.append(get_qk_attribute((q_matmul, k_matmul), "bits", True))
        qk_matmul.attribute.append(get_qk_attribute((q_matmul, k_matmul), "block_size", True))
        qk_matmul.attribute.append(get_qk_attribute((q_matmul, k_matmul), "K", True))
        qk_matmul.attribute.append(get_qk_attribute((q_matmul, k_matmul), "N", False))

        new_nodes.append(qk_matmul)

        output_shape[-1] = onnx.helper.get_node_attr_value(qk_matmul, "N")
        output_dtype = ryzenai_onnx_utils.matcher.get_dtype(q_matmul.output[0], extractor)
        output_tvi = onnx.helper.make_tensor_value_info(
            qk_output_name,
            output_dtype,
            output_shape,
        )
        new_tvis.append(output_tvi)
    else:
        new_nodes.append(q_matmul)
        new_nodes.append(k_matmul)
    new_nodes.append(v_matmul)

    v_matmul_shape = ryzenai_onnx_utils.matcher.get_shape(v_matmul.output[0], extractor)
    present_k_shape = ryzenai_onnx_utils.matcher.get_shape(gqa.output[1], extractor)
    present_v_shape = ryzenai_onnx_utils.matcher.get_shape(gqa.output[2], extractor)

    replace_output_shape(
        extractor.graph,
        gqa.output[1],
        ryzenai_onnx_utils.matcher.get_dtype(gqa.output[1], extractor),
        present_k_shape,
    )
    replace_output_shape(
        extractor.graph,
        gqa.output[2],
        ryzenai_onnx_utils.matcher.get_dtype(gqa.output[2], extractor),
        present_v_shape,
    )

    assert isinstance(present_v_shape[1], int) and isinstance(present_v_shape[3], int)
    assert present_v_shape[0] == v_matmul_shape[0]
    assert present_v_shape[1] * present_v_shape[3] == v_matmul_shape[2]
    reshaped_shape = [present_v_shape[0], present_v_shape[1], 1, present_v_shape[3]]
    reshaped_output = v_matmul.output[0] + ".reshaped"
    v_matmul_dtype = ryzenai_onnx_utils.matcher.get_dtype(v_matmul.output[0], extractor)
    reshape, reshape_tvis, reshape_tensors = ryzenai_onnx_utils.transform.reshape.add_reshape(
        v_matmul.output[0],
        "present_v_reshape",
        reshaped_output,
        v_matmul_dtype,
        v_matmul_shape,
        reshaped_shape,
    )
    new_nodes.append(reshape)
    new_tvis.extend(reshape_tvis)
    new_initializers.append(reshape_tensors)

    pad_tensor = onnx.helper.make_tensor(
        "pad_tensor",
        onnx.TensorProto.INT64,
        [8],
        # this is a placeholder that should get removed when the op is replaced
        [0, 0, 0, 0, 0, 0, 4096 - 1, 0],
    )
    new_initializers.append(pad_tensor)
    pad = onnx.helper.make_node(
        "Pad",
        inputs=[reshaped_output, "pad_tensor"],
        outputs=[gqa.output[2]],
        name=v_matmul.name + ".pad",
    )
    new_nodes.append(pad)

    if fuse_qk_mha:
        pre_cast_qk, pre_cast_qk_tvi = add_cast_dtype_to_bfloat16_auto(
            qk_output_name, pass_id, domain, extractor, output_shape, output_dtype
        )
        new_nodes.extend(pre_cast_qk)
        new_tvis.extend(pre_cast_qk_tvi)
    else:
        output_shape = list(ryzenai_onnx_utils.matcher.get_shape(gqa.input[0], extractor))
        output_dtype = ryzenai_onnx_utils.matcher.get_dtype(gqa.input[0], extractor)
        pre_cast_q, pre_cast_q_tvi = add_cast_dtype_to_bfloat16_auto(
            gqa.input[0], pass_id, domain, extractor, output_shape, output_dtype
        )
        new_nodes.extend(pre_cast_q)
        new_tvis.extend(pre_cast_q_tvi)
        output_shape = list(ryzenai_onnx_utils.matcher.get_shape(gqa.input[1], extractor))
        output_dtype = ryzenai_onnx_utils.matcher.get_dtype(gqa.input[1], extractor)
        pre_cast_k, pre_cast_k_tvi = add_cast_dtype_to_bfloat16_auto(
            gqa.input[1], pass_id, domain, extractor, output_shape, output_dtype
        )
        new_nodes.extend(pre_cast_k)
        new_tvis.extend(pre_cast_k_tvi)

    if ryzenai_onnx_utils.matcher.get_dtype(gqa.input[3], extractor) != onnx.TensorProto.BFLOAT16:
        pre_cast_past_k, pre_cast_past_k_tvi = add_cast_dtype_to_bfloat16_auto(gqa.input[3], pass_id, domain, extractor)
        new_nodes.extend(pre_cast_past_k)
        new_tvis.extend(pre_cast_past_k_tvi)

        pre_cast_past_v, pre_cast_past_v_tvi = add_cast_dtype_to_bfloat16_auto(gqa.input[4], pass_id, domain, extractor)
        new_nodes.extend(pre_cast_past_v)
        new_tvis.extend(pre_cast_past_v_tvi)

        kv_inputs = [pre_cast_past_k[0].output[0], pre_cast_past_v[0].output[0]]
    else:
        kv_inputs = [gqa.input[3], gqa.input[4]]

    # use a common tensor across the whole model
    sin_cos_exists = ryzenai_onnx_utils.matcher.find_consts("sin_cos_cache_token", extractor.graph)
    if not sin_cos_exists:
        sin_cos_cache, sin_cos_cache_tvi = create_sin_cos_cache(extractor)
        new_nodes.extend(sin_cos_cache)
        new_tvis.extend(sin_cos_cache_tvi)

    attention_mask_exists = ryzenai_onnx_utils.matcher.find_nodes_by_output(
        "attention_mask_const_uint", extractor.graph
    )
    if not attention_mask_exists:
        attn_mask_nodes, attn_mask_tvis = add_attention_mask()
        new_nodes.extend(attn_mask_nodes)
        new_tvis.extend(attn_mask_tvis)

    new_inputs = [pre_cast_qk[0].output[0]] if fuse_qk_mha else [pre_cast_q[0].output[0], pre_cast_k[0].output[0]]

    new_inputs.extend(
        [
            # past key and value
            *kv_inputs,
            # attn mask - this should be discovered by tracing back gqa.input[5] to the model input
            "attention_mask_const_uint",
            # sin cos cache
            "sin_cos_cache_token",
        ]
    )
    past_k_shape = ryzenai_onnx_utils.matcher.get_shape(gqa.input[3], extractor)
    kv_num_heads = past_k_shape[1]
    num_heads = onnx.helper.get_node_attr_value(gqa, "num_heads")
    seq_len = output_shape[1]
    total_seq_len = past_k_shape[2]
    token_len = total_seq_len
    head_size = past_k_shape[3]
    is_dyn = False
    if (
        "dynamic_shape_list" in params.attributes
        and "attention_mask_padded" in params.attributes["dynamic_shape_list"][0]
    ):
        new_inputs.append("attention_mask_padded")
        mask_shape = ryzenai_onnx_utils.matcher.get_shape("attention_mask_padded", extractor)
        token_len = mask_shape[1]
        is_dyn = True
    post_cast_output, post_cast_output_tvi = add_cast_bfloat16_to_dtype_auto(gqa.output[0], pass_id, domain, extractor)
    new_nodes.extend(post_cast_output)
    new_tvis.extend(post_cast_output_tvi)

    if ryzenai_onnx_utils.matcher.get_dtype(gqa.output[1], extractor) != onnx.TensorProto.BFLOAT16:
        post_cast_present_k, post_cast_present_k_tvi = add_cast_bfloat16_to_dtype_auto(
            gqa.output[1], pass_id, domain, extractor
        )
        new_nodes.extend(post_cast_present_k)
        new_tvis.extend(post_cast_present_k_tvi)

        kv_output = post_cast_present_k[0].input[0]
    else:
        kv_output = gqa.output[1]

    new_outputs = [post_cast_output[0].input[0], kv_output]

    new_node = onnx.helper.make_node(
        "FLATMHA",
        inputs=new_inputs,
        outputs=new_outputs,
        name=gqa.name,
        domain=domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(gqa, new_node)

    kv_num_heads = past_k_shape[1]
    num_heads = onnx.helper.get_node_attr_value(gqa, "num_heads")
    seq_len = output_shape[1]
    total_seq_len = past_k_shape[2]
    head_size = past_k_shape[3]
    if is_dyn:
        input_shapes = [kv_num_heads, num_heads, seq_len, token_len, head_size, total_seq_len]
    else:
        input_shapes = [kv_num_heads, num_heads, seq_len, total_seq_len, head_size]
    if ryzenai_onnx_utils.matcher.has_attribute(gqa, "rotary_embedding_dim"):
        rope_dim = onnx.helper.get_node_attr_value(gqa, "rotary_embedding_dim")
        ryzenai_onnx_utils.matcher.add_attribute(new_node, "rotary_embedding_dim", rope_dim)
        input_shapes.append(rope_dim)
    ryzenai_onnx_utils.matcher.add_attribute(
        new_node,
        "input_shape",
        input_shapes,
    )
    if not fuse_qk_mha:
        ryzenai_onnx_utils.matcher.add_attribute(new_node, "split_qk", 1)
    lora = params.get_bool_attr("lora", False)
    if lora:
        ryzenai_onnx_utils.matcher.add_attribute(new_node, "lora", lora)
    enable_ctrl_pkt = params.get_bool_attr("enable_ctrl_pkt", False)
    if enable_ctrl_pkt:
        ryzenai_onnx_utils.matcher.add_attribute(new_node, "enable_ctrl_pkt", enable_ctrl_pkt)

    new_nodes.append(new_node)

    return new_nodes, new_initializers, new_tvis


PATTERN = [
    SubPass(
        "Basic",
        [
            "MatMulNBits([?,?,?,?,?], [a3])",  # q
            "MatMulNBits([?,?,?,?,?], [a4])",  # k
            "MatMulNBits([?,?,?,?,?], [a2])",  # v
            "GroupQueryAttention([a3,a4,a2,?,?,?,?,?,?], [?,?,?])",
        ],
    ),
    SubPass(
        "Per-head normalization",
        [
            "MatMulNBits([?,?,?,?,?], [a0])",  # q
            "Reshape([a0, ?], [c0])",
            "SimplifiedLayerNormalization([c0,?], b0)",
            "Reshape([b0, ?], [d0])",
            "MatMulNBits([?,?,?,?,?], [a1])",  # k
            "Reshape([a1, ?], [c1])",
            "SimplifiedLayerNormalization([c1,?], b1)",
            "Reshape([b1, ?], [d1])",
            "MatMulNBits([?,?,?,?,?], [a2])",  # v
            "GroupQueryAttention([d0,d1,a2,?,?,?,?,?,?], [?,?,?])",
        ],
    ),
]
REPLACEMENT = replacement
