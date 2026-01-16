# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass unfuses QKV matmul that feeds into GroupQueryAttention.
Fusion flow requires this due to specific matmuls used for transpose/updating kv cache.
"""

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs, SubPass


def get_new_name(name: str, target: str) -> str:
    if "qkv_proj" in name:
        return name.replace("qkv_proj", target)
    else:
        return name + "." + target


def get_split(name: str):
    # Note: there are downstream passes that rely on name match on v_proj etc to
    # associate it with v matmul
    q_name = get_new_name(name, "q_proj")
    k_name = get_new_name(name, "k_proj")
    v_name = get_new_name(name, "v_proj")

    return q_name, k_name, v_name


def rename_matmulnbits_tensors(weight_name, bias_name, scales_name, zero_point_name, target):
    new_weight_name = get_new_name(weight_name, target)
    new_bias_name = get_new_name(bias_name, target)
    new_scales_name = get_new_name(scales_name, target)
    new_zero_point_name = get_new_name(zero_point_name, target)

    return new_weight_name, new_bias_name, new_scales_name, new_zero_point_name


def create_tvi(name, dtype, shape):
    return onnx.helper.make_tensor_value_info(name, dtype, shape)


def create_matmulnbits_const_tvi(
    weight_name,
    bias_name,
    scales_name,
    zero_point_name,
    weight_tensor,
    bias_tensor,
    scales_tensor,
    zero_point_tensor,
    weight_dtype,
    bias_dtype,
    scale_dtype,
    zero_point_dtype,
):
    new_tvis = []

    new_tvis.append(onnx.helper.make_tensor_value_info(weight_name, weight_dtype, weight_tensor.shape))

    new_tvis.append(onnx.helper.make_tensor_value_info(bias_name, bias_dtype, bias_tensor.shape))

    new_tvis.append(onnx.helper.make_tensor_value_info(scales_name, scale_dtype, scales_tensor.shape))

    new_tvis.append(onnx.helper.make_tensor_value_info(zero_point_name, zero_point_dtype, zero_point_tensor.shape))

    return new_tvis


def create_matmulnbits_initializers(
    weight_name,
    bias_name,
    scales_name,
    zero_point_name,
    weight_tensor,
    bias_tensor,
    scales_tensor,
    zero_point_tensor,
    weight_dtype,
    bias_dtype,
    scale_dtype,
    zero_point_dtype,
):
    initializer = []

    initializer.append(
        onnx.helper.make_tensor(
            weight_name,
            weight_dtype,
            weight_tensor.shape,
            weight_tensor,
        )
    )
    initializer.append(
        onnx.helper.make_tensor(
            bias_name,
            bias_dtype,
            bias_tensor.shape,
            bias_tensor,
        )
    )
    initializer.append(
        onnx.helper.make_tensor(
            scales_name,
            scale_dtype,
            scales_tensor.shape,
            scales_tensor,
        )
    )
    initializer.append(
        onnx.helper.make_tensor(
            zero_point_name,
            zero_point_dtype,
            zero_point_tensor.shape,
            zero_point_tensor,
        )
    )

    return initializer


def split_matmulnbits_node(
    qkv_matmul: onnx.NodeProto, N_q: int, N_k: int, N_v: int, output_names: list[str], extractor: onnx.utils.Extractor
):
    new_nodes = []
    new_initializers = []
    new_tvis = []

    if output_names:
        qoutput_name, koutput_name, voutput_name = output_names
    else:
        qoutput_name, koutput_name, voutput_name = get_split(qkv_matmul.output[0])
    qmatmul_name, kmatmul_name, vmatmul_name = get_split(qkv_matmul.name)

    qkv_matmul_shape = ryzenai_onnx_utils.matcher.get_shape(qkv_matmul.output[0], extractor)

    out_dtype = ryzenai_onnx_utils.matcher.get_dtype(qkv_matmul.output[0], extractor)

    K_qkv = onnx.helper.get_node_attr_value(qkv_matmul, "K")
    block_size_qkv = onnx.helper.get_node_attr_value(qkv_matmul, "block_size")
    bits_qkv = onnx.helper.get_node_attr_value(qkv_matmul, "bits")

    q_matmul_shape = qkv_matmul_shape[:-1] + (N_q,)
    k_matmul_shape = qkv_matmul_shape[:-1] + (N_k,)
    v_matmul_shape = qkv_matmul_shape[:-1] + (N_v,)

    # assumption here is normalize matmulnbits has already been called
    # [N_qkv, num_blocks, BLOCK_SIZE = 128/64], num_blocks = ceil(K_qkv / BLOCK_SIZE)
    num_blocks = (K_qkv + block_size_qkv - 1) // block_size_qkv
    weight_name = qkv_matmul.input[1]
    weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(weight_name, extractor)
    weight_dtype = ryzenai_onnx_utils.matcher.get_dtype(weight_name, extractor)
    # [N_qkv, num_blocks]
    scales_name = qkv_matmul.input[2]
    scales = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(scales_name, extractor)
    scales_dtype = ryzenai_onnx_utils.matcher.get_dtype(scales_name, extractor)
    # [N_qkv, ceil(num_blocks * bits / 8)]
    # essentially if num_blocks is odd, there is padding end of row
    num_blocks_zp = (num_blocks * bits_qkv + 8 - 1) // 8
    zero_point_name = qkv_matmul.input[3]
    zero_point = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(zero_point_name, extractor)
    zero_point_dtype = ryzenai_onnx_utils.matcher.get_dtype(zero_point_name, extractor)
    # [N_qkv, 1]
    bias_name = qkv_matmul.input[5]
    bias = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(bias_name, extractor)
    bias_dtype = ryzenai_onnx_utils.matcher.get_dtype(bias_name, extractor)

    # weight and bias are already split on column basis
    # scales and zero-point are single dim that needs to be split

    # Q matmul input tensors/names
    q_weight = weight[0:N_q]
    q_bias = bias[0:N_q]
    q_scales = scales[0 : N_q * num_blocks]
    q_zp = zero_point[0 : N_q * num_blocks_zp]

    q_weight_name, q_bias_name, q_scales_name, q_zp_name = rename_matmulnbits_tensors(
        weight_name, bias_name, scales_name, zero_point_name, "q_proj"
    )

    new_tvis.extend(
        create_matmulnbits_const_tvi(
            q_weight_name,
            q_bias_name,
            q_scales_name,
            q_zp_name,
            q_weight,
            q_bias,
            q_scales,
            q_zp,
            weight_dtype,
            bias_dtype,
            scales_dtype,
            zero_point_dtype,
        )
    )

    new_initializers.extend(
        create_matmulnbits_initializers(
            q_weight_name,
            q_bias_name,
            q_scales_name,
            q_zp_name,
            q_weight,
            q_bias,
            q_scales,
            q_zp,
            weight_dtype,
            bias_dtype,
            scales_dtype,
            zero_point_dtype,
        )
    )

    # K matmul input tensors/names
    k_weight = weight[N_q : N_q + N_k]
    k_bias = bias[N_q : N_q + N_k]
    k_scales = scales[N_q * num_blocks : (N_q + N_k) * num_blocks]
    k_zp = zero_point[N_q * num_blocks_zp : (N_q + N_k) * num_blocks_zp]

    k_weight_name, k_bias_name, k_scales_name, k_zp_name = rename_matmulnbits_tensors(
        weight_name, bias_name, scales_name, zero_point_name, "k_proj"
    )

    new_tvis.extend(
        create_matmulnbits_const_tvi(
            k_weight_name,
            k_bias_name,
            k_scales_name,
            k_zp_name,
            k_weight,
            k_bias,
            k_scales,
            k_zp,
            weight_dtype,
            bias_dtype,
            scales_dtype,
            zero_point_dtype,
        )
    )

    new_initializers.extend(
        create_matmulnbits_initializers(
            k_weight_name,
            k_bias_name,
            k_scales_name,
            k_zp_name,
            k_weight,
            k_bias,
            k_scales,
            k_zp,
            weight_dtype,
            bias_dtype,
            scales_dtype,
            zero_point_dtype,
        )
    )

    # V matmul input tensors/names
    v_weight = weight[N_q + N_k :]
    v_bias = bias[N_q + N_k :]
    v_scales = scales[(N_q + N_k) * num_blocks :]
    v_zp = zero_point[(N_q + N_k) * num_blocks_zp :]

    v_weight_name, v_bias_name, v_scales_name, v_zp_name = rename_matmulnbits_tensors(
        weight_name, bias_name, scales_name, zero_point_name, "v_proj"
    )

    new_tvis.extend(
        create_matmulnbits_const_tvi(
            v_weight_name,
            v_bias_name,
            v_scales_name,
            v_zp_name,
            v_weight,
            v_bias,
            v_scales,
            v_zp,
            weight_dtype,
            bias_dtype,
            scales_dtype,
            zero_point_dtype,
        )
    )

    new_initializers.extend(
        create_matmulnbits_initializers(
            v_weight_name,
            v_bias_name,
            v_scales_name,
            v_zp_name,
            v_weight,
            v_bias,
            v_scales,
            v_zp,
            weight_dtype,
            bias_dtype,
            scales_dtype,
            zero_point_dtype,
        )
    )

    q_node = onnx.helper.make_node(
        "MatMulNBits",
        inputs=[qkv_matmul.input[0], q_weight_name, q_scales_name, q_zp_name, "", q_bias_name],
        outputs=[qoutput_name],
        name=qmatmul_name,
        domain=qkv_matmul.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(qkv_matmul, q_node)
    ryzenai_onnx_utils.matcher.set_attribute(q_node, "N", N_q)

    k_node = onnx.helper.make_node(
        "MatMulNBits",
        inputs=[qkv_matmul.input[0], k_weight_name, k_scales_name, k_zp_name, "", k_bias_name],
        outputs=[koutput_name],
        name=kmatmul_name,
        domain=qkv_matmul.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(qkv_matmul, k_node)
    ryzenai_onnx_utils.matcher.set_attribute(k_node, "N", N_k)

    v_node = onnx.helper.make_node(
        "MatMulNBits",
        inputs=[qkv_matmul.input[0], v_weight_name, v_scales_name, v_zp_name, "", v_bias_name],
        outputs=[voutput_name],
        name=vmatmul_name,
        domain=qkv_matmul.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(qkv_matmul, v_node)
    ryzenai_onnx_utils.matcher.set_attribute(v_node, "N", N_v)

    new_nodes.append(q_node)
    new_nodes.append(k_node)
    new_nodes.append(v_node)

    if not output_names:
        new_tvis.append(create_tvi(qoutput_name, out_dtype, q_matmul_shape))
        new_tvis.append(create_tvi(koutput_name, out_dtype, k_matmul_shape))
        new_tvis.append(create_tvi(voutput_name, out_dtype, v_matmul_shape))

    return new_nodes, new_initializers, new_tvis


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    qkv_matmul = subgraph[0]

    # assuming matmulnbits have weights, scales, zero points and bias
    assert len(qkv_matmul.input) == 6

    qkv_matmul_shape = ryzenai_onnx_utils.matcher.get_shape(qkv_matmul.output[0], extractor)
    N_qkv = qkv_matmul_shape[-1]

    if subgraph[1].op_type == "GroupQueryAttention":
        gqa = subgraph[1]
        if gqa.input[1] != "":
            # non-empty key and value input means Q/K/V are already split
            return subgraph, [], None
        k_cache_shape = ryzenai_onnx_utils.matcher.get_shape(gqa.input[3], extractor)
        v_cache_shape = ryzenai_onnx_utils.matcher.get_shape(gqa.input[4], extractor)

        k_cache_hidden_dim = k_cache_shape[-1]
        v_cache_hidden_dim = v_cache_shape[-1]

        kv_num_heads = onnx.helper.get_node_attr_value(gqa, "kv_num_heads")
        # num_heads = onnx.helper.get_node_attr_value(gqa, "num_heads")

        N_k = kv_num_heads * k_cache_hidden_dim
        N_v = kv_num_heads * v_cache_hidden_dim
        N_q = N_qkv - N_k - N_v

        output_names = []
    elif subgraph[1].op_type == "Split":
        split = subgraph[1]
        axis = ryzenai_onnx_utils.matcher.get_attribute(split, "axis", 0)
        if axis != 2 or len(split.output) != 3:
            # if we're not splitting on the last axis or for something different
            # than 3 outputs, this is not supported
            return subgraph, [], None
        assert N_qkv % 3 == 0, "QKV dimension not divisible by 3 for Split op"
        N_q = N_k = N_v = N_qkv // 3

        output_names = [split.output[0], split.output[1], split.output[2]]
    else:
        raise ValueError("Unsupported second op for hybrid_llm_split_qkv: " + subgraph[0].op_type)

    new_nodes, new_tensors, new_tvis = split_matmulnbits_node(qkv_matmul, N_q, N_k, N_v, output_names, extractor)

    if subgraph[1].op_type == "GroupQueryAttention":
        qoutput_name, koutput_name, voutput_name = get_split(qkv_matmul.output[0])
        gqa.input[0] = qoutput_name
        gqa.input[1] = koutput_name
        gqa.input[2] = voutput_name
        new_nodes.append(gqa)

    return new_nodes, new_tensors, new_tvis


PATTERN = [
    SubPass(
        "GQA",
        [
            "MatMulNBits([?,?,?,?,?], [a0])",  # qkv
            "GroupQueryAttention([a0,?,?,?,?,?,?,?,?], [?,?,?])",
        ],
    ),
    SubPass(
        "Split",
        [
            "MatMulNBits([?,?,?,?,?], [a0])",  # qkv
            "Split([a0,?,?], [?,?,?])",
        ],
    ),
]
REPLACEMENT = replacement
