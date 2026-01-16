# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.transform.reshape import add_reshape
from ryzenai_onnx_utils.transform.transpose import add_transpose


def is_supported_pattern(extractor, trans0, rms0, rms1, trans1, mul0, mul1):
    trans0_perm = onnx.helper.get_node_attr_value(trans0, "perm")
    trans1_perm = onnx.helper.get_node_attr_value(trans1, "perm")
    if trans0_perm != [0, 2, 1, 3]:
        return False
    if trans1_perm != [0, 2, 3, 1]:
        return False
    rms0_axis = onnx.helper.get_node_attr_value(rms0, "axis")
    rms1_axis = onnx.helper.get_node_attr_value(rms1, "axis")
    if rms0_axis != -1 or rms1_axis != -1:
        return False
    rms0_input_shape = ryzenai_onnx_utils.matcher.get_shape(rms0.input[0], extractor)
    rms1_input_shape = ryzenai_onnx_utils.matcher.get_shape(rms1.input[0], extractor)
    if len(rms0_input_shape) != 4 or len(rms1_input_shape) != 4:
        return False
    is_mul0_initializer = ryzenai_onnx_utils.matcher.is_initializer(mul0.input[1], extractor)
    is_mul1_initializer = ryzenai_onnx_utils.matcher.is_initializer(mul1.input[1], extractor)
    return is_mul0_initializer and is_mul1_initializer


def replacement(
    extractor: "onnx.utils.Extractor",
    pass_id: str,
    subgraph: list,
    params: "ryzenai_onnx_utils.ReplaceParams",
):
    trans0, rms0, rms1, trans1, mul0, mul1, matmul0 = subgraph
    if not is_supported_pattern(extractor, trans0, rms0, rms1, trans1, mul0, mul1):
        return subgraph, [], None
    new_nodes = []
    new_tvis = []
    new_inits = []
    matmul_inputs = []
    for branch in ["q_branch", "k_branch"]:
        rms = rms0 if branch == "q_branch" else rms1
        trans = trans0 if branch == "q_branch" else trans1
        mul = mul0 if branch == "q_branch" else mul1
        branch_input_nodes = [trans0] if branch == "q_branch" else [rms1]
        # create rms
        rms_output_shape = ryzenai_onnx_utils.matcher.get_shape(branch_input_nodes[0].input[0], extractor)
        rms_out_tvi = onnx.helper.make_tensor_value_info(
            rms.output[0] + f"_{pass_id}",
            ryzenai_onnx_utils.matcher.get_dtype(rms.output[0], extractor),
            rms_output_shape,
        )
        new_tvis.append(rms_out_tvi)
        new_rmsnorm = onnx.helper.make_node(
            "SimplifiedLayerNormalization",
            inputs=[branch_input_nodes[0].input[0], rms.input[1]],
            outputs=[rms_out_tvi.name],
            name=rms.name + f"_out_{pass_id}",
        )
        ryzenai_onnx_utils.matcher.copy_attributes(rms, new_rmsnorm)
        new_nodes.append(new_rmsnorm)
        # create reshape 4dims->3dims
        new_reshape_in_shape = rms_output_shape
        new_reshape_out_shape = [
            new_reshape_in_shape[0],
            new_reshape_in_shape[1],
            new_reshape_in_shape[2] * new_reshape_in_shape[3],
        ]
        new_reshape, new_reshape_tvis, new_reshape_init = add_reshape(
            input_name=new_rmsnorm.output[0],
            shape_name=new_rmsnorm.output[0] + "_shape_" + pass_id,
            output_name=new_rmsnorm.output[0] + "_reshape_" + pass_id,
            dtype=ryzenai_onnx_utils.matcher.get_dtype(rms0.output[0], extractor),
            in_shape=new_reshape_in_shape,
            out_shape=new_reshape_out_shape,
        )
        new_nodes.append(new_reshape)
        new_tvis.extend(new_reshape_tvis)
        new_inits.append(new_reshape_init)

        # create reshape 3dims->4dims
        new_reshape1_in_shape = new_reshape_out_shape
        new_reshape1_out_shape = rms_output_shape
        new_reshape1, new_reshape1_tvis, new_reshape1_init = add_reshape(
            input_name=new_reshape.output[0],
            shape_name=new_reshape.output[0] + "_shape_" + pass_id,
            output_name=new_reshape.output[0] + "_reshape_" + pass_id,
            dtype=ryzenai_onnx_utils.matcher.get_dtype(rms.output[0], extractor),
            in_shape=new_reshape1_in_shape,
            out_shape=new_reshape1_out_shape,
        )
        new_nodes.append(new_reshape1)
        new_tvis.extend(new_reshape1_tvis)
        new_inits.append(new_reshape1_init)
        # create transpose
        trans_permute = onnx.helper.get_node_attr_value(trans, "perm")
        new_trans_input_shape = new_reshape1_out_shape
        new_trans_output_shape = [new_trans_input_shape[dim] for dim in trans_permute]
        new_trans, new_trans_tvis = add_transpose(
            node_name=trans.name + "_" + pass_id,
            input_name=new_reshape1.output[0],
            output_name=trans.output[0] + "_" + pass_id,
            dtype=ryzenai_onnx_utils.matcher.get_dtype(trans.output[0], extractor),
            shape_prev=new_trans_input_shape,
            shape_after=new_trans_output_shape,
            perm_vec=trans_permute,
        )
        new_nodes.append(new_trans)
        new_tvis.extend(new_trans_tvis)
        # create mul
        new_mul_out_tvi = onnx.helper.make_tensor_value_info(
            mul.output[0] + f"_{pass_id}",
            ryzenai_onnx_utils.matcher.get_dtype(mul.output[0], extractor),
            ryzenai_onnx_utils.matcher.get_shape(mul.output[0], extractor),
        )
        new_tvis.append(new_mul_out_tvi)
        new_mul = onnx.helper.make_node(
            "Mul",
            inputs=[new_trans.output[0], mul.input[1]],
            outputs=[new_mul_out_tvi.name],
            name=mul.name + f"_{pass_id}",
        )
        new_nodes.append(new_mul)
        new_tvis.append(new_mul_out_tvi)
        matmul_inputs.append(new_mul.output[0])
    # create matmul
    new_matmul = onnx.helper.make_node(
        "MatMul",
        inputs=matmul_inputs,
        outputs=matmul0.output,
        name=matmul0.name + f"_{pass_id}",
    )
    new_nodes.append(new_matmul)

    return new_nodes, new_inits, new_tvis


PATTERN = [
    "Transpose([?], t0)",
    "SimplifiedLayerNormalization([t0, ?], s0)",
    "SimplifiedLayerNormalization([?, ?], s1)",
    "Transpose([s1], t1)",
    "Mul([s0, ?], m0)",
    "Mul([t1, ?], m1)",
    "MatMul([m0, m1], m2)",
]
REPLACEMENT = replacement
