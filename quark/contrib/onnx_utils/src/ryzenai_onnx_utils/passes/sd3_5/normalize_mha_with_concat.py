# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.transform.reshape import add_reshape
from ryzenai_onnx_utils.transform.transpose import add_transpose


def is_supported_pattern(extractor, subgraph):
    (
        trans0,
        rms0,
        trans1,
        rms1,
        concat0,
        rms2,
        rms3,
        concat1,
        trans2,
        mul0,
        mul1,
        matmul0,
    ) = subgraph
    trans0_permute = onnx.helper.get_node_attr_value(trans0, "perm")
    trans1_permute = onnx.helper.get_node_attr_value(trans1, "perm")
    trans2_permute = onnx.helper.get_node_attr_value(trans2, "perm")
    if trans0_permute != trans1_permute:
        return False
    if trans0_permute != [0, 2, 1, 3]:
        return False
    if trans2_permute != [0, 2, 3, 1]:
        return False
    rms0_axis = onnx.helper.get_node_attr_value(rms0, "axis")
    rms1_axis = onnx.helper.get_node_attr_value(rms1, "axis")
    rms2_axis = onnx.helper.get_node_attr_value(rms2, "axis")
    rms3_axis = onnx.helper.get_node_attr_value(rms3, "axis")
    if rms0_axis != -1 or rms1_axis != -1 or rms2_axis != -1 or rms3_axis != -1:
        return False
    if len(concat0.input) != 2 or len(concat1.input) != 2:
        return False
    concat0_input_shape = ryzenai_onnx_utils.matcher.get_shape(concat0.input[0], extractor)
    concat1_input_shape = ryzenai_onnx_utils.matcher.get_shape(concat1.input[0], extractor)
    if len(concat0_input_shape) != 4 or len(concat1_input_shape) != 4:
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
    (
        trans0,
        rms0,
        trans1,
        rms1,
        concat0,
        rms2,
        rms3,
        concat1,
        trans2,
        mul0,
        mul1,
        matmul0,
    ) = subgraph
    if not is_supported_pattern(extractor, subgraph):
        return subgraph, [], None
    new_nodes = []
    new_tvis = []
    new_inits = []
    matmul_inputs = []
    for branch in ["q_branch", "k_branch"]:
        branch_input_nodes = [trans0, trans1] if branch == "q_branch" else [rms2, rms3]
        branch_output_nodes = mul0 if branch == "q_branch" else mul1
        rms = [rms0, rms1] if branch == "q_branch" else [rms2, rms3]
        trans = trans0 if branch == "q_branch" else trans2
        concat = concat0 if branch == "q_branch" else concat1
        mul = mul0 if branch == "q_branch" else mul1
        concat_inputs = []
        concat_inputs_shape = []
        concat_axis = 1
        new_reshape0_inputs_shape = []
        # create concat inputs
        for i in range(2):
            rms_output_shape = ryzenai_onnx_utils.matcher.get_shape(branch_input_nodes[i].input[0], extractor)
            rms_out_tvi = onnx.helper.make_tensor_value_info(
                rms[i].output[0] + f"_{pass_id}",
                ryzenai_onnx_utils.matcher.get_dtype(rms[i].output[0], extractor),
                rms_output_shape,
            )
            new_tvis.append(rms_out_tvi)
            new_rmsnorm = onnx.helper.make_node(
                "SimplifiedLayerNormalization",
                inputs=[branch_input_nodes[i].input[0], rms[i].input[1]],
                outputs=[rms_out_tvi.name],
                name=rms[i].name + f"_out_{pass_id}",
            )
            ryzenai_onnx_utils.matcher.copy_attributes(rms[i], new_rmsnorm)
            new_nodes.append(new_rmsnorm)
            # create reshape 4dims->3dims
            if branch == "q_branch":
                trans_permute = onnx.helper.get_node_attr_value(trans, "perm")
                concat_axis = onnx.helper.get_node_attr_value(concat, "axis")
                concat_axis = trans_permute[concat_axis]
            else:
                concat_axis = onnx.helper.get_node_attr_value(concat, "axis")
            assert concat_axis == 1
            new_reshape0_in_shape = rms_output_shape
            new_reshape0_out_shape = [
                new_reshape0_in_shape[0],
                new_reshape0_in_shape[1],
                new_reshape0_in_shape[2] * new_reshape0_in_shape[3],
            ]
            new_reshape0_inputs_shape.append(new_reshape0_in_shape)
            new_reshape0, new_reshape0_tvis, new_reshape0_init = add_reshape(
                input_name=new_rmsnorm.output[0],
                shape_name=new_rmsnorm.output[0] + "_shape_" + pass_id,
                output_name=new_rmsnorm.output[0] + "_reshape_" + pass_id,
                dtype=ryzenai_onnx_utils.matcher.get_dtype(rms[i].output[0], extractor),
                in_shape=new_reshape0_in_shape,
                out_shape=new_reshape0_out_shape,
            )
            new_nodes.append(new_reshape0)
            new_tvis.extend(new_reshape0_tvis)
            new_inits.append(new_reshape0_init)
            concat_inputs.append(new_reshape0.output[0])
            concat_inputs_shape.append(new_reshape0_out_shape)
        # create concat
        concat_out_shape = concat_inputs_shape[0]
        if isinstance(concat_inputs_shape[0][concat_axis], str) or isinstance(concat_inputs_shape[1][concat_axis], str):
            concat_out_shape[concat_axis] = (
                f"{concat_inputs_shape[0][concat_axis]} + {concat_inputs_shape[1][concat_axis]}"
            )
        else:
            concat_out_shape[concat_axis] = concat_inputs_shape[0][concat_axis] + concat_inputs_shape[1][concat_axis]
        concat_out_tvi = onnx.helper.make_tensor_value_info(
            concat.output[0] + f"_{pass_id}",
            ryzenai_onnx_utils.matcher.get_dtype(concat.output[0], extractor),
            concat_out_shape,
        )
        new_tvis.append(concat_out_tvi)
        new_concat = onnx.helper.make_node(
            "Concat",
            inputs=concat_inputs,
            outputs=[concat_out_tvi.name],
            name=concat.name + f"_{pass_id}",
        )
        ryzenai_onnx_utils.matcher.set_attribute(new_concat, "axis", concat_axis)
        new_nodes.append(new_concat)
        # create reshape 3dims->4dims
        new_reshape1_in_shape = concat_out_shape
        new_reshape1_out_shape = list(new_reshape0_inputs_shape[0])
        new_reshape1_out_shape[concat_axis] = concat_out_shape[concat_axis]
        new_reshape1, new_reshape1_tvis, new_reshape1_init = add_reshape(
            input_name=new_concat.output[0],
            shape_name=new_concat.output[0] + "_shape_" + pass_id,
            output_name=new_concat.output[0] + "_reshape_" + pass_id,
            dtype=ryzenai_onnx_utils.matcher.get_dtype(concat.output[0], extractor),
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
            node_name=trans0.name + "_" + pass_id,
            input_name=new_reshape1.output[0],
            output_name=branch_output_nodes.output[0],
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
            ryzenai_onnx_utils.matcher.get_dtype(mul0.output[0], extractor),
            ryzenai_onnx_utils.matcher.get_shape(mul0.output[0], extractor),
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
    new_matmul0 = onnx.helper.make_node(
        "MatMul",
        inputs=matmul_inputs,
        outputs=matmul0.output,
        name=matmul0.name + f"_{pass_id}",
    )
    new_nodes.append(new_matmul0)

    return new_nodes, new_inits, new_tvis


PATTERN = [
    "Transpose([?], t0)",
    "SimplifiedLayerNormalization([t0, ?], s0)",
    "Transpose([?], t1)",
    "SimplifiedLayerNormalization([t1, ?], s1)",
    "Concat([s0, s1], c0)",
    "SimplifiedLayerNormalization([?, ?], s2)",
    "SimplifiedLayerNormalization([?, ?], s3)",
    "Concat([s2, s3], c1)",
    "Transpose([c1], t2)",
    "Mul([c0, ?], m0)",
    "Mul([t2, ?], m1)",
    "MatMul([m0, m1], m2)",
]
REPLACEMENT = replacement
