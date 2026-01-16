# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.transform.reshape import add_reshape
from ryzenai_onnx_utils.transform.transpose import add_transpose
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_3d_instantcenorm(shape, reshape, extractor):
    return len(ryzenai_onnx_utils.matcher.get_shape(reshape.input[0], extractor)) == 3


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    shape, reshape1, instancenorm, reshape2, mul, add, transpose = subgraph
    if not is_3d_instantcenorm(shape, reshape1, extractor):
        return subgraph, [], None

    new_nodes = []
    tvis = []
    initializers: list[onnx.TensorProto] = []
    reshape0 = ryzenai_onnx_utils.matcher.find_nodes_by_output(shape.input[0], extractor.graph)[0]
    shape_4d = list(ryzenai_onnx_utils.matcher.get_shape(reshape0.input[0], extractor))
    shape_4d[0] = ryzenai_onnx_utils.matcher.get_shape(reshape0.output[0], extractor)[0]

    reshape1.input[0] = reshape0.input[0]
    new_nodes.append(reshape1)

    new_shape_tvi = onnx.helper.make_tensor_value_info(shape.output[0] + "_4d", onnx.TensorProto.INT64, [4])
    tvis.append(new_shape_tvi)
    shape.input[0] = reshape0.input[0]
    shape.output[0] = new_shape_tvi.name

    new_nodes.append(shape)
    new_nodes.append(instancenorm)
    reshape2_dtype = ryzenai_onnx_utils.matcher.get_dtype(reshape2.output[0], extractor)
    reshape2.input[1] = new_shape_tvi.name
    reshape2_output_tvi = onnx.helper.make_tensor_value_info(
        reshape2.output[0] + "_4d",
        reshape2_dtype,
        shape_4d,
    )
    reshape2.output[0] = reshape2_output_tvi.name
    new_nodes.append(reshape2)
    tvis.append(reshape2_output_tvi)

    mul_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul.input[1], extractor)
    mul_init_dtype = ryzenai_onnx_utils.matcher.get_dtype(mul.input[1], extractor)
    mul_init_tvi = onnx.helper.make_tensor_value_info(
        mul.input[1] + "_4d",
        mul_init_dtype,
        mul_data.reshape(-1, 1, 1).shape,
    )
    tvis.append(mul_init_tvi)
    mul_init = onnx.helper.make_tensor(
        mul_init_tvi.name,
        mul_init_dtype,
        mul_data.reshape(-1, 1, 1).shape,
        mul_data.tobytes(),
        True,
    )
    initializers.append(mul_init)

    mul_output_tvi = onnx.helper.make_tensor_value_info(mul.output[0] + "_4d", onnx.TensorProto.FLOAT, shape_4d)
    tvis.append(mul_output_tvi)
    mul.input[0] = reshape2_output_tvi.name
    mul.input[1] = mul_init_tvi.name
    mul.output[0] = mul_output_tvi.name
    new_nodes.append(mul)

    add_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add.input[1], extractor)
    add_init_dtype = ryzenai_onnx_utils.matcher.get_dtype(add.input[1], extractor)
    add_init_tvi = onnx.helper.make_tensor_value_info(
        add.input[1] + "_4d",
        add_init_dtype,
        add_data.reshape(-1, 1, 1).shape,
    )
    tvis.append(add_init_tvi)

    add_init = onnx.helper.make_tensor(
        add_init_tvi.name,
        add_init_dtype,
        add_data.reshape(-1, 1, 1).shape,
        add_data.tobytes(),
        True,
    )
    initializers.append(add_init)
    add_out_dtype = ryzenai_onnx_utils.matcher.get_dtype(add.output[0], extractor)
    add_output_tvi = onnx.helper.make_tensor_value_info(add.output[0] + "_4d", add_out_dtype, shape_4d)
    tvis.append(add_output_tvi)
    add.input[0] = mul_output_tvi.name
    add.input[1] = add_init_tvi.name
    add.output[0] = add_output_tvi.name
    new_nodes.append(add)

    new_transpose_node, transpose_tvi = add_transpose(
        transpose.name + "4d",
        add_output_tvi.name,
        transpose.output[0] + "_4d",
        add_out_dtype,
        shape_4d,
        [shape_4d[0], shape_4d[2], shape_4d[3], shape_4d[1]],
        [0, 2, 3, 1],
    )
    new_nodes.append(new_transpose_node)
    tvis.extend(transpose_tvi)

    if isinstance(shape_4d[2], int) and isinstance(shape_4d[3], int) and isinstance(shape_4d[1], int):
        new_reshape_node, new_reshape_tvi, new_reshape_tensor = add_reshape(
            transpose_tvi[-1].name,
            reshape0.input[1] + "_nhwc",
            transpose.output[0],
            add_out_dtype,
            [shape_4d[0], shape_4d[2], shape_4d[3], shape_4d[1]],
            [-1, shape_4d[2] * shape_4d[3], shape_4d[1]],
        )
        new_nodes.append(new_reshape_node)
        tvis.extend(new_reshape_tvi)
        initializers.append(new_reshape_tensor)
    else:
        transpose_shape_tvi = onnx.helper.make_tensor_value_info(
            transpose_tvi[1].name + "_shape4d",
            onnx.TensorProto.INT64,
            [4],
        )
        tvis.append(transpose_shape_tvi)

        transpose_shape_node = onnx.helper.make_node(
            "Shape",
            inputs=[reshape0.input[0]],
            outputs=[transpose_shape_tvi.name],
        )
        new_nodes.append(transpose_shape_node)

        transpose_shape_dim_tvi = []
        gather_dim_nodes = []
        gather_indices_tensors = []

        unsqueeze_dim_nodes = []
        unsqueeze_dim_tvi = []
        unsqueeze_axes_tensors = []
        for i in range(4):
            transpose_shape_dim_tvi.append(
                onnx.helper.make_tensor_value_info(
                    transpose_shape_tvi.name + f"_dim{i}",
                    onnx.TensorProto.INT64,
                    [],
                )
            )
            gather_indices_tensors.append(
                onnx.helper.make_tensor(
                    name=transpose_shape_dim_tvi[i].name + f"_indices{i}",
                    data_type=onnx.TensorProto.INT64,
                    dims=[],
                    vals=[i],
                )
            )
            gather_dim_nodes.append(
                onnx.helper.make_node(
                    "Gather",
                    inputs=[transpose_shape_tvi.name, gather_indices_tensors[i].name],
                    outputs=[transpose_shape_dim_tvi[i].name],
                    axis=0,
                )
            )
        new_nodes.extend(gather_dim_nodes)
        tvis.extend(transpose_shape_dim_tvi)
        initializers.extend(gather_indices_tensors)

        mul_dimhw_tvi = onnx.helper.make_tensor_value_info(
            transpose_shape_tvi.name + "_hw",
            onnx.TensorProto.INT64,
            [],
        )
        tvis.append(mul_dimhw_tvi)
        mul_hw_node = onnx.helper.make_node(
            "Mul",
            inputs=[transpose_shape_dim_tvi[2].name, transpose_shape_dim_tvi[3].name],
            outputs=[mul_dimhw_tvi.name],
        )
        new_nodes.append(mul_hw_node)
        unsqueeze_hw_tvi = onnx.helper.make_tensor_value_info(
            mul_dimhw_tvi.name + "_unsqueeze",
            onnx.TensorProto.INT64,
            [1],
        )
        tvis.append(unsqueeze_hw_tvi)
        unsqueeze_hw_axes_tensor = onnx.helper.make_tensor(
            name=mul_dimhw_tvi.name + "axes",
            data_type=onnx.TensorProto.INT64,
            dims=[1],
            vals=[0],
        )
        initializers.append(unsqueeze_hw_axes_tensor)
        unsqueeze_hw_node = onnx.helper.make_node(
            "Unsqueeze",
            inputs=[mul_dimhw_tvi.name, unsqueeze_hw_axes_tensor.name],
            outputs=[unsqueeze_hw_tvi.name],
        )
        new_nodes.append(unsqueeze_hw_node)
        for i in [0, 1]:
            unsqueeze_dim_tvi.append(
                onnx.helper.make_tensor_value_info(
                    transpose_shape_dim_tvi[i].name + "_unsqueeze",
                    onnx.TensorProto.INT64,
                    [1],
                )
            )
            unsqueeze_axes_tensors.append(
                onnx.helper.make_tensor(
                    name=transpose_shape_dim_tvi[i].name + "_indices",
                    data_type=onnx.TensorProto.INT64,
                    dims=[1],
                    vals=[0],
                )
            )
            unsqueeze_dim_nodes.append(
                onnx.helper.make_node(
                    "Unsqueeze",
                    inputs=[
                        transpose_shape_dim_tvi[i].name,
                        unsqueeze_axes_tensors[i].name,
                    ],
                    outputs=[unsqueeze_dim_tvi[i].name],
                )
            )
        tvis.extend(unsqueeze_dim_tvi)
        new_nodes.extend(unsqueeze_dim_nodes)
        initializers.extend(unsqueeze_axes_tensors)

        concat_dim_tvi = onnx.helper.make_tensor_value_info(
            transpose_shape_tvi.name + "_shape3d",
            onnx.TensorProto.INT64,
            [3],
        )
        tvis.append(concat_dim_tvi)
        concat_dim_node = onnx.helper.make_node(
            "Concat",
            inputs=[
                unsqueeze_dim_tvi[0].name,
                unsqueeze_hw_tvi.name,
                unsqueeze_dim_tvi[1].name,
            ],
            outputs=[concat_dim_tvi.name],
            axis=0,
        )
        new_nodes.append(concat_dim_node)
        new_reshape_node = onnx.helper.make_node(
            "Reshape",
            name=f"{transpose_tvi[-1].name}_reshape",
            inputs=[
                transpose_tvi[-1].name,
                concat_dim_tvi.name,
            ],
            outputs=[transpose.output[0]],
        )
        new_nodes.append(new_reshape_node)
    return new_nodes, initializers, tvis


PATTERN = [
    "Shape([?],c0)",
    "Reshape([?,?],b1)",
    "InstanceNormalization([b1, ?,?], b2)",
    "Reshape([b2,c0],b3)",
    "Mul([b3,?],b4)",
    "Add([b4,?],b5)",
    "Transpose([b5], b6)",
]
REPLACEMENT = replacement
