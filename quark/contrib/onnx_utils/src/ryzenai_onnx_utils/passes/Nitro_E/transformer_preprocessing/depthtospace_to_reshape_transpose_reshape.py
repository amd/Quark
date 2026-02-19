# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import onnx
import onnx.helper as helper

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import (
    get_attribute,
    get_dtype,
    get_shape,
)
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    depth2space_node = subgraph[0]

    input_name = depth2space_node.input[0]
    output_name = depth2space_node.output[0]
    input_shape = get_shape(depth2space_node.input[0], extractor)
    input_dtype = get_dtype(depth2space_node.input[0], extractor)

    blocksize = get_attribute(depth2space_node, "blocksize")
    mode = get_attribute(depth2space_node, "mode").decode("utf-8")

    if mode != "CRD" or len(input_shape) != 4 or input_shape[1] % (blocksize * blocksize) != 0:
        return subgraph, [], None

    N, C, H, W = input_shape
    bs = blocksize
    Cout = C // (bs * bs)

    new_nodes, new_tensors, new_tvis = [], [], []

    # ---------- Reshape0 ----------
    # (N, C, H, W) -> (N, C/(bs*bs), bs, bs, H, W)
    reshape0_out_shape = [N, Cout, bs, bs, H, W]
    reshape0_name = f"{depth2space_node.name}_reshape0"
    reshape0_shape_name = f"{reshape0_name}_shape"
    reshape0_out_name = f"{reshape0_name}_out"
    reshape0 = helper.make_node(
        op_type="Reshape",
        inputs=[input_name, reshape0_shape_name],
        outputs=[reshape0_out_name],
        name=reshape0_name,
    )
    new_nodes.append(reshape0)

    reshape0_out_tvi = helper.make_tensor_value_info(reshape0_out_name, input_dtype, reshape0_out_shape)
    reshape0_shape_tvi = helper.make_tensor_value_info(
        reshape0_shape_name, onnx.TensorProto.INT64, [len(reshape0_out_shape)]
    )
    reshape0_shape_tensor = helper.make_tensor(
        reshape0_shape_name, onnx.TensorProto.INT64, [len(reshape0_out_shape)], reshape0_out_shape
    )
    new_tvis.append(reshape0_out_tvi)
    new_tvis.append(reshape0_shape_tvi)
    new_tensors.append(reshape0_shape_tensor)

    # ---------- Transpose: (0,1,4,2,5,3) ----------
    # (N, C/(bs*bs), bs, bs, H, W) -> (N, C/(bs*bs), H, bs, W, bs)
    transpose_out_shape = [N, Cout, H, bs, W, bs]
    transpose_name = f"{depth2space_node.name}_transpose"
    transpose_out_name = f"{transpose_name}_out"

    transpose = helper.make_node(
        op_type="Transpose",
        inputs=[reshape0_out_name],
        outputs=[transpose_out_name],
        perm=[0, 1, 4, 2, 5, 3],
        name=transpose_name,
    )
    new_nodes.append(transpose)

    transpose_out_tvi = helper.make_tensor_value_info(transpose_out_name, input_dtype, transpose_out_shape)
    new_tvis.append(transpose_out_tvi)

    # ---------- Reshape1 ----------
    # (N, C/(bs*bs), H, bs, W, bs) -> (N, Cout, H*bs, W*bs)
    reshape1_out_shape = [N, Cout, H * bs, W * bs]
    reshape1_name = f"{depth2space_node.name}_reshape1"
    reshape1_shape_name = f"{reshape1_name}_shape"

    reshape1 = helper.make_node(
        op_type="Reshape",
        inputs=[transpose_out_name, reshape1_shape_name],
        outputs=[output_name],
        name=reshape1_name,
    )
    new_nodes.append(reshape1)

    reshape1_shape_tvi = helper.make_tensor_value_info(
        reshape1_shape_name, onnx.TensorProto.INT64, [len(reshape1_out_shape)]
    )
    reshape1_shape_tensor = helper.make_tensor(
        reshape1_shape_name, onnx.TensorProto.INT64, [len(reshape1_out_shape)], reshape1_out_shape
    )
    new_tvis.append(reshape1_shape_tvi)
    new_tensors.append(reshape1_shape_tensor)

    return new_nodes, new_tensors, new_tvis


PATTERN = ["DepthToSpace(?,?)"]
REPLACEMENT = replacement
