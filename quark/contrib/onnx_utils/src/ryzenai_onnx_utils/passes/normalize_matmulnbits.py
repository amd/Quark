# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass normalizes MatMulNBits by adding in any optional inputs that the
downstream passes/custom ops expect. In particular, it adds the optional qzeros
argument if it's missing and checks that there's no existing group ID or bias
already.
"""

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def create_empty_bias_tensor(matmul: onnx.NodeProto, extractor: onnx.utils.Extractor) -> onnx.TensorProto:
    n = onnx.helper.get_node_attr_value(matmul, "N")
    dtype = ryzenai_onnx_utils.matcher.get_dtype(matmul.input[0], extractor)
    empty_bias = np.zeros([n], onnx.helper.tensor_dtype_to_np_dtype(dtype))
    empty_bias_tensor = onnx.numpy_helper.from_array(
        empty_bias,
        ryzenai_onnx_utils.matcher.input_name_from_node_name(matmul.name, "bias"),
    )
    return empty_bias_tensor


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    matmul = subgraph[0]
    input_num = len(matmul.input)
    assert input_num <= 6, f"MatMulNBits {matmul.name} has more than 6 inputs"

    # inputs for matmulnbits
    # A: input tensor, unquantized, T1
    # B: weight quantized packed tensor, T2
    # scales: quantization scale, T1
    # zero points: optional, present with AWQ
    # group idx: optional
    # bias: optional

    # this is the desired case: all inputs present
    if input_num == 6:
        return subgraph, [], None

    new_inputs = matmul.input[:3]
    new_initializers = []

    # here, matmulnbits has weights and scales but no zero points
    # current kernels assume zero point is present, so just create one with zeros
    # this does increase memory, but will at least be functional
    if input_num <= 3:
        # If input zero_points is stored as uint8_t it has the same
        # packing method as input B. - [N * CeilDiv(n_blocks_per_col * bits, 8)]
        # e.g. for N = 4608, K = 4096, block_size = 128, bits = 4 this gives it size of
        # n_blocks_per_col = (K + block_size - 1) / block_size = 32
        # 4608 * CeilDiv(32 * 4, 8) = 4608 * 16 = 73728

        n = onnx.helper.get_node_attr_value(matmul, "N")
        k = onnx.helper.get_node_attr_value(matmul, "K")
        bits = onnx.helper.get_node_attr_value(matmul, "bits")
        block_size = onnx.helper.get_node_attr_value(matmul, "block_size")

        n_blocks_per_col = (k + block_size - 1) // block_size
        zero_points_blob_size = n * ((n_blocks_per_col * bits + 7) // 8)
        # assuming uint8 zero points and 4 bit packing.
        # Use default value of 2**(bits-1) = 8 -> 0x88 packed
        zero_points_np = np.full([zero_points_blob_size], 0x88, np.uint8)

        zero_points_tensor_name = ryzenai_onnx_utils.matcher.input_name_from_node_name(matmul.name, "qzeros")
        new_inputs.append(zero_points_tensor_name)
        new_initializers.append(
            onnx.helper.make_tensor(
                zero_points_tensor_name,
                onnx.TensorProto.UINT8,
                zero_points_np.shape,
                zero_points_np,
            )
        )
    else:
        new_inputs.append(matmul.input[3])

    # here, matmulnbits has zero points but no group_idx. The custom op implementation
    # does not currently support group_idx downstream so this will be removed if
    # the op is changed to the custom domain. For now, just add a placeholder.
    if input_num <= 4:
        new_inputs.append("")

    new_nodes = []

    new_outputs = matmul.output
    if input_num <= 5:
        if len(subgraph) > 1:
            # if there's an add after, take its value as the bias
            add = subgraph[1]
            if ryzenai_onnx_utils.matcher.is_initializer(add.input[0], extractor):
                new_inputs.append(add.input[0])
                new_outputs = add.output
            elif ryzenai_onnx_utils.matcher.is_initializer(add.input[1], extractor):
                new_inputs.append(add.input[1])
                new_outputs = add.output
            else:
                # if both inputs are not initializers, we can't fuse the add
                empty_bias_tensor = create_empty_bias_tensor(matmul, extractor)
                new_inputs.append(empty_bias_tensor.name)
                new_initializers.append(empty_bias_tensor)
                new_nodes.append(add)
        else:
            empty_bias_tensor = create_empty_bias_tensor(matmul, extractor)
            new_inputs.append(empty_bias_tensor.name)
            new_initializers.append(empty_bias_tensor)

    new_node_name = matmul.name
    # Find indices of "layers" and "MatMulNBits" in the node name
    layers_index = matmul.name.find("layers")
    matmulnbits_index = matmul.name.find("MatMul")
    if layers_index != -1 and matmulnbits_index != -1:
        new_node_name = new_node_name[layers_index : matmulnbits_index - 1].replace("/", ".")

    new_node = onnx.helper.make_node(
        "MatMulNBits",
        inputs=new_inputs,
        outputs=new_outputs,
        name=new_node_name,
        domain=matmul.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(matmul, new_node)
    new_nodes.append(new_node)

    return new_nodes, new_initializers, None


PATTERN = [
    ["MatMulNBits([?,?,?,?], [a0])", "Add([a0,?], ?)"],
    ["MatMulNBits([?,?,?,?], [a0])", "Add([?,a0], ?)"],
    ["MatMulNBits([?,?,?], ?)"],
]
REPLACEMENT = [replacement] * len(PATTERN)
