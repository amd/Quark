# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import numpy.typing as npt
import onnx
from onnx import numpy_helper

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def extract_int4_tensor(tensor_proto: onnx.TensorProto) -> npt.NDArray[np.int8]:
    """Extract INT4 tensor data from ONNX tensor_proto."""
    if tensor_proto.data_type != onnx.TensorProto.INT4:
        return numpy_helper.to_array(tensor_proto)

    shape = tensor_proto.dims
    raw_data = tensor_proto.raw_data
    if not shape or not raw_data:
        raise ValueError("Tensor has no shape information or raw data")

    num_elements = np.prod(shape)
    byte_array = np.frombuffer(raw_data, dtype=np.uint8)
    packed_elements = min(len(raw_data) * 2, num_elements)
    byte_array = byte_array[: (packed_elements + 1) // 2]

    # Unpack bytes into INT4
    unpacked = np.zeros(packed_elements, dtype=np.int8)
    unpacked[0::2] = byte_array & 0x0F  # Lower nibble
    odd_indices = np.arange(1, packed_elements, 2)
    if len(odd_indices) > 0:
        unpacked[odd_indices] = (byte_array[: len(odd_indices)] >> 4) & 0x0F

    # Convert [0..15] to [-8..7]
    return np.where(unpacked > 7, unpacked - 16, unpacked).reshape(shape)


def pack_int4_to_uint8(int4_array: npt.NDArray[np.int8]) -> npt.NDArray[np.uint8]:
    """Pack INT4 values (range -8..7) into uint8 array."""
    shifted = (int4_array + 8).astype(np.uint8)  # Shift to [0..15]
    length = shifted.size
    packed_length = (length + 1) // 2
    packed = np.zeros(packed_length, dtype=np.uint8)

    for i in range(0, length, 2):
        low = shifted[i] & 0xF
        high = shifted[i + 1] & 0xF if i + 1 < length else 0
        packed[i // 2] = (high << 4) | low

    return packed


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    """
    Replacement function for converting DequantizeLinear and MatMul to MatMulNBits.
    This pass may be slow, as it involves weight packing/unpacking/transpose
    """

    # Configure parameters
    block_size = 128
    bits = 4
    ms_domain = "com.microsoft"
    accuracy_level = 0

    # Extract nodes
    dq_node = subgraph[0]
    matmul_node = subgraph[1]

    # Verify DQ output feeds MatMul input[1] (weights)
    if matmul_node.input[1] != dq_node.output[0]:
        return subgraph, [], None

    # Check if DQ has multiple consumers
    dq_consumers = ryzenai_onnx_utils.matcher.find_nodes_by_input(dq_node.output[0], extractor.graph)
    if len(dq_consumers) != 1:
        return subgraph, [], None

    # Extract inputs from DequantizeLinear
    if len(dq_node.input) != 3:
        return subgraph, [], None

    q_data_name, scale_name, zpoint_name = dq_node.input[:3]

    # Get initializers
    q_data_init = ryzenai_onnx_utils.matcher.get_initializer(q_data_name, extractor)
    scale_arr = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(scale_name, extractor)
    zpoint_init = ryzenai_onnx_utils.matcher.get_initializer(zpoint_name, extractor)

    if q_data_init is None or scale_arr is None or zpoint_init is None:
        return subgraph, [], None

    # Verify and extract INT4 weights
    if q_data_init.data_type != zpoint_init.data_type != onnx.TensorProto.INT4:
        return subgraph, [], None

    weights_arr = extract_int4_tensor(q_data_init)
    weight_shape = q_data_init.dims

    if len(weight_shape) != 2:
        return subgraph, [], None

    # Weights expected to come in shape of [K, N]
    K, N = int(weight_shape[0]), int(weight_shape[1])

    # Get zero point arrays
    zpoint_arr = extract_int4_tensor(zpoint_init)

    # Calculate block dimensions
    n_blocks_per_col = (K + block_size - 1) // block_size
    bytes_per_block = (block_size + 1) // 2  # INT4 packing, 2 values per byte

    # Duplcate scales (by n_blocks_per_col) over each column
    scale_flat = np.zeros(N * n_blocks_per_col, dtype=scale_arr.dtype)
    if len(scale_arr.shape) == 1 and scale_arr.shape[0] == N:
        for n in range(N):
            scale_flat[n * n_blocks_per_col : (n + 1) * n_blocks_per_col] = scale_arr[n]
    else:
        scale_flat = scale_arr.flatten()

    # Create scale initializer
    new_scale_name = f"flat_scale_{pass_id}"
    new_scale = numpy_helper.from_array(scale_flat, name=new_scale_name)

    # Process zero points
    zp_flat = np.zeros(N * n_blocks_per_col, dtype=zpoint_arr.dtype)
    if len(zpoint_arr.shape) == 1 and zpoint_arr.shape[0] == N:
        for n in range(N):
            zp_flat[n * n_blocks_per_col : (n + 1) * n_blocks_per_col] = zpoint_arr[n]
    else:
        zp_flat = zpoint_arr.flatten()

    # Create zero point initializer
    zp_safe = np.clip(zp_flat, -8, 7).astype(np.int8)
    packed_zp = pack_int4_to_uint8(zp_safe)
    packed_zp_name = f"packed_zp_{pass_id}"
    new_zp = numpy_helper.from_array(packed_zp, name=packed_zp_name)

    # TRANSPOSE FIX: Transpose weights from [K, N] to [N, K] before packing
    weights_arr = weights_arr.transpose(1, 0)  # Now shaped [N, K]

    # Pack weights
    packed_weights_flat = pack_int4_to_uint8(weights_arr.flatten())

    # Ensure proper shape
    total_packed_size = N * n_blocks_per_col * bytes_per_block
    if len(packed_weights_flat) < total_packed_size:
        padded = np.zeros(total_packed_size, dtype=np.uint8)
        padded[: len(packed_weights_flat)] = packed_weights_flat
        packed_weights_flat = padded

    # Reshape and create weight initializer
    packed_weights_reshaped = packed_weights_flat.reshape(N, n_blocks_per_col, bytes_per_block)
    packed_weights_name = f"packed_weights_{pass_id}"
    new_weights = numpy_helper.from_array(packed_weights_reshaped, name=packed_weights_name)

    # Create MatmulNBits node
    new_node = onnx.helper.make_node(
        op_type="MatMulNBits",
        inputs=[
            matmul_node.input[0],
            packed_weights_name,
            new_scale_name,
            packed_zp_name,
        ],
        outputs=matmul_node.output,
        # TODO(varunsh): using the pass_id at the end as other MatMulNBits creates
        # an issue with JIT layer ID inference. This needs to be corrected
        name=f"{pass_id}_matmulnbits_{matmul_node.name}",
        domain=ms_domain,
    )

    # Add attributes
    new_node.attribute.extend(
        [
            onnx.helper.make_attribute("K", K),
            onnx.helper.make_attribute("N", N),
            onnx.helper.make_attribute("bits", bits),
            onnx.helper.make_attribute("block_size", block_size),
            onnx.helper.make_attribute("accuracy_level", accuracy_level),
        ]
    )

    return [new_node], [new_weights, new_scale, new_zp], None


PATTERN = ["DequantizeLinear(?, a0)", "MatMul([?,a0], ?)"]
REPLACEMENT = replacement
