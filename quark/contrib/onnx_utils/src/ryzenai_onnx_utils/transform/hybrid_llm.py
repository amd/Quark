# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import hashlib
import logging
import math
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
from ryzenai_dynamic_dispatch import Attributes, matmulnbits

import ryzenai_onnx_utils
import ryzenai_onnx_utils.utils

_logger = logging.getLogger(__name__)


def _extract_arrays(
    node: onnx.NodeProto,
    start_index: int,
    extractor: onnx.utils.Extractor,
    n: int,
    bias_offset: int | None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], bool, str]:
    weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(node.input[start_index], extractor)
    scales = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(node.input[start_index + 1], extractor)
    # TODO(varunsh): should detect if present and set asymmetric if so
    zero_point = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(node.input[start_index + 2], extractor)

    if bias_offset is not None:
        bias = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(node.input[start_index + bias_offset], extractor)
    else:
        bias = np.zeros((n, 1))

    # TODO(varunsh): update
    asymmetric_quant = True

    try:
        is_bfp16 = onnx.helper.get_node_attr_value(node, "is_bfp16").decode("utf-8")
    except ValueError:
        mladf_version = "v1"
    else:
        if is_bfp16 == "weights":
            mladf_version = "v2"
        elif is_bfp16 == "flat":
            mladf_version = "flat"
        elif not is_bfp16:
            mladf_version = "v1"
        else:
            raise ValueError(f"Unsupported is_bfp16: {is_bfp16}")

    return (weight, scales, zero_point, bias, asymmetric_quant, mladf_version)


def preprocess_matmulnbits_weights(
    node: onnx.NodeProto,
    start_index: int,
    extractor: onnx.utils.Extractor,
    k: int,
    n: int,
    block_size: int,
    bias_offset: int,
    lora: bool,
) -> tuple[onnx.TensorProto, onnx.TensorProto, onnx.TensorProto, onnx.TensorProto]:
    (weight, scales, zero_point, bias, asymmetric_quant, mladf_version) = _extract_arrays(
        node, start_index, extractor, n, bias_offset
    )
    bias_enable = bias_offset > 0

    if bias_enable:
        bias_name = node.input[start_index + bias_offset]
    else:
        bias_name = ryzenai_onnx_utils.matcher.input_name_from_node_name(node.name, "bias")

    allow_weights_pad = False
    attr = Attributes()
    attr.set("K", k)
    attr.set("N", n)
    attr.set("lora", lora)
    attr.set("allow_weights_pad", allow_weights_pad)
    attr.set("mladf_version", mladf_version)
    attr.set("asymmetric_quant", asymmetric_quant)
    attr.set("bias_en", bias_enable)
    attr.set("block_size", block_size)
    attr.set("verify", True)

    try:
        new_weight, new_bias, new_scales, new_zeros = matmulnbits.matmulnbits_preformat(
            weight.astype(np.uint8),
            bias.astype(np.float32),
            scales.astype(np.float32),
            zero_point.astype(np.uint8),
            attr,
        )
    except RuntimeError as e:
        _logger.warning(f"{node.name} failed to generate NPU prepacked weights: " + str(e))
        raise

    suffix = ".preformat"

    return (
        onnx.numpy_helper.from_array(new_weight.astype(np.int8).reshape((k, n)), node.input[start_index] + suffix),
        onnx.numpy_helper.from_array(new_bias, bias_name + suffix),
        onnx.numpy_helper.from_array(new_scales, node.input[start_index + 1] + suffix),
        onnx.numpy_helper.from_array(
            new_zeros.astype(np.int8),
            node.input[start_index + 2] + suffix,
        ),
    )


def buffer_md5sum(data: bytes) -> str:
    """
    This is a Python implementation of a similar function in DynamicDispatch:
    calculate_md5sum()

    Args:
        data (bytes): Data to compute MD5 checksum for.

    Returns:
        str: MD5 checksum of the input data.
    """
    buffer_size = 1024 * 4
    md5 = hashlib.md5()
    for i in range(0, len(data), buffer_size):
        chunk = data[i : i + buffer_size]
        md5.update(chunk)
    return md5.hexdigest()


def preprocess_matmulnbits_packed_weights_interleaved(
    node: onnx.NodeProto,
    gate_start_index: int,
    up_start_index: int,
    extractor: onnx.utils.Extractor,
    k: int,
    n: int,
    block_size: int,
    bias_offset: int | None,
    lora: bool = False,
) -> tuple[onnx.TensorProto, str]:
    (weight, scales, zero_point, bias, asymmetric_quant, mladf_version) = _extract_arrays(
        node, gate_start_index, extractor, n, bias_offset
    )
    (weight2, scales2, zero_point2, bias2, asymmetric_quant, mladf_version) = _extract_arrays(
        node, up_start_index, extractor, n, bias_offset
    )
    bias_enable = bias_offset is not None
    N = n * 2

    weight = np.concatenate([weight, weight2])
    scales = np.concatenate([scales, scales2])
    zero_point = np.concatenate([zero_point, zero_point2])
    bias = np.concatenate([bias, bias2])
    mladf_version = mladf_version + "_wts_interleaved"
    allow_weights_pad = False
    attr = Attributes()
    attr.set("K", k)
    attr.set("N", N)
    attr.set("lora", lora)
    attr.set("allow_weights_pad", allow_weights_pad)
    attr.set("mladf_version", mladf_version)
    attr.set("asymmetric_quant", asymmetric_quant)
    attr.set("bias_en", bias_enable)
    attr.set("block_size", block_size)
    attr.set("wts_interleaved", True)

    try:
        packed_weight, total_bytes, real_K, real_N = matmulnbits.matmulnbits_pack_const_float32(
            weight.astype(np.uint8),
            bias.astype(np.float32),
            scales.astype(np.float32),
            zero_point.astype(np.uint8),
            attr,
        )
    except RuntimeError:
        print(f"{node.name} failed to generate NPU prepacked weights")
        raise

    packed_weight_name = node.input[gate_start_index] + ".packed"
    packed_weight_tensor = onnx.helper.make_tensor(
        packed_weight_name,
        onnx.TensorProto.INT8,
        packed_weight.shape,
        packed_weight.tobytes(),
        True,
    )
    hash_val = buffer_md5sum(packed_weight.tobytes())

    return packed_weight_tensor, hash_val


def preprocess_matmulnbits_packed_weights(
    node: onnx.NodeProto,
    start_index: int,
    extractor: onnx.utils.Extractor,
    k: int,
    n: int,
    block_size: int,
    bias_offset: int | None,
    lora: bool = False,
    allow_pad: bool = False,
    enable_ctrl_pkt: bool = False,
) -> tuple[onnx.TensorProto, str, tuple[int, int]]:
    (weight, scales, zero_point, bias, asymmetric_quant, mladf_version) = _extract_arrays(
        node, start_index, extractor, n, bias_offset
    )
    wts_k = math.prod(weight.shape[1:]) * 2

    bias_enable = bias_offset is not None
    attr = Attributes()
    attr.set("K", wts_k)
    attr.set("N", n)
    attr.set("lora", lora)
    attr.set("allow_weights_pad", allow_pad)
    attr.set("mladf_version", mladf_version)
    attr.set("asymmetric_quant", asymmetric_quant)
    attr.set("bias_en", bias_enable)
    attr.set("block_size", block_size)
    attr.set("enable_ctrl_pkt", enable_ctrl_pkt)
    if mladf_version == "flat":
        attr.set("max_m", 1)
    else:
        attr.set("max_m", 4096)

    try:
        packed_weight: npt.NDArray[Any]
        packed_weight, total_bytes, real_K, real_N = matmulnbits.matmulnbits_pack_const_float32(
            weight.astype(np.uint8),
            bias.astype(np.float32),
            scales.astype(np.float32),
            zero_point.astype(np.uint8),
            attr,
        )
    except RuntimeError as e:
        _logger.warning(f"{node.name} failed to generate NPU prepacked weights: " + str(e))
        raise

    packed_weight_name = node.input[start_index] + ".packed"
    packed_weight_bytes = packed_weight.tobytes()
    packed_weight_tensor = onnx.helper.make_tensor(
        packed_weight_name,
        onnx.TensorProto.INT8,
        packed_weight.shape,
        packed_weight_bytes,
        True,
    )
    hash_val = buffer_md5sum(packed_weight_bytes)

    return packed_weight_tensor, hash_val, (real_K, real_N)


def _extract_qmoe_arrays(
    node: onnx.NodeProto, start_index: int, bits: int, extractor: onnx.utils.Extractor
) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any], int, int, int, int, bool, str]:
    weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(node.input[start_index], extractor)
    scales = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(node.input[start_index + 1], extractor)
    bias = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(node.input[start_index + 2], extractor)

    (num_experts, n, k_packed) = weight.shape

    assert bits in (2, 4, 8), f"unexpected value for bits = {bits}"

    k = (8 // bits) * k_packed

    block_size = -1
    scales_need_reformat = False

    if scales.ndim == 2:
        _num_experts, _n = scales.shape
        # assert num_experts == _num_experts and n == _n, f"{num_experts}, {_num_experts} vs {n}, {_n}"
        assert num_experts == _num_experts, f"{num_experts}, {_num_experts}"
        scales_need_reformat = True
        block_size = 32
        assert k % block_size == 0, "Invalid default block size when trying to reformat for kernel"
    else:
        assert scales.ndim == 3, f"Expect for QMoE {node.name} scale to have dim == 3, {scales.ndim}"
        scale_k: int
        _num_experts, _n, scale_k = scales.shape
        assert k % scale_k == 0, f"Expect for QMoE {node.name} scale K dim {scale_k} to be divisor of {k}"
        block_size = k // scale_k

    if scales_need_reformat:
        # need to change from [num_experts, n] to [num_experts, n, k/block_size]
        # by repeating innermost dim
        scales = np.repeat(scales[:, :, np.newaxis], k // block_size, axis=2)

    # spec does not have zero-point for QMoE currently
    # insert our own one
    # assuming uint8 zero points and 4 bit packing.
    # Use default value of 2**(bits-1) = 8, packed value 0x88
    zp_k_packed = (k // block_size) // (8 // bits)
    zero_point = np.full((num_experts, n, zp_k_packed), 0x88, dtype=np.uint8)

    # TODO: update
    asymmetric_quant = True

    try:
        is_bfp16 = onnx.helper.get_node_attr_value(node, "is_bfp16").decode("utf-8")
    except ValueError:
        mladf_version = "v1"
    else:
        if is_bfp16 == "weights":
            mladf_version = "v2"
        elif not is_bfp16:
            mladf_version = "v1"
        else:
            raise ValueError(f"Unsupported is_bfp16: {is_bfp16}")

    return (weight, scales, zero_point, bias, k, n, num_experts, block_size, asymmetric_quant, mladf_version)


def preprocess_qmoe_packed_weights(
    node: onnx.NodeProto, extractor: onnx.utils.Extractor, bits: int, block_size_orig: int
) -> tuple[list[onnx.TensorProto], list[int], list[tuple[int, int, int, int]]]:
    lora = False

    packed_weight_tensors = []
    packed_expert_sizes: list[int] = []
    meta_info: list[tuple[int, int, int, int]] = []

    # 0: input
    # 1: router probs
    # 2: FC1 weights
    EXPERT_WEIGHT_START_INDEX = 2

    # weights, scale, bias
    # for now spec does not have zero point
    NUM_CONST_PER_EXPERT = 3

    # have FC1 and FC2 by default, optionally might have FC3
    MAX_NUM_EXPERTS = 3

    BO_ALIGNMENT_BYTES = 4096

    def pad_to_multiple(arr, n, fill_value=0):
        current_size = arr.size
        remainder = current_size % n
        if remainder == 0:
            return arr
        padding_size = n - remainder
        # pad at end of array
        return np.pad(arr, (0, padding_size), mode="constant", constant_values=fill_value)

    # process FC1, FC2 and optionally FC3
    for i in range(MAX_NUM_EXPERTS):
        try:
            (weight, scales, zero_point, bias, k, n, num_experts, block_size, asymmetric_quant, mladf_version) = (
                _extract_qmoe_arrays(node, EXPERT_WEIGHT_START_INDEX + NUM_CONST_PER_EXPERT * i, bits, extractor)
            )
        except KeyError:
            num_experts = 0

        if num_experts == 0:
            assert i == 2, "only expect FC3 to be empty"
            continue

        meta_info.append((k, n, block_size, num_experts))
        # print(f"processing layer for {i}")
        # print(f"weight = {weight.shape}")
        # print(f"scales = {scales.shape}")
        # print(f"zero_point = {zero_point.shape}")
        # print(f"bias = {bias.shape}")
        # print(f"k = {k}, n = {n}, num_experts = {num_experts}, block_size = {block_size}, asymmetric_quant = {asymmetric_quant}, mladf_version = {mladf_version}")

        packed_weights = []

        for expert_idx in range(num_experts):
            bias_enable = bool(np.any(bias[expert_idx]))
            attr = Attributes()
            attr.set("K", k)
            attr.set("N", n)
            attr.set("lora", lora)
            attr.set("mladf_version", mladf_version)
            attr.set("asymmetric_quant", asymmetric_quant)
            attr.set("bias_en", bias_enable)
            attr.set("block_size", block_size)
            try:
                packed_weight, total_bytes, real_K, real_N = matmulnbits.matmulnbits_pack_const_float32(
                    weight[expert_idx].astype(np.uint8),
                    bias[expert_idx].astype(np.float32),
                    scales[expert_idx].astype(np.float32),
                    zero_point[expert_idx].astype(np.uint8),
                    attr,
                )

                packed_weight = pad_to_multiple(packed_weight, BO_ALIGNMENT_BYTES)

                packed_weights.append(packed_weight)
            except RuntimeError as e:
                _logger.warning(
                    f"{node.name} failed to generate NPU prepacked weights for expert {expert_idx}: " + str(e)
                )
                raise

        flat_packed_weight = np.concatenate(packed_weights)
        packed_expert_size = int(np.prod(packed_weights[0].shape))
        assert packed_expert_size % BO_ALIGNMENT_BYTES == 0, (
            f"Expect {BO_ALIGNMENT_BYTES}-byte alignment for packed expert size = {packed_expert_size}"
        )

        # NOTE: this controls our granularity for pulling out weights based on initializer name
        packed_suffix = ".packed.qexperts"
        packed_weight_name = node.input[EXPERT_WEIGHT_START_INDEX + NUM_CONST_PER_EXPERT * i] + packed_suffix

        packed_weight_tensor = onnx.helper.make_tensor(
            packed_weight_name,
            onnx.TensorProto.INT8,
            flat_packed_weight.shape,
            flat_packed_weight.tobytes(),
            True,
        )

        packed_weight_tensors.append(packed_weight_tensor)
        packed_expert_sizes.append(packed_expert_size)

    return packed_weight_tensors, packed_expert_sizes, meta_info
