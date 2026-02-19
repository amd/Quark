# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
LoRA Adapter Preprocessing Tool

This tool preprocesses LoRA (Low-Rank Adaptation) adapter files for use with ONNX models.
It converts LoRA weights from SafeTensors format to a custom binary format with BFP16
quantization and specific tensor layouts optimized for hardware acceleration.

Key Features:
- Loads LoRA adapters from SafeTensors files
- Converts weights to BFP16 format for efficient inference
- Reshapes tensors for optimized memory layouts
- Generates header and binary files for runtime loading
- Supports verification of written files
- Performs QKV fusion plus SVD-based delta matrix factorization
- Pads lower-rank matrices to target rank for consistency


Output files:
- {lora_name}.pb.bin: Header file containing tensor metadata
- {lora_name}.bin: Binary file containing quantized tensor data
"""

from __future__ import annotations

import argparse
import os
import re
from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx

# from safetensors import safe_open
import safetensors

import ryzenai_onnx_utils.proto as proto
from ryzenai_onnx_utils.strategy_builder import MladfVersion

LORA_K_SV_DIM = 32
LORA_N_SV_DIM = 64
LORA_RANK = 128


# Calculate the size of LoRA input for NPU kernel mladfmatmulbias_lora, based on K and N dimensions.
# Converted from DD C++ code: https://gitenterprise.xilinx.com/VitisAI/DynamicDispatch/blob/961a118ff2bbee974fd0986b191dacb099268b89/src/ops/mladfmatmulbias/mladfmatmulbias_utils.hpp#L174
def calculate_lora_input_size(K: int, N: int, is_flat: bool = False) -> int:
    Rank = 128
    if is_flat:
        bf16_size = 2  # sizeof(uint16_t)
        return K * Rank * bf16_size + N * Rank * bf16_size

    # Calculate the size of the lora_a.
    # Lora_a comes from a matrix of shape (K, Rank) and applied with padding, in
    # the type of BF16.
    lora_a_size = 2 * 2 * 2 * (1 + K / 128) * 4 * (Rank / 16) * 8 * 8
    a_data_band = 2  # sizeof(uint16_t);
    lora_a_data_size = lora_a_size * a_data_band

    # Calculate the size of the lora_b.
    # Lora_b comes from a matrix of shape (Rank, N) and applied with padding, in
    # the type of BFP16.
    ebs = 8
    b_data_band = 1  # sizeof(uint8_t)
    lora_b_data_size = ((Rank * N) / ebs) * (ebs + 1) * b_data_band
    return int(lora_a_data_size + lora_b_data_size)


class LoraWgt:
    def __init__(self, lora_a: npt.NDArray[Any], lora_b: npt.NDArray[Any], scaling_factor: float) -> None:
        self.lora_a = lora_a
        self.lora_b = lora_b
        self.scaling_factor = scaling_factor


def f2bfp(
    f: npt.NDArray[Any],
    bits: int = 16,
    ebs: int = 1,
    dim: int = -1,
    limit_range: bool = False,
    m2_0: bool = False,
    m2_0_rnd: bool | None = None,
) -> tuple[npt.NDArray[Any], npt.NDArray[Any]]:
    if dim < 0:
        dim += len(f.shape)
    if m2_0_rnd is None:
        m2_0_rnd = True
    shape = list(f.shape)
    shape[dim] = ebs
    shape.insert(dim, f.shape[dim] // ebs)
    fi = f.reshape(shape).getfield(np.int32)

    fis = np.sign(fi)
    fie = (fi >> 23) & 255
    fim = ((fie > 0) << 23) | (fi & ((1 << 23) - 1))

    me = np.max(fie - m2_0 * (fis * fim == (-1 << 23)), dim + 1)

    shift = np.expand_dims(me, dim + 1) - fie + 33 - bits
    bm_max = (1 << (bits - 9)) - 1
    if limit_range:
        shift = 33 - bits + np.mod(shift, bits - 4)
        me = 127 + np.mod(me - 127, bits - 4)
        fim &= bm_max << shift

    test_im = fim + (1 << np.minimum(23, np.maximum(0, shift - 1))) * (shift > 0)
    test_bm = (test_im >> np.minimum(32, shift)) * (fis if m2_0_rnd else 1)
    overflow = np.any(test_bm > bm_max, dim + 1)
    if np.any(overflow):
        shift += np.expand_dims(overflow, dim + 1)
        me += overflow
    fim += (1 << np.minimum(23, np.maximum(0, shift - 1))) * (shift > 0)
    bm = (fim >> np.minimum(32, shift)) * fis
    bfp = (bm.reshape(f.shape), me - 256 * (me >= 128))
    return bfp


def cast_bf16bfp16(input: npt.NDArray[Any]) -> Any:
    m, e = f2bfp(input.flatten().astype(np.float32), ebs=8, m2_0_rnd=True)
    m = m.reshape(-1, 8)
    e = e.reshape(-1, 1)
    output_bfp16 = np.concatenate((e, m), axis=1)

    # print(output_bfp16.dtype)
    # print(output_bfp16.shape)
    # print(output_bfp16[0:2,:])
    return output_bfp16


def aie_srs(input: npt.NDArray[Any], aie_mode: list[int]) -> Any:
    # Assign aiemode parameters into variables
    data_width = aie_mode[0]
    shift = aie_mode[1]
    mode = aie_mode[2]
    unsigned = aie_mode[3]

    # Shift
    output = input / (2 ** (shift))
    # Round
    if mode == 0:
        # simply truncate LSBs
        output = np.fix(output)
    elif mode == 2:
        # round to nearest, round half up (towards +infinity)
        output = np.round(output)
        # detect negative values at the exact half point (-0.5) and add -0.5 before rounding
        i = (input - np.fix(input)) == -0.5
        output[i] = np.round(input + 0.5)[i]
    elif mode == 6:
        # round to nearest, round half to even
        output = np.around(output)
    else:
        print("Unexpected rounding mode")

    if unsigned == 0:
        # positive clip
        output[(output >= 2 ** (data_width - 1) - 1)] = 2 ** (data_width - 1) - 1
        # negative clip
        output[(output <= -1 * 2 ** (data_width - 1))] = -1 * 2 ** (data_width - 1)
    else:
        output[(output > 2 ** (data_width) - 1)] = 2 ** (data_width) - 1
        output[(output < 0)] = 0

    return output


def np_fp32_2_bf16(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    x = x.view(np.uint32)
    x = aie_srs(x, [16, 16, 6, 1])
    x = x.astype(np.uint16) * 2**16
    x = x.view(np.float32)
    return x


def np_fp32_2_int16(x: npt.NDArray[Any]) -> npt.NDArray[Any]:
    x = x.view(np.uint32)
    x = aie_srs(x, [16, 16, 6, 1])
    x = x.astype(np.uint16)
    return x


def reshape_lora_a_v2(lora_a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Relayout lora_a: [K, R] -> [K//sv_k, 2, sv_k//8, R//8//2, 8, 8]"""
    K, R = lora_a.shape
    sv_k = 32
    K_32 = 2
    K_64 = 2

    K_package = 8
    K_pacakge_num = 2

    R_package = 8
    R_pacakge_num = 2

    lora_a = lora_a.reshape(
        K // sv_k // K_32 // K_64,  # 0
        K_32,  # 1
        K_64,  # 2
        sv_k // K_package,  # 3
        K_package,  # 4
        K_pacakge_num,  # 5
        R // R_pacakge_num // R_package,  # 6
        R_package,
    )
    lora_a = lora_a.transpose(2, 1, 5, 0, 3, 6, 4, 7)  # [2_1, 2_2, 2, K//sv_k//2_1//2_2, sv_k//8, R//8//2, k8, r8]
    s = lora_a.shape
    pad_val = np.zeros((s[0], s[1], s[2], 1, s[4], s[5], s[6], s[7]), dtype=lora_a.dtype)
    lora_a = np.concatenate([pad_val, lora_a], axis=3)
    return lora_a


def reshape_lora_b(lora_b: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """Relayout lora_b: [R, N] -> [N//sv_n, 4, sv_n//8, R//8//4, 8, 8]"""
    R, N = lora_b.shape
    sv_n = LORA_N_SV_DIM
    R_package_num = 4
    R_package = 8
    N_package = 8
    lora_b = lora_b.reshape(
        R_package_num,
        R // R_package_num // R_package,
        R_package,
        N // sv_n,
        sv_n // N_package,
        N_package,
    )
    lora_b = lora_b.transpose(3, 0, 4, 1, 5, 2)
    return lora_b


def reformat_array_lora_flat(array: npt.NDArray[Any]) -> Any:
    k, n = array.shape
    sv_k = 128
    sv_n = 32
    n_gran = 8
    if n % n_gran != 0:
        raise ValueError("n:", n, "can't be divided by n_gran:", n_gran, " when reformat array")
    # reshaped_array = array.reshape(k//sv_k, sv_k, n//sv_n, sv_n//n_gran, n_gran) #(0, 1, 2, 3, 4)
    # reformated_array = reshaped_array.transpose(2, 0, 3, 1, 4)
    reshaped_array = array.reshape(k // sv_k, sv_k // n_gran, n_gran, n // sv_n, sv_n // n_gran, n_gran)
    #                              0        1             2       3        4             5
    reformated_array = reshaped_array.transpose(3, 0, 1, 4, 2, 5)
    return reformated_array


def process_lora_flat(lora_data: npt.NDArray[Any]) -> Any:
    lora_ref = np_fp32_2_int16(reformat_array_lora_flat(lora_data))
    lowest_8_bits = (lora_ref & 0xFF).astype(np.int8)
    highest_8_bits = ((lora_ref >> 8) & 0xFF).astype(np.int8)
    lora_int8 = np.stack((lowest_8_bits, highest_8_bits), axis=-1)
    return lora_int8


def process_lora_a_v2(lora_a_data: npt.NDArray[Any]) -> Any:
    # Processing lora_a into bf16
    lora_a_re = reshape_lora_a_v2(lora_a_data)
    lora_a_int16 = np_fp32_2_int16(lora_a_re)
    lowest_8_bits = (lora_a_int16 & 0xFF).astype(np.int8)
    highest_8_bits = ((lora_a_int16 >> 8) & 0xFF).astype(np.int8)
    lora_a_int8 = np.stack((lowest_8_bits, highest_8_bits), axis=-1)
    return lora_a_int8


def process_lora_b_v2(lora_b_data: npt.NDArray[Any]) -> Any:
    # Processing lora_b into bfp16
    lora_b_re = reshape_lora_b(lora_b_data)
    lora_b_final = cast_bf16bfp16(lora_b_re).astype(np.int8)
    return lora_b_final


def test_written_file(
    header_file_name: str, bin_file_name: str, lora_data_bfp: dict[str, list[npt.NDArray[Any] | None]]
) -> None:
    """Test function to verify written files by reading back and comparing with original data"""
    print("\n=== Testing written files ===")

    header = proto.open_header(header_file_name)

    # Read weights file
    with open(bin_file_name, "rb") as f:
        weights_bytes = f.read()

    # Process each operator in header
    for op_name, op_data in header.operators.items():
        print(f"\nTesting operator: {op_name}")

        if op_name not in lora_data_bfp:
            print(f"Warning: {op_name} not found in lora_data_bfp")
            continue

        expected_tensors = lora_data_bfp[op_name]

        if len(expected_tensors) != len(op_data.data):
            print(f"Warning: Expected {len(expected_tensors)} tensors but header has {len(op_data.data)}")
            continue

        # Read tensors from weights file using header info
        for i, tensor_info in enumerate(op_data.data):
            offset = tensor_info.offset
            size = tensor_info.size
            shape = list(tensor_info.shape)
            dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_info.data_type)

            # Extract tensor bytes
            tensor_bytes = weights_bytes[offset : offset + size]

            # Convert bytes back to numpy array
            read_tensor = np.frombuffer(tensor_bytes, dtype=dtype).reshape(shape)

            expected_tensor = expected_tensors[i]
            if expected_tensor is None:
                continue  # Skip if expected tensor is None
            print(f"Tensor {i} - Expected shape: {expected_tensor.shape}, Read shape: {read_tensor.shape}")

            if np.array_equal(expected_tensor, read_tensor):
                print(f"✓ Tensor {i} matches!")
            else:
                print(f"✗ Tensor {i} does not match!")
                print(f"Max diff: {np.max(np.abs(expected_tensor.astype(float) - read_tensor.astype(float)))}")


def print_header_content(header: proto.external_data.Header) -> None:
    """Print the content of the header for debugging purposes"""
    print("\n=== Header Content ===")
    for op_name, op_data in header.operators.items():
        print(f"\nOperator: {op_name}")
        print(f"  Op Type: {op_data.op_type}")
        for i, data in enumerate(op_data.data):
            print(f"  Data {i}:")
            print(f"    Offset: {data.offset}")
            print(f"    Size: {data.size}")
            print(f"    Data Type: {onnx.helper.tensor_dtype_to_np_dtype(data.data_type)}")
            print(f"    Shape: {list(data.shape)}")


def get_operator_name_prefix(args: argparse.Namespace, base_header: proto.external_data.Header) -> list[Sequence[Any]]:
    """Get the prefix for operator names based on the base model header"""
    prefixes = []

    # Look for operators containing the target substrings
    mm_names = []
    ssmlp_names = []
    gqo_names = []
    for op_name, op_info in base_header.operators.items():
        if "MatMulNBits" in op_info.op_type:
            if "lm_head" in op_name:
                continue
            mm_names.append(op_name)
        elif "SSMLP" in op_info.op_type:
            ssmlp_names.append(op_name)
        elif "GQO" in op_info.op_type:
            gqo_names.append(op_name)

    # find the common prefix for each group
    def find_common_prefix(names: Sequence[str]) -> str:
        if not names:
            return ""
        # Start with the first name as the initial prefix
        prefix = names[0]
        # Compare with each subsequent name
        for name in names[1:]:
            # Find the common prefix between current prefix and this name
            i = 0
            while i < len(prefix) and i < len(name) and prefix[i] == name[i]:
                i += 1
            prefix = prefix[:i]
            # If no common prefix found, break early
            if not prefix:
                break
        return prefix

    # Find common prefixes for each group
    mm_prefix = find_common_prefix(mm_names) if mm_names else ""
    ssmlp_prefix = find_common_prefix(ssmlp_names) if ssmlp_names else ""
    gqo_prefix = find_common_prefix(gqo_names) if gqo_names else ""

    prefixes = [mm_prefix, ssmlp_prefix, gqo_prefix, mm_names, ssmlp_names, gqo_names]

    if args.verbose:
        print(f"MatMulNBits prefix: '{mm_prefix}'")
        print(f"ssmlp prefix: '{ssmlp_prefix}'")
        print(f"gqo prefix: '{gqo_prefix}'")

    return prefixes


def lora_process(
    args: argparse.Namespace, lora_a_data: npt.NDArray[Any], lora_b_data: npt.NDArray[Any], proj_name: str
) -> Any:
    if args.verbose:
        print(f"At {proj_name} lora data...")
    lora_a_data = np.array(lora_a_data.transpose(), dtype=np.float32)
    lora_b_data = np.array(lora_b_data.transpose(), dtype=np.float32)
    if args.verbose:
        print(f"lora_a_data.shape = {lora_a_data.shape}")
        print(f"lora_b_data.shape = {lora_b_data.shape}")
    rank = lora_a_data.shape[1]
    assert rank == lora_b_data.shape[0], f"Rank mismatch: {rank} != {lora_b_data.shape[0]}"
    lora_a_data = np_fp32_2_bf16(
        np.pad(
            lora_a_data,
            pad_width=((0, 0), (0, LORA_RANK - rank)),
            mode="constant",
            constant_values=0,
        )
    )
    lora_b_data = np_fp32_2_bf16(
        np.pad(
            lora_b_data,
            pad_width=((0, LORA_RANK - rank), (0, 0)),
            mode="constant",
            constant_values=0,
        )
    )
    if args.verbose:
        print(f"lora_a_data.shape after padding = {lora_a_data.shape}")
        print(f"lora_b_data.shape after padding = {lora_b_data.shape}")

    if args.lora_type == MladfVersion.AIE2_V2:
        lora_a_final = process_lora_a_v2(lora_a_data * args.scaling_factor)
        lora_b_final = process_lora_b_v2(lora_b_data)
    else:
        lora_a_final = process_lora_flat(lora_a_data * args.scaling_factor)
        lora_b_final = process_lora_flat(lora_b_data)

    if args.verbose:
        print(f"lora_a_final.shape = {lora_a_final.shape}, dtype = {lora_a_final.dtype}")
        print(f"lora_b_final.shape = {lora_b_final.shape}, dtype = {lora_b_final.dtype}")

    lora_a_final = lora_a_final.flatten()
    lora_b_final = lora_b_final.flatten()
    concated_lora = np.concatenate([lora_a_final, lora_b_final], axis=0)
    return concated_lora


def write_operator(
    args: argparse.Namespace,
    k: str,
    bfp_data_list: list[npt.NDArray[Any] | None],
    header: proto.external_data.Header,
    bin_file_name: str,
) -> None:
    if args.verbose:
        print(f"\nWriting operator: {k}")
    header.operators.get_or_create(k)
    header.operators[k].op_type = "LoraAdapter"

    for bfp_data in bfp_data_list:
        header.operators[k].data.add()
        header.operators[k].data[-1].offset = os.path.getsize(bin_file_name) if os.path.exists(bin_file_name) else 0
        if bfp_data is not None:
            header.operators[k].data[-1].size = bfp_data.nbytes
            header.operators[k].data[-1].data_type = onnx.helper.np_dtype_to_tensor_dtype(bfp_data.dtype)
            header.operators[k].data[-1].shape.extend(bfp_data.shape)
            with open(bin_file_name, "ab") as f:
                f.write(bfp_data.tobytes())
        else:
            header.operators[k].data[-1].size = 0
            header.operators[k].data[-1].data_type = onnx.TensorProto.INT8
            header.operators[k].data[-1].shape.extend([])
        if args.verbose and bfp_data is not None:
            print(f"Written {bfp_data.nbytes} bytes for {k}, shape: {bfp_data.shape}, dtype: {bfp_data.dtype}")


# TODO(varunsh): this should be updated to inherit parser arguments from the others somehow
def configure_parser(subparser: argparse._SubParsersAction[Any]) -> argparse.ArgumentParser:
    lora_parser: argparse.ArgumentParser = subparser.add_parser("lora-process")

    lora_parser.add_argument(
        "--lora-file",
        type=str,
        required=True,
        help="Path to lora adapter",
    )
    lora_parser.add_argument(
        "--base-model-header",
        type=str,
        required=True,
        help="Base model header file to use for lora adapter",
    )

    lora_parser.add_argument(
        "--lora-type",
        type=str,
        default=MladfVersion.FLAT,
        choices=[MladfVersion.FLAT, MladfVersion.AIE2_V2],
        help="Type of LoRA adapter to process. Options: 'flat', 'v2'",
    )

    lora_parser.add_argument(
        "--scaling-factor",
        type=float,
        help="Scaling factor of lora weights",
        default=1.0,
    )
    lora_parser.add_argument(
        "--output-dir",
        type=str,
        default="lora_bin",
        help="Directory to save output files",
    )
    lora_parser.add_argument(
        "--mode",
        type=str,
        default="prefill",
        choices=["prefill", "token"],
        help="Different modes for processing lora adapters. Options: 'prefill', 'token'",
    )
    lora_parser.add_argument("--lora-name", type=str, default="lora", help="Name of the lora adapter")
    lora_parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify the written files with original data",
    )
    lora_parser.add_argument("--verbose", action="store_true", help="Enable verbose output for debugging")

    return lora_parser


def lora(args: argparse.Namespace) -> None:
    if args.mode == "prefill":
        args.lora_type = MladfVersion.AIE2_V2
    if args.mode == "token":
        args.lora_type = MladfVersion.FLAT
    header = proto.external_data.Header()
    header_file_name: str = args.output_dir + f"/{args.lora_name}_{args.mode}.pb.bin"
    bin_file_name: str = args.output_dir + f"/{args.lora_name}_{args.mode}.bin"
    os.makedirs(args.output_dir, exist_ok=True)
    # if the output files already exist, remove them
    if os.path.exists(header_file_name):
        os.remove(header_file_name)
    if os.path.exists(bin_file_name):
        os.remove(bin_file_name)

    header.external_data.npu = True

    base_header = proto.open_header(args.base_model_header)
    mm_prefix, ssmlp_prefix, gqo_prefix, mm_names, ssmlp_names, gqo_names = get_operator_name_prefix(args, base_header)
    print(f"Reading node names from base model header: {args.base_model_header}...")
    print(
        f"-- Found {len(mm_names)} MatMulNBits operators, {len(ssmlp_names)} SSMLP operators, and {len(gqo_names)} GQO operators."
    )

    lora_raw_data: dict[str, dict[str, npt.NDArray[Any]]] = {}
    lora_data_bfp: dict[str, list[npt.NDArray[Any] | None]] = {}
    num_layers = 0
    with safetensors.safe_open(args.lora_file, framework="np", device="cpu") as f:
        for k in f.keys():  # noqa: SIM118
            layers_idx = k.find("layers")
            lora_idx = k.find("lora_")
            layer_name = k[layers_idx:lora_idx].rstrip(".") if layers_idx != -1 and lora_idx != -1 else k
            # find the number in the layer_name, and assign the largest to num_layers
            numbers = re.findall(r"\d+", layer_name)
            if numbers:
                layer_num = max(map(int, numbers))
                num_layers = max(num_layers, layer_num + 1)
            if "self_attn.q_proj" in layer_name:
                layer_name = "q_proj_lora." + str(layer_num)
            elif "self_attn.k_proj" in layer_name:
                layer_name = "k_proj_lora." + str(layer_num)
            elif "self_attn.v_proj" in layer_name:
                layer_name = "v_proj_lora." + str(layer_num)
            elif "self_attn.o_proj" in layer_name:
                layer_name = "o_proj_lora." + str(layer_num)
            elif "mlp.gate_proj" in layer_name:
                layer_name = "gate_proj_lora." + str(layer_num)
            elif "mlp.up_proj" in layer_name:
                layer_name = "up_proj_lora." + str(layer_num)
            elif "mlp.down_proj" in layer_name:
                layer_name = "down_proj_lora." + str(layer_num)
            else:
                print(f"Warning: Unknown tensor key format: {k}")
                exit()

            if layer_name not in lora_raw_data:
                lora_raw_data[layer_name] = {}

            if "lora_A" in k:
                lora_raw_data[layer_name]["a"] = f.get_tensor(k)
            elif "lora_B" in k:
                lora_raw_data[layer_name]["b"] = f.get_tensor(k)
            else:
                print(f"Warning: Unknown tensor key format: {k}")
                exit()
            if args.verbose:
                print(f"Preprocess LoRA raw data for {layer_name}: {k}")
    print(f"Reading lora data from {args.lora_file}...")
    print(f"-- Found {len(lora_raw_data)} tensors with lora data, num_layers = {num_layers}")
    written_tensors = []

    if args.mode == "token":
        for layer_name in lora_raw_data:
            if args.verbose:
                print(f"\nProcessing LoRA layer: {layer_name}")
            lora_a_data = lora_raw_data[layer_name]["a"]
            lora_b_data = lora_raw_data[layer_name]["b"]
            lora_oproj_bfp = lora_process(args, lora_a_data, lora_b_data, layer_name)
            write_operator(args, layer_name, [lora_oproj_bfp], header, bin_file_name)
            written_tensors.append(layer_name)
    else:
        for mm_name in mm_names:
            if args.verbose:
                print(f"\nProcessing MatMulNBits layer: {mm_name}")
            l_num = mm_name.split(".")[1]
            mm_type = mm_name.split(".")[-1]
            lora_name = mm_type + "_lora." + l_num
            if lora_name in lora_raw_data:
                if args.verbose:
                    print(f"-- Found LoRA data for {lora_name} in MatMulNBits layer {mm_name}...")
                lora_a_data = lora_raw_data[lora_name]["a"]
                lora_b_data = lora_raw_data[lora_name]["b"]
                lora_oproj_bfp = lora_process(args, lora_a_data, lora_b_data, lora_name)
                lora_data_bfp[mm_name] = [lora_oproj_bfp]
                write_operator(args, mm_name, lora_data_bfp[mm_name], header, bin_file_name)
                written_tensors.append(mm_name)
            else:
                if args.verbose:
                    print(f"-- No LoRA data found for {mm_name}, skipping, lora name: {lora_name}.")
        for gqo_name in gqo_names:
            if args.verbose:
                print(f"\nProcessing GQO layer: {gqo_name}")
            lora_name = "o_proj_lora." + gqo_name.split("_")[-1]
            if lora_name in lora_raw_data:
                if args.verbose:
                    print(f"-- Found LoRA data for {lora_name} in GQO layer {gqo_name}...")
                lora_a_data = lora_raw_data[lora_name]["a"]
                lora_b_data = lora_raw_data[lora_name]["b"]
                lora_bfp = lora_process(args, lora_a_data, lora_b_data, lora_name)
                lora_data_bfp[gqo_name] = [lora_bfp]
                write_operator(args, gqo_name, lora_data_bfp[gqo_name], header, bin_file_name)
                written_tensors.append(gqo_name)
            else:
                if args.verbose:
                    print(f"-- No LoRA data found for {gqo_name}, skipping.")
        for ssmlp_name in ssmlp_names:
            if args.verbose:
                print(f"\nProcessing SSMLP layer: {ssmlp_name}")
            lora_name_gate = "gate_proj_lora." + ssmlp_name.split("_")[-1]
            lora_data_bfp[ssmlp_name] = []
            if lora_name_gate in lora_raw_data:
                if args.verbose:
                    print(f"-- Found LoRA data for {lora_name_gate} gate_proj in SSMLP layer {ssmlp_name}...")
                lora_a_data = lora_raw_data[lora_name_gate]["a"]
                lora_b_data = lora_raw_data[lora_name_gate]["b"]
                lora_bfp_gate = lora_process(args, lora_a_data, lora_b_data, lora_name_gate)
                lora_data_bfp[ssmlp_name].append(lora_bfp_gate)
            else:
                lora_data_bfp[ssmlp_name].append(None)
                if args.verbose:
                    print(f"-- No LoRA data found for {ssmlp_name} gate_proj.")

            lora_name_up = "up_proj_lora." + ssmlp_name.split("_")[-1]
            if lora_name_up in lora_raw_data:
                if args.verbose:
                    print(f"-- Found LoRA data for {lora_name_up} up_proj in SSMLP layer {ssmlp_name}...")
                lora_a_data = lora_raw_data[lora_name_up]["a"]
                lora_b_data = lora_raw_data[lora_name_up]["b"]
                lora_bfp_up = lora_process(args, lora_a_data, lora_b_data, lora_name_up)
                lora_data_bfp[ssmlp_name].append(lora_bfp_up)
            else:
                lora_data_bfp[ssmlp_name].append(None)
                if args.verbose:
                    print(f"-- No LoRA data found for {lora_name_up} up_proj.")

            lora_name_down = "down_proj_lora." + ssmlp_name.split("_")[-1]
            if lora_name_down in lora_raw_data:
                if args.verbose:
                    print(f"-- Found LoRA data for {lora_name_down} down_proj in SSMLP layer {ssmlp_name}...")
                lora_a_data = lora_raw_data[lora_name_down]["a"]
                lora_b_data = lora_raw_data[lora_name_down]["b"]
                lora_bfp_down = lora_process(args, lora_a_data, lora_b_data, lora_name_down)
                lora_data_bfp[ssmlp_name].append(lora_bfp_down)
            else:
                lora_data_bfp[ssmlp_name].append(None)
                if args.verbose:
                    print(f"-- No LoRA data found for {lora_name_down} down_proj.")
            if len(lora_data_bfp[ssmlp_name]):
                write_operator(args, ssmlp_name, lora_data_bfp[ssmlp_name], header, bin_file_name)
                written_tensors.append(ssmlp_name)

    proto.save_header(header, header_file_name)
    print("Processed LoRA data successfully.")
    print(f"-- Header written to {header_file_name}")
    print(f"-- Binary data written to {bin_file_name}")

    if args.verbose:
        print_header_content(header)

    if args.verify:
        print("\nVerifying written files...")
        test_written_file(header_file_name, bin_file_name, lora_data_bfp)

    if args.verbose:
        print("\n=== Summary ===")
        print(f"Total layers processed: {num_layers}")
        print(f"Total tensors written: {len(written_tensors)}")
        written_tensors.sort()
        for tensor in written_tensors:
            print(f"- {tensor}")


def main(args: argparse.Namespace) -> None:
    lora(args)
