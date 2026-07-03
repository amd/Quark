#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


import numpy as np
import pytest
import torch

import quark.torch.kernel
from quark.common.data_type import BaseFP8_E5M3
from quark.common.utils.testing_utils import local_test_only, require_torch_cuda, torch_device
from quark.torch.quantization.config.config import AmdFP4Spec, OCP_MXFP4Spec
from quark.torch.quantization.tensor_quantize import DynamicScaledFakeQuantize

E5M3_MAX = BaseFP8_E5M3.max_value


def generate_all_fp8_e5m3_values():
    """Generate all representable positive FP8_E5M3 values.

    E5M3 format: 5 exponent bits, 3 mantissa bits, no sign bit (unsigned).
    Exponent bias: 15
    """
    values = []
    EXPONENT_BIAS = 15

    # Iterate through all possible 8-bit combinations (0-255)
    for bits in range(256):
        exponent = (bits >> 3) & 0x1F  # 5 exponent bits
        mantissa = bits & 0x7  # 3 mantissa bits

        # Special case: NaN (exponent = 31 AND mantissa = 7)
        if exponent == 31 and mantissa == 7:
            continue  # Skip NaN

        # Special case: Zero
        if exponent == 0 and mantissa == 0:
            values.append(0.0)
            continue

        # Subnormal numbers (exponent = 0, mantissa != 0)
        if exponent == 0:
            value = (2 ** (-EXPONENT_BIAS + 1)) * (mantissa * 2 ** (-3))
        else:
            # Normal numbers: 2^(exponent - bias) * (1 + mantissa/8)
            value = (2 ** (exponent - EXPONENT_BIAS)) * (1 + mantissa * 2 ** (-3))

        values.append(value)

    return sorted(values)


def round_to_nearest_fp8_e5m3(tensor):
    """Round FP32 tensor to nearest representable FP8_E5M3 values.

    Args:
        tensor: Float32 tensor with positive values

    Returns:
        Float32 tensor with values rounded to nearest E5M3 representation
    """
    # Get all representable E5M3 values
    e5m3_values = generate_all_fp8_e5m3_values()
    e5m3_tensor = torch.tensor(e5m3_values, dtype=torch.float32, device=tensor.device)

    # Flatten input for processing
    original_shape = tensor.shape
    flat_tensor = tensor.flatten()

    # For each input value, find nearest E5M3 value
    result = torch.zeros_like(flat_tensor)

    for i, val in enumerate(flat_tensor):
        # Find index of nearest value using binary search
        # torch.searchsorted finds where to insert val to keep sorted order
        idx = torch.searchsorted(e5m3_tensor, val)

        # Handle edge cases
        if idx == 0:
            result[i] = e5m3_tensor[0]
        elif idx == len(e5m3_tensor):
            result[i] = e5m3_tensor[-1]
        else:
            # Round to nearest with ties to even
            lower = e5m3_tensor[idx - 1]
            upper = e5m3_tensor[idx]
            dist_lower = abs(val - lower)
            dist_upper = abs(val - upper)

            if dist_lower < dist_upper:
                # Closer to lower
                result[i] = lower
            elif dist_upper < dist_lower:
                # Closer to upper
                result[i] = upper
            else:
                # For E5M3, we need to check the actual E5M3 representation
                # Find which E5M3 bit pattern these correspond to
                lower_idx = idx - 1

                # The E5M3 bit pattern's LSB is bit 0 (mantissa LSB)
                # Check if lower or upper has even mantissa (bit 0 = 0)
                lower_mantissa_lsb = lower_idx & 1

                if lower_mantissa_lsb == 0:
                    result[i] = lower  # Lower is even
                else:
                    result[i] = upper  # Upper is even

    return result.reshape(original_shape)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp8_e5m3_quantize(dtype, device: str):
    """Test FP8 E5M3 fake quantization."""
    if torch.cuda.device_count() == 0 and device == "cuda":
        pytest.skip("cuda not available")

    # Initialize tensor in range [0, 114688] (E5M3 max)
    test_tensor = torch.rand(32, 512, dtype=torch.float32) * E5M3_MAX

    if dtype == torch.float32:
        test_tensor_subnormal = torch.rand(32, 512, dtype=torch.float32) * 2.0**-16

        test_tensor = torch.cat((test_tensor, test_tensor_subnormal), dim=0)

    # Convert to target dtype and move to device
    test_tensor = test_tensor.to(dtype=dtype, device=device)

    # Get reference: round to nearest E5M3 values (convert to fp32 for reference)
    reference = round_to_nearest_fp8_e5m3(test_tensor.to(torch.float32)).to(dtype)

    result = quark.torch.kernel.non_scaled_fake_quantize(
        test_tensor,
        "fp8_e5m3",
        "",  # mx_element_dtype
        -1,  # ch_axis
        1,  # group_size=1, i.e. elementwise fake quantization.
        "",  # scale_calculation_mode
    )

    assert result.shape == test_tensor.shape
    assert result.dtype == dtype
    assert torch.equal(result, reference)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp8_e5m3_real_quantize(device: str):
    """Test FP8 E5M3 real quantization produces correct uint8 bit patterns.

    This test verifies that the E5M3 bit patterns produced by scaled_real_quantize
    correctly dequantize back to valid E5M3 representable values.
    """
    if torch.cuda.device_count() == 0 and device == "cuda":
        pytest.skip("cuda not available")

    test_tensor_orig = torch.rand(32, 256, dtype=torch.float32) * E5M3_MAX
    test_tensor_orig = test_tensor_orig.to(device=device)

    # Call scaled_real_quantize to get the uint8 representation
    result_uint8 = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
        "fp8_e5m3",
        test_tensor_orig,
        None,  # scale
        None,  # zero_point
        -1,  # ch_axis
        -1,  # group_size
        0.0,  # quant_min
        E5M3_MAX,  # quant_max
        2,  # round_method (round)
        "per_tensor",  # qscheme
    )

    # Verify output is uint8
    assert result_uint8.dtype == torch.uint8, f"Expected uint8, got {result_uint8.dtype}"
    assert result_uint8.shape == test_tensor_orig.shape

    # Dequantize the E5M3 bits back to FP32
    dequantized = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        "fp8_e5m3",
        result_uint8,
        None,  # scale
        None,  # zero_point
        -1,  # ch_axis
        -1,  # group_size
        "per_tensor",  # qscheme
    )

    # For each value, verify the bit pattern produces a valid E5M3 value
    result_cpu = result_uint8.cpu()
    dequant_cpu = dequantized.cpu()

    for i in range(result_cpu.numel()):
        e5m3_bits = result_cpu.flatten()[i].item()
        dequant_val = dequant_cpu.flatten()[i].item()

        # Extract exponent and mantissa from E5M3 uint8
        exp_e5m3 = (e5m3_bits >> 3) & 0x1F  # 5 bits
        mantissa_e5m3 = e5m3_bits & 0x7  # 3 bits

        # Compute the expected FP32 value from E5M3 bits
        if exp_e5m3 == 0 and mantissa_e5m3 == 0:
            # Zero
            expected_val = 0.0
        elif exp_e5m3 == 0:
            # Subnormal
            expected_val = (2 ** (-15 + 1)) * (mantissa_e5m3 * 2 ** (-3))
        elif exp_e5m3 == 31 and mantissa_e5m3 == 7:
            # NaN - skip
            continue
        else:
            # Normal
            expected_val = (2 ** (exp_e5m3 - 15)) * (1 + mantissa_e5m3 * 2 ** (-3))

        # Verify dequantized value matches what the bits represent
        assert dequant_val == expected_val


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp8_e5m3_quantize_dequantize_match_fake_quantize(device: str):
    """Test that real_quantize + dequantize matches fake_quantize for E5M3."""
    if torch.cuda.device_count() == 0 and device == "cuda":
        pytest.skip("cuda not available")

    test_tensor = torch.rand(32, 512, dtype=torch.float32) * E5M3_MAX
    test_tensor = test_tensor.to(device=device)

    # Get result from fake quantize
    fake_quantized = quark.torch.kernel.non_scaled_fake_quantize(
        test_tensor,
        "fp8_e5m3",
        "",  # mx_element_dtype
        -1,  # ch_axis
        1,  # group_size=1, i.e. elementwise fake quantization.
        "",  # scale_calculation_mode
    )

    # Get result from real quantize + dequantize
    real_quantized = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
        "fp8_e5m3",
        test_tensor,
        None,  # scale
        None,  # zero_point
        -1,  # ch_axis
        -1,  # group_size
        0.0,  # quant_min
        E5M3_MAX,  # quant_max
        2,  # round_method (round)
        "per_tensor",  # qscheme
    )

    dequantized = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        "fp8_e5m3",
        real_quantized,
        None,  # scale
        None,  # zero_point
        -1,  # ch_axis
        -1,  # group_size
        "per_tensor",  # qscheme
    )

    assert fake_quantized.shape == dequantized.shape
    assert fake_quantized.dtype == dequantized.dtype
    assert torch.equal(fake_quantized, dequantized)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fp8_e5m3_subnormals(device: str):
    """Test E5M3 subnormal handling.

    E5M3 subnormal values have exponent=0 and mantissa != 0.

    - FP32 -> E5M3 conversion: some normal numbers become subnormal.
    - E5M3 -> FP32 conversion: some subnormal numbers become normal.
    """
    if torch.cuda.device_count() == 0 and device == "cuda":
        pytest.skip("cuda not available")

    # Test specific subnormal values
    subnormal_values = torch.tensor(
        [
            2.0**-17,  # E5M3: exp=0, mant=1 (smallest subnormal)
            2.0**-16,  # E5M3: exp=0, mant=2
            3.0 * 2.0**-17,  # E5M3: exp=0, mant=3
            2.0**-15,  # E5M3: exp=0, mant=4
            5.0 * 2.0**-17,  # E5M3: exp=0, mant=5
            6.0 * 2.0**-17,  # E5M3: exp=0, mant=6
            7.0 * 2.0**-17,  # E5M3: exp=0, mant=7 (largest subnormal)
        ],
        dtype=torch.float32,
        device=device,
    )

    # Expected E5M3 bit patterns: exp=0 (5 bits), mant=[1,2,3,4,5,6,7] (3 bits)
    # Format: EEEEE MMM = 00000 MMM
    expected_bits = torch.tensor([1, 2, 3, 4, 5, 6, 7], dtype=torch.uint8, device=device)

    # Quantize to E5M3
    result_bits = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
        "fp8_e5m3",
        subnormal_values,
        None,  # scale
        None,  # zero_point
        -1,  # ch_axis
        -1,  # group_size
        0.0,  # quant_min
        E5M3_MAX,  # quant_max
        2,  # round_method (round)
        "per_tensor",  # qscheme
    )

    assert torch.equal(result_bits, expected_bits)

    # Dequantize back and verify values
    dequantized = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        "fp8_e5m3",
        result_bits,
        None,  # scale
        None,  # zero_point
        -1,  # ch_axis
        -1,  # group_size
        "per_tensor",  # qscheme
    )

    assert torch.equal(subnormal_values, dequantized)


def do_bench(f, kwargs, num_runs, num_warmup):
    # warmup
    for _ in range(num_warmup):
        f(**kwargs)

    torch.cuda.synchronize()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_runs)]

    torch.cuda.synchronize()

    cache_size = 512 * 1024 * 1024
    cache = torch.empty(int(cache_size // 4), dtype=torch.int, device="cuda")

    for i in range(num_runs):
        cache.zero_()  # L2 eviction.

        start_events[i].record()
        f(**kwargs)
        end_events[i].record()

    torch.cuda.synchronize()

    times = [start_events[i].elapsed_time(end_events[i]) for i in range(num_runs)]

    mean = np.mean(times)
    median = np.median(times)

    return mean, median


@local_test_only
@require_torch_cuda
def test_e5m3_latency():
    # NOTE: Run this test with QUARK_COUNT_OBSERVED_SAMPLES=0 QUARK_DISABLE_COMPILE=1 for profiling.
    dtype = torch.bfloat16
    M = 16
    N = 4096
    K = 4096
    num_runs = 300
    num_warmup = 30

    # Create input and weight tensors
    input_tensor = torch.randn(M, K, dtype=dtype, device=torch_device)
    weight = torch.randn(K, N, dtype=dtype, device=torch_device)

    # Setup amdfp4 quantizer with is_dynamic=True, group_size=16
    amdfp4_spec = AmdFP4Spec(ch_axis=-1, group_size=16, is_dynamic=True)
    amdfp4_qtensor_config = amdfp4_spec.to_quantization_spec()
    amdfp4_fake_quantize = DynamicScaledFakeQuantize(amdfp4_qtensor_config, torch_device)

    # Setup OCP_MXFP4 quantizer with is_dynamic=True
    ocp_mxfp4_spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=True)
    ocp_mxfp4_qtensor_config = ocp_mxfp4_spec.to_quantization_spec()
    ocp_mxfp4_fake_quantize = DynamicScaledFakeQuantize(ocp_mxfp4_qtensor_config, torch_device)

    # Baseline: BF16 GEMM
    def bf16_gemm(x, w):
        return torch.matmul(x, w)

    # With input QDQ using amdfp4
    def bf16_gemm_with_amdfp4_qdq(x, w):
        x_qdq = amdfp4_fake_quantize(x)
        return torch.matmul(x_qdq, w)

    # With input QDQ using OCP_MXFP4
    def bf16_gemm_with_ocp_mxfp4_qdq(x, w):
        x_qdq = ocp_mxfp4_fake_quantize(x)
        return torch.matmul(x_qdq, w)

    # Benchmark BF16 GEMM
    mean_bf16, median_bf16 = do_bench(bf16_gemm, {"x": input_tensor, "w": weight}, num_runs, num_warmup)

    # Benchmark with amdfp4 input QDQ
    mean_amdfp4, median_amdfp4 = do_bench(
        bf16_gemm_with_amdfp4_qdq, {"x": input_tensor, "w": weight}, num_runs, num_warmup
    )

    # Benchmark with OCP_MXFP4 input QDQ
    mean_ocp_mxfp4, median_ocp_mxfp4 = do_bench(
        bf16_gemm_with_ocp_mxfp4_qdq, {"x": input_tensor, "w": weight}, num_runs, num_warmup
    )

    print(f"\nBF16 GEMM: mean={mean_bf16:.3f}ms, median={median_bf16:.3f}ms")
    print(f"BF16 GEMM with input QDQ (amdfp4): mean={mean_amdfp4:.3f}ms, median={median_amdfp4:.3f}ms")
    print(f"  Overhead: {((mean_amdfp4 - mean_bf16) / mean_bf16 * 100):.2f}%")
    print(f"BF16 GEMM with input QDQ (OCP_MXFP4): mean={mean_ocp_mxfp4:.3f}ms, median={median_ocp_mxfp4:.3f}ms")
    print(f"  Overhead: {((mean_ocp_mxfp4 - mean_bf16) / mean_bf16 * 100):.2f}%")
