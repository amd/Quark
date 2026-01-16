#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import _get_build_directory

from quark.shares.utils.import_utils import is_triton_available
from quark.shares.utils.testing_utils import require_linux, require_torch_cuda, require_torch_hip
from quark.torch.export.nn.modules.realquantizer import DynamicScaledQuantizer, StaticScaledRealQuantizer
from quark.torch.kernel import mx as mx_kernel
from quark.torch.kernel.hw_emulation.extensions import compile_kernel, kernel_ext
from quark.torch.quantization.config.config import FP4PerGroupSpec


def detect_architecture_from_binary(binary_path: str):
    try:
        result = subprocess.run(
            "/opt/rocm/lib/llvm/bin/llvm-objdump --full-contents " + binary_path + " | grep gfx",
            shell=True,
            capture_output=True,
            text=True,
        )
        # Match e.g. `gfx942` from `gfx942.amdhsa`.
        return set(re.findall(r"gfx\d+[a-zA-Z]*(?=\.+)", result.stdout.strip()))
    except Exception as e:
        print(f"Error processing {binary_path}: {e}")
        return set()


@require_torch_cuda
@require_torch_hip
@require_linux
def test_compile_kernel_rocm():
    is_cuda_runtime = 0
    extra_cuda_cflags = ["-DIS_CUDA_RUNTIME=" + str(is_cuda_runtime)]
    extra_cflags = ["-DIS_CUDA_RUNTIME=" + str(is_cuda_runtime)]
    extra_cuda_cflags.extend(["-O2"])

    os.environ.pop("PYTORCH_ROCM_ARCH", None)
    kernel_dir = "test_kernel_ext_singlearch"
    compile_dir = Path(_get_build_directory(kernel_dir, False))
    if compile_dir.exists() and compile_dir.is_dir():
        shutil.rmtree(compile_dir, ignore_errors=True)

    start = time.time()
    compile_kernel(kernel_dir, None, extra_cuda_cflags, extra_cflags)
    single_arch_compile_time = time.time() - start

    offload_archs_single = set()
    with open(Path(compile_dir, "build.ninja")) as file:
        logs = file.read().rstrip()
        offload_archs_single = set(re.findall(r"gfx\d+[a-zA-Z]*(?=[,\s])", logs))

    os.environ["PYTORCH_ROCM_ARCH"] = "gfx90a,gfx942,gfx906"
    kernel_dir = "test_kernel_ext_multiarch"
    compile_dir = Path(_get_build_directory(kernel_dir, False))
    if compile_dir.exists() and compile_dir.is_dir():
        shutil.rmtree(compile_dir, ignore_errors=True)

    start = time.time()
    compile_kernel(kernel_dir, None, extra_cuda_cflags, extra_cflags)
    multi_arch_compile_time = time.time() - start

    offload_archs_multi = set()
    with open(Path(compile_dir, "build.ninja")) as file:
        logs = file.read().rstrip()
        offload_archs_multi = set(re.findall(r"gfx\d+[a-zA-Z]*(?=[,\s])", logs))

    assert len(offload_archs_multi - offload_archs_single) == 2
    assert single_arch_compile_time < 0.6 * multi_arch_compile_time


@pytest.mark.parametrize("scale", [1.0, 2.0, 0.5])
@pytest.mark.parametrize(
    "device",
    [
        "cuda:0",
        pytest.param(
            "cuda:1",
            marks=[
                pytest.mark.require_dual_gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires CUDA multi-gpu"),
            ],
        ),
    ],
)
def test_mxfp4_dequant(scale: float, device: str):
    hidden_size = 512
    num_tokens = 1

    inp = torch.zeros(num_tokens, hidden_size // 2, dtype=torch.uint8, device=device)

    scales = torch.ones(num_tokens, hidden_size // 32, dtype=torch.float16, device=device) * scale

    scales[:, 1] = scales[:, 1] * 4

    out = torch.zeros(num_tokens, hidden_size, dtype=torch.float16, device=device)

    ref = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    for i in range(16):
        inp[:, i] = i

    for i in range(16):
        inp[:, 16 + i] = i << 4

    kernel_ext.dq_uint8_mxfp4_to_half(inp, scales, out, 32)

    for i in range(16):
        assert out[:, 2 * i] == ref[i] * scale

    for i in range(16):
        assert out[:, 32 + 2 * i + 1] == ref[i] * scale * 4


def round_ref(x):
    if x < -5.0:
        return -6.0
    elif x >= -5.0 and x <= -3.5:
        return -4.0
    elif x > -3.5 and x < -2.5:
        return -3.0
    elif x >= -2.5 and x <= -1.75:
        return -2.0
    elif x > -1.75 and x < -1.25:
        return -1.5
    elif x >= -1.25 and x <= -0.75:
        return -1.0
    elif x > -0.75 and x < -0.25:
        return -0.5
    elif x >= -0.25 and x < 0.0:
        return -0.0
    elif x >= 0.0 and x <= 0.25:
        return 0.0
    elif x > 0.25 and x < 0.75:
        return 0.5
    elif x >= 0.75 and x <= 1.25:
        return 1.0
    elif x > 1.25 and x < 1.75:
        return 1.5
    elif x >= 1.75 and x <= 2.5:
        return 2.0
    elif x > 2.5 and x < 3.5:
        return 3.0
    elif x >= 3.5 and x <= 5.0:
        return 4.0
    elif x > 5.0:
        return 6.0


def ref_mxfp4_qdq(x, scale):
    return scale * round_ref(x / scale)


@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
def test_mxfp4_fused_qdq(float_dtype: torch.dtype):
    hidden_size = 128
    num_tokens = 1

    inp = torch.rand(num_tokens, hidden_size, dtype=float_dtype, device="cuda") - 0.5

    # Force scale to be 1.
    for i in range(128 // 32):
        inp[0, 32 * i] = 6.2
    inp = torch.clamp(inp, -6.5, 6.5)

    inp_clone = inp.clone()

    kernel_ext.qdq_mxfp4_(inp, 32)

    for i, val in enumerate(inp[0]):
        assert ref_mxfp4_qdq(inp_clone[0, i].item(), 2**0) == val.item()

    # Force scale to be [2**2, 2**3, 2**(-1), 2**(-2)].
    inp = torch.rand(num_tokens, hidden_size, dtype=float_dtype, device="cuda") - 0.5

    inp[:, :32] = (torch.rand(32) - 0.5) * 2 * 17.4
    inp[:, 12] = 17.4

    inp[:, 32:64] = (torch.rand(32) - 0.5) * 2 * 34.8
    inp[:, 40] = -34.8

    inp[:, 64:96] = (torch.rand(32) - 0.5) * 2 * 3.2
    inp[:, 40] = 3.2

    inp[:, 96:] = (torch.rand(32) - 0.5) * 2 * 1.2
    inp[:, 40] = -1.2

    inp_clone = inp.clone()
    kernel_ext.qdq_mxfp4_(inp, 32)

    for i, val in enumerate(inp[0, :32]):
        assert ref_mxfp4_qdq(inp_clone[0, i].item(), 2**2) == val.item()

    for i, val in enumerate(inp[0, 32:64]):
        assert ref_mxfp4_qdq(inp_clone[0, 32 + i].item(), 2**3) == val.item()

    for i, val in enumerate(inp[0, 64:96]):
        assert ref_mxfp4_qdq(inp_clone[0, 64 + i].item(), 2 ** (-1)) == val.item()

    for i, val in enumerate(inp[0, 96:]):
        assert ref_mxfp4_qdq(inp_clone[0, 96 + i].item(), 2 ** (-2)) == val.item()


@pytest.mark.parametrize("hidden_size", [64 * 32, 2880, 128 * 7])
@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scalings", [[2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]])
@pytest.mark.parametrize("inplace", [True, False])
@pytest.mark.parametrize(
    "kernel",
    [
        "hip",
        pytest.param(
            "triton",
            marks=pytest.mark.skipif(not is_triton_available(), reason="Triton is not installed."),
        ),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cuda:0",
        pytest.param(
            "cuda:1",
            marks=[
                pytest.mark.require_dual_gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires CUDA multi-gpu"),
            ],
        ),
    ],
)
def test_mxfp4_fused_qdq_match_quark(
    float_dtype: torch.dtype, scalings: list[int], inplace: bool, kernel: str, device: str, hidden_size: int
):
    torch.manual_seed(0)
    qspec = FP4PerGroupSpec(
        ch_axis=-1,
        group_size=32,
        scale_format="e8m0",
        scale_calculation_mode="even",
        is_dynamic=True,
    ).to_quantization_spec()

    quantizer = DynamicScaledQuantizer(
        qspec=qspec,
        float_dtype=float_dtype,
        device=device,
    )

    inp = (torch.rand(1, hidden_size, dtype=float_dtype, device=device) - 0.5) * 2
    for i in range(hidden_size // 32):
        inp[:, i * 32 : (i + 1) * 32] = inp[:, i * 32 : (i + 1) * 32] * scalings[i % len(scalings)]

    inp_qdq_ref = quantizer(inp)

    inp_kernel = inp.clone()

    if kernel == "hip":
        if inplace:
            kernel_ext.qdq_mxfp4_(inp_kernel, 32)
        else:
            inp_kernel_clone = inp_kernel.clone()
            inp_kernel_clone2 = inp_kernel.clone()

            inp_kernel = mx_kernel.qdq_mxfp4_hip(inp_kernel_clone, "even")

            assert torch.equal(inp_kernel_clone, inp_kernel_clone2)

    elif kernel == "triton":
        if inplace:
            # not supported
            return

        inp_kernel = mx_kernel.qdq_mxfp4_triton(inp_kernel, "even")

    for i in range(hidden_size // 32):
        assert torch.all(torch.isfinite(inp_qdq_ref[:, i * 32 : (i + 1) * 32]))
        assert torch.all(torch.isfinite(inp_kernel[:, i * 32 : (i + 1) * 32]))

        if kernel == "triton":
            # NOTE: Triton kernel does slight different rounding during float32 -> float4 casting.
            atol = 5
            rtol = 1e-2
        else:
            atol = None
            rtol = None
        torch.testing.assert_close(
            inp_qdq_ref[:, i * 32 : (i + 1) * 32], inp_kernel[:, i * 32 : (i + 1) * 32], atol=atol, rtol=rtol
        )


@pytest.mark.parametrize("scale_dtype", ["uint8", "float"])
@pytest.mark.parametrize("float_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("scalings", [[2.3, 0.03, 7.3, 0.1, 0.004, 17.3, 1e4, 1e-4]])
@pytest.mark.parametrize(
    "kernel",
    [
        "hip",
        pytest.param(
            "triton",
            marks=pytest.mark.skipif(not is_triton_available(), reason="Triton is not installed."),
        ),
    ],
)
@pytest.mark.parametrize(
    "device",
    [
        "cuda:0",
        pytest.param(
            "cuda:1",
            marks=[
                pytest.mark.require_dual_gpu,
                pytest.mark.skipif(torch.cuda.device_count() < 2, reason="test requires CUDA multi-gpu"),
            ],
        ),
    ],
)
def test_mxfp4_dequant_kernel_match_quark(
    scale_dtype: str, float_dtype: torch.dtype, scalings: list[int], kernel: str, device: str
):
    qspec = FP4PerGroupSpec(
        ch_axis=-1,
        group_size=32,
        scale_format="e8m0",
        scale_calculation_mode="even",
        is_dynamic=False,
    ).to_quantization_spec()

    weight_quantizer = StaticScaledRealQuantizer(
        qspec=qspec,
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=float_dtype,
        device=device,
    )

    observer = qspec.observer_cls(qspec, device=device)

    hidden_size = 512
    shape = (11008, hidden_size)

    w = (torch.rand(shape, device=device, dtype=float_dtype) - 0.5) * 2

    # Make it so that different groups have different scales.
    for i in range(hidden_size // 32):
        w[:, i * 32 : (i + 1) * 32] = w[:, i * 32 : (i + 1) * 32] * scalings[i % len(scalings)]

    observer(w)
    scale, _ = observer._calculate_qparams()
    weight_quantizer.scale = scale

    w_mxfp4 = weight_quantizer.to_real_quantize_params(w).to(device)
    weight_quantizer.maybe_convert_and_transpose_scale()

    if scale_dtype == "float":
        scale = scale.to(float_dtype)
    else:
        scale = weight_quantizer.scale
    w_qdq = weight_quantizer(w_mxfp4).to(float_dtype)

    out = torch.zeros(shape, device=device, dtype=float_dtype)
    if kernel == "hip":
        out = mx_kernel.dq_mxfp4_hip(w_mxfp4, scale, float_dtype)
    elif kernel == "triton":
        if scale_dtype == "float":
            # not supported
            return
        out = mx_kernel.dq_mxfp4_triton(w_mxfp4, scale, float_dtype)

    assert torch.equal(w_qdq, out)
