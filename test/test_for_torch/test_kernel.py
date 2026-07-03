#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import math
import re
import shutil
import subprocess
import time
from importlib import import_module
from pathlib import Path

import pytest
import torch
from torch.utils.cpp_extension import _get_build_directory
from torch_testing_utils import run_torch_op_variants  # type: ignore[import-not-found]

from quark.common.utils.import_utils import is_triton_available
from quark.common.utils.testing_utils import (
    assert_outputs_equivalent,
    require_linux,
    require_torch_cuda,
    require_torch_hip,
    set_environment_variables,
    torch_device,
)
from quark.torch.export.nn.modules.realquantizer import DynamicScaledQuantizer, StaticScaledRealQuantizer
from quark.torch.kernel import mx as mx_kernel
from quark.torch.kernel.hw_emulation import extensions
from quark.torch.kernel.hw_emulation.extensions import compile_kernel
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

    # Test 1: Without PYTORCH_ROCM_ARCH (single arch)
    with set_environment_variables(PYTORCH_ROCM_ARCH=None):
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

    # Test 2: With multiple architectures
    with set_environment_variables(PYTORCH_ROCM_ARCH="gfx90a,gfx942,gfx906,gfx950"):
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

    assert len(offload_archs_multi - offload_archs_single) == 3
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

    ref = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
    for i in range(16):
        inp[:, i] = i

    for i in range(16):
        inp[:, 16 + i] = i << 4

    def pipeline() -> torch.Tensor:
        out_local = torch.zeros(num_tokens, hidden_size, dtype=torch.float16, device=device)
        extensions.kernel_ext.dq_uint8_mxfp4_to_half(inp, scales, out_local, 32)
        return out_local

    out = run_torch_op_variants(pipeline)

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


def _qdq_mxfp4_inplace_via_runner(inp_seed: torch.Tensor, group_size: int) -> torch.Tensor:
    """Run ``kernel_ext.qdq_mxfp4_`` on a fresh clone of ``inp_seed`` through the
    dual-variant runner; returns the (bit-exact) stable result.
    """

    def pipeline() -> torch.Tensor:
        a = inp_seed.clone()
        extensions.kernel_ext.qdq_mxfp4_(a, group_size)
        return a

    return run_torch_op_variants(pipeline)


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
    inp = _qdq_mxfp4_inplace_via_runner(inp, 32)

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
    inp = _qdq_mxfp4_inplace_via_runner(inp, 32)

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
            inp_kernel = _qdq_mxfp4_inplace_via_runner(inp_kernel, 32)
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


@pytest.mark.parametrize("shape", [(11008, 512), (256, 4194304)])
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
    scale_dtype: str, float_dtype: torch.dtype, scalings: list[int], kernel: str, device: str, shape: tuple[int, int]
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

    hidden_size = shape[1]

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
        assert torch.equal(w_qdq, out)
    elif kernel == "triton":
        if scale_dtype == "float":
            # not supported
            return

        if hidden_size > 65536:
            with pytest.raises(ValueError, match=r"is larger than the supported grid dimension"):
                out = mx_kernel.dq_mxfp4_triton(w_mxfp4, scale, float_dtype)
        else:
            out = mx_kernel.dq_mxfp4_triton(w_mxfp4, scale, float_dtype)

            assert torch.equal(w_qdq, out)


@require_torch_cuda
@require_torch_hip
@require_linux
def test_aiter_mxfp4_asm_requested_path_matches_aiter_reference():
    try:
        aiter = import_module("aiter")
        fp4_utils = import_module("aiter.utility.fp4_utils")
        fp4_linear = import_module("quark.torch.quantization.nn.modules.aiter_fp4_inference_linear")
    except (ImportError, AttributeError, RuntimeError) as exc:
        pytest.skip(f"Aiter MXFP4 ASM kernels are not available: {exc}")
    if fp4_linear._get_triton_quant is None or fp4_linear._shuffle_weight is None:
        pytest.skip("Aiter MXFP4 ASM shuffled weight path is not available")

    torch.manual_seed(123)
    device = "cuda"
    dtype = torch.bfloat16
    m, n, k = 64, 1280, 8192

    x = torch.randn(m, k, device=device, dtype=dtype) * 0.5
    weight = torch.randn(n, k, device=device, dtype=dtype) * 0.5
    quant_func = aiter.get_triton_quant(aiter.QuantType.per_1x32)
    x_q, _ = quant_func(x, shuffle=True)
    weight_q, _ = quant_func(weight, shuffle=True)
    _, x_scale = quant_func(x, shuffle=False)
    _, weight_scale = quant_func(weight, shuffle=False)

    weight_q_shuffle, weight_scale_shuffle = fp4_linear._pack_weight_asm(weight)
    actual = fp4_linear._gemm_with_dynamic_quant(
        x=x,
        weight=weight_q_shuffle.view(torch.uint8),
        weight_scale=weight_scale_shuffle.view(torch.uint8),
        use_asm_gemm=True,
        out_dtype=dtype,
    )

    x_f32 = fp4_utils.mxfp4_to_f32(x_q)
    weight_f32 = fp4_utils.mxfp4_to_f32(weight_q)
    x_scale_f32 = fp4_utils.e8m0_to_f32(x_scale.view(torch.uint8)[:m].repeat_interleave(32, dim=1))
    weight_scale_f32 = fp4_utils.e8m0_to_f32(weight_scale.view(torch.uint8)[:n].repeat_interleave(32, dim=1))
    expected = ((x_f32 * x_scale_f32) @ (weight_f32 * weight_scale_f32).T).to(dtype)

    assert actual.shape == (m, n)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-1)


@require_torch_cuda
@require_torch_hip
@require_linux
def test_aiter_mxfp4_asm_requested_small_k_fallback_matches_dequant_reference():
    try:
        import_module("aiter")
        fp4_linear = import_module("quark.torch.quantization.nn.modules.aiter_fp4_inference_linear")
    except (ImportError, AttributeError, RuntimeError) as exc:
        pytest.skip(f"Aiter MXFP4 ASM kernels are not available: {exc}")
    if fp4_linear._per_1x32_f4_quant_hip is None:
        pytest.skip("Aiter MXFP4 ASM quantization path is not available")

    torch.manual_seed(123)
    device = "cuda"
    dtype = torch.bfloat16
    m, n, k = 64, 128, 64

    x = torch.randn(m, k, device=device, dtype=dtype) * 0.5
    weight = torch.randn(n, k, device=device, dtype=dtype) * 0.5
    weight_q, weight_scale = fp4_linear._pack_weight_asm(weight)

    actual = fp4_linear._gemm_with_dynamic_quant(
        x=x,
        weight=weight_q.view(torch.uint8),
        weight_scale=weight_scale.view(torch.uint8),
        use_asm_gemm=True,
        out_dtype=dtype,
    )

    x_q, x_scale = fp4_linear._per_1x32_f4_quant_hip(x, shuffle=False)
    x_dq = mx_kernel.dq_mxfp4_hip(x_q.view(torch.uint8), x_scale.view(torch.uint8), dtype)
    weight_dq = mx_kernel.dq_mxfp4_hip(weight_q.view(torch.uint8), weight_scale.view(torch.uint8), dtype)
    expected = x_dq @ weight_dq.T

    assert actual.shape == (m, n)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-1)


# Op coverage routed through ``run_torch_op_variants``. Each pipeline below
# is a single-op driver against ``extensions.kernel_ext``; the runner
# currently exercises just that legacy surface and degrades equivalence
# checking to a no-op until the upcoming stable-ABI migration lands, at
# which point it picks up bit-exact dual-variant assertions against
# ``torch.ops.quark_hw_emulation`` automatically.


def _variant_randn(shape, device="cpu", dtype=torch.float32, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(shape, generator=g, dtype=torch.float32).to(dtype=dtype, device=device)


_FQ_RANGES = [
    (-128, 127),  # int8
    (0, 255),  # uint8
    (-8, 7),  # int4
    (-32768, 32767),  # int16
]
_FQ_ROUND_MODES = [2, 3, 8]  # floor-half, round-half-away, banker's


@require_torch_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("shape", [(16,), (4, 16), (2, 3, 32)])
@pytest.mark.parametrize("qmin,qmax", _FQ_RANGES)
@pytest.mark.parametrize("zp", [0, 5, -3])
@pytest.mark.parametrize("scale", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("round_mode", _FQ_ROUND_MODES)
def test_fake_quantize_per_tensor_affine_variants(dtype, shape, qmin, qmax, zp, scale, round_mode):
    x = _variant_randn(shape, device="cuda", dtype=dtype)
    scale_t = torch.tensor([scale], device="cuda", dtype=dtype)
    zp_t = torch.tensor([zp], device="cuda", dtype=torch.int32)

    def pipeline() -> torch.Tensor:
        return extensions.kernel_ext.fake_quantize_per_tensor_affine(x.clone(), scale_t, zp_t, qmin, qmax, round_mode)

    run_torch_op_variants(pipeline)


# (ebits, mbits, max_norm) for the FP4/FP6/FP8 low-precision FP formats.
_LOW_PREC_FP_FORMATS = [
    (2, 1, 6.0),  # fp4_e2m1
    (2, 3, 7.5),  # fp6_e2m3
    (3, 2, 28.0),  # fp6_e3m2
    (4, 3, 448.0),  # fp8_e4m3
    (5, 2, 57344.0),  # fp8_e5m2
]


@pytest.mark.parametrize("fmt", _LOW_PREC_FP_FORMATS, ids=lambda f: f"e{f[0]}m{f[1]}")
@pytest.mark.parametrize("round_mode", [2, 3, 8])
@pytest.mark.parametrize("shape", [(16,), (4, 16), (2, 3, 16)])
def test_fake_quantize_to_low_precision_fp_variants(fmt, round_mode, shape):
    ebits, mbits, max_norm = fmt
    x = _variant_randn(shape, device=torch_device, dtype=torch.float32)

    def pipeline() -> torch.Tensor:
        return extensions.kernel_ext.fake_quantize_to_low_precision_fp(x.clone(), ebits, mbits, max_norm, round_mode)

    run_torch_op_variants(pipeline)


@pytest.mark.parametrize("seed", [0, 1, 2, 42, 12345])
def test_fake_quantize_to_low_precision_fp_seed_variety(seed):
    x = _variant_randn((4, 16), seed=seed)

    def pipeline() -> torch.Tensor:
        return extensions.kernel_ext.fake_quantize_to_low_precision_fp(x.clone(), 2, 1, 6.0, 2)

    run_torch_op_variants(pipeline)


@pytest.mark.parametrize(
    "name,tensor",
    [
        ("zeros", torch.zeros(4, 16)),
        ("ones", torch.ones(4, 16)),
        ("neg_ones", -torch.ones(4, 16)),
        ("saturating", torch.full((4, 16), 1e6)),
        ("tiny", torch.full((4, 16), 1e-6)),
    ],
)
def test_fake_quantize_to_low_precision_fp_edge_cases(name, tensor):
    def pipeline() -> torch.Tensor:
        return extensions.kernel_ext.fake_quantize_to_low_precision_fp(tensor.clone(), 4, 3, 448.0, 2)

    run_torch_op_variants(pipeline)


# ---------------------------------------------------------------------------
# tqt_backward (CPU + CUDA; shared kernel -> bit-exact).
# ---------------------------------------------------------------------------

_TQT_CONFIGS = [
    pytest.param({"scale": 0.1, "qmin": -128.0, "qmax": 127.0}, id="int8_in_range"),
    pytest.param({"scale": 0.1, "qmin": -8.0, "qmax": 7.0}, id="int4_saturating"),
    pytest.param({"scale": 30.0, "qmin": -128.0, "qmax": 127.0}, id="loose_in_range_only"),
]


def _tqt_make_inputs(config, shape, device, seed):
    scale_t = torch.tensor([config["scale"]], device=device, dtype=torch.float32)
    qmax_t = torch.tensor([config["qmax"]], device=device, dtype=torch.float32)
    qmin_t = torch.tensor([config["qmin"]], device=device, dtype=torch.float32)
    logt_t = torch.tensor([0.5], device=device, dtype=torch.float32)
    x = _variant_randn(shape, device=device, seed=seed)
    grad_out = _variant_randn(shape, device=device, seed=seed + 100)
    return x, grad_out, scale_t, qmax_t, qmin_t, logt_t


def _tqt_make_negative_tie_inputs(device):
    scale_val = 0.5
    scaled = torch.tensor(
        [-0.5, -1.5, -2.5, -3.5, -4.5, -5.5, -6.5, -7.5, -8.5, -9.5, -10.5, -11.5, -12.5, -13.5, -14.5, -15.5],
        dtype=torch.float32,
        device=device,
    )
    x = scaled * scale_val
    grad_out = torch.full_like(x, 0.25)
    scale_t = torch.tensor([scale_val], device=device, dtype=torch.float32)
    qmax_t = torch.tensor([127.0], device=device, dtype=torch.float32)
    qmin_t = torch.tensor([-128.0], device=device, dtype=torch.float32)
    logt_t = torch.tensor([0.5], device=device, dtype=torch.float32)
    return x, grad_out, scale_t, qmax_t, qmin_t, logt_t


def _tqt_backward_cpu_reference(x, scale, quant_max, quant_min, logt, grad_output):
    scaled_x = x / scale
    rounded_scaled_x = torch.where(
        (scaled_x < 0) & (scaled_x - torch.floor(scaled_x) == 0.5),
        torch.ceil(scaled_x),
        torch.round(scaled_x),
    )
    is_lt_min = rounded_scaled_x < quant_min
    is_gt_max = rounded_scaled_x > quant_max
    is_ge_min_and_le_max = ~is_lt_min & ~is_gt_max
    grad_logt = grad_output * scale * math.log(2)
    grad_logt = torch.where(is_ge_min_and_le_max, grad_logt * (rounded_scaled_x - scaled_x), grad_logt)
    grad_logt = torch.where(is_lt_min, grad_logt * quant_min, grad_logt)
    grad_logt = torch.where(is_gt_max, grad_logt * quant_max, grad_logt)
    grad_logt = grad_logt.sum().expand_as(logt)
    grad_x = grad_output.clone()
    grad_x = torch.where(is_ge_min_and_le_max, grad_x, 0 * grad_x)
    return grad_x, grad_logt


@pytest.mark.parametrize("shape", [(16,), (4, 16), (2, 3, 16)])
@pytest.mark.parametrize("seed", [0, 1, 42])
@pytest.mark.parametrize("config", _TQT_CONFIGS)
def test_tqt_backward_variants(shape, seed, config):
    x, grad_out, scale, quant_max, quant_min, logt = _tqt_make_inputs(config, shape, torch_device, seed)

    def pipeline():
        return extensions.kernel_ext.tqt_backward(
            x.clone(), scale.clone(), quant_max.clone(), quant_min.clone(), logt.clone(), grad_out.clone()
        )

    run_torch_op_variants(pipeline)


def test_tqt_backward_negative_tie_inputs():
    x, grad_out, scale, quant_max, quant_min, logt = _tqt_make_negative_tie_inputs(torch_device)

    def pipeline():
        return extensions.kernel_ext.tqt_backward(
            x.clone(), scale.clone(), quant_max.clone(), quant_min.clone(), logt.clone(), grad_out.clone()
        )

    run_torch_op_variants(pipeline)


_TQT_REF_CONFIGS = [
    pytest.param(
        config.values[0],
        lambda device, r=config.values[0]: _tqt_make_inputs(r, (4, 16), device, seed=0),
        id=config.id,
    )
    for config in _TQT_CONFIGS
] + [
    pytest.param(
        {"scale": 0.5, "qmin": -128.0, "qmax": 127.0},
        _tqt_make_negative_tie_inputs,
        id="negative_ties",
    ),
]


@pytest.mark.parametrize("config,build_inputs", _TQT_REF_CONFIGS)
def test_tqt_backward_cpu_matches_pure_pytorch_reference(config, build_inputs):
    x, grad_out, scale, quant_max, quant_min, logt = build_inputs("cpu")

    def pipeline():
        return extensions.kernel_ext.tqt_backward(
            x.clone(), scale.clone(), quant_max.clone(), quant_min.clone(), logt.clone(), grad_out.clone()
        )

    actual_grad_x, actual_grad_logt = run_torch_op_variants(pipeline)
    expected_grad_x, expected_grad_logt = _tqt_backward_cpu_reference(x, scale, quant_max, quant_min, logt, grad_out)

    assert_outputs_equivalent(actual_grad_x, expected_grad_x, atol=0, rtol=0, ctx=f"tqt_backward CPU grad_x [{config}]")
    assert_outputs_equivalent(
        actual_grad_logt, expected_grad_logt, atol=0, rtol=0, ctx=f"tqt_backward CPU grad_logt [{config}]"
    )


# MXFP4 op-variant coverage; reference-value correctness lives in
# ``test_mxfp4_dequant`` / ``test_mxfp4_fused_qdq`` above.
@require_torch_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(64,), (4, 32), (2, 3, 64)])
@pytest.mark.parametrize("group_size", [32])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_qdq_mxfp4_variants(dtype, shape, group_size, seed):
    if shape[-1] % group_size != 0:
        pytest.skip(f"last dim {shape[-1]} not divisible by group_size {group_size}")
    x = _variant_randn(shape, device="cuda", dtype=dtype, seed=seed)

    def pipeline() -> torch.Tensor:
        return extensions.kernel_ext.qdq_mxfp4(x.clone(), group_size)

    run_torch_op_variants(pipeline)


@require_torch_cuda
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("shape", [(64,), (4, 32), (2, 3, 64)])
@pytest.mark.parametrize("group_size", [32])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_qdq_mxfp4_inplace_variants(dtype, shape, group_size, seed):
    if shape[-1] % group_size != 0:
        pytest.skip(f"last dim {shape[-1]} not divisible by group_size {group_size}")
    base = _variant_randn(shape, device="cuda", dtype=dtype, seed=seed)

    def pipeline() -> torch.Tensor:
        # ``qdq_mxfp4_`` mutates its input; clone per variant run.
        a = base.clone()
        extensions.kernel_ext.qdq_mxfp4_(a, group_size)
        return a

    run_torch_op_variants(pipeline)


@require_torch_cuda
@pytest.mark.parametrize("n_groups", [16, 64])
@pytest.mark.parametrize("group_size", [32])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dq_uint8_mxfp4_to_half_variants(n_groups, group_size, seed):
    # Input packing: 2 fp4 values per uint8 byte.
    numel = n_groups * group_size
    g = torch.Generator(device="cpu").manual_seed(seed)
    inp = torch.randint(0, 256, (numel // 2,), generator=g, dtype=torch.uint8).cuda()
    scales = torch.randint(0, 128, (n_groups,), generator=g, dtype=torch.uint8).cuda()

    def pipeline() -> torch.Tensor:
        out = torch.empty(numel, dtype=torch.float16, device="cuda")
        extensions.kernel_ext.dq_uint8_mxfp4_to_half(inp, scales, out, group_size)
        return out

    run_torch_op_variants(pipeline)


# Property tests routed through both variants.
@require_torch_cuda
def test_fake_quantize_per_tensor_affine_preserves_metadata():
    x = torch.randn(4, 16, device="cuda")
    scale = torch.tensor([0.1], device="cuda")
    zp = torch.tensor([0], device="cuda", dtype=torch.int32)

    def pipeline() -> torch.Tensor:
        return extensions.kernel_ext.fake_quantize_per_tensor_affine(x.clone(), scale, zp, -128, 127, 2)

    out = run_torch_op_variants(pipeline)
    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device


def test_fake_quantize_to_low_precision_fp_preserves_metadata():
    x = torch.randn(4, 16)

    def pipeline() -> torch.Tensor:
        return extensions.kernel_ext.fake_quantize_to_low_precision_fp(x.clone(), 4, 3, 448.0, 2)

    out = run_torch_op_variants(pipeline)
    assert out.shape == x.shape and out.dtype == x.dtype and out.device == x.device


def test_fake_quantize_fp_does_not_mutate_input():
    x = torch.randn(4, 16)
    x_copy = x.clone()

    def pipeline() -> torch.Tensor:
        return extensions.kernel_ext.fake_quantize_to_low_precision_fp(x, 4, 3, 448.0, 2)

    run_torch_op_variants(pipeline)
    assert_outputs_equivalent(x, x_copy, atol=0, rtol=0, ctx="fake_quantize_to_low_precision_fp mutated input")


# Spec tests for ``fake_quantize_per_tensor_affine``. Each asserts against
# values that don't depend on a separate Python implementation of the op:
#   - tie-breaking expectations are hand-computed integer outputs
#   - special-value expectations are pinned to the input itself (finite
#     inputs pass through unchanged; ±0 / NaN / Inf must not crash)
#   - idempotency is a property of the op, not a comparison to a golden
@require_torch_cuda
@pytest.mark.parametrize(
    "round_mode, expected",
    [
        # Inputs are ``[0.5, -0.5, 1.5, -1.5, 2.5]`` with scale=1, zp=0.
        (2, [1.0, 0.0, 2.0, -1.0, 3.0]),  # floor(x + 0.5): ties to +inf
        (3, [1.0, -1.0, 2.0, -2.0, 3.0]),  # sign(x) * floor(|x| + 0.5): ties away from zero
        (8, [0.0, 0.0, 2.0, -2.0, 2.0]),  # nearbyint: ties to even (banker's)
    ],
    ids=["mode2_ties_toward_pos_inf", "mode3_ties_away_from_zero", "mode8_ties_to_even"],
)
def test_fake_quantize_per_tensor_affine_tie_breaking(round_mode, expected):
    x = torch.tensor([0.5, -0.5, 1.5, -1.5, 2.5], dtype=torch.float32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    zp = torch.tensor([0], dtype=torch.int32, device="cuda")

    def pipeline() -> torch.Tensor:
        return extensions.kernel_ext.fake_quantize_per_tensor_affine(x.clone(), scale, zp, -128, 127, round_mode)

    out = run_torch_op_variants(pipeline)
    # Outputs are exact integers in float32, so ``==`` is safe here.
    assert out.cpu().tolist() == expected


@require_torch_cuda
@pytest.mark.parametrize(
    "x_values, expected_finite",
    [
        ([0.0, -0.0], {0: 0.0, 1: 0.0}),
        ([1.0, float("nan"), 3.0], {0: 1.0, 2: 3.0}),
        ([float("inf"), float("-inf"), 0.0], {2: 0.0}),
    ],
    ids=["positive_and_negative_zero", "nan_propagation", "inf_does_not_crash"],
)
def test_fake_quantize_per_tensor_affine_special_values(x_values, expected_finite):
    """Kernel must not crash on ±0 / NaN / Inf; finite inputs are preserved."""
    x = torch.tensor(x_values, dtype=torch.float32, device="cuda")
    scale = torch.tensor([1.0], dtype=torch.float32, device="cuda")
    zp = torch.tensor([0], dtype=torch.int32, device="cuda")

    def pipeline() -> torch.Tensor:
        return extensions.kernel_ext.fake_quantize_per_tensor_affine(x.clone(), scale, zp, -128, 127, 8)

    out = run_torch_op_variants(pipeline).cpu()
    assert out.shape == x.shape
    for idx, expected in expected_finite.items():
        assert out[idx].item() == expected


@require_torch_cuda
@pytest.mark.parametrize(
    "round_mode, scale_val, atol, rtol, x_factory",
    [
        pytest.param(
            8,
            0.1,
            1e-6,
            1e-5,
            lambda: torch.tensor([0.0, 0.1, -0.1, 0.5, -0.5, 1.0, -1.0], dtype=torch.float32, device="cuda"),
            id="float32_nearbyint",
        ),
        pytest.param(
            3,
            0.05,
            1e-3,
            1e-3,
            lambda: torch.randn(64, dtype=torch.float16, device="cuda"),
            id="float16_round",
        ),
    ],
)
def test_fake_quantize_per_tensor_affine_idempotent(round_mode, scale_val, atol, rtol, x_factory):
    """Q(Q(x)) == Q(x). Tolerances absorb float roundtrip on values like 0.1."""
    x = x_factory()
    scale = torch.tensor([scale_val], dtype=x.dtype, device=x.device)
    zp = torch.tensor([0], dtype=torch.int32, device=x.device)

    def pipeline():
        first = extensions.kernel_ext.fake_quantize_per_tensor_affine(x, scale, zp, -128, 127, round_mode)
        second = extensions.kernel_ext.fake_quantize_per_tensor_affine(first, scale, zp, -128, 127, round_mode)
        return first, second

    first, second = run_torch_op_variants(pipeline)
    assert_outputs_equivalent(second, first, atol=atol, rtol=rtol, ctx="fake_quantize_per_tensor_affine not idempotent")
