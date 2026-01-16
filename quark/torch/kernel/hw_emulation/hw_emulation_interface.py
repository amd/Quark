#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
from functools import partial
from typing import Any

import torch
from packaging.version import Version
from torch._inductor.decomposition import register_decomposition
from torch.library import Library, impl
from torch.types import Number

from quark.shares.utils.import_utils import TORCH_HIGHER_OR_EQUAL_2_7
from quark.shares.utils.log import ScreenLogger, log_errors
from quark.torch.kernel.hw_emulation.extensions import kernel_ext
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.utils import (
    calculate_qmin_qmax,
    even_round,
    get_dtype_params,
    reshape_to_blocks,
    t_exponent,
)
from quark.torch.utils import TRITON_GPU_SUPPORTS_FP8, assert_no_nan

# impl_abstract renamed to register_fake in PyTorch 2.4
# TODO: Remove this once we drop pytorch<=2.4 support.
if Version(torch.__version__) < Version("2.4.0"):  # pragma: no cover
    from torch.library import impl_abstract as register_fake
else:  # pragma: no cover
    from torch.library import register_fake

logger = ScreenLogger(__name__)

# NOTE: The default of PyTorch is 8 guard failures allowed with `fullgraph=True`, after that an error `torch._dynamo.exc.FailOnRecompileLimitHit: recompile_limit reached with one_graph=True.` is raised.
# In the case of `scaled_fake_quantize`, the default of 8 is too small as there are guard failures for:
# - size mismatch (`size mismatch at index 2. expected 4096, actual 12288`, `size mismatch at index 0. expected 1, actual 64`), as specialization occurs in some cases (specifically, specialization on `1` and `>1`).
# - rank mismatch (`rank mismatch. expected 2, actual 3`, `tensor 'args[1]' rank mismatch. expected 3, actual 2`), as we pass both 2D weights and 3D inputs (and potentially 4D inputs!) into `scaled_fake_quantize`.
# - running under no_grad/with gradient (e.g. if QAT is run, and later `with torch.no_grad()` is used).
# - running different quantization specs that dispatch to different path within `scaled_fake_quantize`.
# - running with different devices (`tensor 'args[1]' Tensor device index mismatch. Expected device index to be 3, actual 5`).
RECOMPILATION_LIMIT = 24 * max(1, torch.cuda.device_count())

if Version(torch.__version__) >= Version("2.7"):
    if torch._dynamo.config.recompile_limit == 8:
        torch._dynamo.config.recompile_limit = RECOMPILATION_LIMIT
    else:
        if torch._dynamo.config.recompile_limit < RECOMPILATION_LIMIT:
            logger.warning(
                f"Detected user-specified torch._dynamo.config.recompile_limit={torch._dynamo.config.recompile_limit}. This may be too small for AMD Quark needs that expects >={RECOMPILATION_LIMIT} (as different quantization schemes, different devices may trigger `scaled_fake_quantize` recompilations), and may trigger `torch._dynamo.exc.FailOnRecompileLimitHit` error."
            )
elif Version(torch.__version__) >= Version("2.5"):
    if torch._dynamo.config.cache_size_limit == 8:
        torch._dynamo.config.cache_size_limit = RECOMPILATION_LIMIT
    else:
        if torch._dynamo.config.cache_size_limit < RECOMPILATION_LIMIT:
            logger.warning(
                f"Detected user-specified torch._dynamo.config.recompile_limit={torch._dynamo.config.cache_size_limit}. This may be too small for AMD Quark needs that expects >={RECOMPILATION_LIMIT} (as different quantization schemes, different devices may trigger `scaled_fake_quantize` recompilations), and may trigger `torch._dynamo.exc.FailOnRecompileLimitHit` error."
            )

__all__ = [
    "quant_fp8_e4m3",
    "dequant_fp8_e4m3",
    "quant_fp8_e4m3_with_scale",
    "dequant_fp8_e4m3_with_scale",
    "quant_fp8_e5m2",
    "dequant_fp8_e5m2",
    "quant_fp8_e5m2_with_scale",
    "dequant_fp8_e5m2_with_scale",
    "scaled_fake_quantize",
]


def cast_to_float8_and_back(x: torch.Tensor, float8_dtype: torch.dtype) -> torch.Tensor:
    """
    Casting to float8 is broken with torch.compile with some torch versions with `torch<2.7`.

    This utility function bypasses the issue by guarding the bugged op to not be traced by `torch.compile` for `torch<2.7`.
    """
    x_dtype = x.dtype

    if TORCH_HIGHER_OR_EQUAL_2_7 and (
        x.device.type == "cpu" or (x.dtype not in {torch.float16, torch.bfloat16} and TRITON_GPU_SUPPORTS_FP8)
    ):
        return x.to(float8_dtype).to(x_dtype)
    elif float8_dtype == torch.float8_e4m3fn:
        # For torch<=2.7, casting to float8_e4m3fn is bugged (even on gfx942).
        # (CompilationError at `tmp8 = tmp7.to(tl.float8e4nv)`)
        # On gfx90a, float8_e4m3fn conversion is bugged as well:
        # ValueError: type fp8e4nv not supported in this architecture. The supported fp8 dtypes are ('fp8e5',)
        # The tests
        # - test/test_for_torch/test_hf_export_import.py::test_fp8_inp_weight_out_import[real_quantized]
        # - test/test_for_torch/test_hf_export_import.py::test_fp8_kv_cache_import[real_quantized-no-kv]
        # - test/test_for_torch/test_hf_export_import.py::test_wfp8_int4perchannel_afp8_import[real_quantized]
        # do NOT pass when using `torch.compile` for the op `x.to(float8_dtype).to(x_dtype)` when `x_dtype` is torch.float16 or torch.bfloat16.
        # It seems like this cast with torch.compile is not deterministic, or has some issue that needs to be
        # investigated. As a temporary solution, to prioritize correctness, we do not use torch.compile for
        # this cast for these dtypes.
        return torch.ops.torch_guard.cast_to_float8_e4m3fn_and_back(x)
    elif float8_dtype == torch.float8_e5m2:
        # No issue with torch.float8_e5m2 on torch<=2.7.
        # No issue with torch.float8_e5m2 on gfx90a.
        return x.to(float8_dtype).to(x_dtype)
    else:
        raise ValueError(f"Wrong dtype={float8_dtype}")


def prepare_inputs_per_group(
    inputs: torch.Tensor, scale: torch.Tensor, axis: int, group_size: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Ensures that the inputs and scale have the following shapes:

    * ``scale``: [-1, inputs.size(axis) // group_size, 1]
    * ``inputs```: [-1, inputs.size(axis) // group_size, group_size]

    For example, the input:
    * ``inputs`` of shape (256, 128, 64)
    * ``scale`` of shape (256, 4, 64)
    * ``axis=1``
    * ``group_size=32``

    gives ``inputs`` of shape (256 * 128, 4, 32)
          ``scale`` of shape (256 * 128, 4, 1)

    Passing the argument ``scales`` with the broadcasting dimension already present is accepted, for example:

    * ``inputs`` of shape (256, 128, 64)
    * ``scale`` of shape (8, 1, 128, 64)
    * ``axis=0``
    * ``group_size=32``

    gives ``inputs`` of shape (128 * 64, 8, 32)
          ``scale`` of shape (128 * 64, 8, 1)
    """
    inputs_ndim = inputs.ndim
    inputs = inputs.to(torch.float32)
    scale = scale.to(torch.float32)

    inputs = reshape_to_blocks(inputs, group_size, axis)

    # `reshape_to_blocks` reshapes to a 3D tensor [-1, axis_size // group_size, group_size].
    # For the scale, the `1` dimension should be the last. Remove the dim from
    # the scale tensor if it is already present.
    # `axis_positive` axis is expected to be of dimension `axis_size // group_size`.
    if scale.dim() == inputs_ndim + 1:
        axis_positive = (inputs_ndim + axis) % inputs_ndim
        scale = scale.squeeze(axis_positive + 1)

    # Move the group dimension to the end: [..., inputs.size(axis) // group_size, 1].
    scale = scale.transpose(axis, -1)
    scale = scale.unsqueeze(-1)

    # Finally, merge all dimensions except the end [inputs.size(axis) // group_size, 1].
    if scale.dim() > inputs.dim():
        scale = scale.reshape(-1, scale.shape[-2], scale.shape[-1])

    return inputs, scale


# namespace
quant_scope_lib = Library("quark", "DEF")

quant_scope_lib.define("quant_fp8_e4m3(Tensor x) -> Tensor")


@impl(quant_scope_lib, "quant_fp8_e4m3", "CompositeExplicitAutograd")
def quant_fp8_e4m3(inputs: torch.Tensor) -> torch.Tensor:
    inputs = torch.clamp(inputs, min=-448, max=448)
    return inputs.to(torch.float8_e4m3fn)


quant_scope_lib.define("quant_fp8_e5m2(Tensor x) -> Tensor")


@impl(quant_scope_lib, "quant_fp8_e5m2", "CompositeExplicitAutograd")
def quant_fp8_e5m2(inputs: torch.Tensor) -> torch.Tensor:
    inputs = torch.clamp(inputs, min=-57344, max=57344)
    return inputs.to(torch.float8_e5m2)


quant_scope_lib.define("dequant_fp8_e4m3(Tensor x) -> Tensor")


@impl(quant_scope_lib, "dequant_fp8_e4m3", "CompositeExplicitAutograd")
def dequant_fp8_e4m3(inputs: torch.Tensor) -> torch.Tensor:
    return inputs.to(torch.float16)


quant_scope_lib.define("dequant_fp8_e5m2(Tensor x) -> Tensor")


@impl(quant_scope_lib, "dequant_fp8_e5m2", "CompositeExplicitAutograd")
def dequant_fp8_e5m2(inputs: torch.Tensor) -> torch.Tensor:
    return inputs.to(torch.float16)


quant_scope_lib.define("quant_fp8_e4m3_with_scale(Tensor x, float scale) -> Tensor")


@impl(quant_scope_lib, "quant_fp8_e4m3_with_scale", "CompositeExplicitAutograd")
def quant_fp8_e4m3_with_scale(inputs: torch.Tensor, scale: float) -> torch.Tensor:
    inputs = inputs / scale
    inputs = torch.clamp(inputs, min=-448, max=448)
    return inputs.to(torch.float8_e4m3fn)


quant_scope_lib.define("quant_fp8_e5m2_with_scale(Tensor x, float scale) -> Tensor")


@impl(quant_scope_lib, "quant_fp8_e5m2_with_scale", "CompositeExplicitAutograd")
def quant_fp8_e5m2_with_scale(inputs: torch.Tensor, scale: float) -> torch.Tensor:
    inputs = inputs / scale
    inputs = torch.clamp(inputs, min=-57344, max=57344)
    return inputs.to(torch.float8_e5m2)


quant_scope_lib.define("dequant_fp8_e4m3_with_scale(Tensor x, float scale) -> Tensor")


@impl(quant_scope_lib, "dequant_fp8_e4m3_with_scale", "CompositeExplicitAutograd")
def dequant_fp8_e4m3_with_scale(inputs: torch.Tensor, scale: float) -> torch.Tensor:
    return inputs.to(torch.float16) * scale


quant_scope_lib.define("dequant_fp8_e5m2_with_scale(Tensor x, float scale) -> Tensor")


@impl(quant_scope_lib, "dequant_fp8_e5m2_with_scale", "CompositeExplicitAutograd")
def dequant_fp8_e5m2_with_scale(inputs: torch.Tensor, scale: float) -> torch.Tensor:
    return inputs.to(torch.float16) * scale


quant_scope_lib.define("quant_dequant_fp8_e4m3(Tensor x) -> Tensor")


@impl(quant_scope_lib, "quant_dequant_fp8_e4m3", "CompositeExplicitAutograd")
def quant_dequant_fp8_e4m3(inputs: torch.Tensor) -> torch.Tensor:
    inputs_type = inputs.dtype
    inputs = torch.clamp(inputs, min=-448, max=448)
    outputs = (inputs).to(torch.float8_e4m3fn).to(inputs_type)
    return outputs


quant_scope_lib.define("quant_dequant_fp8_e5m2(Tensor x) -> Tensor")


@impl(quant_scope_lib, "quant_dequant_fp8_e5m2", "CompositeExplicitAutograd")
def quant_dequant_fp8_e5m2(inputs: torch.Tensor) -> torch.Tensor:
    inputs_type = inputs.dtype
    inputs = torch.clamp(inputs, min=-57344, max=57344)
    outputs = (inputs).to(torch.float8_e5m2).to(inputs_type)
    return outputs


quant_scope_lib.define(
    "scaled_fake_quantize(str quant_dtype, Tensor inputs, Tensor scale, Tensor zero_point, SymInt axis, SymInt group_size, float quant_min, float quant_max, SymInt round_mode, str qscheme, str mx_element_dtype) -> Tensor"
)


@impl(quant_scope_lib, "scaled_fake_quantize", "CompositeExplicitAutograd")
def scaled_fake_quantize(
    quant_dtype: str,
    inputs: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
    group_size: int,
    quant_min: float,
    quant_max: float,
    round_mode: int,
    qscheme: str,
    mx_element_dtype: str,
) -> torch.Tensor:
    return scaled_fake_quantize_impl(
        quant_dtype,
        inputs,
        scale,
        zero_point,
        axis,
        group_size,
        quant_min,
        quant_max,
        round_mode,
        qscheme,
        mx_element_dtype,
    )


@register_fake("quark::scaled_fake_quantize")
def _(
    quant_dtype: str,
    inputs: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
    group_size: int,
    quant_min: float,
    quant_max: float,
    round_mode: int,
    qscheme: str,
    mx_element_dtype: str,
) -> torch.Tensor:
    return torch.empty_like(inputs)


@register_decomposition(torch.ops.quark.scaled_fake_quantize)
def scaled_fake_quantize_impl(
    quant_dtype: str,
    inputs: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
    group_size: int,
    quant_min: float,
    quant_max: float,
    round_mode: int,
    qscheme: str,
    mx_element_dtype: str,
) -> torch.Tensor:
    if quant_dtype not in DTYPE_TO_FAKE_QUANTIZE_FUNCS:
        raise ValueError(f"Unsupported Quant Data Type: {quant_dtype}")  # pragma: no cover

    return DTYPE_TO_FAKE_QUANTIZE_FUNCS[quant_dtype](  # type: ignore[operator]
        inputs,
        scale=scale,
        zero_point=zero_point,
        axis=axis,
        group_size=group_size,
        quant_min=quant_min,
        quant_max=quant_max,
        round_mode=round_mode,
        qscheme=qscheme,
        quant_dtype=quant_dtype,
        mx_element_dtype=mx_element_dtype,
    )


quant_scope_lib.define(
    'non_scaled_fake_quantize(Tensor input_tensor, str quant_dtype, str mx_element_dtype, SymInt axis, SymInt block_size, str scale_calculation_mode="even") -> Tensor'
)


@impl(quant_scope_lib, "non_scaled_fake_quantize", "CompositeExplicitAutograd")
def non_scaled_fake_quantize(
    input_tensor: torch.Tensor,
    quant_dtype: str,
    mx_element_dtype: str,
    axis: int,
    block_size: int,
    scale_calculation_mode: str = "even",
) -> torch.Tensor:
    return non_scaled_fake_quantize_impl(
        input_tensor, quant_dtype, mx_element_dtype, axis, block_size, scale_calculation_mode
    )


@register_fake("quark::non_scaled_fake_quantize")
def _(
    input_tensor: torch.Tensor,
    quant_dtype: str,
    mx_element_dtype: str,
    axis: int,
    block_size: int,
    scale_calculation_mode: str = "even",
) -> torch.Tensor:
    return torch.empty_like(input_tensor)


@register_decomposition(torch.ops.quark.non_scaled_fake_quantize)
def non_scaled_fake_quantize_impl(
    input_tensor: torch.Tensor,
    quant_dtype: str,
    mx_element_dtype: str,
    axis: int,
    block_size: int,
    scale_calculation_mode: str = "even",
) -> torch.Tensor:
    if quant_dtype not in DTYPE_TO_NON_SCALED_FAKE_QUANTIZE_FUNCS:
        logger.error(f"Unsupported Quant Data Type: {quant_dtype}")  # pragma: no cover

    if quant_dtype == Dtype.mx.value:
        fake_quantize_func = partial(fake_quantize_mx, scale_calculation_mode=scale_calculation_mode)
    else:
        fake_quantize_func = DTYPE_TO_NON_SCALED_FAKE_QUANTIZE_FUNCS[quant_dtype]  # type: ignore[assignment]

    return fake_quantize_func(
        input_tensor=input_tensor,
        quant_dtype=quant_dtype,
        mx_element_dtype=mx_element_dtype,
        axis=axis,
        block_size=block_size,
    )


# Custom ops used to avoid dynamo entering into certain non-compatible ops.
torch_guard_lib = Library("torch_guard", "DEF")


def fake_quantize_int(
    inputs: torch.Tensor,
    scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    axis: int | None = None,
    group_size: int | None = None,
    quant_min: float | None = None,
    quant_max: float | None = None,
    round_mode: int | None = None,
    qscheme: str | None = None,
    quant_dtype: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    quant_min = int(quant_min)
    quant_max = int(quant_max)
    if qscheme == QSchemeType.per_tensor.value:
        return fake_quantize_int_per_tensor_affine(inputs, scale, zero_point, quant_min, quant_max, round_mode)
    elif qscheme == QSchemeType.per_channel.value:
        return fake_quantize_int_per_channel_affine(inputs, scale, zero_point, axis, quant_min, quant_max, round_mode)
    elif qscheme == QSchemeType.per_group.value:
        return fake_quantize_int_per_group_affine(
            inputs, scale, zero_point, axis, group_size, quant_min, quant_max, round_mode
        )
    else:
        raise ValueError(f"Unsupported QuantSchema: {qscheme} for quant_dtype: {quant_dtype}")  # pragma: no cover


def fake_quantize_fp8_e4m3(
    inputs: torch.Tensor,
    scale: torch.Tensor | None = None,
    axis: int | None = None,
    qscheme: str | None = None,
    quant_dtype: str | None = None,
    group_size: int | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    if qscheme == QSchemeType.per_tensor.value:
        return fake_quantize_fp8_e4m3_per_tensor_with_scale(inputs, scale)
    elif qscheme == QSchemeType.per_channel.value:
        return fake_quantize_fp8_e4m3_per_channel_with_scale(inputs, scale, axis)
    elif qscheme == QSchemeType.per_group.value:
        return fake_quantize_fp8_per_group_with_scale(inputs, scale, axis, group_size, fp8_dtype=torch.float8_e4m3fn)
    else:
        raise ValueError(f"Unsupported QuantSchema: {qscheme} for quant_dtype: {quant_dtype}")  # pragma: no cover


def fake_quantize_fp8_e5m2(
    inputs: torch.Tensor,
    scale: torch.Tensor | None = None,
    axis: int | None = None,
    qscheme: str | None = None,
    quant_dtype: str | None = None,
    group_size: int | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    if qscheme == QSchemeType.per_tensor.value:
        return fake_quantize_fp8_e5m2_per_tensor_with_scale(inputs, scale)
    elif qscheme == QSchemeType.per_channel.value:
        return fake_quantize_fp8_e5m2_per_channel_with_scale(inputs, scale, axis)
    elif qscheme == QSchemeType.per_group.value:
        return fake_quantize_fp8_per_group_with_scale(inputs, scale, axis, group_size, fp8_dtype=torch.float8_e5m2)
    else:
        raise ValueError(f"Unsupported QuantSchema: {qscheme} for quant_dtype: {quant_dtype}")  # pragma: no cover


def fake_quantize_fp8_per_group_with_scale(
    input_tensor: torch.Tensor, scale: torch.Tensor, axis: int, group_size: int, fp8_dtype: torch.dtype, **kwargs: Any
) -> torch.Tensor:
    input_shape = list(input_tensor.shape)
    input_shape[-1], input_shape[axis] = input_shape[axis], input_shape[-1]

    eps = torch.finfo(torch.float32).eps
    scale = scale.masked_fill(scale == 0.0, eps)

    input_dtype = input_tensor.dtype
    input_tensor = input_tensor.to(torch.float32)
    scale = scale.to(torch.float32)

    input_tensor = reshape_to_blocks(input_tensor, group_size, axis)
    if scale.dim() < input_tensor.dim():
        scale = scale.unsqueeze(-1)

    input_tensor = input_tensor / scale

    quark_dtype = Dtype.from_torch_dtype(fp8_dtype)
    quant_min, quant_max = calculate_qmin_qmax(quark_dtype)
    input_tensor = torch.clamp(input_tensor, quant_min, quant_max)

    output_tensor = cast_to_float8_and_back(input_tensor, float8_dtype=fp8_dtype)
    output_tensor = output_tensor * scale

    output_tensor = output_tensor.reshape(output_tensor.size(0), -1)
    output_tensor = output_tensor[:, : input_shape[-1]].reshape(input_shape).to(input_dtype)
    if scale.dim() > output_tensor.dim():
        scale = scale.squeeze(-1)
    assert_no_nan(output_tensor, message="fake_quantize_fp8_per_group_with_scale output_tensor contains NaN!")
    return output_tensor.transpose(axis, -1)


def fake_quantize_fp4_fp6(
    inputs: torch.Tensor,
    scale: torch.Tensor | None = None,
    axis: int | None = None,
    qscheme: str | None = None,
    quant_dtype: str | None = None,
    group_size: int | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    if qscheme == QSchemeType.per_tensor.value:
        return fake_quantize_fp4_per_tensor_with_scale(inputs, scale)
    elif qscheme == QSchemeType.per_channel.value:
        return fake_quantize_fp4_fp6_per_channel_with_scale(inputs, scale, axis, quant_dtype)
    elif qscheme == QSchemeType.per_group.value:
        return fake_quantize_fp4_fp6_per_group_with_scale(inputs, scale, axis, group_size, quant_dtype, **kwargs)
    else:
        raise ValueError(f"Unsupported QuantSchema: {qscheme} for quant_dtype: {quant_dtype}")  # pragma: no cover


def fake_quantize_fp4_per_tensor_with_scale(inputs: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    pass  # TODO


def fake_quantize_fp4_fp6_per_channel_with_scale(
    inputs: torch.Tensor, scale: torch.Tensor, axis: int, quant_dtype: str
) -> torch.Tensor:
    inputs_type = inputs.dtype
    dtype = Dtype.from_str(quant_dtype)
    ebits, mbits, _ = get_dtype_params(dtype)
    _, quant_max = calculate_qmin_qmax(dtype)
    scale = scale.masked_fill(scale == 0.0, torch.finfo(torch.float32).eps).to(inputs_type).to(inputs.device)
    if axis >= 0:
        for k in range(inputs.dim() - axis - 1):
            scale = scale.unsqueeze(-1)
    else:
        for k in range(-1 - axis):
            scale = scale.unsqueeze(-1)
    inputs = inputs / scale
    qinputs = torch.ops.quark.fake_quantize_to_low_precision_fp(inputs.contiguous(), ebits, mbits, quant_max, 0)

    # NOTE: This can not be done in-place, as otherwise raises:
    # torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
    # AssertionError: aot_autograd expected to have an entirely functional graph, but found %mul_ : [num_users=1] = call_function[target=torch.ops.aten.mul_.Tensor](args = (%fake_quantize_to_low_precision_fp, %where), kwargs = {})
    outputs = qinputs * scale

    return outputs.to(inputs.dtype)


def fake_quantize_fp4_fp6_per_group_with_scale(
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
    axis: int,
    group_size: int,
    quant_dtype: str | None,
    **kwargs: Any,
) -> torch.Tensor:
    input_shape = list(input_tensor.shape)

    input_shape[-1], input_shape[axis] = input_shape[axis], input_shape[-1]

    ebits, mbits, _ = get_dtype_params(quant_dtype)
    eps = torch.finfo(torch.float32).eps
    scale = scale.masked_fill(scale == 0.0, eps)

    input_dtype = input_tensor.dtype
    input_tensor = input_tensor.to(torch.float32)
    scale = scale.to(torch.float32)

    input_tensor = reshape_to_blocks(input_tensor, group_size, axis)
    if scale.dim() < input_tensor.dim():
        scale = scale.unsqueeze(-1)

    max_exp = pow(2.0, ebits) - 1
    offset_exp = pow(2.0, ebits - 1) - 1
    quant_max = pow(2.0, max_exp - offset_exp) * (1 + (pow(2.0, mbits) - 1) / (pow(2.0, mbits)))

    input_tensor = input_tensor / scale

    output_tensor = torch.ops.quark.fake_quantize_to_low_precision_fp(input_tensor, ebits, mbits, quant_max, 0)

    # NOTE: This can not be done in-place, as otherwise raises:
    # torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
    # AssertionError: aot_autograd expected to have an entirely functional graph, but found %mul_ : [num_users=1] = call_function[target=torch.ops.aten.mul_.Tensor](args = (%fake_quantize_to_low_precision_fp, %where), kwargs = {})
    output_tensor = output_tensor * scale

    # We can not simply reshape to `(output_tensor.size(0), -1)` because
    # in some cases the first dimension may be 0-sized, resulting in
    # the error `cannot reshape tensor of 0 elements into shape [0, -1]`.
    # This is e.g. the case with mixtral and qwen2_moe before
    # https://github.com/huggingface/transformers/pull/32429,
    # i.e. with transformers<=4.51.
    output_tensor = output_tensor.reshape(output_tensor.size(0), output_tensor.shape[1:].numel())

    output_tensor = output_tensor[:, : input_shape[-1]].reshape(input_shape).to(input_dtype)

    if scale.dim() > output_tensor.dim():
        scale = scale.squeeze(-1)

    return output_tensor.transpose(axis, -1)


# `test/test_for_torch/test_realq_deq.py::test_fp16_per_channel_quantize_axis_range`
# `torch.equal` otherwise does not exactly pass. There is not much to fuse anyway.
torch_guard_lib.define("_fake_quantize_with_dtype_convert(Tensor inputs, str quant_dtype) -> Tensor")


@impl(torch_guard_lib, "_fake_quantize_with_dtype_convert", "CompositeExplicitAutograd")
def _fake_quantize_with_dtype_convert(inputs: torch.Tensor, quant_dtype: str) -> torch.Tensor:
    quant_torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(quant_dtype)
    input_origin_type = inputs.dtype
    if input_origin_type != quant_torch_dtype:
        inputs = inputs.to(quant_torch_dtype)
        inputs = inputs.to(input_origin_type)
    return inputs


@register_fake("torch_guard::_fake_quantize_with_dtype_convert")
def _(inputs: torch.Tensor, quant_dtype: str) -> torch.Tensor:
    return torch.empty_like(inputs)


def fake_quantize_with_dtype_convert(
    inputs: torch.Tensor, quant_dtype: str | None = None, **kwargs: Any
) -> torch.Tensor:
    return torch.ops.torch_guard._fake_quantize_with_dtype_convert(inputs, quant_dtype)


def fake_quantize_fp8_per_tensor_with_scale(
    inputs: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, max_value: Number
) -> torch.Tensor:
    inputs = inputs / scale
    inputs = torch.clamp(inputs, min=-max_value, max=max_value)

    inputs = cast_to_float8_and_back(inputs, float8_dtype=dtype) * scale

    return inputs


fake_quantize_fp8_e4m3_per_tensor_with_scale = partial(
    fake_quantize_fp8_per_tensor_with_scale, dtype=torch.float8_e4m3fn, max_value=448
)
fake_quantize_fp8_e5m2_per_tensor_with_scale = partial(
    fake_quantize_fp8_per_tensor_with_scale, dtype=torch.float8_e5m2, max_value=57344
)


def fake_quantize_fp8_per_channel_with_scale(
    inputs: torch.Tensor, scale: torch.Tensor, axis: int, dtype: torch.dtype, max_value: Number
) -> torch.Tensor:
    inputs_dtype = inputs.dtype
    scale = scale.to(inputs_dtype).to(inputs.device)
    if axis >= 0:
        for k in range(inputs.dim() - axis - 1):
            scale = scale.unsqueeze(-1)
    else:
        for k in range(-1 - axis):
            scale = scale.unsqueeze(-1)
    inputs = inputs / scale
    inputs = torch.clamp(inputs, min=-max_value, max=max_value)

    inputs = cast_to_float8_and_back(inputs, float8_dtype=dtype) * scale

    return inputs


fake_quantize_fp8_e4m3_per_channel_with_scale = partial(
    fake_quantize_fp8_per_channel_with_scale, dtype=torch.float8_e4m3fn, max_value=448
)
fake_quantize_fp8_e5m2_per_channel_with_scale = partial(
    fake_quantize_fp8_per_channel_with_scale, dtype=torch.float8_e5m2, max_value=57344
)

# Operations using kernels from `kernel_ext` can not be traced into by inductor.
# As the decorator `@torch.compiler.disable` either fails to prevent inductor attempting to enter them, or prevents to use `fullgraph=True`, we instead define
# custom ops wrapping `kernel_ext` operations to prevent torch.compile to inspect
# them.
# Reference: # Reference: https://github.com/pytorch/pytorch/issues/156322#issuecomment-2996141115
quant_scope_lib.define(
    "fake_quantize_per_tensor_affine(Tensor inputs, Tensor scale, Tensor zero_point, float quant_min, float quant_max, int round_mode) -> Tensor"
)


@impl(quant_scope_lib, "fake_quantize_per_tensor_affine", "CompositeExplicitAutograd")
def fake_quantize_per_tensor_affine(
    inputs: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: float,
    quant_max: float,
    round_mode: int,
) -> torch.Tensor:
    quant_min = int(quant_min)
    quant_max = int(quant_max)
    return kernel_ext.fake_quantize_per_tensor_affine(inputs, scale, zero_point, quant_min, quant_max, round_mode)


@register_fake("quark::fake_quantize_per_tensor_affine")
def _(
    inputs: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    quant_min: float,
    quant_max: float,
    round_mode: int,
) -> torch.Tensor:
    return torch.empty_like(inputs)


torch_guard_lib.define(
    "fake_quantize_per_tensor_affine_guard(Tensor inputs, Tensor scale, Tensor zero_point, int quant_min, int quant_max) -> Tensor"
)


@impl(torch_guard_lib, "fake_quantize_per_tensor_affine_guard", "CompositeExplicitAutograd")
def fake_quantize_per_tensor_affine_guard(
    inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int, quant_max: int
) -> torch.Tensor:
    # `torch.fake_quantize_per_tensor_affine` can not be traced into by inductor with the error `Backend compiler `inductor` failed with aten._fake_quantize_per_tensor_affine_cachemask_tensor_qparams.default`.
    # Thus, we guard it in a custom op.
    # Reference: https://github.com/pytorch/pytorch/issues/156322#issuecomment-2996141115
    return torch.fake_quantize_per_tensor_affine(inputs, scale, zero_point, quant_min, quant_max)


@register_fake("torch_guard::fake_quantize_per_tensor_affine_guard")
def _(
    inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: float, quant_max: float
) -> torch.Tensor:
    return torch.empty_like(inputs)


# See the comment in `fake_quantize_fp8_per_tensor_with_scale` for the reason this custom operator is defined.
torch_guard_lib.define("cast_to_float8_e4m3fn_and_back(Tensor inputs) -> Tensor")


@impl(torch_guard_lib, "cast_to_float8_e4m3fn_and_back", "CompositeExplicitAutograd")
def cast_to_float8_e4m3fn_and_back(inputs: torch.Tensor) -> torch.Tensor:
    inputs_dtype = inputs.dtype
    return inputs.to(torch.float8_e4m3fn).to(inputs_dtype)


@register_fake("torch_guard::cast_to_float8_e4m3fn_and_back")
def _(inputs: torch.Tensor) -> torch.Tensor:
    return torch.empty_like(inputs)


# NOTE: `@torch.compiler.disable` does not work when using `register_decomposition` as we do.
# As a workaround, we register as custom ops the functions which can not be traced into.
def fake_quantize_int_per_tensor_affine(
    inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int, quant_max: int, round_mode: int
) -> torch.Tensor:
    inputs_type = inputs.dtype
    scale_type = scale.dtype
    if inputs_type != torch.float:
        inputs = inputs.to(torch.float)
    if scale_type != inputs.dtype:
        scale = scale.to(inputs.dtype)

    if kernel_ext is not None and inputs.device != torch.device("cpu"):
        if scale.device != inputs.device:
            scale = scale.to(inputs.device)
        if zero_point.device != inputs.device:
            zero_point = zero_point.to(inputs.device)
        inputs = inputs.contiguous()
        res = torch.ops.quark.fake_quantize_per_tensor_affine(
            inputs, scale, zero_point, quant_min, quant_max, round_mode
        )
    else:
        res = torch.ops.torch_guard.fake_quantize_per_tensor_affine_guard(
            inputs, scale, zero_point, quant_min, quant_max
        )

    if inputs_type != res.dtype:
        res = res.to(inputs_type)
    if scale_type != scale.dtype:
        scale = scale.to(scale_type)
    return res


def fake_quantize_per_channel_affine(
    input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, axis: int, quant_min: int, quant_max: int
) -> torch.Tensor:
    """
    Implements ``torch.fake_quantize_per_channel_affine`` (https://docs.pytorch.org/docs/stable/generated/torch.fake_quantize_per_channel_affine.html).

    The function ``torch.fake_quantize_per_channel_affine`` does not support CUDA Graph, but this one does.
    """
    QUARK_AWQ_MEMORY_OPTIMIZATION = os.environ.get("QUARK_AWQ_MEMORY_OPTIMIZATION", None) == "1"
    # Currently torch.fake_quantize_per_channel_affine saves more gpu memory
    if QUARK_AWQ_MEMORY_OPTIMIZATION:
        return torch.fake_quantize_per_channel_affine(input, scale, zero_point, axis, quant_min, quant_max)
    unsqueeze_slice = (None,) * axis + (...,) + (None,) * (input.ndim - axis - 1)
    scale = scale[unsqueeze_slice]
    zero_point = zero_point[unsqueeze_slice]

    # PyTorch uses an aten::mul operation to divide by the scale in its implementation: https://github.com/pytorch/pytorch/blob/v2.7.1/aten/src/ATen/native/quantized/cuda/FakeQuantizeCore.cu#L186.
    # In order to have matching logits compared to `torch.fake_quantize_per_channel_affine`, we use an aten::div followed by an aten::mul op as well here.
    # Check for bounds of scale to avoid NaN/Inf in the output.
    scale_min = torch.finfo(scale.dtype).tiny
    scale = torch.clamp(scale, min=scale_min)
    inv_scale = 1.0 / scale

    # PyTorch uses `std::nearbyint(input_val * inv_scale) + zero_point`. This may yield different results than `std::nearbyint(input_val * inv_scale + zero_point)`, one needs to be extra careful here.
    # Reference: https://github.com/pytorch/pytorch/issues/49779.
    q_x = torch.clamp(torch.round(input * inv_scale) + zero_point, quant_min, quant_max)
    qdq_x = (q_x - zero_point) * scale
    return qdq_x


def fake_quantize_int_per_channel_affine(
    inputs: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
    round_mode: int,
) -> torch.Tensor:
    inputs_type = inputs.dtype
    scale_type = scale.dtype
    if inputs_type != torch.float:
        inputs = inputs.to(torch.float)
    if scale_type != inputs.dtype:
        scale = scale.to(inputs.dtype)

    # We do not use `torch.fake_quantize_per_channel_affine` as this operator does not support CUDA Graph capture.
    # Reference: https://github.com/pytorch/pytorch/issues/155231.
    res = fake_quantize_per_channel_affine(inputs, scale, zero_point, axis, quant_min, quant_max)

    if inputs_type != res.dtype:
        res = res.to(inputs_type)
    if scale_type != scale.dtype:
        scale = scale.to(scale_type)
    return res


def fake_quantize_int_per_group_affine(
    inputs: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
    group_size: int,
    quant_min: int,
    quant_max: int,
    round_mode: int,
) -> torch.Tensor:
    # Reshape input tensor to [-1, group_size] and then use per channel kernel"
    inputs_dim = inputs.size()
    new_axis_list = [i for i in range(len(inputs_dim))]  # noqa: C416

    new_axis_list[axis] = -1
    new_axis_list[-1] = axis

    inputs = inputs.permute(new_axis_list)
    inputs_shape = inputs.shape
    if group_size > 0:
        inputs = inputs.reshape(-1, group_size)

    inputs_type = inputs.dtype
    scale_type = scale.dtype
    if inputs_type != torch.float:
        inputs = inputs.to(torch.float)
    if scale_type != inputs.dtype:
        scale = scale.to(inputs.dtype)

    # We do not use `torch.fake_quantize_per_channel_affine` as this operator does not support CUDA Graph capture.
    # Reference: https://github.com/pytorch/pytorch/issues/155231.
    res = fake_quantize_per_channel_affine(inputs, scale.reshape(-1), zero_point.reshape(-1), 0, quant_min, quant_max)

    if inputs_type != res.dtype:
        res = res.to(inputs_type)
    if scale_type != scale.dtype:
        scale = scale.to(scale_type)

    # Reshape back input tensor
    res = res.reshape(inputs_shape).permute(new_axis_list)
    return res


def fake_quantize_bfp16(input_tensor: torch.Tensor, axis: int, block_size: int, **kwargs: Any) -> torch.Tensor:
    block_size = 8
    input_shape = list(input_tensor.shape)
    input_shape[-1], input_shape[axis] = input_shape[axis], input_shape[-1]
    eps = torch.finfo(torch.float32).eps

    block_x = reshape_to_blocks(input_tensor.detach(), block_size, axis)  # type: ignore
    block_x = block_x.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0)
    amax, _ = torch.max(torch.abs(block_x), dim=-1)

    scale = torch.pow(2, torch.floor(torch.log2(amax)) - 6)
    scale = scale.masked_fill(scale == 0.0, eps)
    zero_point = torch.zeros_like(scale).to(torch.int32)

    quantized_block_x_int = fake_quantize_int(
        inputs=block_x,
        scale=scale,
        zero_point=zero_point,
        axis=-1,
        group_size=block_size,
        quant_min=-129,
        quant_max=128,
        qscheme=QSchemeType.per_group.value,
    ) / scale.unsqueeze(-1)
    bool_mask = torch.logical_or(quantized_block_x_int >= 128, quantized_block_x_int < -128)
    scale_adjust = torch.pow(2, torch.any(bool_mask, dim=-1).to(torch.float32))
    amax *= scale_adjust

    scale = torch.pow(2, torch.floor(torch.log2(amax)) - 6)
    scale = scale.masked_fill(scale == 0.0, eps)
    zero_point = torch.zeros_like(scale).to(torch.int32)

    output_tensor = fake_quantize_int(
        inputs=block_x,
        scale=scale,
        zero_point=zero_point,
        axis=-1,
        group_size=block_size,
        quant_min=-128,
        quant_max=127,
        qscheme=QSchemeType.per_group.value,
    )

    output_tensor = output_tensor.reshape(-1, output_tensor.size(-1) * output_tensor.size(-2))[:, : input_shape[-1]]
    output_tensor = output_tensor.reshape(input_shape)
    return output_tensor.transpose(axis, -1)


def fake_quantize_mx(
    input_tensor: torch.Tensor, axis: int, block_size: int, scale_calculation_mode: str = "even", **kwargs: Any
) -> torch.Tensor:
    mx_element_dtype = kwargs["mx_element_dtype"]
    input_shape = list(input_tensor.shape)
    input_shape[-1], input_shape[axis] = input_shape[axis], input_shape[-1]

    block_x = reshape_to_blocks(input_tensor.detach(), block_size, axis)
    amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
    ebits, mbits, emax = get_dtype_params(mx_element_dtype)
    if scale_calculation_mode == "floor":
        scale = torch.pow(2, torch.floor(torch.log2(amax)) - emax)
    elif scale_calculation_mode == "ceil":
        scale = torch.pow(2, torch.ceil(torch.log2(amax)) - emax)
    else:
        scale = even_round(amax, Dtype(mx_element_dtype))
    eps = torch.finfo(torch.float32).eps
    scale = scale.masked_fill(scale == 0.0, eps)

    element_dtype = Dtype(mx_element_dtype)

    input_dtype = input_tensor.dtype
    input_tensor = input_tensor.to(torch.float32)
    scale = scale.to(torch.float32)

    input_tensor = reshape_to_blocks(input_tensor, block_size, axis)

    # convert input_tensor to different element_dtypes
    if element_dtype == Dtype.int8:
        output_tensor = fake_quantize_int_per_group_affine(
            inputs=input_tensor,
            scale=scale / 64,
            zero_point=torch.zeros_like(scale),
            axis=-1,
            group_size=block_size,
            quant_min=-127,
            quant_max=127,
            round_mode=0,
        )
    elif element_dtype == Dtype.fp8_e4m3:
        output_tensor = fake_quantize_fp8_e4m3_per_channel_with_scale(inputs=input_tensor, scale=scale, axis=-1)
    elif element_dtype == Dtype.fp8_e5m2:
        output_tensor = fake_quantize_fp8_e5m2_per_channel_with_scale(inputs=input_tensor, scale=scale, axis=-1)
    elif element_dtype in [Dtype.fp4, Dtype.fp6_e3m2, Dtype.fp6_e2m3]:
        # bias mode: ieee_wo_inf_and_nan
        quant_bit_e, quant_bit_m, _ = get_dtype_params(element_dtype)
        max_exp = pow(2.0, quant_bit_e) - 1
        offset_exp = pow(2.0, quant_bit_e - 1) - 1
        quant_max = pow(2.0, max_exp - offset_exp) * (1 + (pow(2.0, quant_bit_m) - 1) / (pow(2.0, quant_bit_m)))

        input_tensor = input_tensor / scale
        output_tensor = torch.ops.quark.fake_quantize_to_low_precision_fp(
            input_tensor.contiguous(), ebits, mbits, quant_max, 0
        )

        # NOTE: This can not be done in-place, as otherwise raises:
        # torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
        # AssertionError: aot_autograd expected to have an entirely functional graph, but found %mul_ : [num_users=1] = call_function[target=torch.ops.aten.mul_.Tensor](args = (%fake_quantize_to_low_precision_fp, %where), kwargs = {})
        output_tensor = output_tensor * scale
    else:
        raise ValueError(f"unsupported element dtype : {element_dtype}")  # pragma: no cover

    output_tensor = output_tensor.reshape(output_tensor.size(0), -1)
    output_tensor = output_tensor[:, : input_shape[-1]].reshape(input_shape).to(input_dtype)
    return output_tensor.transpose(axis, -1)


def fake_quantize_mx6_mx9(input_tensor: torch.Tensor, axis: int, block_size: int, **kwargs: Any) -> torch.Tensor:
    quant_bit = kwargs["quant_bit"]
    input_shape = list(input_tensor.shape)
    input_shape[-1], input_shape[axis] = input_shape[axis], input_shape[-1]

    block_x = reshape_to_blocks(input_tensor.detach(), block_size, axis)

    # NOTE: This can not be done in-place with Tensor._nan_to_num, as otherwise raises:
    # torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
    # AssertionError: aot_autograd expected to have an entirely functional graph=
    block_x = torch.nan_to_num(block_x, nan=0.0, posinf=0.0, neginf=0.0)

    amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
    scale = t_exponent(amax)

    input_dtype = input_tensor.dtype
    max_exp = scale

    shape_list = list(input_tensor.shape)
    shape_list[axis], shape_list[-1] = shape_list[-1], shape_list[axis]
    input_tensor = reshape_to_blocks(input_tensor, block_size, axis)

    t_exp = t_exponent(input_tensor)
    idx2 = max_exp - t_exp >= 1

    # shared prime bit
    number_count_shared_prime_bit = 2
    assert idx2.shape[-1] % number_count_shared_prime_bit == 0
    old_shape = idx2.shape
    target_shape = old_shape[0:-1] + (old_shape[-1] // number_count_shared_prime_bit, number_count_shared_prime_bit)
    idx2 = idx2.reshape(target_shape)
    idx2 = torch.sum(idx2, -1, keepdim=True) == number_count_shared_prime_bit
    repeat_times = [1 for i in range(len(idx2.shape))]
    repeat_times[-1] = number_count_shared_prime_bit
    idx2 = idx2.repeat(repeat_times)
    idx2 = idx2.reshape(old_shape)

    shared_exp = idx2 * (-1) + max_exp
    scale = torch.pow(2.0, shared_exp - quant_bit + 2)

    quant_max = (
        torch.clamp_max(torch.pow(2.0, max_exp.to(torch.float64) + 1) - scale, torch.finfo(torch.float32).max).to(
            torch.float32
        )
        / scale
    )

    output_tensor = torch.round(input_tensor / scale)
    output_tensor = torch.clamp(output_tensor, -quant_max, quant_max) * scale

    output_tensor = output_tensor.reshape(output_tensor.size(0), -1)
    output_tensor = output_tensor[:, : input_shape[-1]].reshape(input_shape).to(input_dtype)
    return output_tensor.transpose(axis, -1)


def fake_quantize_non_mx(input_tensor: torch.Tensor, element_dtype: Dtype, axis: int, block_size: int) -> torch.Tensor:
    input_shape = list(input_tensor.shape)
    input_shape[-1], input_shape[axis] = input_shape[axis], input_shape[-1]

    block_x = reshape_to_blocks(input_tensor.detach(), block_size, axis)
    amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
    ebits, mbits, emax = get_dtype_params(element_dtype)
    element_dtype_str = Dtype(element_dtype)

    quant_bit_e, quant_bit_m, _ = get_dtype_params(element_dtype_str)
    max_exp = pow(2.0, quant_bit_e) - 1
    offset_exp = pow(2.0, quant_bit_e - 1) - 1
    quant_max = pow(2.0, max_exp - offset_exp) * (1 + (pow(2.0, quant_bit_m) - 1) / (pow(2.0, quant_bit_m)))

    scale = torch.div(amax, quant_max)

    eps = torch.finfo(torch.float32).eps
    scale = scale.masked_fill(scale == 0.0, eps)

    input_dtype = input_tensor.dtype
    input_tensor = input_tensor.to(torch.float32)
    scale = scale.to(torch.float32)

    input_tensor = reshape_to_blocks(input_tensor, block_size, axis)

    input_tensor = input_tensor / scale
    output_tensor = torch.ops.quark.fake_quantize_to_low_precision_fp(
        input_tensor.contiguous(), ebits, mbits, quant_max, 0
    )

    # NOTE: This can not be done in-place, as otherwise raises:
    # torch._dynamo.exc.BackendCompilerFailed: backend='inductor' raised:
    # AssertionError: aot_autograd expected to have an entirely functional graph, but found %mul_ : [num_users=1] = call_function[target=torch.ops.aten.mul_.Tensor](args = (%fake_quantize_to_low_precision_fp, %where), kwargs = {})
    output_tensor = output_tensor * scale

    output_tensor = output_tensor.reshape(output_tensor.size(0), -1)
    output_tensor = output_tensor[:, : input_shape[-1]].reshape(input_shape).to(input_dtype)
    return output_tensor.transpose(axis, -1)


quant_scope_lib.define(
    "fake_quantize_to_low_precision_fp(Tensor inputs, SymInt ebits, SymInt mbits, float quant_max, SymInt round_mode) -> Tensor"
)


@impl(quant_scope_lib, "fake_quantize_to_low_precision_fp", "CompositeExplicitAutograd")
def fake_quantize_to_low_precision_fp(
    inputs: torch.Tensor, ebits: int, mbits: int, quant_max: float, round_mode: int
) -> torch.Tensor:
    return kernel_ext.fake_quantize_to_low_precision_fp(inputs.contiguous(), ebits, mbits, quant_max, round_mode)


@register_fake("quark::fake_quantize_to_low_precision_fp")
def _(inputs: torch.Tensor, ebits: int, mbits: int, quant_max: float, round_mode: int) -> torch.Tensor:
    return torch.empty_like(inputs, memory_format=torch.contiguous_format)


quant_scope_lib.define(
    "scaled_real_quantize(str quant_dtype, Tensor inputs, Tensor scale, Tensor zero_point, SymInt axis, SymInt group_size, float quant_min, float quant_max, SymInt round_mode, str qscheme) -> Tensor"
)


@log_errors
@impl(quant_scope_lib, "scaled_real_quantize", "CompositeExplicitAutograd")
def scaled_real_quantize(
    quant_dtype: str,
    inputs: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
    group_size: int,
    quant_min: float,
    quant_max: float,
    round_mode: int,
    qscheme: str,
) -> torch.Tensor:
    return scaled_real_quantize_impl(
        quant_dtype, inputs, scale, zero_point, axis, group_size, quant_min, quant_max, round_mode, qscheme
    )


@register_decomposition(torch.ops.quark.scaled_real_quantize)
def scaled_real_quantize_impl(
    quant_dtype: str,
    inputs: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
    group_size: int,
    quant_min: float,
    quant_max: float,
    round_mode: int,
    qscheme: str,
) -> torch.Tensor:
    if quant_dtype not in DTYPE_TO_REAL_QUANTIZE_FUNCS:
        raise ValueError(f"Unsupported Quant Data Type: {quant_dtype}")  # pragma: no cover

    return DTYPE_TO_REAL_QUANTIZE_FUNCS[quant_dtype](  # type: ignore[operator]
        inputs,
        scale=scale,
        zero_point=zero_point,
        axis=axis,
        group_size=group_size,
        quant_min=quant_min,
        quant_max=quant_max,
        round_mode=round_mode,
        qscheme=qscheme,
        quant_dtype=quant_dtype,
    )


quant_scope_lib.define(
    "non_scaled_real_quantize(Tensor input_tensor, str quant_dtype, str mx_element_dtype, SymInt axis, SymInt block_size) -> Tensor"
)


@log_errors
@impl(quant_scope_lib, "non_scaled_real_quantize", "CompositeExplicitAutograd")
def non_scaled_real_quantize(
    input_tensor: torch.Tensor, quant_dtype: str, mx_element_dtype: str, axis: int, block_size: int
) -> torch.Tensor:
    return non_scaled_real_quantize_impl(input_tensor, quant_dtype, mx_element_dtype, axis, block_size)


@register_decomposition(torch.ops.quark.non_scaled_real_quantize)
def non_scaled_real_quantize_impl(
    input_tensor: torch.Tensor, quant_dtype: str, mx_element_dtype: str, axis: int, block_size: int
) -> torch.Tensor:
    assert quant_dtype == "mx" and mx_element_dtype in ["fp4", "fp6_e2m3", "fp6_e3m2"], (
        "Only mxfp4, mxfp6_e2m3 and mxfp6_e3m2 is supported!"
    )

    return real_quantize_mxfp(
        input_tensor=input_tensor, mx_element_dtype=mx_element_dtype, axis=axis, block_size=block_size
    )


def real_quantize_mxfp(
    input_tensor: torch.Tensor, mx_element_dtype: str, axis: int, block_size: int, **kwargs: Any
) -> torch.Tensor:
    assert mx_element_dtype in ["fp4", "fp6_e2m3", "fp6_e3m2"], "Only mxfp4, mxfp6_e2m3 and mxfp6_e3m2 is supported!"
    assert input_tensor.shape[-1] % 32 == 0
    input_shape = list(input_tensor.shape)
    input_shape[-1], input_shape[axis] = input_shape[axis], input_shape[-1]

    block_x = reshape_to_blocks(input_tensor.detach(), block_size, axis)
    amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
    ebits, mbits, emax = get_dtype_params(mx_element_dtype)
    scale = torch.pow(2, torch.floor(torch.log2(amax)) - emax)
    eps = torch.finfo(torch.float32).eps
    scale = scale.masked_fill(scale == 0.0, eps)

    element_dtype = Dtype(mx_element_dtype)

    input_dtype = input_tensor.dtype
    input_tensor = input_tensor.to(torch.float32)
    scale = scale.to(torch.float32)

    input_tensor = reshape_to_blocks(input_tensor, block_size, axis)

    quant_bit_e, quant_bit_m, _ = get_dtype_params(element_dtype)
    max_exp = pow(2.0, quant_bit_e) - 1
    offset_exp = pow(2.0, quant_bit_e - 1) - 1
    quant_max = pow(2.0, max_exp - offset_exp) * (1 + (pow(2.0, quant_bit_m) - 1) / (pow(2.0, quant_bit_m)))

    input_tensor = input_tensor / scale
    output_tensor = torch.ops.quark.fake_quantize_to_low_precision_fp(
        input_tensor.contiguous(), ebits, mbits, quant_max, 0
    )
    output_tensor = torch.cat([scale, output_tensor], dim=-1)
    output_tensor = output_tensor.reshape(output_tensor.size(0), -1)
    input_shape[-1] = input_shape[-1] // 32 * 33
    output_tensor = output_tensor[:, : input_shape[-1]].reshape(input_shape)

    input_tensor = input_tensor.to(input_dtype)
    return output_tensor.transpose(axis, -1)


def real_quantize_int(
    inputs: torch.Tensor,
    scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    axis: int | None = None,
    group_size: int | None = None,
    quant_min: float | None = None,
    quant_max: float | None = None,
    round_mode: int | None = None,
    qscheme: str | None = None,
    quant_dtype: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    quant_min = int(quant_min)
    quant_max = int(quant_max)
    if qscheme == QSchemeType.per_tensor.value:
        return real_quantize_int_per_tensor_affine(inputs, scale, zero_point, quant_min, quant_max, round_mode)
    elif qscheme == QSchemeType.per_channel.value:
        return real_quantize_int_per_channel_affine(inputs, scale, zero_point, axis, quant_min, quant_max, round_mode)
    elif qscheme == QSchemeType.per_group.value:
        return real_quantize_int_per_group_affine(
            inputs, scale, zero_point, axis, group_size, quant_min, quant_max, round_mode
        )
    else:
        raise ValueError(f"Unsupported QuantSchema: {qscheme} for quant_dtype: {quant_dtype}")  # pragma: no cover


def real_quantize_fp8_e4m3(
    inputs: torch.Tensor,
    scale: torch.Tensor | None = None,
    axis: int | None = None,
    qscheme: str | None = None,
    quant_dtype: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    if qscheme == QSchemeType.per_tensor.value:
        return real_quantize_fp8_e4m3_per_tensor_with_scale(inputs, scale)
    elif qscheme == QSchemeType.per_channel.value:
        return real_quantize_fp8_e4m3_per_channel_with_scale(inputs, scale, axis)
    else:
        raise ValueError(f"Unsupported QuantSchema: {qscheme} for quant_dtype: {quant_dtype}")  # pragma: no cover


def real_quantize_fp8_e5m2(
    inputs: torch.Tensor,
    scale: torch.Tensor | None = None,
    axis: int | None = None,
    qscheme: str | None = None,
    quant_dtype: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    if qscheme == QSchemeType.per_tensor.value:
        return real_quantize_fp8_e5m2_per_tensor_with_scale(inputs, scale)
    elif qscheme == QSchemeType.per_channel.value:
        return real_quantize_fp8_e5m2_per_channel_with_scale(inputs, scale, axis)
    else:
        raise ValueError(f"Unsupported QuantSchema: {qscheme} for quant_dtype: {quant_dtype}")  # pragma: no cover


def real_quantize_with_dtype_convert(
    inputs: torch.Tensor, quant_dtype: str | None = None, **kwargs: Any
) -> torch.Tensor:
    return _real_quantize_with_dtype_convert(inputs, quant_dtype)


def real_quantize_int_per_tensor_affine(
    inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int, quant_max: int, round_mode: int
) -> torch.Tensor:
    inputs_type = inputs.dtype
    scale_type = scale.dtype
    if inputs_type != torch.float:
        inputs = inputs.to(torch.float)
    if scale_type != inputs.dtype:
        scale = scale.to(inputs.dtype)

    # PyTorch uses an aten::mul operation to divide by the scale in its implementation: https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp#L2535
    # In order to have matching logits compared to `torch.fake_quantize_per_tensor_affine`, we use an aten::mul op as well here.
    inv_scale = 1.0 / scale
    res = torch.round(inputs * inv_scale + zero_point).clamp_(quant_min, quant_max)

    if inputs.dtype != inputs_type:
        inputs = inputs.to(inputs_type)
    if scale.dtype != scale_type:
        scale = scale.to(scale_type)

    return res.contiguous().to(torch.int)


def real_quantize_int_per_channel_affine(
    inputs: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
    quant_min: int,
    quant_max: int,
    round_mode: int,
) -> torch.Tensor:
    inputs_type = inputs.dtype
    scale_type = scale.dtype
    if inputs_type != torch.float:
        inputs = inputs.to(torch.float)
    if scale_type != inputs.dtype:
        scale = scale.to(inputs.dtype)

    inputs_transpose = torch.transpose(inputs, axis, -1)

    # PyTorch uses an aten::mul operation to divide by the scale in its implementation: https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp#L2656
    # In order to have matching logits compared to `torch.fake_quantize_per_tensor_affine`, we use an aten::mul op as well here.
    inv_scale = 1.0 / scale

    res = torch.round(inputs_transpose * inv_scale + zero_point).clamp(quant_min, quant_max)
    res = res.transpose(-1, axis)

    if inputs.dtype != inputs_type:
        inputs = inputs.to(inputs_type)
    if scale.dtype != scale_type:
        scale = scale.to(scale_type)

    return res.contiguous().to(torch.int)


def real_quantize_int_per_group_affine(
    inputs: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
    group_size: int,
    quant_min: int,
    quant_max: int,
    round_mode: int,
) -> torch.Tensor:
    inputs_type = inputs.dtype
    scale_type = scale.dtype
    if inputs_type != torch.float:
        inputs = inputs.to(torch.float)
    if scale_type != inputs.dtype:
        scale = scale.to(inputs.dtype)

    inputs_transpose = torch.transpose(inputs, axis, -1)
    scale_transpose = torch.transpose(scale, axis, -1)
    zp_transpose = torch.transpose(zero_point, axis, -1)

    inputs_shape = inputs_transpose.shape
    scale_shape = scale_transpose.shape
    g_size = inputs_shape[-1] // scale_shape[-1]

    inputs_reshape = torch.reshape(inputs_transpose, (-1, inputs_transpose.shape[-1]))
    scale_reshape = torch.reshape(scale_transpose, (-1, scale_transpose.shape[-1]))
    zp_reshape = torch.reshape(zp_transpose, (-1, zp_transpose.shape[-1]))

    quant_dim = inputs_transpose.shape[-1]

    # PyTorch uses an aten::mul operation to divide by the scale in its implementation: https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/quantized/cpu/kernels/QuantizedOpKernels.cpp#L2535
    # In order to have matching logits compared to `torch.fake_quantize_per_channel_affine`, we use an aten::mul op as well
    # here.
    scale_reshape_inv = 1.0 / scale_reshape[:, torch.arange(quant_dim) // g_size]

    inputs_div = inputs_reshape * scale_reshape_inv + zp_reshape[:, torch.arange(quant_dim) // g_size]
    res = torch.round(inputs_div).clamp(quant_min, quant_max)
    res = res.reshape(inputs_shape).transpose(-1, axis)

    if inputs.dtype != inputs_type:
        inputs = inputs.to(inputs_type)
    if scale.dtype != scale_type:
        scale = scale.to(scale_type)

    return res.contiguous().to(torch.int)


def real_quantize_fp8_per_tensor_with_scale(
    inputs: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype, max_value: Number
) -> torch.Tensor:
    res = inputs / scale
    res = torch.clamp(res, min=-max_value, max=max_value)
    return res.to(dtype)


real_quantize_fp8_e4m3_per_tensor_with_scale = partial(
    real_quantize_fp8_per_tensor_with_scale, dtype=torch.float8_e4m3fn, max_value=448
)
real_quantize_fp8_e5m2_per_tensor_with_scale = partial(
    real_quantize_fp8_per_tensor_with_scale, dtype=torch.float8_e5m2, max_value=57344
)


def real_quantize_fp8_per_channel_with_scale(
    inputs: torch.Tensor, scale: torch.Tensor, axis: int, dtype: torch.dtype, max_value: Number
) -> torch.Tensor:
    inputs_type = inputs.dtype
    scale = scale.to(inputs_type).to(inputs.device)
    if axis >= 0:
        for k in range(inputs.dim() - axis - 1):
            scale = scale.unsqueeze(-1)
    else:
        for k in range(-1 - axis):
            scale = scale.unsqueeze(-1)
    res = inputs / scale
    res = torch.clamp(res, min=-max_value, max=max_value)
    return res.to(dtype)


real_quantize_fp8_e4m3_per_channel_with_scale = partial(
    real_quantize_fp8_per_channel_with_scale, dtype=torch.float8_e4m3fn, max_value=448
)
real_quantize_fp8_e5m2_per_channel_with_scale = partial(
    real_quantize_fp8_per_channel_with_scale, dtype=torch.float8_e5m2, max_value=57344
)


def real_quantize_fp4_fp6_per_group(
    input_tensor: torch.Tensor,
    scale: torch.Tensor,
    axis: int,
    group_size: int,
    quant_dtype: str | None,
    **kwargs: Any,
) -> torch.Tensor:
    input_shape = list(input_tensor.shape)
    input_shape[-1], input_shape[axis] = input_shape[axis], input_shape[-1]

    ebits, mbits, _ = get_dtype_params(quant_dtype)
    eps = torch.finfo(torch.float32).eps
    scale = scale.masked_fill(scale == 0.0, eps)

    input_dtype = input_tensor.dtype

    input_tensor, scale = prepare_inputs_per_group(input_tensor, scale, axis, group_size)

    max_exp = pow(2.0, ebits) - 1
    offset_exp = pow(2.0, ebits - 1) - 1
    quant_max = pow(2.0, max_exp - offset_exp) * (1 + (pow(2.0, mbits) - 1) / (pow(2.0, mbits)))

    input_tensor = input_tensor / scale
    output_tensor = torch.ops.quark.fake_quantize_to_low_precision_fp(
        input_tensor.contiguous(), ebits, mbits, quant_max, 0
    )

    output_tensor = output_tensor.reshape(output_tensor.size(0), -1)
    output_tensor = output_tensor[:, : input_shape[-1]].reshape(input_shape)

    input_tensor = input_tensor.to(input_dtype)

    return output_tensor.transpose(axis, -1)


def _real_quantize_with_dtype_convert(inputs: torch.Tensor, quant_dtype: str) -> torch.Tensor:
    quant_torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(quant_dtype)
    input_origin_type = inputs.dtype
    res = inputs
    if input_origin_type != quant_torch_dtype:
        res = res.to(quant_torch_dtype)
    return res


quant_scope_lib.define(
    "dequantize(str quant_dtype, Tensor inputs, Tensor scale, Tensor zero_point, SymInt axis, SymInt group_size, str qscheme) -> Tensor"
)


@impl(quant_scope_lib, "dequantize", "CompositeExplicitAutograd")
def dequantize(
    quant_dtype: str,
    inputs: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
    group_size: int,
    qscheme: str,
) -> torch.Tensor:
    """
    Dequantizes the unpacked tensor ``inputs`` using the quantization parameters ``scale`` and ``zero_point``.

    In case ``qscheme="per_group"``, ``axis`` indicates the dimension for which the quantization parameters ``scale`` and ``zero_point`` are shared per group of ``group_size``.

    In this case, an example of valid input is:
    * ``inputs`` of shape (256, 128, 64)
    * ``scale`` of shape (256, 128, 2)
    * ``axis=-1``
    * ``group_size=32``.
    """
    return dequantize_impl(quant_dtype, inputs, scale, zero_point, axis, group_size, qscheme)


@register_decomposition(torch.ops.quark.dequantize)
def dequantize_impl(
    quant_dtype: str,
    inputs: torch.Tensor,
    scale: torch.Tensor,
    zero_point: torch.Tensor,
    axis: int,
    group_size: int,
    qscheme: str,
) -> torch.Tensor:
    if quant_dtype not in DTYPE_TO_DEQUANTIZE_FUNCS:
        raise ValueError(f"Unsupported Quant Data Type: {quant_dtype}")  # pragma: no cover

    return DTYPE_TO_DEQUANTIZE_FUNCS[quant_dtype](  # type: ignore[operator]
        inputs,
        scale=scale,
        zero_point=zero_point,
        axis=axis,
        group_size=group_size,
        qscheme=qscheme,
        quant_dtype=quant_dtype,
    )


def dequantize_int(
    inputs: torch.Tensor,
    scale: torch.Tensor | None = None,
    zero_point: torch.Tensor | None = None,
    axis: int | None = None,
    group_size: int | None = None,
    qscheme: str | None = None,
    quant_dtype: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    if qscheme == QSchemeType.per_tensor.value:
        return dequantize_int_per_tensor_affine(inputs, scale, zero_point)
    elif qscheme == QSchemeType.per_channel.value:
        return dequantize_int_per_channel_affine(inputs, scale, zero_point, axis)
    elif qscheme == QSchemeType.per_group.value:
        return dequantize_int_per_group_affine(inputs, scale, zero_point, axis, group_size)
    else:
        raise ValueError(f"Unsupported QuantSchema: {qscheme} for quant_dtype: {quant_dtype}")  # pragma: no cover


def dequantize_fp8(
    inputs: torch.Tensor,
    scale: torch.Tensor | None = None,
    axis: int | None = None,
    qscheme: str | None = None,
    quant_dtype: str | None = None,
    **kwargs: Any,
) -> torch.Tensor:
    if qscheme == QSchemeType.per_tensor.value:
        return dequantize_fp8_per_tensor_with_scale(inputs, scale)
    elif qscheme == QSchemeType.per_channel.value:
        return dequantize_fp8_per_channel_with_scale(inputs, scale, axis)
    else:
        raise ValueError(f"Unsupported QuantSchema: {qscheme} for quant_dtype: {quant_dtype}")  # pragma: no cover


def dequantize_with_dtype_convert(inputs: torch.Tensor, quant_dtype: str | None = None, **kwargs: Any) -> torch.Tensor:
    return _dequantize_with_dtype_convert(inputs, quant_dtype)


def dequantize_int_per_tensor_affine(
    inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor
) -> torch.Tensor:
    scale_type = scale.dtype
    if scale_type != torch.float:
        scale = scale.to(torch.float)

    res = (inputs - zero_point).to(torch.float) * scale

    if scale.dtype != scale_type:
        scale = scale.to(scale_type)

    return res.contiguous()


def dequantize_int_per_channel_affine(
    inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, axis: int
) -> torch.Tensor:
    scale_type = scale.dtype
    if scale_type != torch.float:
        scale = scale.to(torch.float)

    inputs_transpose = torch.transpose(inputs, axis, -1)
    res = (inputs_transpose.to(torch.float) - zero_point) * scale
    res = res.transpose(-1, axis)

    if scale.dtype != scale_type:
        scale = scale.to(scale_type)

    return res.contiguous()


def dequantize_int_per_group_affine(
    inputs: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, axis: int, group_size: int
) -> torch.Tensor:
    """
    Dequantizes `inputs`, which is assumed to be quantized per-group.

    If axis = 1:
    - `scale` is assumed to be of shape (out_features, in_features // group_size).
    - `zero_point` is assumed to be of shape (out_features, in_features // group_size).
    """
    scale_type = scale.dtype
    if scale_type != torch.float:
        scale = scale.to(torch.float)

    inputs_transpose = torch.transpose(inputs, axis, -1)
    scale_transpose = torch.transpose(scale, axis, -1)
    zp_transpose = torch.transpose(zero_point, axis, -1)

    inputs_shape = inputs_transpose.shape
    scale_shape = scale_transpose.shape
    g_size = inputs_shape[-1] // scale_shape[-1]

    inputs_reshape = torch.reshape(inputs_transpose, (-1, inputs_transpose.shape[-1]))
    scale_reshape = torch.reshape(scale_transpose, (-1, scale_transpose.shape[-1]))
    zp_reshape = torch.reshape(zp_transpose, (-1, zp_transpose.shape[-1]))

    quant_dim = inputs_transpose.shape[-1]
    dequantized = (inputs_reshape.to(torch.float) - zp_reshape[:, torch.arange(quant_dim) // g_size]) * scale_reshape[
        :, torch.arange(quant_dim) // g_size
    ]

    res = dequantized.reshape(inputs_shape).transpose(-1, axis)

    if scale.dtype != scale_type:
        scale = scale.to(scale_type)

    return res.contiguous()


def dequantize_fp8_per_tensor_with_scale(inputs: torch.Tensor, scale: float) -> torch.Tensor:
    return (inputs.to(torch.float32) * scale).contiguous()


def dequantize_fp8_per_channel_with_scale(inputs: torch.Tensor, scale: torch.Tensor, axis: int) -> torch.Tensor:
    if axis >= 0:
        for k in range(inputs.dim() - axis - 1):
            scale = scale.unsqueeze(-1)
    else:
        for k in range(-1 - axis):
            scale = scale.unsqueeze(-1)
    res = inputs.to(torch.float32) * scale.to(torch.float32)
    return res.contiguous()


def dequantize_fp4_fp6_per_group(
    inputs: torch.Tensor, scale: torch.Tensor, axis: int, group_size: int, **kwargs: Any
) -> torch.Tensor:
    input_shape = list(inputs.shape)
    input_shape[-1], input_shape[axis] = input_shape[axis], input_shape[-1]

    inputs_dtype = inputs.dtype

    inputs, scale = prepare_inputs_per_group(inputs, scale, axis, group_size)

    outputs = inputs * scale

    outputs = outputs.reshape(outputs.size(0), -1)
    outputs = outputs[:, : input_shape[-1]].reshape(input_shape).to(inputs_dtype)

    if scale.dim() > outputs.dim():
        scale = scale.squeeze(-1)

    return outputs.transpose(axis, -1).contiguous()


def _dequantize_with_dtype_convert(inputs: torch.Tensor, quant_dtype: str) -> torch.Tensor:
    return inputs


DTYPE_TO_DEQUANTIZE_FUNCS = {
    Dtype.int2.value: dequantize_int,
    Dtype.int3.value: dequantize_int,
    Dtype.int4.value: dequantize_int,
    Dtype.uint4.value: dequantize_int,
    Dtype.int8.value: dequantize_int,
    Dtype.uint8.value: dequantize_int,
    Dtype.fp8_e4m3.value: dequantize_fp8,
    Dtype.fp8_e5m2.value: dequantize_fp8,
    Dtype.bfloat16.value: dequantize_with_dtype_convert,
    Dtype.float16.value: dequantize_with_dtype_convert,
    Dtype.fp4.value: dequantize_fp4_fp6_per_group,
    Dtype.fp6_e2m3.value: dequantize_fp4_fp6_per_group,
    Dtype.fp6_e3m2.value: dequantize_fp4_fp6_per_group,
}

DTYPE_TO_FAKE_QUANTIZE_FUNCS = {
    Dtype.int2.value: fake_quantize_int,
    Dtype.int3.value: fake_quantize_int,
    Dtype.int4.value: fake_quantize_int,
    Dtype.uint16.value: fake_quantize_int,
    Dtype.int16.value: fake_quantize_int,
    Dtype.int32.value: fake_quantize_int,
    Dtype.uint4.value: fake_quantize_int,
    Dtype.int8.value: fake_quantize_int,
    Dtype.uint8.value: fake_quantize_int,
    Dtype.fp8_e4m3.value: fake_quantize_fp8_e4m3,
    Dtype.fp8_e5m2.value: fake_quantize_fp8_e5m2,
    Dtype.bfloat16.value: fake_quantize_with_dtype_convert,
    Dtype.float16.value: fake_quantize_with_dtype_convert,
    Dtype.fp4.value: fake_quantize_fp4_fp6,
    Dtype.fp6_e3m2.value: fake_quantize_fp4_fp6,
    Dtype.fp6_e2m3.value: fake_quantize_fp4_fp6,
}

DTYPE_TO_NON_SCALED_FAKE_QUANTIZE_FUNCS = {
    Dtype.bfp16.value: fake_quantize_bfp16,
    Dtype.mx.value: fake_quantize_mx,
    Dtype.mx6.value: partial(fake_quantize_mx6_mx9, quant_bit=5),
    Dtype.mx9.value: partial(fake_quantize_mx6_mx9, quant_bit=8),
}

DTYPE_TO_REAL_QUANTIZE_FUNCS = {
    Dtype.int2.value: real_quantize_int,
    Dtype.int3.value: real_quantize_int,
    Dtype.int4.value: real_quantize_int,
    Dtype.uint4.value: real_quantize_int,
    Dtype.int8.value: real_quantize_int,
    Dtype.uint8.value: real_quantize_int,
    Dtype.fp8_e4m3.value: real_quantize_fp8_e4m3,
    Dtype.fp8_e5m2.value: real_quantize_fp8_e5m2,
    Dtype.bfloat16.value: real_quantize_with_dtype_convert,
    Dtype.float16.value: real_quantize_with_dtype_convert,
    Dtype.fp4.value: real_quantize_fp4_fp6_per_group,
    Dtype.fp6_e2m3.value: real_quantize_fp4_fp6_per_group,
    Dtype.fp6_e3m2.value: real_quantize_fp4_fp6_per_group,
}
