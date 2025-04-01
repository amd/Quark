#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# type: ignore

import os
import torch
from torch.library import Library, impl
from types import ModuleType
from typing import Any, List, Optional
from torch import ops  # type: ignore[attr-defined]
from .hw_emulation import hw_emulation_interface
from .hw_emulation.extensions import kernel_ext
from torch.autograd import Function
from typing import Any, Union
from torch.onnx._internal import jit_utils
from torch.onnx import errors, symbolic_helper
import torch._C._onnx as _C_onnx


class QuantE4M3Function(Function):

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, scale: Union[float, None] = None) -> Any:  # type: ignore
        if scale is None:
            return ops.quark.quant_fp8_e4m3(inputs)
        else:
            return ops.quark.quant_fp8_e4m3_with_scale(inputs, scale)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:  # type: ignore # pragma: no cover
        return grad_outputs, None


class DequantE4M3Function(Function):

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, scale: Union[float, None] = None) -> Any:  # type: ignore
        if scale is None:
            return ops.quark.dequant_fp8_e4m3(inputs)
        else:
            return ops.quark.dequant_fp8_e4m3_with_scale(inputs, scale)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:  # type: ignore # pragma: no cover
        return grad_outputs, None


quant_fp8_e4m3 = QuantE4M3Function.apply
dequant_fp8_e4m3 = DequantE4M3Function.apply


class QuantE5M2Function(Function):

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, scale: Union[float, None] = None) -> Any:  # type: ignore
        if scale is None:
            return ops.quark.quant_fp8_e5m2(inputs)
        else:
            return ops.quark.quant_fp8_e5m2_with_scale(inputs, scale)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:  # type: ignore # pragma: no cover
        return grad_outputs, None


class DequantE5M2Function(Function):

    @staticmethod
    def forward(ctx: Any, inputs: torch.Tensor, scale: Union[float, None] = None) -> Any:  # type: ignore
        if scale is None:
            return ops.quark.dequant_fp8_e5m2(inputs)
        else:
            return ops.quark.dequant_fp8_e5m2_with_scale(inputs, scale)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:  # type: ignore # pragma: no cover
        return grad_outputs, None


quant_fp8_e5m2 = QuantE5M2Function.apply
dequant_fp8_e5m2 = DequantE5M2Function.apply


class ScaledFakeQuantizeFunction(Function):

    @staticmethod
    def forward(ctx: Any, quant_dtype: str, inputs: torch.Tensor, scale: torch.Tensor,
                zero_point: Optional[torch.Tensor], axis: Optional[int], group_size: Optional[int],
                quant_min: Union[int, float], quant_max: Union[int, float], round_mode: Optional[int],
                qscheme: Optional[str], mx_element_dtype: Optional[str]) -> torch.Tensor:

        # Default value setting
        zero_point = zero_point if zero_point is not None else torch.Tensor([])  # Set illegal value for zero_point
        axis = axis if axis is not None else 1  # Set same default value as ONNX QuantLinear
        group_size = group_size if group_size is not None else 1  # Set same default value as ONNX QuantLinear
        round_mode = round_mode if round_mode is not None else -1  # Set illegal value for round_mode
        qscheme = qscheme if qscheme is not None else 'None'
        mx_element_dtype = 'None' if mx_element_dtype is None else mx_element_dtype

        return ops.quark.scaled_fake_quantize(quant_dtype, inputs, scale, zero_point, axis, group_size, quant_min,
                                              quant_max, round_mode, qscheme, mx_element_dtype)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:
        return None, grad_outputs, None, None, None, None, None, None, None, None, None

    @staticmethod
    @symbolic_helper.parse_args("s", "v", "v", "v", "i", "i", "i", "i", "i", "s", "s")
    def symbolic(g: jit_utils.GraphContext,
                 quant_dtype: str,
                 inputs: torch.Tensor,
                 scale: torch.Tensor,
                 zero_point: torch.Tensor = None,
                 axis: int = None,
                 group_size: int = None,
                 quant_min: int = None,
                 quant_max: int = None,
                 round_mode: Union[int, None] = None,
                 qscheme: Union[str, None] = None,
                 mx_element_dtype: Union[str, None] = None) -> torch.Value:

        if quant_dtype == 'mx':
            raise NotImplementedError("Exporting MX datatypes to ONNX is not supported yet.")  # pragma: no cover
        if quant_dtype in ['mx6', 'mx9']:
            raise NotImplementedError(
                "Exporting MX6 or MX9 datatypes to ONNX is not supported yet.")  # pragma: no cover

        if quant_dtype == 'fp8_e4m3':
            zero_point = torch.tensor(0, dtype=torch.float8_e4m3fn)
            quantized = g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis)
            return g.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis)
        elif quant_dtype == 'fp8_e5m2':
            zero_point = torch.tensor(0, dtype=torch.float8_e5m2)
            quantized = g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis)
            return g.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis)
        elif quant_dtype in ["int4", "uint4", "int8", "uint8"]:
            if (quant_min, quant_max) not in [(0, 255), (-128, 127), (-8, 7), (0, 15)]:
                raise errors.SymbolicValueError(
                    "For (quant_min, quant_max), ONNX allows only (0, 255), (-128, 127), (-8, 7) and (0, 15). "
                    f"Got ({quant_min}, {quant_max})", )
            if quant_min == 0:
                zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.UINT8)
            else:
                zero_point = g.op("Cast", zero_point, to_i=_C_onnx.TensorProtoDataType.INT8)
            quantized = g.op("QuantizeLinear", inputs, scale, zero_point, axis_i=axis)
            return g.op("DequantizeLinear", quantized, scale, zero_point, axis_i=axis)
        elif quant_dtype in ["bfloat16", "float16"]:
            dequant_torch_dtype = {
                torch.float32: _C_onnx.TensorProtoDataType.FLOAT,
                torch.float16: _C_onnx.TensorProtoDataType.FLOAT16,
                torch.bfloat16: _C_onnx.TensorProtoDataType.BFLOAT16,
            }.get(inputs.type().dtype())
            quant_torch_dtype = {
                "bfloat16": _C_onnx.TensorProtoDataType.BFLOAT16,
                "float16": _C_onnx.TensorProtoDataType.FLOAT16,
                "float32": _C_onnx.TensorProtoDataType.FLOAT
            }.get(quant_dtype)
            quantized = g.op("Cast", inputs, to_i=quant_torch_dtype)
            return g.op("Cast", quantized, to_i=dequant_torch_dtype)
        else:
            raise ValueError(f"Unsupported mode: {quant_dtype}")


scaled_fake_quantize = ScaledFakeQuantizeFunction.apply


class NonScaledFakeQuantizeFunction(Function):

    @staticmethod
    def forward(ctx: Any, input_tensor: torch.Tensor, quant_dtype: str, mx_element_dtype: str, axis: int,
                block_size: int) -> torch.Tensor:
        return ops.quark.non_scaled_fake_quantize(input_tensor=input_tensor,
                                                  quant_dtype=quant_dtype,
                                                  mx_element_dtype=mx_element_dtype,
                                                  axis=axis,
                                                  block_size=block_size)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:
        return grad_outputs, None, None, None, None

    @staticmethod
    @symbolic_helper.parse_args("s", "v")
    def symbolic(g: jit_utils.GraphContext, quant_dtype: str, inputs: torch.Tensor) -> torch.Value:
        return None


non_scaled_fake_quantize = NonScaledFakeQuantizeFunction.apply


class ScaledRealQuantizeFunction(Function):

    @staticmethod
    def forward(ctx: Any, quant_dtype: str, inputs: torch.Tensor, scale: torch.Tensor, zero_point: Union[torch.Tensor,
                                                                                                         None],
                axis: Union[int, None], group_size: Union[int, None], quant_min: Union[int, float],
                quant_max: Union[int, float], round_mode: Union[int, None], qscheme: Union[str, None]) -> torch.Tensor:

        # Default value setting
        zero_point = zero_point if zero_point is not None else torch.Tensor([])  # Set illegal value for zero_point
        axis = axis if axis is not None else 1  # Set same default value as ONNX QuantLinear
        group_size = group_size if group_size is not None else 1  # Set same default value as ONNX QuantLinear
        round_mode = round_mode if round_mode is not None else -1  # Set illegal value for round_mode
        qscheme = qscheme if qscheme is not None else 'None'

        return ops.quark.scaled_real_quantize(quant_dtype, inputs, scale, zero_point, axis, group_size, quant_min,
                                              quant_max, round_mode, qscheme)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:
        return None, grad_outputs, None, None, None, None, None, None, None, None


scaled_real_quantize = ScaledRealQuantizeFunction.apply


class NonScaledRealQuantizeFunction(Function):

    @staticmethod
    def forward(ctx: Any, input_tensor: torch.Tensor, quant_dtype: str, mx_element_dtype: str, axis: int,
                block_size: int) -> torch.Tensor:

        return ops.quark.non_scaled_real_quantize(input_tensor, quant_dtype, mx_element_dtype, axis, block_size)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:
        return None, grad_outputs


non_scaled_real_quantize = NonScaledRealQuantizeFunction.apply


class DeQuantizeFunction(Function):

    @staticmethod
    def forward(ctx: Any, quant_dtype: str, inputs: torch.Tensor, scale: torch.Tensor, zero_point: Union[torch.Tensor,
                                                                                                         None],
                axis: Union[int, None], group_size: Union[int, None], qscheme: Union[str, None]) -> torch.Tensor:

        # Default value setting
        zero_point = zero_point if zero_point is not None else torch.Tensor([])  # Set illegal value for zero_point
        axis = axis if axis is not None else 1  # Set same default value as ONNX QuantLinear
        group_size = group_size if group_size is not None else 1  # Set same default value as ONNX QuantLinear
        qscheme = qscheme if qscheme is not None else 'None'

        return ops.quark.dequantize(quant_dtype, inputs, scale, zero_point, axis, group_size, qscheme)

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:
        return None, grad_outputs, None, None, None, None, None


dequantize = DeQuantizeFunction.apply


class TQTQuantize(Function):

    @staticmethod
    def forward(  # type: ignore
            ctx: Any, inputs: torch.Tensor, log_threshold: torch.Tensor, zero_point: torch.Tensor, domain: torch.Tensor,
            round_mode: int) -> Any:
        scale = 2**(torch.ceil(log_threshold)) / domain
        quant_max = domain - 1
        quant_min = -domain
        ctx.save_for_backward(inputs, scale, quant_max, quant_min, log_threshold)
        if kernel_ext is not None and inputs.device != torch.device("cpu"):
            return kernel_ext.fake_quantize_per_tensor_affine(inputs, scale, zero_point, quant_min, quant_max,
                                                              round_mode)
        else:
            return torch.fake_quantize_per_tensor_affine(inputs, scale, zero_point, int(quant_min.item()),
                                                         int(quant_max.item()))

    @staticmethod
    def backward(ctx: Any, grad_outputs: torch.Tensor) -> Any:  # type: ignore # pragma: no cover
        inputs, scale, quant_max, quant_min, log_threshold = ctx.saved_tensors
        grad_inputs, grad_log_threshold = kernel_ext.tqt_backward(inputs, scale, quant_max, quant_min, log_threshold,
                                                                  grad_outputs)
        return grad_inputs, grad_log_threshold, None, None, None


tqt_quantize = TQTQuantize.apply


class LSQQuantize(Function):

    @staticmethod
    def forward(
            ctx: Any,  # type: ignore
            x: torch.tensor,
            scale: torch.tensor,
            zero_point: torch.tensor,
            grad_factor: float,
            quant_min: int,
            quant_max: int,
            ch_axis: Optional[int] = None,
            rounding_key: int = 8) -> Any:
        ctx.save_for_backward(x, scale, zero_point)
        ctx.others = quant_min, quant_max, ch_axis, grad_factor

        if ch_axis is not None:
            # Per channel
            result = hw_emulation_interface.fake_quantize_int_per_channel_affine(x, scale, zero_point, ch_axis,
                                                                                 quant_min, quant_max, rounding_key)
        else:
            result = kernel_ext.fake_quantize_per_tensor_affine(x, scale, zero_point, quant_min, quant_max,
                                                                rounding_key)
        return result

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:  # type: ignore # pragma: no cover
        x, scale, zero_point = ctx.saved_tensors
        quant_min, quant_max, ch_axis, grad_factor = ctx.others

        if ch_axis is not None:
            scaled_x = x.transpose(ch_axis, -1) / scale + zero_point
            scaled_x = scaled_x.transpose(ch_axis, -1)
            dims = [i for i in range(x.dim()) if i != ch_axis]
        else:
            scaled_x = x / scale + zero_point
            dims = [i for i in range(x.dim())]

        rounded_scaled_x = torch.round(scaled_x)
        is_lt_min = (rounded_scaled_x < quant_min).float()
        is_gt_max = (rounded_scaled_x > quant_max).float()
        is_ge_min_and_le_max = torch.ones(is_lt_min.shape, device=is_lt_min.device) - is_lt_min - is_gt_max

        grad_x = is_ge_min_and_le_max * grad_output
        grad_scale = ((is_lt_min * quant_min + is_gt_max * quant_max + is_ge_min_and_le_max *
                       (rounded_scaled_x - scaled_x)) * grad_output * grad_factor)
        # For channel-only tensor like conv2d's bias we don't do sum computation.
        if dims:
            # torch.sum() will return a scalar tensor if the computation is performed
            # on whole dims of the tensor. Use unsequeeze to make it a normal tensor.
            grad_scale = grad_scale.sum(dims).unsqueeze(0)
        return grad_x, grad_scale, None, None, None, None, None, None


lsq_quantize = LSQQuantize.apply
