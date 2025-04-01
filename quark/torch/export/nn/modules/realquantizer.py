#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import quark.torch.kernel  # noqa
from typing import Optional, Tuple
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from quark.torch.quantization.config.config import QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.utils import calculate_qmin_qmax
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase
from quark.torch.utils.pack import create_pack_method
from quark.torch.export.constants import INT_QUANT_DTYPES


class RealQuantizerBase(ABC, nn.Module):

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def pack_zero_point(self) -> None:
        pass

    @abstractmethod
    def maybe_transpose_scale(self) -> None:
        pass

    @staticmethod
    def from_fake_quantizer(qspec: QuantizationSpec,
                            quantizer: Optional[FakeQuantizeBase],
                            reorder: bool,
                            real_quantized: bool,
                            float_dtype: torch.dtype,
                            device: Optional[torch.device] = torch.device("cuda"),
                            scale_shape: Optional[Tuple[int, ...]] = None,
                            zero_point_shape: Optional[Tuple[int, ...]] = None) -> "RealQuantizerBase":
        if qspec.dtype in [Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.bfp16]:
            qspec.is_dynamic = False
            return NonScaledRealQuantizer(qspec=qspec,
                                          quantizer=quantizer,
                                          reorder=reorder,
                                          real_quantized=False,
                                          device=device,
                                          float_dtype=float_dtype)
        else:
            return ScaledRealQuantizer(qspec=qspec,
                                       quantizer=quantizer,
                                       reorder=reorder,
                                       real_quantized=real_quantized,
                                       float_dtype=float_dtype,
                                       device=device,
                                       scale_shape=scale_shape,
                                       zero_point_shape=zero_point_shape)


class ScaledRealQuantizer(RealQuantizerBase):
    '''
    On export, performs transpose on scale and pack on zeropint. Called by parent class, performs real quantization on weight, bias.
    On import, performs dequantization of weight, bias, and fakequantization of input, output via forward method.
    '''

    def __init__(self,
                 qspec: QuantizationSpec,
                 quantizer: Optional[FakeQuantizeBase],
                 reorder: bool,
                 real_quantized: bool,
                 float_dtype: torch.dtype,
                 device: Optional[torch.device] = torch.device("cuda"),
                 scale_shape: Optional[Tuple[int, ...]] = None,
                 zero_point_shape: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__()
        self.qspec = qspec
        self.reorder = reorder
        self.real_quantized = real_quantized
        self.scale_shape = scale_shape
        self.zero_point_shape = zero_point_shape
        self.device = device
        self.float_dtype = float_dtype
        self.transpose_scale: bool = False
        if quantizer is None:
            quant_torch_dtype = self.qspec.dtype.to_torch_packed_dtype()
            if self.scale_shape is not None:
                self.register_parameter(
                    "scale",
                    torch.nn.Parameter(torch.empty(self.scale_shape, device=self.device, dtype=float_dtype),
                                       requires_grad=False))
            else:
                self.register_parameter(
                    "scale",
                    torch.nn.Parameter(torch.empty((), device=self.device, dtype=float_dtype), requires_grad=False))
            self.zero_point = None
            if self.qspec.dtype in INT_QUANT_DTYPES:
                if self.zero_point_shape is not None:
                    self.zero_point = torch.nn.Parameter(torch.empty(self.zero_point_shape,
                                                                     device=self.device,
                                                                     dtype=quant_torch_dtype),
                                                         requires_grad=False)
                else:
                    self.zero_point = torch.nn.Parameter(torch.empty((), device=self.device, dtype=quant_torch_dtype),
                                                         requires_grad=False)

            if self.qspec.qscheme == QSchemeType.per_group and self.qspec.dtype in [Dtype.int4, Dtype.uint4]:
                self.transpose_scale = True
            else:
                self.transpose_scale = False

            if self.qspec.dtype is Dtype.mx:  # pragma: no cover
                assert self.qspec.mx_element_dtype is not None
                self.quant_min, self.quant_max = calculate_qmin_qmax(self.qspec.mx_element_dtype)
            elif self.qspec.dtype in [Dtype.mx6, Dtype.mx9]:  # pragma: no cover
                self.quant_min = self.quant_max = 0.0
            else:
                self.quant_min, self.quant_max = calculate_qmin_qmax(self.qspec.dtype)
        else:
            # TODO: check here
            # self.scale = torch.nn.Parameter(quantizer.scale.to(torch.float), requires_grad=False)
            self.scale = torch.nn.Parameter(quantizer.scale, requires_grad=False)
            self.zero_point = torch.nn.Parameter(quantizer.zero_point.to(torch.int),
                                                 requires_grad=False) if self.qspec.dtype in INT_QUANT_DTYPES else None
            self.quant_min, self.quant_max = calculate_qmin_qmax(self.qspec.dtype)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # If real_quantized, unpack and dequantize tensor.
        # If not real_quantized, unpack and fakequantize tensor.
        pack_method = create_pack_method(
            qscheme=self.qspec.qscheme.value,  # type: ignore[union-attr]
            dtype=self.qspec.dtype.value)

        if self.real_quantized:
            # for weight, bias

            X = pack_method.unpack(
                X, self.reorder,
                **({
                    "origin_packed_axis_size": self.scale.shape[-1]
                } if self.scale.shape != torch.Size([]) else {}))

            # Some quantization schemes do not use `zero_point`, hence initialized to `None`.
            zero_point = None

            if hasattr(self, "zero_point") and self.zero_point is not None:
                zero_point = pack_method.unpack(
                    self.zero_point, self.reorder,
                    **({
                        "origin_packed_axis_size": self.scale.shape[-1]
                    } if self.scale.shape != torch.Size([]) else {}))

            if self.transpose_scale:
                # transpose_scale of bias is always false in qparamslinear.py
                scale = self.scale.data.t().contiguous()
            else:
                scale = self.scale
            X = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                self.qspec.dtype.value, X, scale, zero_point, self.qspec.ch_axis, self.qspec.group_size,
                self.qspec.qscheme.value)  # type: ignore[union-attr]
        else:
            zero_point = None

            if hasattr(self, "zero_point") and self.zero_point is not None:
                zero_point = pack_method.unpack(
                    self.zero_point, self.reorder,
                    **({
                        "origin_packed_axis_size": self.scale.shape[-1]
                    } if self.scale.shape != torch.Size([]) else {}))

            if self.transpose_scale:
                scale = self.scale.data.t().contiguous()
            else:
                scale = self.scale
            X = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
                self.qspec.dtype.value,
                X,
                scale,
                zero_point,
                self.qspec.ch_axis,
                self.qspec.group_size,
                self.quant_min,
                self.quant_max,
                getattr(self.qspec.round_method, "value", None),
                self.qspec.qscheme.value,  # type: ignore[union-attr]
                None)
        return X

    def to_real_quantize_params(self, param: torch.Tensor) -> torch.Tensor:
        '''
        Quantize weight and bias on low-bit precision datatypes, and pack them if required.
        '''
        dtype = self.qspec.dtype.value
        ch_axis = self.qspec.ch_axis
        group_size = self.qspec.group_size
        round_method = getattr(self.qspec.round_method, "value", None)
        qscheme_str_name = getattr(self.qspec.qscheme, "value", None)
        quant_min = self.quant_min
        quant_max = self.quant_max
        scale = self.scale
        zero_point = self.zero_point
        w_res = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
            dtype, param if param is None else param.cpu(), scale if scale is None else scale.cpu(),
            zero_point if zero_point is None else zero_point.cpu(), ch_axis, group_size, quant_min, quant_max,
            round_method, qscheme_str_name)
        weight_pack = create_pack_method(qscheme_str_name, self.qspec.dtype.value)
        w_res = weight_pack.pack(w_res, self.reorder)
        return w_res

    # Pack zero point
    def pack_zero_point(self) -> None:
        if self.zero_point is not None and self.qspec and hasattr(self.qspec, "dtype"):
            qscheme_str_name = getattr(self.qspec.qscheme, "value", None)
            zero_point_pack = create_pack_method(qscheme_str_name, self.qspec.dtype.value)
            self.zero_point = nn.Parameter(zero_point_pack.pack(self.zero_point, self.reorder), requires_grad=False)

    # Try to transpose scale
    def maybe_transpose_scale(self) -> None:
        if getattr(self.qspec.dtype, "value", None) in ["int8", "uint8", "int4", "uint4"]:
            if self.scale.ndim > 2:
                raise ValueError("Only supports self.scale with dimensions not greater than 2.")
            if getattr(self.qspec.qscheme, "value", None) == "per_group":
                self.scale.data = self.scale.data.t().contiguous()


class NonScaledRealQuantizer(RealQuantizerBase):
    '''
    On export, performs transpose on scale and pack on zeropint. Called by parent class, performs real quantization on weight, bias.
    On import, performs dequantization of weight, bias, and fakequantization of input, output via forward method.
    '''

    def __init__(self,
                 qspec: QuantizationSpec,
                 quantizer: Optional[FakeQuantizeBase],
                 reorder: bool,
                 real_quantized: bool,
                 float_dtype: torch.dtype,
                 device: Optional[torch.device] = torch.device("cuda"),
                 scale_shape: Optional[Tuple[int, ...]] = None,
                 zero_point_shape: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__()
        self.qspec = qspec
        self.reorder = reorder
        self.real_quantized = real_quantized
        self.device = device
        self.float_dtype = float_dtype
        self.transpose_scale: bool = False  # For consistency

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # If real_quantized, unpack and dequantize tensor.
        # If not real_quantized, unpack and fakequantize tensor.
        pack_method = create_pack_method(
            qscheme=self.qspec.qscheme.value,  # type: ignore[union-attr]
            dtype=self.qspec.dtype.value)

        if self.real_quantized:
            # for weight, bias
            X = pack_method.unpack(X, self.reorder)
            # Some quantization schemes do not use `zero_point`, hence initialized to `None`.
            zero_point = None

            if hasattr(self, "zero_point") and self.zero_point is not None:
                zero_point = pack_method.unpack(self.zero_point, self.reorder)

            if self.transpose_scale:
                # transpose_scale of bias is always false in qparamslinear.py
                scale = self.scale.data.t().contiguous()
            else:
                scale = self.scale
            X = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                self.qspec.dtype.value, X, scale, zero_point, self.qspec.ch_axis, self.qspec.group_size,
                self.qspec.qscheme.value)  # type: ignore[union-attr]
        else:
            zero_point = None

            if hasattr(self, "zero_point") and self.zero_point is not None:
                zero_point = pack_method.unpack(self.zero_point, self.reorder)

            if self.transpose_scale:
                scale = self.scale.data.t().contiguous()
            else:
                scale = self.scale
            X = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
                self.qspec.dtype.value,
                X,
                scale,
                zero_point,
                self.qspec.ch_axis,
                self.qspec.group_size,
                self.quant_min,
                self.quant_max,
                getattr(self.qspec.round_method, "value", None),
                self.qspec.qscheme.value,  # type: ignore[union-attr]
                None)
        return X

    def to_real_quantize_params(self, param: torch.Tensor) -> torch.Tensor:
        '''
        Quantize weight and bias on low-bit precision datatypes, and pack them if required.
        '''
        dtype = self.qspec.dtype.value
        assert self.qspec.mx_element_dtype is not None
        mx_element_dtype = self.qspec.mx_element_dtype.value
        axis = self.qspec.ch_axis
        block_size = self.qspec.group_size
        qscheme_str_name = getattr(self.qspec.qscheme, "value", None)
        w_res = quark.torch.kernel.non_scaled_real_quantize(  # type: ignore[attr-defined]
            param if param is None else param.cpu(), dtype, mx_element_dtype, axis, block_size)
        weight_pack = create_pack_method(qscheme_str_name, self.qspec.dtype.value)
        w_res = weight_pack.pack(w_res, self.reorder)
        return w_res

    def pack_zero_point(self) -> None:
        pass

    def maybe_transpose_scale(self) -> None:
        pass
