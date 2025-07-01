#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import quark.torch.kernel  # noqa
from typing import Optional, Tuple, Union, List
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from quark.torch.quantization.config.config import QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.utils import calculate_qmin_qmax
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, SequentialQuantize
from quark.torch.utils.pack import create_pack_method
from quark.torch.quantization.constants import INT_QUANT_DTYPES
from quark.torch.quantization.observer.tqt_observer import TQTObserver
from quark.torch.quantization.observer.lsq_observer import LSQObserver
from quark.torch.quantization.observer.observer import ObserverBase, PlaceholderObserver
from quark.torch.quantization.utils import get_num_bits
from quark.torch.quantization.config.type import ZeroPointType
from quark.torch.quantization import debug
from torch.distributed._tensor.experimental import implicit_replication
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class RealQuantizerBase(ABC, nn.Module):

    def __init__(self, qspec: QuantizationSpec) -> None:
        super().__init__()
        self.qspec = qspec
        self.is_dynamic = qspec.is_dynamic
        self.is_scale_quant = qspec.is_scale_quant

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def pack_zero_point(self) -> None:
        pass

    @abstractmethod
    def maybe_convert_and_transpose_scale(self) -> None:
        pass

    @abstractmethod
    def to_real_quantize_params(self, param: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def has_static_scale(self) -> bool:
        pass

    def update_dynamic_params(self, X: torch.Tensor) -> None:
        pass

    @abstractmethod
    def unpack_tensor(self, X: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def unpack_params(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        pass


class StaticRealQuantizer(RealQuantizerBase, ABC):

    def __init__(self,
                 qspec: QuantizationSpec,
                 quantizer: Optional[FakeQuantizeBase],
                 reorder: bool,
                 real_quantized: bool,
                 float_dtype: torch.dtype,
                 device: Optional[torch.device] = torch.device("cuda"),
                 scale_shape: Optional[Tuple[int, ...]] = None,
                 zero_point_shape: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__(qspec)
        self.reorder = reorder
        self.real_quantized = real_quantized
        self.device = device
        self.float_dtype = float_dtype
        self.transpose_scale: bool = False
        self.scale_shape = scale_shape
        self.zero_point_shape = zero_point_shape
        self.pack_method = create_pack_method(qscheme=getattr(self.qspec.qscheme, "value", None),
                                              dtype=self.qspec.dtype.value,
                                              mx_element_dtype=getattr(self.qspec.mx_element_dtype, "value", None))

        if self.qspec.qscheme == QSchemeType.per_group and self.qspec.dtype in [
                Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8, Dtype.int2
        ]:
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

    def unpack_tensor(self, X: torch.Tensor) -> torch.Tensor:
        unpacked_X = self.pack_method.unpack(
            X, self.reorder,
            **({
                "origin_packed_axis_size": self.scale.shape[-1]
            } if self.scale.shape != torch.Size([]) else {}))
        return unpacked_X

    def unpack_params(self) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        zero_point = None
        if getattr(self, "zero_point", None) is not None:
            zero_point = self.pack_method.unpack(
                self.zero_point, self.reorder,
                **({
                    "origin_packed_axis_size": self.scale.shape[-1]
                } if self.scale.shape != torch.Size([]) else {}))

        if self.transpose_scale:
            # transpose_scale of bias is always false in qparamslinear.py
            scale = self.scale.data.t().contiguous()
        else:
            scale = self.scale

        if getattr(self.qspec, "scale_format", None) == "e8m0":
            scale = 2**(scale.view(torch.uint8).to(torch.int16) - 127).to(self.float_dtype)

        return scale, zero_point

    def has_static_scale(self) -> bool:
        return True


class StaticScaledRealQuantizer(StaticRealQuantizer):
    '''
    On export, performs transpose on scale and pack on zeropint. Called by parent class, performs real quantization on weight, bias.
    On import, performs dequantization of weight, bias, and fakequantization of input, output via forward method.
    '''
    scale: torch.Tensor

    def __init__(self,
                 qspec: QuantizationSpec,
                 quantizer: Optional[FakeQuantizeBase],
                 reorder: bool,
                 real_quantized: bool,
                 float_dtype: torch.dtype,
                 device: Optional[torch.device] = torch.device("cuda"),
                 scale_shape: Optional[Tuple[int, ...]] = None,
                 zero_point_shape: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__(qspec, quantizer, reorder, real_quantized, float_dtype, device, scale_shape, zero_point_shape)
        if quantizer is None:
            quant_torch_dtype = self.qspec.dtype.to_torch_packed_dtype()
            if self.scale_shape is not None:
                self.register_buffer("scale", torch.empty(self.scale_shape, device=self.device, dtype=float_dtype))
            else:
                self.register_buffer("scale", torch.empty((), device=self.device, dtype=float_dtype))
            # self.zero_point = None
            if self.qspec.dtype in INT_QUANT_DTYPES:
                if self.zero_point_shape is not None:
                    self.register_buffer(
                        "zero_point", torch.empty(self.zero_point_shape, device=self.device, dtype=quant_torch_dtype))
                else:
                    self.register_buffer("zero_point", torch.empty((), device=self.device, dtype=quant_torch_dtype))
        else:
            # TODO: check here
            self.register_buffer("scale", quantizer.scale)
            if self.qspec.dtype in INT_QUANT_DTYPES:
                self.register_buffer("zero_point", quantizer.zero_point.to(torch.int))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # If real_quantized, unpack and dequantize tensor.
        # If not real_quantized, unpack and fakequantize tensor.

        if self.real_quantized:
            # for weight, bias
            X = self.unpack_tensor(X)
            scale, zero_point = self.unpack_params()
            with implicit_replication():
                X = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                    self.qspec.dtype.value, X, scale, zero_point, self.qspec.ch_axis, self.qspec.group_size,
                    self.qspec.qscheme.value)  # type: ignore[union-attr]
        else:
            # X = self.unpack_tensor(X)
            scale, zero_point = self.unpack_params()
            with implicit_replication():
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
        zero_point = getattr(self, "zero_point", None)
        if scale.device != param.device:
            scale = scale.to(param.device)
        if zero_point is not None and zero_point.device != param.device:
            zero_point = zero_point.to(param.device)
        w_res = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
            dtype, param, scale, zero_point, ch_axis, group_size, quant_min, quant_max, round_method, qscheme_str_name)
        w_res = self.pack_method.pack(w_res, self.reorder)
        w_res = w_res.to('cpu')
        torch.cuda.empty_cache()
        return w_res

    def to_fake_quantize_params(self, param: torch.Tensor) -> torch.Tensor:
        '''
        Fake quantize weight and bias
        '''
        dtype = self.qspec.dtype.value
        ch_axis = self.qspec.ch_axis
        group_size = self.qspec.group_size
        round_method = getattr(self.qspec.round_method, "value", None)
        qscheme_str_name = getattr(self.qspec.qscheme, "value", None)
        quant_min = self.quant_min
        quant_max = self.quant_max
        scale = self.scale
        zero_point = getattr(self, "zero_point", None)

        res = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
            dtype, param, scale, zero_point, ch_axis, group_size, quant_min, quant_max, round_method, qscheme_str_name,
            None)
        assert isinstance(res, torch.Tensor), "res must be a torch.Tensor!"
        return res

    # Pack zero point
    def pack_zero_point(self) -> None:
        if getattr(self, "zero_point", None) is not None and self.qspec and hasattr(self.qspec, "dtype"):
            self.zero_point: torch.Tensor = self.pack_method.pack(self.zero_point, self.reorder)

    # Try to convert scale to int8 and transpose scale
    def maybe_convert_and_transpose_scale(self) -> None:
        if getattr(self.qspec, "scale_format", None) == "e8m0":
            self.scale = (torch.log2(self.scale).round().to(torch.int16).clamp(-127, 127) + 127).to(torch.uint8)

        if getattr(self.qspec.dtype, "value", None) in ["int8", "uint8", "int4", "uint4", "int2"]:
            if self.scale.ndim > 2:
                raise ValueError("Only supports self.scale with dimensions not greater than 2.")
            if getattr(self.qspec.qscheme, "value", None) == "per_group":
                self.scale = self.scale.t().contiguous()


class StaticNonScaledRealQuantizer(StaticRealQuantizer):
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
        super().__init__(qspec, quantizer, reorder, real_quantized, float_dtype, device, scale_shape, zero_point_shape)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward method for StaticNonScaledRealQuantizer is not supported currently.")

    def to_real_quantize_params(self, param: torch.Tensor) -> torch.Tensor:
        '''
        Quantize weight and bias on low-bit precision datatypes, and pack them if required.
        '''
        dtype = self.qspec.dtype.value
        assert self.qspec.mx_element_dtype is not None
        mx_element_dtype = self.qspec.mx_element_dtype.value
        axis = self.qspec.ch_axis
        block_size = self.qspec.group_size
        w_res = quark.torch.kernel.non_scaled_real_quantize(  # type: ignore[attr-defined]
            param, dtype, mx_element_dtype, axis, block_size)
        w_res = self.pack_method.pack(w_res, self.reorder)
        w_res = w_res.to('cpu')
        torch.cuda.empty_cache()
        return w_res

    def pack_zero_point(self) -> None:
        pass

    def maybe_convert_and_transpose_scale(self) -> None:
        pass


class DynamicScaledQuantizer(RealQuantizerBase):

    def __init__(self,
                 qspec: QuantizationSpec,
                 quantizer: Optional[FakeQuantizeBase] = None,
                 device: Optional[torch.device] = torch.device("cuda"),
                 float_dtype: Optional[torch.dtype] = None,
                 scale_shape: Optional[Tuple[int, ...]] = None,
                 zero_point_shape: Optional[Tuple[int, ...]] = None) -> None:
        super().__init__(qspec)
        self.device = device
        self.float_dtype = float_dtype
        self.scale_shape = scale_shape
        self.zero_point_shape = zero_point_shape
        self.dtype = qspec.dtype
        self.mx_element_dtype = qspec.mx_element_dtype
        self.qscheme = qspec.qscheme
        self.qscheme_str_name = getattr(qspec.qscheme, "value", None)
        self.ch_axis = qspec.ch_axis
        self.group_size = qspec.group_size
        self.symmetric = qspec.symmetric
        self.round_method = getattr(qspec.round_method, "value", None)
        self.scale_type = qspec.scale_type
        self._num_bits = get_num_bits(qspec.dtype)
        self.zero_point_type = qspec.zero_point_type
        self.real_quantized = False

        assert self.dtype not in [Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.bfloat16, Dtype.float16, Dtype.bfp16], \
            "Not supported for mx, mx6, mx9, bfloat16, float16, bfp16 quantization!"
        assert self.zero_point_type == ZeroPointType.int32, "Only support int32 zero point!"

        # For dynamic quantizer, when it is scale per tensor quantizer, the scale and zero point are not None
        # and are registered as buffers. Otherwise, they are None and are not registered as buffers.
        self.register_buffer("scale", None)
        self.register_buffer("zero_point", None)
        if quantizer is not None and self.has_static_scale():
            self.register_buffer("scale", quantizer.scale)
            if quantizer.zero_point is not None and self.qspec.dtype in INT_QUANT_DTYPES:
                self.register_buffer("zero_point", quantizer.zero_point.to(torch.int))
        else:
            quant_torch_dtype = self.qspec.dtype.to_torch_packed_dtype()
            if self.has_static_scale():
                if self.scale_shape is not None:
                    self.register_buffer("scale",
                                         torch.empty(self.scale_shape, device=self.device, dtype=self.float_dtype))
                else:
                    self.register_buffer("scale", torch.empty((), device=self.device, dtype=self.float_dtype))
                if self.qspec.dtype in INT_QUANT_DTYPES:
                    if self.zero_point_shape is not None:
                        self.register_buffer(
                            "zero_point", torch.empty(self.zero_point_shape,
                                                      device=self.device,
                                                      dtype=quant_torch_dtype))
                    else:
                        self.register_buffer("zero_point", torch.empty((), device=self.device, dtype=quant_torch_dtype))

        self.quant_min, self.quant_max = calculate_qmin_qmax(self.dtype)

    @staticmethod
    def create_observer(quant_spec: QuantizationSpec, device: Optional[torch.device] = None) -> ObserverBase:
        if quant_spec.observer_cls is not None:
            return quant_spec.observer_cls(quant_spec, device)
        else:
            return PlaceholderObserver(quant_spec)

    def calculate_qparams(self, observer: ObserverBase,
                          X: torch.Tensor) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        debug.assert_no_nan(X, "tensor contains NaN!")
        qparams = observer._calculate_qparams()
        scale, zero_point = None, None
        if qparams is not None:
            scale, zero_point = qparams
            debug.assert_no_nan(scale, "scale contains NaN!")
            debug.assert_no_nan(zero_point, "zero_point contains NaN!")
        return scale, zero_point

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        debug.assert_no_nan(X, "input tensor X contains NaN!")
        self.update_dynamic_params(X)
        # Do fake quantize
        mx_element_dtype = None if self.mx_element_dtype is None else self.mx_element_dtype.value
        X_quantized: torch.Tensor = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
            self.dtype.value, X, self.scale,
            self.zero_point.to(torch.int) if self.zero_point is not None else None, self.ch_axis, self.group_size,
            self.quant_min, self.quant_max, self.round_method, self.qscheme_str_name, mx_element_dtype)

        debug.assert_no_nan(X_quantized, "output tensor X contains NaN!")
        return X_quantized

    def update_dynamic_params(self, X: torch.Tensor) -> None:
        if not self.has_static_scale():
            observer = self.create_observer(self.qspec, self.device)
            assert not isinstance(observer, (TQTObserver, LSQObserver)), \
                "Not supported for TQT and LSQ observer!"

            # Do observation
            observer(X.detach())
            self.scale, self.zero_point = self.calculate_qparams(observer, X)

    def has_static_scale(self) -> bool:
        return self.qspec.is_scale_quant and self.qspec.qscheme == QSchemeType.per_tensor

    def to_real_quantize_params(self, param: torch.Tensor) -> torch.Tensor:
        return param

    def to_fake_quantize_params(self, param: torch.Tensor) -> torch.Tensor:
        return param

    def unpack_tensor(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def unpack_params(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        return self.scale, self.zero_point

    def pack_zero_point(self) -> None:
        pass

    def maybe_convert_and_transpose_scale(self) -> None:
        pass


class SequentialRealQuantizer(nn.Sequential):
    """A sequential container for RealQuantizerBase modules.

    This module is used to quantize a tensor in multiple formats sequentially using real quantization.
    It takes RealQuantizerBase modules as input and containerizes them similar to torch.nn.Sequential.

    Args:
        quantizers (RealQuantizerBase): RealQuantizerBase modules to be added to the container.
    """

    def __init__(self, *quantizers: RealQuantizerBase):
        """Initialize SequentialRealQuantizer module."""
        assert not any(not isinstance(q, RealQuantizerBase)
                       for q in quantizers), ("All quantizers must be a RealQuantizerBase.")
        super().__init__(*quantizers)

        assert all(not isinstance(quantizer, StaticNonScaledRealQuantizer) for quantizer in quantizers), \
            "StaticNonScaledRealQuantizer is not supported in SequentialRealQuantizer currently."

        # Verify all quantizers have consistent is_dynamic configuration
        assert all(quantizer.is_dynamic == quantizers[0].is_dynamic for quantizer in quantizers), \
            "The is_dynamic configuration of all quantizers should be the same"
        self.is_dynamic = quantizers[0].is_dynamic

        # Verify all quantizers have consistent real_quantized configuration
        assert all(quantizer.real_quantized == quantizers[0].real_quantized for quantizer in quantizers), \
            "The real_quantized configuration of all quantizers should be the same"

        self.real_quantized = quantizers[0].real_quantized

    def to_real_quantize_params(self, param: torch.Tensor) -> torch.Tensor:
        param_dtype = param.dtype
        for i, module in enumerate(self):
            if not module.is_scale_quant:
                param = param.to(param_dtype)
                if i < len(self) - 1:
                    next_module = self[i + 1]
                    if next_module.is_scale_quant:
                        # if the scale quantizer of current module exists, use the scale quantizer
                        # to fake quantize the scale of current module, as we need the real float scale
                        # to do quantization of current module
                        module.scale = next_module.to_fake_quantize_params(module.scale)
                param = module.to_real_quantize_params(param)
        return param

    def pack_zero_point(self) -> None:
        for module in self:
            module.pack_zero_point()

    def maybe_convert_and_transpose_scale(self) -> None:
        for i, module in enumerate(self):
            if not module.is_scale_quant:
                if i < len(self) - 1 and self[i + 1].is_scale_quant and not self.is_dynamic:
                    # if the scale quantizer of current module exists, use the scale quantizer
                    # to real quantize the scale of current module
                    module.scale = self[i + 1].to_real_quantize_params(module.scale)
                    self[i + 1].maybe_convert_and_transpose_scale()
                module.maybe_convert_and_transpose_scale()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through all quantizers in sequence.

        Args:
            x: Input tensor to be quantized

        Returns:
            Quantized tensor after passing through all quantizers
        """
        if not self.real_quantized:
            for module_index, module in enumerate(self):
                if not module.is_scale_quant:
                    module.update_dynamic_params(x)
                    scale, zero_point = module.unpack_params()
                    if module_index < len(self) - 1:
                        next_module = self[module_index + 1]
                        if next_module.is_scale_quant:
                            next_module.update_dynamic_params(scale)
                            n_scale, n_zero_point = next_module.unpack_params()
                            scale = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
                                next_module.qspec.dtype.value, scale.to(next_module.float_dtype), n_scale,
                                n_zero_point.to(torch.int) if n_zero_point is not None else None,
                                next_module.qspec.ch_axis, next_module.qspec.group_size, next_module.quant_min,
                                next_module.quant_max, next_module.round_method, next_module.qspec.qscheme.value, None)

                    x = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
                        module.qspec.dtype.value, x.to(module.float_dtype), scale, zero_point, module.qspec.ch_axis,
                        module.qspec.group_size, module.quant_min, module.quant_max,
                        getattr(module.qspec.round_method, "value", None), getattr(module.qspec.qscheme, "value", None))

            for module_index, module in enumerate(reversed(self)):
                if not module.is_scale_quant:
                    scale, zero_point = module.unpack_params()
                    if module_index > 0:
                        previous_module = self[len(self) - module_index]
                        if previous_module.is_scale_quant:
                            p_scale, p_zero_point = previous_module.unpack_params()
                            scale = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
                                previous_module.qspec.dtype.value, scale.to(previous_module.float_dtype), p_scale,
                                p_zero_point.to(torch.int) if p_zero_point is not None else None,
                                previous_module.qspec.ch_axis, previous_module.qspec.group_size,
                                previous_module.quant_min, previous_module.quant_max, previous_module.round_method,
                                previous_module.qspec.qscheme.value, None)

                    x = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                        module.qspec.dtype.value, x.to(module.float_dtype), scale, zero_point, module.qspec.ch_axis,
                        module.qspec.group_size, module.qspec.qscheme.value)
        else:
            for module_index, module in enumerate(reversed(self)):
                if not module.is_scale_quant:
                    module.update_dynamic_params(x)
                    x = module.unpack_tensor(x)
                    scale, zero_point = module.unpack_params()
                    if module_index > 0:
                        previous_module = self[len(self) - module_index]
                        if previous_module.is_scale_quant:
                            previous_module.update_dynamic_params(scale)
                            p_scale, p_zero_point = previous_module.unpack_params()
                            scale = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                                previous_module.qspec.dtype.value, scale.to(previous_module.float_dtype), p_scale,
                                p_zero_point, previous_module.qspec.ch_axis, previous_module.qspec.group_size,
                                previous_module.qspec.qscheme.value)
                    x = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                        module.qspec.dtype.value, x.to(module.float_dtype), scale, zero_point, module.qspec.ch_axis,
                        module.qspec.group_size, module.qspec.qscheme.value)
        return x


def get_real_quantizer(
    qspec: Union[QuantizationSpec, List[QuantizationSpec]],
    quantizer: Optional[Union[FakeQuantizeBase, SequentialQuantize]],
    reorder: Optional[bool] = None,
    real_quantized: Optional[bool] = None,
    float_dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = torch.device("cuda"),
    scale_shape: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None,
    zero_point_shape: Optional[Union[Tuple[int, ...], List[Tuple[int, ...]]]] = None
) -> Union[RealQuantizerBase, SequentialRealQuantizer]:
    if isinstance(qspec, list):
        quantizers: List[RealQuantizerBase] = []
        for i, q in enumerate(qspec):
            assert not isinstance(
                quantizer, FakeQuantizeBase), "quantizer must be a SequentialQuantize for sequential real quantizer!"
            q_quantizer = None if quantizer is None else quantizer[len(quantizers)]
            s_shape = scale_shape[i] if scale_shape is not None else None
            assert not isinstance(s_shape, int), "scale_shape must be a list of tuples"
            zp_shape = zero_point_shape[i] if zero_point_shape is not None else None
            assert not isinstance(zp_shape, int), "zero_point_shape must be a list of tuples"
            real_quantizer = get_real_quantizer(qspec=q,
                                                quantizer=q_quantizer,
                                                reorder=reorder,
                                                real_quantized=real_quantized,
                                                float_dtype=float_dtype,
                                                device=device,
                                                scale_shape=s_shape,
                                                zero_point_shape=zp_shape)
            assert isinstance(real_quantizer, RealQuantizerBase), "real_quantizer must be a RealQuantizerBase!"
            quantizers.append(real_quantizer)
        return SequentialRealQuantizer(*quantizers)
    else:
        assert not isinstance(quantizer,
                              SequentialQuantize), "quantizer must be a FakeQuantizeBase for single real quantizer!"
        assert not isinstance(scale_shape, list), "scale_shape must be a int tuple for single real quantizer!"
        assert not isinstance(zero_point_shape, list), "zero_point_shape must be a int tuple for single real quantizer!"
        qspec.is_dynamic = False if qspec.dtype in [Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.bfp16] else qspec.is_dynamic

        if qspec.is_dynamic:
            assert qspec.dtype not in [Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.bfp16, Dtype.bfloat16, Dtype.float16], \
                "Not supported for mx, mx6, mx9, bfp16, bfloat16, float16 quantization in dynamic real quantizer!"
            return DynamicScaledQuantizer(qspec=qspec,
                                          quantizer=quantizer,
                                          device=device,
                                          float_dtype=float_dtype,
                                          scale_shape=scale_shape,
                                          zero_point_shape=zero_point_shape)

        assert reorder is not None, "reorder must be provided for static real quantizer!"
        assert float_dtype is not None, "float_dtype must be provided for static real quantizer!"
        if qspec.dtype in [Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.bfp16]:
            return StaticNonScaledRealQuantizer(qspec=qspec,
                                                quantizer=quantizer,
                                                reorder=reorder,
                                                real_quantized=False,
                                                device=device,
                                                float_dtype=float_dtype)
        else:
            assert real_quantized is not None, "real_quantized must be provided for static real scaled quantizer!"
            return StaticScaledRealQuantizer(qspec=qspec,
                                             quantizer=quantizer,
                                             reorder=reorder,
                                             real_quantized=real_quantized,
                                             float_dtype=float_dtype,
                                             device=device,
                                             scale_shape=scale_shape,
                                             zero_point_shape=zero_point_shape)
