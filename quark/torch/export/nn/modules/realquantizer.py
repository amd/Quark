#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import quark.torch.kernel  # noqa
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from quark.torch.quantization.config.config import QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.utils import calculate_qmin_qmax
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase, SequentialQuantize
from quark.torch.utils import create_pack_method
from quark.torch.utils.numerics import to_e8m0_uint8
from quark.torch.quantization.constants import INT_QUANT_DTYPES, PER_GROUP_INT_TRANSPOSE_DTYPES
from quark.torch.quantization.observer.tqt_observer import TQTObserver
from quark.torch.quantization.observer.lsq_observer import LSQObserver
from quark.torch.quantization.observer.observer import (
    ObserverBase,
    PerBlockMXBufferReuseObserver,
    PerBlockMXObserver,
    PlaceholderObserver,
)
from quark.torch.quantization.utils import get_num_bits
from quark.torch.quantization.config.type import ZeroPointType
from quark.torch.utils import assert_no_nan
from torch.distributed._tensor.experimental import implicit_replication  # type: ignore
from quark.common.utils.log import ScreenLogger
from quark.common.data_type import BaseFP8_E5M3

logger = ScreenLogger(__name__)


class RealQuantizerBase(ABC, nn.Module):
    def __init__(self, qspec: QTensorConfig) -> None:
        super().__init__()
        self.qspec = qspec
        self.is_dynamic = qspec.is_dynamic
        self.is_scale_quant = qspec.is_scale_quant

    @property
    def quant_spec(self) -> QTensorConfig:
        return self.qspec

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

    def update_dynamic_params(self, X: torch.Tensor) -> None:
        pass

    @abstractmethod
    def unpack_tensor(self, X: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def unpack_params(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        pass


class StaticRealQuantizer(RealQuantizerBase, ABC):
    def __init__(
        self,
        qspec: QTensorConfig,
        quantizer: FakeQuantizeBase | None,
        reorder: bool,
        real_quantized: bool,
        float_dtype: torch.dtype,
        device: torch.device | None = torch.device("cuda"),  # noqa: B008
        scale_shape: tuple[int, ...] | None = None,
        zero_point_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(qspec)
        self.reorder = reorder
        self.real_quantized = real_quantized
        self.device = device
        self.float_dtype = float_dtype
        self.transpose_scale: bool = False
        self.scale_shape = scale_shape
        self.zero_point_shape = zero_point_shape
        self.pack_method = create_pack_method(
            qscheme=getattr(self.qspec.qscheme, "value", None),
            dtype=self.qspec.dtype.value,
            mx_element_dtype=getattr(self.qspec.mx_element_dtype, "value", None),
        )

        if self.qspec.qscheme == QSchemeType.per_group and self.qspec.dtype in PER_GROUP_INT_TRANSPOSE_DTYPES:
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
            X,
            self.reorder,
            **({"origin_packed_axis_size": self.scale.shape[-1]} if self.scale.shape != torch.Size([]) else {}),
        )
        return unpacked_X

    def unpack_params(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        zero_point = None
        if getattr(self, "zero_point", None) is not None:
            zero_point = self.pack_method.unpack(
                self.zero_point,
                self.reorder,
                **({"origin_packed_axis_size": self.scale.shape[-1]} if self.scale.shape != torch.Size([]) else {}),
            )

        if self.transpose_scale:
            # transpose_scale of bias is always false in qparamslinear.py
            scale = self.scale.data.t().contiguous()
        else:
            scale = self.scale

        if getattr(self.qspec, "scale_format", None) == "e8m0":
            scale = 2 ** (scale.view(torch.uint8).to(torch.int16) - 127).to(self.float_dtype)
        elif getattr(self.qspec, "scale_format", None) == "e5m3":
            # Dequantize E5M3 (uint8) to float32
            scale = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                "fp8_e5m3", scale, None, None, -1, -1, "per_tensor"
            ).to(self.float_dtype)

        return scale, zero_point


class StaticScaledRealQuantizer(StaticRealQuantizer):
    """
    On export, performs transpose on scale and pack on zeropint. Called by parent class, performs real quantization on weight, bias.
    On import, performs dequantization of weight, bias, and fakequantization of input, output via forward method.
    """

    scale: torch.Tensor

    def __init__(
        self,
        qspec: QTensorConfig,
        quantizer: FakeQuantizeBase | None,
        reorder: bool,
        real_quantized: bool,
        float_dtype: torch.dtype,
        device: torch.device | None = torch.device("cuda"),  # noqa: B008
        scale_shape: tuple[int, ...] | None = None,
        zero_point_shape: tuple[int, ...] | None = None,
    ) -> None:
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
                        "zero_point", torch.empty(self.zero_point_shape, device=self.device, dtype=quant_torch_dtype)
                    )
                else:
                    self.register_buffer("zero_point", torch.empty((), device=self.device, dtype=quant_torch_dtype))
        else:
            # TODO: check here
            self.register_buffer("scale", quantizer.scale)
            if self.qspec.dtype in INT_QUANT_DTYPES:
                self.register_buffer("zero_point", quantizer.zero_point.to(torch.int))
            if hasattr(quantizer, "_quantized_block_scale"):
                self._quantized_block_scale = quantizer._quantized_block_scale

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # If real_quantized, unpack and dequantize tensor.
        # If not real_quantized, unpack and fakequantize tensor.

        if self.real_quantized:
            # for weight, bias
            X = self.unpack_tensor(X)
            scale, zero_point = self.unpack_params()
            # Ensure scale and zero_point are on the same device as X
            if scale.device != X.device:
                scale = scale.to(X.device)
            if zero_point is not None and zero_point.device != X.device:
                zero_point = zero_point.to(X.device)
            with implicit_replication():
                X = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                    self.qspec.dtype.value,
                    X,
                    scale,
                    zero_point,
                    self.qspec.ch_axis,
                    self.qspec.group_size,
                    self.qspec.qscheme.value,  # type: ignore[union-attr]
                )
        else:
            # X = self.unpack_tensor(X)
            scale, zero_point = self.unpack_params()
            # Ensure scale and zero_point are on the same device as X
            if scale.device != X.device:
                scale = scale.to(X.device)
            if zero_point is not None and zero_point.device != X.device:
                zero_point = zero_point.to(X.device)
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
                    None,
                )
        return X

    def to_real_quantize_params(self, param: torch.Tensor) -> torch.Tensor:
        """
        Quantize weight and bias on low-bit precision datatypes, and pack them if required.
        """
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
            dtype, param, scale, zero_point, ch_axis, group_size, quant_min, quant_max, round_method, qscheme_str_name
        )
        w_res = self.pack_method.pack(w_res, self.reorder)
        return w_res

    def to_fake_quantize_params(self, param: torch.Tensor) -> torch.Tensor:
        """
        Fake quantize weight and bias
        """
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
            dtype,
            param,
            scale,
            zero_point,
            ch_axis,
            group_size,
            quant_min,
            quant_max,
            round_method,
            qscheme_str_name,
            None,
        )
        assert isinstance(res, torch.Tensor), "res must be a torch.Tensor!"
        return res

    # Pack zero point
    def pack_zero_point(self) -> None:
        if getattr(self, "zero_point", None) is not None and self.qspec and hasattr(self.qspec, "dtype"):
            self.zero_point: torch.Tensor = self.pack_method.pack(self.zero_point, self.reorder)

    # Try to convert scale to int8 and transpose scale
    def maybe_convert_and_transpose_scale(self) -> None:
        if getattr(self.qspec, "scale_format", None) == "e8m0":
            self.scale = to_e8m0_uint8(self.scale)

        if getattr(self.qspec.dtype, "value", None) in ["int8", "uint8", "int4", "uint4", "int2"]:
            if self.scale.ndim > 2:
                raise ValueError("Only supports self.scale with dimensions not greater than 2.")
            if getattr(self.qspec.qscheme, "value", None) == "per_group":
                self.scale = self.scale.t().contiguous()

        if getattr(self.qspec, "scale_format", None) == "e5m3":
            # Quantize the scale to E5M3 format
            self.scale = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
                Dtype.fp8_e5m3.value,
                self.scale,
                None,  # scale (not needed - simple conversion to E5M3)
                None,  # zero_point
                -1,  # axis
                1,  # group_size
                BaseFP8_E5M3.min_value,
                BaseFP8_E5M3.max_value,
                2,  # round_method (round)
                "per_group",
            )


class StaticNonScaledRealQuantizer(StaticRealQuantizer):
    """
    On export, performs transpose on scale and pack on zeropint. Called by parent class, performs real quantization on weight, bias.
    On import, performs dequantization of weight, bias, and fakequantization of input, output via forward method.
    """

    def __init__(
        self,
        qspec: QTensorConfig,
        quantizer: FakeQuantizeBase | None,
        reorder: bool,
        real_quantized: bool,
        float_dtype: torch.dtype,
        device: torch.device | None = torch.device("cuda"),  # noqa: B008
        scale_shape: tuple[int, ...] | None = None,
        zero_point_shape: tuple[int, ...] | None = None,
    ) -> None:
        super().__init__(qspec, quantizer, reorder, real_quantized, float_dtype, device, scale_shape, zero_point_shape)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Forward method for StaticNonScaledRealQuantizer is not supported currently.")

    def to_real_quantize_params(self, param: torch.Tensor) -> torch.Tensor:
        """
        Quantize weight and bias on low-bit precision datatypes, and pack them if required.
        """
        dtype = self.qspec.dtype.value
        assert self.qspec.mx_element_dtype is not None
        mx_element_dtype = self.qspec.mx_element_dtype.value
        axis = self.qspec.ch_axis
        block_size = self.qspec.group_size
        w_res = quark.torch.kernel.non_scaled_real_quantize(  # type: ignore[attr-defined]
            param, dtype, mx_element_dtype, axis, block_size
        )
        w_res = self.pack_method.pack(w_res, self.reorder)
        return w_res

    def pack_zero_point(self) -> None:
        pass

    def maybe_convert_and_transpose_scale(self) -> None:
        pass


class DynamicScaledQuantizer(RealQuantizerBase):
    def __init__(
        self,
        qspec: QTensorConfig,
        device: torch.device | None = torch.device("cuda"),  # noqa: B008
        float_dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__(qspec)
        self.device = device
        self.float_dtype = float_dtype
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

        assert self.dtype not in [Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.bfloat16, Dtype.float16, Dtype.bfp16], (
            "Not supported for mx, mx6, mx9, bfloat16, float16, bfp16 quantization!"
        )
        assert self.zero_point_type == ZeroPointType.int32, "Only support int32 zero point!"

        # Dynamic quantizers always compute qparams from the current input tensor.
        # Keep buffers uninitialized and fill them during update_dynamic_params().
        self.register_buffer("scale", None, persistent=False)
        self.register_buffer("zero_point", None, persistent=False)

        self.quant_min, self.quant_max = calculate_qmin_qmax(self.dtype)

    @staticmethod
    def create_observer(quant_spec: QTensorConfig, device: torch.device | None = None) -> ObserverBase:
        if quant_spec.observer_cls is not None:
            observer_cls = quant_spec.observer_cls
            if observer_cls is PerBlockMXObserver and quant_spec.enable_buffer_reuse:
                observer_cls = PerBlockMXBufferReuseObserver
            return observer_cls(quant_spec, device)
        else:
            return PlaceholderObserver(quant_spec)

    def calculate_qparams(
        self, observer: ObserverBase, X: torch.Tensor
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        assert_no_nan(X, "tensor contains NaN!")
        qparams = observer._calculate_qparams()
        scale, zero_point = None, None
        if qparams is not None:
            scale, zero_point = qparams
            assert_no_nan(scale, "scale contains NaN!")
            assert_no_nan(zero_point, "zero_point contains NaN!")
        return scale, zero_point

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        assert_no_nan(X, "input tensor X contains NaN!")
        self.update_dynamic_params(X)
        # Do fake quantize
        mx_element_dtype = None if self.mx_element_dtype is None else self.mx_element_dtype.value
        # Ensure scale and zero_point are on the same device as X
        scale = self.scale
        zero_point = self.zero_point
        if scale is not None and scale.device != X.device:
            scale = scale.to(X.device)
        if zero_point is not None and zero_point.device != X.device:
            zero_point = zero_point.to(X.device)
        X_quantized: torch.Tensor = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
            self.dtype.value,
            X,
            scale,
            zero_point.to(torch.int) if zero_point is not None else None,
            self.ch_axis,
            self.group_size,
            self.quant_min,
            self.quant_max,
            self.round_method,
            self.qscheme_str_name,
            mx_element_dtype,
        )

        assert_no_nan(X_quantized, "output tensor X contains NaN!")
        return X_quantized

    def update_dynamic_params(self, X: torch.Tensor) -> None:
        observer = self.create_observer(self.qspec, X.device)
        assert not isinstance(observer, TQTObserver | LSQObserver), "Not supported for TQT and LSQ observer!"

        # Do observation
        observer(X.detach())
        self.scale, self.zero_point = self.calculate_qparams(observer, X)

    def to_real_quantize_params(self, param: torch.Tensor) -> torch.Tensor:
        return param

    def to_fake_quantize_params(self, param: torch.Tensor) -> torch.Tensor:
        return param

    def unpack_tensor(self, X: torch.Tensor) -> torch.Tensor:
        return X

    def unpack_params(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
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
        assert not any(not isinstance(q, RealQuantizerBase) for q in quantizers), (
            "All quantizers must be a RealQuantizerBase."
        )
        super().__init__(*quantizers)

        assert all(not isinstance(quantizer, StaticNonScaledRealQuantizer) for quantizer in quantizers), (
            "StaticNonScaledRealQuantizer is not supported in SequentialRealQuantizer currently."
        )

        tensor_quantizers = [quantizer for quantizer in quantizers if quantizer.is_scale_quant is False]
        assert len(tensor_quantizers) > 0, "SequentialRealQuantizer must contain at least one tensor quantizer"
        assert all(
            tensor_quantizer.is_dynamic == tensor_quantizers[0].is_dynamic for tensor_quantizer in tensor_quantizers
        ), "Tensor quantizers in SequentialRealQuantizer must share the same is_dynamic configuration"
        self.is_dynamic = tensor_quantizers[0].is_dynamic

        # Verify all quantizers have consistent real_quantized configuration
        assert all(quantizer.real_quantized == quantizers[0].real_quantized for quantizer in quantizers), (
            "The real_quantized configuration of all quantizers should be the same"
        )

        self.real_quantized = quantizers[0].real_quantized

    def to_real_quantize_params(self, param: torch.Tensor) -> torch.Tensor:
        param_dtype = param.dtype
        for i, module in enumerate(self):
            if not module.is_scale_quant:
                param = param.to(param_dtype)
                if i < len(self) - 1:
                    next_module = self[i + 1]
                    if next_module.is_scale_quant:
                        # Skip if scale was pre-computed in get_real_quantizer (combined division)
                        if not hasattr(module, "_quantized_block_scale"):
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
                    quantized_scale = getattr(module, "_quantized_block_scale", None)
                    if quantized_scale is not None:
                        # Use pre-computed quantized block scale (combined division)
                        device = module.scale.device
                        module.scale = self[i + 1].pack_method.pack(quantized_scale, self[i + 1].reorder)
                        module.scale = module.scale.to(device)
                    else:
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
                            scale = next_module(scale.to(next_module.float_dtype))

                    x = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
                        module.qspec.dtype.value,
                        x.to(module.float_dtype),
                        scale,
                        zero_point,
                        module.qspec.ch_axis,
                        module.qspec.group_size,
                        module.quant_min,
                        module.quant_max,
                        getattr(module.qspec.round_method, "value", None),
                        getattr(module.qspec.qscheme, "value", None),
                    )

            for module_index, module in enumerate(reversed(self)):
                if not module.is_scale_quant:
                    scale, zero_point = module.unpack_params()
                    if module_index > 0:
                        previous_module = self[len(self) - module_index]
                        if previous_module.is_scale_quant:
                            scale = previous_module(scale.to(previous_module.float_dtype))

                    x = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                        module.qspec.dtype.value,
                        x.to(module.float_dtype),
                        scale,
                        zero_point,
                        module.qspec.ch_axis,
                        module.qspec.group_size,
                        module.qspec.qscheme.value,
                    )
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
                                previous_module.qspec.dtype.value,
                                scale.to(previous_module.float_dtype),
                                p_scale,
                                p_zero_point,
                                previous_module.qspec.ch_axis,
                                previous_module.qspec.group_size,
                                previous_module.qspec.qscheme.value,
                            )
                    x = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                        module.qspec.dtype.value,
                        x.to(module.float_dtype),
                        scale,
                        zero_point,
                        module.qspec.ch_axis,
                        module.qspec.group_size,
                        module.qspec.qscheme.value,
                    )
        return x


def get_real_quantizer(
    qspec: QTensorConfig | list[QTensorConfig],
    quantizer: FakeQuantizeBase | SequentialQuantize | None,
    reorder: bool | None = None,
    real_quantized: bool | None = None,
    float_dtype: torch.dtype | None = None,
    device: torch.device | None = torch.device("cuda"),  # noqa: B008
    scale_shape: tuple[int, ...] | list[tuple[int, ...]] | None = None,
    zero_point_shape: tuple[int, ...] | list[tuple[int, ...]] | None = None,
) -> RealQuantizerBase | SequentialRealQuantizer:
    if isinstance(qspec, list):
        quantizers: list[RealQuantizerBase] = []
        for i, q in enumerate(qspec):
            assert not isinstance(quantizer, FakeQuantizeBase), (
                "quantizer must be a SequentialQuantize for sequential real quantizer!"
            )
            q_quantizer = None if quantizer is None else quantizer[len(quantizers)]
            s_shape = scale_shape[i] if scale_shape is not None else None
            assert not isinstance(s_shape, int), "scale_shape must be a list of tuples"
            zp_shape = zero_point_shape[i] if zero_point_shape is not None else None
            assert not isinstance(zp_shape, int), "zero_point_shape must be a list of tuples"
            real_quantizer = get_real_quantizer(
                qspec=q,
                quantizer=q_quantizer,
                reorder=reorder,
                real_quantized=real_quantized,
                float_dtype=float_dtype,
                device=device,
                scale_shape=s_shape,
                zero_point_shape=zp_shape,
            )
            assert isinstance(real_quantizer, RealQuantizerBase), "real_quantizer must be a RealQuantizerBase!"
            quantizers.append(real_quantizer)

        return SequentialRealQuantizer(*quantizers)
    else:
        assert not isinstance(quantizer, SequentialQuantize), (
            "quantizer must be a FakeQuantizeBase for single real quantizer!"
        )
        assert not isinstance(scale_shape, list), "scale_shape must be a int tuple for single real quantizer!"
        assert not isinstance(zero_point_shape, list), "zero_point_shape must be a int tuple for single real quantizer!"
        qspec.is_dynamic = False if qspec.dtype in [Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.bfp16] else qspec.is_dynamic

        if qspec.is_dynamic:
            assert qspec.dtype not in [Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.bfp16, Dtype.bfloat16, Dtype.float16], (
                "Not supported for mx, mx6, mx9, bfp16, bfloat16, float16 quantization in dynamic real quantizer!"
            )
            return DynamicScaledQuantizer(
                qspec=qspec,
                device=device,
                float_dtype=float_dtype,
            )

        assert reorder is not None, "reorder must be provided for static real quantizer!"
        assert float_dtype is not None, "float_dtype must be provided for static real quantizer!"
        if qspec.dtype in [Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.bfp16]:
            return StaticNonScaledRealQuantizer(
                qspec=qspec,
                quantizer=quantizer,
                reorder=reorder,
                real_quantized=False,
                device=device,
                float_dtype=float_dtype,
            )
        else:
            assert real_quantized is not None, "real_quantized must be provided for static real scaled quantizer!"
            return StaticScaledRealQuantizer(
                qspec=qspec,
                quantizer=quantizer,
                reorder=reorder,
                real_quantized=real_quantized,
                float_dtype=float_dtype,
                device=device,
                scale_shape=scale_shape,
                zero_point_shape=zero_point_shape,
            )
