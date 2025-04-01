#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import quark.torch.kernel  # noqa

from typing import Optional, List, Dict, Any, Union
from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
from quark.torch.quantization.observer.observer import ObserverBase, PlaceholderObserver
from quark.torch.quantization.config.config import QuantizationSpec
from quark.torch.quantization.observer.tqt_observer import TQTObserver
from quark.torch.quantization.observer.lsq_observer import LSQObserver
from quark.torch.quantization.config.type import Dtype, QSchemeType, ZeroPointType, ScaleType
from quark.torch.quantization.utils import calculate_qmin_qmax, get_num_bits


class FakeQuantizeBase(ABC, nn.Module):
    r"""Base fake quantize module.

    Base fake quantize module
    Any fake quantize implementation should derive from this class.

    Concrete fake quantize module should follow the same API. In forward, they will update
    the statistics of the observed Tensor and fake quantize the input. They should also provide a
    `calculate_qparams` function that computes the quantization parameters given
    the collected statistics.

    """

    fake_quant_enabled: torch.Tensor
    observer_enabled: torch.Tensor

    def __init__(self, quant_spec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        """Set fake_quant_enabled and observer_enabled."""
        super().__init__()

        self.quant_spec = quant_spec

        # fake_quant_enabled and observer_enabled are buffers to support their
        # replication in DDP. Data type is uint8 because Multi-GPU does not support
        # bool tensors.
        self.register_buffer('fake_quant_enabled', torch.tensor([1], dtype=torch.uint8, device=device))
        self.register_buffer('observer_enabled', torch.tensor([1], dtype=torch.uint8, device=device))

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def calculate_qparams(self, x: torch.Tensor) -> None:
        pass

    @abstractmethod
    def to_freezed_module(self) -> nn.Module:
        pass

    def enable_fake_quant(self, enabled: bool = True) -> None:
        self.fake_quant_enabled[0] = 1 if enabled else 0

    def disable_fake_quant(self) -> None:
        self.enable_fake_quant(False)

    def enable_observer(self, enabled: bool = True) -> None:
        self.observer_enabled[0] = 1 if enabled else 0

    def disable_observer(self) -> None:
        self.enable_observer(False)

    @property
    def is_observer_enabled(self) -> bool:
        return self.observer_enabled[0].item() == 1

    @property
    def is_fake_quant_enabled(self) -> bool:
        return self.fake_quant_enabled[0].item() == 1

    def update_buffer(self, buffer_name: str, new_value: Union[torch.Tensor, None],
                      input_tensor_device: torch.device) -> None:
        """
        Update the value of a registered buffer while ensuring that its shape,
        device, and data type match the input tensor.

        Parameters:
        - buffer_name: The name of the buffer to update
        - new_value: The new value to assign to the buffer
        - input_tensor_device: The target device (e.g., torch.device('cuda') or torch.device('cpu'))
        """

        buffer = getattr(self, buffer_name)

        if new_value is not None:
            if buffer.shape != new_value.shape:
                buffer.resize_(new_value.shape)
            buffer = buffer.to(new_value.dtype)
            buffer.copy_(new_value)

        buffer = buffer.to(input_tensor_device)
        setattr(self, buffer_name, buffer)

    @staticmethod  # type: ignore
    def get_fake_quantize(quant_spec: QuantizationSpec,
                          device: Optional[torch.device] = None,
                          **kwargs) -> 'FakeQuantizeBase':
        if quant_spec.dtype in [Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.bfp16]:
            return NonScaledFakeQuantize(quant_spec=quant_spec, device=device)
        else:
            return ScaledFakeQuantize(quant_spec=quant_spec, device=device, **kwargs)


class ScaledFakeQuantize(FakeQuantizeBase):
    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(
            self,
            quant_spec: QuantizationSpec,
            device: Optional[torch.device] = None,
            **kwargs: Any,  # TODO: Delete kwargs here
    ) -> None:
        super().__init__(quant_spec, device)

        # Set properties with Quant Config
        self.dtype = quant_spec.dtype
        self.mx_element_dtype = quant_spec.mx_element_dtype
        self.is_dynamic = quant_spec.is_dynamic
        self.qscheme = quant_spec.qscheme
        self.qscheme_str_name = getattr(quant_spec.qscheme, "value", None)
        self.ch_axis = quant_spec.ch_axis
        self.group_size = quant_spec.group_size
        self.symmetric = quant_spec.symmetric
        self.round_method = getattr(quant_spec.round_method, "value", None)
        self.scale_type = quant_spec.scale_type
        self._num_bits = get_num_bits(quant_spec.dtype)

        self.scale_torch_dtype = None
        if self.scale_type in [ScaleType.float32, ScaleType.float16, ScaleType.bfloat16]:
            self.scale_torch_dtype = self.scale_type.to_torch_dtype()

        self.zero_point_type = quant_spec.zero_point_type

        if self.dtype is Dtype.mx:
            assert self.mx_element_dtype is not None
            self.quant_min, self.quant_max = calculate_qmin_qmax(self.mx_element_dtype)
        elif self.dtype in [Dtype.mx6, Dtype.mx9]:
            self.quant_min = self.quant_max = 0.0
        else:
            self.quant_min, self.quant_max = calculate_qmin_qmax(self.dtype)
        self.observer = self.create_observer(quant_spec, device)
        self.verify_observer(quant_spec, self.observer)
        self.register_buffer('scale', torch.tensor(1.0, dtype=self.scale_torch_dtype, device=device))

        if self.zero_point_type == ZeroPointType.float32:
            self.register_buffer('zero_point', torch.tensor(0.0, dtype=torch.float, device=device))
        else:
            self.register_buffer('zero_point', torch.tensor(0, dtype=torch.int, device=device))

    @staticmethod
    def create_observer(quant_spec: QuantizationSpec, device: Optional[torch.device] = None) -> ObserverBase:
        if quant_spec.observer_cls is not None:
            return quant_spec.observer_cls(quant_spec, device)
        else:
            return PlaceholderObserver(quant_spec)

    # TODO: Add verify_observer to init.
    @staticmethod
    def verify_observer(quant_spec: QuantizationSpec, observer: ObserverBase) -> None:
        if quant_spec.dtype in [Dtype.bfloat16, Dtype.float16]:
            assert isinstance(observer, PlaceholderObserver), f"{quant_spec.dtype} only suuport for PlaceholderObserver"
        # elif quant_spec.dtype in [Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8]:
        #     assert isinstance(observer, UniformScalingObserver)
        # elif quant_spec.dtype in [Dtype.fp8_e4m3]:
        #     assert isinstance(
        #         observer,
        #         (PerTensorMinMaxObserver, PerTensorMSEObserver, PerTensorPercentileObserver, PerChannelMinMaxObserver))

    def calculate_qparams(self, X: torch.Tensor) -> None:
        qparams = self.observer._calculate_qparams()
        if qparams is not None:
            _scale, _zero_point = qparams
            self.update_buffer('scale', _scale, X.device)
            self.update_buffer('zero_point', _zero_point, X.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Do observation
        if self.is_observer_enabled or self.is_dynamic:
            self.observer(X.detach())
            self.calculate_qparams(X)

        # Do fake quantize
        if self.is_fake_quant_enabled:
            if isinstance(self.observer, TQTObserver):
                X = quark.torch.kernel.tqt_quantize(  # type: ignore[attr-defined]
                    X, self.observer.log_threshold, self.zero_point, self.observer.domain, self.round_method)
            elif isinstance(self.observer, LSQObserver):
                grad_factor = 1.0 / math.sqrt(X.numel() * self.observer.quant_max)
                X = quark.torch.kernel.lsq_quantize(  # type: ignore[attr-defined]
                    X, self.observer.scale + self.observer.eps, self.observer.zero_point, grad_factor,
                    self.observer.quant_min, self.observer.quant_max, self.ch_axis, self.round_method)
            else:
                mx_element_dtype_value = 'None' if self.mx_element_dtype is None else self.mx_element_dtype.value
                if self.zero_point_type == ZeroPointType.float32:
                    X = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
                        self.dtype.value,
                        X,
                        self.scale,
                        self.zero_point.to(torch.float),
                        self.ch_axis,
                        self.group_size,
                        self.quant_min,
                        self.quant_max,
                        self.round_method,
                        self.qscheme_str_name,
                        mx_element_dtype_value,
                    )
                else:
                    X = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
                        self.dtype.value,
                        X,
                        self.scale,
                        self.zero_point.to(torch.int),
                        self.ch_axis,
                        self.group_size,
                        self.quant_min,
                        self.quant_max,
                        self.round_method,
                        self.qscheme_str_name,
                        mx_element_dtype_value,
                    )

        return X

    def extra_repr(self) -> str:
        return 'fake_quant_enabled={}, observer_enabled={}, ' \
                   'quant_min={}, quant_max={}, dtype={}, qscheme={}, mx_element_dtype={}, ch_axis={}, ' \
                   'scale={}, zero_point={}'.format(
                       self.fake_quant_enabled, self.observer_enabled,
                       self.quant_min, self.quant_max,
                       self.dtype, self.qscheme, self.mx_element_dtype, self.ch_axis, self.scale, self.zero_point)

    def _save_to_state_dict(self, destination: Dict[str, Union[torch.nn.Parameter, torch.Tensor]], prefix: str,
                            keep_vars: bool) -> None:
        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)  # type: ignore
        if self.dtype in [
                Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8, Dtype.fp8_e4m3, Dtype.fp8_e5m2, Dtype.mx, Dtype.mx6,
                Dtype.mx9
        ]:
            destination[prefix + 'scale'] = self.scale
            destination[prefix + 'zero_point'] = self.zero_point

    def _load_from_state_dict(self, state_dict: Dict[str, Union[torch.nn.Parameter, torch.Tensor]], prefix: str,
                              local_metadata: Dict[str, Any], strict: bool, missing_keys: List[str],
                              unexpected_keys: List[str], error_msgs: List[str]) -> None:
        # Removing this function throws an error that the the size of the loaded tensor does not match the original size
        # i.e., These buffers start out with numel 0 and become numel 1 once they have their first forward pass.
        local_state = ['scale', 'zero_point']
        for name in local_state:
            key = prefix + name
            if key in state_dict:
                val = state_dict[key]
                # Custom handling to allow loading scale and zero_point
                # of size N into uninitialized buffers of size 0. The
                # buffers are resized here, and the values are copied in
                # the default state_dict loading code of the parent.
                if name == 'scale':
                    self.scale.resize_(val.shape)
                else:
                    assert name == 'zero_point'
                    self.zero_point.resize_(val.shape)
                # For torchscript module we need to update the attributes here since we do not
                # call the `_load_from_state_dict` function defined module.py
                if torch.jit.is_scripting():  # type: ignore[attr-defined]
                    if name == 'scale':
                        self.scale.copy_(val)
                    else:
                        assert name == 'zero_point'
                        self.zero_point.copy_(val)
            elif strict:
                missing_keys.append(key)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)  # type: ignore

    def to_freezed_module(self) -> nn.Module:
        freezed_fake_quantize_model = FreezedScaledFakeQuantize(self.dtype, self.quant_spec)
        if self.dtype in [
                Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8, Dtype.fp8_e4m3, Dtype.fp8_e5m2, Dtype.mx, Dtype.mx6,
                Dtype.mx9
        ]:
            freezed_fake_quantize_model.register_buffer('scale', self.scale)
            freezed_fake_quantize_model.register_buffer('zero_point', self.zero_point)
        freezed_fake_quantize_model.qscheme = self.qscheme
        freezed_fake_quantize_model.qscheme_str_name = self.qscheme_str_name
        freezed_fake_quantize_model.ch_axis = self.ch_axis
        freezed_fake_quantize_model.group_size = self.group_size
        freezed_fake_quantize_model.round_method = self.round_method
        freezed_fake_quantize_model.quant_min = getattr(self, 'quant_min', None)
        freezed_fake_quantize_model.quant_max = getattr(self, 'quant_max', None)
        freezed_fake_quantize_model.mx_element_dtype = self.mx_element_dtype
        freezed_fake_quantize_model.zero_point_type = self.zero_point_type
        freezed_fake_quantize_model.quant_spec = self.quant_spec
        return freezed_fake_quantize_model


class FreezedScaledFakeQuantize(nn.Module):
    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, dtype: Dtype, quant_spec: QuantizationSpec) -> None:
        super(FreezedScaledFakeQuantize, self).__init__()
        self.zero_point_type: Optional[ZeroPointType] = quant_spec.zero_point_type
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float))
        if self.zero_point_type == ZeroPointType.float32:
            self.register_buffer('zero_point', torch.tensor([0.0], dtype=torch.float))
        else:
            self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int))
        self.dtype: Dtype = dtype
        self.quant_spec = quant_spec
        self.quant_min: Optional[int] = None
        self.quant_max: Optional[int] = None
        self.qscheme: Optional[QSchemeType] = None
        self.qscheme_str_name: Optional[str] = None
        self.ch_axis: Optional[int] = None
        self.group_size: Optional[int] = None
        self.round_method: Optional[int] = None
        self.mx_element_dtype: Optional[Dtype] = None

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        mx_element_dtype_value = 'None' if self.mx_element_dtype is None else self.mx_element_dtype.value
        if self.zero_point_type == ZeroPointType.float32:
            X = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
                self.dtype.value, X, self.scale, self.zero_point.to(torch.float), self.ch_axis, self.group_size,
                self.quant_min, self.quant_max, self.round_method, self.qscheme_str_name, mx_element_dtype_value)
        else:
            X = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
                self.dtype.value, X, self.scale, self.zero_point.to(torch.int), self.ch_axis, self.group_size,
                self.quant_min, self.quant_max, self.round_method, self.qscheme_str_name, mx_element_dtype_value)
        assert isinstance(X, torch.Tensor)

        return X

    def _load_from_state_dict(
        self,
        state_dict: Dict[str, Any],
        prefix: str,
        local_metadata: Dict[str, Any],
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:

        for name, value in state_dict.items():
            if "scale" in name:
                self.scale.resize_(value.shape)
            if "zero_point" in name:
                self.zero_point.resize_(value.shape)

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)  # type: ignore

        if self.dtype not in [Dtype.int2, Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8]:
            zero_point_key = prefix + 'zero_point'
            if zero_point_key in missing_keys:
                missing_keys.remove(zero_point_key)


class NonScaledFakeQuantize(FakeQuantizeBase):

    def __init__(self, quant_spec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        super().__init__(quant_spec, device)

        self.dtype = quant_spec.dtype
        self.mx_element_dtype = quant_spec.mx_element_dtype
        self.axis = quant_spec.ch_axis
        self.group_size = quant_spec.group_size
        self.is_dynamic = quant_spec.is_dynamic

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = quark.torch.kernel.non_scaled_fake_quantize(  # type: ignore[attr-defined]
            X, self.dtype.value, self.mx_element_dtype.value if self.mx_element_dtype is not None else "", self.axis,
            self.group_size)
        assert isinstance(X, torch.Tensor)
        return X

    def to_freezed_module(self) -> nn.Module:
        return self
