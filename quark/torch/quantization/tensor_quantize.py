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
from quark.torch.quantization.constants import INT_QUANT_DTYPES


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
    is_dynamic: Optional[bool] = None

    def __init__(self, quant_spec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        """Set fake_quant_enabled and observer_enabled."""
        super().__init__()

        self.quant_spec = quant_spec
        self.is_dynamic = quant_spec.is_dynamic

        # fake_quant_enabled and observer_enabled are buffers to support their
        # replication in DDP. Data type is uint8 because Multi-GPU does not support
        # bool tensors.
        self.register_buffer('fake_quant_enabled',
                             torch.tensor([1], dtype=torch.uint8, device=device),
                             persistent=False)
        self.register_buffer('observer_enabled', torch.tensor([1], dtype=torch.uint8, device=device), persistent=False)

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
    def get_fake_quantize(quant_spec: Union[QuantizationSpec, List[QuantizationSpec]],
                          device: Optional[torch.device] = None,
                          **kwargs) -> Union['FakeQuantizeBase', 'SequentialQuantize']:
        # Handle list of specs for sequential quantization
        if isinstance(quant_spec, list):
            return SequentialQuantize(quant_specs=quant_spec, device=device)
            # quantizers: List[FakeQuantizeBase] = []
            # quantizer = None
            # for spec in quant_spec:
            #     quantizer = FakeQuantizeBase.get_fake_quantize(spec, device, **kwargs)
            #     assert isinstance(quantizer, FakeQuantizeBase), "quantizer should be a FakeQuantizeBase instance"
            #     quantizers.append(quantizer)
            # return SequentialQuantize(*quantizers)

        # Handle single spec case
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
        self.is_scale_quant = quant_spec.is_scale_quant

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

        persistent = (not self.is_dynamic) or (self.is_scale_quant and self.qscheme == QSchemeType.per_tensor)
        self.register_buffer('scale',
                             torch.tensor(1.0, dtype=self.scale_torch_dtype, device=device),
                             persistent=persistent)

        persistent = persistent and self.dtype in INT_QUANT_DTYPES
        if self.zero_point_type == ZeroPointType.float32:
            self.register_buffer('zero_point',
                                 torch.tensor(0.0, dtype=torch.float, device=device),
                                 persistent=persistent)
        else:
            self.register_buffer('zero_point', torch.tensor(0, dtype=torch.int, device=device), persistent=persistent)

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
        assert not torch.isnan(X).any().item(), "tensor contains NaN!"
        qparams = self.observer._calculate_qparams()
        if qparams is not None:
            _scale, _zero_point = qparams
            assert not torch.isnan(_scale).any().item(), "scale contains NaN!"
            assert not torch.isnan(_zero_point).any().item(), "zero_point contains NaN!"
            self.update_buffer('scale', _scale, X.device)
            self.update_buffer('zero_point', _zero_point, X.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        # Do observation
        if self.is_observer_enabled:
            self.observer.record_observed_tokens(X)
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
                        self.scale.to(X.device),
                        self.zero_point.to(torch.float).to(X.device),
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
                        self.scale.to(X.device),
                        self.zero_point.to(torch.int).to(X.device),
                        self.ch_axis,
                        self.group_size,
                        self.quant_min,
                        self.quant_max,
                        self.round_method,
                        self.qscheme_str_name,
                        mx_element_dtype_value,
                    )
        assert not torch.isnan(X).any().item(), "output tensor X contains NaN!"
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
        # TODO: do we really need this? state_dict() already contains persistent buffers!

        # We cannot currently register scalar values as buffers, so need to manually
        # specify serialization here.
        super()._save_to_state_dict(destination, prefix, keep_vars)  # type: ignore
        if self.dtype in [
                Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.uint8, Dtype.fp8_e4m3, Dtype.fp8_e5m2, Dtype.mx, Dtype.mx6,
                Dtype.mx9
        ]:
            if not self.is_dynamic:
                destination[prefix + 'scale'] = self.scale

                if self.dtype in INT_QUANT_DTYPES:
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

        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)  # type: ignore

    def to_freezed_module(self) -> nn.Module:
        freezed_fake_quantize_model = FreezedScaledFakeQuantize(self.dtype, self.quant_spec)
        if self.dtype in [
                Dtype.int4, Dtype.uint4, Dtype.int8, Dtype.int16, Dtype.uint16, Dtype.uint8, Dtype.int32,
                Dtype.fp8_e4m3, Dtype.fp8_e5m2, Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.fp4, Dtype.fp6_e2m3,
                Dtype.fp6_e3m2, Dtype.int2
        ]:
            persistent = (not self.is_dynamic) or (self.is_scale_quant and self.qscheme == QSchemeType.per_tensor)
            freezed_fake_quantize_model.register_buffer('scale', self.scale, persistent=persistent)

            persistent = persistent and self.dtype in INT_QUANT_DTYPES
            freezed_fake_quantize_model.register_buffer('zero_point', self.zero_point, persistent=persistent)

        freezed_fake_quantize_model.qscheme = self.qscheme
        freezed_fake_quantize_model.qscheme_str_name = self.qscheme_str_name
        freezed_fake_quantize_model.ch_axis = self.ch_axis
        freezed_fake_quantize_model.group_size = self.group_size
        freezed_fake_quantize_model.round_method = self.round_method
        freezed_fake_quantize_model.quant_min = getattr(self, 'quant_min', None)
        freezed_fake_quantize_model.quant_max = getattr(self, 'quant_max', None)
        freezed_fake_quantize_model.mx_element_dtype = self.mx_element_dtype
        freezed_fake_quantize_model.zero_point_type = self.zero_point_type
        freezed_fake_quantize_model.is_scale_quant = self.is_scale_quant
        freezed_fake_quantize_model.quant_spec = self.quant_spec
        return freezed_fake_quantize_model


class FreezedScaledFakeQuantize(nn.Module):
    scale: torch.Tensor
    zero_point: torch.Tensor

    def __init__(self, dtype: Dtype, quant_spec: QuantizationSpec) -> None:
        super(FreezedScaledFakeQuantize, self).__init__()
        self.zero_point_type: Optional[ZeroPointType] = quant_spec.zero_point_type

        persistent = (not quant_spec.is_dynamic) or (quant_spec.is_scale_quant
                                                     and quant_spec.qscheme == QSchemeType.per_tensor)
        self.register_buffer('scale', torch.tensor([1.0], dtype=torch.float), persistent=persistent)

        persistent = persistent and quant_spec.dtype in INT_QUANT_DTYPES
        if self.zero_point_type == ZeroPointType.float32:
            self.register_buffer('zero_point', torch.tensor([0.0], dtype=torch.float), persistent=persistent)
        else:
            self.register_buffer('zero_point', torch.tensor([0], dtype=torch.int), persistent=persistent)

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
        self.is_scale_quant: bool = False

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


class NonScaledFakeQuantize(FakeQuantizeBase):

    def __init__(self, quant_spec: QuantizationSpec, device: Optional[torch.device] = None) -> None:
        super().__init__(quant_spec, device)

        self.dtype = quant_spec.dtype
        self.mx_element_dtype = quant_spec.mx_element_dtype
        self.axis = quant_spec.ch_axis
        self.group_size = quant_spec.group_size
        self.is_dynamic = quant_spec.is_dynamic
        self.scale_calculation_mode = quant_spec.scale_calculation_mode
        self.is_scale_quant = quant_spec.is_scale_quant
        self.observer = self.create_observer(quant_spec, device)

    @staticmethod
    def create_observer(quant_spec: QuantizationSpec, device: Optional[torch.device] = None) -> ObserverBase:
        return PlaceholderObserver(quant_spec)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.is_fake_quant_enabled:
            X = quark.torch.kernel.non_scaled_fake_quantize(  # type: ignore[attr-defined]
                X, self.dtype.value, self.mx_element_dtype.value if self.mx_element_dtype is not None else "",
                self.axis, self.group_size, self.scale_calculation_mode)
        assert isinstance(X, torch.Tensor)
        return X

    def to_freezed_module(self) -> nn.Module:
        return self


class SequentialQuantize(nn.Sequential):
    """A sequential container for  :class:`FakeQuantizeBase` modules.

    This modules is used to quantize a tensor in multiple formats sequentially. It takes as input
    a list of quantization specifications and containerize them similar to :class:`torch.nn.Sequential`.

    Args:
        quant_specs (List[QuantizationSpec]): List of quantization specifications.
        device (Optional[torch.device]): Device to run the quantization on.

    """
    is_dynamic: Optional[bool] = None

    def __init__(self, quant_specs: List[QuantizationSpec], device: Optional[torch.device] = None) -> None:
        """Initialize SequentialQuantize module."""
        quantizers = [FakeQuantizeBase.get_fake_quantize(spec, device) for spec in quant_specs]
        assert not any(not isinstance(q, FakeQuantizeBase)
                       for q in quantizers), ("All quantizers must be a FakeQuantizeBase.")
        super().__init__(*quantizers)

        # the is_dynamic configuration of all quantizers should be the same
        assert all(quantizer.is_dynamic == quantizers[0].is_dynamic for quantizer in quantizers), \
            "The is_dynamic configuration of all quantizers should be the same"
        assert all(not isinstance(quantizer, NonScaledFakeQuantize) for quantizer in quantizers), \
            "NonScaledFakeQuantize is not supported in SequentialQuantize currently"
        assert all(not isinstance(quantizer.observer, (TQTObserver, LSQObserver)) for quantizer in quantizers), \
            "TQTObserver and LSQObserver are not supported in SequentialQuantize currently"
        assert all((not quantizer.zero_point_type == ZeroPointType.float32) for quantizer in quantizers), \
            "Float32 zero point is not supported in SequentialQuantize currently"

        self.is_dynamic = quantizers[0].is_dynamic

    def disable_fake_quant(self) -> None:
        for module in self:
            if isinstance(module, FreezedScaledFakeQuantize):
                continue
            module.disable_fake_quant()

    def enable_observer(self, enabled: bool = True) -> None:
        for module in self:
            if isinstance(module, FreezedScaledFakeQuantize):
                continue
            module.enable_observer(enabled)

    @property
    def is_observer_enabled(self) -> bool:
        return any(
            (not isinstance(module, FreezedScaledFakeQuantize)) and module.is_observer_enabled for module in self)

    @property
    def is_fake_quant_enabled(self) -> bool:
        return any(isinstance(module, FreezedScaledFakeQuantize) or module.is_fake_quant_enabled for module in self)

    # store the config of the is_observer_enabled status of each module
    def get_observer_enabled_config(self) -> List[Optional[bool]]:
        return [
            module.is_observer_enabled if not isinstance(module, FreezedScaledFakeQuantize) else None for module in self
        ]

    # store the config of the is_fake_quant_enabled status of each module
    def get_fake_quant_enabled_config(self) -> List[Optional[bool]]:
        return [
            module.is_fake_quant_enabled if not isinstance(module, FreezedScaledFakeQuantize) else None
            for module in self
        ]

    # restore the config of the is_fake_quant_enabled status of each module
    def restore_fake_quant_enabled_config(self, config: List[Optional[bool]]) -> None:
        for module, enabled in zip(self, config):
            if isinstance(module, FreezedScaledFakeQuantize):
                continue
            module.enable_fake_quant(enabled)

    # check the class contains at least two tensor quantizers
    def contains_at_least_two_tensor_quantizers(self) -> bool:
        tensor_quantizers = [module for module in self if module.is_scale_quant is False]
        return len(tensor_quantizers) >= 2

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass for sequential quantization."""
        # Observer mode
        if self.is_observer_enabled:
            fake_quant_enabled_config = self.get_fake_quant_enabled_config()
            self.disable_fake_quant()

            previous_tensor_quantizer = None
            quant_x = X.clone() if self.contains_at_least_two_tensor_quantizers() else X

            for module_index, module in enumerate(self):
                if module.is_scale_quant:
                    self._validate_scale_quantizer(module_index)
                    module(self[module_index - 1].scale)
                else:
                    quant_x = self._process_tensor_quantizer(module_index, module, previous_tensor_quantizer, quant_x,
                                                             X.dtype)
                    previous_tensor_quantizer = module

            if self.contains_at_least_two_tensor_quantizers():
                del quant_x
            self.restore_fake_quant_enabled_config(fake_quant_enabled_config)

        # Fake quantization mode
        if self.is_fake_quant_enabled:
            # Forward quantization
            X = self._apply_forward_quantization(X)
            # Reverse dequantization
            X = self._apply_reverse_dequantization(X)

        return X

    def _validate_scale_quantizer(self, module_index: int) -> None:
        """Validate scale quantizer configuration."""
        assert module_index > 0, "Scale quantizer could not be the first one"
        previous_module = self[module_index - 1]
        assert hasattr(previous_module, 'scale'), "Previous module must have scale attribute"
        assert isinstance(previous_module, ScaledFakeQuantize), "Previous module must be ScaledFakeQuantize"
        assert previous_module.qscheme_str_name == "per_group", "Only per_group scheme supported"
        assert not previous_module.is_scale_quant, "Previous module cannot be scale quantizer"

    def _process_tensor_quantizer(self, module_index: int, module: Any, previous_quantizer: Any, x: torch.Tensor,
                                  target_dtype: torch.dtype) -> torch.Tensor:
        """Process tensor quantizer and apply quantization."""
        if previous_quantizer is None:
            module(x)
            return x

        if self[module_index - 1].is_scale_quant:
            prev_module = self[module_index - 1]
            previous_quantizer.scale = self._apply_scale_quantization(previous_quantizer, prev_module)

        x = self._apply_tensor_quantization(previous_quantizer, x, target_dtype)
        module(x)
        return x

    def _apply_forward_quantization(self, X: torch.Tensor) -> torch.Tensor:
        """Apply forward quantization to input tensor."""
        x_dtype = X.dtype
        for module_index, module in enumerate(self):
            if not module.is_scale_quant:
                if module_index < len(self) - 1 and self[module_index + 1].is_scale_quant:
                    next_module = self[module_index + 1]
                    module.scale = self._apply_scale_quantization(module, next_module)
                X = self._apply_tensor_quantization(module, X, x_dtype)
        return X

    def _apply_reverse_dequantization(self, X: torch.Tensor) -> torch.Tensor:
        """Apply reverse dequantization to input tensor."""
        x_dtype = X.dtype
        for module in reversed(self):
            if not module.is_scale_quant:
                X = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
                    module.dtype.value, X, module.scale,
                    module.zero_point.to(torch.int) if module.zero_point is not None else None, module.ch_axis,
                    module.group_size, module.qscheme_str_name)
        return X.to(x_dtype)

    def _apply_scale_quantization(self, source_module: Any, scale_module: Any) -> torch.Tensor:
        """Apply scale quantization between modules."""
        result = quark.torch.kernel.scaled_fake_quantize(  # type: ignore[attr-defined]
            scale_module.dtype.value, source_module.scale, scale_module.scale,
            scale_module.zero_point.to(torch.int) if scale_module.zero_point is not None else None,
            scale_module.ch_axis, scale_module.group_size, scale_module.quant_min, scale_module.quant_max,
            scale_module.round_method, scale_module.qscheme_str_name, None)
        assert isinstance(result, torch.Tensor)  # Runtime check to ensure correct type
        return result

    def _apply_tensor_quantization(self, module: Any, x: torch.Tensor, target_dtype: torch.dtype) -> torch.Tensor:
        """Apply tensor quantization using module parameters."""
        x = quark.torch.kernel.scaled_real_quantize(  # type: ignore[attr-defined]
            module.dtype.value, x, module.scale,
            module.zero_point.to(torch.int) if module.zero_point is not None else None, module.ch_axis,
            module.group_size, module.quant_min, module.quant_max, module.round_method, module.qscheme_str_name)
        assert isinstance(x, torch.Tensor)  # Runtime check to ensure correct type
        return x.to(target_dtype)


def enable_or_disable_quantizer(quantizer: Union[FakeQuantizeBase, SequentialQuantize],
                                enable: Optional[bool] = False) -> None:
    quantizers = [quantizer] if isinstance(quantizer, ScaledFakeQuantize) else quantizer
    assert isinstance(quantizers, list)
    for _quantizer in quantizers:
        if enable:
            _quantizer.enable_observer()
            _quantizer.enable_fake_quant()
        else:
            _quantizer.disable_observer()
            _quantizer.disable_fake_quant()
