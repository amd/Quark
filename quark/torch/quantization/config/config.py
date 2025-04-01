#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization Config API for PyTorch"""
from __future__ import annotations
import json
import torch.nn as nn
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from typing import Union, Optional, Dict, List, Type, Any, cast, TypeVar
from quark.torch.quantization.observer.observer import ObserverBase, PerBlockMXObserver, PlaceholderObserver, \
    PerTensorMinMaxObserver, PerTensorHistogramObserver, PerTensorMSEObserver, PerTensorPercentileObserver, \
    PerChannelMinMaxObserver, PerGroupMinMaxObserver, PerBlockBFPObserver, PerTensorHistogramObserverPro, OBSERVER_MAP
from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType, DeviceType, QuantizationMode, TQTThresholdInitMeth, ZeroPointType
from quark.torch.quantization.constants import QUARK_LAYER_TYPES
from quark.shares.utils.log import ScreenLogger
from quark.torch.quantization.config.utils import dataclass_pretty_string

logger = ScreenLogger(__name__)

PER_TENSOR_OBSERVER_METHOD_MAP: Dict[str, Type[ObserverBase]] = {
    "min_max": PerTensorMinMaxObserver,
    "histogram": PerTensorHistogramObserver,
    "histogrampro": PerTensorHistogramObserverPro,
    "MSE": PerTensorMSEObserver,
    "percentile": PerTensorPercentileObserver
}

SCALE_TYPE_MAP = {"float": ScaleType.float, "power_of_2": ScaleType.pof2}

ROUND_METHOD_MAP = {"round": RoundType.round, "floor": RoundType.floor, "half_even": RoundType.half_even}

ZERO_POINT_TYPE_MAP = {"int32": ZeroPointType.int32, "float32": ZeroPointType.float32}


def get_per_tensor_observer(observer_method: Optional[str] = None) -> Optional[Type[ObserverBase]]:
    if observer_method:
        assert observer_method in PER_TENSOR_OBSERVER_METHOD_MAP, \
            f"Invalid observer method. Valid observer methods are {list(PER_TENSOR_OBSERVER_METHOD_MAP.keys())}"
        observer_cls = PER_TENSOR_OBSERVER_METHOD_MAP[observer_method]
    else:
        observer_cls = None
    return observer_cls


def get_scale_type(scale_type: Optional[str] = None) -> Optional[ScaleType]:
    if scale_type:
        assert scale_type in SCALE_TYPE_MAP, f"Invalid scale type. Valid scale types are {list(SCALE_TYPE_MAP.keys())}"
        ret = SCALE_TYPE_MAP[scale_type]
    else:
        ret = None
    return ret


def get_round_method(round_method: Optional[str] = None) -> Optional[RoundType]:
    if round_method:
        assert round_method in ROUND_METHOD_MAP, f"Invalid round method. Valid round methods are {list(ROUND_METHOD_MAP.keys())}"
        ret = ROUND_METHOD_MAP[round_method]
    else:
        ret = None
    return ret


def get_zero_point_type(zero_point_type: Optional[str] = None) -> Optional[ZeroPointType]:
    if zero_point_type:
        assert zero_point_type in ZERO_POINT_TYPE_MAP, f"Invalid zero point type, Valid zero point type method are {list(ZERO_POINT_TYPE_MAP.keys())}"
        ret = ZERO_POINT_TYPE_MAP[zero_point_type]
    else:
        ret = None
    return ret


T = TypeVar('T', bound='ConfigBase')


@dataclass(eq=True)
class ConfigBase(ABC):

    name = ""

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        return cls(**data)

    def update_from_dict(self, data: Dict[str, Any]) -> None:
        for field_name in data:
            setattr(self, field_name, data[field_name])

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(eq=True)
class Config(ConfigBase):
    """
    A class that encapsulates comprehensive quantization configurations for a machine learning model, allowing for detailed and hierarchical control over quantization parameters across different model components.

    :param QuantizationConfig global_quant_config: Global quantization configuration applied to the entire model unless overridden at the layer level.
    :param Dict[str, QuantizationConfig] layer_type_quant_config: A dictionary mapping from layer types (e.g., nn.Conv2d, nn.Linear) to their quantization configurations.
    :param Dict[str, QuantizationConfig] layer_quant_config: A dictionary mapping from layer names to their quantization configurations, allowing for per-layer customization. Default is an empty dictionary.
    :param Dict[str, QuantizationConfig] kv_cache_quant_config: A dictionary mapping from layer names to kv_cache quantization configurations. Default is an empty dictionary.
    :param Optional[QuantizationSpec] softmax_quant_spec: A quantization specifications of nn.functional.softmax output. Default is None.
    :param List[str] exclude: A list of layer names to be excluded from quantization, enabling selective quantization of the model. Default is an empty list.
    :param Optional[AlgoConfig] algo_config: Optional configuration for the quantization algorithm, such as GPTQ and AWQ. After this process, the datatype/fake_datatype of weights will be changed with quantization scales. Default is None.
    :param QuantizationMode quant_mode: The quantization mode to be used (eager_mode or fx_graph_mode). Default is eager_mode.
    :param List[PreQuantOptConfig] pre_quant_opt_config: Optional pre-processing optimization, such as Equalization and SmoothQuant. After this process, the value of weights will be changed, but the dtype/fake_dtype will be the same. Default is an empty list.
    :param Optional[int] log_severity_level: 0:DEBUG, 1:INFO, 2:WARNING. 3:ERROR, 4:CRITICAL/FATAL. Default is 1.
    """

    # Global quantization configuration applied to the entire model unless overridden at the layer level.
    global_quant_config: QuantizationConfig

    # A dictionary mapping from layer types (e.g., nn.Conv2d, nn.Linear) to their quantization configurations.
    layer_type_quant_config: Dict[Type[nn.Module], QuantizationConfig] = field(default_factory=dict)

    # A dictionary mapping from layer names to their quantization configurations, allowing for per-layer customization.
    layer_quant_config: Dict[str, QuantizationConfig] = field(default_factory=dict)

    # A dictionary mapping from layer names to kv_cache quantization configurations.
    kv_cache_quant_config: Dict[str, QuantizationConfig] = field(default_factory=dict)

    # A quantization specifications of nn.functional.softmax output.
    softmax_quant_spec: Optional[QuantizationSpec] = None

    # A list of layer names to be excluded from quantization, enabling selective quantization of the model.
    exclude: List[str] = field(default_factory=list)

    # Optional configuration for the quantization algorithm, such as GPTQ and AWQ
    # After this process, the datatype/fake_datatype of weights will be changed with quantization scales.
    algo_config: Optional[AlgoConfig] = None

    # Optional pre-processing optimization, such as Equalization and SmoothQuant.
    # After this process, the value of weights will be changed, but the dtype/fake_dtype will be the same.
    pre_quant_opt_config: List[PreQuantOptConfig] = field(default_factory=list)

    # The quantization mode to be used (eager_mode or fx_graph_mode)
    quant_mode: QuantizationMode = QuantizationMode.eager_mode

    # Log level for printing on screen
    log_severity_level: Optional[int] = 1

    def to_dict(self) -> Dict[str, Any]:
        config_dict: Dict[str, Any] = {
            "global_quant_config": self.global_quant_config.to_dict(),
            "exclude": self.exclude,
            "algo_config": self.algo_config.to_dict() if self.algo_config is not None else None,
            "softmax_quant_spec": self.softmax_quant_spec.to_dict() if self.softmax_quant_spec is not None else None,
            "quant_method": "quark"
        }

        layer_type_quant_config_dict: Dict[str, Any] = {}
        for type, config in self.layer_type_quant_config.items():
            layer_type_quant_config_dict[type.__name__] = config.to_dict()
        config_dict["layer_type_quant_config"] = layer_type_quant_config_dict

        layer_quant_config_dict: Dict[str, Any] = {}
        for name, config in self.layer_quant_config.items():
            layer_quant_config_dict[name] = config.to_dict()
        config_dict["layer_quant_config"] = layer_quant_config_dict

        config_dict["quant_mode"] = self.quant_mode.name

        return config_dict

    def __str__(self) -> str:
        s = dataclass_pretty_string(self)
        return s

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> Config:
        global_quant_config = QuantizationConfig.from_dict(config_dict["global_quant_config"])

        # TODO: Deprecate legacy configuration and remove the None check here.
        # Legacy (quark<1.0) configuration used to allow layer_type_quant_config=None in the serialized config, inconstitant with
        # the type hints of the dataclass.
        layer_type_quant_config = {}
        if config_dict["layer_type_quant_config"] is not None:
            for layer_type_name, layer_type_quantization_config in config_dict["layer_type_quant_config"].items():
                if layer_type_name in QUARK_LAYER_TYPES:
                    layer_type_quant_config[QUARK_LAYER_TYPES[layer_type_name]] = QuantizationConfig.from_dict(
                        layer_type_quantization_config)
                else:
                    raise NotImplementedError(
                        f"Quark does not support reloading a quantization `Config` from a dictionary using custom `layer_type_quantization_config`. Found `'{layer_type_name}'` in `layer_type_quantization_config`, which is not among the supported {QUARK_LAYER_TYPES}."
                    )

        # TODO: Deprecate legacy configuration and remove the None check here.
        # Legacy (quark<1.0) configuration used to allow layer_quant_config=None in the serialized config, inconstitant with
        # the type hints of the dataclass.
        if config_dict["layer_quant_config"] is not None:
            layer_quant_config = {
                layer_name: QuantizationConfig.from_dict(quant_config_dict)
                for layer_name, quant_config_dict in config_dict["layer_quant_config"].items()
            }
        else:
            layer_quant_config = {}

        # TODO: Deprecate legacy (quark<1.0) configuration and remove the check here.
        # `exclude` used to be serialized as `None` when there was no exclude layer, instead of `[]`.
        if config_dict["exclude"] is None:  # pragma: no cover
            exclude = []
        else:
            exclude = config_dict["exclude"]

        algo_config = _load_quant_algo_config_from_dict(
            config_dict["algo_config"]) if config_dict["algo_config"] is not None else None

        # Get softmax_quant_spec configuration from config_dict
        softmax_quant_spec = QuantizationSpec.from_dict(config_dict["softmax_quant_spec"]) if (
            "softmax_quant_spec" in config_dict and config_dict["softmax_quant_spec"] is not None) else None

        if "quant_mode" in config_dict:
            quant_mode = QuantizationMode[config_dict["quant_mode"]]  # Access by name and not by value.
        else:
            # TODO: Deprecate legacy (quark<1.0) configuration and remove the check here.
            # The key `"quant_mode"` used not to be serialized in the legacy quantization_config, inconstitant with
            # the type hints of the dataclass.
            quant_mode = QuantizationMode.eager_mode

        log_severity_level = 1  # `log_severity_level` is not saved.

        # TODO: Currently, `pre_quant_opt_config` is not saved but it might be nice to have?
        return cls(global_quant_config=global_quant_config,
                   layer_type_quant_config=layer_type_quant_config,
                   layer_quant_config=layer_quant_config,
                   exclude=exclude,
                   algo_config=algo_config,
                   softmax_quant_spec=softmax_quant_spec,
                   pre_quant_opt_config=[],
                   quant_mode=quant_mode,
                   log_severity_level=log_severity_level)

    def set_algo_config(self, algo_config: Optional[AlgoConfig]) -> None:
        """
        Sets the algorithm configuration for quantization.

        :param Optional[AlgoConfig] algo_config: The quantization algorithm configuration to be set.
        """
        self.algo_config = algo_config

    def add_pre_optimization_config(self, pre_quant_opt_config: PreQuantOptConfig) -> None:
        """
        Adds a pre-processing optimization configuration to the list of existing pre-quant optimization configs.

        :param PreQuantOptConfig pre_quant_opt_config: The pre-quantization optimization configuration to add.
        """
        assert isinstance(pre_quant_opt_config, PreQuantOptConfig)
        self.pre_quant_opt_config.append(pre_quant_opt_config)


@dataclass(eq=True)
class QuantizationConfig:
    """
    A data class that specifies quantization configurations for different components of a module, allowing hierarchical control over how each tensor type is quantized.

    :param Optional[QuantizationSpec] input_tensors: Input tensors quantization specification. If None, following the hierarchical quantization setup. e.g. If the input_tensors in layer_type_quant_config is None, the configuration from global_quant_config will be used instead. Defaults to None. If None in global_quant_config, input_tensors are not quantized.
    :param Optional[QuantizationSpec] output_tensors: Output tensors quantization specification. Defaults to None. If None, the same as above.
    :param Optional[QuantizationSpec] weight: The weights tensors quantization specification. Defaults to None. If None, the same as above.
    :param Optional[QuantizationSpec] bias: The bias tensors quantization specification. Defaults to None. If None, the same as above.
    :param Optional[DeviceType] target_device: Configuration specifying the target device (e.g., CPU, GPU, IPU) for the quantized model.

    """

    input_tensors: Optional[QuantizationSpec] = None

    output_tensors: Optional[QuantizationSpec] = None

    weight: Optional[QuantizationSpec] = None

    bias: Optional[QuantizationSpec] = None

    target_device: Optional[DeviceType] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_tensors": self.input_tensors.to_dict() if self.input_tensors is not None else None,
            "output_tensors": self.output_tensors.to_dict() if self.output_tensors is not None else None,
            "weight": self.weight.to_dict() if self.weight is not None else None,
            "bias": self.bias.to_dict() if self.bias is not None else None,
            "target_device": self.target_device.value if self.target_device is not None else None
        }

    @classmethod
    def from_dict(cls, quantization_config: Dict[str, Any]) -> QuantizationConfig:
        input_tensors = QuantizationSpec.from_dict(quantization_config["input_tensors"])
        output_tensors = QuantizationSpec.from_dict(quantization_config["output_tensors"])
        weight = QuantizationSpec.from_dict(quantization_config["weight"])
        bias = QuantizationSpec.from_dict(quantization_config["bias"])

        # TODO: Deprecate legacy configuration.
        # Legacy (quark<1.0) saved quantization_config does not have the key `"target_device"`.
        target_device = quantization_config["target_device"] if "target_device" in quantization_config else None
        target_device = DeviceType(target_device) if target_device is not None else None

        return cls(input_tensors=input_tensors,
                   output_tensors=output_tensors,
                   weight=weight,
                   bias=bias,
                   target_device=target_device)


class DataTypeSpec(ConfigBase):

    @abstractmethod
    def to_quantization_spec(self) -> QuantizationSpec:
        pass


@dataclass
class Uint4PerTensorSpec(DataTypeSpec):
    observer_method: Optional[str] = None
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    is_dynamic: Optional[bool] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.uint4,
                                observer_cls=get_per_tensor_observer(self.observer_method),
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_tensor,
                                is_dynamic=self.is_dynamic)


@dataclass
class Uint4PerChannelSpec(DataTypeSpec):
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    ch_axis: Optional[int] = None
    is_dynamic: Optional[bool] = None
    zero_point_type: Optional[str] = "int32"

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.uint4,
                                observer_cls=PerChannelMinMaxObserver,
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_channel,
                                ch_axis=self.ch_axis,
                                is_dynamic=self.is_dynamic,
                                zero_point_type=get_zero_point_type(self.zero_point_type))


@dataclass
class Uint4PerGroupSpec(DataTypeSpec):
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    ch_axis: Optional[int] = None
    is_dynamic: Optional[bool] = None
    group_size: Optional[int] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.uint4,
                                observer_cls=PerGroupMinMaxObserver,
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_group,
                                ch_axis=self.ch_axis,
                                is_dynamic=self.is_dynamic,
                                group_size=self.group_size)


@dataclass
class Int4PerTensorSpec(DataTypeSpec):
    observer_method: Optional[str] = None
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    is_dynamic: Optional[bool] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.int4,
                                observer_cls=get_per_tensor_observer(self.observer_method),
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_tensor,
                                is_dynamic=self.is_dynamic)


@dataclass
class Int4PerChannelSpec(DataTypeSpec):
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    ch_axis: Optional[int] = None
    is_dynamic: Optional[bool] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.int4,
                                observer_cls=PerChannelMinMaxObserver,
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_channel,
                                ch_axis=self.ch_axis,
                                is_dynamic=self.is_dynamic)


@dataclass
class Int4PerGroupSpec(DataTypeSpec):
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    ch_axis: Optional[int] = None
    is_dynamic: Optional[bool] = None
    group_size: Optional[int] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.int4,
                                observer_cls=PerGroupMinMaxObserver,
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_group,
                                ch_axis=self.ch_axis,
                                is_dynamic=self.is_dynamic,
                                group_size=self.group_size)


@dataclass
class Uint8PerTensorSpec(DataTypeSpec):
    observer_method: Optional[str] = None
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    is_dynamic: Optional[bool] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.uint8,
                                observer_cls=get_per_tensor_observer(self.observer_method),
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_tensor,
                                is_dynamic=self.is_dynamic)


@dataclass
class Uint8PerChannelSpec(DataTypeSpec):
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    ch_axis: Optional[int] = None
    is_dynamic: Optional[bool] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.uint8,
                                observer_cls=PerChannelMinMaxObserver,
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_channel,
                                ch_axis=self.ch_axis,
                                is_dynamic=self.is_dynamic)


@dataclass
class Uint8PerGroupSpec(DataTypeSpec):
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    ch_axis: Optional[int] = None
    is_dynamic: Optional[bool] = None
    group_size: Optional[int] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.uint8,
                                observer_cls=PerGroupMinMaxObserver,
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_group,
                                ch_axis=self.ch_axis,
                                is_dynamic=self.is_dynamic,
                                group_size=self.group_size)


@dataclass
class Int8PerTensorSpec(DataTypeSpec):
    observer_method: Optional[str] = None
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    is_dynamic: Optional[bool] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.int8,
                                observer_cls=get_per_tensor_observer(self.observer_method),
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_tensor,
                                is_dynamic=self.is_dynamic)


@dataclass
class Int8PerChannelSpec(DataTypeSpec):
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    ch_axis: Optional[int] = None
    is_dynamic: Optional[bool] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.int8,
                                observer_cls=PerChannelMinMaxObserver,
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_channel,
                                ch_axis=self.ch_axis,
                                is_dynamic=self.is_dynamic)


@dataclass
class Int8PerGroupSpec(DataTypeSpec):
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    ch_axis: Optional[int] = None
    is_dynamic: Optional[bool] = None
    group_size: Optional[int] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.int8,
                                observer_cls=PerGroupMinMaxObserver,
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_group,
                                ch_axis=self.ch_axis,
                                is_dynamic=self.is_dynamic,
                                group_size=self.group_size)


@dataclass
class FP8E4M3PerTensorSpec(DataTypeSpec):
    observer_method: Optional[str] = None
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    is_dynamic: Optional[bool] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                observer_cls=get_per_tensor_observer(self.observer_method),
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_tensor,
                                is_dynamic=self.is_dynamic)


@dataclass
class FP8E4M3PerChannelSpec(DataTypeSpec):
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    ch_axis: Optional[int] = None
    is_dynamic: Optional[bool] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                observer_cls=PerChannelMinMaxObserver,
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_channel,
                                ch_axis=self.ch_axis,
                                is_dynamic=self.is_dynamic)


@dataclass
class FP8E4M3PerGroupSpec(DataTypeSpec):
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    ch_axis: Optional[int] = None
    is_dynamic: Optional[bool] = None
    group_size: Optional[int] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.fp8_e4m3,
                                observer_cls=PerGroupMinMaxObserver,
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_group,
                                ch_axis=self.ch_axis,
                                is_dynamic=self.is_dynamic,
                                group_size=self.group_size)


@dataclass
class FP8E5M2PerTensorSpec(DataTypeSpec):
    observer_method: Optional[str] = None
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    is_dynamic: Optional[bool] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.fp8_e5m2,
                                observer_cls=get_per_tensor_observer(self.observer_method),
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_tensor,
                                is_dynamic=self.is_dynamic)


@dataclass
class FP8E5M2PerChannelSpec(DataTypeSpec):
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    ch_axis: Optional[int] = None
    is_dynamic: Optional[bool] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.fp8_e5m2,
                                observer_cls=PerChannelMinMaxObserver,
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_channel,
                                ch_axis=self.ch_axis,
                                is_dynamic=self.is_dynamic)


@dataclass
class FP8E5M2PerGroupSpec(DataTypeSpec):
    symmetric: Optional[bool] = None
    scale_type: Optional[str] = None
    round_method: Optional[str] = None
    ch_axis: Optional[int] = None
    is_dynamic: Optional[bool] = None
    group_size: Optional[int] = None

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.fp8_e5m2,
                                observer_cls=PerGroupMinMaxObserver,
                                symmetric=self.symmetric,
                                scale_type=get_scale_type(self.scale_type),
                                round_method=get_round_method(self.round_method),
                                qscheme=QSchemeType.per_group,
                                ch_axis=self.ch_axis,
                                is_dynamic=self.is_dynamic,
                                group_size=self.group_size)


@dataclass(eq=True)
class Float16Spec(DataTypeSpec):

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.float16, observer_cls=PlaceholderObserver)


@dataclass(eq=True)
class Bfloat16Spec(DataTypeSpec):

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.bfloat16, observer_cls=PlaceholderObserver)


@dataclass(eq=True)
class MXSpec(DataTypeSpec):
    mx_element_dtype: str = "fp8_e4m3"
    ch_axis: int = -1
    block_size: int = 32
    is_dynamic: bool = True

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.mx,
                                observer_cls=PerBlockMXObserver,
                                qscheme=QSchemeType.per_group,
                                mx_element_dtype=Dtype.from_str(self.mx_element_dtype),
                                ch_axis=self.ch_axis,
                                group_size=self.block_size,
                                is_dynamic=self.is_dynamic,
                                round_method=RoundType.half_even)


@dataclass(eq=True)
class MX6Spec(DataTypeSpec):
    ch_axis: int = -1
    block_size: int = 32
    is_dynamic: bool = True

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.mx6,
                                observer_cls=PerBlockMXObserver,
                                qscheme=QSchemeType.per_group,
                                ch_axis=self.ch_axis,
                                group_size=self.block_size,
                                is_dynamic=self.is_dynamic,
                                round_method=RoundType.half_even)


@dataclass(eq=True)
class MX9Spec(DataTypeSpec):
    ch_axis: int = -1
    block_size: int = 32
    is_dynamic: bool = True

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.mx9,
                                observer_cls=PerBlockMXObserver,
                                qscheme=QSchemeType.per_group,
                                ch_axis=self.ch_axis,
                                group_size=self.block_size,
                                is_dynamic=self.is_dynamic,
                                round_method=RoundType.half_even)


@dataclass
class BFP16Spec(DataTypeSpec):
    is_dynamic: bool = True
    ch_axis: int = -1

    def to_quantization_spec(self) -> QuantizationSpec:
        return QuantizationSpec(dtype=Dtype.bfp16,
                                symmetric=True,
                                observer_cls=PerBlockBFPObserver,
                                qscheme=QSchemeType.per_group,
                                is_dynamic=self.is_dynamic,
                                ch_axis=self.ch_axis,
                                scale_type=ScaleType.float,
                                group_size=8,
                                round_method=RoundType.half_even)


@dataclass(eq=True)
class QuantizationSpec:
    """
    A data class that defines the specifications for quantizing tensors within a model.

    :param Dtype dtype: The data type for quantization (e.g., int8, int4).
    :param Optional[bool] is_dynamic: Specifies whether dynamic or static quantization should be used. Default is None, which indicates no specification.
    :param Optional[Type[ObserverBase]] observer_cls: The class of observer to be used for determining quantization parameters like min/max values. Default is None.
    :param Optional[QSchemeType] qscheme: The quantization scheme to use, such as per_tensor, per_channel or per_group. Default is None.
    :param Optional[int] ch_axis: The channel axis for per-channel quantization. Default is None.
    :param Optional[int] group_size: The size of the group for per-group quantization, also the block size for MX datatypes. Default is None.
    :param Optional[bool] symmetric: Indicates if the quantization should be symmetric around zero. If True, quantization is symmetric. If None, it defers to a higher-level or global setting. Default is None.
    :param Optional[RoundType] round_method: The rounding method during quantization, such as half_even. If None, it defers to a higher-level or default method. Default is None.
    :param Optional[ScaleType] scale_type: Defines the scale type to be used for quantization, like power of two or float. If None, it defers to a higher-level setting or uses a default method. Default is None.
    :param Optional[Dtype] mx_element_dtype: Defines the data type to be used for the element type when using mx datatypes, the shared scale effectively uses FP8 E8M0.
    """

    ###################################################################################################
    # Quantization Specification for Dtype in [Bfloat16, FP8, Int, MX]

    dtype: Dtype

    observer_cls: Optional[Type[ObserverBase]] = None

    ###################################################################################################
    # Quantization Specification for Dtype in [FP8, Int, MX]

    is_dynamic: Optional[bool] = None

    qscheme: Optional[QSchemeType] = None

    ch_axis: Optional[int] = None

    group_size: Optional[int] = None

    ###################################################################################################
    # Quantization Specification for Dtype in [Int]

    symmetric: Optional[bool] = None

    round_method: Optional[RoundType] = None

    scale_type: Optional[ScaleType] = None

    qat_spec: Optional[QATSpec] = None
    ###################################################################################################
    # Quantization Specification for Dtype in [MX]
    mx_element_dtype: Optional[Dtype] = None
    ###################################################################################################
    # Quantization zero point Specification for Dtype
    zero_point_type: Optional[ZeroPointType] = ZeroPointType.int32

    def __post_init__(self) -> None:
        self._verify_config()

    def _verify_config(self) -> None:
        if self.qscheme == QSchemeType.per_group:
            assert self.ch_axis is not None
            assert self.group_size is not None

    def set_group_size(self, group_size: int) -> None:
        assert isinstance(group_size, int) and (group_size > 0 or group_size == -1), \
            "Group size must be a positive integer or -1 (which means group size equals to dimension size)."
        self.group_size = group_size

    def to_dict(self) -> Dict[str, Any]:
        # TODO: qat_spec, mx_element_dtype missing.
        return {
            "dtype": self.dtype.name,
            "is_dynamic": self.is_dynamic,
            "qscheme": self.qscheme.name if self.qscheme is not None else None,
            "ch_axis": self.ch_axis,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "round_method": self.round_method.name if self.round_method is not None else None,
            "scale_type": self.scale_type.name if self.scale_type is not None else None,
            "observer_cls": self.observer_cls.__name__ if self.observer_cls is not None else None
        }

    @classmethod
    def from_dict(cls, config_dict: Optional[Dict[str, Any]]) -> Optional[QuantizationSpec]:
        if config_dict is None:
            return None

        dtype = Dtype[config_dict["dtype"]]

        if config_dict.get("mx_element_dtype", None) is not None:
            mx_element_dtype = Dtype[config_dict["mx_element_dtype"]]
        else:
            mx_element_dtype = None

        if config_dict.get("qscheme", None) is not None:
            qscheme = QSchemeType[config_dict["qscheme"]]
        else:
            qscheme = None

        if config_dict.get("round_method", None) is not None:
            round_method = RoundType[config_dict["round_method"]]
        else:
            round_method = None

        if config_dict.get("scale_type", None) is not None:
            scale_type = ScaleType[config_dict["scale_type"]]
        else:
            scale_type = None

        # TODO: Deprecate legacy configuration.
        # Accomodate the legacy (quark<1.0) export which used custom keys.
        is_dynamic = config_dict["is_dynamic"] if "is_dynamic" in config_dict else config_dict["dynamic"]
        ch_axis = config_dict["ch_axis"] if "ch_axis" in config_dict else config_dict["axis"]

        group_size = config_dict["group_size"]
        symmetric = config_dict["symmetric"]

        if "observer_cls" in config_dict:
            if config_dict["observer_cls"] in OBSERVER_MAP:
                observer_cls = OBSERVER_MAP[config_dict["observer_cls"]]
            else:  # pragma: no cover
                logger.warning(
                    f"Unknown observer_cls={config_dict['observer_cls']}. Loading the QuantizationSpec with observer_cls=PlaceholderObserver."
                )
                observer_cls = PlaceholderObserver
        else:  # pragma: no cover
            # quark<1.0 used not to save the `observer_cls` in `QuantizationSpec.to_dict()`.
            observer_cls = PlaceholderObserver

        return cls(dtype=dtype,
                   is_dynamic=is_dynamic,
                   qscheme=qscheme,
                   ch_axis=ch_axis,
                   group_size=group_size,
                   symmetric=symmetric,
                   round_method=round_method,
                   scale_type=scale_type,
                   mx_element_dtype=mx_element_dtype,
                   observer_cls=observer_cls)  # type: ignore[arg-type]


@dataclass
class TQTSpec(ConfigBase):
    threshold_init_meth: Optional[TQTThresholdInitMeth] = None


QATSpec = Union[TQTSpec]


def load_pre_optimization_config_from_file(file_path: str) -> PreQuantOptConfig:
    """
    Load pre-optimization configuration from a JSON file.

    :param file_path: The path to the JSON file containing the pre-optimization configuration.
    :type file_path: str
    :return: The pre-optimization configuration.
    :rtype: PreQuantOptConfig
    """
    with open(file_path, 'r') as file:
        algo_config_info = json.load(file)
    return _load_pre_optimization_config_from_dict(algo_config_info)


def load_quant_algo_config_from_file(file_path: str) -> AlgoConfig:
    """
    Load quantization algorithm configuration from a JSON file.

    :param file_path: The path to the JSON file containing the quantization algorithm configuration.
    :type file_path: str
    :return: The quantization algorithm configuration.
    :rtype: AlgoConfig
    """
    with open(file_path, 'r') as file:
        algo_config_info = json.load(file)
    return _load_quant_algo_config_from_dict(algo_config_info)


def _load_pre_optimization_config_from_dict(pre_optimization_config_dict: Dict[str, Any]) -> PreQuantOptConfig:
    """
    Load pre-optimization configuration from a dictionary.

    :param pre_optimization_config_dict: A dictionary containing the pre-optimization configuration.
    :type pre_optimization_config_dict: Dict[str, Any]
    :return: The pre-optimization configuration.
    :rtype: PreQuantOptConfig
    :raises ValueError: If the configuration name is not recognized.
    """
    if pre_optimization_config_dict["name"] == "rotation":
        return cast(PreQuantOptConfig, RotationConfig.from_dict(pre_optimization_config_dict))
    elif pre_optimization_config_dict["name"] == "quarot":
        return cast(PreQuantOptConfig, QuaRotConfig.from_dict(pre_optimization_config_dict))
    elif pre_optimization_config_dict["name"] == "smooth":
        return cast(PreQuantOptConfig, SmoothQuantConfig.from_dict(pre_optimization_config_dict))
    else:
        raise ValueError(f"Unknown algorithm name {pre_optimization_config_dict['name']}")


def _load_quant_algo_config_from_dict(algo_config_dict: Dict[str, Any]) -> AlgoConfig:
    """
    Load quantization algorithm configuration from a dictionary.

    :param algo_config_dict: A dictionary containing the quantization algorithm configuration.
    :type algo_config_dict: Dict[str, Any]
    :return: The quantization algorithm configuration.
    :rtype: AlgoConfig
    :raises ValueError: If the configuration name is not recognized.
    """
    if algo_config_dict["name"] == "awq":
        return cast(AlgoConfig, AWQConfig.from_dict(algo_config_dict))
    elif algo_config_dict["name"] == "gptq":  # pragma: no cover
        return cast(AlgoConfig, GPTQConfig.from_dict(algo_config_dict))
    elif algo_config_dict["name"] == "autosmoothquant":  # pragma: no cover:
        return cast(AlgoConfig, AutoSmoothQuantConfig.from_dict(algo_config_dict))
    else:
        raise ValueError(f"Unknown algorithm name {algo_config_dict['name']}")


@dataclass
class AlgoConfigBase(ConfigBase):
    pass


@dataclass
class PreQuantOptConfig(AlgoConfigBase):
    pass


@dataclass
class AlgoConfig(AlgoConfigBase):
    pass


@dataclass
class SmoothQuantConfig(PreQuantOptConfig):
    """
    A data class that defines the specifications for Smooth Quantization.

    :param str name: The name of the configuration, typically used to identify different quantization settings. Default is "smoothquant".
    :param int alpha: The factor of adjustment in the quantization formula, influencing how aggressively weights are quantized. Default is 1.
    :param float scale_clamp_min: The minimum scaling factor to be used during quantization, preventing the scale from becoming too small. Default is 1e-3.
    :param List[Dict[str, str]] scaling_layers: Specific settings for scaling layers, allowing customization of quantization parameters for different layers within the model. Default is None.
    :param str model_decoder_layers: Specifies any particular decoder layers in the model that might have unique quantization requirements. Default is None.
    """
    name: str = "smooth"
    alpha: float = 1
    scale_clamp_min: float = 1e-3
    scaling_layers: List[Dict[str, str]] = field(default_factory=list)
    model_decoder_layers: str = field(default_factory=str)
    # For GQA models, num_attention_heads and num_key_value_heads must be set in config file,
    # otherwise o_project in GQA will not be smoothed (other linears are still smoothed).
    num_attention_heads: int = -1  # o_project in GQA will not be smoothed if num_attention_heads is not set in config file.
    num_key_value_heads: int = -1  # o_project in GQA will not be smoothed if num_attention_heads is not set in config file.


@dataclass
class RotationConfig(PreQuantOptConfig):
    """
    A data class that defines the specifications for rotation settings in processing algorithms.

    :param str name: The name of the configuration, typically used to identify different rotation settings. Default is "rotation".
    :param bool random: A boolean flag indicating whether the rotation should be applied randomly. This can be useful for data augmentation purposes where random rotations may be required. Default is False.
    :param List[Dict[str, str]] scaling_layers: Specific settings for scaling layers, allowing customization of quantization parameters for different layers within the model. Default is None.
    """
    name: str = "rotation"
    random: bool = False
    scaling_layers: List[Dict[str, str]] = field(default_factory=list)


@dataclass
class QuaRotConfig(RotationConfig):
    """
    A data class that defines the specifications for the QuaRot algorithm.
    :param str name: The name of the configuration, typically used to identify different rotation settings. Default is "quarot".
    :param bool random: A boolean flag indicating whether R1 should be applied randomly. This can be useful for data augmentation purposes where random rotations may be required. Default is False.
    :param bool random2: A boolean flag indicating whether R2 should be applied randomly. This can be useful for data augmentation purposes where random rotations may be required. Default is False.
        random and random2 are only relevant if we are using Hadamard rotations for R1 and R2. If optimized_rotation_path specified,
        then we will load R1 and R2 matrices from a file instad of using Hadamard matrices.
    :param List[Dict[str, str]] scaling_layers: Specific settings for scaling layers, allowing customization of quantization parameters for different layers within the model. Default is None.
    :param bool had: A boolean flag indicating whether online hadamard operations R3 and R4 should be performed.
    :param Optional[str] optimized_rotation_path: The path to the file 'R.bin' that has saved optimized R1 and (per decoder) R2 matrices.
        If this is specified, R1 and R2 rotations will be loaded from this file. Otherwise they will be Hadamard matrices.
    :param bool kv_cache_quant: A boolean flag indicating whether there is kv-cache quantization. R3 rotation is applied only if there is.
    :param bool act_quant: A boolean flag indicating whether there is kv-cache quantization. R3 rotation is applied only if there is.
    :param str backbone: A string indicating the path to the model backbone.
    :param str model_decoder_layers: A string indicating the path to the list of decoder layers.
    :param str v_proj: A string indicating the path to the v projection layer, starting from the decoder layer it is in.
    :param str o_proj: A string indicating the path to the o projection layer, starting from the decoder layer it is in.
    :param str self_attn: A string indicating the path to the self attention block, starting from the decoder layer it is in.
    :param str mlp: A string indicating the path to the multilayer perceptron layer, starting from the decoder layer it is in.
    """
    name: str = "quarot"
    random: bool = False
    random2: bool = False
    scaling_layers: List[Dict[str, str]] = field(default_factory=list)
    online_had: bool = True
    optimized_rotation_path: Optional[str] = None
    kv_cache_quant: bool = True
    act_quant: bool = True
    backbone: str = "model"
    model_decoder_layers: str = "model.layers"
    v_proj: str = "self_attn.v_proj"
    o_proj: str = "self_attn.o_proj"
    self_attn: str = "self_attn"
    mlp: str = "mlp"


@dataclass
class AutoSmoothQuantConfig(AlgoConfig):
    """
    A data class that defines the specifications for AutoSmoothQuant.

    :param str name: The name of the quantization configuration. Default is "autosmoothquant".
    :param List[Dict[str, str]] scaling_layers: Configuration details for scaling layers within the model, specifying custom scaling parameters per layer. Default is None.
    :param str compute_scale_loss: Calculate the best scale loss, "MSE" or "MAE". Default is "MSE".
    :param str model_decoder_layers: Specifies the layers involved in model decoding that may require different quantization parameters. Default is None.
    """
    name: str = "autosmoothquant"
    scaling_layers: Optional[List[Dict[str, str]]] = None
    model_decoder_layers: Optional[str] = None
    compute_scale_loss: Optional[str] = "MSE"
    # For GQA models, num_attention_heads and num_key_value_heads must be set in config file,
    # otherwise o_project in GQA will not be smoothed (other linears are still smoothed).
    num_attention_heads: int = -1  # o_project in GQA will not be smoothed if num_attention_heads is not set in config file.
    num_key_value_heads: int = -1  # o_project in GQA will not be smoothed if num_attention_heads is not set in config file.


@dataclass
class AWQConfig(AlgoConfig):
    """
    Configuration for Activation-aware Weight Quantization (AWQ).

    :param str name: The name of the quantization configuration. Default is "awq".
    :param List[Dict[str, str]] scaling_layers: Configuration details for scaling layers within the model, specifying custom scaling parameters per layer. Default is None.
    :param str model_decoder_layers: Specifies the layers involved in model decoding that may require different quantization parameters. Default is None.
    """
    name: str = "awq"
    scaling_layers: List[Dict[str, str]] = field(default_factory=list)
    model_decoder_layers: str = field(default_factory=str)
    # For GQA models, num_attention_heads and num_key_value_heads must be set in config file,
    # otherwise o_project in GQA will not be smoothed (other linears are still smoothed).
    num_attention_heads: int = -1  # o_project in GQA will not be smoothed if num_attention_heads is not set in config file.
    num_key_value_heads: int = -1  # o_project in GQA will not be smoothed if num_attention_heads is not set in config file.


@dataclass
class GPTQConfig(AlgoConfig):
    """
    A data class that defines the specifications for Accurate Post-Training Quantization for Generative Pre-trained Transformers (GPTQ).

    :param str name: The configuration name. Default is "gptq".
    :param float damp_percent: The percentage used to dampen the quantization effect, aiding in the maintenance of accuracy post-quantization. Default is 0.01.
    :param bool desc_act: Indicates whether descending activation is used, typically to enhance model performance with quantization. Default is True.
    :param bool static_groups: Specifies whether the order of groups for quantization are static or can be dynamically adjusted. Default is True. Quark export only support static_groups as True.
    :param bool true_sequential: Indicates whether the quantization should be applied in a truly sequential manner across the layers. Default is True.
    :param List[str] inside_layer_modules: Lists the names of internal layer modules within the model that require specific quantization handling. Default is None.
    :param str model_decoder_layers: Specifies custom settings for quantization on specific decoder layers of the model. Default is None.
    """
    name: str = "gptq"
    damp_percent: float = 0.01
    desc_act: bool = True
    static_groups: bool = True
    true_sequential: bool = True
    inside_layer_modules: List[str] = field(default_factory=list)
    model_decoder_layers: str = field(default_factory=str)
