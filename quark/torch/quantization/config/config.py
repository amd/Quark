#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization Config API for PyTorch"""

from __future__ import annotations

import json
from abc import abstractmethod
from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch.nn as nn

if TYPE_CHECKING:
    from quark.torch.quantization.config.template import LLMTemplate

from quark.common.config import BaseAlgoConfig, BaseConfigImpl, BaseQConfig, BaseQLayerConfig, BaseQTensorConfig
from quark.common.utils.doc import add_start_docstring
from quark.common.utils.log import ScreenLogger
from quark.torch.quantization.config.type import (
    ALL_DATA_TYPES,
    DeviceType,
    Dtype,
    QSchemeType,
    QuantizationMode,
    RoundType,
    ScaleType,
    TQTThresholdInitMeth,
    ZeroPointType,
)
from quark.torch.quantization.config.utils import dataclass_pretty_string
from quark.torch.quantization.constants import ONLY_DTYPE_CHANGE, QUARK_LAYER_TYPES, USING_NON_SCALED_QUANT
from quark.torch.quantization.observer import (
    OBSERVER_CLASSES,
    OBSERVER_MAP,
    PER_CHANNEL_OBSERVERS,
    PER_GROUP_OBSERVERS,
    PER_TENSOR_OBSERVERS,
    ObserverBase,
    PerBlockMXDiffsObserver,
    PerBlockMXObserver,
    PerChannelMinMaxObserver,
    PerGroupMinMaxObserver,
    PerTensorHistogramObserver,
    PerTensorHistogramObserverPro,
    PerTensorMinMaxObserver,
    PerTensorMSEObserver,
    PerTensorPercentileObserver,
    PlaceholderObserver,
)

logger = ScreenLogger(__name__)

DATA_TYPE_SPEC_DOCSTRING = r"""Helper class to define a :py:class:`.QTensorConfig` using {0}.

Example:

.. code-block:: python

    quantization_spec = {1}({2}).to_quantization_spec()
"""

PER_TENSOR_OBSERVER_METHOD_MAP: dict[str, type[ObserverBase]] = {
    "min_max": PerTensorMinMaxObserver,
    "histogram": PerTensorHistogramObserver,
    "histogrampro": PerTensorHistogramObserverPro,
    "MSE": PerTensorMSEObserver,
    "percentile": PerTensorPercentileObserver,
}

SCALE_TYPE_MAP = {
    "float32": ScaleType.float32,
    "float16": ScaleType.float16,
    "bfloat16": ScaleType.bfloat16,
    "float": ScaleType.float,
    "power_of_2": ScaleType.pof2,
    "float8_e5m3": ScaleType.float8_e5m3,
}

ROUND_METHOD_MAP = {"round": RoundType.round, "floor": RoundType.floor, "half_even": RoundType.half_even}

ZERO_POINT_TYPE_MAP = {"int32": ZeroPointType.int32, "float32": ZeroPointType.float32}


def get_per_tensor_observer(observer_method: str | None = None) -> type[ObserverBase] | None:
    if observer_method:
        if observer_method not in PER_TENSOR_OBSERVER_METHOD_MAP:
            raise ValueError(
                f"Invalid observer_method '{observer_method}'. "
                f"Valid options: {list(PER_TENSOR_OBSERVER_METHOD_MAP.keys())}"
            )
        observer_cls = PER_TENSOR_OBSERVER_METHOD_MAP[observer_method]
    else:
        observer_cls = None
    return observer_cls


def get_scale_type(scale_type: str | None = None) -> ScaleType | None:
    if scale_type:
        if scale_type not in SCALE_TYPE_MAP:
            raise ValueError(f"Invalid scale_type '{scale_type}'. Valid options: {list(SCALE_TYPE_MAP.keys())}")
        ret = SCALE_TYPE_MAP[scale_type]
    else:
        ret = None
    return ret


def get_round_method(round_method: str | None = None) -> RoundType | None:
    if round_method:
        if round_method not in ROUND_METHOD_MAP:
            raise ValueError(f"Invalid round_method '{round_method}'. Valid options: {list(ROUND_METHOD_MAP.keys())}")
        ret = ROUND_METHOD_MAP[round_method]
    else:
        ret = None
    return ret


def get_zero_point_type(zero_point_type: str | None = None) -> ZeroPointType | None:
    if zero_point_type:
        if zero_point_type not in ZERO_POINT_TYPE_MAP:
            raise ValueError(
                f"Invalid zero_point_type '{zero_point_type}'. Valid options: {list(ZERO_POINT_TYPE_MAP.keys())}"
            )
        ret = ZERO_POINT_TYPE_MAP[zero_point_type]
    else:
        ret = None
    return ret


QCT = TypeVar("QCT", bound="QConfig")


@dataclass(eq=True)
class QConfig(BaseConfigImpl, BaseQConfig):
    """
    A class that encapsulates comprehensive quantization configurations for a machine learning model, allowing for detailed and hierarchical control over quantization parameters across different model components.

    :param QLayerConfig global_quant_config: Global quantization configuration applied to the entire model unless overridden at the layer level.
    :param Dict[torch.nn.Module, QLayerConfig] layer_type_quant_config: A dictionary mapping from layer types (e.g., nn.Conv2d, nn.Linear) to their quantization configurations.
    :param Dict[str, QLayerConfig] layer_quant_config: A dictionary mapping from layer names to their quantization configurations, allowing for per-layer customization. Default is ``{}``.
    :param Dict[str, QLayerConfig] kv_cache_quant_config: A dictionary mapping from layer names to kv_cache quantization configurations. Default is ``{}``.
    :param Optional[QTensorConfig] softmax_quant_spec: A quantization specifications of nn.functional.softmax output. Default is ``None``.
    :param List[str] exclude: A list of layer names to be excluded from quantization, enabling selective quantization of the model. Default is ``[]``.
    :param Optional[AlgoConfig] algo_config: Optional configuration for the quantization algorithm, such as GPTQ, AWQ and Qronos. After this process, the datatype/fake_datatype of weights will be changed with quantization scales. Default is ``None``.
    :param QuantizationMode quant_mode: The quantization mode to be used (``eager_mode`` or ``fx_graph_mode``). Default is ``QuantizationMode.eager_mode``.
    :param bool sync_moe_expert_input_amax: Whether to synchronize the post-calibration
        input amax across experts of the same projection name within each MoE layer.
        When enabled, Quark updates the observer ranges first and then recomputes the
        quantizer qparams from those synced ranges. Helpers such as ``LLMTemplate``
        may enable this flag by default for schemes that require MoE expert input
        synchronization, such as ``nvfp4``. Default is ``False``.
    :param bool keep_prequantized_layers: Export behavior for pre-quantized layers
        excluded from Quark quantization. ``True`` (default) preserves the source
        quantized bytes via :py:class:`QParamsLinear`; ``False`` exports them as
        dequantized bf16 ``nn.Linear``. Also gates pre-quantized routing at the start
        of quantization. The ``False`` path retains the legacy dequantize-on-export
        behavior and can be removed if it proves unnecessary.
    """

    # Note: `global_quant_config`, `exclude`, `algo_config`, `log_severity_level`, `version` are inherited from `BaseQConfig`

    # A dictionary mapping from layer types (e.g., nn.Conv2d, nn.Linear) to their quantization configurations.
    layer_type_quant_config: dict[type[nn.Module], QLayerConfig] = field(default_factory=dict)

    # A dictionary mapping from layer names to their quantization configurations, allowing for per-layer customization.
    layer_quant_config: dict[str, QLayerConfig] = field(default_factory=dict)

    # A dictionary mapping from layer names to kv_cache quantization configurations.
    kv_cache_quant_config: dict[str, QLayerConfig] = field(default_factory=dict)

    # A list of layer names to be grouped for kv_cache quantization, enabling per-group customization.
    kv_cache_group: list[str] = field(default_factory=list)

    # The minimum scale of kv_cache quantization.
    min_kv_scale: float = 0.0

    # Control whether KV-cache quantization is applied post-RoPE (inside HF Cache.update).
    # When False (default), legacy pre-RoPE behaviour applies (module-level K/V output quantizers).
    kv_cache_post_rope: bool = False

    # Groups of layer name suffixes that should share the global-scale quantizer.
    # Each inner list represents a group of parallel layer suffixes
    # (e.g. ``["q_proj", "k_proj", "v_proj"]``).
    # The system recursively finds sibling layers ending with these suffixes and
    # shares observers among them so they produce a single unified scale.
    shared_scale_groups: list[list[str]] = field(default_factory=list)

    # A quantization specifications of nn.functional.softmax output.
    softmax_quant_spec: QTensorConfig | None = None

    # The quantization mode to be used (eager_mode or fx_graph_mode)
    quant_mode: QuantizationMode = QuantizationMode.eager_mode

    # Synchronize post-calibration input amax across MoE experts that share the
    # same projection name inside a single MoE layer. This is configured
    # explicitly or by helpers such as `LLMTemplate` for specific schemes.
    sync_moe_expert_input_amax: bool = False

    keep_prequantized_layers: bool = True

    def to_dict(self) -> dict[str, Any]:
        config_dict: dict[str, Any] = {
            "global_quant_config": self.global_quant_config.to_dict(),
            "exclude": self.exclude,
            "algo_config": [config.to_dict() for config in self.algo_config] if self.algo_config is not None else None,
            "softmax_quant_spec": self.softmax_quant_spec.to_dict() if self.softmax_quant_spec is not None else None,
            "quant_method": "quark",
        }

        layer_type_quant_config_dict: dict[str, Any] = {}
        for layer_type, config in self.layer_type_quant_config.items():
            layer_type_quant_config_dict[layer_type.__name__] = config.to_dict()
        config_dict["layer_type_quant_config"] = layer_type_quant_config_dict

        layer_quant_config_dict: dict[str, Any] = {}
        for name, config in self.layer_quant_config.items():
            layer_quant_config_dict[name] = config.to_dict()
        config_dict["layer_quant_config"] = layer_quant_config_dict

        kv_cache_quant_config_dict: dict[str, Any] = {}
        for name, config in self.kv_cache_quant_config.items():
            kv_cache_quant_config_dict[name] = config.to_dict()
        config_dict["kv_cache_quant_config"] = kv_cache_quant_config_dict

        # Only serialize kv_cache_post_rope - it's needed to determine cache integration behavior during import
        # kv_cache_group and min_kv_scale are export-time processing parameters and not needed for import
        config_dict["kv_cache_post_rope"] = self.kv_cache_post_rope

        config_dict["quant_mode"] = self.quant_mode.name

        config_dict["version"] = self.version

        return config_dict

    def __str__(self) -> str:
        s = dataclass_pretty_string(self)
        return s

    @classmethod
    def from_dict(cls: type[QCT], config_dict: dict[str, Any]) -> QCT:
        global_quant_config = QLayerConfig.from_dict(config_dict["global_quant_config"])

        # TODO: Deprecate legacy configuration and remove the None check here.
        # Legacy (quark<1.0) configuration used to allow layer_type_quant_config=None in the serialized config, inconstitant with
        # the type hints of the dataclass.
        layer_type_quant_config = {}
        if config_dict["layer_type_quant_config"] is not None:
            for layer_type_name, layer_type_quantization_config in config_dict["layer_type_quant_config"].items():
                if layer_type_name in QUARK_LAYER_TYPES:
                    layer_type_quant_config[QUARK_LAYER_TYPES[layer_type_name]] = QLayerConfig.from_dict(
                        layer_type_quantization_config
                    )
                else:
                    raise NotImplementedError(
                        f"Quark does not support reloading a quantization `Config` from a dictionary using custom `layer_type_quantization_config`. Found `'{layer_type_name}'` in `layer_type_quantization_config`, which is not among the supported {QUARK_LAYER_TYPES}."
                    )

        # TODO: Deprecate legacy configuration and remove the None check here.
        # Legacy (quark<1.0) configuration used to allow layer_quant_config=None in the serialized config, inconstitant with
        # the type hints of the dataclass.
        if config_dict["layer_quant_config"] is not None:
            layer_quant_config = {
                layer_name: QLayerConfig.from_dict(quant_config_dict)
                for layer_name, quant_config_dict in config_dict["layer_quant_config"].items()
            }
        else:
            layer_quant_config = {}

        if config_dict.get("kv_cache_quant_config") is not None:
            kv_cache_quant_config = {
                kv_cache_name: QLayerConfig.from_dict(kv_cache_config_dict)
                for kv_cache_name, kv_cache_config_dict in config_dict["kv_cache_quant_config"].items()
            }
        else:
            kv_cache_quant_config = {}

        # TODO: Deprecate legacy (quark<1.0) configuration and remove the check here.
        # `exclude` used to be serialized as `None` when there was no exclude layer, instead of `[]`.
        if config_dict["exclude"] is None:  # pragma: no cover
            exclude = []
        else:
            exclude = config_dict["exclude"]

        if "algo_config" in config_dict and config_dict["algo_config"] is not None:
            if isinstance(config_dict["algo_config"], list):  # new config
                algo_config = [_load_quant_algo_config_from_dict(config) for config in config_dict["algo_config"]]
            else:  # old config
                algo_config = [_load_quant_algo_config_from_dict(config_dict["algo_config"])]
        else:
            algo_config = None

        # Get softmax_quant_spec configuration from config_dict
        softmax_quant_spec = (
            QTensorConfig.from_dict(config_dict["softmax_quant_spec"])
            if ("softmax_quant_spec" in config_dict and config_dict["softmax_quant_spec"] is not None)
            else None
        )

        if "quant_mode" in config_dict:
            quant_mode = QuantizationMode[config_dict["quant_mode"]]  # Access by name and not by value.
        else:
            # TODO: Deprecate legacy (quark<1.0) configuration and remove the check here.
            # The key `"quant_mode"` used not to be serialized in the legacy quantization_config, inconstitant with
            # the type hints of the dataclass.
            quant_mode = QuantizationMode.eager_mode

        # get version from config_dict, if not found (e.g. models exported with amd-quark<=0.8), set it to `None`.
        version = config_dict["version"] if "version" in config_dict else None

        kv_cache_post_rope = config_dict.get("kv_cache_post_rope", False)

        return cls(
            global_quant_config=global_quant_config,
            layer_type_quant_config=layer_type_quant_config,
            layer_quant_config=layer_quant_config,
            kv_cache_quant_config=kv_cache_quant_config,
            kv_cache_post_rope=kv_cache_post_rope,
            exclude=exclude,
            algo_config=algo_config,
            softmax_quant_spec=softmax_quant_spec,
            quant_mode=quant_mode,
            version=version,
        )

    @staticmethod
    def with_llm_template(
        template: LLMTemplate,
        scheme: str,
        algorithm: str | None = None,
        kv_cache_scheme: str | None = None,
        min_kv_scale: float = 0.0,
        attention_scheme: str | None = None,
        layer_config: dict[str, str] | None = None,
        layer_type_config: dict[type[nn.Module], str] | None = None,
        exclude_layers: list[str] | None = None,
    ) -> QConfig:
        return template.get_config(
            scheme=scheme,
            algorithm=algorithm,
            kv_cache_scheme=kv_cache_scheme,
            min_kv_scale=min_kv_scale,
            attention_scheme=attention_scheme,
            layer_config=layer_config,
            layer_type_config=layer_type_config,
            exclude_layers=exclude_layers,
        )

    def __post_init__(self) -> None:
        # When uses want to exclude *.gate, they actually want to all the sublayers of the gate layer.
        # So exlude = ["*.gate"] can be expanded as ["*.gate", "*.gate.*"]
        new_exclude = []
        for item in self.exclude:
            if not item.endswith(".*"):
                logger.warning(f"Expanding exclude pattern {[item]} to {[item, item + '.*']}")
                new_exclude.append(item + ".*")
            new_exclude.append(item)
        self.exclude = new_exclude
        if self.algo_config is not None:
            for algo_config in self.algo_config:
                if algo_config.name == "rotation":
                    algo_config_cls_name = algo_config.__class__.__name__
                    if len(self.kv_cache_quant_config) > 0 and not algo_config.r3:  # type: ignore
                        logger.warning(
                            f"The R3 rotation is disabled, but the KV cache is configured to quantized with: {self.kv_cache_quant_config}. KV cache quantization may benefit from R3 rotation if keys are quantized. Consider using `r3=True` in {algo_config_cls_name} configuration."
                        )

                    if len(self.kv_cache_quant_config) == 0 and algo_config.r3:  # type: ignore
                        logger.warning(
                            f"No KV cache quantization configuration provided, but `{algo_config_cls_name}.r3` is set to `True`. This setting is only useful in case KV cache quantization is used. Consider using `r3=False`."
                        )

    def get_rotation_config(self) -> RotationConfig | None:
        rotation_config = None
        if self.algo_config is not None:
            for algo_config in self.algo_config:
                if isinstance(algo_config, RotationConfig):
                    rotation_config = algo_config
                    break

        return rotation_config


QLT = TypeVar("QLT", bound="QLayerConfig")


@dataclass(eq=True)
class QLayerConfig(BaseQLayerConfig):
    """
    A data class that specifies quantization configurations for different components of a module, allowing hierarchical control over how each tensor type is quantized.

    :param Optional[Union[QTensorConfig, List[QTensorConfig]]] input_tensors: Input tensors quantization specification. If None, following the hierarchical quantization setup. e.g. If the ``input_tensors`` in ``layer_type_quant_config`` is ``None``, the configuration from ``global_quant_config`` will be used instead. Defaults to ``None``. If None in ``global_quant_config``, ``input_tensors`` are not quantized.
    :param Optional[Union[QTensorConfig, List[QTensorConfig]]] output_tensors: Output tensors quantization specification. Defaults to ``None``. If ``None``, the same as above.
    :param Optional[Union[QTensorConfig, List[QTensorConfig]]] weight: The weights tensors quantization specification. Defaults to ``None``. If ``None``, the same as above.
    :param Optional[Union[QTensorConfig, List[QTensorConfig]]] bias: The bias tensors quantization specification. Defaults to ``None``. If ``None``, the same as above.
    :param Optional[DeviceType] target_device: Configuration specifying the target device (e.g., CPU, GPU, IPU) for the quantized model.
    """

    # Note: `input_tensors`, ou`tput_tensors, wei`ght, and bi`as are inherited from `BaseQLayerConfig`
    # with type Union[BaseQTensorConfig, list[BaseQTensorConfig]] | None
    # They are not redeclared here to avoid type incompatibility issues

    target_device: DeviceType | None = None

    def __post_init__(self) -> None:
        for tensor_name, quantization_spec in [
            ("input_tensors", self.input_tensors),
            ("output_tensors", self.output_tensors),
        ]:
            if quantization_spec is not None:
                if isinstance(quantization_spec, QTensorConfig):
                    quantization_spec = [quantization_spec]
                if not isinstance(quantization_spec, list):
                    raise TypeError(
                        f"quantization_spec for '{tensor_name}' must be a QTensorConfig or list of QTensorConfig, "
                        f"got {type(quantization_spec).__name__}."
                    )
                for quant_spec in quantization_spec:
                    if quant_spec.qscheme == QSchemeType.per_group and not quant_spec.is_dynamic:
                        raise ValueError(
                            f"The parameterization `qscheme=QSchemeType.per_group` along with `is_dynamic=False` is currently not supported in AMD Quark for `QLayerConfig.input_tensors` and `QLayerConfig.output_tensors`, got `QLayerConfig.{tensor_name}.qscheme={quant_spec.qscheme}` and `QLayerConfig.{tensor_name}.is_dynamic={quant_spec.is_dynamic}`. Consider using `QLayerConfig.{tensor_name}.is_dynamic=True` as static per-group quantization for activations is rarely relevant."
                        )

    def to_dict(self) -> dict[str, Any]:
        def convert_spec_to_dict(
            spec: QTensorConfig | list[QTensorConfig] | None,
        ) -> dict[str, Any] | list[dict[str, Any]] | None:
            if spec is None:
                return None
            elif isinstance(spec, list):
                return [s.to_dict() for s in spec]
            else:
                return spec.to_dict()

        return {
            "input_tensors": convert_spec_to_dict(self.input_tensors),
            "output_tensors": convert_spec_to_dict(self.output_tensors),
            "weight": convert_spec_to_dict(self.weight),
            "bias": convert_spec_to_dict(self.bias),
            "target_device": self.target_device.value if self.target_device is not None else None,
        }

    @classmethod
    def from_dict(cls: type[QLT], quantization_config: dict[str, Any]) -> QLT:
        def convert_dict_to_spec(
            config: dict[str, Any] | list[dict[str, Any]] | None,
        ) -> QTensorConfig | list[QTensorConfig] | None:
            if config is None:
                return None
            elif isinstance(config, list):
                specs = [QTensorConfig.from_dict(c) for c in config]
                if not all(spec is not None for spec in specs):
                    raise ValueError(
                        "All quantization specs in the config must be valid (not None). "
                        "Check that the config dict contains valid QTensorConfig data."
                    )
                # After verification, all items are guaranteed to be QTensorConfig
                return specs  # type: ignore[return-value]
            else:
                return QTensorConfig.from_dict(config)

        input_tensors = convert_dict_to_spec(quantization_config["input_tensors"])
        output_tensors = convert_dict_to_spec(quantization_config["output_tensors"])
        weight = convert_dict_to_spec(quantization_config["weight"])
        bias = convert_dict_to_spec(quantization_config["bias"])

        # TODO: Deprecate legacy configuration.
        # Legacy (quark<1.0) saved quantization_config does not have the key `"target_device"`.
        target_device = quantization_config["target_device"] if "target_device" in quantization_config else None
        target_device = DeviceType(target_device) if target_device is not None else None

        return cls(
            input_tensors=input_tensors,
            output_tensors=output_tensors,
            weight=weight,
            bias=bias,
            target_device=target_device,
        )


@dataclass
class TwoStageSpec(BaseConfigImpl):
    """
    A data class that specifies two-stage quantization configurations for different components of a module,
    allowing hierarchical control over how each tensor type is quantized.
    """

    first_stage: DataTypeSpec | QTensorConfig
    second_stage: DataTypeSpec | QTensorConfig

    @abstractmethod
    def to_quantization_spec(self) -> list[QTensorConfig]:
        pass


@dataclass
class ProgressiveSpec(TwoStageSpec):
    """
    A data class that specifies a progressive quantization specification for a tensor.
    The first stage quantizes the input tensor, while the second stage quantizes the output from the first stage.

    For example, to progressively quantize a float16 tensor:

    1. First quantize it to fp8_e4m3 using fp8_e4m3 per-tensor quantization, get a fp8_e4m3 tensor.
    2. Then quantize the fp8_e4m3 tensor to int4 using int4 per-channel quantization, get a int4 tensor.

    The configuration for this example would be:

    .. code-block:: python

        quant_spec = ProgressiveSpec(
            first_stage=FP8E4M3PerTensorSpec(observer_method="min_max",
                                             is_dynamic=False),
            second_stage=Int4PerChannelSpec(symmetric=False,
                                            scale_type="float",
                                            round_method="half_even",
                                            ch_axis=0,
                                            is_dynamic=False)
        ).to_quantization_spec()
    """

    def to_quantization_spec(self) -> list[QTensorConfig]:
        return [
            self.first_stage.to_quantization_spec() if isinstance(self.first_stage, DataTypeSpec) else self.first_stage,
            self.second_stage.to_quantization_spec()
            if isinstance(self.second_stage, DataTypeSpec)
            else self.second_stage,
        ]


@dataclass
class ScaleQuantSpec(TwoStageSpec):
    """
    A data class that specifies a two-stage quantization process for scale quantization.

    The quantization happens in two stages:

    1. First stage quantizes the input tensor itself.
    2. Second stage quantizes the scale values from the first stage quantization.

    For example, given a float16 tensor:

    1. First quantize the tensor to fp4_e2m1 using fp4_e2m1 per-group quantization, producing a fp4_e2m1 tensor with float16 scale values.
    2. Then quantize those float16 scale values to fp8_e4m3 using fp8_e4m3 per-tensor quantization.

    The configuration for this example would be:

    .. code-block:: python

        quant_spec = ScaleQuantSpec(
            first_stage=FP4PerGroupSpec(group_size=16, is_dynamic=False),
            second_stage=FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False)
        ).to_quantization_spec()
    """

    def to_quantization_spec(self) -> list[QTensorConfig]:
        second_stage_spec = (
            self.second_stage.to_quantization_spec()
            if isinstance(self.second_stage, DataTypeSpec)
            else self.second_stage
        )
        second_stage_spec.is_scale_quant = True
        return [
            self.first_stage.to_quantization_spec() if isinstance(self.first_stage, DataTypeSpec) else self.first_stage,
            second_stage_spec,
        ]


class DataTypeSpec(BaseConfigImpl):
    @abstractmethod
    def to_quantization_spec(self) -> QTensorConfig:
        pass


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "uint4 per tensor quantization", "Uint4PerTensorSpec", "is_dynamic=True, symmetric=False"
    )
)
class Uint4PerTensorSpec(DataTypeSpec):
    observer_method: str | None = "min_max"
    symmetric: bool | None = False
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.uint4,
            observer_cls=get_per_tensor_observer(self.observer_method),
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_tensor,
            is_dynamic=self.is_dynamic,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "uint4 per channel quantization",
        "Uint4PerChannelSpec",
        r"""
        symmetric=True,
        scale_type="float",
        round_method="half_even",
        ch_axis=0,
        is_dynamic=False
    """,
    )
)
class Uint4PerChannelSpec(DataTypeSpec):
    ch_axis: int
    symmetric: bool | None = False
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = None
    zero_point_type: str | None = "int32"

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.uint4,
            observer_cls=PerChannelMinMaxObserver,
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_channel,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
            zero_point_type=get_zero_point_type(self.zero_point_type),
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "uint4 per group quantization",
        "Uint4PerGroupSpec",
        r"""
        symmetric=False,
        scale_type="float",
        round_method="half_even",
        ch_axis=1,
        is_dynamic=False,
        group_size=128
    """,
    )
)
class Uint4PerGroupSpec(DataTypeSpec):
    ch_axis: int
    group_size: int
    symmetric: bool = False
    is_dynamic: bool | None = False
    scale_type: str | None = "float"
    round_method: str | None = "half_even"

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.uint4,
            observer_cls=PerGroupMinMaxObserver,
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_group,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
            group_size=self.group_size,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "int3 per group quantization",
        "Int3PerGroupSpec",
        r"""
        symmetric=True,
        scale_type="float",
        round_method="half_even",
        is_dynamic=False,
        group_size=32,
    """,
    )
)
class Int3PerGroupSpec(DataTypeSpec):
    ch_axis: int
    group_size: int
    symmetric: bool | None = True
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.int3,
            observer_cls=PerGroupMinMaxObserver,
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_group,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
            group_size=self.group_size,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "int3 per channel quantization",
        "Int3PerChannelSpec",
        r"""
        symmetric=False,
        scale_type="float",
        round_method="half_even",
        ch_axis=0,
        is_dynamic=False
    """,
    )
)
class Int3PerChannelSpec(DataTypeSpec):
    ch_axis: int
    symmetric: bool | None = False
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.int3,
            observer_cls=PerChannelMinMaxObserver,
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_channel,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "int2 per group quantization",
        "Int2PerGroupSpec",
        r"""
        symmetric=True,
        scale_type="float",
        round_method="half_even",
        is_dynamic=False,
        group_size=32,
    """,
    )
)
class Int2PerGroupSpec(DataTypeSpec):
    ch_axis: int
    group_size: int
    symmetric: bool | None = True
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.int2,
            observer_cls=PerGroupMinMaxObserver,
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_group,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
            group_size=self.group_size,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "int4 per tensor quantization",
        "Int4PerTensorSpec",
        r"""
        observer_method="min_max",
        symmetric=True,
        scale_type="float",
        round_method="half_even",
        is_dynamic=False
    """,
    )
)
class Int4PerTensorSpec(DataTypeSpec):
    observer_method: str | None = "min_max"
    symmetric: bool | None = True
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.int4,
            observer_cls=get_per_tensor_observer(self.observer_method),
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_tensor,
            is_dynamic=self.is_dynamic,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "int4 per channel quantization",
        "Int4PerChannelSpec",
        r"""
        symmetric=False,
        scale_type="float",
        round_method="half_even",
        ch_axis=0,
        is_dynamic=False
    """,
    )
)
class Int4PerChannelSpec(DataTypeSpec):
    ch_axis: int
    symmetric: bool | None = True
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.int4,
            observer_cls=PerChannelMinMaxObserver,
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_channel,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "int4 per group quantization",
        "Int4PerGroupSpec",
        r"""
        symmetric=True,
        scale_type="float",
        round_method="half_even",
        ch_axis=1,
        is_dynamic=False,
        group_size=128
    """,
    )
)
class Int4PerGroupSpec(DataTypeSpec):
    ch_axis: int
    group_size: int
    symmetric: bool = True
    is_dynamic: bool | None = False
    scale_type: str | None = "float"
    round_method: str | None = "half_even"

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.int4,
            observer_cls=PerGroupMinMaxObserver,
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_group,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
            group_size=self.group_size,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "uint8 per tensor quantization",
        "Uint8PerTensorSpec",
        r"""
        observer_method="percentile",
        symmetric=True,
        scale_type="float",
        round_method="half_even",
        is_dynamic=False
    """,
    )
)
class Uint8PerTensorSpec(DataTypeSpec):
    observer_method: str | None = "min_max"
    symmetric: bool | None = False
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.uint8,
            observer_cls=get_per_tensor_observer(self.observer_method),
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_tensor,
            is_dynamic=self.is_dynamic,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "uint8 per channel quantization",
        "Uint8PerChannelSpec",
        r"""
        symmetric=True,
        scale_type="float",
        round_method="half_even",
        ch_axis=0,
        is_dynamic=False
    """,
    )
)
class Uint8PerChannelSpec(DataTypeSpec):
    ch_axis: int
    symmetric: bool | None = True
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.uint8,
            observer_cls=PerChannelMinMaxObserver,
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_channel,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "uint8 per group quantization",
        "Uint8PerGroupSpec",
        r"""
        symmetric=False,
        scale_type="float",
        round_method="half_even",
        ch_axis=1,
        is_dynamic=False,
        group_size=128
    """,
    )
)
class Uint8PerGroupSpec(DataTypeSpec):
    ch_axis: int
    group_size: int
    symmetric: bool | None = False
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.uint8,
            observer_cls=PerGroupMinMaxObserver,
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_group,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
            group_size=self.group_size,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "int8 per tensor quantization",
        "Int8PerTensorSpec",
        r"""
        observer_method="min_max",
        symmetric=True,
        scale_type="float",
        round_method="half_even",
        is_dynamic=False
    """,
    )
)
class Int8PerTensorSpec(DataTypeSpec):
    observer_method: str | None = "min_max"
    symmetric: bool | None = True
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.int8,
            observer_cls=get_per_tensor_observer(self.observer_method),
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_tensor,
            is_dynamic=self.is_dynamic,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "int8 per channel quantization",
        "Int8PerChannelSpec",
        r"""
        symmetric=False,
        scale_type="float",
        round_method="half_even",
        ch_axis=0,
        is_dynamic=False
    """,
    )
)
class Int8PerChannelSpec(DataTypeSpec):
    ch_axis: int
    symmetric: bool | None = False
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.int8,
            observer_cls=PerChannelMinMaxObserver,
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_channel,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "int8 per group quantization",
        "Int8PerGroupSpec",
        r"""
        symmetric=True,
        scale_type="float",
        round_method="half_even",
        ch_axis=1,
        is_dynamic=False,
        group_size=128
    """,
    )
)
class Int8PerGroupSpec(DataTypeSpec):
    ch_axis: int
    group_size: int
    symmetric: bool | None = True
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.int8,
            observer_cls=PerGroupMinMaxObserver,
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_group,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
            group_size=self.group_size,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "FP8E4M3 per tensor quantization",
        "FP8E4M3PerTensorSpec",
        r"""
        observer_method="min_max",
        is_dynamic=False
    """,
    )
)
class FP8E4M3PerTensorSpec(DataTypeSpec):
    observer_method: str | None = "min_max"
    scale_type: str | None = None
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp8_e4m3,
            observer_cls=get_per_tensor_observer(self.observer_method),
            symmetric=True,
            scale_type=get_scale_type(self.scale_type),
            round_method=RoundType.half_even,
            qscheme=QSchemeType.per_tensor,
            is_dynamic=self.is_dynamic,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "FP8E5M3 per tensor quantization",
        "FP8E5M3PerTensorSpec",
        r"""
        observer_method="min_max",
        is_dynamic=False
    """,
    )
)
class FP8E5M3PerTensorSpec(DataTypeSpec):
    observer_method: str | None = "min_max"
    scale_type: str | None = None
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp8_e5m3,
            observer_cls=get_per_tensor_observer(self.observer_method),
            symmetric=True,
            scale_type=get_scale_type(self.scale_type),
            round_method=RoundType.half_even,
            qscheme=QSchemeType.per_tensor,
            is_dynamic=self.is_dynamic,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "FP8E4M3 per channel quantization", "FP8E4M3PerChannelSpec", "is_dynamic=False, ch_axis=0"
    )
)
class FP8E4M3PerChannelSpec(DataTypeSpec):
    ch_axis: int
    symmetric: bool | None = True
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp8_e4m3,
            observer_cls=PerChannelMinMaxObserver,
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_channel,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "FP8E4M3 per group quantization",
        "FP8E4M3PerGroupSpec",
        r"""
        ch_axis=-1,
        group_size=group_size,
        is_dynamic=True
    """,
    )
)
class FP8E4M3PerGroupSpec(DataTypeSpec):
    ch_axis: int
    group_size: int
    scale_format: str | None = "float32"
    scale_calculation_mode: str | None = None
    is_dynamic: bool | None = True

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp8_e4m3,
            observer_cls=PerBlockMXObserver,
            symmetric=None,
            scale_type=ScaleType.float,
            scale_format=self.scale_format,
            scale_calculation_mode=self.scale_calculation_mode,
            round_method=RoundType.half_even,
            qscheme=QSchemeType.per_group,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
            group_size=self.group_size,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "FP8E5M2 per tensor quantization",
        "FP8E5M2PerTensorSpec",
        r"""
        observer_method="min_max",
        is_dynamic=False
    """,
    )
)
class FP8E5M2PerTensorSpec(DataTypeSpec):
    observer_method: str | None = "min_max"
    symmetric: bool | None = True
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp8_e5m2,
            observer_cls=get_per_tensor_observer(self.observer_method),
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_tensor,
            is_dynamic=self.is_dynamic,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "FP8E5M2 per channel quantization", "FP8E5M2PerChannelSpec", "is_dynamic=False, ch_axis=0"
    )
)
class FP8E5M2PerChannelSpec(DataTypeSpec):
    ch_axis: int
    symmetric: bool | None = True
    scale_type: str | None = "float"
    round_method: str | None = "half_even"
    is_dynamic: bool | None = False

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp8_e5m2,
            observer_cls=PerChannelMinMaxObserver,
            symmetric=self.symmetric,
            scale_type=get_scale_type(self.scale_type),
            round_method=get_round_method(self.round_method),
            qscheme=QSchemeType.per_channel,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "FP8E5M2 per group quantization",
        "FP8E5M2PerGroupSpec",
        r"""
        ch_axis=-1,
        group_size=group_size,
        is_dynamic=True
    """,
    )
)
class FP8E5M2PerGroupSpec(DataTypeSpec):
    ch_axis: int
    group_size: int

    scale_format: str | None = "float32"
    scale_calculation_mode: str | None = None
    is_dynamic: bool | None = True

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp8_e5m2,
            observer_cls=PerBlockMXObserver,
            symmetric=None,
            scale_type=ScaleType.float,
            scale_format=self.scale_format,
            scale_calculation_mode=self.scale_calculation_mode,
            round_method=RoundType.half_even,
            qscheme=QSchemeType.per_group,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
            group_size=self.group_size,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "FP4 per group quantization",
        "FP4PerGroupSpec",
        r"""
        ch_axis=-1,
        group_size=group_size,
        is_dynamic=True
    """,
    )
)
class FP4PerGroupSpec(DataTypeSpec):
    ch_axis: int
    group_size: int
    scale_format: str | None = "float32"
    scale_calculation_mode: str | None = None
    is_dynamic: bool | None = True
    scale_type: str | None = "float"

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp4,
            observer_cls=PerBlockMXObserver,
            symmetric=None,
            scale_type=get_scale_type(self.scale_type),
            scale_format=self.scale_format,
            scale_calculation_mode=self.scale_calculation_mode,
            round_method=RoundType.half_even,
            qscheme=QSchemeType.per_group,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
            group_size=self.group_size,
        )


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "amdfp4 per group quantization with E5M3 scale format",
        "AmdFP4Spec",
        r"""
        ch_axis=-1,
        group_size=16,
        is_dynamic=True
    """,
    )
)
class AmdFP4Spec(DataTypeSpec):
    ch_axis: int
    group_size: int
    is_dynamic: bool = True

    def __post_init__(self) -> None:
        if self.group_size not in [16, 32]:
            raise ValueError(f"amdfp4 group_size must be 16 or 32, got group_size={self.group_size}.")

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp4,
            observer_cls=PerBlockMXObserver,
            symmetric=None,
            scale_type=ScaleType.float8_e5m3,
            scale_format="e5m3",
            scale_calculation_mode=None,
            round_method=RoundType.half_even,
            qscheme=QSchemeType.per_group,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
            group_size=self.group_size,
        )


@dataclass
class FP6E2M3PerGroupSpec(DataTypeSpec):
    ch_axis: int
    group_size: int
    scale_format: str | None = "float32"
    scale_calculation_mode: str | None = None
    is_dynamic: bool | None = True

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp6_e2m3,
            observer_cls=PerBlockMXObserver,
            symmetric=None,
            scale_type=ScaleType.float,
            scale_format=self.scale_format,
            scale_calculation_mode=self.scale_calculation_mode,
            round_method=RoundType.half_even,
            qscheme=QSchemeType.per_group,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
            group_size=self.group_size,
        )


@dataclass
class FP6E3M2PerGroupSpec(DataTypeSpec):
    ch_axis: int
    group_size: int
    scale_format: str | None = "float32"
    scale_calculation_mode: str | None = None
    is_dynamic: bool | None = True

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp6_e3m2,
            observer_cls=PerBlockMXObserver,
            symmetric=None,
            scale_type=ScaleType.float,
            scale_format=self.scale_format,
            scale_calculation_mode=self.scale_calculation_mode,
            round_method=RoundType.half_even,
            qscheme=QSchemeType.per_group,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
            group_size=self.group_size,
        )


@dataclass(eq=True)
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "float16 data type. The resulting QTensorConfig does not quantize the tensor.", "Float16Spec", ""
    )
)
class Float16Spec(DataTypeSpec):
    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(dtype=Dtype.float16, observer_cls=PlaceholderObserver)


@dataclass(eq=True)
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "bfloat16 data type. The resulting QTensorConfig does not quantize the tensor.", "Bfloat16Spec", ""
    )
)
class Bfloat16Spec(DataTypeSpec):
    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(dtype=Dtype.bfloat16, observer_cls=PlaceholderObserver)


class OCP_MXSpec(DataTypeSpec):
    OCP_MX_SPEC_KWARGS = {
        "observer_cls": PerBlockMXObserver,
        "symmetric": None,
        "scale_type": ScaleType.float,
        "round_method": RoundType.half_even,
        "scale_format": "e8m0",
        "qscheme": QSchemeType.per_group,
        "group_size": 32,
    }


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "MX OCP data type using FP8E4M3",
        "OCP_MXFP8E4M3Spec",
        r"""
        ch_axis=-1,
        is_dynamic=False
    """,
    )
)
class OCP_MXFP8E4M3Spec(OCP_MXSpec):
    ch_axis: int
    is_dynamic: bool = True
    scale_calculation_mode: str = "even"

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp8_e4m3,
            scale_calculation_mode=self.scale_calculation_mode,
            is_dynamic=self.is_dynamic,
            ch_axis=self.ch_axis,
            **self.OCP_MX_SPEC_KWARGS,
        )  # type: ignore[arg-type]


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "MX OCP data type using FP8E5M2",
        "OCP_MXFP8E5M2Spec",
        r"""
        ch_axis=-1,
        is_dynamic=False
    """,
    )
)
class OCP_MXFP8E5M2Spec(OCP_MXSpec):
    ch_axis: int
    is_dynamic: bool = True
    scale_calculation_mode: str = "even"

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp8_e5m2,
            scale_calculation_mode=self.scale_calculation_mode,
            is_dynamic=self.is_dynamic,
            ch_axis=self.ch_axis,
            **self.OCP_MX_SPEC_KWARGS,
        )  # type: ignore[arg-type]


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "MX OCP data type using FP6E3M2",
        "OCP_MXFP6E3M2Spec",
        r"""
        ch_axis=-1,
        is_dynamic=False
    """,
    )
)
class OCP_MXFP6E3M2Spec(OCP_MXSpec):
    ch_axis: int
    is_dynamic: bool = True
    scale_calculation_mode: str = "even"

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp6_e3m2,
            scale_calculation_mode=self.scale_calculation_mode,
            is_dynamic=self.is_dynamic,
            ch_axis=self.ch_axis,
            **self.OCP_MX_SPEC_KWARGS,
        )  # type: ignore[arg-type]


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "MX OCP data type using FP6E2M3",
        "OCP_MXFP6E2M3Spec",
        r"""
        ch_axis=-1,
        is_dynamic=False
    """,
    )
)
class OCP_MXFP6E2M3Spec(OCP_MXSpec):
    ch_axis: int
    is_dynamic: bool = True
    scale_calculation_mode: str = "even"

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp6_e2m3,
            scale_calculation_mode=self.scale_calculation_mode,
            is_dynamic=self.is_dynamic,
            ch_axis=self.ch_axis,
            **self.OCP_MX_SPEC_KWARGS,
        )  # type: ignore[arg-type]


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "MX OCP data type using FP4",
        "OCP_MXFP4Spec",
        r"""
        ch_axis=-1,
        is_dynamic=False
    """,
    )
)
class OCP_MXFP4Spec(OCP_MXSpec):
    ch_axis: int
    is_dynamic: bool = True
    scale_calculation_mode: str = "even"

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp4,
            scale_calculation_mode=self.scale_calculation_mode,
            is_dynamic=self.is_dynamic,
            ch_axis=self.ch_axis,
            **self.OCP_MX_SPEC_KWARGS,
        )  # type: ignore[arg-type]


@dataclass
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "MX OCP data type using INT8",
        "OCP_MXINT8Spec",
        r"""
        ch_axis=-1,
        is_dynamic=False
    """,
    )
)
class OCP_MXINT8Spec(OCP_MXSpec):
    ch_axis: int
    is_dynamic: bool = True
    scale_calculation_mode: str = "even"

    # TODO: support Dtype.int8 in PerBlockMXObserver.
    # Dtype.int8 still uses NonScaledFakeQuantize (see tensor_quantize.py),
    # which it needs not to.
    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.mx,
            mx_element_dtype=Dtype.int8,
            scale_calculation_mode=self.scale_calculation_mode,
            ch_axis=self.ch_axis,
            group_size=32,
        )  # type: ignore[arg-type]


@dataclass
class OCP_MXFP4DiffsSpec(DataTypeSpec):
    ch_axis: int
    is_dynamic: bool | None = False
    scale_calculation_mode: str | None = "even"

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.fp4,
            observer_cls=PerBlockMXDiffsObserver,
            symmetric=None,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            scale_format="e8m0",
            scale_calculation_mode=self.scale_calculation_mode,
            qscheme=QSchemeType.per_group,
            ch_axis=self.ch_axis,
            is_dynamic=self.is_dynamic,
            group_size=32,
        )


@dataclass(eq=True)
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "MX6 data type as defined in https://arxiv.org/pdf/2302.08007. More details are available in the :doc:`Two Level Quantization Formats </pytorch/adv_two_level>` documentation",
        "MX6Spec",
        "is_dynamic=False",
    )
)
class MX6Spec(DataTypeSpec):
    ch_axis: int
    block_size: int
    scale_calculation_mode: str | None = "even"

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.mx6,
            ch_axis=self.ch_axis,
            group_size=self.block_size,
            scale_calculation_mode=self.scale_calculation_mode,
        )


@dataclass(eq=True)
@add_start_docstring(
    DATA_TYPE_SPEC_DOCSTRING.format(
        "MX9 data type as defined in https://arxiv.org/pdf/2302.08007. More details are available in the :doc:`Two Level Quantization Formats </pytorch/adv_two_level>` documentation",
        "MX9Spec",
        "is_dynamic=False",
    )
)
class MX9Spec(DataTypeSpec):
    ch_axis: int
    block_size: int
    scale_calculation_mode: str | None = "even"

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.mx9,
            ch_axis=self.ch_axis,
            group_size=self.block_size,
            scale_calculation_mode=self.scale_calculation_mode,
        )


@dataclass
@add_start_docstring(DATA_TYPE_SPEC_DOCSTRING.format("bfp16 data type", "BFP16Spec", "is_dynamic=False"))
class BFP16Spec(DataTypeSpec):
    ch_axis: int
    scale_calculation_mode: str | None = "even"

    def to_quantization_spec(self) -> QTensorConfig:
        return QTensorConfig(
            dtype=Dtype.bfp16, ch_axis=self.ch_axis, group_size=8, scale_calculation_mode=self.scale_calculation_mode
        )


QTT = TypeVar("QTT", bound="QTensorConfig")


@dataclass(eq=True)
class QTensorConfig(BaseQTensorConfig):
    """
    A data class that defines the specifications for quantizing tensors within a model.

    :param Dtype dtype: The data type for quantization (e.g., int8, int4).
    :param Optional[bool] is_dynamic: Specifies whether dynamic or static quantization should be used. Default is ``None``, which indicates no specification.
    :param Optional[Type[ObserverBase]] observer_cls: The class of observer to be used for determining quantization parameters like min/max values. Default is ``None``.
    :param Optional[QSchemeType] qscheme: The quantization scheme to use, such as per_tensor, per_channel or per_group. Default is ``None``.
    :param Optional[int] ch_axis: The channel axis for per-channel quantization. Default is ``None``.
    :param Optional[int] group_size: The size of the group for per-group quantization, also the block size for MX datatypes. Default is ``None``.
    :param Optional[bool] symmetric: Indicates if the quantization should be symmetric around zero. If True, quantization is symmetric. If ``None``, it defers to a higher-level or global setting. Default is ``None``.
    :param Optional[RoundType] round_method: The rounding method during quantization, such as half_even. If None, it defers to a higher-level or default method. Default is ``None``.
    :param Optional[ScaleType] scale_type: Defines the scale type to be used for quantization, like power of two or float. If ``None``, it defers to a higher-level setting or uses a default method. Default is ``None``.
    :param Optional[Dtype] mx_element_dtype: Defines the data type to be used for the element type when using mx datatypes, the shared scale effectively uses FP8 E8M0.
    :param Optional[bool] is_scale_quant: Indicates whether this spec is for quantizing scales rather than tensors. Default is ``False``.

    Example:

    .. code-block:: python

        from quark.torch.quantization.config.type import Dtype, ScaleType, RoundType, QSchemeType
        from quark.torch.quantization.config.config import QTensorConfig
        from quark.torch.quantization.observer.observer import PerChannelMinMaxObserver

        quantization_spec = QTensorConfig(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_channel,
            observer_cls=PerChannelMinMaxObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            is_dynamic=False,
            ch_axis=1,
        )
    """

    # Note: `dtype`, `observer_cls`, `is_dynamic`, `qscheme`, `ch_axis`, `group_size`, `symmetric`, `round_method`, `scale_type`, `scale_format`, `scale_calculation_mode`, `qat_spec`, `mx_element_dtype`, `zero_point_type` are inherited from `BaseQTensorConfig`

    qscheme: QSchemeType | None = None

    observer_cls: type[ObserverBase] | None = None

    dtype: Dtype

    scale_type: ScaleType | None = None

    mx_element_dtype: Dtype | None = None

    zero_point_type: ZeroPointType | None = ZeroPointType.int32

    is_scale_quant: bool = False

    # Whether to use the dedicated memory-efficient buffer-reuse observer implementation
    # when observer_cls is PerBlockMXObserver.
    enable_buffer_reuse: bool = False

    # Upper bound (in flattened block elements) for applying buffer-reuse path.
    max_input_numel: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        """
        When the user init a QTensorConfig, we need to check whether the config is valid.
        for example:
            1. observer_cls -> PerTensorPowOf2MinMSEObserver
            2. qscheme      -> QSchemeType.per_channel

        For the above config, the `per_channel` is in conflict with PerTensorPowOf2MinMSEObserver
        Target:
            Once user config a Config like above that contains any conflict, we need to \
                throw an exception and tell the user what the conflict is.
        """
        # NOTE: for developers, every time a new dtype is added, please add the corresponding check for the new dtype.

        if self.dtype in ONLY_DTYPE_CHANGE and self.observer_cls != PlaceholderObserver:
            raise ValueError(
                f"{self.dtype} only supports observer_cls=PlaceholderObserver (as only type casting is used), got observer_cls={self.observer_cls}."
            )

        if self.dtype not in ALL_DATA_TYPES:
            raise ValueError(f"The value dtype={self.dtype} is not among the supported dtypes {ALL_DATA_TYPES}.")

        # NOTE: for developers, every time a new observer is added, please add the corresponding check for the new observer.
        if self.observer_cls is not None and self.observer_cls not in OBSERVER_CLASSES:
            raise ValueError(
                f"The value observer_cls={self.observer_cls} is not among the supported observer_cls: {OBSERVER_CLASSES}."
            )

        if self.dtype in [
            Dtype.int8,
            Dtype.uint8,
            Dtype.int16,
            Dtype.uint16,
            Dtype.int4,
            Dtype.uint4,
            Dtype.int3,
            Dtype.int2,
            Dtype.int32,
            Dtype.fp8_e4m3,
            Dtype.fp8_e5m2,
        ]:
            if self.is_dynamic is None:
                raise ValueError(
                    f"The field `is_dynamic` cannot be None when Dtype is {self.dtype.name} in QTensorConfig."
                )
            if self.observer_cls is None:
                raise ValueError(
                    f"The field `observer_cls` cannot be None when Dtype is {self.dtype.name} in QTensorConfig."
                )
            if self.qscheme is None:
                raise ValueError(
                    f"The field `qscheme` cannot be None when Dtype is {self.dtype.name} in QTensorConfig. Please reconfigure the quantization settings accordingly."
                )
        if self.dtype in [
            Dtype.int8,
            Dtype.uint8,
            Dtype.int16,
            Dtype.uint16,
            Dtype.int4,
            Dtype.uint4,
            Dtype.int3,
            Dtype.int32,
        ]:
            if self.symmetric is None:
                raise ValueError(
                    f"The field `symmetric` cannot be None when Dtype is {self.dtype.name} in QTensorConfig. Please reconfigure the quantization settings accordingly."
                )
            if self.round_method is None:
                raise ValueError(
                    f"The field `round_method` cannot be None when Dtype is {self.dtype.name} in QTensorConfig. Please reconfigure the quantization settings accordingly."
                )
            if self.scale_type is None:
                raise ValueError(
                    f"The field `scale_type` cannot be None when Dtype is {self.dtype.name} in QTensorConfig. Please reconfigure the quantization settings accordingly."
                )

        # CASE 1:  will only init NonScaledFakeQuantize and PlaceholderObserver,
        #   NOTE: quark/torch/quantization/tensor_quantize.py FakeQuantizeBase:get_fake_quantize
        #   in this case will using NonScaledFakeQuantize, and only init PlaceholderObserver()
        #   As a results, quant forward func: quark.torch.kernel.non_scaled_fake_quantize
        #   /torch/kernel/hw_emulation/hw_emulation_interface.py: def non_scaled_fake_quantize
        if self.dtype in USING_NON_SCALED_QUANT:
            # 1.In quark/torch/kernel/__init__.py: class NonScaledFakeQuantizeFunction
            #   During quantization: the following needed
            #    1.input_tensor 2.quant_dtype 2.mx_element_dtype 3.axis 4.block_size 5.scale_calculation_mode needed
            # 2.In init PlaceholderObserver, only qspec.dtype needed
            # Summary for QTensorConfig: 1.dtype(r) 2.mx_element_dtype(o) 3.axis(r) 4.group_size(r) 5.scale_calculation_mode(o)

            required_fields = ["dtype", "ch_axis", "group_size"]
            oprional_fields = ["mx_element_dtype", "scale_calculation_mode"]

            if self.ch_axis is None:
                raise ValueError(
                    f"When using dtype={self.dtype}, quantization_spec.ch_axis must be specified. Got `ch_axis=None`."
                )

            if self.group_size is None:
                raise ValueError(
                    f"When using dtype={self.dtype}, quantization_spec.group_size must be specified. Got `group_size=None`."
                )

            if self.dtype == Dtype.mx and self.mx_element_dtype is None:
                raise ValueError(
                    f"When using dtype={self.dtype}, quantization_spec.mx_element_dtype must be specified. Got `mx_element_dtype=None`."
                )

            for each_field in fields(self):
                if each_field.name not in required_fields + oprional_fields:
                    value = getattr(self, each_field.name)
                    default_value = each_field.default
                    if value != default_value:
                        logger.warning(
                            f"When using dtype={self.dtype}, QTensorConfig.{each_field.name} will not take effect. Got {each_field.name}={value} but the default is {each_field.name}={default_value}."
                        )
            return

        # CASE 2: # NOTE only quantization_spec.dtype is needed
        # in quantization actually call: fake_quantize_with_dtype_convert
        if self.dtype in ONLY_DTYPE_CHANGE:
            required_fields = ["dtype", "observer_cls"]
            for each_field in fields(self):
                if each_field.name not in required_fields:
                    value = getattr(self, each_field.name)
                    default_value = each_field.default
                    if value != default_value:
                        logger.warning(
                            f"In {self.dtype} quant, QTensorConfig.{each_field.name} will not take effect. User supplied: {value} User should skip setting this field"
                        )

            return

        # CASE 3
        # NOTE: quark/torch/quantization/tensor_quantize.py FakeQuantizeBase:get_fake_quantize
        # we will init ScaledFakeQuantize and the corresponding observer
        # def scaled_fake_quantize, in Quark/quark/torch/kernel/hw_emulation/hw_emulation_interface.py
        #    1. fake_quantize_int:      qscheme, axis (if channel/group), group_size, round_mode
        #    2. fake_quantize_fp8_e4m3: qscheme, axis (if channel/group), group_size
        #    3. fake_quantize_fp8_e5m2: qscheme, axis (if channel/group), group_size
        #    4. fake_quantize_fp4_fp6:  qscheme, axis (if channel/group), group_size, quant_dtype(channel/group)
        if self.observer_cls is None:
            raise ValueError(
                "QTensorConfig requires observer_cls to be specified. "
                "Please provide an observer class (e.g., PerTensorMinMaxObserver)."
            )
        if self.qscheme is None:
            raise ValueError(
                "QTensorConfig requires qscheme to be specified. "
                "Please provide a QSchemeType (e.g., QSchemeType.per_tensor)."
            )

        if self.qscheme == QSchemeType.per_tensor:
            if self.observer_cls not in PER_TENSOR_OBSERVERS:
                raise ValueError(
                    f"Observer {self.observer_cls} does not support per-tensor quantization. "
                    f"Supported observers: {PER_TENSOR_OBSERVERS}"
                )

        elif self.qscheme == QSchemeType.per_channel:
            if self.observer_cls not in PER_CHANNEL_OBSERVERS:
                raise ValueError(
                    f"You select channel wise quant, the observer_cls you select is {self.observer_cls} not support channel wise quant."
                )

            if not isinstance(self.ch_axis, int):
                raise ValueError("You select channel wise quant, user must assigned int num to ch_axis.")

        elif self.qscheme == QSchemeType.per_group:
            if self.observer_cls not in PER_GROUP_OBSERVERS:
                raise ValueError(
                    f"The combination qscheme={self.qscheme} and observer_cls={self.observer_cls} is not supported. For per group quantization, please use an observer from {PER_GROUP_OBSERVERS}."
                )
            if not isinstance(self.ch_axis, int):
                raise ValueError(
                    f"Got ch_axis={self.ch_axis} in QTensorConfig initialization with qscheme={self.qscheme}. A correct positive integer value is required for per-group quantization."
                )
            if not isinstance(self.group_size, int):
                raise ValueError(
                    f"Got group_size={self.group_size} in QTensorConfig initialization with qscheme={self.qscheme}. A correct positive integer value is required for per-group quantization."
                )
        elif self.qscheme == QSchemeType.per_block:
            if not (
                isinstance(self.block_size, tuple | list)
                and len(self.block_size) == 2
                and all(isinstance(x, int) for x in self.block_size)
            ):
                raise ValueError(
                    f"Got block_size={self.block_size} in QTensorConfig initialization with qscheme={self.qscheme}. A correct tuple or list of two positive integers is required for per-block quantization."
                )
        else:  # NOTE for developer
            raise ValueError(
                f"Please decide {self.observer_cls.__name__} belongs to which kind of quant (tensor/channel/group)."
            )

    def set_group_size(self, group_size: int) -> None:
        if not isinstance(group_size, int):
            raise TypeError(f"group_size must be an integer, got {type(group_size).__name__}.")
        if not (group_size > 0 or group_size == -1):
            raise ValueError(
                "group_size must be a positive integer or -1 "
                f"(which means group size equals to dimension size), got {group_size}."
            )
        self.group_size = group_size

    def to_dict(self) -> dict[str, Any]:
        # TODO: qat_spec, mx_element_dtype missing.
        return {
            "dtype": self.dtype.name,
            "is_dynamic": self.is_dynamic,
            "qscheme": self.qscheme.name if self.qscheme is not None else None,
            "ch_axis": self.ch_axis,
            "group_size": self.group_size,
            "block_size": self.block_size,
            "symmetric": self.symmetric,
            "round_method": self.round_method.name if self.round_method is not None else None,
            "scale_type": self.scale_type.name if self.scale_type is not None else None,
            "scale_format": self.scale_format,
            "scale_calculation_mode": self.scale_calculation_mode,
            "mx_element_dtype": self.mx_element_dtype.name if self.mx_element_dtype is not None else None,
            "observer_cls": self.observer_cls.__name__ if self.observer_cls is not None else None,
            "is_scale_quant": self.is_scale_quant,
            "enable_buffer_reuse": self.enable_buffer_reuse,
            "max_input_numel": self.max_input_numel,
        }

    @classmethod
    def from_dict(cls: type[QTT], config_dict: dict[str, Any]) -> QTT:
        dtype = Dtype[config_dict["dtype"]]

        if config_dict.get("mx_element_dtype") is not None:
            mx_element_dtype = Dtype[config_dict["mx_element_dtype"]]
        else:
            mx_element_dtype = None

        if config_dict.get("qscheme") is not None:
            qscheme = QSchemeType[config_dict["qscheme"]]
        else:
            qscheme = None

        if config_dict.get("round_method") is not None:
            round_method = RoundType[config_dict["round_method"]]
        else:
            round_method = None

        if config_dict.get("scale_type") is not None:
            scale_type = ScaleType[config_dict["scale_type"]]
        else:
            scale_type = None

        if config_dict.get("scale_format") is not None:
            scale_format = config_dict["scale_format"]
        else:
            scale_format = None

        if config_dict.get("scale_calculation_mode") is not None:
            scale_calculation_mode = config_dict["scale_calculation_mode"]
        else:
            scale_calculation_mode = None

        # TODO: Deprecate legacy configuration.
        # Accomodate the legacy (quark<1.0) export which used custom keys.
        is_dynamic = config_dict["is_dynamic"] if "is_dynamic" in config_dict else config_dict["dynamic"]
        ch_axis = config_dict["ch_axis"] if "ch_axis" in config_dict else config_dict["axis"]

        group_size = config_dict["group_size"]
        block_size = config_dict.get("block_size")
        symmetric = config_dict["symmetric"]

        if "observer_cls" in config_dict:
            if config_dict["observer_cls"] in OBSERVER_MAP:
                observer_cls = OBSERVER_MAP[config_dict["observer_cls"]]
            else:  # pragma: no cover
                logger.warning(
                    f"Unknown observer_cls={config_dict['observer_cls']}. Loading the QTensorConfig with observer_cls=PlaceholderObserver."
                )
                observer_cls = PlaceholderObserver
        else:  # pragma: no cover
            # quark<1.0 used not to save the `observer_cls` in `QTensorConfig.to_dict()`.
            observer_cls = PlaceholderObserver

        is_scale_quant = config_dict.get("is_scale_quant", False)
        enable_buffer_reuse = config_dict.get("enable_buffer_reuse", False)
        max_input_numel = config_dict.get("max_input_numel", 4 * 1024 * 1024)

        return cls(
            dtype=dtype,
            is_dynamic=is_dynamic,
            qscheme=qscheme,
            ch_axis=ch_axis,
            group_size=group_size,
            block_size=block_size,
            symmetric=symmetric,
            round_method=round_method,
            scale_type=scale_type,
            scale_format=scale_format,
            scale_calculation_mode=scale_calculation_mode,
            mx_element_dtype=mx_element_dtype,
            observer_cls=observer_cls,  # type: ignore[arg-type]
            is_scale_quant=is_scale_quant,
            enable_buffer_reuse=enable_buffer_reuse,
            max_input_numel=max_input_numel,
        )

    def is_ocp_mxfp4(self) -> bool:
        quant_spec_is_mxfp4 = True
        for key, value in OCP_MXSpec.OCP_MX_SPEC_KWARGS.items():
            if getattr(self, key) != value:
                quant_spec_is_mxfp4 = False
                break

        if self.dtype != Dtype.fp4:
            quant_spec_is_mxfp4 = False

        return quant_spec_is_mxfp4


@dataclass
class QATSpec(BaseConfigImpl):
    pass


@dataclass
class TQTSpec(QATSpec):
    """
    Configuration for the Trained Quantization Thresholds (TQT) post-training quantization method, implementing https://arxiv.org/abs/1903.08066.
    """

    threshold_init_meth: TQTThresholdInitMeth | None = None


def load_pre_optimization_config_from_file(file_path: str) -> PreQuantOptConfig:
    """
    Load pre-optimization configuration from a JSON file.

    :param file_path: The path to the JSON file containing the pre-optimization configuration.
    :type file_path: str
    :return: The pre-optimization configuration.
    :rtype: PreQuantOptConfig
    """
    with open(file_path) as file:
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
    with open(file_path) as file:
        algo_config_info = json.load(file)
    return _load_quant_algo_config_from_dict(algo_config_info)


def _migrate_deprecated_rotation_fields(config_dict: dict[str, Any]) -> None:
    """Migrate deprecated rotation config fields for backward compatibility.

    Handles two deprecated patterns:
    1. ``random`` field -> ``random_r1`` and ``random_r2``
    2. ``quarot`` name -> ``rotation`` with r1-r4 all True
    """
    if "random" in config_dict:
        logger.warning(
            "Config field 'random' is removed. Use 'random_r1' and 'random_r2' instead. "
            "Auto-converting for backward compatibility."
        )
        random_val = config_dict.pop("random")
        config_dict.setdefault("random_r1", random_val)
        config_dict.setdefault("random_r2", random_val)

    if config_dict.get("name") == "quarot":
        logger.warning(
            "Config name 'quarot' is deprecated. Use 'rotation' with r1=True, r2=True, "
            "r3=True, r4=True instead. Auto-converting for backward compatibility."
        )
        for key in ("r1", "r2", "r3", "r4"):
            config_dict.setdefault(key, True)
        config_dict.pop("optimized_rotation_path", None)
        config_dict["name"] = "rotation"


def _load_pre_optimization_config_from_dict(pre_optimization_config_dict: dict[str, Any]) -> PreQuantOptConfig:
    """
    Load pre-optimization configuration from a dictionary.

    :param pre_optimization_config_dict: A dictionary containing the pre-optimization configuration.
    :type pre_optimization_config_dict: Dict[str, Any]
    :return: The pre-optimization configuration.
    :rtype: PreQuantOptConfig
    :raises ValueError: If the configuration name is not recognized.
    """
    # Deprecate old settings for GQA
    pre_optimization_config_dict.pop("num_attention_heads", None)
    pre_optimization_config_dict.pop("num_key_value_heads", None)

    # Handle deprecated rotation fields (random, quarot)
    _migrate_deprecated_rotation_fields(pre_optimization_config_dict)

    if pre_optimization_config_dict["name"] == "rotation":
        return cast(PreQuantOptConfig, RotationConfig.from_dict(pre_optimization_config_dict))
    elif pre_optimization_config_dict["name"] == "smooth":
        return cast(PreQuantOptConfig, SmoothQuantConfig.from_dict(pre_optimization_config_dict))
    else:
        raise ValueError(f"Unknown algorithm name {pre_optimization_config_dict['name']}")


def _load_quant_algo_config_from_dict(algo_config_dict: dict[str, Any]) -> AlgoConfig:
    """
    Load quantization algorithm configuration from a dictionary.

    :param algo_config_dict: A dictionary containing the quantization algorithm configuration.
    :type algo_config_dict: Dict[str, Any]
    :return: The quantization algorithm configuration.
    :rtype: AlgoConfig
    :raises ValueError: If the configuration name is not recognized.
    """
    # Deprecate old settings for GQA
    algo_config_dict.pop("num_attention_heads", None)
    algo_config_dict.pop("num_key_value_heads", None)

    # Handle deprecated rotation fields (random, quarot)
    _migrate_deprecated_rotation_fields(algo_config_dict)

    if algo_config_dict["name"] == "rotation":
        return cast(AlgoConfig, RotationConfig.from_dict(algo_config_dict))
    elif algo_config_dict["name"] == "smooth":
        return cast(AlgoConfig, SmoothQuantConfig.from_dict(algo_config_dict))
    elif algo_config_dict["name"] == "awq":
        return cast(AlgoConfig, AWQConfig.from_dict(algo_config_dict))
    elif algo_config_dict["name"] == "gptq":  # pragma: no cover
        return cast(AlgoConfig, GPTQConfig.from_dict(algo_config_dict))
    elif algo_config_dict["name"] == "gptaq":  # pragma: no cover
        return cast(AlgoConfig, GPTAQConfig.from_dict(algo_config_dict))
    elif algo_config_dict["name"] == "autosmoothquant":  # pragma: no cover:
        return cast(AlgoConfig, AutoSmoothQuantConfig.from_dict(algo_config_dict))
    elif algo_config_dict["name"] == "qronos":  # pragma: no cover
        return cast(AlgoConfig, QronosConfig.from_dict(algo_config_dict))
    elif algo_config_dict["name"] == "svdquant":
        return cast(AlgoConfig, SVDQuantConfig.from_dict(algo_config_dict))
    else:
        raise ValueError(f"Unknown algorithm name {algo_config_dict['name']}")


@dataclass
class PreQuantOptConfig(BaseAlgoConfig):
    pass


@dataclass
class AlgoConfig(BaseAlgoConfig):
    pass


@dataclass
class SmoothQuantConfig(AlgoConfig):
    """
    A data class that defines the specifications for Smooth Quantization.

    :param str name: The name of the configuration, typically used to identify different quantization settings. Default is ``"smooth"``.
    :param int alpha: The factor of adjustment in the quantization formula, influencing how aggressively weights are quantized. Default is ``1``.
    :param float scale_clamp_min: The minimum scaling factor to be used during quantization, preventing the scale from becoming too small. Default is ``1e-3``.
    :param List[Dict[str, Any]] scaling_layers: Specific settings for scaling layers, allowing customization of quantization parameters for different layers within the model. Default is ``None``.
    :param str model_decoder_layers: Specifies any particular decoder layers in the model that might have unique quantization requirements. Default is ``None``.

    The parameter ``scaling_layers`` can be left to an empty list (default), in which case they will be automatically detected.

    Example:

    .. code-block:: python

        from quark.torch.quantization.config.config import SmoothQuantConfig

        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn"
            },
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp"
            }
        ]

        smoothquant_config = SmoothQuantConfig(
            scaling_layers=scaling_layers,
            model_decoder_layers="model.layers"
        )
    """

    name: str = "smooth"
    alpha: float = 1
    scale_clamp_min: float = 1e-3
    scaling_layers: list[dict[str, Any]] = field(default_factory=list)
    model_decoder_layers: str = ""


@dataclass
class RotationConfig(AlgoConfig):
    """
    A data class that defines the specifications for the rotation algorithms.

    :param str name: The name of the configuration, typically used to identify different rotation settings. Default is ``"rotation"``.
    :param bool r1: Whether to apply ``R1`` rotation. See `SpinQuant paper <https://arxiv.org/abs/2405.16406>`__ for details. Defaults to ``True``.
    :param bool r2: Whether to apply ``R2`` rotation. See `SpinQuant paper <https://arxiv.org/abs/2405.16406>`__ for details. Defaults to ``True``.
    :param bool r3: Whether to apply ``R3`` rotation. It is only useful when using KV cache quantization. See `SpinQuant paper <https://arxiv.org/abs/2405.16406>`__ for details. Defaults to ``True``.
    :param bool r4: Whether to apply ``R4`` rotation. See `SpinQuant paper <https://arxiv.org/abs/2405.16406>`__ for details. Defaults to ``True``.
    :param Optional[int] rotation_size: The size of rotations to apply on activations/weights. By default, the activation last dimension (e.g. ``hidden_size``), or weight input/output channel dimension is used as rotation size. In case the parameter ``rotation_size`` is specified, smaller rotations of size ``(rotation_size, rotation_size)`` are applied per-block. Defaults to ``None``.
    :param bool random_r1: A boolean flag indicating whether ``R1`` should be a random Hadamard matrix. See `SpinQuant paper <https://arxiv.org/abs/2405.16406>`__ for details. This can be useful for data augmentation purposes where random rotations may be required. Default is ``False``.
    :param bool random_r2: A boolean flag indicating whether ``R2`` should be a random Hadamard matrix. See `SpinQuant paper <https://arxiv.org/abs/2405.16406>`__ for details. This can be useful for data augmentation purposes where random rotations may be required. Default is ``False``. ``random_r1`` and ``random_r2`` are only relevant if we are using Hadamard rotations for ``R1`` and ``R2``.
    :param List[Dict[str, str]] scaling_layers: Specific settings for scaling layers, specifying the layer names where ``R1`` rotations must be applied if chosen, or where smoothing scales must be trained, if ``train_smooth=True``. It is a dictionary with keys ``"first_layer"``, ``"middle_layers"`` and ``"last_layer"``, which are dictionaries specifying:

        - ``"prev_modules"``: The list of previous modules the activation rotation may be fused into. **This is optional/unused for online R1**.
        - ``"norm_module"``: The list of normalization layer that is in between the modules in ``"prev_modules"`` and ``"next_modules"``. The normalization weight will typically need to be merged first into the modules of ``"next_modules"`` in order to permute the activation rotation with the normalization, and fuse the activation rotation in the modules of ``"prev_modules"``.  **This is optional/unused for online R1**.
        - ``"next_modules"``: The list of modules to fuse inverse rotation into, on the input features dimension. This corresponds to weight rotation on the input features dimension.  **This is optional/unused for online R1**.
        - ``"target_modules"``: If specified, the list of modules to apply online rotation on, this may be only on a subset of ``"next_modules"``. This is for example useful for MOE models, in case the experts gate/router is not quantized (skipping online rotation for it). **Useful only for online ``R1`` rotations, optional.** If not provided, ``"next_modules"`` is used instead as the list of modules to apply rotation on.

    :param str backbone: A string indicating the path to the model backbone.
    :param str model_decoder_layers: A string indicating the path to the list of decoder layers.
    :param str v_proj: A string indicating the path to the v projection layer, starting from the decoder layer it is in.
    :param str o_proj: A string indicating the path to the o projection layer, starting from the decoder layer it is in.
    :param str self_attn: A string indicating the path to the self attention block, starting from the decoder layer it is in.
    :param str mlp: A string indicating the path to the multilayer perceptron layer, starting from the decoder layer it is in.
    :param Optional[bool] online_r1_rotation: Whether the activation rotation ``R1`` should be kept online instead of being fused into the preceding layer weights. In case ``online_r1_rotation=False``, a single ``R1`` rotation is shared across all layers, whereas specialized rotations can be used per linear layer in case the activation rotation is done online. Defaults to ``False``, i.e. ``R1`` is fused offline. If `R1` is not used, defaults to ``None``.
    :param bool trainable: Whether to insert trainable rotations.
    :param Optional[OnlineRotationConfig] online_config: Configuration specific to online rotations. Refer to :py:class:`.OnlineRotationConfig`. Defaults to ``OnlineRotationConfig(shared_parallel=False)`` for online trainable ``R1``, otherwise defaults to ``None`` (no effect).
    :param Optional[bool] train_smooth: Whether to train SmoothQuant scales. The trainable transform is then ``T = D @ R`` (or ``T = R @ D``) with the rotation ``R`` and the SmoothQuant scales ``D`` (seen as a 1D vector, or a diagonal matrix). This parameter is similar to the approach in `OSTQuant <https://arxiv.org/abs/2501.13987>`__ paper. Defaults to ``None``. In case ``trainable=True``, defaults to ``False``.
    :param Optional[List[str]] smooth_positions: A list of rotation positions to train smoothing scales for. Acceptable list values are ``"r1"``, ``"r2"``, ``"r4"``. Defaults to ``None``. In case ``trainable=True``, defaults to ``[]``.

    Example for llama model, offline rotations:

    .. code-block:: python

        from quark.torch.quantization.config.config import RotationConfig

        scaling_layers = {
            "first_layer": [
                {
                    "prev_modules": ["model.embed_tokens"],
                    "norm_module": "model.layers.layer_id.input_layernorm",
                    "next_modules": [
                        "model.layers.layer_id.self_attn.q_proj",
                        "model.layers.layer_id.self_attn.k_proj",
                        "model.layers.layer_id.self_attn.v_proj",
                    ],
                },
                {
                    "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                    "norm_module": "model.layers.layer_id.post_attention_layernorm",
                    "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"],
                },
            ],
            "middle_layers": [
                {
                    "prev_modules": ["model.layers.pre_layer_id.mlp.down_proj"],
                    "norm_module": "model.layers.layer_id.input_layernorm",
                    "next_modules": [
                        "model.layers.layer_id.self_attn.q_proj",
                        "model.layers.layer_id.self_attn.k_proj",
                        "model.layers.layer_id.self_attn.v_proj",
                    ],
                },
                {
                    "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                    "norm_module": "model.layers.layer_id.post_attention_layernorm",
                    "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"],
                },
            ],
            "last_layer": [
                {
                    "prev_modules": ["model.layers.layer_id.mlp.down_proj"],
                    "norm_module": "model.norm",
                    "next_modules": ["lm_head"],
                }
            ],
        }

        rotation_config = RotationConfig(
            model_decoder_layers="model.layers",
            v_proj="self_attn.v_proj",
            o_proj="self_attn.o_proj",
            self_attn="self_attn",
            mlp="mlp",
            scaling_layers=scaling_layers,
            r1=True,
            r2=True,
        )

    Example for llama model, online rotations:

    .. code-block:: python

        from quark.torch.quantization.config.config import RotationConfig

        scaling_layers = {
            "first_layer": [
                {
                    "target_modules": [
                        "model.layers.layer_id.self_attn.q_proj",
                        "model.layers.layer_id.self_attn.k_proj",
                        "model.layers.layer_id.self_attn.v_proj",
                    ]
                },
                {
                    "target_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"]
                },
            ],
            "middle_layers": [
                {
                    "target_modules": [
                        "model.layers.layer_id.self_attn.q_proj",
                        "model.layers.layer_id.self_attn.k_proj",
                        "model.layers.layer_id.self_attn.v_proj",
                    ]
                },
                {
                    "target_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"]
                },
            ],
            "last_layer": [
                {
                    "target_modules": [],
                }
            ],
        }

        rotation_config = RotationConfig(
            model_decoder_layers="model.layers",
            v_proj="self_attn.v_proj",
            o_proj="self_attn.o_proj",
            self_attn="self_attn",
            mlp="mlp",
            scaling_layers=scaling_layers,
            r1=True,
            r2=True,
            online_r1_rotation=True,
            rotation_size=32,
        )
    """

    scaling_layers: dict[str, list[dict[str, Any]]]
    name: str = "rotation"
    r1: bool = True
    r2: bool = False
    r3: bool = False
    r4: bool = False
    rotation_size: int | None = None
    random_r1: bool = False
    random_r2: bool = False
    backbone: str = "model"
    model_decoder_layers: str = "model.layers"
    v_proj: str = "self_attn.v_proj"
    o_proj: str = "self_attn.o_proj"
    self_attn: str = "self_attn"
    mlp: str = "mlp"
    online_r1_rotation: bool | None = None
    online_config: OnlineRotationConfig | None = None
    trainable: bool = False
    train_smooth: bool | None = None
    smooth_positions: list[str] | None = None

    def __post_init__(self) -> None:
        if (self.random_r1 or self.random_r2) and self.rotation_size is not None:
            raise NotImplementedError(
                f"random_r1=True or random_r2=True along with a custom rotation_size={self.rotation_size} is not supported at the moment in RotationConfig. Please open an issue."
            )

        if self.r1:
            if self.online_r1_rotation is None:
                self.online_r1_rotation = False
        else:
            if self.online_r1_rotation is not None:
                raise ValueError(
                    f"The configuration online_r1_rotation={self.online_r1_rotation} has no effect along r1={self.r1}."
                )

        if self.trainable and self.r3:
            raise NotImplementedError("trainable=True along `r3=True` is not implemented.")

        if self.online_config is not None:
            if (not self.online_r1_rotation and not self.r1) and not self.r4:
                raise ValueError(
                    f"Got online_r1_rotation={self.online_r1_rotation}, r1={self.r1}, r4={self.r4}, for which the provided online_config has no effect: {self.online_config}."
                )
        else:
            if self.r1 and self.online_r1_rotation:
                self.online_config = OnlineRotationConfig(shared_parallel=False)

        if self.trainable and self.train_smooth is None:
            self.train_smooth = False
            self.smooth_positions = []

        if self.train_smooth and not self.trainable:
            raise ValueError(f"train_smooth={self.train_smooth} along trainable={self.trainable} is not supported.")

        if self.train_smooth and self.smooth_positions is None:
            raise ValueError(
                "RotationConfig.smooth_positions needs to be set in case RotationConfig.train_smooth=True, got None."
            )

    @classmethod
    def from_dict(cls, rotation_dict: dict[str, Any]) -> RotationConfig:
        if "online_config" in rotation_dict and isinstance(rotation_dict["online_config"], dict):
            online_config = OnlineRotationConfig(**rotation_dict["online_config"])
            rotation_dict["online_config"] = online_config

        return cls(**rotation_dict)


@dataclass
class OnlineRotationConfig(BaseConfigImpl):
    shared_parallel: bool
    online_rotation_layers: list[str] | None = None


@dataclass
class AutoSmoothQuantConfig(AlgoConfig):
    """
    A data class that defines the specifications for AutoSmoothQuant.

    :param str name: The name of the quantization configuration. Default is ``"autosmoothquant"``.
    :param List[Dict[str, str]] scaling_layers: Configuration details for scaling layers within the model, specifying custom scaling parameters per layer. Default is ``None``.
    :param str compute_scale_loss: Calculate the best scale loss, "MSE" or "MAE". Default is ``"MSE"``.
    :param str model_decoder_layers: Specifies the layers involved in model decoding that may require different quantization parameters. Default is ``None``.

    Example:

    .. code-block:: python

        from quark.torch.quantization.config.config import AutoSmoothQuantConfig

        scaling_layers = [
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn"
            },
            {
                "prev_op": "self_attn.v_proj",
                "layers": ["self_attn.o_proj"],
                "inp": "self_attn.o_proj"
            },
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp"
            },
            {
                "prev_op": "mlp.up_proj",
                "layers": ["mlp.down_proj"],
                "inp": "mlp.down_proj"
            }
        ]

        autosmoothquant_config = AutoSmoothQuantConfig(
            model_decoder_layers="model.layers",
            scaling_layers=scaling_layers
        )
    """

    name: str = "autosmoothquant"
    scaling_layers: list[dict[str, Any]] | None = None
    model_decoder_layers: str | None = None
    compute_scale_loss: str | None = "MSE"


@dataclass
class AWQConfig(AlgoConfig):
    """
    Configuration for Activation-aware Weight Quantization (AWQ).

    :param str name: The name of the quantization configuration. Default is ``"awq"``.
    :param List[Dict[str, Any]] scaling_layers: Configuration details for scaling layers within the model, specifying custom scaling parameters per layer. Default is ``None``.
    :param str model_decoder_layers: Specifies the layers involved in model decoding that may require different quantization parameters. Default is ``None``.

    The parameter ``scaling_layers`` can be left to an empty list (default), in which case they will be automatically detected.

    Example:

    .. code-block:: python

        from quark.torch.quantization.config.config import AWQConfig

        scaling_layers = [
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn"
            },
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp"
            },
        ]

        awq_config = AWQConfig(
            model_decoder_layers="model.layers",
            scaling_layers=scaling_layers
        )
    """

    name: str = "awq"
    scaling_layers: list[dict[str, Any]] = field(default_factory=list)
    model_decoder_layers: str = field(default_factory=str)


@dataclass
class GPTQConfig(AlgoConfig):
    """
    A data class that defines the specifications for Accurate Post-Training Quantization for Generative Pre-trained Transformers (GPTQ).

    :param str name: The configuration name. Default is ``"gptq"``.
    :param int block_size: GPTQ divides the columns into blocks of size block_size and quantizes each block separately. Default is ``128``.
    :param float damp_percent: The percentage used to dampen the quantization effect, aiding in the maintenance of accuracy post-quantization. Default is ``0.01``.
    :param bool desc_act: Indicates whether descending activation is used, typically to enhance model performance with quantization. Default is ``True``.
    :param bool static_groups: Specifies whether the order of groups for quantization are static or can be dynamically adjusted. Default is ``True``. Quark export only support static_groups as True.
    :param List[str] inside_layer_modules: Lists the names of internal layer modules within the model that require specific quantization handling. Default is ``None``.
    :param str model_decoder_layers: Specifies custom settings for quantization on specific decoder layers of the model. Default is ``None``.

    Example:

    .. code-block:: python

        from quark.torch.quantization.config.config import GPTQConfig

        gptq_config = GPTQConfig(
            inside_layer_modules=[
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.q_proj",
                "self_attn.o_proj",
                "mlp.up_proj",
                "mlp.gate_proj",
                "mlp.down_proj"
            ],
            model_decoder_layers="model.layers"
        )

    """

    name: str = "gptq"
    block_size: int = 128
    damp_percent: float = 0.01
    desc_act: bool = True
    static_groups: bool = True
    inside_layer_modules: list[str] = field(default_factory=list)
    model_decoder_layers: str = field(default_factory=str)

    def __post_init__(self) -> None:
        if self.desc_act and not self.static_groups:
            raise ValueError(
                "AMD Quark does not support using GPTQ with `desc_act=True` and `static_groups=False`. Please use `static_groups=True`, or disable `desc_act`."
            )


@dataclass
class GPTAQConfig(AlgoConfig):
    """
    A data class that defines the specifications for Accurate Post-Training Quantization for Generative Pre-trained Transformers (GPTQ).

    :param str name: The configuration name. Default is ``"gptaq"``.
    :param int block_size: GPTAQ divides the columns into blocks of size block_size and quantizes each block separately. Default is ``128``.
    :param float damp_percent: The percentage used to dampen the quantization effect, aiding in the maintenance of accuracy post-quantization. Default is ``0.01``.
    :param bool desc_act: Indicates whether descending activation is used, typically to enhance model performance with quantization. Default is ``True``.
    :param bool static_groups: Specifies whether the order of groups for quantization are static or can be dynamically adjusted. Default is ``True``. Quark export only support static_groups as True.
    :param List[str] inside_layer_modules: Lists the names of internal layer modules within the model that require specific quantization handling. Default is ``None``.
    :param str model_decoder_layers: Specifies custom settings for quantization on specific decoder layers of the model. Default is ``None``.

    Example:

    .. code-block:: python

        from quark.torch.quantization.config.config import GPTAQConfig

        gptq_config = GPTAQConfig(
            inside_layer_modules=[
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.q_proj",
                "self_attn.o_proj",
                "mlp.up_proj",
                "mlp.gate_proj",
                "mlp.down_proj"
            ],
            model_decoder_layers="model.layers"
        )

    """

    name: str = "gptaq"
    block_size: int = 128
    damp_percent: float = 0.01
    alpha: float = 0.25
    desc_act: bool = True
    static_groups: bool = True
    inside_layer_modules: list[str] = field(default_factory=list)
    model_decoder_layers: str = field(default_factory=str)

    def __post_init__(self) -> None:
        if self.desc_act and not self.static_groups:
            raise ValueError(
                "AMD Quark does not support using GPTAQ with `desc_act=True` and `static_groups=False`. Please use `static_groups=True`, or disable `desc_act`."
            )


@dataclass
class QronosConfig(AlgoConfig):
    """
    Configuration for Qronos, an advanced post-training quantization algorithm. Implemented as proposed in https://arxiv.org/pdf/2505.11695

    :param List[str] inside_layer_modules: Lists the names of internal layer modules within the model that require specific quantization handling.
    :param str model_decoder_layers: Specifies custom settings for quantization on specific decoder layers of the model.
    :param str name: The configuration name. Default is ``"qronos"``.
    :param int block_size: Qronos divides the columns into blocks of size block_size and quantizes each block separately. Default is ``128``.
    :param bool desc_act: Indicates whether descending activation is used, typically to enhance model performance with quantization. Default is ``True``.
    :param bool static_groups: Specifies whether the order of groups for quantization are static or can be dynamically adjusted. Default is ``True``. Quark export only supports ``static_groups=True``.
    :param float alpha: Dampening factor for numerical stability during matrix inversions. Default is ``1e-6``.
    :param float beta: Stabilisation factor for Cholesky decomposition. Default is ``1e4``.

    Example:

    .. code-block:: python

        from quark.torch.quantization.config.config import QronosConfig

        qronos_config = QronosConfig(
            inside_layer_modules=[
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.q_proj",
                "self_attn.o_proj",
                "mlp.up_proj",
                "mlp.gate_proj",
                "mlp.down_proj"
            ],
            model_decoder_layers="model.layers"
        )
    """

    inside_layer_modules: list[str]
    model_decoder_layers: str
    name: str = "qronos"
    block_size: int = 128
    desc_act: bool = True
    static_groups: bool = True
    alpha: float = 1e-3
    beta: float = 1e4

    def __post_init__(self) -> None:
        if self.desc_act and not self.static_groups:
            raise ValueError(
                "AMD Quark does not support using Qronos with `desc_act=True` and `static_groups=False`. Please use `static_groups=True`."
            )

        if self.block_size <= 0:
            raise ValueError(f"Number of blocks must be positive, got {self.block_size}.")


@dataclass
class SVDQuantConfig(AlgoConfig):
    """Configuration for SVDQuant (SVD-based low-rank error correction).

    When ``search_alpha`` is True (the default), the smoothing migration
    strength *alpha* is searched independently for each layer to minimise
    post-SVD layer output MSE on calibration data, matching the original
    SVDQuant paper.  Set to False to use a fixed global ``smooth_alpha``.

    When ``use_gptq`` is True, residual weights are quantised via GPTQ
    (Hessian-based column-wise optimisation) instead of RTN.
    """

    name: str = "svdquant"
    svd_rank: int = 32
    smooth_alpha: float = 0.5
    search_alpha: bool = True
    alpha_candidates: list[float] | None = None
    alpha_search_max_samples: int = 8
    exclude_patterns: list[str] = field(
        default_factory=lambda: [
            "time_embedding",
            "add_time_proj",
            "conv_in",
            "conv_out",
            "time_proj",
        ]
    )
    min_layer_size: int = 256
    use_gptq: bool = False
    gptq_n_bits: int = 4
    gptq_symmetric: bool = True
    gptq_group_size: int = -1
    gptq_blocksize: int = 128
    gptq_percdamp: float = 0.01
    gptq_actorder: bool = False
