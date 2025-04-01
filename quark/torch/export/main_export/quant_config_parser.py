#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations
from typing import Dict, Any, List, Optional, Type, Tuple
from dataclasses import dataclass, field
from quark.torch.export.config.config import JsonExporterConfig
from quark.torch.quantization.utils import deep_compare
import torch.nn as nn
import fnmatch
from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig
from quark.torch.quantization.config.config import AWQConfig as TrueAWQConfig
from quark.torch.export.constants import AWQ_QUANT_DTYPES
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver, PerGroupMinMaxObserver
from quark.torch.quantization.config.type import Dtype, QSchemeType, QuantizationMode, RoundType, ScaleType


@dataclass
class FP8Config:
    activation_scheme: Optional[str] = None
    ignored_layers: List[str] = field(default_factory=list)
    kv_cache_scheme: Optional[str] = None
    quant_method: str = "fp8"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activation_scheme": self.activation_scheme,
            "ignored_layers": self.ignored_layers,
            "kv_cache_scheme": self.kv_cache_scheme,
            "quant_method": self.quant_method
        }


@dataclass
class AwqConfig:
    quant_method: str = field(default="awq")
    zero_point: bool = field(default=True)
    group_size: Optional[int] = None
    bits: int = field(default=4)
    version: str = field(default="gemm")
    modules_to_not_convert: Optional[List[str]] = None
    pack_method: str = field(default="reorder")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bits": self.bits,
            "group_size": self.group_size,
            "modules_to_not_convert": self.modules_to_not_convert,
            "quant_method": self.quant_method,
            "version": self.version,
            "zero_point": self.zero_point,
            "pack_method": self.pack_method,
        }


def get_layer_quant_config(quant_config: Config, layer_type: Type[nn.Module],
                           layer_name: str) -> Optional[QuantizationConfig]:
    if layer_type not in [nn.Linear, nn.Conv2d]:
        return None

    for exclude_layer in quant_config.exclude:
        if fnmatch.fnmatch(layer_name, exclude_layer):
            return None

    for name_pattern in quant_config.layer_quant_config.keys():
        if fnmatch.fnmatch(layer_name, name_pattern):
            layer_quantization_config = quant_config.layer_quant_config[name_pattern]
            break
    else:
        if layer_type in quant_config.layer_type_quant_config:  # pragma: no cover
            layer_quantization_config = quant_config.layer_type_quant_config[layer_type]
        else:
            layer_quantization_config = quant_config.global_quant_config

    return layer_quantization_config


class QuantConfigParser:

    def __init__(self, quant_config: Config, json_config: JsonExporterConfig) -> None:
        self._pack_method = json_config.pack_method
        self._config = quant_config
        self._kv_cache_group = json_config.kv_cache_group
        self._fp8_kv_cache_scheme = None
        self._custom_mode: Optional[str] = None
        self._output_quant: bool = False

    @property
    def custom_mode(self) -> Optional[str]:
        return self._custom_mode

    @property
    def output_quant(self) -> bool:
        return self._output_quant

    @property
    def kv_cache_group(self) -> Optional[List[str]]:
        return self._kv_cache_group

    def fp8_kv_cache_check(self, layer_quant_config: Dict[str, Any]) -> None:
        if self.kv_cache_group is None or len(self.kv_cache_group) == 0:
            self._fp8_kv_cache_scheme = None
            return

        layer_names = list(layer_quant_config.keys())
        if set(self.kv_cache_group) != set(layer_names):
            self._fp8_kv_cache_scheme = None
            return

        quant_configs = [layer_quant_config[name] for name in layer_names]
        if not all(deep_compare(quant_config, quant_configs[0]) for quant_config in quant_configs):
            self._fp8_kv_cache_scheme = None
            return

        weight_config = quant_configs[0].weight
        if weight_config is None or weight_config.dtype != Dtype.fp8_e4m3 or weight_config.qscheme != QSchemeType.per_tensor:
            self._fp8_kv_cache_scheme = None
            return

        input_config = quant_configs[0].input_tensors
        if input_config is not None and (input_config.dtype != Dtype.fp8_e4m3
                                         or input_config.qscheme != QSchemeType.per_tensor):
            self._fp8_kv_cache_scheme = None
            return

        output_config = quant_configs[0].output_tensors
        if output_config is None or (output_config.dtype != Dtype.fp8_e4m3
                                     or output_config.qscheme != QSchemeType.per_tensor):
            self._fp8_kv_cache_scheme = None
            return
        self._fp8_kv_cache_scheme = "dynamic" if output_config.is_dynamic is True else "static"  # type: ignore

    def get_custom_config(self) -> Tuple[Dict[str, Any], str]:
        """
        Returns the custom configuration that is required by external libraries for specific quantization schemes.
        """
        custom_mode = "quark"

        if self._config.global_quant_config is None:
            return {}, custom_mode  # pragma: no cover

        weight_config = self._config.global_quant_config.weight
        bias_config = self._config.global_quant_config.bias
        input_config = self._config.global_quant_config.input_tensors
        output_config = self._config.global_quant_config.output_tensors

        if output_config is not None:
            self._output_quant = True

        custom_config = {}
        is_custom = True

        if self._config.layer_type_quant_config is not None and len(self._config.layer_type_quant_config) > 0:
            return {}, custom_mode  # pragma: no cover

        if self._config.layer_quant_config is not None and len(self._config.layer_quant_config) > 0:
            self.fp8_kv_cache_check(self._config.layer_quant_config)
            if self._fp8_kv_cache_scheme is None:
                return {}, custom_mode  # pragma: no cover

        if weight_config is None:  # pragma: no cover
            return {}, custom_mode

        if weight_config.dtype == Dtype.fp8_e4m3 and weight_config.qscheme == QSchemeType.per_tensor:
            # fp8 custom config, for vLLM compatibility.
            if input_config is None:
                activation_scheme = None
            elif (input_config.dtype == Dtype.fp8_e4m3 and input_config.qscheme == QSchemeType.per_tensor):
                if output_config is None or (output_config.dtype == Dtype.fp8_e4m3
                                             and output_config.qscheme == QSchemeType.per_tensor):
                    activation_scheme = "dynamic" if input_config.is_dynamic else "static"
                else:
                    is_custom = False  # pragma: no cover
            else:
                is_custom = False  # pragma: no cover

            if is_custom and (bias_config is None or
                              (bias_config.dtype == Dtype.fp8_e4m3 and bias_config.qscheme == QSchemeType.per_tensor)):
                ignored_layers = self._config.exclude
                custom_mode = "fp8"
                custom_config = FP8Config(activation_scheme=activation_scheme,
                                          kv_cache_scheme=self._fp8_kv_cache_scheme,
                                          ignored_layers=ignored_layers).to_dict()
        elif self._config.algo_config is not None and self._config.algo_config.name.lower() == "awq":
            # AWQ custom config, for vLLM, AutoAWQ (and others) compatibility.
            if weight_config.dtype in AWQ_QUANT_DTYPES and weight_config.qscheme == QSchemeType.per_group:
                if input_config is None and output_config is None and bias_config is None:
                    bits = weight_config.dtype.to_bitwidth()

                    custom_mode = "awq"
                    custom_config = AwqConfig(quant_method="awq",
                                              zero_point=not weight_config.symmetric,
                                              group_size=weight_config.group_size,
                                              bits=bits,
                                              modules_to_not_convert=self._config.exclude,
                                              pack_method=self._pack_method).to_dict()

        return custom_config, custom_mode

    @staticmethod
    def from_custom_config(custom_config_dict: Dict[str, Any], is_bias_quantized: bool, is_kv_cache: bool,
                           kv_layers_name: Optional[List[str]]) -> Config:
        """
        Maps the custom quantization config Fp8Config and AwqConfig back to Quark's Config. Some important keys
        can not be inferred from this custom config, namely whether the outputs of quantized ops are quantized, and whether the bias are quantized.
        By default, we assume the outputs of quantized ops are not quantized.
        We require the user to provide the argument `is_bias_quantized` to determine whether the bias is quantized.
        """
        if custom_config_dict["quant_method"] == "fp8":
            if custom_config_dict["activation_scheme"] is None:
                is_dynamic = False  # pragma: no cover
            else:
                is_dynamic = custom_config_dict["activation_scheme"] == "dynamic"

            q_spec = QuantizationSpec(
                dtype=Dtype.fp8_e4m3,
                observer_cls=PerTensorMinMaxObserver,
                is_dynamic=False,
                qscheme=QSchemeType.per_tensor,
            )

            output_tensors = None  # This one is assumed, we can determine if kv_cache quantization was performed by examining the weights file
            if custom_config_dict["activation_scheme"] is None:
                input_tensors = None  # pragma: no cover
            else:
                input_tensors = QuantizationSpec(
                    dtype=Dtype.fp8_e4m3,
                    observer_cls=PerTensorMinMaxObserver,
                    is_dynamic=is_dynamic,
                    qscheme=QSchemeType.per_tensor,
                )

            global_quant_config = QuantizationConfig(
                input_tensors=input_tensors,
                output_tensors=output_tensors,
                weight=q_spec,
                bias=q_spec if is_bias_quantized else None,
            )

            KV_CACHE_CFG: Dict[str, QuantizationConfig] = {}
            if is_kv_cache:
                if kv_layers_name is not None:
                    # We can check the pth or safetensor files to determine if kv_cache is being used
                    FP8_PER_TENSOR_SPEC = QuantizationSpec(
                        dtype=Dtype.fp8_e4m3,
                        observer_cls=PerTensorMinMaxObserver,
                        is_dynamic=is_dynamic,
                        qscheme=QSchemeType.per_tensor,
                    )
                    for layer_name in kv_layers_name:
                        KV_CACHE_CFG[layer_name] = QuantizationConfig(
                            input_tensors=input_tensors,
                            output_tensors=FP8_PER_TENSOR_SPEC,
                            weight=q_spec,
                            bias=q_spec if is_bias_quantized else None,
                        )
                else:
                    raise ValueError(
                        "Initializing import_config requires kv_cache_layers_info but kv_layers_name is empty")

            config = Config(
                global_quant_config=global_quant_config,
                layer_quant_config=KV_CACHE_CFG,
                exclude=custom_config_dict["ignored_layers"],
                quant_mode=QuantizationMode.eager_mode,
            )

        elif custom_config_dict["quant_method"] == "awq":
            if is_bias_quantized:
                raise ValueError("AWQ does not support bias quantization.")

            symmetric = not custom_config_dict["zero_point"]

            # AWQ uses signed dtypes for symmetric case, unsigned for non-symmetric.
            if custom_config_dict["bits"] == 4 and symmetric:
                dtype = Dtype.int4
            elif custom_config_dict["bits"] == 4 and not symmetric:  # pragma: no cover
                dtype = Dtype.uint4
            elif custom_config_dict["bits"] == 8 and symmetric:  # pragma: no cover
                dtype = Dtype.int8
            elif custom_config_dict["bits"] == 8 and not symmetric:  # pragma: no cover
                dtype = Dtype.uint8
            else:
                raise ValueError(
                    f"AWQ in Quark is supported only for 4-bits and 8-bits quantization, but a configuration with {custom_config_dict['bits']} bits was passed."
                )

            q_spec = QuantizationSpec(
                dtype=dtype,
                observer_cls=PerGroupMinMaxObserver,
                is_dynamic=False,
                qscheme=QSchemeType.per_group,
                group_size=custom_config_dict["group_size"],
                symmetric=not custom_config_dict["zero_point"],
                ch_axis=1,  # The only meaningful value for per group quantization.
                round_method=RoundType.half_even,
                scale_type=ScaleType.float,
            )

            global_quant_config = QuantizationConfig(weight=q_spec)

            config = Config(
                global_quant_config=global_quant_config,
                exclude=custom_config_dict["modules_to_not_convert"],
                quant_mode=QuantizationMode.eager_mode,
                algo_config=TrueAWQConfig(),
            )
        else:
            config = Config.from_dict(custom_config_dict)

        return config
