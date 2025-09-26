#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from typing import Any, Union

import torch.nn as nn

from quark.shares.utils.log import ScreenLogger
from quark.torch.quantization.config.algo_configs import get_algo_config
from quark.torch.quantization.config.config import (
    AutoSmoothQuantConfig,
    AWQConfig,
    BFP16Spec,
    Config,
    FP8E4M3PerTensorSpec,
    GPTQConfig,
    Int4PerGroupSpec,
    Int8PerTensorSpec,
    MX6Spec,
    OCP_MXFP4Spec,
    OCP_MXFP6E2M3Spec,
    OCP_MXFP6E3M2Spec,
    QuantizationConfig,
    RotationConfig,
    SmoothQuantConfig,
    Uint4PerGroupSpec,
)

logger = ScreenLogger(__name__)


class QuantizationScheme:
    """Abstract base class for quantization schemes."""

    def __init__(self, config: QuantizationConfig):
        self._config = config

    @property
    def config(self) -> QuantizationConfig:
        return self._config


class Int4WeightOnlyScheme(QuantizationScheme):
    """Scheme for INT4 weight-only quantization."""

    def __init__(self, group_size: int):
        self.group_size = group_size

    @property
    def config(self) -> QuantizationConfig:
        weight_spec = Int4PerGroupSpec(
            ch_axis=-1, is_dynamic=False, scale_type="float", group_size=self.group_size
        ).to_quantization_spec()
        return QuantizationConfig(weight=weight_spec)


class Uint4WeightOnlyScheme(QuantizationScheme):
    """Scheme for UINT4 weight-only quantization."""

    def __init__(self, group_size: int):
        self.group_size = group_size

    @property
    def config(self) -> QuantizationConfig:
        weight_spec = Uint4PerGroupSpec(
            ch_axis=-1, is_dynamic=False, scale_type="float", group_size=self.group_size
        ).to_quantization_spec()
        return QuantizationConfig(weight=weight_spec)


class Int8Scheme(QuantizationScheme):
    """Scheme for INT8 weight and activation input quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QuantizationConfig:
        spec = Int8PerTensorSpec(
            observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
        ).to_quantization_spec()
        return QuantizationConfig(weight=spec, input_tensors=spec)


class FP8Scheme(QuantizationScheme):
    """Scheme for FP8 quantization (e4m3 format)."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QuantizationConfig:
        spec = FP8E4M3PerTensorSpec(
            observer_method="min_max", scale_type="float", is_dynamic=False
        ).to_quantization_spec()
        return QuantizationConfig(weight=spec, input_tensors=spec)


class MXFP4Scheme(QuantizationScheme):
    """Scheme for MXFP4 quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QuantizationConfig:
        spec = OCP_MXFP4Spec(is_dynamic=False).to_quantization_spec()
        spec_dynamic = OCP_MXFP4Spec(is_dynamic=True).to_quantization_spec()
        return QuantizationConfig(weight=spec, input_tensors=spec_dynamic)


class MXFP6E3M2Scheme(QuantizationScheme):
    """Scheme for MXFP6E3M2 quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QuantizationConfig:
        spec = OCP_MXFP6E3M2Spec(is_dynamic=False).to_quantization_spec()
        spec_dynamic = OCP_MXFP6E3M2Spec(is_dynamic=True).to_quantization_spec()
        return QuantizationConfig(weight=spec, input_tensors=spec_dynamic)


class MXFP6E2M3Scheme(QuantizationScheme):
    """Scheme for MXFP6E2M3 quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QuantizationConfig:
        spec = OCP_MXFP6E2M3Spec(is_dynamic=False).to_quantization_spec()
        spec_dynamic = OCP_MXFP6E2M3Spec(is_dynamic=True).to_quantization_spec()
        return QuantizationConfig(weight=spec, input_tensors=spec_dynamic)


class MX6Scheme(QuantizationScheme):
    """Scheme for MX6 quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QuantizationConfig:
        spec = MX6Spec(ch_axis=-1, block_size=32).to_quantization_spec()
        return QuantizationConfig(weight=spec, input_tensors=spec)


class BFP16Scheme(QuantizationScheme):
    """Scheme for BFP16 quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QuantizationConfig:
        spec = BFP16Spec(ch_axis=-1).to_quantization_spec()
        return QuantizationConfig(weight=spec, input_tensors=spec)


class QuantizationSchemeCollection:
    """Collection for quantization schemes."""

    def __init__(self) -> None:
        self._schemes: dict[str, QuantizationScheme] = {}
        self._collect_supported_schemes()

    def _collect_supported_schemes(self) -> None:
        """Collect all supported quantization schemes."""
        # INT4 weight-only schemes
        self._schemes["int4_wo_32"] = Int4WeightOnlyScheme(group_size=32)
        self._schemes["int4_wo_64"] = Int4WeightOnlyScheme(group_size=64)
        self._schemes["int4_wo_128"] = Int4WeightOnlyScheme(group_size=128)

        # UINT4 weight-only schemes
        self._schemes["uint4_wo_32"] = Uint4WeightOnlyScheme(group_size=32)
        self._schemes["uint4_wo_64"] = Uint4WeightOnlyScheme(group_size=64)
        self._schemes["uint4_wo_128"] = Uint4WeightOnlyScheme(group_size=128)

        # INT8 scheme
        self._schemes["int8"] = Int8Scheme()

        # FP8 quantization schemes
        self._schemes["fp8"] = FP8Scheme()

        # OCP MXFP quantization schemes
        self._schemes["mxfp4"] = MXFP4Scheme()
        self._schemes["mxfp6_e3m2"] = MXFP6E3M2Scheme()
        self._schemes["mxfp6_e2m3"] = MXFP6E2M3Scheme()

        # MX6 quantization schemes
        self._schemes["mx6"] = MX6Scheme()

        # BFP16 quantization schemes
        self._schemes["bfp16"] = BFP16Scheme()

    def register_scheme(self, scheme_name: str, scheme: QuantizationScheme) -> None:
        """Register a quantization scheme."""
        self._schemes[scheme_name] = scheme

    def unregister_scheme(self, scheme_name: str) -> None:
        """Unregister a quantization scheme."""
        del self._schemes[scheme_name]

    def get_supported_schemes(self) -> list[str]:
        """Get list of supported quantization schemes."""
        return list(self._schemes.keys())

    def get_scheme(self, scheme_name: str) -> QuantizationScheme:
        """Get a quantization scheme by name."""
        return self._schemes[scheme_name]


class LLMTemplate:
    """
    A configuration template that defines how to quantize specific types of LLM models.

    Each LLM architecture (like llama, qwen, deepseek, etc.) has its own unique structure and naming patterns
    for layers. This template allows specifying those patterns and quantization settings in a reusable way.

    :param str model_type: Type of the LLM model.
    :param List[str] kv_layers_name: List of k_proj and v_proj layer name patterns to match. Default is ``None``.
    :param Union[str, List[str]] q_layer_name: q_proj layer name pattern to match. Default is ``None``.
    :param List[str] exclude_layers_name: List of layer name patterns to exclude from quantization. Default is ``[]``.
    :param AWQConfig awq_config: Configuration for AWQ algorithm. Default is ``None``.
    :param GPTQConfig gptq_config: Configuration for GPTQ algorithm. Default is ``None``.
    :param SmoothQuantConfig smoothquant_config: Configuration for SmoothQuant algorithm. Default is ``None``.
    :param AutoSmoothQuantConfig autosmoothquant_config: Configuration for AutoSmoothQuant algorithm. Default is ``None``.
    :param RotationConfig rotation_config: Configuration for Rotation algorithm. Default is ``None``.

    Note:
        - The quantization schemes supported by the template are:
            - fp8
            - int4_wo_32
            - int4_wo_64
            - int4_wo_128
            - uint4_wo_32
            - uint4_wo_64
            - uint4_wo_128
            - int8
            - mxfp4
            - mxfp6_e3m2
            - mxfp6_e2m3
            - mx6
            - bfp16
        - The quantization algorithms supported by the template are:
            - awq
            - gptq
            - smoothquant
            - autosmoothquant
            - rotation
        - The KV cache schemes supported by the template are:
            - fp8
        - The attention schemes supported by the template are:
            - fp8

    Creating a Custom Template:

    To create a custom template for a new model type, you can define layer name patterns and algorithm configurations
    specific to your model architecture. Take `moonshotai/Kimi-K2-Instruct <https://huggingface.co/moonshotai/Kimi-K2-Instruct>`__
    as an example:

    .. code-block:: python

        from quark.torch import LLMTemplate

        # Create a new template
        template = LLMTemplate(
            model_type="kimi_k2",
            kv_layers_name=["*kv_b_proj"],
            exclude_layers_name=["lm_head"]
        )

        # Register the template to LLMTemplate class (optional, if you want to use the template in other places)
        LLMTemplate.register_template(template)
    """

    _templates: dict[str, LLMTemplate] = {}
    _SCHEME_COLLECTION = QuantizationSchemeCollection()
    _SUPPORTED_SCHEMES = _SCHEME_COLLECTION.get_supported_schemes()
    _SUPPORTED_ALGORITHMS = ["awq", "gptq", "smoothquant", "autosmoothquant", "rotation"]
    _SUPPORTED_KV_CACHE_SCHEMES = ["fp8"]
    _SUPPORTED_ATTENTION_SCHEMES = ["fp8"]

    def __init__(
        self,
        model_type: str,
        kv_layers_name: list[str] | None = None,
        q_layer_name: str | list[str] | None = None,
        exclude_layers_name: list[str] = [],
        awq_config: AWQConfig | None = None,
        gptq_config: GPTQConfig | None = None,
        smoothquant_config: SmoothQuantConfig | None = None,
        autosmoothquant_config: AutoSmoothQuantConfig | None = None,
        rotation_config: RotationConfig | None = None,
    ):
        self.model_type = model_type
        self.kv_layers_name = kv_layers_name
        self.q_layer_name = q_layer_name
        self.exclude_layers_name = exclude_layers_name

        # Algorithm-specific configuration fields
        self.awq_config = awq_config
        self.gptq_config = gptq_config
        self.smoothquant_config = smoothquant_config
        self.autosmoothquant_config = autosmoothquant_config
        self.rotation_config = rotation_config

    @classmethod
    def list_available(cls: type[LLMTemplate]) -> list[str]:
        """
        List all available model names of registered templates.

        :return: List of template names.
        :rtype: List[str]

        Example:

        .. code-block:: python

            from quark.torch import LLMTemplate

            templates = LLMTemplate.list_available()
            print(templates)  # ['llama', 'opt', 'gpt2', ...]
        """
        return list(cls._templates.keys())

    @classmethod
    def register_template(cls, template: LLMTemplate) -> None:
        """
        Register a template.

        :param LLMTemplate template: The template to register.

        Example:

        .. code-block:: python

            from quark.torch import LLMTemplate

            # Create template
            template = LLMTemplate(
                model_type="llama",
                kv_layers_name=["*k_proj", "*v_proj"],
                q_layer_name="*q_proj"
            )

            # Register template
            LLMTemplate.register_template(template)
        """
        cls._templates[template.model_type] = template

    @classmethod
    def get(cls, model_type: str) -> LLMTemplate:
        """Get a template by model type.

        :param str model_type: Type of the model. It is obtained from the original LLM HuggingFace model's ``model.config.model_type`` attribute. When the model_type field is not defined, the ``model.config.architecture[0]`` is assigned as the model_type..

        Available model types:

            - llama
            - mllama
            - llama4
            - opt
            - qwen2_moe
            - qwen2
            - qwen
            - chatglm
            - phi3
            - phi
            - mistral
            - mixtral
            - gptj
            - grok-1
            - cohere
            - dbrx
            - deepseek_v2
            - deepseek_v3
            - deepseek
            - olmo
            - gemma2
            - gemma3_text
            - gemma3
            - instella
            - gpt_oss

        :return: The template object.
        :rtype: LLMTemplate

        Example:

        .. code-block:: python

            from quark.torch import LLMTemplate

            template = LLMTemplate.get("llama")
            print(template)

        """
        if model_type not in cls._templates:
            available = ", ".join(cls.list_available())
            raise ValueError(
                f"There is no model template defined for the model type '{model_type}'. Available templates: {available}."
            )

        return cls._templates[model_type]

    # Register a new quantization scheme for the template
    @classmethod
    def register_scheme(cls, scheme_name: str, config: QuantizationConfig) -> None:
        """
        Register a new quantization scheme for LLMTemplate class.

        :param str scheme_name: Name of the scheme.
        :param QuantizationConfig config: Configuration for the scheme.

        Example:

        .. code-block:: python

            # Register a new quantization scheme ``int8_wo (int8 weight-only)`` to the template
            from quark.torch import LLMTemplate
            from quark.torch.quantization.config.config import Int8PerTensorSpec, QuantizationConfig

            quant_spec = Int8PerTensorSpec(observer_method="min_max", symmetric=True, scale_type="float",
                                           round_method="half_even", is_dynamic=False).to_quantization_spec()
            global_config = QuantizationConfig(weight=quant_spec)

            LLMTemplate.register_scheme("int8_wo", config=global_config)
        """
        cls._SCHEME_COLLECTION.register_scheme(scheme_name, QuantizationScheme(config))
        cls._SUPPORTED_SCHEMES = cls._SCHEME_COLLECTION.get_supported_schemes()

    @classmethod
    def unregister_scheme(cls, scheme_name: str) -> None:
        """
        Unregister a quantization scheme.

        :param str scheme_name: Name of the scheme to unregister.

        Example:

        .. code-block:: python

            from quark.torch import LLMTemplate

            LLMTemplate.unregister_scheme("int8")
        """
        cls._SCHEME_COLLECTION.unregister_scheme(scheme_name)
        cls._SUPPORTED_SCHEMES = cls._SCHEME_COLLECTION.get_supported_schemes()

    def get_config(
        self,
        scheme: str,
        algorithm: Union[str, list[str]] | None = None,
        kv_cache_scheme: str | None = None,
        min_kv_scale: float = 0.0,
        attention_scheme: str | None = None,
        layer_config: dict[str, str] | None = None,
        layer_type_config: dict[type[nn.Module], str] | None = None,
        exclude_layers: list[str] | None = None,
    ) -> Config:
        """
        Create a quantization configuration based on the provided parameters.

        :param str scheme: Name of the quantization scheme.
        :param Optional[Union[str, List[str]]] algorithm: Name or list of names of quantization algorithms to apply.
        :param Optional[str] kv_cache_scheme: Name of the KV cache quantization scheme.
        :param float min_kv_scale: Minimum value of KV Cache scale.
        :param Optional[str] attention_scheme: Name of the attention quantization scheme.
        :param Optional[Dict[str, str]] layer_config: Dictionary of layer name patterns and quantization scheme names.
        :param Optional[Dict[Type[nn.Module], str]] layer_type_config: Dictionary of layer types and quantization scheme names.
        :param Optional[List[str]] exclude_layers: List of layer names to exclude from quantization.

        Example:

        .. code-block:: python

            from quark.torch import LLMTemplate

            template = LLMTemplate.get("llama")
            config = template.get_config(scheme="fp8", kv_cache_scheme="fp8")
        """
        # Check if the scheme is supported
        if scheme not in LLMTemplate._SUPPORTED_SCHEMES:
            raise ValueError(f"Unsupported quantization scheme: {scheme}")
        # Check if the algorithm is supported
        if algorithm:
            if isinstance(algorithm, str):
                algorithm = [algorithm]
            for algo in algorithm:
                if algo not in self._SUPPORTED_ALGORITHMS:
                    raise ValueError(f"Unsupported algorithm: {algo}")
        # Check if the KV cache scheme is supported
        if kv_cache_scheme and kv_cache_scheme not in self._SUPPORTED_KV_CACHE_SCHEMES:
            raise ValueError(f"Unsupported KV cache scheme: {kv_cache_scheme}")
        # Check if the attention scheme is supported
        if attention_scheme and attention_scheme not in self._SUPPORTED_ATTENTION_SCHEMES:
            raise ValueError(f"Unsupported attention scheme: {attention_scheme}")

        # Set up base global configuration
        global_config = self._create_global_config(scheme)

        # Create config object
        config = Config(
            global_quant_config=global_config,
            min_kv_scale=min_kv_scale,
            exclude=self.exclude_layers_name,
            kv_cache_group=self.kv_layers_name,
        )

        # Apply algorithm if specified
        if algorithm:
            config = self._set_algorithm(config, algorithm)

        # Apply KV cache quantization if specified
        if kv_cache_scheme:
            config = self._set_kv_cache_config(config, kv_cache_scheme)

        # Apply attention quantization if specified
        if attention_scheme:
            config = self._set_attention_config(config, attention_scheme)

        # Apply per-layer configuration overrides
        if layer_config:
            config = self._set_layer_name_config(config, layer_config)
        if layer_type_config:
            config = self._set_layer_type_config(config, layer_type_config)

        # Apply exclude layers configuration
        if exclude_layers is not None:
            config = self._set_exclude_layers_config(config, exclude_layers)

        return config

    def _create_global_config(self, scheme: str) -> QuantizationConfig:
        return LLMTemplate._SCHEME_COLLECTION.get_scheme(scheme).config

    def _set_algorithm(self, config: Config, algorithm: Union[str, list[str]]) -> Config:
        if isinstance(algorithm, str):
            algorithm = [algorithm]
        for algo in algorithm:
            if config.algo_config is None:
                config.algo_config = []
            if algo.lower() == "awq":
                if self.awq_config:
                    config.algo_config.append(self.awq_config)
                else:
                    logger.warning(
                        f"No AWQ config provided for {self.model_type}, "
                        "falling back to default AWQ config. If you need customized AWQ quantization for this model, "
                        "please provide the AWQ config, and pass it to LLMTemplate constructor."
                    )
                    # Fallback to default AWQ config
                    config.algo_config.append(AWQConfig())
            elif algo.lower() == "gptq":
                if self.gptq_config:
                    config.algo_config.append(self.gptq_config)
                else:
                    logger.warning(
                        f"No GPTQ config provided for {self.model_type}, "
                        "falling back to default GPTQ config. If you need customized GPTQ quantization for this model, "
                        "please provide the GPTQ config, and pass it to LLMTemplate constructor."
                    )
                    # Fallback to default GPTQ config
                    config.algo_config.append(GPTQConfig())
            elif algo.lower() == "smoothquant":
                if self.smoothquant_config:
                    config.algo_config.append(self.smoothquant_config)
                else:
                    logger.warning(
                        f"No SmoothQuant config provided for {self.model_type}, "
                        "falling back to default SmoothQuant config. If you need customized SmoothQuant quantization for this model, "
                        "please provide the SmoothQuant config, and pass it to LLMTemplate constructor."
                    )
                    # Fallback to default SmoothQuant config
                    config.algo_config.append(SmoothQuantConfig())
            elif algo.lower() == "autosmoothquant":
                if self.autosmoothquant_config:
                    config.algo_config.append(self.autosmoothquant_config)
                else:
                    logger.warning(
                        f"No AutoSmoothQuant config provided for {self.model_type}, "
                        "falling back to default AutoSmoothQuant config. If you need customized AutoSmoothQuant quantization for this model, "
                        "please provide the AutoSmoothQuant config, and pass it to LLMTemplate constructor."
                    )
                    # Fallback to default AutoSmoothQuant config
                    config.algo_config.append(AutoSmoothQuantConfig())
            elif algo.lower() == "rotation":
                if self.rotation_config:
                    config.algo_config.append(self.rotation_config)
                else:
                    logger.warning(
                        f"No Rotation config provided for {self.model_type}, "
                        "not to use Rotation quantization for this model."
                    )
            else:
                raise ValueError(f"Unsupported algorithm: {algo}")
        return config

    def _set_kv_cache_config(self, config: Config, kv_cache_scheme: str) -> Config:
        # Use pattern matching to identify KV projection layers
        if self.kv_layers_name is None:
            return config

        if kv_cache_scheme == "fp8":
            spec = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()

            for layer_name in self.kv_layers_name:
                layer_config = QuantizationConfig(
                    weight=config.global_quant_config.weight,
                    input_tensors=config.global_quant_config.input_tensors,
                    output_tensors=spec,
                )
                config.layer_quant_config[layer_name] = layer_config
                # Create a separate config for KV cache
                kv_cache_config = QuantizationConfig(
                    weight=config.global_quant_config.weight,
                    input_tensors=config.global_quant_config.input_tensors,
                    output_tensors=spec,
                )
                config.kv_cache_quant_config[layer_name] = kv_cache_config
        else:
            raise ValueError(f"Unsupported KV cache quantization scheme: {kv_cache_scheme}")
        return config

    def _set_attention_config(self, config: Config, attention_scheme: str) -> Config:
        if attention_scheme == "fp8":
            spec = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()
            config.softmax_quant_spec = spec

            if self.q_layer_name is not None:
                if isinstance(self.q_layer_name, str):
                    self.q_layer_name = [self.q_layer_name]
                for q_layer_name in self.q_layer_name:
                    config.layer_quant_config[q_layer_name] = QuantizationConfig(
                        weight=config.global_quant_config.weight,
                        input_tensors=config.global_quant_config.input_tensors,
                        output_tensors=spec,
                    )
        else:
            raise ValueError(f"Unsupported attention quantization scheme: {attention_scheme}")
        return config

    def _set_layer_name_config(self, config: Config, layer_name_config: dict[str, str]) -> Config:
        for layer_name, layer_scheme in layer_name_config.items():
            config.layer_quant_config[layer_name] = LLMTemplate._SCHEME_COLLECTION.get_scheme(layer_scheme).config
        return config

    def _set_layer_type_config(self, config: Config, layer_type_config: dict[type[nn.Module], str]) -> Config:
        for layer_type, layer_scheme in layer_type_config.items():
            config.layer_type_quant_config[layer_type] = LLMTemplate._SCHEME_COLLECTION.get_scheme(layer_scheme).config
        return config

    def _set_exclude_layers_config(self, config: Config, exclude_layers: list[str]) -> Config:
        config.exclude.clear()
        for layer_name in exclude_layers:
            config.exclude.append(layer_name)
        return config


# Default template configurations
DEFAULT_TEMPLATES = {
    "llama": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
        "awq_config": "llama",
        "gptq_config": "llama",
        "smoothquant_config": "llama",
        "autosmoothquant_config": "llama",
        "rotation_config": "llama",
    },
    "mllama": {
        "kv_layers_name": ["*language_model.*k_proj", "*language_model.*v_proj"],
        "q_layer_name": "*self_attn.q_proj",
        "exclude_layers_name": ["*lm_head", "*patch_embedding", "multi_modal_projector"],
        "awq_config": "mllama",
        "gptq_config": "mllama",
        "smoothquant_config": "mllama",
        "autosmoothquant_config": "mllama",
        "rotation_config": "mllama",
    },
    "llama4": {
        "kv_layers_name": ["*language_model.*.k_proj", "*language_model.*.v_proj"],
        "q_layer_name": "*language_model.*.q_proj",
        "exclude_layers_name": [
            "multi_modal_projector*",
            "*feed_forward.router",
            "vision_model*",
            "*lm_head",
        ],
        "awq_config": "llama4",
        "gptq_config": "llama4",
        "smoothquant_config": "llama4",
        "autosmoothquant_config": "llama4",
        "rotation_config": "llama4",
    },
    "opt": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
        "awq_config": "opt",
        "gptq_config": "opt",
        "smoothquant_config": "opt",
        "autosmoothquant_config": "opt",
        "rotation_config": "opt",
    },
    "qwen2_moe": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head", "*.gate", "*.shared_expert_gate"],
        "awq_config": "qwen2_moe",
        "gptq_config": "qwen2_moe",
        "smoothquant_config": "qwen2_moe",
        "autosmoothquant_config": "qwen2_moe",
        "rotation_config": "qwen2_moe",
    },
    "qwen2": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
        "awq_config": "qwen2",
        "gptq_config": "qwen2",
        "smoothquant_config": "qwen2",
        "autosmoothquant_config": "qwen2",
        "rotation_config": "qwen2",
    },
    "qwen": {
        "kv_layers_name": ["*c_attn"],
        "q_layer_name": "*c_attn",
        "exclude_layers_name": ["lm_head"],
        "awq_config": "qwen",
        "gptq_config": "qwen",
        "smoothquant_config": "qwen",
        "autosmoothquant_config": "qwen",
        "rotation_config": "qwen",
    },
    "chatglm": {
        "kv_layers_name": ["*query_key_value"],
        "q_layer_name": "*query_key_value",
        "exclude_layers_name": ["transformer.output_layer"],
        "awq_config": "chatglm",
        "gptq_config": "chatglm",
        "smoothquant_config": "chatglm",
        "autosmoothquant_config": "chatglm",
        "rotation_config": "chatglm",
    },
    "phi3": {
        "kv_layers_name": ["*qkv_proj"],
        "q_layer_name": "*qkv_proj",
        "exclude_layers_name": ["lm_head"],
        "awq_config": "phi3",
        "gptq_config": "phi3",
        "smoothquant_config": "phi3",
        "autosmoothquant_config": "phi3",
        "rotation_config": "phi3",
    },
    "phi": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
        "awq_config": "phi",
        "gptq_config": "phi",
        "smoothquant_config": "phi",
        "autosmoothquant_config": "phi",
        "rotation_config": "phi",
    },
    "mistral": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
        "awq_config": "mistral",
        "gptq_config": "mistral",
        "smoothquant_config": "mistral",
        "autosmoothquant_config": "mistral",
        "rotation_config": "mistral",
    },
    "mixtral": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head", "*.gate"],
        "awq_config": "mixtral",
        "gptq_config": "mixtral",
        "smoothquant_config": "mixtral",
        "autosmoothquant_config": "mixtral",
        "rotation_config": "mixtral",
    },
    "gptj": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
        "awq_config": "gptj",
        "gptq_config": "gptj",
        "smoothquant_config": "gptj",
        "autosmoothquant_config": "gptj",
        "rotation_config": "gptj",
    },
    "grok-1": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head", "*.gate"],
        "awq_config": "grok-1",
        "gptq_config": "grok-1",
        "smoothquant_config": "grok-1",
        "autosmoothquant_config": "grok-1",
        "rotation_config": "grok-1",
    },
    "cohere": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
        "awq_config": "cohere",
        "gptq_config": "cohere",
        "smoothquant_config": "cohere",
        "autosmoothquant_config": "cohere",
        "rotation_config": "cohere",
    },
    "dbrx": {
        "kv_layers_name": ["*Wqkv"],
        "q_layer_name": "*Wqkv",
        "exclude_layers_name": ["lm_head", "*router.layer"],
        "awq_config": "dbrx",
        "gptq_config": "dbrx",
        "smoothquant_config": "dbrx",
        "autosmoothquant_config": "dbrx",
        "rotation_config": "dbrx",
    },
    "deepseek_v2": {
        "kv_layers_name": ["*kv_b_proj"],
        "q_layer_name": ["*q_a_proj", "*q_b_proj"],
        "exclude_layers_name": ["lm_head", "*self_attn*", "*mlp.gate"],
        "awq_config": "deepseek_v2",
        "gptq_config": "deepseek_v2",
        "smoothquant_config": "deepseek_v2",
        "autosmoothquant_config": "deepseek_v2",
        "rotation_config": "deepseek_v2",
    },
    "deepseek_v3": {
        "kv_layers_name": ["*kv_b_proj"],
        "q_layer_name": ["*q_a_proj", "*q_b_proj"],
        "exclude_layers_name": ["lm_head", "*self_attn*", "*mlp.gate"],
        "awq_config": "deepseek_v3",
        "gptq_config": "deepseek_v3",
        "smoothquant_config": "deepseek_v3",
        "autosmoothquant_config": "deepseek_v3",
        "rotation_config": "deepseek_v3",
    },
    "deepseek": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head", "*.gate"],
        "awq_config": "deepseek",
        "gptq_config": "deepseek",
        "smoothquant_config": "deepseek",
        "autosmoothquant_config": "deepseek",
        "rotation_config": "deepseek",
    },
    "olmo": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
        "awq_config": "olmo",
        "gptq_config": "olmo",
        "smoothquant_config": "olmo",
        "autosmoothquant_config": "olmo",
        "rotation_config": "olmo",
    },
    "gemma2": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
        "awq_config": "gemma2",
        "gptq_config": "gemma2",
        "smoothquant_config": "gemma2",
        "autosmoothquant_config": "gemma2",
        "rotation_config": "gemma2",
    },
    "gemma3_text": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["*lm_head"],
        "awq_config": "gemma3_text",
        "gptq_config": "gemma3_text",
        "smoothquant_config": "gemma3_text",
        "autosmoothquant_config": "gemma3_text",
        "rotation_config": "gemma3_text",
    },
    "gemma3": {
        "kv_layers_name": ["*language_model.*k_proj", "*language_model.*v_proj"],
        "q_layer_name": "*language_model.*q_proj",
        "exclude_layers_name": ["*vision_tower*", "*multi_modal_projector*", "*lm_head"],
        "awq_config": "gemma3",
        "gptq_config": "gemma3",
        "smoothquant_config": "gemma3",
        "autosmoothquant_config": "gemma3",
        "rotation_config": "gemma3",
    },
    "instella": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
        "awq_config": "instella",
        "gptq_config": "instella",
        "smoothquant_config": "instella",
        "autosmoothquant_config": "instella",
        "rotation_config": "instella",
    },
    "gpt_oss": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
        "awq_config": "gpt_oss",
        "gptq_config": "gpt_oss",
        "smoothquant_config": "gpt_oss",
        "autosmoothquant_config": "gpt_oss",
        "rotation_config": "gpt_oss",
    },
}


def _create_template_from_config(model_type: str, config: dict[str, Any]) -> LLMTemplate:
    """create a template from configuration dictionary."""
    return LLMTemplate(
        model_type=model_type,
        kv_layers_name=config["kv_layers_name"],
        q_layer_name=config["q_layer_name"],
        exclude_layers_name=config["exclude_layers_name"],
        awq_config=get_algo_config("awq", config["awq_config"]),  # type: ignore
        gptq_config=get_algo_config("gptq", config["gptq_config"]),  # type: ignore
        smoothquant_config=get_algo_config("smoothquant", config["smoothquant_config"]),  # type: ignore
        autosmoothquant_config=get_algo_config("autosmoothquant", config["autosmoothquant_config"]),
        rotation_config=get_algo_config("rotation", config["rotation_config"]),
    )  # type: ignore


"""
Developer Note for Quark Engineers:
====================================

To add a new model template, follow these steps:

    1. Add the model configuration to DEFAULT_TEMPLATES dictionary above.
    2. Add corresponding algorithm configs to the algo config registry if the algorithm is needed.
    (see quark/torch/quantization/config/algo_config.py)
    3. Update the docstring list in LLMTemplate.get() method to include the new model type.
    4. Test the new template with various quantization schemes and algorithms

Example for adding "new_model":

.. code-block:: python

    "new_model": {
        "kv_layers_name": ["*attention.k_proj", "*attention.v_proj"],
        "q_layer_name": "*attention.q_proj",
        "exclude_layers_name": ["lm_head"],
        "awq_config": "new_model",
        "gptq_config": "new_model",
        "smoothquant_config": "new_model",
        "autosmoothquant_config": "new_model",
        "rotation_config": "new_model",
    }
"""

# Register built-in templates
for model_type, config in DEFAULT_TEMPLATES.items():
    if model_type not in LLMTemplate._templates:
        template = _create_template_from_config(model_type, config)
        LLMTemplate.register_template(template)
