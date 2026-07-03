#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from typing import Any, cast

import torch.nn as nn

from quark.common.utils.log import ScreenLogger
from quark.torch.export.main_export.quant_config_parser import get_layer_quant_config
from quark.torch.quantization.config.algo_configs import get_algo_config, get_supported_algorithm_types
from quark.torch.quantization.config.config import (
    AlgoConfig,
    AmdFP4Spec,
    BFP16Spec,
    FP4PerGroupSpec,
    FP8E4M3PerChannelSpec,
    FP8E4M3PerTensorSpec,
    FP8E5M3PerTensorSpec,
    Int4PerChannelSpec,
    Int4PerGroupSpec,
    Int8PerTensorSpec,
    MX6Spec,
    OCP_MXFP4Spec,
    OCP_MXFP6E2M3Spec,
    OCP_MXFP6E3M2Spec,
    ProgressiveSpec,
    QConfig,
    QLayerConfig,
    ScaleQuantSpec,
    Uint4PerChannelSpec,
    Uint4PerGroupSpec,
)
from quark.torch.quantization.weight_convert import SplitFusedExperts, WeightConverter

logger = ScreenLogger(__name__)


class QuantizationScheme:
    """Abstract base class for quantization schemes."""

    def __init__(self, config: QLayerConfig):
        self._config = config

    @property
    def config(self) -> QLayerConfig:
        return self._config


class Int4WeightOnlyScheme(QuantizationScheme):
    """Scheme for INT4 weight-only quantization."""

    def __init__(self, group_size: int):
        self.group_size = group_size

    @property
    def config(self) -> QLayerConfig:
        weight_spec = Int4PerGroupSpec(
            ch_axis=-1, is_dynamic=False, scale_type="float", group_size=self.group_size
        ).to_quantization_spec()
        return QLayerConfig(weight=weight_spec)


class Int4WeightAndActivationScheme(QuantizationScheme):
    """Scheme for INT4 weight and activation quantization."""

    def __init__(self, group_size: int):
        self.group_size = group_size

    @property
    def config(self) -> QLayerConfig:
        weight_spec = Int4PerGroupSpec(
            ch_axis=-1, is_dynamic=False, scale_type="float", group_size=self.group_size
        ).to_quantization_spec()
        act_spec = Int4PerGroupSpec(ch_axis=-1, is_dynamic=True, group_size=self.group_size).to_quantization_spec()
        return QLayerConfig(weight=weight_spec, input_tensors=act_spec)


class Int4WeightOnlyPerChannelScheme(QuantizationScheme):
    """Scheme for INT4 weight-only per-channel quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        weight_spec = Int4PerChannelSpec(is_dynamic=False, ch_axis=0).to_quantization_spec()
        return QLayerConfig(weight=weight_spec)


class Uint4WeightOnlyScheme(QuantizationScheme):
    """Scheme for UINT4 weight-only quantization."""

    def __init__(self, group_size: int):
        self.group_size = group_size

    @property
    def config(self) -> QLayerConfig:
        weight_spec = Uint4PerGroupSpec(
            ch_axis=-1, is_dynamic=False, scale_type="float", group_size=self.group_size
        ).to_quantization_spec()
        return QLayerConfig(weight=weight_spec)


class Uint4WeightOnlyPerChannelScheme(QuantizationScheme):
    """Scheme for UINT4 weight-only per-channel quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        weight_spec = Uint4PerChannelSpec(is_dynamic=False, ch_axis=0).to_quantization_spec()
        return QLayerConfig(weight=weight_spec)


class Int8Scheme(QuantizationScheme):
    """Scheme for INT8 weight and activation input quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        spec = Int8PerTensorSpec(
            observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
        ).to_quantization_spec()
        return QLayerConfig(weight=spec, input_tensors=spec)


class FP8Scheme(QuantizationScheme):
    """Scheme for FP8 quantization (e4m3 format)."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        spec = FP8E4M3PerTensorSpec(
            observer_method="min_max", scale_type="float", is_dynamic=False
        ).to_quantization_spec()
        return QLayerConfig(weight=spec, input_tensors=spec)


class MXFP4Scheme(QuantizationScheme):
    """Scheme for MXFP4 quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
        spec_dynamic = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=True).to_quantization_spec()
        return QLayerConfig(weight=spec, input_tensors=spec_dynamic)


class MXFP4WeightOnlyScheme(QuantizationScheme):
    """Scheme for weight-only MXFP4 quantization (e.g. gpt-oss source format)."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
        return QLayerConfig(weight=spec)


class MXFP6E3M2Scheme(QuantizationScheme):
    """Scheme for MXFP6E3M2 quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        spec = OCP_MXFP6E3M2Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
        spec_dynamic = OCP_MXFP6E3M2Spec(ch_axis=-1, is_dynamic=True).to_quantization_spec()
        return QLayerConfig(weight=spec, input_tensors=spec_dynamic)


class MXFP6E2M3Scheme(QuantizationScheme):
    """Scheme for MXFP6E2M3 quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        spec = OCP_MXFP6E2M3Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
        spec_dynamic = OCP_MXFP6E2M3Spec(ch_axis=-1, is_dynamic=True).to_quantization_spec()
        return QLayerConfig(weight=spec, input_tensors=spec_dynamic)


class MXFP4_MXFP6E2M3Scheme(QuantizationScheme):
    """Scheme for MXFP4 weight and MXFP6E2M3 activation input quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        weight_spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
        input_spec = OCP_MXFP6E2M3Spec(ch_axis=-1, is_dynamic=True).to_quantization_spec()
        return QLayerConfig(weight=weight_spec, input_tensors=input_spec)


class AmdFP4Scheme(QuantizationScheme):
    """
    Scheme for amdfp4 quantization with E5M3 scale format.

    Supports only ``group_size=16`` or ``group_size=32``.
    """

    def __init__(self, group_size: int = 16) -> None:
        self.group_size = group_size

    @property
    def config(self) -> QLayerConfig:
        weight_spec = AmdFP4Spec(ch_axis=-1, group_size=self.group_size, is_dynamic=False).to_quantization_spec()
        input_spec = AmdFP4Spec(ch_axis=-1, group_size=self.group_size, is_dynamic=True).to_quantization_spec()
        return QLayerConfig(weight=weight_spec, input_tensors=input_spec)


class MX6Scheme(QuantizationScheme):
    """Scheme for MX6 quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        spec = MX6Spec(ch_axis=-1, block_size=32).to_quantization_spec()
        return QLayerConfig(weight=spec, input_tensors=spec)


class BFP16Scheme(QuantizationScheme):
    """Scheme for BFP16 quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        spec = BFP16Spec(ch_axis=-1).to_quantization_spec()
        return QLayerConfig(weight=spec, input_tensors=spec)


class MXFP4_FP8Scheme(QuantizationScheme):
    """Scheme for MXFP4 weight and FP8 activation input quantization."""

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        weight_spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False, scale_calculation_mode="even").to_quantization_spec()
        input_spec = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()
        return QLayerConfig(weight=weight_spec, input_tensors=input_spec)


class PTPCFP8Scheme(QuantizationScheme):
    """Scheme for PTPC FP8 quantization (Dynamic activation per-token quantization, weight quantization per-channel).

    Uses FP8 Per-Channel Static for weights and FP8 Per-Token Dynamic for activations.
    """

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        weight_spec = FP8E4M3PerChannelSpec(is_dynamic=False, ch_axis=0).to_quantization_spec()
        input_spec = FP8E4M3PerChannelSpec(is_dynamic=True, ch_axis=1).to_quantization_spec()
        return QLayerConfig(weight=weight_spec, input_tensors=input_spec)


class FP4Block16ScaleE4M3Scheme(QuantizationScheme):
    """Scheme for FP4 per-group quantization with FP8 E4M3 scale quantization for both weights and activations.

    Uses FP4 per-group (group_size=16) with FP8 E4M3 per-tensor scale quantization.
    This is a two-stage quantization where the scale itself is quantized to FP8 E4M3 format.
    Weights use static quantization while activations use dynamic quantization.
    """

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        weight_spec = ScaleQuantSpec(
            first_stage=FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=False, scale_type="float32"),
            second_stage=FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False, scale_type="float32"),
        ).to_quantization_spec()
        input_spec = ScaleQuantSpec(
            first_stage=FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=True, scale_type="float32"),
            second_stage=FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False, scale_type="float32"),
        ).to_quantization_spec()
        return QLayerConfig(weight=weight_spec, input_tensors=input_spec)


class AmdFP4GlobalScaleScheme(QuantizationScheme):
    """Scheme for FP4 per-group quantization with FP8 E5M3 global scale quantization for both weights and activations.

    Uses FP4 per-group with FP8 E5M3 per-tensor global scale quantization.
    This is a two-stage quantization where the scale itself is quantized to FP8 E5M3 format.
    Weights use static quantization while activations use dynamic quantization.
    """

    def __init__(self, group_size: int) -> None:
        self.group_size = group_size

    @property
    def config(self) -> QLayerConfig:
        weight_spec = ScaleQuantSpec(
            first_stage=FP4PerGroupSpec(ch_axis=-1, group_size=self.group_size, is_dynamic=False, scale_type="float32"),
            second_stage=FP8E5M3PerTensorSpec(observer_method="min_max", is_dynamic=False, scale_type="float32"),
        ).to_quantization_spec()
        input_spec = ScaleQuantSpec(
            first_stage=FP4PerGroupSpec(ch_axis=-1, group_size=self.group_size, is_dynamic=True, scale_type="float32"),
            second_stage=FP8E5M3PerTensorSpec(observer_method="min_max", is_dynamic=False, scale_type="float32"),
        ).to_quantization_spec()
        return QLayerConfig(weight=weight_spec, input_tensors=input_spec)


class INT4_FP8Scheme(QuantizationScheme):
    """Scheme with INT4 weights and FP8 activations (a.k.a. "W4A8").

    The scheme name follows the ``<weight_format>_<activation_format>`` convention, the same
    style as ``mxfp4_fp8``. Concretely:

    Weight  (4-bit, INT4):
        - Quantized to INT4 (4-bit signed integer), the final stored weight format.
        - Quantization is *progressive* (two stages): the high-precision weight is first
          quantized to FP8 E4M3 per-tensor, then that result is re-quantized to INT4.
        - INT4 stage is per-channel (ch_axis=0), symmetric, static (no runtime calibration),
          using min-max observation, half-even rounding, and a float32 scale.

    Activation (8-bit, FP8):
        - Quantized to FP8 E4M3 (8-bit floating point, 4 exponent / 3 mantissa bits).
        - Per-tensor, dynamic (scale computed at runtime from each input), min-max, float32 scale.

    This matches the AMD-Quark INT4-weight / FP8-activation recipe used by models such as
    ``amd/Kimi-K2-Thinking-W4A8``.
    """

    def __init__(self) -> None:
        pass

    @property
    def config(self) -> QLayerConfig:
        # Weight: progressive two-stage quantization, FP8 E4M3 per-tensor static -> INT4 per-channel static.
        weight_spec = ProgressiveSpec(
            first_stage=FP8E4M3PerTensorSpec(observer_method="min_max", scale_type="float32", is_dynamic=False),
            second_stage=Int4PerChannelSpec(
                symmetric=True, scale_type="float32", round_method="half_even", ch_axis=0, is_dynamic=False
            ),
        ).to_quantization_spec()
        # Activation: FP8 E4M3 per-tensor dynamic.
        input_spec = FP8E4M3PerTensorSpec(
            observer_method="min_max", scale_type="float32", is_dynamic=True
        ).to_quantization_spec()
        return QLayerConfig(weight=weight_spec, input_tensors=input_spec)


class QuantizationSchemeCollection:
    """Collection for quantization schemes."""

    def __init__(self) -> None:
        self._schemes: dict[str, QuantizationScheme] = {}
        self._collect_supported_schemes()
        self._custom_registered_schemes: list[str] = []

    def _collect_supported_schemes(self) -> None:
        """Collect all supported quantization schemes."""
        # INT4 weight-only schemes
        self._schemes["int4_wo_32"] = Int4WeightOnlyScheme(group_size=32)
        self._schemes["int4_wo_64"] = Int4WeightOnlyScheme(group_size=64)
        self._schemes["int4_wo_128"] = Int4WeightOnlyScheme(group_size=128)
        self._schemes["int4_wo_per_channel"] = Int4WeightOnlyPerChannelScheme()

        # INT4 weight + activation schemes
        self._schemes["int4_wa_64"] = Int4WeightAndActivationScheme(group_size=64)

        # UINT4 weight-only schemes
        self._schemes["uint4_wo_32"] = Uint4WeightOnlyScheme(group_size=32)
        self._schemes["uint4_wo_64"] = Uint4WeightOnlyScheme(group_size=64)
        self._schemes["uint4_wo_128"] = Uint4WeightOnlyScheme(group_size=128)
        self._schemes["uint4_wo_per_channel"] = Uint4WeightOnlyPerChannelScheme()

        # INT8 scheme
        self._schemes["int8"] = Int8Scheme()

        # FP8 quantization schemes
        self._schemes["fp8"] = FP8Scheme()
        self._schemes["ptpc_fp8"] = PTPCFP8Scheme()

        # OCP MXFP quantization schemes
        self._schemes["mxfp4"] = MXFP4Scheme()
        self._schemes["mxfp4_weight_only"] = MXFP4WeightOnlyScheme()
        self._schemes["mxfp6_e3m2"] = MXFP6E3M2Scheme()
        self._schemes["mxfp6_e2m3"] = MXFP6E2M3Scheme()
        self._schemes["mxfp4_mxfp6_e2m3"] = MXFP4_MXFP6E2M3Scheme()
        self._schemes["mxfp4_fp8"] = MXFP4_FP8Scheme()

        # amdfp4 quantization schemes
        self._schemes["amdfp4"] = AmdFP4Scheme(group_size=16)
        self._schemes["amdfp4_g32"] = AmdFP4Scheme(group_size=32)

        # INT4 weight (progressive FP8 E4M3 -> INT4 per-channel) + FP8 E4M3 dynamic activation ("W4A8")
        self._schemes["int4_fp8"] = INT4_FP8Scheme()

        # MX6 quantization schemes
        self._schemes["mx6"] = MX6Scheme()

        # BFP16 quantization schemes
        self._schemes["bfp16"] = BFP16Scheme()

        # Block-scale schemes
        # NVFP4: FP4 with group_size=16 and FP8 E4M3 per-group scale.
        self._schemes["nvfp4"] = FP4Block16ScaleE4M3Scheme()
        # Legacy alias kept for backward compatibility with saved configs / older scripts.
        self._schemes["fp4_block16_scale_e4m3"] = self._schemes["nvfp4"]
        self._schemes["amdfp4_global16"] = AmdFP4GlobalScaleScheme(group_size=16)  # E5M3 global scale, group_size=16
        self._schemes["amdfp4_global32"] = AmdFP4GlobalScaleScheme(group_size=32)  # E5M3 global scale, group_size=32

        # TODO: add later the following (names to be defined) that do not involve a global scale:
        # fp4_block16, with E8M0 scale.
        # fp4_block16, with E5M3 scale + global scale.
        # fp4_block16, with E5M3 scale + NO global scale.
        # fp4_block32, with E4M3 scale.
        # fp4_block32, with E5M3 scale.
        # fp4_block32 with E8M0 scale already exists under the name "mxfp4".

    def register_scheme(self, scheme_name: str, scheme: QuantizationScheme) -> None:
        """Register a quantization scheme."""
        if scheme_name in self._schemes:
            raise ValueError(f"Scheme '{scheme_name}' already registered, please use a different name.")
        self._schemes[scheme_name] = scheme
        self._custom_registered_schemes.append(scheme_name)

    def unregister_scheme(self, scheme_name: str) -> None:
        """Unregister a quantization scheme."""
        if scheme_name not in self._schemes:
            raise ValueError(f"Scheme '{scheme_name}' not found, please check the name.")
        if scheme_name not in self._custom_registered_schemes:
            raise ValueError(
                f"Scheme '{scheme_name}' not registered as custom scheme, quark built-in schemes cannot be unregistered."
            )
        del self._schemes[scheme_name]
        self._custom_registered_schemes.remove(scheme_name)

    def get_supported_schemes(self) -> list[str]:
        """Get list of supported quantization schemes."""
        return list(self._schemes.keys())

    def get_scheme(self, scheme_name: str) -> QuantizationScheme:
        """Get a quantization scheme by name."""
        return self._schemes[scheme_name]


# Substrings that mark a user pattern as targeting a self-attention container
# (and therefore valid as the base spec for kv/q projections within it).
_ATTN_TOKENS = ("self_attn", "self_attention", "attn", "attention")


def _resolve_projection_base_spec(config: QConfig, projection_pattern: str) -> QLayerConfig:
    """Resolve the base ``QLayerConfig`` for a kv/q projection pattern.

    Pure structural scan over ``layer_quant_config`` keys. Two ways to match:

    1. The user pattern's last segment equals this projection's leaf — catches
       direct writes like ``{"*k_proj": ...}`` or ``model.layers.5.self_attn.k_proj``.
    2. The user pattern contains a self-attention token — catches ``*self_attn*``,
       ``*layers.0.self_attn.*``, chatglm ``*self_attention*``, dbrx
       ``*norm_attn_norm.attn.*``.

    Falls back to ``global_quant_config`` when both miss. Out of scope:
    non-attention overrides (``*mlp*``), per-layer differentiation, and
    positional broad overrides like ``*model.layers.0.*`` that don't carry an
    attention token. The broader fix is to stop synthesising entries back into
    ``layer_quant_config`` and compose ``kv_cache_quant_config`` at real-layer
    resolution time instead.
    """
    proj_leaf = projection_pattern.rsplit(".", 1)[-1].strip("*")
    for pattern, spec in config.layer_quant_config.items():
        if proj_leaf and pattern.endswith(proj_leaf):
            return spec
        if any(token in pattern for token in _ATTN_TOKENS):
            return spec
    return cast(QLayerConfig, config.global_quant_config)


class LLMTemplate:
    """
    A configuration template that defines how to quantize specific types of LLM models.

    Each LLM architecture (like llama, qwen, deepseek, etc.) has its own unique structure and naming patterns
    for layers. This template allows specifying those patterns and quantization settings in a reusable way.

    :param str model_type: Type of the LLM model.
    :param List[str] kv_layers_name: List of k_proj and v_proj layer name patterns to match. Default is ``None``.
    :param Union[str, List[str]] q_layer_name: q_proj layer name pattern to match. Default is ``None``.
    :param List[str] exclude_layers_name: List of layer name patterns to exclude from quantization. Default is ``[]``.
    :param Optional[Dict[str, AlgoConfig]] algorithm_configs: Dictionary of algorithm names to algorithm
        configurations. Example: ``{"awq": custom_awq_config, "gptq": custom_gptq_config}``. Default is ``None``.
    :param Dict[str, AlgoConfig] legacy_algorithm_parameters: Legacy keyword arguments in ``<algorithm>_config``
        form (for backward compatibility). Passing these will emit a deprecation warning. Use
        ``algorithm_configs`` for new code.

    Note:
        - The quantization schemes supported by the template are:
            - fp8
            - ptpc_fp8
            - int4_wo_32
            - int4_wo_64
            - int4_wo_128
            - int4_wo_per_channel
            - uint4_wo_32
            - uint4_wo_64
            - uint4_wo_128
            - uint4_wo_per_channel
            - int8
            - mxfp4
            - mxfp6_e3m2
            - mxfp6_e2m3
            - mx6
            - bfp16
            - int4_fp8
        - The quantization algorithms supported by the template are:
            - awq
            - gptq
            - gptaq
            - smoothquant
            - autosmoothquant
            - qronos
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
    _SUPPORTED_ALGORITHMS = get_supported_algorithm_types()
    _SUPPORTED_KV_CACHE_SCHEMES = ["fp8"]
    _SUPPORTED_ATTENTION_SCHEMES = ["fp8"]

    def __init__(
        self,
        model_type: str,
        kv_layers_name: list[str] | None = None,
        q_layer_name: str | list[str] | None = None,
        gate_up_layers_name: list[str] | None = None,
        exclude_layers_name: list[str] = [],
        algorithm_configs: dict[str, AlgoConfig | None] | None = None,
        f2f_weight_converters: list[WeightConverter] | None = None,
        **legacy_algorithm_parameters: AlgoConfig | None,
    ):
        self.model_type = model_type
        self.kv_layers_name = kv_layers_name
        self.q_layer_name = q_layer_name
        self.exclude_layers_name = exclude_layers_name
        # Model-specific gate/up projection layer names for shared scale groups.
        self.gate_up_layers_name = gate_up_layers_name if gate_up_layers_name is not None else ["gate_proj", "up_proj"]
        # Pre-quantization checkpoint transformations applied only on the file-to-file path.
        # For models whose checkpoint stores fused MoE expert tensors, these converters
        # unfuse them into per-expert tensors so the resulting shards match the model's
        # forward-pass layout. ``None`` means no transformation is required.
        self.f2f_weight_converters = f2f_weight_converters

        # Algorithm-specific configuration fields. New code should use
        # `algorithm_configs`; `legacy_algorithm_parameters` is kept only
        # for backward compatibility and emits a deprecation warning.
        self.algo_config: dict[str, AlgoConfig | None] = {}
        for supported_algorithm_name in self._SUPPORTED_ALGORITHMS:
            self.algo_config[supported_algorithm_name] = None

        supported_legacy_parameter_to_algorithm_name = {
            f"{supported_algorithm_name}_config": supported_algorithm_name
            for supported_algorithm_name in self._SUPPORTED_ALGORITHMS
        }

        legacy_algorithm_parameter_names: list[str] = []
        for legacy_parameter_name, algorithm_config in legacy_algorithm_parameters.items():
            algorithm_name = supported_legacy_parameter_to_algorithm_name.get(legacy_parameter_name)
            if algorithm_name is None:
                supported_legacy_parameter_names = sorted(supported_legacy_parameter_to_algorithm_name.keys())
                raise ValueError(
                    f"Unsupported legacy algorithm keyword '{legacy_parameter_name}'. "
                    "Legacy algorithm keyword arguments must use the '<algorithm>_config' format "
                    f"and one of {supported_legacy_parameter_names}. "
                    "Use `algorithm_configs` for new code."
                )

            self.algo_config[algorithm_name] = algorithm_config
            legacy_algorithm_parameter_names.append(legacy_parameter_name)

        if legacy_algorithm_parameter_names:
            sorted_parameter_names = sorted(legacy_algorithm_parameter_names)
            deprecation_message = (
                "Deprecated keyword arguments were used when initializing `LLMTemplate`: "
                f"{sorted_parameter_names}. These legacy keyword arguments will be removed soon. "
                "Please configure algorithm parameters with `algorithm_configs`, for example "
                "`LLMTemplate(..., algorithm_configs={'awq': custom_awq_config})`."
            )
            logger.warning(deprecation_message)

        if algorithm_configs is not None:
            for algorithm_name, algorithm_config in algorithm_configs.items():
                normalized_algorithm_name = algorithm_name.lower()
                if normalized_algorithm_name not in self._SUPPORTED_ALGORITHMS:
                    raise ValueError(
                        f"Unsupported algorithm '{algorithm_name}' in `algorithm_configs`. "
                        f"Supported algorithms: {self._SUPPORTED_ALGORITHMS}."
                    )
                self.algo_config[normalized_algorithm_name] = algorithm_config

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
                q_layer_name="*q_proj",
                exclude_layers_name=["lm_head"],
            )

            # Register template
            LLMTemplate.register_template(template)
        """
        if template.model_type in cls._templates:
            logger.warning(
                f"Template '{template.model_type}' already registered, will overwrite the existing template."
            )
        cls._templates[template.model_type] = template

    @classmethod
    def get(cls, model_type: str) -> LLMTemplate:
        """Get a template by model type.

        :param str model_type: Type of the model. It is obtained from the original LLM HuggingFace model's ``model.config.model_type`` attribute. When the model_type field is not defined, the ``model.config.architecture[0]`` is assigned as the model_type..

        Available model types:

            - chatglm
            - cohere
            - dbrx
            - deepseek
            - deepseek_v2
            - deepseek_v3
            - deepseek_v32
            - deepseek_v4
            - deepseek_vl_v2
            - gemma2
            - gemma3
            - gemma3_text
            - glm4_moe
            - glm4_moe_lite
            - glm_moe_dsa
            - gptj
            - gpt_oss
            - granitemoehybrid
            - grok-1
            - instella
            - kimi_k2
            - kimi_k25
            - llama
            - llama4
            - minimax_m2
            - minimax_m3_vl
            - mistral
            - mixtral
            - mllama
            - olmo
            - opt
            - phi
            - phi3
            - qwen
            - qwen2
            - qwen2_moe
            - qwen3
            - qwen3_moe
            - qwen3_next
            - qwen3_vl_moe
            - qwen3_5_moe

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
                "you can refer to the example comments within the `register_template` function for instructions on how to register a new model."
            )

        return cls._templates[model_type]

    # Register a new quantization scheme for the template
    @classmethod
    def register_scheme(cls, scheme_name: str, config: QLayerConfig) -> None:
        """
        Register a new quantization scheme for LLMTemplate class.

        :param str scheme_name: Name of the scheme.
        :param QLayerConfig config: Configuration for the scheme.

        Example:

        .. code-block:: python

            # Register a new quantization scheme ``int8_wo (int8 weight-only)`` to the template
            from quark.torch import LLMTemplate
            from quark.torch.quantization.config.config import Int8PerTensorSpec, QLayerConfig

            quant_spec = Int8PerTensorSpec(observer_method="min_max", symmetric=True, scale_type="float",
                                           round_method="half_even", is_dynamic=False).to_quantization_spec()
            global_config = QLayerConfig(weight=quant_spec)

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

    @classmethod
    def get_supported_schemes(cls) -> list[str]:
        """Get list of supported quantization schemes."""
        return cls._SUPPORTED_SCHEMES

    def get_config(
        self,
        scheme: str,
        algorithm: str | list[str] | None = None,
        kv_cache_scheme: str | None = None,
        min_kv_scale: float = 0.0,
        attention_scheme: str | None = None,
        layer_config: dict[str, str] | None = None,
        layer_type_config: dict[type[nn.Module], str] | None = None,
        exclude_layers: list[str] | None = None,
        algo_configs: dict[str, AlgoConfig] | None = None,
        shared_scale_groups: list[list[str]] | None = None,
    ) -> QConfig:
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
        :param Optional[Dict[str, AlgoConfig]] algo_configs: Dictionary of algorithm names to their configurations.
        :param Optional[List[List[str]]] shared_scale_groups: Groups of layer name suffixes that should share the global-scale observer. Each inner list represents a group of parallel layer suffixes (e.g. ``["q_proj", "k_proj", "v_proj"]``). If ``None``, the default for the scheme is used. Pass ``[]`` to disable.

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
                normalized_algorithm_name = algo.lower()
                if normalized_algorithm_name not in self._SUPPORTED_ALGORITHMS:
                    raise ValueError(f"Unsupported algorithm: {algo}")
        # Check if the KV cache scheme is supported
        if kv_cache_scheme and kv_cache_scheme not in self._SUPPORTED_KV_CACHE_SCHEMES:
            raise ValueError(f"Unsupported KV cache scheme: {kv_cache_scheme}")
        # Check if the attention scheme is supported
        if attention_scheme and attention_scheme not in self._SUPPORTED_ATTENTION_SCHEMES:
            raise ValueError(f"Unsupported attention scheme: {attention_scheme}")

        # Set up base global configuration
        global_config = self._create_global_config(scheme)

        # Resolve shared-scale defaults for the selected scheme.
        sync_moe_expert_input_amax = False
        if shared_scale_groups is None:
            shared_scale_groups, sync_moe_expert_input_amax = self._get_default_shared_scale_settings(scheme)

        # Create config object
        config = QConfig(
            global_quant_config=global_config,
            min_kv_scale=min_kv_scale,
            exclude=self.exclude_layers_name if exclude_layers is None else exclude_layers,
            kv_cache_group=self.kv_layers_name,
            shared_scale_groups=shared_scale_groups,
            sync_moe_expert_input_amax=sync_moe_expert_input_amax,
        )
        # Apply algorithm if specified
        if algorithm:
            config = self._set_algorithm(config, algorithm, algo_configs)

        # Set layer quantization configuration first
        # Apply per-layer configuration overrides
        if layer_config:
            config = self._set_layer_name_config(config, layer_config)
        if layer_type_config:
            config = self._set_layer_type_config(config, layer_type_config)

        # Set KV cache quantization quantization configuration after setting layer quantization configuration to aviod the conflicts
        # Apply KV cache quantization if specified
        if kv_cache_scheme:
            config = self._set_kv_cache_config(config, kv_cache_scheme)

        # Apply attention quantization if specified
        if attention_scheme:
            config = self._set_attention_config(config, attention_scheme)

        return config

    def _create_global_config(self, scheme: str) -> QLayerConfig:
        return LLMTemplate._SCHEME_COLLECTION.get_scheme(scheme).config

    def _get_default_shared_scale_settings(self, scheme: str) -> tuple[list[list[str]], bool]:
        """Return default shared-scale settings for a given quantization scheme.

        The returned tuple contains ``shared_scale_groups`` and the default
        ``sync_moe_expert_input_amax`` value. Currently only the ``nvfp4``
        scheme (also exposed as the legacy alias ``fp4_block16_scale_e4m3``)
        uses non-empty shared-scale groups and enables MoE expert input amax
        synchronization by default.
        """
        if scheme not in ("nvfp4", "fp4_block16_scale_e4m3"):
            return [], False
        return [["q_proj", "k_proj", "v_proj"], self.gate_up_layers_name], True

    def _set_algorithm(
        self, config: QConfig, algorithm: str | list[str], algo_configs: dict[str, AlgoConfig] | None = None
    ) -> QConfig:
        if isinstance(algorithm, str):
            algorithm = [algorithm]

        # Clone algo_config to avoid modifying the original algo_config
        effective_algo_config = dict(self.algo_config)
        if algo_configs:
            for algo_name, algo_cfg in algo_configs.items():
                effective_algo_config[algo_name.lower()] = algo_cfg

        for algo in algorithm:
            if config.algo_config is None:
                config.algo_config = []
            algorithm_name = algo.lower()

            if algorithm_name not in self._SUPPORTED_ALGORITHMS:
                raise ValueError(
                    f"The algorithm {algorithm_name} is not supported in Quark. Are you sure it is one of {self._SUPPORTED_ALGORITHMS}?"
                )

            if effective_algo_config[algorithm_name] is None:
                raise NotImplementedError(
                    f"No built-in {algorithm_name} configuration is available for the '{self.model_type}' architecture. "
                    f"Pass a custom configuration via `algo_configs={{'{algorithm_name}': <your AlgoConfig>}}` "
                    f"to `LLMTemplate.get_config()`. "
                    f"See the Quark documentation for details."
                )

            config.algo_config.append(effective_algo_config[algorithm_name])

        return config

    def _set_kv_cache_config(self, config: QConfig, kv_cache_scheme: str) -> QConfig:
        # Use pattern matching to identify KV projection layers
        if self.kv_layers_name is None:
            return config

        if kv_cache_scheme == "fp8":
            spec = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()

            for layer_name in self.kv_layers_name:
                # Canonical resolver first; returns None for excluded patterns,
                # global for "no override matched", or the user spec otherwise.
                layer_quant_config = get_layer_quant_config(config, nn.Linear, layer_name)
                if layer_quant_config is not None and layer_quant_config is config.global_quant_config:
                    layer_quant_config = _resolve_projection_base_spec(config, layer_name)

                # exclude applies to weight/input only; kv-cache (output) is
                # independent — matches the runtime exclude-with-kv branch in
                # get_layer_quant_config.
                if layer_quant_config is None:
                    weight = None
                    input_tensors = None
                else:
                    weight = layer_quant_config.weight
                    input_tensors = layer_quant_config.input_tensors
                    config.layer_quant_config[layer_name] = QLayerConfig(
                        weight=weight,
                        input_tensors=input_tensors,
                        output_tensors=spec,
                    )
                config.kv_cache_quant_config[layer_name] = QLayerConfig(
                    weight=weight,
                    input_tensors=input_tensors,
                    output_tensors=spec,
                )
        else:
            raise ValueError(f"Unsupported KV cache quantization scheme: {kv_cache_scheme}")
        return config

    def _set_attention_config(self, config: QConfig, attention_scheme: str) -> QConfig:
        if attention_scheme == "fp8":
            spec = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()
            config.softmax_quant_spec = spec

            if self.q_layer_name is not None:
                if isinstance(self.q_layer_name, str):
                    self.q_layer_name = [self.q_layer_name]
                for q_layer_name in self.q_layer_name:
                    layer_quant_config = get_layer_quant_config(config, nn.Linear, q_layer_name)
                    if layer_quant_config is None:
                        continue
                    if layer_quant_config is config.global_quant_config:
                        layer_quant_config = _resolve_projection_base_spec(config, q_layer_name)
                    config.layer_quant_config[q_layer_name] = QLayerConfig(
                        weight=layer_quant_config.weight,
                        input_tensors=layer_quant_config.input_tensors,
                        output_tensors=spec,
                    )
        else:
            raise ValueError(f"Unsupported attention quantization scheme: {attention_scheme}")
        return config

    def _set_layer_name_config(self, config: QConfig, layer_name_config: dict[str, str]) -> QConfig:
        for layer_name, layer_scheme in layer_name_config.items():
            config.layer_quant_config[layer_name] = LLMTemplate._SCHEME_COLLECTION.get_scheme(layer_scheme).config
        return config

    def _set_layer_type_config(self, config: QConfig, layer_type_config: dict[type[nn.Module], str]) -> QConfig:
        for layer_type, layer_scheme in layer_type_config.items():
            config.layer_type_quant_config[layer_type] = LLMTemplate._SCHEME_COLLECTION.get_scheme(layer_scheme).config
        return config


# Default template configurations
DEFAULT_TEMPLATES = {
    "chatglm": {
        "kv_layers_name": ["*query_key_value"],
        "q_layer_name": "*query_key_value",
        "exclude_layers_name": ["transformer.output_layer"],
    },
    "cohere": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
    },
    "dbrx": {
        "kv_layers_name": ["*Wqkv"],
        "q_layer_name": "*Wqkv",
        "exclude_layers_name": ["lm_head", "*router.layer"],
        "gate_up_layers_name": ["w1", "v1"],
    },
    "deepseek": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head", "*.gate", "*.gate.linear"],
    },
    "deepseek_v2": {
        "kv_layers_name": ["*kv_b_proj"],
        "q_layer_name": ["*q_a_proj", "*q_b_proj"],
        "exclude_layers_name": ["lm_head", "*self_attn*", "*mlp.gate", "*mlp.gate.linear"],
    },
    "deepseek_v3": {
        "kv_layers_name": ["*kv_b_proj"],
        "q_layer_name": ["*q_a_proj", "*q_b_proj"],
        "exclude_layers_name": ["lm_head", "*self_attn*", "*mlp.gate", "*mlp.gate.linear"],
    },
    "deepseek_v32": {
        "kv_layers_name": ["*kv_b_proj"],
        "q_layer_name": ["*q_a_proj", "*q_b_proj"],
        "exclude_layers_name": ["lm_head", "*mlp.gate", "*mlp.gate.linear", "model.layers.61.*", "*self_attn*"],
    },
    "deepseek_v4": {
        "kv_layers_name": ["*wkv"],
        "q_layer_name": ["*wq_a", "*wq_b"],
        "exclude_layers_name": [
            "*attn*",
            "embed",
            "head",
            "*ffn.gate*",
            "hc_*",
            "mtp.*",
        ],
    },
    "deepseek_vl_v2": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": ["*q_proj"],
        "exclude_layers_name": ["lm_head", "model.sam_model*", "model.vision_model*", "model.projector*"],
    },
    "gemma2": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
    },
    "gemma3": {
        "kv_layers_name": ["*language_model.*k_proj", "*language_model.*v_proj"],
        "q_layer_name": "*language_model.*q_proj",
        "exclude_layers_name": ["*vision_tower*", "*multi_modal_projector*", "*lm_head"],
    },
    "gemma3_text": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["*lm_head"],
    },
    "glm4_moe": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": [
            "lm_head",
            "*mlp.gate",
            "*mlp.gate.linear",
            "*self_attn*",
            "*shared_experts.*",
            "*mlp.down_proj",
            "*mlp.gate_proj",
            "*mlp.up_proj",
        ],
    },
    "glm4_moe_lite": {
        "kv_layers_name": ["*kv_a_proj_with_mqa", "*kv_b_proj"],
        "q_layer_name": "*q_a_proj",
        "exclude_layers_name": [
            "lm_head",
            "*self_attn*",
            "*mlp.gate",
        ],
    },
    "glm_moe_dsa": {
        "kv_layers_name": ["*kv_a_proj_with_mqa", "*kv_b_proj"],
        "q_layer_name": "*q_a_proj",
        "exclude_layers_name": [
            "*self_attn*",
            "*mlp.gate",
            "*lm_head",
            "*mlp.gate_proj",
            "*mlp.up_proj",
            "*mlp.down_proj",
        ],
    },
    "gptj": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
    },
    "gpt_oss": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head", "*router*"],
    },
    "granitemoehybrid": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head", "*router*"],
    },
    "grok-1": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head", "*.gate", "*.gate.linear"],
        "gate_up_layers_name": ["linear", "linear_v"],
    },
    "instella": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
    },
    "kimi_k2": {
        "kv_layers_name": ["*kv_a_proj_with_mqa", "*kv_b_proj"],
        "q_layer_name": "*q_a_proj",
        "exclude_layers_name": [
            "*self_attn*",
            "*mlp.gate",
            "*lm_head",
            "*mlp.gate_proj",
            "*mlp.up_proj",
            "*mlp.down_proj",
            "*shared_experts*",
        ],
    },
    "kimi_k25": {
        "kv_layers_name": ["*kv_a_proj_with_mqa", "*kv_b_proj"],
        "q_layer_name": "*q_a_proj",
        "exclude_layers_name": [
            "*self_attn*",
            "*mlp.gate",
            "*mlp.gate.linear",
            "*lm_head",
            "*mlp.gate_proj",
            "*mlp.up_proj",
            "*mlp.down_proj",
            "*shared_experts*",
            "*mm_projector*",
            "*vision_tower*",
        ],
    },
    "llama": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
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
    },
    "minimax_m2": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head", "*block_sparse_moe.gate*", "*self_attn*"],
        "gate_up_layers_name": ["w1", "w3"],
    },
    "minimax_m3_vl": {
        "kv_layers_name": ["*language_model.*k_proj", "*language_model.*v_proj"],
        "q_layer_name": "*language_model.*q_proj",
        "exclude_layers_name": [
            "*lm_head",
            "*vision_tower*",
            "*multi_modal_projector*",
            "*patch_merge_mlp*",
            "*block_sparse_moe.gate",
            "*self_attn*",
        ],
    },
    "mistral": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
    },
    "mixtral": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head", "*.gate", "*.gate.linear"],
    },
    "mllama": {
        "kv_layers_name": ["*language_model.*k_proj", "*language_model.*v_proj"],
        "q_layer_name": "*self_attn.q_proj",
        "exclude_layers_name": ["*lm_head", "*patch_embedding", "multi_modal_projector"],
    },
    "olmo": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
    },
    "opt": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
    },
    "phi": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
    },
    "phi3": {
        "kv_layers_name": ["*qkv_proj"],
        "q_layer_name": "*qkv_proj",
        "exclude_layers_name": ["lm_head"],
    },
    "qwen": {
        "kv_layers_name": ["*c_attn"],
        "q_layer_name": "*c_attn",
        "exclude_layers_name": ["lm_head"],
    },
    "qwen2": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
    },
    "qwen2_moe": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head", "*.gate", "*.gate.linear", "*.shared_expert_gate"],
    },
    "qwen3": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head"],
    },
    "qwen3_moe": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head", "*.gate", "*.gate.linear"],
    },
    "qwen3_next": {
        "kv_layers_name": ["*qkvz"],
        "q_layer_name": "*qkvz",
        "exclude_layers_name": [
            "lm_head",
            "mtp.fc",
            "*linear_attn.in_proj_ba",
            "*linear_attn.in_proj_qkvz",
            "*mlp.gate",
            "*mlp.gate.linear",
            "*mlp.shared_expert_gate",
            "*self_attn.k_proj",
            "*self_attn.q_proj",
            "*self_attn.v_proj",
        ],
    },
    "qwen3_vl_moe": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": ["lm_head", "*mlp.gate", "*mlp.gate.linear", "*.visual.*"],
    },
    "qwen3_5_moe": {
        "kv_layers_name": ["*k_proj", "*v_proj"],
        "q_layer_name": "*q_proj",
        "exclude_layers_name": [
            "lm_head",
            "model.visual.*",
            "mtp.*",
            "*mlp.gate",
            "*mlp.gate.linear",
            "*shared_expert_gate*",
            "*.linear_attn.*",
            "*.self_attn.*",
            "*.shared_expert.*",
        ],
        # qwen3_5_moe stores experts as fused tensors:
        #   gate_up_proj: (num_experts, 2*intermediate, hidden)
        #   down_proj:    (num_experts, hidden, intermediate)
        # The file-to-file path needs to unfuse them into per-expert tensors so the
        # resulting shards match the per-expert nn.Linear forward layout.
        "f2f_weight_converters": [
            WeightConverter(
                "gate_up_proj",
                ["gate_proj.weight", "up_proj.weight"],
                operations=[SplitFusedExperts(split_axis=0)],
            ),
            WeightConverter(
                "down_proj",
                ["down_proj.weight"],
                operations=[SplitFusedExperts(split_axis=0)],
            ),
        ],
    },
}


def _create_template_from_config(model_type: str, config: dict[str, Any]) -> LLMTemplate:
    """create a template from configuration dictionary."""
    algorithm_configs: dict[str, AlgoConfig | None] = {}
    for supported_algorithm_name in LLMTemplate._SUPPORTED_ALGORITHMS:
        algorithm_configs[supported_algorithm_name] = get_algo_config(supported_algorithm_name, model_type)

    return LLMTemplate(
        model_type=model_type,
        kv_layers_name=config["kv_layers_name"],
        q_layer_name=config["q_layer_name"],
        gate_up_layers_name=config.get("gate_up_layers_name"),
        exclude_layers_name=config["exclude_layers_name"],
        algorithm_configs=algorithm_configs,
        f2f_weight_converters=config.get("f2f_weight_converters"),
    )  # type: ignore


"""
Developer Note for Quark Engineers:
====================================

To add a new model template, follow these steps:

    1. Add the model configuration to DEFAULT_TEMPLATES dictionary above.
    2. Add corresponding algorithm configs to the algo config registry if the algorithm is needed.
    (see quark/torch/quantization/config/algo_config.py)
    3. Algorithm configuration lookup always uses ``model_type`` for built-in templates.
    4. Update the docstring list in LLMTemplate.get() method to include the new model type.
    5. Test the new template with various quantization schemes and algorithms

Example for adding "new_model":

.. code-block:: python

    "new_model": {
        "kv_layers_name": ["*attention.k_proj", "*attention.v_proj"],
        "q_layer_name": "*attention.q_proj",
        "exclude_layers_name": ["lm_head"],
    }
"""

# Register built-in templates
for model_type, config in DEFAULT_TEMPLATES.items():
    if model_type not in LLMTemplate._templates:
        template = _create_template_from_config(model_type, config)
        LLMTemplate.register_template(template)
