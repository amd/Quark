#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Config verificiation helper functions for Brevitas quantizer."""

from typing import List, Optional

import quark.torch.quantization.config.type as quark_config_type
import quark.torch.extensions.brevitas.config as brevitas_config
import quark.torch.extensions.brevitas.algos as brevitas_algos

from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)


class ConfigVerifier:
    """
    This is a helper utility to inspect Brevitas quantization configs and ensure they are valid. It'll warn the user about parameters that need to be set or that won't have any effect and it will highlight possible improvements where possible.
    """

    @classmethod
    def verify_config(cls, config: brevitas_config.Config) -> None:
        if config.backend is not brevitas_config.Backend.layerwise:
            raise ValueError(f"Backend type: {config.backend} is not supported.")

        cls._verify_global_config(config.global_quant_config)
        cls._verify_pre_quant_configs(config.pre_quant_opt_config, config)
        cls._verify_post_quant_configs(config.algo_config, config)

    @classmethod
    def _verify_global_config(cls, global_config: brevitas_config.QuantizationConfig) -> None:
        if global_config.bias is not None and global_config.input_tensors is None:
            raise ValueError("There must be input quantization if bias quantization is used.")

        cls._verify_activation_quant_spec(global_config.input_tensors)
        cls._verify_activation_quant_spec(global_config.output_tensors)
        cls._verify_weight_quant_spec(global_config.weight)
        cls._verify_bias_quant_spec(global_config.bias)

    @classmethod
    def _verify_spec_common(cls, spec: brevitas_config.QuantizationSpec) -> None:
        if spec.quant_type is brevitas_config.QuantType.float_quant:
            if spec.exponent_bit_width is None or spec.mantissa_bit_width is None:
                raise ValueError("Exponent and mantissa bit width must be specified for float quantization.")

            if spec.symmetric is False:
                raise ValueError("Asymmetric quantization is not supported with float quantization.")

    @classmethod
    def _verify_activation_quant_spec(cls, spec: Optional[brevitas_config.QuantizationSpec]) -> None:
        if spec is not None:
            cls._verify_spec_common(spec)

    @classmethod
    def _verify_weight_quant_spec(cls, spec: Optional[brevitas_config.QuantizationSpec]) -> None:
        if spec is not None:
            cls._verify_spec_common(spec)

    @classmethod
    def _verify_bias_quant_spec(cls, spec: Optional[brevitas_config.QuantizationSpec]) -> None:
        if spec is not None:
            cls._verify_spec_common(spec)

            if spec.quant_type is not brevitas_config.QuantType.int_quant:
                raise ValueError("Only integer bias quantization is supported.")

            if spec.qscheme is not quark_config_type.QSchemeType.per_tensor:
                logger.warning(
                    "qscheme for bias quantization is implicitly per_tensor, different values will be ignored.")

            if spec.symmetric is False:
                logger.warning("symmetric is not used for bias quantization.")

            if spec.scale_type is not quark_config_type.ScaleType.float:
                logger.warning("scale_type is not used for bias quantization.")

            if spec.param_type is not brevitas_config.ParamType.stats:
                logger.warning("param_type is not used for bias quantization.")

            if spec.exponent_bit_width is not None:
                logger.warning("exponent_bit_width is not used for bias quantization")

            if spec.mantissa_bit_width is not None:
                logger.warning("mantissa_bit_width is not used for bias quantization")

    @classmethod
    def _verify_pre_quant_configs(cls, pre_quant_configs: List[brevitas_algos.PreQuantOptConfig],
                                  config: brevitas_config.Config) -> None:
        if len(pre_quant_configs) > 0:
            # check if preprocess is in the list
            using_preprocess = False
            for x in pre_quant_configs:
                if isinstance(x, brevitas_algos.Preprocess):
                    using_preprocess = True

            if using_preprocess is False:
                logger.warning(
                    "Preprocess is not being applied, you may want to consider adding it to improve quantization.")

            for idx, pre_config in enumerate(pre_quant_configs):
                if isinstance(pre_config, brevitas_algos.Preprocess):
                    if idx != 0:
                        logger.warning(
                            "Preprocess is being applied after other optimizations, it probably should be first.")
                elif isinstance(pre_config, brevitas_algos.ActivationEqualization):
                    if config.backend == brevitas_config.Backend.layerwise and pre_config.is_layerwise is False:
                        raise ValueError(
                            "ActivationEqualization has is_layerwise set to false but config backend is set to layerwise."
                        )

    @classmethod
    def _verify_post_quant_configs(cls, post_quant_configs: List[brevitas_algos.AlgoConfig],
                                   config: brevitas_config.Config) -> None:
        # check if GPTQ, GPFQ or GPFA2Q are being combined which they shouldn't
        if len(post_quant_configs) > 0:
            count = 0
            count += 1 if any(isinstance(x, brevitas_algos.GPTQ) for x in post_quant_configs) else 0
            count += 1 if any(isinstance(x, brevitas_algos.GPFQ) for x in post_quant_configs) else 0
            count += 1 if any(isinstance(x, brevitas_algos.GPFA2Q) for x in post_quant_configs) else 0

            if count > 1:
                raise ValueError("GPTQ, GPFQ or GPFA2Q should not be mixed, please just use one.")

        if any(isinstance(x, brevitas_algos.GPFQ)
               for x in post_quant_configs) or any(isinstance(x, brevitas_algos.GPFA2Q) for x in post_quant_configs):
            if config.global_quant_config.input_tensors is None:
                raise ValueError("GPFQ and GPFA2Q need input_tensor quantization to be defined.")
