#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy
from dataclasses import fields

from quark.shares.utils.log import ScreenLogger
from quark.torch.quantization.config.config import QuantizationConfig, QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType

logger = ScreenLogger(__name__)


# TODO: can be optimized
def init_quantization_config(quantization_config: QuantizationConfig) -> tuple[bool, bool, bool, bool]:
    is_dynamic = True  # TODO: remove it
    is_weight_only = True

    for field in fields(QuantizationConfig):
        quantization_spec = getattr(quantization_config, field.name)
        if isinstance(quantization_spec, (QuantizationSpec, list)):
            specs_to_check = (
                [quantization_spec] if isinstance(quantization_spec, QuantizationSpec) else quantization_spec
            )
            for spec in specs_to_check:
                if not isinstance(spec, QuantizationSpec):
                    continue
                if (
                    spec.dtype
                    in [
                        Dtype.int8,
                        Dtype.uint8,
                        Dtype.int4,
                        Dtype.uint4,
                        Dtype.int3,
                        Dtype.int2,
                        Dtype.fp8_e4m3,
                        Dtype.fp8_e5m2,
                        Dtype.fp6_e3m2,
                        Dtype.fp6_e2m3,
                        Dtype.mx,
                        Dtype.mx6,
                        Dtype.mx9,
                        Dtype.fp4,
                    ]
                    and spec.is_dynamic is False
                ):
                    is_dynamic = False
                if field.name in ["input_tensors", "output_tensors"] and spec.dtype not in [
                    Dtype.float16,
                    Dtype.bfloat16,
                ]:
                    is_weight_only = False

    if not is_weight_only:
        is_input_dynamic = True
        is_output_dynamic = True
        is_input_contain_scale_per_tensor = False
        is_output_contain_scale_per_tensor = False
        if quantization_config.input_tensors is not None:
            specs_to_check = (
                [quantization_config.input_tensors]
                if isinstance(quantization_config.input_tensors, QuantizationSpec)
                else quantization_config.input_tensors
            )
            is_input_dynamic = all(spec.is_dynamic for spec in specs_to_check if isinstance(spec, QuantizationSpec))
            if not isinstance(quantization_config.input_tensors, QuantizationSpec):
                is_input_contain_scale_per_tensor = any(
                    spec.is_scale_quant and spec.qscheme == QSchemeType.per_tensor
                    for spec in specs_to_check
                    if isinstance(spec, QuantizationSpec)
                )

        if quantization_config.output_tensors is not None:
            specs_to_check = (
                [quantization_config.output_tensors]
                if isinstance(quantization_config.output_tensors, QuantizationSpec)
                else quantization_config.output_tensors
            )
            is_output_dynamic = all(spec.is_dynamic for spec in specs_to_check if isinstance(spec, QuantizationSpec))
            if not isinstance(quantization_config.output_tensors, QuantizationSpec):
                is_output_contain_scale_per_tensor = any(
                    spec.is_scale_quant and spec.qscheme == QSchemeType.per_tensor
                    for spec in specs_to_check
                    if isinstance(spec, QuantizationSpec)
                )

        is_act_dynamic = is_input_dynamic and is_output_dynamic
        is_act_contain_scale_per_tensor = is_input_contain_scale_per_tensor or is_output_contain_scale_per_tensor
    else:
        is_act_dynamic = False
        is_act_contain_scale_per_tensor = False

    return is_dynamic, is_weight_only, is_act_dynamic, is_act_contain_scale_per_tensor


def check_and_adjust_quant_config(quantization_config: QuantizationConfig) -> QuantizationConfig:
    assert isinstance(quantization_config, QuantizationConfig), "Only support check on 'QuantizationConfig'"
    if quantization_config.input_tensors is not None and isinstance(
        quantization_config.input_tensors, QuantizationSpec
    ):
        if (
            quantization_config.input_tensors.qscheme == QSchemeType.per_group
            and quantization_config.input_tensors.group_size is not None
            and quantization_config.input_tensors.group_size > 0
        ):
            if not quantization_config.input_tensors.is_dynamic:
                logger.warning(
                    "For input_tensors, quantization must be dynamic using per-group granularity, forcely set is_dynamic=True."
                )
                input_quant_config_copied = copy.deepcopy(quantization_config.input_tensors)
                input_quant_config_copied.is_dynamic = True
                quantization_config.input_tensors = input_quant_config_copied
                del input_quant_config_copied
    if quantization_config.output_tensors is not None and isinstance(
        quantization_config.output_tensors, QuantizationSpec
    ):
        if (
            quantization_config.output_tensors.qscheme == QSchemeType.per_group
            and quantization_config.output_tensors.group_size is not None
            and quantization_config.output_tensors.group_size > 0
        ):
            if not quantization_config.output_tensors.is_dynamic:
                logger.warning(
                    "For output_tensors, quantization must be dynamic using per-group granularity, forcely set is_dynamic=True."
                )
                output_quant_config_copied = copy.deepcopy(quantization_config.output_tensors)
                output_quant_config_copied.is_dynamic = True
                quantization_config.output_tensors = output_quant_config_copied
                del output_quant_config_copied

    return quantization_config
