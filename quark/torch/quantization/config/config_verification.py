#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy
from typing import Any
from dataclasses import fields

from quark.torch.quantization.config.config import QuantizationConfig, QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType
import quark.torch.quantization.observer.observer as qobs
from quark.shares.utils.log import ScreenLogger, log_errors

logger = ScreenLogger(__name__)


def verify_quantization_spec(any_config: Any) -> None:
    if isinstance(any_config, QuantizationSpec):
        check_quantization_spec(any_config)
    elif hasattr(any_config, '__dataclass_fields__'):
        for field in fields(any_config):
            field_value = getattr(any_config, field.name)
            verify_quantization_spec(field_value)
    elif isinstance(any_config, dict):
        for value in any_config.values():
            verify_quantization_spec(value)
    elif isinstance(any_config, (list, tuple)):
        for item in any_config:
            verify_quantization_spec(item)


@log_errors
def check_quantization_spec(quantization_spec: QuantizationSpec) -> None:
    if quantization_spec.dtype in [
            Dtype.int8, Dtype.uint8, Dtype.int4, Dtype.uint4, Dtype.int32, Dtype.fp8_e4m3, Dtype.fp8_e5m2, Dtype.mx,
            Dtype.mx6, Dtype.mx9
    ]:
        if quantization_spec.is_dynamic is None:
            raise ValueError(
                f"The field `is_dynamic` cannot be None when Dtype is {quantization_spec.dtype.name} in QuantizationSpec."
            )
        if quantization_spec.observer_cls is None:
            raise ValueError(
                f"The field `observer_cls` cannot be None when Dtype is {quantization_spec.dtype.name} in QuantizationSpec."
            )
        if quantization_spec.qscheme is None:
            raise ValueError(
                f"The field `qscheme` cannot be None when Dtype is {quantization_spec.dtype.name} in QuantizationSpec. Please reconfigure the quantization settings accordingly."
            )
    if quantization_spec.dtype in [Dtype.int8, Dtype.uint8, Dtype.int4, Dtype.uint4, Dtype.int32]:
        if quantization_spec.symmetric is None:
            raise ValueError(
                f"The field `symmetric` cannot be None when Dtype is {quantization_spec.dtype.name} in QuantizationSpec. Please reconfigure the quantization settings accordingly."
            )
        if quantization_spec.round_method is None:
            raise ValueError(
                f"The field `round_method` cannot be None when Dtype is {quantization_spec.dtype.name} in QuantizationSpec. Please reconfigure the quantization settings accordingly."
            )
        if quantization_spec.scale_type is None:
            raise ValueError(
                f"The field `scale_type` cannot be None when Dtype is {quantization_spec.dtype.name} in QuantizationSpec. Please reconfigure the quantization settings accordingly."
            )
    if quantization_spec.dtype in [Dtype.float16, Dtype.bfloat16]:
        for field in fields(quantization_spec):
            quant_config_spec = getattr(quantization_spec, field.name)
            if (field.name not in ['observer_cls', 'dtype']) and \
                    (quant_config_spec is not None):
                logger.warning(
                    f"{field.name} will not be used when Dtype is {quantization_spec.dtype.name} in QuantizationSpec.")
    if quantization_spec.dtype in [Dtype.mx, Dtype.mx6, Dtype.mx9]:
        if quantization_spec.dtype == Dtype.mx and quantization_spec.mx_element_dtype is None:
            raise ValueError(f"Element dtype cannot be none for Dtype: {quantization_spec.dtype.name}.")

        if quantization_spec.observer_cls is not qobs.PerBlockMXObserver:
            raise ValueError(f"MX datatypes must use the {qobs.PerBlockMXObserver} observer.")

        if quantization_spec.ch_axis is None:
            raise ValueError(f"The channel axis must be specified when using Dtype: {quantization_spec.dtype.name}.")

        if quantization_spec.group_size is None:
            raise ValueError(
                f"The group/block size must be specified when using Dtype: {quantization_spec.dtype.name}.")


# TODO: can be optimized
def init_quantization_config(quantization_config: QuantizationConfig) -> tuple[bool, bool, bool, bool]:
    is_dynamic = True  # TODO: remove it
    is_weight_only = True

    for field in fields(QuantizationConfig):
        quantization_spec = getattr(quantization_config, field.name)
        if isinstance(quantization_spec, (QuantizationSpec, list)):
            specs_to_check = [quantization_spec] if isinstance(quantization_spec,
                                                               QuantizationSpec) else quantization_spec
            for spec in specs_to_check:
                if not isinstance(spec, QuantizationSpec):
                    continue
                if spec.dtype in [
                        Dtype.int8, Dtype.uint8, Dtype.int4, Dtype.uint4, Dtype.fp8_e4m3, Dtype.fp8_e5m2,
                        Dtype.fp6_e3m2, Dtype.fp6_e2m3, Dtype.mx, Dtype.mx6, Dtype.mx9, Dtype.fp4, Dtype.int2
                ] and spec.is_dynamic is False:
                    is_dynamic = False
                if field.name in ["input_tensors", "output_tensors"
                                  ] and spec.dtype not in [Dtype.float16, Dtype.bfloat16]:
                    is_weight_only = False

    if not is_weight_only:
        is_input_dynamic = True
        is_output_dynamic = True
        is_input_contain_scale_per_tensor = False
        is_output_contain_scale_per_tensor = False
        if quantization_config.input_tensors is not None:
            specs_to_check = [quantization_config.input_tensors] \
                if isinstance(quantization_config.input_tensors, QuantizationSpec) \
                else quantization_config.input_tensors
            is_input_dynamic = all(spec.is_dynamic for spec in specs_to_check if isinstance(spec, QuantizationSpec))
            if not isinstance(quantization_config.input_tensors, QuantizationSpec):
                is_input_contain_scale_per_tensor = any(spec.is_scale_quant and spec.qscheme == QSchemeType.per_tensor
                                                        for spec in specs_to_check
                                                        if isinstance(spec, QuantizationSpec))

        if quantization_config.output_tensors is not None:
            specs_to_check = [quantization_config.output_tensors] \
                if isinstance(quantization_config.output_tensors, QuantizationSpec) \
                else quantization_config.output_tensors
            is_output_dynamic = all(spec.is_dynamic for spec in specs_to_check if isinstance(spec, QuantizationSpec))
            if not isinstance(quantization_config.output_tensors, QuantizationSpec):
                is_output_contain_scale_per_tensor = any(spec.is_scale_quant and spec.qscheme == QSchemeType.per_tensor
                                                         for spec in specs_to_check
                                                         if isinstance(spec, QuantizationSpec))

        is_act_dynamic = is_input_dynamic and is_output_dynamic
        is_act_contain_scale_per_tensor = is_input_contain_scale_per_tensor or is_output_contain_scale_per_tensor
    else:
        is_act_dynamic = False
        is_act_contain_scale_per_tensor = False

    return is_dynamic, is_weight_only, is_act_dynamic, is_act_contain_scale_per_tensor


def check_and_adjust_quant_config(quantization_config: QuantizationConfig) -> QuantizationConfig:
    assert isinstance(quantization_config, QuantizationConfig), "Only support check on 'QuantizationConfig'"
    if quantization_config.input_tensors is not None and isinstance(quantization_config.input_tensors,
                                                                    QuantizationSpec):
        if quantization_config.input_tensors.qscheme == QSchemeType.per_group and quantization_config.input_tensors.group_size is not None and quantization_config.input_tensors.group_size > 0:
            if not quantization_config.input_tensors.is_dynamic:
                logger.warning(
                    "For input_tensors, quantization must be dynamic using per-group granularity, forcely set is_dynamic=True."
                )
                input_quant_config_copied = copy.deepcopy(quantization_config.input_tensors)
                input_quant_config_copied.is_dynamic = True
                quantization_config.input_tensors = input_quant_config_copied
                del input_quant_config_copied
    if quantization_config.output_tensors is not None and isinstance(quantization_config.output_tensors,
                                                                     QuantizationSpec):
        if quantization_config.output_tensors.qscheme == QSchemeType.per_group and quantization_config.output_tensors.group_size is not None and quantization_config.output_tensors.group_size > 0:
            if not quantization_config.output_tensors.is_dynamic:
                logger.warning(
                    "For output_tensors, quantization must be dynamic using per-group granularity, forcely set is_dynamic=True."
                )
                output_quant_config_copied = copy.deepcopy(quantization_config.output_tensors)
                output_quant_config_copied.is_dynamic = True
                quantization_config.output_tensors = output_quant_config_copied
                del output_quant_config_copied

    return quantization_config
