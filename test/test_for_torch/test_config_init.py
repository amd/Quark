#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
from quark.torch.quantization.config.type import Dtype, QSchemeType
from quark.torch.quantization.observer.observer import PerBlockMXObserver
from quark.torch.quantization.config.config import Config, QuantizationSpec, QuantizationConfig
from quark.torch import ModelQuantizer


@pytest.mark.parametrize("activation,dtype,is_dynamic", [
    ("input_tensors", Dtype.fp4, True),
    ("output_tensors", Dtype.fp4, True),
    ("input_tensors", Dtype.fp4, False),
    ("output_tensors", Dtype.fp4, False),
    ("input_tensors", Dtype.fp8_e4m3, True),
    ("output_tensors", Dtype.fp8_e4m3, True),
    ("input_tensors", Dtype.fp8_e4m3, False),
    ("output_tensors", Dtype.fp8_e4m3, False),
    ("input_tensors", Dtype.fp8_e5m2, True),
    ("output_tensors", Dtype.fp8_e5m2, True),
    ("input_tensors", Dtype.fp8_e5m2, False),
    ("output_tensors", Dtype.fp8_e5m2, False),
])
def test_activation_fp_per_group_config_auto_check_and_adjust(activation, dtype, is_dynamic):

    FP_PER_GROUP_SPEC = QuantizationSpec(dtype=dtype,
                                         qscheme=QSchemeType.per_group,
                                         observer_cls=PerBlockMXObserver,
                                         ch_axis=-1,
                                         group_size=32,
                                         is_dynamic=is_dynamic)
    if activation == "input_tensors":
        ACTIVATION_CONFIG = QuantizationConfig(input_tensors=FP_PER_GROUP_SPEC)
    if activation == "output_tensors":
        ACTIVATION_CONFIG = QuantizationConfig(output_tensors=FP_PER_GROUP_SPEC)

    quant_config = Config(global_quant_config=ACTIVATION_CONFIG)
    quantizer = ModelQuantizer(quant_config)
    quantizer.init_config()

    if activation == "input_tensors":
        assert quantizer.config.global_quant_config.input_tensors.is_dynamic
    if activation == "output_tensors":
        assert quantizer.config.global_quant_config.output_tensors.is_dynamic
