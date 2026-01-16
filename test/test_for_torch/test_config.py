#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
import torch.nn as nn

from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver


def test_reload_config():
    quantization_spec = QTensorConfig(
        dtype=Dtype.int8,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
        qscheme=QSchemeType.per_tensor,
        ch_axis=None,
        group_size=None,
        symmetric=False,
        round_method=RoundType.round,
        scale_type=ScaleType.float,
    )
    quantization_config = QLayerConfig(weight=quantization_spec)
    config = QConfig(global_quant_config=quantization_config, layer_type_quant_config={nn.Linear: quantization_config})

    config_dict = config.to_dict()

    config_reloaded = QConfig.from_dict(config_dict)

    assert config == config_reloaded

    quantization_spec = QTensorConfig(
        dtype=Dtype.int8,
        observer_cls=PerTensorMinMaxObserver,
        is_dynamic=False,
        qscheme=QSchemeType.per_tensor,
        ch_axis=None,
        group_size=None,
        symmetric=False,
        round_method=RoundType.round,
        scale_type=ScaleType.float,
    )
    quantization_config = QLayerConfig(weight=quantization_spec)
    config = QConfig(
        global_quant_config=quantization_config, layer_type_quant_config={nn.LayerNorm: quantization_config}
    )

    config_dict = config.to_dict()

    with pytest.raises(Exception) as e_info:
        _ = QConfig.from_dict(config_dict)
    assert "from a dictionary using custom `layer_type_quantization_config`" in str(e_info.value)
