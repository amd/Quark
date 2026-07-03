#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn

from quark.torch.algorithm.svdquant.svdquant import ErrorCorrectedModule, LowRankCorrectionModule
from quark.torch.export.config.config import JsonExporterConfig
from quark.torch.export.json_export.builder.native_model_info_builder import NativeModelInfoBuilder


def test_native_model_info_builder_buffer_export():
    layer = nn.Linear(8, 8, bias=False)
    correction = LowRankCorrectionModule(8, 8, rank=2)
    sf = torch.ones(8) * 2.0
    ecm = ErrorCorrectedModule(correction, layer, smooth_factor=sf)

    class WrapperModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.ecm = ecm

    model = WrapperModel()
    config = JsonExporterConfig()
    builder = NativeModelInfoBuilder(model, config)
    param_dict: dict[str, torch.Tensor] = {}
    builder.build_model_info(param_dict, compressed=False, reorder=False)

    assert "ecm.smooth_factor" in param_dict
    assert torch.allclose(param_dict["ecm.smooth_factor"], sf)
