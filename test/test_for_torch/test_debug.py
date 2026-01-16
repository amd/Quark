#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import os
from unittest.mock import patch

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from quark.shares.utils.testing_utils import use_temporary_directory
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver


class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(10, 20)
        self.relu = nn.ReLU()
        self.lin2 = nn.Linear(20, 30)

    def forward(self, x):
        return self.lin2(self.relu(self.lin1(x)))


UINT8_PER_TENSOR_ASYM_SPEC = QTensorConfig(
    dtype=Dtype.uint8,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=False,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    qscheme=QSchemeType.per_tensor,
    is_dynamic=True,
)

REFERENCE_FILES = [
    "lin2.output_ref_histogram_absmean_ch0.png",
    "lin1.output_ref_histogram_absmean_ch0.png",
    "lin1.input_ref_histogram.png",
    "lin2.input_ref_histogram.png",
    "lin2.weight.png",
    "summary_ref_input_error.png",
    "summary_ref_output_error.png",
    "lin2.input_ref_histogram_absmean_ch1.png",
    "lin2.bias_stats.json",
    "lin2.weight_stats.json",
    "lin1.output_qdq_histogram.png",
    "lin1.input_ref_histogram_absmean_ch1.png",
    "lin2.output_histogram.png",
    "summary_io_quantization_error.png",
    "lin2.input_qdq_histogram.png",
    "lin1.weight.png",
    "lin1.bias_stats.json",
    "lin1.output_ref_histogram.png",
    "lin2.output_ref_histogram.png",
    "lin1.output_ref_histogram_absmean_ch1.png",
    "lin1.input_ref_histogram_absmean_ch0.png",
    "lin1.input_qdq_histogram.png",
    "lin2.input_histogram.png",
    "lin1.weight_stats.json",
    "lin2.output_qdq_histogram.png",
    "lin1.input_histogram.png",
    "summary_weight_error.png",
    "lin2.output_ref_histogram_absmean_ch1.png",
    "lin1.output_histogram.png",
    "lin2.input_ref_histogram_absmean_ch0.png",
]


# QUARK_DEBUG_ACT_HIST environment variable can not be used here, as `SAVE_ACTIVATIONS_HISTOGRAM` is a constant in quark.
@use_temporary_directory
@patch("quark.torch.quantization.debug.SAVE_ACTIVATIONS_HISTOGRAM", True)
def test_smoke_debug(tmpdir: str):
    global_quant_config = QLayerConfig(weight=UINT8_PER_TENSOR_ASYM_SPEC)
    config = QConfig(global_quant_config=global_quant_config)

    quantizer = ModelQuantizer(config)

    model = MyModel()
    model = model.to(torch.float16)

    os.environ["QUARK_DEBUG"] = tmpdir

    _ = quantizer.quantize_model(model)

    del os.environ["QUARK_DEBUG"]

    dir_content = os.listdir(tmpdir)
    for filename in [
        "lin2.weight.png",
        "lin2.weight_stats.json",
        "lin1.weight.png",
        "lin1.weight_stats.json",
        "summary_weight_error.png",
    ]:
        assert filename in dir_content


@use_temporary_directory
@patch("quark.torch.quantization.debug.SAVE_ACTIVATIONS_HISTOGRAM", True)
def test_smoke_debug_all(tmpdir: str):
    quant_spec = copy.deepcopy(UINT8_PER_TENSOR_ASYM_SPEC)
    quant_spec.is_dynamic = False

    dataloader = DataLoader([torch.rand(10, dtype=torch.float16), torch.rand(10, dtype=torch.float16)])

    global_quant_config = QLayerConfig(
        weight=quant_spec, input_tensors=quant_spec, bias=quant_spec, output_tensors=quant_spec
    )
    config = QConfig(global_quant_config=global_quant_config)

    quantizer = ModelQuantizer(config)

    model = MyModel()
    model = model.to(torch.float16)

    os.environ["QUARK_DEBUG"] = tmpdir

    _ = quantizer.quantize_model(model, dataloader=dataloader)

    del os.environ["QUARK_DEBUG"]

    dir_content = os.listdir(tmpdir)
    for filename in REFERENCE_FILES:
        assert filename in dir_content
