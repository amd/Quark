#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from unittest.mock import MagicMock, patch
import torch
import torch.nn as nn

from quark.torch.quantization.config.config import Config, RotationConfig, SmoothQuantConfig, AWQConfig, QuantizationConfig, Float16Spec
from quark.torch.algorithm.utils.auto_config import dump_config_to_json
from quark.torch.algorithm.api import add_algorithm_config_by_model

FLOAT16_SPEC = Float16Spec().to_quantization_spec()
DEFAULT_CONFIG = QuantizationConfig(weight=FLOAT16_SPEC)

@patch("os.makedirs")
@patch("builtins.open")
@patch("json.dump")
def test_smoke_dump_config_to_json(mock_json_dump, mock_open, mock_makedirs):
    # TODO: make it a unit test rather than smoke test
    # Arrange
    model = MagicMock(spec=nn.Module)
    model.__class__.__name__ = "TestModel"
    file_name = "test_file"
    data_dict = {"key": "value"}

    # Act
    dump_config_to_json(model, file_name, data_dict)

def test_smoke_enhance_algorithm_config():
    # TODO: make it a unit test rather than smoke test
    # Arrange
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-0.5B")
    config = Config(global_quant_config=DEFAULT_CONFIG, pre_quant_opt_config=[RotationConfig(), SmoothQuantConfig()], algo_config=AWQConfig())
    dataloader = torch.utils.data.DataLoader(torch.tensor([[1, 2, 3, 4]]))
    updated_config = add_algorithm_config_by_model(model, dataloader, config)
