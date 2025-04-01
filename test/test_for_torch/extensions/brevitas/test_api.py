#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest

import os.path

import torch
import torch.nn as nn
import brevitas.nn as qnn

import quark.torch.extensions.brevitas.api as brevitas_api
import quark.torch.extensions.brevitas.config as brevitas_config
import quark.torch.extensions.brevitas.algos as brevitas_algos

class SimpleNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1, bias=False)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.fc1 = nn.Linear(8 * 8, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

backends = [
    brevitas_config.Backend.layerwise
]

@pytest.mark.parametrize("backend", backends)
def test_basic_quantization_no_calibration_data(backend):
    model = SimpleNetwork().to('cpu')

    weight_spec = brevitas_config.QuantizationSpec()
    global_config = brevitas_config.QuantizationConfig(weight=weight_spec)
    config = brevitas_config.Config(global_quant_config=global_config, backend=backend)

    quantizer = brevitas_api.ModelQuantizer(config)
    quant_model = quantizer.quantize_model(model, None)

    assert quant_model is not None

    if backend == brevitas_config.Backend.layerwise:
        for idx, module in enumerate(quant_model.children()):
            if idx == 0 or idx == 2:
                assert isinstance(module, qnn.QuantConv2d)
            elif idx == 3 or idx == 4:
                assert isinstance(module, qnn.QuantLinear)
    else:
        # add more cases here as backend support is extended
        pass

def test_missing_calibration_data():
    model = SimpleNetwork().to('cpu')
    input_spec = brevitas_config.QuantizationSpec()
    weight_spec = brevitas_config.QuantizationSpec()
    global_config = brevitas_config.QuantizationConfig(input_tensors=input_spec, weight=weight_spec)
    config = brevitas_config.Config(global_quant_config=global_config,
                                    # this algorithm needs calibration data
                                    algo_config=[brevitas_algos.GPTQ()])

    quantizer = brevitas_api.ModelQuantizer(config)
    with pytest.raises(ValueError):
        _ = quantizer.quantize_model(model, None)

def test_parse_none():
    assert brevitas_api.ModelQuantizer._parse_activation_quant_spec(None) is None
    assert brevitas_api.ModelQuantizer._parse_bias_quant_spec(None) is None
    assert brevitas_api.ModelQuantizer._parse_weight_quant_spec(None) is None

def test_export_path():
    path = "test.onnx"
    exporter = brevitas_api.ModelExporter(path)
    assert exporter.export_path == path

def test_basic_export_no_error():
    model = SimpleNetwork().to('cpu')

    weight_spec = brevitas_config.QuantizationSpec()
    global_config = brevitas_config.QuantizationConfig(weight=weight_spec)
    config = brevitas_config.Config(global_quant_config=global_config)

    quantizer = brevitas_api.ModelQuantizer(config)
    quant_model = quantizer.quantize_model(model, None)

    exporter = brevitas_api.ModelExporter("test.onnx")
    exporter.export_onnx_model(quant_model, torch.ones(1, 3, 32, 32))

    assert os.path.isfile("test.onnx"), "Exported file could not be found!"

valid_bias_bit_widths = [
    8, 16, 24, 32
]

@pytest.mark.parametrize("bit_width", valid_bias_bit_widths)
def test_parse_bias_spec(bit_width):
    bias_spec = brevitas_config.QuantizationSpec(bit_width=bit_width)

    assert brevitas_api.ModelQuantizer._parse_bias_quant_spec(bias_spec) is not None, "Bias quant spect wasn't parsed correctly."

def test_parse_invaid_bias_spec():
    bias_spec = brevitas_config.QuantizationSpec(bit_width=7)
    with pytest.raises(KeyError):
        brevitas_api.ModelQuantizer._parse_bias_quant_spec(bias_spec)

def parse_activation_float_invalid():
    activation_spec = brevitas_config.QuantizationSpec(bit_width=8,
                                                       quant_type=brevitas_config.QuantType.float_quant,
                                                       exponent_bit_width=4,
                                                       mantissa_bit_width=3,
                                                       symmetric=False)
    with pytest.raises(KeyError):
        brevitas_api.ModelQuantizer._parse_activation_quant_spec(activation_spec)

def parse_activation_float_bit_widths():
    activation_spec = brevitas_config.QuantizationSpec(bit_width=8,
                                                       quant_type=brevitas_config.QuantType.float_quant,
                                                       exponent_bit_width=4,
                                                       mantissa_bit_width=3)
    quant = brevitas_api.ModelQuantizer._parse_activation_quant_spec(activation_spec)

    assert quant.high_percentile_q is not None
    assert quant.bit_width == 8
    assert quant.exponent_bit_width == 4
    assert quant.mantissa_bit_width == 3

def parse_activation_non_symmetric():
    activation_spec = brevitas_config.QuantizationSpec(bit_width=8,
                                                       quant_type=brevitas_config.QuantType.float_quant,
                                                       exponent_bit_width=4,
                                                       mantissa_bit_width=3)
    quant = brevitas_api.ModelQuantizer._parse_activation_quant_spec(activation_spec)

    assert quant.high_percentile_q is not None
    assert quant.low_percentile_q is not None
    assert quant.bit_width == 8

def parse_weight_float_bit_widths():
    weight_spec = brevitas_config.QuantizationSpec(bit_width=8,
                                                   quant_type=brevitas_config.QuantType.int_quant,
                                                   exponent_bit_width=4,
                                                   mantissa_bit_width=3,
                                                   symmetric=False)
    quant = brevitas_api.ModelQuantizer._parse_weight_quant_spec(weight_spec)

    assert quant.bit_width == 8
    assert quant.exponent_bit_width == 4
    assert quant.mantissa_bit_width == 3
    assert quant.zero_point_shape is not None
