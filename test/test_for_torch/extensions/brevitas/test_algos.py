#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import quark.torch.extensions.brevitas.algos as brevitas_algos
import quark.torch.extensions.brevitas.api as brevitas_api
import quark.torch.extensions.brevitas.config as brevitas_config


def create_dataloader():
    class SimpleDataset(Dataset):
        def __init__(self):
            return

        def __len__(self):
            return 10

        def __getitem__(self, _):
            return torch.randn(1), torch.randn(1)

    return DataLoader(SimpleDataset())


def create_simple_model():
    class SimpleNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.seq = nn.Sequential(nn.Linear(1, 10), nn.ReLU(), nn.Linear(10, 1))

        def forward(self, x):
            return self.seq(x)

    return SimpleNetwork().to("cpu")


def create_quantized_model(calib_loader=None, input_quant=True):
    input_spec = brevitas_config.QuantizationSpec() if input_quant else None
    weight_spec = brevitas_config.QuantizationSpec()
    global_config = brevitas_config.QuantizationConfig(weight=weight_spec, input_tensors=input_spec)
    config = brevitas_config.Config(global_quant_config=global_config, pre_quant_opt_config=[])

    quantizer = brevitas_api.ModelQuantizer(config)
    return quantizer.quantize_model(create_simple_model(), calib_loader)


def test_pre_quant_default_apply():
    pre_quant_opt_config = brevitas_algos.PreQuantOptConfig()
    with pytest.raises(NotImplementedError):
        pre_quant_opt_config.apply(nn.Linear(20, 30), None)


def test_algo_config_default_apply():
    algo_config = brevitas_algos.AlgoConfig()
    with pytest.raises(NotImplementedError):
        algo_config.apply(nn.Linear(20, 30), None)


algos_requiring_calib = [
    brevitas_algos.ActivationEqualization(),
    brevitas_algos.GPFQ(),
    brevitas_algos.GPFA2Q(),
    brevitas_algos.GPTQ(),
    brevitas_algos.CalibrateBatchNorm(),
    brevitas_algos.BiasCorrection(),
]


@pytest.mark.parametrize("algo", algos_requiring_calib)
def test_algo_require_calib(algo):
    with pytest.raises(ValueError):
        algo.apply(nn.Linear(20, 30), None)


# add more algorithms here as more are exposed
algos = [
    brevitas_algos.GPFQ(),
    brevitas_algos.GPFA2Q(),
    brevitas_algos.GPTQ(),
    brevitas_algos.CalibrateBatchNorm(),
    brevitas_algos.BiasCorrection(),
]


@pytest.mark.parametrize("algo", algos)
def test_algos_no_exception(algo):
    quant_model = create_quantized_model()
    algo.apply(quant_model, create_dataloader())


pre_quant = [brevitas_algos.Preprocess(), brevitas_algos.ActivationEqualization()]


@pytest.mark.parametrize("pre_quant", pre_quant)
def test_pre_quant_no_exception(pre_quant):
    quant_model = create_simple_model()
    pre_quant.apply(quant_model, create_dataloader())


def test_calibrate():
    quant_model = create_quantized_model(None, False)
    assert brevitas_algos._calibrate(create_dataloader(), quant_model) is not None
