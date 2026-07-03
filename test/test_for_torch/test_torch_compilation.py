#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy

import pytest
import torch
import torch.nn as nn
from packaging import version
from torch._dynamo.utils import get_metrics_context
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.common.utils.testing_utils import PatchEverywhere, torch_device
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import (
    FP8E4M3PerTensorSpec,
    FP8E5M2PerTensorSpec,
    Int4PerTensorSpec,
    Int8PerChannelSpec,
    OCP_MXFP4Spec,
    QConfig,
    QLayerConfig,
    QTensorConfig,
)
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import (
    PerBlockMXObserver,
    PerChannelMinMaxObserver,
    PerGroupMinMaxObserver,
    PerTensorMinMaxObserver,
)

TEST_DEVICES = ["cpu"] if torch.device(torch_device).type == "cpu" else ["cpu", torch_device]


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 100)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)

        x = torch.nn.functional.softmax(x, dim=-1)

        return x


def quantize_and_compile_model(device, weight_spec: QTensorConfig, activation_spec: QTensorConfig):
    device = torch.device(device)
    is_fp8 = weight_spec.dtype in [Dtype.fp8_e4m3, Dtype.fp8_e5m2] or weight_spec.mx_element_dtype in [
        Dtype.fp8_e4m3,
        Dtype.fp8_e5m2,
    ]

    model = SimpleModel().to(device)
    input_data = torch.randn(10, 512).to(device)
    calib_dataloader = DataLoader(input_data, batch_size=10, shuffle=True)

    global_quant_config = QLayerConfig(weight=weight_spec, input_tensors=activation_spec)
    quant_config = QConfig(global_quant_config=global_quant_config)

    quantizer = ModelQuantizer(quant_config)
    quant_model = quantizer.quantize_model(model, calib_dataloader)

    input_data = torch.randn(20, 512).to(device)

    with torch.no_grad():
        output1 = quant_model(input_data)

        # Freeze quantized model
        frozen_quant_model = quantizer.freeze(quant_model)

        output2 = frozen_quant_model(input_data)

    # Test the result is same after freeze
    assert torch.equal(output1, output2)

    with PatchEverywhere("QUARK_DISABLE_COMPILE", True, module_name_prefix="quark"), torch.no_grad():
        output2_nocompile = frozen_quant_model(input_data)

    if not is_fp8 or (version.parse(torch.__version__) >= version.parse("2.9") or device.type == "cpu"):
        # NOTE: There is some non-determinism introduced by torch.compile on float8 for torch<=2.8 (at least on MI300, did not test on H100).
        # On MI300 + torch==2.9.0+rocm6.4, the output is deterministic.
        assert torch.equal(output2, output2_nocompile)
    else:
        absdiff = (output2 - output2_nocompile).abs()
        reldiff = absdiff / (output2.abs() + 1e-3)

        assert reldiff.max() < 2e-1
        assert reldiff.mean() < 5e-3

    def custom_backend(gm, example_inputs):
        print("Model graph with custom_backend:")
        print(gm.graph)

        return gm.forward

    for backend in ["inductor", custom_backend]:
        frozen_quant_model = copy.deepcopy(frozen_quant_model)

        frozen_quant_model = torch.compile(frozen_quant_model, backend=backend)

        with torch.no_grad():
            output2 = frozen_quant_model(input_data)

        absdiff = (output2 - output2_nocompile).abs()
        reldiff = absdiff / (output2.abs() + 1e-3)

        if not is_fp8 or (version.parse(torch.__version__) >= version.parse("2.9") or device.type == "cpu"):
            # See the note above for this controlflow.
            assert torch.allclose(output2_nocompile, output2, atol=1e-4, rtol=1e-4)
        else:
            assert reldiff.max() < 2e-1
            assert reldiff.mean() < 1e-2


@pytest.mark.parametrize("device", [pytest.param(val, id=f"device:{val}") for val in TEST_DEVICES])
def test_int_per_tensor_for_torch_compile(device: str | torch.device):
    weight_spec = QTensorConfig(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )

    quantize_and_compile_model(device, weight_spec=weight_spec, activation_spec=weight_spec)


@pytest.mark.parametrize("device", [pytest.param(val, id=f"device:{val}") for val in TEST_DEVICES])
def test_int_per_channel_for_torch_compile(device):
    weight_spec = QTensorConfig(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_channel,
        observer_cls=PerChannelMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        ch_axis=1,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    quantize_and_compile_model(device, weight_spec=weight_spec, activation_spec=weight_spec)


@pytest.mark.parametrize("device", [pytest.param(val, id=f"device:{val}") for val in TEST_DEVICES])
def test_int_per_group_for_torch_compile(device):
    weight_spec = QTensorConfig(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_group,
        observer_cls=PerGroupMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        ch_axis=1,
        group_size=2,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )

    activation_spec = copy.deepcopy(weight_spec)
    activation_spec.is_dynamic = True

    quantize_and_compile_model(device, weight_spec=weight_spec, activation_spec=activation_spec)


@pytest.mark.parametrize(
    "dtype",
    [
        (Dtype.fp8_e4m3),
        (Dtype.fp8_e5m2),
    ],
)
@pytest.mark.parametrize("device", [pytest.param(val, id=f"device:{val}") for val in TEST_DEVICES])
def test_fp8_per_tensor_for_torch_compile(dtype, device):
    weight_spec = QTensorConfig(
        dtype=dtype,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    quantize_and_compile_model(device, weight_spec=weight_spec, activation_spec=weight_spec)


@pytest.mark.parametrize(
    "dtype,element_dtype",
    [
        (Dtype.mx6, None),
        (Dtype.mx9, None),
        (Dtype.mx, Dtype.int8),
        (Dtype.mx, Dtype.fp8_e4m3),
        (Dtype.mx, Dtype.fp8_e5m2),
        (Dtype.mx, Dtype.fp6_e2m3),
        (Dtype.mx, Dtype.fp6_e3m2),
        (Dtype.mx, Dtype.fp4),
    ],
)
@pytest.mark.parametrize("device", [pytest.param(val, id=f"device:{val}") for val in TEST_DEVICES])
def test_mx_for_torch_compile(dtype, element_dtype, device):
    weight_spec = QTensorConfig(
        dtype=dtype,
        mx_element_dtype=element_dtype,
        qscheme=QSchemeType.per_group,
        observer_cls=PerBlockMXObserver,
        ch_axis=-1,
        group_size=16,
        round_method=RoundType.half_even,
        is_dynamic=False,
        scale_calculation_mode="floor",
    )

    activation_spec = copy.deepcopy(weight_spec)
    activation_spec.is_dynamic = True

    quantize_and_compile_model(device, weight_spec=weight_spec, activation_spec=activation_spec)


def test_recompilations():
    torch.compiler.reset()
    torch._dynamo.reset()

    torch._logging.set_logs(recompiles=True)

    weight_specs = [
        FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec(),
        FP8E5M2PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec(),
        Int4PerTensorSpec(
            observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
        ).to_quantization_spec(),
        Int8PerChannelSpec(
            symmetric=True, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
        ).to_quantization_spec(),
        OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False, scale_calculation_mode="even").to_quantization_spec(),
    ]

    act_specs = [
        FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=True).to_quantization_spec(),
        FP8E5M2PerTensorSpec(observer_method="min_max", is_dynamic=True).to_quantization_spec(),
        Int8PerChannelSpec(
            symmetric=True, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=True
        ).to_quantization_spec(),
        Int4PerTensorSpec(
            observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=True
        ).to_quantization_spec(),
        OCP_MXFP4Spec(ch_axis=-1, is_dynamic=True, scale_calculation_mode="even").to_quantization_spec(),
    ]

    model_id = "trl-internal-testing/tiny-random-LlamaForCausalLM"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for weight_spec in weight_specs:
        for activation_spec in act_specs:
            print(f"\n\n\n\n----- weight_spec: {weight_spec}, activation_spec: {activation_spec}")
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto")
            model = model.to(torch_device)

            global_quant_config = QLayerConfig(weight=weight_spec, input_tensors=activation_spec)
            quant_config = QConfig(global_quant_config=global_quant_config, exclude=["lm_head"])

            quantizer = ModelQuantizer(quant_config)
            model = quantizer.quantize_model(model)

            model = model.eval()

            prompt = "this is me here"
            with torch.no_grad():
                for seqlen in [50, 1, 2, 3, 4, 8, 12, 5, 100]:
                    for bs in [1, 2, 3, 4, 16, 5, 32]:
                        inp = tokenizer([prompt * seqlen] * bs, return_tensors="pt").to(torch_device)

                        assert inp["input_ids"].shape[0] == bs

                        with torch.no_grad():
                            _ = model(**inp)

    metrics = get_metrics_context()._metrics

    print("dynamo metrics", metrics)

    recompilation_count = metrics["cache_size"]

    # TODO: locally we get strictly `recompilation_count == 18`, however in the CI we get 25.
    # Clarify why. For some reason torch.compile recompilations are not logged in the CI.
    assert recompilation_count <= 49
