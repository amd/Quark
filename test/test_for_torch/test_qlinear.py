#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from quark.common.utils.import_utils import export_for_training
from quark.common.utils.testing_utils import torch_device
from quark.torch import ModelQuantizer
from quark.torch.quantization import AutoSmoothQuantConfig, Int8PerTensorSpec, QConfig, QLayerConfig
from quark.torch.quantization.config.config import QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, QuantizationMode, RoundType, ScaleType
from quark.torch.quantization.nn.modules.quantize_linear import QLoRaQuantLinear, QuantLinear
from quark.torch.quantization.observer.observer import PerTensorPowOf2MinMaxObserver
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase

INT8_PER_TENSOR_SPEC = Int8PerTensorSpec(is_dynamic=False).to_quantization_spec()

DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG = QLayerConfig(
    input_tensors=INT8_PER_TENSOR_SPEC,
    weight=INT8_PER_TENSOR_SPEC,
    bias=INT8_PER_TENSOR_SPEC,
    output_tensors=INT8_PER_TENSOR_SPEC,
)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


input_tensor = torch.randn(1, 64, 64)


def test_net():
    class MyDataset(Dataset):
        def __init__(self):
            return

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return input_tensor

    model = SimpleCNN(num_classes=10)
    dataset = MyDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    quant_config = QConfig(global_quant_config=DEFAULT_W_INT8_A_INT8_PER_TENSOR_CONFIG)
    quantizer = ModelQuantizer(quant_config)
    quant_model = quantizer.quantize_model(model, dataloader)

    assert (
        hasattr(quant_model.fc, "_input_quantizer")
        and hasattr(quant_model.fc, "_weight_quantizer")
        and hasattr(quant_model.fc, "_bias_quantizer")
        and hasattr(quant_model.fc, "_output_quantizer")
    )


def test_quantlinear_eq_qloraquantlinear():
    empty_quant_config = QLayerConfig()
    in_feature = 128
    out_feature = 256
    qlinear = (
        QuantLinear(
            in_feature,
            out_feature,
            device=torch_device,
            bias=True,
            quant_config=empty_quant_config,
        )
        .to(torch_device)
        .to(torch.bfloat16)
    )
    qlinear.weight.data = torch.ones_like(qlinear.weight.data)

    qloralinear = (
        QLoRaQuantLinear(
            in_feature,
            out_feature,
            device=torch_device,
            bias=True,
            quant_config=empty_quant_config,
        )
        .to(torch_device)
        .to(torch.bfloat16)
    )

    qloralinear.weight.data = qlinear.weight.data
    qloralinear.bias.data = qlinear.bias.data

    example_inputs = torch.ones(1, 128).to(torch_device).to(torch.bfloat16)

    qlinear(example_inputs)
    qloralinear(example_inputs)
    qloralinear.active_adapters = True
    qloralinear(example_inputs)


@pytest.mark.parametrize("algo", [pytest.param(val, id=f"algo:{val}") for val in ["autosmoothquant", None]])
def test_quantizer_enabled_disabled(monkeypatch, algo: None | str):
    config = AutoConfig.from_pretrained("trl-internal-testing/tiny-random-LlamaForCausalLM")
    model = AutoModelForCausalLM.from_config(config)

    int8_act_spec = Int8PerTensorSpec(is_dynamic=True).to_quantization_spec()
    int8_weight_spec = Int8PerTensorSpec(is_dynamic=False).to_quantization_spec()

    layer_config = QLayerConfig(
        input_tensors=int8_act_spec,
        weight=int8_weight_spec,
    )

    if algo == "autosmoothquant":
        algo_config = [AutoSmoothQuantConfig(model_decoder_layers="model.layers", scaling_layers=[])]
    else:
        algo_config = []

    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained("trl-internal-testing/tiny-random-LlamaForCausalLM")
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"])

    original_method = ModelQuantizer._do_calibration

    def patched_method(self, model, dataloader):
        model = original_method(self, model, dataloader)
        for name, module in model.named_modules():
            if isinstance(module, FakeQuantizeBase):
                assert module.is_fake_quant_enabled

                if "_weight_quantizer" not in name:
                    assert module.is_dynamic
                    assert module.observer_enabled

        return model

    monkeypatch.setattr(ModelQuantizer, "_do_calibration", patched_method)

    quant_config = QConfig(global_quant_config=layer_config, algo_config=algo_config)
    quantizer = ModelQuantizer(quant_config)
    _ = quantizer.quantize_model(model, calib_dataloader)


# =============================================================================
# Tests for QuantLinear pre-quantized model support
# =============================================================================


class _FakeQuantizationStatus:
    """Mimics compressed_tensors QuantizationStatus enum for testing."""

    value = "compressed"


class FakeCompressedLinear(nn.Linear):
    """Minimal fake compressed-tensors module for unit-testing QuantLinear.from_prequantized.

    Mimics the >=0.15 compressed-tensors API where compressed modules are plain
    ``nn.Linear`` with ``quantization_status == COMPRESSED``.
    """

    def __init__(self, in_features: int = 64, out_features: int = 32, bias: bool = False) -> None:
        super().__init__(in_features, out_features, bias=bias)
        self.weight = nn.Parameter(torch.randn(out_features, in_features).to(torch.float8_e4m3fn))
        self.weight_scale = torch.ones(out_features, 1)
        self.weight_zero_point = None
        self.weight_shape = None
        self.weight_g_idx = None
        self.quantization_status = _FakeQuantizationStatus()
        self.quantization_scheme = None


class TestQuantLinearPrequantized:
    """Tests for QuantLinear extensions supporting pre-quantized models."""

    def test_from_prequantized(self) -> None:
        """from_prequantized creates QuantLinear with correct attributes and bias handling."""
        from unittest.mock import patch

        from quark.torch.quantization.inverse_quantizer import CompressedLinearInverseQuantizer

        for has_bias in (False, True):
            fake = FakeCompressedLinear(64, 32, bias=has_bias)
            with patch(
                "quark.torch.quantization.inverse_quantizer.create_inverse_quantizer",
                return_value=CompressedLinearInverseQuantizer(fake),
            ):
                ql = QuantLinear.from_prequantized(fake, QLayerConfig())

            assert isinstance(ql, QuantLinear)
            assert ql.is_prequantized is True
            assert ql.in_features == 64 and ql.out_features == 32
            assert (ql.bias is not None) == has_bias

    def test_get_quant_weight_and_dequantized_weight(self) -> None:
        """get_quant_weight and get_dequantized_weight use inverse quantizer for pre-quantized;
        get_dequantized_weight returns self.weight for normal QuantLinear."""
        from unittest.mock import MagicMock, patch

        fake = FakeCompressedLinear(64, 32)
        mock_inv = MagicMock()
        mock_inv.dequantize.return_value = torch.randn(32, 64, dtype=torch.float32)

        with patch(
            "quark.torch.quantization.inverse_quantizer.create_inverse_quantizer",
            return_value=mock_inv,
        ):
            ql = QuantLinear.from_prequantized(fake, QLayerConfig())

        # Pre-quantized path
        result = ql.get_quant_weight(ql.weight)
        assert result.dtype in (torch.float16, torch.bfloat16, torch.float32)
        dequant = ql.get_dequantized_weight()
        assert dequant.shape == (32, 64)
        assert mock_inv.dequantize.call_count == 2

        # Normal QuantLinear path: get_dequantized_weight returns self.weight directly
        normal = QuantLinear(64, 32, device=torch.device("cpu"), bias=True, quant_config=QLayerConfig())
        assert normal.is_prequantized is False
        assert normal.get_dequantized_weight().data_ptr() == normal.weight.data_ptr()


# =============================================================================
# Tests for QuantLinear gradient flow (PTQ then backward / optimizer step)
# =============================================================================
# Same quant spec as test_fx_quant_align_hw_pow_of_2.py L67-82, using distinct names to avoid conflict with INT8_PER_TENSOR_SPEC above.
INT8_PER_TENSOR_POW2_SPEC = QTensorConfig(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorPowOf2MinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)
_global_quant_config = QLayerConfig(
    input_tensors=INT8_PER_TENSOR_POW2_SPEC,
    output_tensors=INT8_PER_TENSOR_POW2_SPEC,
    weight=INT8_PER_TENSOR_POW2_SPEC,
    bias=INT8_PER_TENSOR_POW2_SPEC,
)
_fx_mode_quant_config = QConfig(
    global_quant_config=_global_quant_config,
    quant_mode=QuantizationMode.fx_graph_mode,
)
_eager_mode_quant_config = QConfig(
    global_quant_config=_global_quant_config,
    quant_mode=QuantizationMode.eager_mode,
)


class TinyLinearModel(nn.Module):
    """Single Linear layer for PTQ + gradient flow tests."""

    def __init__(self, in_features: int = 4, out_features: int = 8):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def _get_quant_linear_after_ptq_fx(device: torch.device, in_features: int = 4, out_features: int = 8):
    """Build tiny model -> export to FX GraphModule -> PTQ (fx_graph_mode) -> return (model, QuantLinear)."""
    float_model = TinyLinearModel(in_features=in_features, out_features=out_features).to(device).eval()
    example_inputs = (torch.randn(4, in_features, device=device),)
    graph_model = export_for_training(float_model, example_inputs).module()
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)

    quantizer = ModelQuantizer(_fx_mode_quant_config)
    calib_data = [torch.randn(4, in_features, device=device) for _ in range(6)]
    quantized_model = quantizer.quantize_model(graph_model, calib_data)

    quant_linear = None
    for _, module in quantized_model.named_modules():
        if isinstance(module, QuantLinear):
            quant_linear = module
            break
    assert quant_linear is not None, "PTQ model (fx_graph_mode) should contain at least one QuantLinear"
    return quantized_model, quant_linear


def _get_quant_linear_after_ptq_eager(device: torch.device, in_features: int = 4, out_features: int = 8):
    """Build tiny model -> PTQ (eager_mode) -> return (model, QuantLinear)."""
    float_model = TinyLinearModel(in_features=in_features, out_features=out_features).to(device).eval()

    quantizer = ModelQuantizer(_eager_mode_quant_config)
    calib_data = [torch.randn(4, in_features, device=device) for _ in range(6)]
    quantized_model = quantizer.quantize_model(float_model, calib_data)

    quant_linear = None
    for _, module in quantized_model.named_modules():
        if isinstance(module, QuantLinear):
            quant_linear = module
            break
    assert quant_linear is not None, "PTQ model (eager_mode) should contain at least one QuantLinear"
    return quantized_model, quant_linear


def test_quant_linear_weight_receives_gradient_after_ptq_fx():
    """fx_graph_mode PTQ: QuantLinear with requires_grad=True must receive non-zero weight.grad."""
    device = torch_device
    quantized_model, quant_linear = _get_quant_linear_after_ptq_fx(device)
    in_features = 4

    quant_linear.weight.requires_grad = True
    if quant_linear.bias is not None:
        quant_linear.bias.requires_grad = True

    quantized_model.train()
    x = torch.randn(2, in_features, device=device)
    out = quantized_model(x)
    loss = out.sum()
    loss.backward()

    assert quant_linear.weight.grad is not None, (
        "QuantLinear.weight must receive gradient after backward (PTQ-calibrated model)"
    )
    assert quant_linear.weight.grad.abs().sum().item() > 0, (
        "QuantLinear.weight.grad must be non-zero (gradient flow was broken, e.g. by weight.data)"
    )
    if quant_linear.bias is not None:
        assert quant_linear.bias.grad is not None, "QuantLinear.bias must receive gradient after backward"
        assert quant_linear.bias.grad.abs().sum().item() > 0, "QuantLinear.bias.grad must be non-zero"


def test_quant_linear_weight_updates_with_optimizer_after_ptq_fx():
    """fx_graph_mode PTQ: optimizer.step() must update QuantLinear weight."""
    device = torch_device
    quantized_model, quant_linear = _get_quant_linear_after_ptq_fx(device)
    in_features = 4

    quant_linear.weight.requires_grad = True
    if quant_linear.bias is not None:
        quant_linear.bias.requires_grad = True

    optimizer = torch.optim.SGD(quant_linear.parameters(), lr=0.1)
    weight_before = quant_linear.weight.data.clone()

    quantized_model.train()
    x = torch.randn(2, in_features, device=device)
    out = quantized_model(x)
    loss = out.sum()
    loss.backward()
    optimizer.step()

    assert quant_linear.weight.grad is not None and quant_linear.weight.grad.abs().sum().item() > 0, (
        "Weight gradient must be non-zero for this test to be meaningful"
    )
    weight_after = quant_linear.weight.data
    assert not torch.equal(weight_before, weight_after), (
        "QuantLinear.weight must change after optimizer.step() (training is effective)"
    )


def test_quant_linear_weight_receives_gradient_after_ptq_eager():
    """Eager_mode PTQ: QuantLinear with requires_grad=True must receive non-zero weight.grad."""
    device = torch_device
    quantized_model, quant_linear = _get_quant_linear_after_ptq_eager(device)
    in_features = 4

    quant_linear.weight.requires_grad = True
    if quant_linear.bias is not None:
        quant_linear.bias.requires_grad = True

    quantized_model.train()
    x = torch.randn(2, in_features, device=device)
    out = quantized_model(x)
    loss = out.sum()
    loss.backward()

    assert quant_linear.weight.grad is not None, (
        "QuantLinear.weight must receive gradient after backward (eager_mode PTQ)"
    )
    assert quant_linear.weight.grad.abs().sum().item() > 0, "QuantLinear.weight.grad must be non-zero (eager_mode)"
    if quant_linear.bias is not None:
        assert quant_linear.bias.grad is not None, "QuantLinear.bias must receive gradient after backward"
        assert quant_linear.bias.grad.abs().sum().item() > 0, "QuantLinear.bias.grad must be non-zero"


def test_quant_linear_weight_updates_with_optimizer_after_ptq_eager():
    """Eager_mode PTQ: optimizer.step() must update QuantLinear weight."""
    device = torch_device
    quantized_model, quant_linear = _get_quant_linear_after_ptq_eager(device)
    in_features = 4

    quant_linear.weight.requires_grad = True
    if quant_linear.bias is not None:
        quant_linear.bias.requires_grad = True

    optimizer = torch.optim.SGD(quant_linear.parameters(), lr=0.1)
    weight_before = quant_linear.weight.data.clone()

    quantized_model.train()
    x = torch.randn(2, in_features, device=device)
    out = quantized_model(x)
    loss = out.sum()
    loss.backward()
    optimizer.step()

    assert quant_linear.weight.grad is not None and quant_linear.weight.grad.abs().sum().item() > 0, (
        "Weight gradient must be non-zero for this test to be meaningful (eager_mode)"
    )
    weight_after = quant_linear.weight.data
    assert not torch.equal(weight_before, weight_after), (
        "QuantLinear.weight must change after optimizer.step() (eager_mode)"
    )
