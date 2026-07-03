#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys

sys.path.append("..")
import quark.torch.kernel  # noqa
import torch
import torch.nn as nn
import quark.torch.quantization.nn.modules.quantize_conv_bn_fused as conv_bn_fused
from quark.torch.quantization.config.config import QTensorConfig, QLayerConfig, QConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType, QuantizationMode
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
from quark.torch.quantization.nn.modules.quantize_conv import QuantConvTranspose2d
from quark.torch import ModelQuantizer
from quark.common.utils.import_utils import export_for_training
from quark.common.utils.testing_utils import torch_device

INT8_PER_TENSOR_SPEC = QTensorConfig(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)


def test_QuantizedConvBatchNorm2d():
    torch.cuda.empty_cache()
    conv_map = {
        torch.nn.Conv2d: (torch.nn.BatchNorm2d, (1, 3, 16, 16), conv_bn_fused.QuantizedConvBatchNorm2d),
        torch.nn.ConvTranspose2d: (torch.nn.BatchNorm2d, (1, 3, 28, 28), conv_bn_fused.QuantConvTransposeBatchNorm2d),
        # TODO haoliang
        # torch.nn.Conv3d: (torch.nn.BatchNorm3d, (1, 3, 16, 16, 16), conv_bn_fused.QuantizedConvBatchNorm3d),
        # torch.nn.ConvTranspose3d: (torch.nn.BatchNorm3d, (1, 3, 16, 16, 16), conv_bn_fused.QuantizedConvTransposeBatchNorm3d),
    }
    empty_config = QLayerConfig()
    for conv, (bn, input_size, q_conv) in conv_map.items():
        input = torch.ones(input_size).to(torch_device)
        float_conv = conv(in_channels=3, out_channels=16, kernel_size=3, stride=1, bias=False).to(torch_device)
        float_bn = bn(16).to(torch_device)
        quantized_conv = q_conv.from_float(float_conv, float_bn, empty_config).to(torch_device)
        float_out = float_bn.eval()(float_conv(input))
        quantized_out = quantized_conv.eval()(input)
        assert torch.allclose(float_out.mean(), quantized_out.mean()), (
            f"{float_conv.__class__.__name__} vs {quantized_conv.__class__.__name__} diffs in mean"
        )
        assert torch.allclose(float_out.std(), quantized_out.std()), (
            f"{float_conv.__class__.__name__} vs {quantized_conv.__class__.__name__} diffs in std"
        )

    # test forward
    quant_config = QLayerConfig(weight=INT8_PER_TENSOR_SPEC, bias=INT8_PER_TENSOR_SPEC)
    for conv, (bn, input_size, q_conv) in conv_map.items():
        input = torch.randn(input_size).to(torch_device)
        float_conv = conv(in_channels=3, out_channels=16, kernel_size=3, stride=1, bias=True).to(torch_device)
        float_bn = bn(16).to(torch_device)
        quantized_conv = q_conv.from_float(float_conv, float_bn, quant_config)
        quantized_out = quantized_conv(input)
    torch.cuda.empty_cache()


# =======================test QuantConvTranspose2d function =======================


def test_transpose_conv():
    torch.cuda.empty_cache()
    transposeconv = torch.nn.ConvTranspose2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1).to(
        torch_device
    )
    quant_config = QLayerConfig()
    quant_transpose_conv = QuantConvTranspose2d.from_float(
        transposeconv, quant_config, True, transposeconv.weight, transposeconv.bias
    )

    dummy_input = torch.randn(1, 3, 112, 112).to(torch_device)
    output_1 = transposeconv(dummy_input).to(torch_device)
    output_2 = quant_transpose_conv(dummy_input).to(torch_device)
    assert torch.allclose(output_1, output_2, atol=1e-5), "ConvTranspose2d diff with QuantConvTranspose2d"
    print(f"Finish Test: {torch.nn.ConvTranspose2d.__name__} equal to {QuantConvTranspose2d.__name__}")
    torch.cuda.empty_cache()


# =======================test QuantConvTranspose2d eager&fx mode quant =======================


def test_transpose_model_quant():
    class SimpleCNNWithTransposeConv(nn.Module):
        def __init__(self, num_classes=10):
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(32)
            self.relu1 = nn.ReLU()
            self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, bias=False)
            self.convtrans1 = nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3)
            self.relu2 = nn.ReLU()
            self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(in_features=32, out_features=num_classes)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu1(x)
            x = self.conv2(x)
            x = self.convtrans1(x)
            x = self.relu2(x)
            x = self.adaptive_avg_pool2d(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

    torch.cuda.empty_cache()
    batch_shape = [4, 3, 56, 56]
    example_inputs = torch.rand(batch_shape).to(torch_device)
    quant_config = QLayerConfig(input_tensors=None, output_tensors=None, weight=None, bias=None)
    # eager mode quant
    eager_quant_conf = QConfig(global_quant_config=quant_config, quant_mode=QuantizationMode.eager_mode)
    quantizer = ModelQuantizer(eager_quant_conf)
    model = SimpleCNNWithTransposeConv().eval().to(torch_device)
    quantized_model = quantizer.quantize_model(model, [example_inputs for _ in range(2)])
    org_model_out = model(example_inputs)
    quant_model_out = quantized_model(example_inputs)
    assert torch.allclose(org_model_out, quant_model_out), (
        "have diff init QuantConvTranspose2d by `from_float` func from ConvTranspose2d"
    )
    print("Finish test: SimpleCNNWithTransposeConv model eager mode quant")

    # fx model quant without quant
    float_model = SimpleCNNWithTransposeConv().to(torch_device).eval()
    float_out = float_model(example_inputs)
    graph_model = export_for_training(float_model, (example_inputs,)).module()
    fx_quant_conf = QConfig(global_quant_config=quant_config, quant_mode=QuantizationMode.fx_graph_mode)
    quantizer = ModelQuantizer(fx_quant_conf)
    quantized_model = quantizer.quantize_model(graph_model, [example_inputs for _ in range(2)])
    quant_out = quantized_model(example_inputs)
    assert torch.allclose(float_out, quant_out, atol=1e-3), (
        "On the condition no quant, FP32 model's output should be same with FX model's output"
    )
    print("Finish test: SimpleCNNWithTransposeConv model FX mode quant(no quant)")

    # fx model quant with int8 quant
    int8_quant_config = QLayerConfig(
        weight=INT8_PER_TENSOR_SPEC,
        bias=INT8_PER_TENSOR_SPEC,
        output_tensors=INT8_PER_TENSOR_SPEC,
        input_tensors=INT8_PER_TENSOR_SPEC,
    )
    float_model = SimpleCNNWithTransposeConv().to(torch_device).eval()
    graph_model = export_for_training(float_model, (example_inputs,)).module()
    fx_quant_conf = QConfig(global_quant_config=int8_quant_config, quant_mode=QuantizationMode.fx_graph_mode)
    quantizer = ModelQuantizer(fx_quant_conf)
    quantized_model = quantizer.quantize_model(graph_model, [example_inputs for _ in range(2)])
    quant_out = quantized_model(example_inputs)
    print("Finish test: SimpleCNNWithTransposeConv model FX mode quant(int8 quant)")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    test_QuantizedConvBatchNorm2d()
    test_transpose_conv()
    test_transpose_model_quant()
    torch.cuda.empty_cache()
