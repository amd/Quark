#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
sys.path.append("..")
import torch
import torch.nn as nn

from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QuantizationSpec, QuantizationConfig, Config
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType, QuantizationMode
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
import quark.torch.kernel  # noqa
from torch._export import capture_pre_autograd_graph
from quark.shares.utils.testing_utils import torch_device

from quark.shares.utils.testing_utils import use_temporary_directory
from pathlib import Path

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu1 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.relu2 = nn.ReLU(inplace=True)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu3(out)
        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation,
                  norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def _resnet(arch, block, layers, pretrained):
    model = ResNet(block, layers)
    if pretrained is not None:
        model.load_state_dict(torch.load(pretrained))
    return model


def resnet18(pretrained):
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained)


def test_replace_convbn_to_qt_convnb():
    torch.cuda.empty_cache()
    from quark.torch.quantization.graph.optimization.pre_quant.replace_linear_to_qtlinear import replace_linear_qtlinear
    from quark.torch.quantization.graph.optimization.pre_quant.replace_conv_bn_to_qt_model import replace_conv2dbn_quantizedconv_module
    from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantizedConvBatchNorm2d
    from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
    model = resnet18(pretrained=None).to(torch_device).eval()

    count_conv2d, count_bn2d, count_linear = 0, 0, 0
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            count_conv2d += 1
        if isinstance(module, nn.BatchNorm2d):
            count_bn2d += 1
        if isinstance(module, nn.Linear):
            count_linear += 1

    batch_shape = [4, 3, 112, 112]
    example_inputs = (torch.ones(batch_shape).to(torch_device), )
    graph_model = capture_pre_autograd_graph(model, example_inputs)
    replace_conv2dbn_quantizedconv_module(graph_model)
    replace_linear_qtlinear(graph_model)
    model_out = model(example_inputs[0])
    count_quantconvbn2d, count_quantlinear = 0, 0
    for module in graph_model.modules():
        if isinstance(module, QuantizedConvBatchNorm2d):
            module.eval()
            count_quantconvbn2d += 1
        if isinstance(module, QuantLinear):
            count_quantlinear += 1
    graph_out = graph_model(example_inputs[0])
    assert count_conv2d == count_bn2d and count_bn2d == count_quantconvbn2d, "replace conv bn number not equal"
    assert count_linear == count_quantlinear, "replace linear number not equal"
    assert torch.allclose(model_out, graph_out, atol=1e-5), "float model's out diffs vs graph model's out"
    torch.cuda.empty_cache()

@use_temporary_directory
def test_quant_res18_and_export(tmpdir: str):
    torch.cuda.empty_cache()
    # prepard float model and graph model
    model = resnet18(pretrained=None).to(torch_device).eval()
    example_inputs = (torch.rand(16, 3, 224, 224).to(torch_device), )
    graph_model = capture_pre_autograd_graph(model, example_inputs)
    # init config
    INT8_PER_TENSOR_SPEC = QuantizationSpec(dtype=Dtype.int8,
                                            qscheme=QSchemeType.per_tensor,
                                            observer_cls=PerTensorMinMaxObserver,
                                            symmetric=True,
                                            scale_type=ScaleType.float,
                                            round_method=RoundType.half_even,
                                            is_dynamic=False)
    quant_config = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC,
                                      output_tensors=INT8_PER_TENSOR_SPEC,
                                      weight=INT8_PER_TENSOR_SPEC,
                                      bias=INT8_PER_TENSOR_SPEC)
    quant_config = Config(global_quant_config=quant_config, quant_mode=QuantizationMode.fx_graph_mode)
    quantizer = ModelQuantizer(quant_config)

    prepared_model = quantizer._prepare_model(graph_model)

    # calibration
    for module in prepared_model.modules():
        if isinstance(module, ScaledFakeQuantize):
            module.enable_observer()
            module.disable_fake_quant()
    prepared_model(*example_inputs)
    for module in prepared_model.modules():
        if isinstance(module, ScaledFakeQuantize):
            module.disable_observer()
            module.enable_fake_quant()

    freezeded_model = quantizer.freeze(prepared_model)

    # ===========================test onnx ===========================
    from quark.torch import ModelExporter
    from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig
    try:
        config = ExporterConfig(json_export_config=JsonExporterConfig())
        exporter = ModelExporter(config=config, export_dir="./")
        example_inputs = (torch.rand(15, 3, 224, 224).to(torch_device), )
        exporter.export_onnx_model(freezeded_model, example_inputs[0])
    except Exception:
        print("Graph model  onnx export faild")
    print("Graph model  onnx export successful")

    # =======================test torch.export =======================
    example_inputs = example_inputs
    model_file_path = Path(tmpdir, "test_model_quantized.pth").as_posix()

    try:
        from quark.torch.export.api import save_params
        save_params(freezeded_model,
                    model_type="test_model",
                    args=example_inputs,
                    export_dir="./",
                    quant_mode=QuantizationMode.fx_graph_mode)
        # exported_model = torch.export.export(freezeded_model, example_inputs)
        # torch.export.save(exported_model, model_file_path)

        from quark.torch.quantization.api import load_params
        loaded_quantized_model = load_params(pth_path=model_file_path, quant_mode=QuantizationMode.fx_graph_mode)
        # loaded_exported_model = torch.export.load(model_file_path)
        # loaded_quantized_model = loaded_exported_model.module()

        ref_out = freezeded_model(*example_inputs)
        exp_out = loaded_quantized_model(*example_inputs)
        assert torch.allclose(ref_out, exp_out, atol=1e-5)
        print("Graph model  torch.export successful")
    except Exception:
        print("Graph model torch.export faild")
    torch.cuda.empty_cache()
