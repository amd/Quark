#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys

sys.path.append("..")
import onnx
import torch
import torch.nn as nn
from torch.fx import GraphModule

from quark.shares.utils.log import ScreenLogger
from quark.shares.utils.testing_utils import torch_device, use_temporary_directory
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, QuantizationMode, RoundType, ScaleType
from quark.torch.quantization.graph.graph_modelquantizer import FxGraphQuantizer
from quark.torch.quantization.graph.optimization.pre_quant.fold_bn_after_concat import fold_bn_after_concat
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d, QuantConvTranspose2d
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import (
    QuantConvTransposeBatchNorm2d,
    QuantizedConvBatchNorm2d,
)
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

logger = ScreenLogger(__name__)

TEST_TOPIC = "New Fx quant API\n"

INT8_PER_TENSOR_SPEC = QTensorConfig(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)
float_scale_quant_config = QLayerConfig(
    input_tensors=INT8_PER_TENSOR_SPEC,
    output_tensors=INT8_PER_TENSOR_SPEC,
    weight=INT8_PER_TENSOR_SPEC,
    bias=INT8_PER_TENSOR_SPEC,
)
fp_scale_quant_config = QConfig(global_quant_config=float_scale_quant_config, quant_mode=QuantizationMode.fx_graph_mode)


def onnx_contains_op_num(model_path: str, target_op_type: str) -> int:
    model = onnx.load(model_path)
    count = 0
    for node in model.graph.node:
        if node.op_type == target_op_type:
            count += 1
    return count


def fx_contains_op_num(model: GraphModule, check_func) -> int:
    count = 0
    for node in model.graph.nodes:
        if check_func(node):
            count += 1
    return count


def fx_contain_module_num(model: GraphModule, target_module: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, target_module):
            count += 1
    return count


"""
=============== Test model strategy ===============
To test the new API FxGraphQuantizer, meanwhile compliance with the old unified ModelQuantizer API
"""


class TinyShareWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 32, 3, bias=True, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.bn_sg = nn.BatchNorm2d(32)
        # test forward twice
        self.conv2d_1 = nn.Conv2d(32, 32, 3, bias=True)
        self.bn_1 = nn.BatchNorm2d(32)
        self.relu_1 = nn.ReLU(inplace=True)
        # test forward twice
        self.conv2d_2 = nn.Conv2d(32, 32, 3, bias=True)
        # test forward twice
        self.transposconv2d = nn.ConvTranspose2d(32, 32, (3, 3), bias=True)
        # test forward twice
        self.transposconv2d_1 = nn.ConvTranspose2d(32, 32, (3, 3), bias=True)
        self.bn_transposeconv = nn.BatchNorm2d(32)
        # test forward twice
        self.linear_1 = nn.Linear(32, 32)
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        # ===strategy: ops.conv + ops.bn -> QuantzedConv2dBn
        x = self.bn(self.conv2d(x))  # quantizer: w, b, input -> 3
        x = self.relu(x)  # quantizer: outout -> 1
        # ===strategy: ops.bn -> QuantConv
        x = self.bn_sg(x)  # convert to conv: weight, bias, output -> 3

        # ===strategy: split to seperate QuantConv
        x = self.relu_1(self.bn_1(self.conv2d_1(x)))  # quantizer: w, b, output -> 3
        x = self.relu_1(self.bn_1(self.conv2d_1(x)))  # quantizer: w, b, output -> 3
        # forward twice
        x = self.conv2d_2(x)  # quantizer: w, b, output -> 3
        x = self.conv2d_2(x)  # quantizer: w, b, output -> 3
        # forward twice
        x = self.transposconv2d(x)  # quantizer: w, b, output -> 3
        x = self.transposconv2d(x)  # quantizer: w, b, output -> 3
        # forward twice
        x = self.bn_transposeconv(self.transposconv2d_1(x))  # quantizer: w, b, output -> 3
        x = self.bn_transposeconv(self.transposconv2d_1(x))  # quantizer: w, b, output -> 3

        # x = self.adaptive_avg_pool2d(x)
        x = torch.mean(x, dim=(2, 3), keepdim=False)  # split 2 avgpool & adaptive
        x = torch.flatten(x, 1)  # quantizer: output -> 1
        # forward twice
        x = self.linear_1(x)  # quantizer: w, b -> 2
        x = torch.clip(x, 0, 1)  # replaced to relu # output -> 1
        x = self.linear_1(x)  # quantizer: w, b -> 2
        x = torch.clip(x, -1, 1)  # replaced to relu # output -> 1
        x = self.linear(x)  # quantizer: w, b, output -> 3
        return x


@use_temporary_directory
def test_fx_model_quantizer(tmpdir: str):
    """
    test torch model that if one submodel that contain parameter used over once
    , test code will show how the fx graph model is optimized for better deployment.
    """
    torch.cuda.empty_cache()
    float_model = TinyShareWeightModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 112, 112).to(torch_device),)
    _ = float_model.eval()(example_inputs[0])
    # ========== test using graph_model as input ===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, fp_scale_quant_config]:
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        graph_model_2 = torch.export.export_for_training(float_model, example_inputs).module()
        for model in [float_model, graph_model_2]:
            quantized_model = fx_quantizer.quantize_model(model, example_inputs, calibdata=example_inputs)  # only PTQ
            _ = quantized_model.eval()(example_inputs[0])
            # as scale after DPU's adaptive pool so skip : torch.allclose(out_fp32, opt_fx_graph)
            assert fx_contain_module_num(quantized_model, QuantizedConvBatchNorm2d) == 3
            assert fx_contain_module_num(quantized_model, QuantConv2d) == 3
            assert fx_contain_module_num(quantized_model, QuantConvTranspose2d) == 2
            assert fx_contain_module_num(quantized_model, QuantLinear) == 3
            assert fx_contain_module_num(quantized_model, QuantConvTransposeBatchNorm2d) == 2
            assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [42, 0]
            fx_quantizer.export_onnx_model(quantized_model, example_inputs, tmpdir + "w_quantized")
    logger.info(TEST_TOPIC + "split module that used over one to seperate module, Passed")
    torch.cuda.empty_cache()


"""
if linear -> concat -> batchnorm
then: merge the bn to (transposeconv, linear, conv2c)
"""


class Timy_Linear_Cat_BN_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(16, 16)
        self.linear2 = nn.Linear(16, 16)
        self.linear3 = nn.Linear(16, 16)
        self.bn1 = nn.BatchNorm1d(48)

    def forward(self, x):
        x1 = self.linear1(x)  # input , weight, bias, output -> 4
        x2 = self.linear2(x)  # weight, bias, output         -> 3
        x3 = self.linear3(x)  # weight, bias, output         -> 3
        x = torch.concatenate([x1, x2, x3], dim=1)  # output: 1
        x = self.bn1(x)  # recognized as bn1d, so not convert to conv
        return x


@use_temporary_directory
def test_fold_bn_2_linear_after_concat_strategy(tmpdir: str):
    """
    TODO this is a strategy need to be supprted
    """
    torch.cuda.empty_cache()
    float_model = Timy_Linear_Cat_BN_Model().to(torch_device).eval()
    example_inputs = (torch.rand(2, 16).to(torch_device),)
    out = float_model(example_inputs[0])
    # ========== unit fold_bn_after_concat ===============
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    graph_model = fold_bn_after_concat(graph_model)
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    out1 = graph_model.eval()(example_inputs[0])
    assert torch.allclose(out, out1, atol=1e-7)
    assert fx_contain_module_num(graph_model, QuantLinear) == 0

    # ========== test quant pipeline===============
    fx_quantizer = FxGraphQuantizer(fp_scale_quant_config)
    quantized_model = fx_quantizer.quantize_model(float_model, example_inputs, calibdata=example_inputs)  # only PTQ
    assert fx_contain_module_num(quantized_model, QuantLinear) == 3
    assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) == 11
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    test_fx_model_quantizer()
    test_fold_bn_2_linear_after_concat_strategy()
    torch.cuda.empty_cache()
