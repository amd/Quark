#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import math
import sys

sys.path.append("..")
import copy

import onnx
import torch
import torch.nn as nn
from torch.fx import GraphModule

import quark.torch.quantization.graph.optimization.post_quant.opt_pass_after_quant_powof2_scale as opt_after_qt_pow2_s
import quark.torch.quantization.graph.optimization.pre_quant.opt_pass_before_quant as opt_befor_qt
import quark.torch.quantization.graph.optimization.pre_quant.opt_pass_before_quant as opt_pre_qt_pass
from quark.shares.utils.log import ScreenLogger
from quark.shares.utils.testing_utils import retry_flaky_test, torch_device, use_temporary_directory
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, QuantizationMode, RoundType, ScaleType
from quark.torch.quantization.graph.graph_modelquantizer import FxGraphQuantizer
from quark.torch.quantization.graph.optimization.model_optimization import trans_opsfunc_2_quant_module
from quark.torch.quantization.graph.optimization.post_calib import AdjustBiasScaleQOPass
from quark.torch.quantization.graph.optimization.pre_quant.fold_bn_after_concat import fold_bn_after_concat
from quark.torch.quantization.graph.optimization.pre_quant.replace_silu_2_sigmoid_mul import replace_silu_node
from quark.torch.quantization.graph.optimization.remove_dropout_node import RemoveDropoutNode
from quark.torch.quantization.graph.torch_utils import (
    QUANT_CONV_WITH_BN,
    _is_sample_split_node,
    _is_split_with_size_node,
    is_adaptive_avg_pool2d_node,
    is_avg_pool2d_node,
    is_batchnorm_node,
    is_clip_node,
    is_dropout_node,
    is_hardsigmoid_node,
    is_hardswish_node,
    is_mean_node,
    is_mul_node,
    is_relu_act_node,
    is_sigmoid_node,
    is_silu_node,
    is_slice_node,
    is_split_node,
)
from quark.torch.quantization.nn.modules import (
    QuantAdaptiveAvgPool2d,
    QuantAvgPool2d,
    QuantConv2d,
    QuantConvTranspose2d,
    QuantConvTransposeBatchNorm2d,
    QuantizedConvBatchNorm2d,
    QuantLeakyReLU,
    QuantLinear,
)
from quark.torch.quantization.observer.observer import PerTensorPowOf2MinMaxObserver, PerTensorPowOf2MinMSEObserver
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

logger = ScreenLogger(__name__)

TEST_TOPIC = "torch FX graph mode quantization, align with hw deploy need\n"
ABSOLUTE_TOL = 1e-6
INT8_PER_TENSOR_SPEC = QTensorConfig(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorPowOf2MinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)
quant_tensor_config = QLayerConfig(
    input_tensors=INT8_PER_TENSOR_SPEC,
    output_tensors=INT8_PER_TENSOR_SPEC,
    weight=INT8_PER_TENSOR_SPEC,
    bias=INT8_PER_TENSOR_SPEC,
)
quant_config = QConfig(global_quant_config=quant_tensor_config, quant_mode=QuantizationMode.fx_graph_mode)


def onnx_contains_op_type(model_path: str, target_op_type: str) -> bool:
    model = onnx.load(model_path)
    return any(node.op_type == target_op_type for node in model.graph.node)


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
=============== Test if one module used over once ===============
if one module used over once, like the below example
(conv2d_1, conv2d_2, transposconv2d, linear_1)

Although, these modules will have one copy in torch module. In hardware (e.g IPU),
for better deployments, even though they share the same weight/bias, we treat each conv operation in the forward as different conv.
"""


class TinyShareConvbnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 32, 3, bias=True, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.relu(self.bn(self.conv2d(x1)))  # quantizer: w, b, input, output -> 4
        x2 = self.relu(self.bn(self.conv2d(x2)))  # quantizer: w, b, input, output -> 4
        return x1, x2


class TinyShareConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 32, 3, bias=True, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.relu(self.conv2d(x1))  # quantizer: w, b, input, output -> 4
        x2 = self.relu(self.conv2d(x2))  # quantizer: w, b, input, output -> 4
        return x1, x2


class TinyShareConvTransposeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transposeconv2d = nn.ConvTranspose2d(3, 32, 3, bias=True, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.relu(self.transposeconv2d(x1))  # quantizer: w, b, input, output -> 4
        x2 = self.relu(self.transposeconv2d(x2))  # quantizer: w, b, input, output -> 4
        return x1, x2


class TinyShareConvTransposeBnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.transposeconv2d = nn.ConvTranspose2d(3, 32, 3, bias=True, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.relu(self.bn(self.transposeconv2d(x1)))  # quantizer: w, b, input, output -> 4
        x2 = self.relu(self.bn(self.transposeconv2d(x2)))  # quantizer: w, b, input, output -> 4
        return x1, x2


class TinyShareLinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(3, 10, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x1, x2):
        x1 = self.relu(self.linear(torch.flatten(self.pool(x1), 1)))
        x2 = self.relu(self.linear(torch.flatten(self.pool(x2), 1)))
        return x1, x2


@retry_flaky_test()
def test_use_over_once_module_optim():
    """
    In optimized Graph, will has two QuantizedConvBatchNorm2d/or similiar module
    """
    torch.cuda.empty_cache()
    float_model1 = TinyShareConvbnModel().to(torch_device).eval()
    float_model2 = TinyShareConvModel().to(torch_device).eval()
    float_model3 = TinyShareConvTransposeModel().to(torch_device).eval()
    float_model4 = TinyShareLinearModel().to(torch_device).eval()
    float_model5 = TinyShareConvTransposeBnModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 32, 32).to(torch_device), torch.rand(1, 3, 32, 32).to(torch_device))
    opt_instance = opt_pre_qt_pass.SplitQuantModuleCalledOverOnce()
    for each_fp_model in [float_model1, float_model2, float_model3, float_model4, float_model5]:
        each_fp_model(*example_inputs)
        # session 1
        # ========== test using hardware constrain ===============
        graph_model_0 = torch.export.export_for_training(each_fp_model, example_inputs).module()
        graph_model = trans_opsfunc_2_quant_module(graph_model_0)
        graph_model = opt_instance(graph_model)
        out_fp32 = each_fp_model.eval()(*example_inputs)
        count_QuantizeModule = 0
        for module in graph_model.modules():
            if isinstance(
                module,
                (
                    QuantizedConvBatchNorm2d,
                    QuantConvTransposeBatchNorm2d,
                    QuantConv2d,
                    QuantConvTranspose2d,
                    QuantLinear,
                ),
            ):
                module.freeze_bn_stats() if isinstance(module, QUANT_CONV_WITH_BN) else None
                count_QuantizeModule += 1
        out_opt_fx_graph = graph_model(*example_inputs)
        assert count_QuantizeModule == 2, (
            f"In model {each_fp_model.__class__.__name__}, the QuantizedModule num should ne 2"
        )
        assert torch.allclose(out_fp32[0], out_opt_fx_graph[0], atol=ABSOLUTE_TOL)
        assert torch.allclose(out_fp32[1], out_opt_fx_graph[1], atol=ABSOLUTE_TOL)

        # session 2 not using hw constrain, will not copy another QuantModule instance
        graph_model = torch.export.export_for_training(each_fp_model, example_inputs).module()
        graph_model = trans_opsfunc_2_quant_module(graph_model)
        out_fp32 = each_fp_model.eval()(*example_inputs)
        count_QuantizeModule = 0
        for module in graph_model.modules():
            if isinstance(
                module,
                (
                    QuantizedConvBatchNorm2d,
                    QuantConvTransposeBatchNorm2d,
                    QuantConv2d,
                    QuantConvTranspose2d,
                    QuantLinear,
                ),
            ):
                module.freeze_bn_stats() if isinstance(module, QUANT_CONV_WITH_BN) else None
                count_QuantizeModule += 1
        out_opt_fx_graph = graph_model(*example_inputs)
        assert count_QuantizeModule == 1, (
            f"In model {each_fp_model.__class__.__name__}, the QuantizedModule num should ne 1"
        )
        assert torch.allclose(out_fp32[0], out_opt_fx_graph[0], atol=ABSOLUTE_TOL)
        assert torch.allclose(out_fp32[1], out_opt_fx_graph[1], atol=ABSOLUTE_TOL)

    logger.info(TEST_TOPIC + "[5 small model] split module used over onee to seperate modul. Passed")
    torch.cuda.empty_cache()


"""
=============== Test if one module used over once ===============
if one module used over once, like the below example
(conv2d_1, conv2d_2, transposconv2d, linear_1)

Although, these modules will have one copy in torch module. In hardware (e.g IPU),
for better deployments, we will regard every convolutional operation in the forward path as a different conv, even though they share the same weight/bias.
"""


class TinyShareWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 32, 3, bias=True, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)

        # test forward twice
        self.conv2d_1 = nn.Conv2d(32, 32, 3, bias=True)
        self.bn_1 = nn.BatchNorm2d(32)
        self.relu_1 = nn.ReLU(inplace=True)
        # test forward 3
        self.conv2d_2 = nn.Conv2d(32, 32, 3, bias=True)
        # test forward 3
        self.transposconv2d = nn.ConvTranspose2d(32, 32, (3, 3), bias=True)
        # test forward 3
        self.transposconv2d_1 = nn.ConvTranspose2d(32, 32, (3, 3), bias=True)
        self.bn_2 = nn.BatchNorm2d(32)
        # test forward twice
        self.linear_1 = nn.Linear(32, 32)

        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        x = self.bn(self.conv2d(x))  # quantizer: w, b, input -> 3
        x = self.relu(x)  # quantizer: outout -> 1
        # forward twice
        x = self.relu_1(self.bn_1(self.conv2d_1(x)))  # quantizer: w, b, output -> 3
        x = self.relu_1(self.bn_1(self.conv2d_1(x)))  # quantizer: w, b, output -> 3
        # forward 3
        x = self.conv2d_2(x)  # quantizer: w, b, output -> 3
        x = self.conv2d_2(x)  # quantizer: w, b, output -> 3
        x = self.conv2d_2(x)  # quantizer: w, b, output -> 3
        # forward 3
        x = self.transposconv2d(x)  # quantizer: w, b, output -> 3
        x = self.transposconv2d(x)  # quantizer: w, b, output -> 3
        x = self.transposconv2d(x)  # quantizer: w, b, output -> 3
        # forward3
        x = self.bn_2(self.transposconv2d_1(x))  # quantizer: w, b, output -> 3
        x = self.bn_2(self.transposconv2d_1(x))  # quantizer: w, b, output -> 3
        x = self.bn_2(self.transposconv2d_1(x))  # quantizer: w, b, output -> 3

        x = x[:, :, 0, 0]
        x = torch.flatten(x, 1)  # quantizer: output -> 1
        # forward twice
        x = self.linear_1(x)  # quantizer: w, b, output -> 3
        x = self.linear_1(x)  # quantizer: w, b, output -> 3
        x = self.linear_1(x)  # quantizer: w, b, output -> 3
        x = self.linear_1(x)  # quantizer: w, b, output -> 3
        x = self.linear(x)  # quantizer: w, b, output -> 3
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_module_used_over_once_optim_strategy(tmpdir: str):
    """
    test torch model that if one submodel that contain parameter used over once
    , test code will show how the fx graph model is optimized for better deployment.
    """
    INT8_PER_TENSOR_MSE_POW2_SPEC = QTensorConfig(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorPowOf2MinMSEObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    powof2_mse_tensor_config = QLayerConfig(
        input_tensors=INT8_PER_TENSOR_MSE_POW2_SPEC,
        output_tensors=INT8_PER_TENSOR_MSE_POW2_SPEC,
        weight=INT8_PER_TENSOR_MSE_POW2_SPEC,
        bias=INT8_PER_TENSOR_MSE_POW2_SPEC,
    )
    mse_pow2_config = QConfig(global_quant_config=powof2_mse_tensor_config, quant_mode=QuantizationMode.fx_graph_mode)

    torch.cuda.empty_cache()
    float_model = TinyShareWeightModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 15, 15).to(torch_device),)
    # session 1
    # ========== test using hardware constrain ===============
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    graph_model = trans_opsfunc_2_quant_module(graph_model)  # default op
    opt_instance = opt_pre_qt_pass.SplitQuantModuleCalledOverOnce()
    graph_model = opt_instance(graph_model)

    out_fp32 = float_model.eval()(example_inputs[0])
    for module in graph_model.modules():
        if isinstance(module, QUANT_CONV_WITH_BN):
            module.freeze_bn_stats()
    out_opt_fx_graph = graph_model(example_inputs[0])
    assert fx_contain_module_num(graph_model, QuantizedConvBatchNorm2d) == 3
    assert fx_contain_module_num(graph_model, QuantConv2d) == 3
    assert fx_contain_module_num(graph_model, QuantConvTranspose2d) == 3
    assert fx_contain_module_num(graph_model, QuantLinear) == 5
    assert fx_contain_module_num(graph_model, QuantConvTransposeBatchNorm2d) == 3
    assert torch.allclose(out_fp32, out_opt_fx_graph)
    # ========== test not using hardware constrain ===============
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    opt_fx_before_qt = trans_opsfunc_2_quant_module(graph_model)  # default op
    out_fp32 = float_model.eval()(example_inputs[0])
    for module in graph_model.modules():
        if isinstance(module, QUANT_CONV_WITH_BN):
            module.freeze_bn_stats()
    out_opt_fx_graph = opt_fx_before_qt(example_inputs[0])
    assert fx_contain_module_num(opt_fx_before_qt, QuantizedConvBatchNorm2d) == 2
    assert fx_contain_module_num(opt_fx_before_qt, QuantConv2d) == 1
    assert fx_contain_module_num(opt_fx_before_qt, QuantConvTranspose2d) == 1
    assert fx_contain_module_num(graph_model, QuantConvTransposeBatchNorm2d) == 1
    assert fx_contain_module_num(opt_fx_before_qt, QuantLinear) == 2
    assert torch.allclose(out_fp32, out_opt_fx_graph)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [mse_pow2_config, emp_quant_config, quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()
        quantized_model = quantizer.quantize_model(graph_model, [example_inputs[0]])
        out_2 = quantized_model.eval()(*example_inputs)
        if each_quant_config == emp_quant_config:
            assert torch.allclose(out_fp32, out_2)
        assert fx_contain_module_num(graph_model, QuantizedConvBatchNorm2d) == 3
        assert fx_contain_module_num(graph_model, QuantConv2d) == 3
        assert fx_contain_module_num(graph_model, QuantConvTranspose2d) == 3
        assert fx_contain_module_num(graph_model, QuantLinear) == 5
        assert fx_contain_module_num(graph_model, QuantConvTransposeBatchNorm2d) == 3
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [53, 0]
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        opt_graph_module(*example_inputs)
        onnx_dir = tmpdir + "/module_called_over_once.onnx"
        torch.onnx.export(opt_graph_module, example_inputs, onnx_dir, dynamo=False)
        assert onnx_contains_op_num(onnx_dir, "Conv") == 6
        assert onnx_contains_op_num(onnx_dir, "ConvTranspose") == 6
        assert onnx_contains_op_num(onnx_dir, "Gemm") == 5
    logger.info(TEST_TOPIC + "split module that used over one to seperate module, Passed")
    torch.cuda.empty_cache()


"""
Align with onnx model optimization before quantization
"""


class TinyConvertBn2ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 16, 3, bias=True, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn_1 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.conv2d(x)  # input, w, b,
        x = self.relu(x)  # output
        #  the bn_1 -> conv
        x = self.bn_1(x)  # w, b, out
        return x


@retry_flaky_test()
def test_torch_sg_bn2d_to_conv2d_optim_strategy():
    """
    For better allign with hw requirements for deployment,
    transfer one single batchnorm2d to conv2d.
    """
    torch.cuda.empty_cache()
    float_model = TinyConvertBn2ConvModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 64, 64).to(torch_device),)
    # session 1
    # ========== test using hardware constrain ===============
    graph_model_2 = torch.export.export_for_training(float_model, example_inputs).module()
    opt_graph_module = trans_opsfunc_2_quant_module(graph_model_2)  # default op
    opt_instance = opt_pre_qt_pass.ConvertBn2D2ConvQOPass()
    opt_graph_module = opt_instance(opt_graph_module)

    out_fp32 = float_model.eval()(example_inputs[0])
    out_fx_graph = opt_graph_module(example_inputs[0])
    assert fx_contain_module_num(opt_graph_module, QuantConv2d) == 2
    assert torch.allclose(out_fp32, out_fx_graph, atol=ABSOLUTE_TOL)

    # ========== test quant pipeline==========
    graph_model_2 = torch.export.export_for_training(float_model, example_inputs).module()
    quantizer = ModelQuantizer(quant_config)
    quantized_model = quantizer.quantize_model(
        graph_model_2, [torch.rand(4, 3, 32, 32).to(torch_device) for _ in range(3)]
    )

    assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) == 7, (
        "The total quantizer in this model should be 7"
    )
    quantized_model(example_inputs[0])
    torch.cuda.empty_cache()


"""
Tiny_Convert_Clip_To_Relu: post quantize optim strategy:
After quantization, conver a clip operation to Relu (with restriction, need to check the clip param)
"""


class Tiny_Convert_Clip_To_Relu(nn.Module):
    def __init__(self):
        super(Tiny_Convert_Clip_To_Relu, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)  # input, weight, bias
        x = self.relu(x)  # output
        x = torch.clip(x, 0, 1)  # output ,   will be transfer to Relu
        x = torch.clip_(x, -1, 1)  # output ,  will not be transfer to Relu
        x = torch.clamp(x, -0.5, 1)
        x = torch.clamp_(x, -0.2, 1)
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_clip_2_relu_optim_strategy(tmpdir: str):
    """
    For better allign with hw requirements for deployment,
    transfer one clip node to relu node after quantization.
    """
    torch.cuda.empty_cache()
    float_model = Tiny_Convert_Clip_To_Relu().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 16, 16).to(torch_device),)
    # ========== test using hardware constrain ===============
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    opt_graph_module = opt_after_qt_pow2_s.ConvertClip2ReLUQOPass()(graph_model)
    assert fx_contains_op_num(opt_graph_module, is_relu_act_node) == 1
    assert fx_contains_op_num(opt_graph_module, is_clip_node) == 4
    # ========== test quant pipeline===============
    quantizer = ModelQuantizer(quant_config)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    quantized_model = quantizer.quantize_model(graph_model, [example_inputs[0]])
    _ = quantized_model.eval()(*example_inputs)
    assert fx_contains_op_num(quantized_model, is_relu_act_node) == 1
    assert fx_contains_op_num(quantized_model, is_clip_node) == 4
    assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) == 8
    opt_graph_module = quantizer.freeze(quantized_model.eval())
    assert fx_contains_op_num(opt_graph_module, is_relu_act_node) == 2
    assert fx_contains_op_num(opt_graph_module, is_clip_node) == 3
    opt_graph_module(*example_inputs)
    onnx_dir = tmpdir + "/clip_2_relu.onnx"
    torch.onnx.export(opt_graph_module, example_inputs, onnx_dir, dynamo=False)
    assert onnx_contains_op_num(onnx_dir, "Relu") == 2
    assert onnx_contains_op_num(onnx_dir, "Clip") == 3
    torch.cuda.empty_cache()


"""
TinyMean2GAP: pre quantize optim strategy:
Before quantization, conver a mean to GAP (with restriction, need to check mean equal to GAP)
"""


class TinyMean2GAP(nn.Module):
    def __init__(self):
        super(TinyMean2GAP, self).__init__()
        self.conv = nn.Conv2d(3, 8, 3, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool3 = nn.AvgPool2d(3, stride=2)

    def forward(self, x0, x1, x2, x3, x4, x5, x6, x7, x8):
        x0 = torch.mean(x0, dim=(2), keepdim=False)  # onnx: reducemean
        # input output  -> 2
        x1 = torch.mean(x1, dim=(2), keepdim=True)  # onnx: reducemean
        # input output  -> 2
        x2 = torch.mean(
            x2, dim=(2, 3), keepdim=True
        )  # onnx: GlobalAveragePool  # torch2.5 & cap: torch.ops.aten.mean.dim()
        # input output  -> 2
        x3 = self.pool1(
            x3
        )  # onnx: GlobalAveragePool  #torch2.5 & torchcap torch.ops.aten.adaptive_avg_pool2d.default()
        # input output  -> 2
        x4 = nn.functional.adaptive_avg_pool2d(x4, (1, 1))  # onnx: GlobalAveragePool
        # input output  -> 2
        x5 = self.pool2(x5)  # onnx: AveragePool  #torch.ops.aten.avg_pool2d.default();
        # input output  -> 2
        x6 = self.pool3(x6)  # onnx: AveragePool  #torch.ops.aten.avg_pool2d.default();
        # input output  -> 2
        x7 = self.conv(x7)
        # input output weight, bias -> 4
        x8 = torch.mean(x8, dim=(3, -2), keepdim=True)
        return x0, x1, x2, x3, x4, x5, x6, x7, x8


@retry_flaky_test()
@use_temporary_directory
def test_mean_2_pooling_strategy(tmpdir: str):
    """
    For better allign with hw requirements for deployment,
    transfer one clip node to relu node after quantization.
    """
    torch.cuda.empty_cache()
    float_model = TinyMean2GAP().to(torch_device).eval()
    example_inputs = (
        torch.rand(1, 3, 6, 6).to(torch_device),
        torch.rand(1, 3, 8, 8).to(torch_device),
        torch.rand(1, 3, 10, 10).to(torch_device),
        torch.rand(1, 3, 12, 12).to(torch_device),
        torch.rand(1, 3, 14, 14).to(torch_device),
        torch.rand(1, 3, 16, 16).to(torch_device),
        torch.rand(1, 3, 18, 18).to(torch_device),
        torch.rand(1, 3, 20, 20).to(torch_device),
        torch.rand(1, 3, 22, 22).to(torch_device),
    )
    quant_inputs = {"x" + str(index): value for index, value in enumerate(example_inputs)}
    fp_out = float_model(*example_inputs)
    # ========== test using hardware constrain ===============
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    assert fx_contains_op_num(graph_model, is_mean_node) == 4
    assert fx_contains_op_num(graph_model, is_adaptive_avg_pool2d_node) == 3
    graph_model = opt_befor_qt.ConvertReduceMean2GapQOPass()(graph_model)
    assert fx_contains_op_num(graph_model, is_mean_node) == 2
    assert fx_contains_op_num(graph_model, is_adaptive_avg_pool2d_node) == 5
    graph_out = graph_model(*example_inputs)
    assert all([torch.allclose(x[0], x[1]) for x in zip(fp_out, graph_out, strict=False)])
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()
        quantized_model = quantizer.quantize_model(graph_model, [quant_inputs])
        out_2 = quantized_model.eval()(*example_inputs)
        if each_quant_config == emp_quant_config:
            assert [i for i, x in enumerate(zip(fp_out, out_2, strict=False)) if torch.allclose(x[0], x[1])] == [
                0,
                1,
                5,
                7,
            ]
        assert fx_contains_op_num(graph_model, is_mean_node) == 2
        assert fx_contains_op_num(graph_model, is_adaptive_avg_pool2d_node) == 1
        assert fx_contain_module_num(graph_model, QuantAdaptiveAvgPool2d) == 4
        assert fx_contain_module_num(graph_model, QuantAvgPool2d) == 1
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [20, 0]
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        opt_graph_module(*example_inputs)
        onnx_dir = tmpdir + "/mean_2_pooling.onnx"
        torch.onnx.export(opt_graph_module, example_inputs, onnx_dir, dynamo=False)
        assert onnx_contains_op_num(onnx_dir, "GlobalAveragePool") == 4
        assert onnx_contains_op_num(onnx_dir, "ReduceMean") == 2
        assert onnx_contains_op_num(onnx_dir, "AveragePool") == 2
        assert onnx_contains_op_num(onnx_dir, "Mul") == 5
    torch.cuda.empty_cache()


"""
convert split to slice:
"""


class TinyConvertSplit2Slice(nn.Module):
    def __init__(self):
        super(TinyConvertSplit2Slice, self).__init__()
        self.conv = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True)

    def forward(self, x):
        # input quant
        x1 = self.conv(x)  # weight, bias, out
        x2 = self.conv(x)  # weight, bias, out
        x3, x4, x5, x6 = torch.split(x1, [8, 16, 4, 4], 1)  # out1, out2, out3, out4
        # torch.ops.aten.split_with_sizes.default(conv2d, [8, 16, 4, 4], 1)
        x7, x8, x9, x10 = torch.split(x2, 9, dim=1)  # out1, out2, out3, out4
        # torch.ops.aten.split.Tensor(conv2d_1, 8, 1)
        x11 = torch.cat([x3, x4, x5, x6], dim=1)  # out
        x12 = torch.cat([x7, x8, x9, x10], dim=1)  # out

        x13, x14, x15, x16 = torch.split(x1, [7, 7, 6, 8], -1)  # out1, out2, out3, out4
        # torch.ops.aten.split_with_sizes.default(conv2d, [7, 7, 6, 8], -1)
        x17, x18, x19, x20 = torch.split(x2, 7, dim=-1)  # out1, out2, out3, out4
        # torch.ops.aten.split.Tensor(conv2d_1, 7, -1)
        x21 = torch.cat([x13, x14, x15, x16], dim=-1)  # out
        x22 = torch.cat([x17, x18, x19, x20], dim=-1)  # out

        return x11, x12, x21, x22


@retry_flaky_test()
@use_temporary_directory
def test_torch_convert_split_2_slice_strategy(tmpdir: str):
    """
    For better allign with hw requirements for deployment,
    transfer one split node to multi slice node before quantization.
    """
    torch.cuda.empty_cache()
    float_model = TinyConvertSplit2Slice().to(torch_device).eval()
    example_inputs = (torch.rand(1, 16, 28, 28).to(torch_device),)
    out = float_model(example_inputs[0])
    # ========== unit test ConvertSplit2SliceQOPass ===============
    graph_model_2 = torch.export.export_for_training(float_model, example_inputs).module()
    for graph_model in [graph_model_2]:
        assert fx_contains_op_num(graph_model, _is_split_with_size_node) == 2
        assert fx_contains_op_num(graph_model, _is_sample_split_node) == 2
        graph_model = opt_befor_qt.ConvertSplit2SliceQOPass()(graph_model)
        assert fx_contains_op_num(graph_model, is_split_node) == 0
        assert fx_contains_op_num(graph_model, is_slice_node) == 16
        graph_out = graph_model(example_inputs[0])
        assert all([torch.allclose(x[0], x[1]) for x in zip(out, graph_out, strict=False)])
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()
        quantized_model = quantizer.quantize_model(graph_model, [example_inputs[0]])
        out_2 = quantized_model.eval()(example_inputs[0])
        if each_quant_config == emp_quant_config:
            assert all([torch.allclose(x[0], x[1]) for x in zip(out, out_2, strict=False)])
        assert fx_contains_op_num(graph_model, is_split_node) == 0
        assert fx_contains_op_num(graph_model, is_slice_node) == 16
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [27, 0]
        assert fx_contain_module_num(quantized_model, QuantConv2d) == 2
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        opt_graph_module(example_inputs[0])
        onnx_dir = tmpdir + "/split_2_slice.onnx"
        torch.onnx.export(opt_graph_module, *example_inputs, onnx_dir, dynamo=False)
        assert onnx_contains_op_num(onnx_dir, "Slice") == 16
        assert onnx_contains_op_num(onnx_dir, "Conv") == 2
    torch.cuda.empty_cache()


"""
if (transposeconv, linear, conv2c)'s output folled by: concat -> batchnorm
then: merge the bn to (transposeconv, linear, conv2c)
"""


class Tiny_Fold_Batch_Norm_After_Concat_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=0)

        self.conv_transpose1 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=0, bias=None)
        self.conv_transpose2 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=0)
        self.conv_transpose3 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(48)

        self.conv1 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(48)

    def forward(self, x):
        x = self.conv(x)  # input, weight, bias output: 4
        x1, x2, x3 = torch.split(x, [16, 16, 16], 1)  # out1, out2, out3: 3

        # called once
        x1 = self.conv_transpose1(x1)  # weight, bias, output: 3
        x2 = self.conv_transpose2(x2)  # weight, bias, output: 3
        x3 = self.conv_transpose3(x3)  # weight, bias, output: 3
        x = torch.concatenate([x1, x2, x3], dim=1)  # output: 1
        x = self.bn1(x)  # merged

        x1, x2, x3 = torch.split(x, [16, 16, 16], 1)  # out1, out2, out3: 3
        # called twice
        x1 = self.conv_transpose1(x1)  # weight, bias, output: 3
        x2 = self.conv_transpose2(x2)  # weight, bias, output: 3
        x3 = self.conv_transpose3(x3)  # weight, bias, output: 3
        x = torch.concatenate([x1, x2, x3], dim=1)  # output: 1
        x = self.bn1(x)

        x1, x2 = torch.split(x, 24, 1)  # out1, out2: 2
        # called once
        x1 = self.conv1(x1)  # weight, bias, output: 3
        x2 = self.conv2(x2)  # weight, bias, output: 3
        x = torch.concat([x1, x2], dim=1)  # output: 1
        x = self.bn2(x)

        x1, x2 = torch.split(x, 24, 1)  # out1, out2: 2
        # called once
        x1 = self.conv1(x1)  # weight, bias, output: 3
        x2 = self.conv2(x2)  # weight, bias, output: 3
        x = torch.concat([x1, x2], dim=1)  # output: 1
        x = self.bn2(x)
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_fold_bn_after_concat_strategy(tmpdir: str):
    """
    For better allign with hw requirements for deployment,
    before:
        (conv2d, transposed2d, linear) -> concat -> bn
    after:
        (conv2d, transposed2d, linear) -> concat

    example:
        before:
          conv  transposeconv2d
            |     |
               cat
                |
               BN
        after optimization:
          conv  transposeconv2d
            |     |
               cat
                |
                |
    """
    torch.cuda.empty_cache()
    float_model = Tiny_Fold_Batch_Norm_After_Concat_Model().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    out = float_model(example_inputs[0])
    # ========== unit fold_bn_after_concat ===============
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    graph_model = fold_bn_after_concat(graph_model)
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    out1 = graph_model.eval()(example_inputs[0])
    assert torch.allclose(out, out1, atol=ABSOLUTE_TOL)
    assert fx_contain_module_num(graph_model, QuantizedConvBatchNorm2d) == 2
    assert fx_contain_module_num(graph_model, QuantConvTransposeBatchNorm2d) == 3
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [quant_config, emp_quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()
        quantized_model = quantizer.quantize_model(graph_model, [example_inputs[0]])
        out_2 = quantized_model.eval()(example_inputs[0])
        if each_quant_config == emp_quant_config:
            assert torch.allclose(out, out_2, atol=ABSOLUTE_TOL)
        assert fx_contain_module_num(quantized_model, QuantizedConvBatchNorm2d) == 4
        assert fx_contain_module_num(quantized_model, QuantConvTransposeBatchNorm2d) == 6
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [48, 0]
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        opt_graph_module(example_inputs[0])
        onnx_dir = tmpdir + "/fold_bn_afterconcat.onnx"
        torch.onnx.export(opt_graph_module, *example_inputs, onnx_dir, dynamo=False)
        assert onnx_contains_op_num(onnx_dir, "ConvTranspose") == 6
        assert onnx_contains_op_num(onnx_dir, "Conv") == 5
    torch.cuda.empty_cache()


"""
if transposed2d 's group is not 1, can not perform fold
"""


class Tiny_Bn_After_Concat_Not_Fold_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_transpose1 = nn.ConvTranspose2d(16, 16, kernel_size=3, groups=2)
        self.conv_transpose2 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=0)
        self.conv_transpose3 = nn.ConvTranspose2d(16, 16, kernel_size=3, padding=0)
        self.bn1 = nn.BatchNorm2d(48)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU6()
        self.bn2 = nn.BatchNorm2d(96)
        self.conv_transpose4 = nn.ConvTranspose2d(96, 96, kernel_size=3)
        self.conv_transpose5 = nn.ConvTranspose2d(96, 96, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(192)
        self.relu_3 = nn.ReLU()

    def forward(self, x):  # input :1
        x1 = self.conv_transpose1(x)  # weight, bias, output: 3
        x2 = self.conv_transpose2(x)  # weight, bias, output: 3
        x3 = self.conv_transpose3(x)  # weight, bias, output: 3
        x = torch.concatenate([x1, x2, x3], dim=1)  # output: 1
        x = self.bn1(x)  # can not be merged/ replaced to conv2d weight, bias, output: 3
        x1 = self.relu_1(x)  # output: 1
        x2 = self.relu_2(x)  # output: 1
        x = torch.concatenate([x1, x2], dim=1)  # output: 1
        x = self.bn2(x)  # can not fold , convert to conv weight, bias, output: 3

        x1 = self.conv_transpose4(x)  # weight, bias, outout : 3
        x2 = self.conv_transpose5(x)  # weight, bias, outout : 3
        x = torch.concatenate([x1, x2], dim=1)  # output: 1
        x = self.bn3(x)  # weight, bias, outout : 3
        x3 = self.relu_3(x1)  # outout : 1
        x = torch.cat([x, x3], dim=1)  # outout : 1
        return x


@retry_flaky_test()
@use_temporary_directory
def test_not_fold_bn_after_concat_strategy(tmpdir: str):
    """
    In this example, as transposedconv2d's group is not 1, so can not perform fold bn
    """
    torch.cuda.empty_cache()
    float_model = Tiny_Bn_After_Concat_Not_Fold_Model().to(torch_device).eval()
    example_inputs = (torch.rand(1, 16, 28, 28).to(torch_device),)
    out = float_model(example_inputs[0])
    # ========== unit fold_bn_after_concat ===============
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    graph_model = fold_bn_after_concat(graph_model)
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    out1 = graph_model.eval()(example_inputs[0])
    assert torch.allclose(out, out1, atol=ABSOLUTE_TOL)
    assert fx_contains_op_num(graph_model, is_batchnorm_node) == 3
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()
        quantized_model = quantizer.quantize_model(graph_model, [example_inputs[0]])
        out_2 = quantized_model.eval()(example_inputs[0])
        if each_quant_config == emp_quant_config:
            assert torch.allclose(out, out_2, atol=ABSOLUTE_TOL)
        assert fx_contain_module_num(quantized_model, QuantConvTranspose2d) == 5
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [32, 0]
        assert fx_contain_module_num(quantized_model, QuantConv2d) == 3
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        onnx_dir = tmpdir + "/not_fold_bn_afterconcat.onnx"
        torch.onnx.export(opt_graph_module, *example_inputs, onnx_dir, dynamo=False)
        assert onnx_contains_op_num(onnx_dir, "Conv") == 3
        assert onnx_contains_op_num(onnx_dir, "ConvTranspose") == 5
    torch.cuda.empty_cache()


"""
NOTE haoliang LayerNorm node
At present, torch.ops.aten.layer_norm.default will be translated to Onnx's LayerNormalization.
No need to process this problem.
This mainly caused by the opset version
"""


class Tiny_LayerNorm_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(32, 64)
        self.layernorm = nn.LayerNorm(64)
        self.linear1 = nn.Linear(64, 64)

    def forward(self, x):
        x = self.linear(x)
        x = self.layernorm(x)
        x = self.linear1(x)
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_layerNorm_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = Tiny_LayerNorm_Model().to(torch_device).eval()
    example_inputs = (torch.rand(4, 32).to(torch_device),)
    out = float_model(example_inputs[0])
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()
        quantized_model = quantizer.quantize_model(graph_model, [example_inputs[0]])
        out_2 = quantized_model.eval()(example_inputs[0])
        if each_quant_config == emp_quant_config:
            assert torch.allclose(out, out_2)
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [10, 0]
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        opt_graph_module(example_inputs[0])
        out_onnx_dir = tmpdir + "/lm.onnx"
        torch.onnx.export(opt_graph_module, *example_inputs, out_onnx_dir, dynamo=False)
        assert onnx_contains_op_num(out_onnx_dir, "LayerNormalization") == 1
    torch.cuda.empty_cache()


"""
TODO NOTE GELU node need to optimize ONNX model, rather than in Fx quant
"""


class Tiny_Fuse_Gelu_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.conv(x)  # input, weight, bias
        return self.gelu(x)  # output


@retry_flaky_test()
@use_temporary_directory
def test_torch_fuse_gelu_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = Tiny_Fuse_Gelu_Model().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    out = float_model(example_inputs[0])
    # graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()

        quantized_model = quantizer.quantize_model(graph_model, [example_inputs[0]])
        out_2 = quantized_model.eval()(example_inputs[0])
        if each_quant_config == emp_quant_config:
            assert torch.allclose(out, out_2)
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [4, 0]
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        opt_graph_module(example_inputs[0])
        out_onnx_path = tmpdir + "/gelu_layer.onnx"
        torch.onnx.export(opt_graph_module, *example_inputs, out_onnx_path, dynamo=False)
        # TODO Gelu should be done in onnx level
    torch.cuda.empty_cache()


"""
Split a large pooling kernel to smaller one
"""


class Tiny_Split_Large_Kernel_Pool_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(3, 16, kernel_size=3, stride=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x1 = self.conv1(x)  # input, weight, bias, output -> 4
        x2 = self.conv2(x)  # weight, bias, output -> 3
        x1 = self.pool(x1)  # output1, output2 -> 2
        x2 = self.pool(x2)  # output1, output2 -> 2
        x = torch.cat([x1, x2], dim=1)  # output-> 1
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_split_large_kernel_pool_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = Tiny_Split_Large_Kernel_Pool_Model().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 64, 100).to(torch_device),)
    out = float_model(example_inputs[0])
    # ========== unit SplitLargeKernelPoolQOPass ===============
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    out1 = graph_model.eval()(example_inputs[0])
    # checkt avgpooling num
    assert fx_contains_op_num(graph_model, is_adaptive_avg_pool2d_node) == 2
    assert fx_contains_op_num(graph_model, is_avg_pool2d_node) == 0
    opt_graph = opt_befor_qt.SplitLargeKernelPoolQOPass()(graph_model)
    out2 = opt_graph.eval()(example_inputs[0])
    assert fx_contains_op_num(opt_graph, is_adaptive_avg_pool2d_node) == 2
    assert fx_contains_op_num(opt_graph, is_avg_pool2d_node) == 2
    assert torch.allclose(out, out1)
    assert torch.allclose(out, out2)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()

        quantized_model = quantizer.quantize_model(graph_model, [example_inputs[0]])
        quantized_model.eval()(example_inputs[0])  # out_2 =
        # as scale after adaptive pool as skip -> assert torch.allclose(out, out_2)
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [12, 0]
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        opt_graph_module(example_inputs[0])
        assert fx_contain_module_num(opt_graph_module, QuantAdaptiveAvgPool2d) == 2
        assert fx_contain_module_num(opt_graph_module, QuantAvgPool2d) == 2
        out_onnx_path = tmpdir + "/split_large_gap_pooling.onnx"
        torch.onnx.export(opt_graph_module, *example_inputs, out_onnx_path, dynamo=False)
        assert onnx_contains_op_num(out_onnx_path, "GlobalAveragePool") == 2
        assert onnx_contains_op_num(out_onnx_path, "AveragePool") == 2
        assert onnx_contains_op_num(out_onnx_path, "Mul") == 4
    torch.cuda.empty_cache()


"""
Delete redundant Slice
"""


class Slice_Tensor_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 48)
        self.conv2d = nn.Conv2d(3, 16, 3)
        self.conv3d = nn.Conv3d(3, 16, 3)

    def forward(self, x0, x1, x2):
        x0 = self.linear(x0)  # input, weight, bias, output
        x1 = self.conv2d(x1)  # input, weight, bias, output
        x2 = self.conv3d(x2)  # input, weight, bias, output
        x0 = x0[:, 10:45]  # 1
        x1 = x1[:, :, 6:8, 6:9]  # 2
        x2 = x2[0:4, :, :, 6:8, 6:8]  # 3
        return x0, x1, x2


@retry_flaky_test()
@use_temporary_directory
def test_torch_delete_slice_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = Slice_Tensor_Model().to(torch_device).eval()
    example_inputs = (
        torch.rand(8, 16).to(torch_device),
        torch.rand(8, 3, 10, 10).to(torch_device),
        torch.rand(8, 3, 16, 16, 16).to(torch_device),
    )
    quant_inputs = {"x" + str(index): value for index, value in enumerate(example_inputs)}

    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    assert fx_contains_op_num(graph_model, is_slice_node) in [11, 6]  # NOTE in torch2.9 this bug fixed
    opt_model = opt_befor_qt.ConvertDeleteRedundantSliceQOPass()(graph_model)
    assert fx_contains_op_num(graph_model, is_slice_node) == 6
    out1 = opt_model.eval()(*example_inputs)
    assert all([torch.allclose(x[0], x[1]) for x in zip(fp_out, out1, strict=False)])

    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()
        quantized_model = quantizer.quantize_model(graph_model, [quant_inputs])
        out_2 = quantized_model.eval()(*example_inputs)
        if each_quant_config == emp_quant_config:
            assert all([torch.allclose(x[0], x[1]) for x in zip(fp_out, out_2, strict=False)])
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [18, 0]
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        opt_graph_module(*example_inputs)
        assert fx_contains_op_num(opt_graph_module, is_slice_node) == 6
        out_onnx_path = tmpdir + "/slice.onnx"
        torch.onnx.export(opt_graph_module, example_inputs, out_onnx_path, dynamo=False)
        assert onnx_contains_op_num(out_onnx_path, "Slice") == 6
    torch.cuda.empty_cache()


"""
Align with nndct IPU quant model optimize strategy
func: replace_sigmoid_with_hsigmoid
"""


class Tiny_Sigmoid_2_Hardsigmoid_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 48)
        self.sigmoid = nn.Sigmoid()
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        x = self.linear(x)  # input, weight, bias
        x = self.sigmoid(x)  # o,
        x = self.hardsigmoid(x)  # o
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_sigmoid_2_hardsigmoid_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = Tiny_Sigmoid_2_Hardsigmoid_Model().to(torch_device).eval()
    example_inputs = (torch.rand(8, 16).to(torch_device),)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    assert fx_contains_op_num(graph_model, is_sigmoid_node) == 1
    opt_model = opt_befor_qt.ConvertSigmoid2HardSigmoidQOPass()(graph_model)
    assert fx_contains_op_num(opt_model, is_sigmoid_node) == 0
    assert fx_contains_op_num(opt_model, is_hardsigmoid_node) == 2

    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [unify_quantizer, fx_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [5, 0]
                freeze_graph_module = quantizer.freeze(quantized_model.eval())
                freeze_graph_module(*example_inputs)
                assert fx_contains_op_num(freeze_graph_module, is_hardsigmoid_node) == 2
                out_onnx_path = tmpdir + "/hardsigmoid.onnx"
                torch.onnx.export(freeze_graph_module, example_inputs, out_onnx_path, dynamo=False)
                assert onnx_contains_op_num(out_onnx_path, "HardSigmoid") == 2
    torch.cuda.empty_cache()


"""
Align with nndct IPU quant model optimizes strategy
func: replace_sigmoid_with_hsigmoid
"""


class Tiny_Silu_2_Hardsigmoid_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(16, 48)
        self.silu = nn.SiLU()
        self.hardswish = nn.Hardswish()

    def forward(self, x):
        x = self.linear(x)  # input, weight, bias
        x = self.silu(x)  # hardsigmoid out, out
        x = self.hardswish(x)  # o
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_silu_2_hardswish_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = Tiny_Silu_2_Hardsigmoid_Model().to(torch_device).eval()
    example_inputs = (torch.rand(8, 16).to(torch_device),)
    float_model(example_inputs[0])
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    # graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    assert fx_contains_op_num(graph_model, is_silu_node) == 1
    opt_model = opt_befor_qt.ConvertSilu2HardswishQOPass()(graph_model)
    assert fx_contains_op_num(opt_model, is_hardswish_node) == 2
    assert fx_contains_op_num(opt_model, is_silu_node) == 0

    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [unify_quantizer, fx_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [6, 0]
                freeze_graph_module = quantizer.freeze(quantized_model.eval())
                freeze_graph_module(*example_inputs)
                assert fx_contains_op_num(freeze_graph_module, is_hardswish_node) == 1
                assert fx_contains_op_num(freeze_graph_module, is_hardsigmoid_node) == 1
                out_onnx_path = tmpdir + "/hardswish.onnx"
                torch.onnx.export(freeze_graph_module, example_inputs, out_onnx_path, dynamo=False)
                assert onnx_contains_op_num(out_onnx_path, "HardSwish") == 1
                assert onnx_contains_op_num(out_onnx_path, "HardSigmoid") == 1
                assert onnx_contains_op_num(out_onnx_path, "Mul") in [2, 1]
    torch.cuda.empty_cache()


"""
Align with nndct NPU quant model optimizes strategy
func: replace_AdaptiveAvgPool2d -> QuantAdaptiveAvgPool2d
"""


class Tiny_AdaptiveAvgPool2d_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 16, kernel_size=1, stride=1)
        self.adaptiveavgpool2d = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, x3, x4, x5, x6):
        x1 = self.conv2d(x1)  # 3 * 3 size  # in, out, w, b,
        x1 = self.adaptiveavgpool2d(x1)  # o

        x2 = self.conv2d(x2)  # 5 * 5 size  # in, out, w, b,
        x2 = self.adaptiveavgpool2d(x2)  # o

        x3 = self.conv2d(x3)  # 6 * 6 size  # in, out, w, b,
        x3 = self.adaptiveavgpool2d(x3)  # o

        x4 = self.conv2d(x4)  # 7 * 7 size  # in, out, w, b,
        x4 = self.adaptiveavgpool2d(x4)  # o

        x5 = self.conv2d(x5)  # 14 * 14 size  # in, out, w, b,
        x5 = self.adaptiveavgpool2d(x5)  # o

        x6 = self.conv2d(x6)  # 15 * 15 size  # in, out, w, b,
        x6 = self.adaptiveavgpool2d(x6)  # o
        return x1, x2, x3, x4, x5, x6


@retry_flaky_test()
@use_temporary_directory
def test_torch_adaptiveavgpool2d_2_qtadaptiveavgpool2d_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = Tiny_AdaptiveAvgPool2d_Model().to(torch_device).eval()

    example_inputs = (
        torch.rand(1, 3, 3, 3).to(torch_device),
        torch.rand(1, 3, 5, 5).to(torch_device),
        torch.rand(1, 3, 6, 6).to(torch_device),
        torch.rand(1, 3, 7, 7).to(torch_device),
        torch.rand(1, 3, 14, 14).to(torch_device),
        torch.rand(1, 3, 15, 15).to(torch_device),
    )
    quant_inputs = {"x" + str(index + 1): value for index, value in enumerate(example_inputs)}
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()

    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    gp_out = graph_model(*example_inputs)
    assert all([torch.allclose(x[0], x[1]) for x in zip(fp_out, gp_out, strict=False)]) is True
    opt_graph = opt_befor_qt.ConvertAdaptiveavgpool2d2Quantadaptiveavgpool2DQOPass()(graph_model)
    assert fx_contain_module_num(opt_graph, QuantAdaptiveAvgPool2d) == 6
    opt_graph(*example_inputs)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [unify_quantizer, fx_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [quant_inputs]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [quant_inputs]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [30, 0]
                freeze_graph_module = quantizer.freeze(quantized_model.eval())
                freeze_graph_module(*example_inputs)
                assert fx_contain_module_num(freeze_graph_module, QuantAdaptiveAvgPool2d) == 6
                out_onnx_path = tmpdir + "/adaptivepooling2d.onnx"
                torch.onnx.export(freeze_graph_module, example_inputs, out_onnx_path, dynamo=False)
                assert onnx_contains_op_num(out_onnx_path, "GlobalAveragePool") == 6
    torch.cuda.empty_cache()


"""
pre quantize optim strategy:
Before quantization, conver AveragePool to QuantQveragePool
"""


class TinyAveragePool2QuantQveragePool(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1, bias=True)
        """
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        """
        self.avgpool1 = nn.AvgPool2d(3, stride=2)
        self.avgpool2 = nn.AvgPool2d(5, stride=2)
        self.avgpool3 = nn.AvgPool2d((3, 6), stride=(3, 5), ceil_mode=True)
        self.avgpool4 = nn.AvgPool2d(7)
        self.avgpool5 = nn.AvgPool2d(14)
        self.avgpool6 = nn.AvgPool2d(15)

    def forward(self, x):
        x = self.conv(x)  # input, w, bias, output
        x0 = self.avgpool1(x)  # out,
        x1 = self.avgpool2(x)  # out,
        x2 = self.avgpool3(x)  # out,
        x3 = self.avgpool4(x)  # out,
        x4 = self.avgpool5(x)  # out,
        x5 = self.avgpool6(x)  # out,
        return x0, x1, x2, x3, x4, x5


@retry_flaky_test()
@use_temporary_directory
def test_torch_avgpool2d_2_qtavgpool2d_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyAveragePool2QuantQveragePool().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert all([torch.allclose(x[0], x[1]) for x in zip(fp_out, gp_out, strict=False)]) is True
    opt_graph = opt_befor_qt.ConverAvgpool2d2QuantAvgPool2dQOPass()(graph_model)
    assert fx_contain_module_num(opt_graph, QuantAvgPool2d) == 6
    opt_graph(*example_inputs)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [unify_quantizer, fx_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [10, 0]
                freeze_graph_module = quantizer.freeze(quantized_model.eval())
                freeze_graph_module(*example_inputs)
                assert fx_contain_module_num(freeze_graph_module, QuantAvgPool2d) == 6
                out_onnx_path = tmpdir + "/avgpooling2d.onnx"
                torch.onnx.export(freeze_graph_module, example_inputs, out_onnx_path, dynamo=False)
                assert onnx_contains_op_num(out_onnx_path, "AveragePool") == 6
                assert onnx_contains_op_num(out_onnx_path, "Mul") == 6
    torch.cuda.empty_cache()


"""
pre quantize optim strategy:
Before quantization, conver LeakyReLU to NPU version QuantLeakyReLU
"""


class TinyLeakyReluModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1, bias=True)
        self.leakyrelu = nn.LeakyReLU()

    def forward(self, x):
        x = self.conv(x)  # input, w, bias
        x = self.leakyrelu(x)  # out,
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_leakyrelu_2_qtleakyrelu_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyLeakyReluModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    opt_graph = opt_befor_qt.ConvertLeakyReLu2QuantLeakyReLuQOPass()(graph_model)
    assert fx_contain_module_num(opt_graph, QuantLeakyReLU) == 1
    opt_graph(*example_inputs)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [unify_quantizer, fx_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [4, 0]
                freeze_graph_module = quantizer.freeze(quantized_model.eval())
                freeze_graph_module(*example_inputs)
                assert fx_contain_module_num(freeze_graph_module, QuantLeakyReLU) == 1
                out_onnx_path = tmpdir + "/leaky_relu.onnx"
                torch.onnx.export(freeze_graph_module, example_inputs, out_onnx_path, dynamo=False)
                assert onnx_contains_op_num(out_onnx_path, "LeakyRelu") == 1
                assert onnx_contains_op_num(out_onnx_path, "Conv") == 1
    torch.cuda.empty_cache()


"""
post quant optimize strategy
After quantization, ApplyConstrain2ConcatQOPass
each input's scale (quantized tensor) of concat must be same
"""


class TinyConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(6, 16, 1)

    def forward(self, x):
        x1 = x * 100
        x2 = x
        x = torch.concat([x1, x2], dim=1)
        x = self.conv(x)
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_postquant_concat_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyConcatModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    opt_graph = opt_after_qt_pow2_s.ApplyConstrain2ConcatQOPass()(graph_model)  # skip
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [7, 0]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    scale_0 = quantized_model.fake_quantizer_0.scale.item()
                    scale_2 = quantized_model.fake_quantizer_2.scale.item()
                    scale_3 = quantized_model.fake_quantizer_3.scale.item()
                    min_scale = min(min(scale_0, scale_2), scale_3)
                    quantized_model.fake_quantizer_3.scale.fill_(min_scale)
                    assert len({scale_0, scale_2, scale_3}) == 2
                    opt_graph = opt_after_qt_pow2_s.ApplyConstrain2ConcatQOPass()(quantized_model)
                    new_scale_0 = opt_graph.fake_quantizer_0.scale.item()
                    new_scale_2 = opt_graph.fake_quantizer_2.scale.item()
                    new_scale_3 = opt_graph.fake_quantizer_3.scale.item()
                    assert len({new_scale_0, new_scale_2, new_scale_3}) == 1
                    opt_graph.fake_quantizer_0 = nn.Identity()
                    opt_graph = opt_after_qt_pow2_s.ApplyConstrain2ConcatQOPass()(opt_graph)
    torch.cuda.empty_cache()


"""
post quant optimize strategy
After quantization, AlignSingleInOutOpScaleQOPass
each input's scale should be same with output's scale.
"""


class TinySingleInOutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_align_single_in_out_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinySingleInOutModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    opt_graph = opt_after_qt_pow2_s.AlignSingleInOutOpScaleQOPass([torch.ops.aten.sigmoid.default])(graph_model)  # skip
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_pow2_s.AlignSingleInOutOpScaleQOPass([torch.ops.aten.hardsigmoid.default])
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [5, 0]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    scale_1 = quantized_model.fake_quantizer_1.scale.item()
                    quantized_model.fake_quantizer_2.scale.fill_(scale_1 * 2)
                    scale_2 = quantized_model.fake_quantizer_2.scale.item()
                    assert len({scale_1, scale_2}) == 2
                    opt_graph = opt_module(quantized_model)
                    new_scale_1 = opt_graph.fake_quantizer_1.scale.item()
                    new_scale_2 = opt_graph.fake_quantizer_2.scale.item()
                    assert len({new_scale_1, new_scale_2}) == 1
                    opt_graph.fake_quantizer_1.scale.fill_(new_scale_2 * 2)
                    new_scale_1 = opt_graph.fake_quantizer_1.scale.item()
                    new_scale_2 = opt_graph.fake_quantizer_2.scale.item()
                    assert len({scale_1, scale_2}) == 2
                    opt_graph = opt_module(opt_graph)
                    new_scale_1 = opt_graph.fake_quantizer_1.scale.item()
                    new_scale_2 = opt_graph.fake_quantizer_2.scale.item()
                    assert len({new_scale_1, new_scale_2}) == 1
                    opt_graph.fake_quantizer_1 = nn.Identity()
                    opt_graph = opt_module(opt_graph)
    torch.cuda.empty_cache()


"""
post quant optimize strategy
After quantization, AlignSingleInOutModuleScaleQOPass
each input's scale should be same with output's scale.
# NOTE aten.pool -> Quant.pool is pre quant strategies
"""


class TinySingleInOutAvgpool2dModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 16, 1)
        self.pool2d = nn.AvgPool2d(3)
        self.relu = nn.ReLU()
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv2d(x)
        x = self.pool2d(x)
        x = self.relu(x)
        x = self.adaptive_pool(x)
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_align_single_in_out_module_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinySingleInOutAvgpool2dModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_pow2_s.AlignSingleInOutModuleScaleQOPass((QuantAvgPool2d, QuantAdaptiveAvgPool2d))
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [7, 0]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    opt_graph = opt_module(quantized_model)
                    new_scale_1_1 = opt_graph.fake_quantizer_1.scale.item()
                    new_scale_2_1 = opt_graph.fake_quantizer_2.scale.item()
                    assert len({new_scale_1_1, new_scale_2_1}) == 1
                    opt_graph.fake_quantizer_2.scale.fill_(new_scale_2_1 * 2)
                    new_scale_1_2 = opt_graph.fake_quantizer_1.scale.item()
                    new_scale_2_2 = opt_graph.fake_quantizer_2.scale.item()
                    assert len({new_scale_1_2, new_scale_2_2}) == 2
                    opt_graph = opt_module(opt_graph)
                    new_scale_1_3 = opt_graph.fake_quantizer_1.scale.item()
                    new_scale_2_3 = opt_graph.fake_quantizer_2.scale.item()
                    assert len({new_scale_1_3, new_scale_2_3}) == 1
                    opt_graph.fake_quantizer_1 = nn.Identity()
                    opt_graph = opt_module(opt_graph)
    torch.cuda.empty_cache()


"""
post quant optimize strategy
After quantization, AdjustShiftReadQOPass
0 <= max(input_quant_pos) - min(ipos) <= 7
"""


class TinyAddModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.relu = nn.ReLU()
        self.relu6 = nn.ReLU6()

    def forward(self, x):
        x = self.conv(x)
        x1 = self.relu(x)
        x2 = self.relu6(x)
        x = x1 + x2
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_adjust_shift_read_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyAddModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_pow2_s.AdjustShiftReadQOPass()
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [7, 0]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    quantized_model.fake_quantizer_2.scale.fill_(1 / 2**1)
                    quantized_model.fake_quantizer_3.scale.fill_(1 / 2**10)
                    scale_1 = quantized_model.fake_quantizer_2.scale.item()
                    scale_3 = quantized_model.fake_quantizer_3.scale.item()
                    assert abs(math.log2(scale_1) - math.log2(scale_3)) == 9
                    opt_graph = opt_module(quantized_model)
                    new_scale_1 = opt_graph.fake_quantizer_2.scale.item()
                    new_scale_3 = opt_graph.fake_quantizer_3.scale.item()
                    assert abs(math.log2(new_scale_1) - math.log2(new_scale_3)) == 7
                    opt_graph.fake_quantizer_1 = nn.Identity()
                    opt_graph = opt_module(opt_graph)
    torch.cuda.empty_cache()


"""
post quant optimize strategy
After quantization, AdjustShiftWriteQOPass
For Add:
    shift_write = min(ipos) - opos
    NPU compiler constraints of shift_write:
    1. -7 <= shift_write <= 25

    For Mul:
    shift_write = sum(ipos) - opos
    NPU compiler constraints of shift_write:
    1. 0 <= shift_write <= 32
"""


class TinyAddMULAdjustWriteModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.relu = nn.ReLU()
        self.avgpool = nn.AvgPool2d(3, 3)

    def forward(self, x):
        x = self.conv(x)  # input, weight, bias
        x = self.relu(x)  # out
        x = x + 100  # scale, out
        x = self.avgpool(x)  # out
        x = x * 10  # scale, out
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_adjust_shift_write_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyAddMULAdjustWriteModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_pow2_s.AdjustShiftWriteQOPass()
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [9, 0]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    # process  add case
                    quantized_model.fake_quantizer_1.scale.fill_(1 / 2**4)
                    quantized_model.fake_quantizer_2.scale.fill_(1 / 2**4)
                    quantized_model.fake_quantizer_3.scale.fill_(1 / 2 ** (8 + 4))
                    scale_1 = quantized_model.fake_quantizer_1.scale.item()
                    scale_3 = quantized_model.fake_quantizer_3.scale.item()
                    assert abs(math.log2(scale_1) - math.log2(scale_3)) == 8
                    opt_graph = opt_module(quantized_model)
                    new_scale_1 = opt_graph.fake_quantizer_1.scale.item()
                    new_scale_3 = opt_graph.fake_quantizer_3.scale.item()
                    assert abs(math.log2(new_scale_1) - math.log2(new_scale_3)) == 7
                    # process  mul case
                    quantized_model.fake_quantizer_4.scale.fill_(1 / 2**4)
                    quantized_model.fake_quantizer_5.scale.fill_(1 / 2**4)
                    quantized_model.fake_quantizer_6.scale.fill_(1 / 2 ** (9))
                    scale_4 = quantized_model.fake_quantizer_4.scale.item()
                    scale_5 = quantized_model.fake_quantizer_5.scale.item()
                    scale_6 = quantized_model.fake_quantizer_6.scale.item()
                    assert abs(math.log2(scale_4) + math.log2(scale_5) - math.log2(scale_6)) == 1
                    opt_graph = opt_module(quantized_model)
                    new_scale_4 = opt_graph.fake_quantizer_4.scale.item()
                    new_scale_5 = opt_graph.fake_quantizer_5.scale.item()
                    new_scale_6 = opt_graph.fake_quantizer_6.scale.item()
                    assert abs(math.log2(new_scale_4) + math.log2(new_scale_5) - math.log2(new_scale_6)) == 0
                    opt_graph.fake_quantizer_1.scale = torch.tensor([1.0, 2.0])
                    opt_graph = opt_module(opt_graph)
                    opt_graph.fake_quantizer_1 = nn.Identity()
                    opt_graph = opt_module(opt_graph)
    torch.cuda.empty_cache()


class TinyAdjustShiftCutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 8, 1)
        self.pool = nn.AvgPool2d(3, 3)
        proj = torch.linspace(0, 7, 8).reshape([1, 8, 1, 1])
        self.register_buffer("proj_conv", proj, persistent=False)

    def forward(self, x):
        x = self.conv(x)  # input, weight, bias
        x = self.relu(x)  # out
        x = self.conv1(x)  # weight, bias, out
        x = self.pool(x)  # out
        x = torch.nn.functional.conv2d(x, weight=self.proj_conv)  # weight, out
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_adjust_shift_cut_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyAdjustShiftCutModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_pow2_s.AdjustShiftCutQOPass()
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [10, 0]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    # Norm Path
                    quantized_model.fake_quantizer_3.scale.fill_(1 / 2**4)
                    quantized_model.fake_quantizer_4.scale.fill_(1 / 2**4)
                    quantized_model.fake_quantizer_5.scale.fill_(1 / 2 ** (9))
                    scale_4_old = quantized_model.fake_quantizer_4.scale.item()
                    opt_graph = opt_module(quantized_model)
                    scale_4_new = opt_graph.fake_quantizer_4.scale.item()
                    assert abs(math.log2(scale_4_old)) == 4
                    assert abs(math.log2(scale_4_new)) == 5
                    # Output quantizer is missing
                    opt_graph.fake_quantizer_5 = nn.Identity()
                    opt_graph = opt_module(opt_graph)
                    # QuantConv's weight quantizer is missing
                    opt_graph.conv2d_quantized_module._weight_quantizer = nn.Identity()
                    opt_graph = opt_module(opt_graph)
                    # ops.conv's weight quantizer is missing
                    opt_graph.fake_quantizer_4 = nn.Identity()
                    opt_graph = opt_module(opt_graph)
    torch.cuda.empty_cache()


class TinyAdjustShiftBiasModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(8, 8, 1)
        self.pool = nn.AvgPool2d(3, 3)
        proj = torch.linspace(0, 7, 8).reshape([1, 8, 1, 1])
        bias = torch.tensor([1.0], dtype=torch.float32)
        self.register_buffer("proj_conv", proj, persistent=False)
        self.register_buffer("bias", bias, persistent=False)

    def forward(self, x):
        x = self.conv(x)  # input, weight, bias
        x = self.relu(x)  # out
        x = self.conv1(x)  # weight, bias, out
        x = self.pool(x)  # out
        x = torch.nn.functional.conv2d(x, weight=self.proj_conv, bias=self.bias)  # weight, bias, out
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_adjust_shift_bias_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyAdjustShiftBiasModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_pow2_s.AdjustShiftBiasQOPass()
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [11, 0]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    # Norm Path
                    quantized_model.fake_quantizer_3.scale.fill_(1 / 2**8)
                    quantized_model.fake_quantizer_4.scale.fill_(1 / 2**8)
                    quantized_model.fake_quantizer_5.scale.fill_(1 / 2**18)
                    quantized_model.fake_quantizer_6.scale.fill_(1 / 2**1)
                    scale_5_old = quantized_model.fake_quantizer_5.scale.item()
                    opt_graph = opt_module(quantized_model)
                    scale_5_new = opt_graph.fake_quantizer_5.scale.item()
                    assert abs(math.log2(scale_5_old)) == 18
                    assert abs(math.log2(scale_5_new)) == 17
                    # scale is not a scale
                    opt_graph.fake_quantizer_5.scale = torch.tensor([1.0, 2.0])
                    opt_graph = opt_module(opt_graph)
                    # QuantConv's bias quantizer is missing
                    opt_graph.conv2d_quantized_module._bias_quantizer = nn.Identity()
                    opt_graph = opt_module(opt_graph)
                    # ops.conv's bias quantizer is missing
                    opt_graph.fake_quantizer_5 = nn.Identity()
                    opt_graph = opt_module(opt_graph)
    torch.cuda.empty_cache()


class TinyAdjustHardSigmoidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.pool = nn.AvgPool2d(3, 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)  # input, weight, bias, out
        x = self.pool(x)  # out
        x = self.sigmoid(x)  # out
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_adjust_hard_sigmoid_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyAdjustHardSigmoidModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_pow2_s.AdjustHardSigmoidQOPass()
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [6, 0]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    # Norm Path
                    quantized_model.fake_quantizer_2.scale.fill_(1 / 2**16)
                    quantized_model.fake_quantizer_3.scale.fill_(1 / 2**6)
                    opt_graph = opt_module(quantized_model)
                    scale_2_new = opt_graph.fake_quantizer_2.scale.item()
                    scale_3_new = opt_graph.fake_quantizer_3.scale.item()
                    assert abs(math.log2(scale_2_new)) == 15
                    assert abs(math.log2(scale_3_new)) == 7
                    # scale is not a scale
                    opt_graph.fake_quantizer_3.scale = torch.tensor([1.0, 2.0])
                    opt_graph = opt_module(opt_graph)
                    # output quantizer is missing
                    opt_graph.fake_quantizer_3 = nn.Identity()
                    opt_graph = opt_module(opt_graph)
    torch.cuda.empty_cache()


class TinyAdjustShiftSwishModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AvgPool2d(3, 3)

    def forward(self, x):
        x = self.conv(x)  # input, weight, bias, out
        x1 = self.sigmoid(x)  # output
        x = x1 * x  # out
        x = self.avgpool(x)  # out
        x2 = self.sigmoid(x)  # out
        x = torch.nn.functional.relu(x)  # out
        x = x2 * x  # out
        x = self.avgpool(x)  # out
        x3 = self.sigmoid(x)  # out
        x3 = torch.nn.functional.relu(x3)  # out
        x = x * x3  # out
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_adjust_shift_swishd_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyAdjustShiftSwishModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_pow2_s.AdjustShiftSwishQOPass()
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [14, 0]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    # Norm Path
                    quantized_model.fake_quantizer_1.scale.fill_(1 / 2**4)
                    quantized_model.fake_quantizer_2.scale.fill_(1 / 2**4)
                    quantized_model.fake_quantizer_3.scale.fill_(1 / 2**9)
                    opt_graph = opt_module(quantized_model)
                    onnx_dir = tmpdir + "/module_shift_swish.onnx"
                    torch.onnx.export(opt_graph, example_inputs, onnx_dir, dynamo=False)
                    assert onnx_contains_op_num(onnx_dir, "QuantizeLinear") == 14
                    assert onnx_contains_op_num(onnx_dir, "Mul") == (3 + 2)  # 2 from adgpool adjust
                    assert onnx_contains_op_num(onnx_dir, "Conv") == 1
                    scale_3_new = opt_graph.fake_quantizer_3.scale.item()
                    assert abs(math.log2(scale_3_new)) == 8
                    # scale is not a scale
                    opt_graph.fake_quantizer_1.scale = torch.tensor([1.0, 2.0])
                    opt_graph = opt_module(opt_graph)
                    # output quantizer is missing
                    opt_graph.fake_quantizer_2 = nn.Identity()
                    opt_graph = opt_module(opt_graph)
    torch.cuda.empty_cache()


"""
post quant optimize strategy
After quantization, ConvertHardSigmoidDpuVersionQOPass
Convert HardSigmoid to DPU version.
"""


class TinySigmoidModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.sigmoid(x)
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_convert_hard_sigmoid_dpu_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinySigmoidModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    opt_graph = opt_after_qt_pow2_s.ConvertHardSigmoidDpuVersionQOPass()(graph_model)  # skip
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_pow2_s.ConvertHardSigmoidDpuVersionQOPass()
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [5, 0]
                opt_graph = opt_module(quantized_model)
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    assert fx_contains_op_num(opt_graph, is_mul_node) == 1
                else:
                    assert fx_contains_op_num(opt_graph, is_mul_node) == 0
                onnx_dir = tmpdir + "/module_Dpu_hardsigmoid.onnx"
                torch.onnx.export(opt_graph, example_inputs, onnx_dir, dynamo=False)
                assert onnx_contains_op_num(onnx_dir, "Conv") == 1
                assert onnx_contains_op_num(onnx_dir, "Mul") in [1, 0]
    torch.cuda.empty_cache()


"""
Befor quant strategy: silu -> x * sigmoid(x)
"""


class TinySiLUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.relu = nn.ReLU()
        self.silu = nn.SiLU()

    def forward(self, x):
        x = self.conv(x)  # input, weight, bias,
        x = self.relu(x)  # out
        x = self.silu(x)  # sigmoid_out, out
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_convert_silu_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinySiLUModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    opt_graph_model = replace_silu_node(graph_model)
    opt_out = opt_graph_model(*example_inputs)
    assert torch.allclose(gp_out, opt_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [unify_quantizer, fx_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                quantized_model.train()
                quantized_model.eval()
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [6, 0]
                onnx_dir = tmpdir + "/module_x_mul_sigmoid.onnx"
                torch.onnx.export(quantized_model, example_inputs, onnx_dir, dynamo=False)
                assert onnx_contains_op_num(onnx_dir, "QuantizeLinear") in [6, 0]
                assert onnx_contains_op_num(onnx_dir, "HardSigmoid") == 1
                assert onnx_contains_op_num(onnx_dir, "Mul") == 1
    torch.cuda.empty_cache()


# ============================test post quant delete dropout layer strategy=================
class TinyDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 32, 3, bias=True, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout()
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.adaptive_avg_pool2d(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.linear(x)
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_delete_dropout_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyDropoutModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    opt_model_func = RemoveDropoutNode()
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [quant_W_A_8_B_32_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [unify_quantizer, fx_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                quant_out1 = quantized_model(example_inputs[0])
                assert fx_contains_op_num(quantized_model, is_dropout_node) == 1
                quantized_model = opt_model_func.apply(quantized_model)
                quant_out2 = quantized_model(example_inputs[0])
                assert fx_contains_op_num(quantized_model, is_dropout_node) == 0
                assert torch.allclose(quant_out1, quant_out2)
    torch.cuda.empty_cache()


# ============================test_torch_bias_int32_strategy=================
# conddition:
#    1. if bias -> int32 format quant -> then -> bias_scale = weight_scale * activation_scale

INT8_PER_TENSOR_SPEC_W_A = QTensorConfig(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorPowOf2MinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)
INT8_PER_TENSOR_SPEC_BIAS = QTensorConfig(
    dtype=Dtype.int32,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorPowOf2MinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)
quant_tensor_W_A_8_B_32_config = QLayerConfig(
    input_tensors=INT8_PER_TENSOR_SPEC_W_A,
    output_tensors=INT8_PER_TENSOR_SPEC_W_A,
    weight=INT8_PER_TENSOR_SPEC_W_A,
    bias=INT8_PER_TENSOR_SPEC_BIAS,
)
quant_W_A_8_B_32_config = QConfig(
    global_quant_config=quant_tensor_W_A_8_B_32_config, quant_mode=QuantizationMode.fx_graph_mode
)


class TinyCONVModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)

    def forward(self, x):
        x = self.conv(x)  # input, weight, bias,
        return x


@retry_flaky_test()
@use_temporary_directory
def test_torch_bias_int32_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyCONVModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    opt_model_func = AdjustBiasScaleQOPass()
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [quant_W_A_8_B_32_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [unify_quantizer, fx_quantizer]:
            graph_model_1 = torch.export.export_for_training(float_model, example_inputs).module()
            input_models = [graph_model_1]
            if isinstance(quantizer, FxGraphQuantizer):
                input_models.append(float_model)
            for each_format_model in input_models:
                input_args = (
                    {"model": each_format_model, "args": example_inputs, "calibdata": [example_inputs[0]]}
                    if isinstance(quantizer, FxGraphQuantizer)
                    else {"model": each_format_model, "dataloader": [example_inputs[0]]}
                )
                quantized_model = quantizer.quantize_model(**input_args)
                quantized_model(example_inputs[0])
                copyed_quantized_model = copy.deepcopy(quantized_model)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [4, 0]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    # Norm Path
                    act_quantizer = quantized_model.fake_quantizer_0
                    conv_w_quantizer = quantized_model.conv2d_quantized_module._weight_quantizer
                    conv_b_quantizer = quantized_model.conv2d_quantized_module._bias_quantizer
                    assert (
                        act_quantizer.scale.numel() == 1
                        and conv_w_quantizer.scale.numel() == 1
                        and conv_b_quantizer.scale.numel() == 1
                    )
                    assert act_quantizer.scale.item() * conv_w_quantizer.scale.item() == conv_b_quantizer.scale.item()
                    conv_b_quantizer.scale.fill_(act_quantizer.scale.item())
                    opt_graph = opt_model_func(quantized_model)
                    assert (
                        act_quantizer.scale.detach().clone() * conv_w_quantizer.scale.detach().clone()
                        == conv_b_quantizer.scale.detach().clone()
                    )
                    # change to int8
                    opt_graph.conv2d_quantized_module._bias_quantizer.dtype = Dtype.int8
                    opt_graph = opt_model_func(opt_graph)
                    opt_graph.conv2d_quantized_module._bias_quantizer.dtype = Dtype.int32
                    # act quantizer is not a single scale
                    org_act_scale = opt_graph.fake_quantizer_0.scale.detach().item()
                    opt_graph.fake_quantizer_0.scale = torch.tensor([1.0, 2.0])
                    opt_graph = opt_model_func(opt_graph)
                    opt_graph.fake_quantizer_0.scale = torch.tensor(org_act_scale)
                    # weight & bias quantizer perchannel
                    quantized_model.conv2d_quantized_module._weight_quantizer.scale = torch.tensor([1.0, 2.0, 3.0])
                    quantized_model.conv2d_quantized_module._bias_quantizer.scale = torch.tensor([1.0, 2.0, 3.0])
                    opt_graph.conv2d_quantized_module._bias_quantizer.qscheme = QSchemeType.per_channel
                    opt_graph.conv2d_quantized_module._weight_quantizer.qscheme = QSchemeType.per_channel
                    opt_graph = opt_model_func(opt_graph)
                    # act & weight quantizer is missing
                    opt_graph.fake_quantizer_0 = nn.Identity()
                    opt_graph.conv2d_quantized_module._weight_quantizer = nn.Identity()
                    opt_graph = opt_model_func(opt_graph)
                onnx_dir = tmpdir + "/module_bias_int32.onnx"
                # freeze_model = quantizer.freeze(copyed_quantized_model.eval())
                torch.onnx.export(copyed_quantized_model.eval(), example_inputs, onnx_dir, dynamo=False)
                assert onnx_contains_op_num(onnx_dir, "QuantizeLinear") in [4, 0]
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    test_use_over_once_module_optim()
    test_torch_module_used_over_once_optim_strategy()
    test_torch_sg_bn2d_to_conv2d_optim_strategy()
    test_torch_clip_2_relu_optim_strategy()
    test_mean_2_pooling_strategy()
    test_torch_convert_split_2_slice_strategy()
    test_torch_fold_bn_after_concat_strategy()
    test_not_fold_bn_after_concat_strategy()
    test_torch_layerNorm_strategy()
    test_torch_fuse_gelu_strategy()
    test_torch_split_large_kernel_pool_strategy()
    test_torch_delete_slice_strategy()
    test_torch_sigmoid_2_hardsigmoid_strategy()
    test_torch_silu_2_hardswish_strategy()
    test_torch_adaptiveavgpool2d_2_qtadaptiveavgpool2d_strategy()
    test_torch_avgpool2d_2_qtavgpool2d_strategy()
    test_torch_leakyrelu_2_qtleakyrelu_strategy()
    test_torch_postquant_concat_strategy()
    test_torch_align_single_in_out_strategy()
    test_torch_align_single_in_out_module_strategy()
    test_torch_adjust_shift_read_strategy()
    test_torch_adjust_shift_write_strategy()
    test_torch_adjust_shift_cut_strategy()
    test_torch_adjust_shift_bias_strategy()
    test_torch_adjust_hard_sigmoid_strategy()
    test_torch_adjust_shift_swishd_strategy()
    test_torch_convert_hard_sigmoid_dpu_strategy()
    test_torch_convert_silu_strategy()
    test_torch_bias_int32_strategy()
    test_torch_delete_dropout_strategy()
    torch.cuda.empty_cache()
