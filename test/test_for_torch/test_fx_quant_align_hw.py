#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys
sys.path.append("..")
import torch
import torch.nn as nn
# from torch.fx import Node
from torch._export import capture_pre_autograd_graph
from quark.torch import ModelQuantizer
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantizedConvBatchNorm2d
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d, QuantConvTranspose2d
from quark.torch.quantization.graph.processor.processor import _pre_quant_optimize
from quark.torch.quantization.config.config import QuantizationSpec, QuantizationConfig, Config
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType, QuantizationMode
from quark.torch.quantization.graph.torch_utils import is_clip_node, is_relu_act_node, is_mean_node, is_adaptive_avg_pool_node
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
import quark.torch.quantization.graph.optimization.post_quant.opt_pass_after_quant as opt_after_qt
from quark.shares.utils.testing_utils import torch_device, use_temporary_directory
from quark.shares.utils.log import ScreenLogger
logger = ScreenLogger(__name__)

TEST_TOPIC = "torch FX graph mode quantization, align with hw deploy need\n"

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

'''
=============== Test if one module used over once ===============
if one module used over once, like the below example
(conv2d_1, conv2d_2, transposconv2d, linear_1)

Although, these modules will have one copy in torch module. In hardware (e.g IPU),
for better deployments, we will regard every convolutional operation in the forward path as a different conv, even though they share the same weight/bias.
'''
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


def test_use_over_once_module_optim():
    '''
    In optmized Graph, will has two QuantizedConvBatchNorm2d
    '''
    float_model1 = TinyShareConvbnModel().to(torch_device).eval()
    float_model2 = TinyShareConvModel().to(torch_device).eval()
    float_model3 = TinyShareConvTransposeModel().to(torch_device).eval()
    float_model4 = TinyShareLinearModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 32, 32).to(torch_device), torch.rand(1, 3, 32, 32).to(torch_device))
    for each_fp_model in [float_model1, float_model2, float_model3, float_model4]:
        each_fp_model(*example_inputs)
        # session 1
        # ========== test using hardware constrain ===============
        graph_model = capture_pre_autograd_graph(each_fp_model, example_inputs)
        # TODO for torch version >= 2.5
        # graph_model = torch.export.export_for_training(float_model, example_inputs).module()
        graph_model = _pre_quant_optimize(graph_model)
        out_fp32 = each_fp_model.eval()(*example_inputs)
        count_QuantizeModule = 0
        for module in graph_model.modules():
            if isinstance(module, (QuantizedConvBatchNorm2d, QuantConv2d, QuantConvTranspose2d, QuantLinear)):
                module.freeze_bn_stats() if isinstance(module, QuantizedConvBatchNorm2d) else None
                count_QuantizeModule += 1
        out_opt_fx_graph = graph_model(*example_inputs)
        assert count_QuantizeModule == 2, "In model {}, the QuantizedModule num should ne 2".format(each_fp_model.__class__.__name__)
        assert torch.allclose(out_fp32[0], out_opt_fx_graph[0], atol=1e-6)
        assert torch.allclose(out_fp32[1], out_opt_fx_graph[1], atol=1e-6)

        # session 2 not using hw constrain, will not copy another QuantModule instance
        graph_model = capture_pre_autograd_graph(each_fp_model, example_inputs)
        graph_model = _pre_quant_optimize(graph_model, hw_constrain=False)
        out_fp32 = each_fp_model.eval()(*example_inputs)
        count_QuantizeModule = 0
        for module in graph_model.modules():
            if isinstance(module, (QuantizedConvBatchNorm2d, QuantConv2d, QuantConvTranspose2d, QuantLinear)):
                module.freeze_bn_stats() if isinstance(module, QuantizedConvBatchNorm2d) else None
                count_QuantizeModule += 1
        out_opt_fx_graph = graph_model(*example_inputs)
        assert count_QuantizeModule == 1, "In model {}, the QuantizedModule num should ne 1".format(each_fp_model.__class__.__name__)
        assert torch.allclose(out_fp32[0], out_opt_fx_graph[0], atol=1e-6)
        assert torch.allclose(out_fp32[1], out_opt_fx_graph[1], atol=1e-6)

    logger.info(TEST_TOPIC + "[4 small model] split module used over onee to seperate modul. Passed")
    torch.cuda.empty_cache()

'''
=============== Test if one module used over once ===============
if one module used over once, like the below example
(conv2d_1, conv2d_2, transposconv2d, linear_1)

Although, these modules will have one copy in torch module. In hardware (e.g IPU),
for better deployments, we will regard every convolutional operation in the forward path as a different conv, even though they share the same weight/bias.
'''

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
        # test forward twice
        self.conv2d_2 = nn.Conv2d(32, 32, 3, bias=True)
        # test forward twice
        self.transposconv2d = nn.ConvTranspose2d(32, 32, (3, 3), bias=True)
        # test forward twice
        self.linear_1 = nn.Linear(32, 32)

        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(32, 10)

    def forward(self, x):
        x = self.bn(self.conv2d(x))  # quantizer: w, b, input -> 3
        x = self.relu(x)     # quantizer: outout -> 1
        # forward twice
        x = self.relu_1(self.bn_1(self.conv2d_1(x)))  # quantizer: w, b, output -> 3
        x = self.relu_1(self.bn_1(self.conv2d_1(x)))  # quantizer: w, b, output -> 3
        # forward twice
        x = self.conv2d_2(x)  # quantizer: w, b, output -> 3
        x = self.conv2d_2(x)  # quantizer: w, b, output -> 3
        # forward twice
        x = self.transposconv2d(x)  # quantizer: w, b, output -> 3
        x = self.transposconv2d(x)  # quantizer: w, b, output -> 3

        x = self.adaptive_avg_pool2d(x)
        x = torch.flatten(x, 1)  # quantizer: output -> 1
        # forward twice
        x = self.linear_1(x)  # quantizer: w, b, output -> 3
        x = self.linear_1(x)  # quantizer: w, b, output -> 3
        x = self.linear_1(x)  # quantizer: w, b, output -> 3
        x = self.linear_1(x)  # quantizer: w, b, output -> 3
        x = self.linear(x)   # quantizer: w, b, output -> 3
        return x

def test_torch_module_used_over_once_optim_strategy():
    '''
    test torch model that if one submodel that contain parameter used over once
    , test code will show how the fx graph model is optimized for better deployment.
    '''
    float_model = TinyShareWeightModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 112, 112).to(torch_device), )
    # session 1
    # ========== test using hardware constrain ===============
    graph_model = capture_pre_autograd_graph(float_model, example_inputs)
    # TODO for torch version >= 2.5
    # graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    graph_model = _pre_quant_optimize(graph_model)
    out_fp32 = float_model.eval()(example_inputs[0])
    for module in graph_model.modules():
        if isinstance(module, QuantizedConvBatchNorm2d):
            module.freeze_bn_stats()
    out_opt_fx_graph = graph_model(example_inputs[0])
    count_QuantizedConvBatchNorm2d, count_QuantConv2d, count_QuantConvTranspose2d, count_QuantLinear = 0, 0, 0, 0
    for module in graph_model.modules():
        if isinstance(module, QuantizedConvBatchNorm2d):
            module.freeze_bn_stats()
            count_QuantizedConvBatchNorm2d += 1
        if isinstance(module, QuantConv2d):
            count_QuantConv2d += 1
        if isinstance(module, QuantConvTranspose2d):
            count_QuantConvTranspose2d += 1
        if isinstance(module, QuantLinear):
            count_QuantLinear += 1
    assert count_QuantizedConvBatchNorm2d == 3, "QuantizedConvBatchNorm2d should be 3"
    assert count_QuantConv2d == 2, "QuantConv2d should be 2"
    assert count_QuantConvTranspose2d == 2, "QuantConvTranspose2d should be 2"
    assert count_QuantLinear == 5, "QuantLinear should be 5"
    assert torch.allclose(out_fp32, out_opt_fx_graph)
    graph_model = capture_pre_autograd_graph(float_model, example_inputs)
    quantizer = ModelQuantizer(quant_config)
    quantized_model = quantizer.quantize_model(graph_model, [torch.rand(4, 3, 112, 112).to(torch_device) for _ in range(3)])
    count_quantizer = 0
    for module in quantized_model.modules():
        if isinstance(module, ScaledFakeQuantize):
            count_quantizer += 1
    assert count_quantizer == 38, "The total quantizer in this model should be 38"
    quantized_model(example_inputs[0])
    # session 2
    # ========== test using hardware constrain ===============
    graph_model = capture_pre_autograd_graph(float_model, example_inputs)
    optimized_fx_graph_before_qt = _pre_quant_optimize(graph_model, hw_constrain=False)
    out_fp32 = float_model.eval()(example_inputs[0])
    for module in graph_model.modules():
        if isinstance(module, QuantizedConvBatchNorm2d):
            module.freeze_bn_stats()
    out_opt_fx_graph = optimized_fx_graph_before_qt(example_inputs[0])
    count_QuantizedConvBatchNorm2d, count_QuantConv2d, count_QuantConvTranspose2d, count_QuantLinear = 0, 0, 0, 0
    for module in graph_model.modules():
        if isinstance(module, QuantizedConvBatchNorm2d):
            module.freeze_bn_stats()
            count_QuantizedConvBatchNorm2d += 1
        if isinstance(module, QuantConv2d):
            count_QuantConv2d += 1
        if isinstance(module, QuantConvTranspose2d):
            count_QuantConvTranspose2d += 1
        if isinstance(module, QuantLinear):
            count_QuantLinear += 1
    assert count_QuantizedConvBatchNorm2d == 2, "QuantizedConvBatchNorm2d should be 2"
    assert count_QuantConv2d == 1, "QuantConv2d should be 1"
    assert count_QuantConvTranspose2d == 1, "QuantConvTranspose2d should be 1"
    assert count_QuantLinear == 2, "QuantLinear should be 2"
    assert torch.allclose(out_fp32, out_opt_fx_graph)
    logger.info(TEST_TOPIC + "split module that used over one to seperate module, Passed")
    torch.cuda.empty_cache()


'''
Align with onnx model optimization before quantization
'''

class TinyConvertBn2ConvModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 16, 3, bias=True, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn_1 = nn.BatchNorm2d(16)

    def forward(self, x):
        x = self.conv2d(x)   # input, w, b,
        x = self.relu(x)  # output
        #  the bn_1 -> conv
        x = self.bn_1(x)  # w, b, out
        return x

def test_torch_sg_bn2d_to_conv2d_optim_strategy():
    '''
    For better allign with hw requirements for deployment,
    transfer one single batchnorm2d to conv2d.
    '''
    float_model = TinyConvertBn2ConvModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 64, 64).to(torch_device), )
    # session 1
    # ========== test using hardware constrain ===============
    graph_model = capture_pre_autograd_graph(float_model, example_inputs)
    # TODO for torch version >= 2.5
    # graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    opt_graph_module = _pre_quant_optimize(graph_model)
    out_fp32 = float_model.eval()(example_inputs[0])
    out_fx_graph = opt_graph_module(example_inputs[0])
    count_QuantConv2d = 0
    for module in opt_graph_module.modules():
        if isinstance(module, QuantConv2d):
            count_QuantConv2d += 1
    assert count_QuantConv2d == 2
    assert torch.allclose(out_fp32, out_fx_graph, atol = 1e-07)
    graph_model = capture_pre_autograd_graph(float_model, example_inputs)
    quantizer = ModelQuantizer(quant_config)
    quantized_model = quantizer.quantize_model(graph_model, [torch.rand(4, 3, 112, 112).to(torch_device) for _ in range(3)])
    count_quantizer = 0
    for module in quantized_model.modules():
        if isinstance(module, ScaledFakeQuantize):
            count_quantizer += 1
    assert count_quantizer == 7, "The total quantizer in this model should be 7"
    quantized_model(example_inputs[0])
    torch.cuda.empty_cache()

'''
Tiny_Convert_Clip_To_Relu: post quantize optim strategy:
After quantization, conver a clip operation to Relu (with restriction, need to check the clip param)
'''
class Tiny_Convert_Clip_To_Relu(nn.Module):
    def __init__(self):
        super(Tiny_Convert_Clip_To_Relu, self).__init__()
        self.conv = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)  # input, weight, bias
        x = self.relu(x)  # output
        x = torch.clip(x, 0, 1)  # output ,   will be transfer to Relu
        x = torch.clip(x, -1, 1)  # output ,  will not be transfer to Relu
        return x

@use_temporary_directory
def test_torch_clip_2_relu_optim_strategy(tmpdir: str):
    '''
    For better allign with hw requirements for deployment,
    transfer one clip node to relu node after quantization.
    '''
    float_model = Tiny_Convert_Clip_To_Relu().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 16, 16).to(torch_device), )
    # ========== test using hardware constrain ===============
    graph_model = capture_pre_autograd_graph(float_model, example_inputs)
    # graph_model = torch.export.export_for_training(float_model, example_inputs).module() # TODO torch2.5
    # === condition check, as no quantizer followd the clip, will not replace ====
    opt_graph_module = opt_after_qt.ConvertClip2ReLUQOPass()(graph_model)
    ops_relu_count, ops_clip_count = 0, 0
    for n in opt_graph_module.graph.nodes:
        if is_relu_act_node(n):
            ops_relu_count += 1
        if is_clip_node(n):
            ops_clip_count += 1
    assert ops_relu_count == 1
    assert ops_clip_count == 2

    # ===quantization quantizer will be inserted
    graph_model = capture_pre_autograd_graph(float_model, example_inputs)
    quantizer = ModelQuantizer(quant_config)
    quantized_model = quantizer.quantize_model(graph_model, [torch.rand(4, 3, 16, 16).to(torch_device) for _ in range(3)])
    count_quantizer = 0
    for module in quantized_model.modules():
        if isinstance(module, ScaledFakeQuantize):
            count_quantizer += 1
    assert count_quantizer == 6, "The total quantizer in this model should be 7"

    # NOTE only perform clip 2 relu after quantization
    ops_relu_count, ops_clip_count = 0, 0
    for n in opt_graph_module.graph.nodes:
        if is_relu_act_node(n):
            ops_relu_count += 1
        if is_clip_node(n):
            ops_clip_count += 1
    assert ops_relu_count == 1
    assert ops_clip_count == 2
    opt_graph_module = opt_after_qt.ConvertClip2ReLUQOPass()(quantized_model)
    ops_relu_count, ops_clip_count = 0, 0
    for n in opt_graph_module.graph.nodes:
        if is_relu_act_node(n):
            ops_relu_count += 1
        if is_clip_node(n):
            ops_clip_count += 1
    assert ops_relu_count == 2, 'After optimize, total relu num should be 2'
    assert ops_clip_count == 1, 'After optimize, total relu num should be 1'
    opt_graph_module = quantizer.freeze(quantized_model.eval())
    quantized_model(example_inputs[0])
    torch.onnx.export(opt_graph_module, *example_inputs, tmpdir + "/clip2relu.onnx")
    torch.cuda.empty_cache()


'''
TinyMean2GAP: pre quantize optim strategy:
Before quantization, conver a mean to GAP (with restriction, need to check mean equal to GAP)
'''
class TinyMean2GAP(nn.Module):
    def __init__(self):
        super(TinyMean2GAP, self).__init__()
        self.conv = nn.Conv2d(3, 8, 3, bias=True)
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool3 = nn.AvgPool2d(3, stride=2)

    def forward(self, x0, x1, x2, x3, x4, x5, x6, x7):
        x0 = torch.mean(x0, dim=(2), keepdim=False)   # onnx: reducemean
        # input output  -> 2
        x1 = torch.mean(x1, dim=(2), keepdim=True)  # onnx: reducemean
        # input output  -> 2
        x2 = torch.mean(x2, dim=(2, 3), keepdim=True)  # onnx: GlobalAveragePool  # torch2.5 & cap: torch.ops.aten.mean.dim()
        # input output  -> 2
        x3 = self.pool1(x3)  # onnx: GlobalAveragePool  #torch2.5 & torchcap torch.ops.aten.adaptive_avg_pool2d.default()
        # input output  -> 2
        x4 = nn.functional.adaptive_avg_pool2d(x4, (1, 1))  # onnx: GlobalAveragePool
        # input output  -> 2
        x5 = self.pool2(x5)  # onnx: AveragePool  #torch.ops.aten.avg_pool2d.default();
        # input output  -> 2
        x6 = self.pool3(x6)  # onnx: AveragePool  #torch.ops.aten.avg_pool2d.default();
        # input output  -> 2
        x7 = self.conv(x7)
        # input output weight, bias -> 4
        return x0, x1, x2, x3, x4, x5, x6, x7

@use_temporary_directory
def test_mean_2_pooling_strategy(tmpdir: str):
    '''
    For better allign with hw requirements for deployment,
    transfer one clip node to relu node after quantization.
    '''
    float_model = TinyMean2GAP().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 10, 10).to(torch_device),
                      torch.rand(1, 3, 20, 20).to(torch_device),
                      torch.rand(1, 3, 30, 30).to(torch_device),
                      torch.rand(1, 3, 40, 40).to(torch_device),
                      torch.rand(1, 3, 50, 50).to(torch_device),
                      torch.rand(1, 3, 60, 60).to(torch_device),
                      torch.rand(1, 3, 70, 70).to(torch_device),
                      torch.rand(1, 3, 80, 80).to(torch_device))
    quant_inputs = {"x" + str(index): value for index, value in enumerate(example_inputs)}
    org_out = float_model(*example_inputs)
    # ========== test using hardware constrain ===============
    graph_model = capture_pre_autograd_graph(float_model, example_inputs)
    from quark.torch.quantization.graph.optimization.pre_quant.opt_pass_before_quant import ConvertReduceMean2GapQOPass
    ops_mean_count, ops_adaptive_avg_pool_count = 0, 0
    for n in graph_model.graph.nodes:
        if is_mean_node(n):
            ops_mean_count += 1
        if is_adaptive_avg_pool_node(n):
            ops_adaptive_avg_pool_count += 1
    assert ops_mean_count == 3
    assert ops_adaptive_avg_pool_count == 3
    graph_model = ConvertReduceMean2GapQOPass()(graph_model)
    ops_mean_count, ops_adaptive_avg_pool_count = 0, 0
    for n in graph_model.graph.nodes:
        if is_mean_node(n):
            ops_mean_count += 1
        if is_adaptive_avg_pool_node(n):
            ops_adaptive_avg_pool_count += 1
    assert ops_mean_count == 2
    assert ops_adaptive_avg_pool_count == 4
    graph_out = graph_model(*example_inputs)
    assert all([torch.allclose(x[0], x[1]) for x in zip(org_out, graph_out)])
    torch.cuda.empty_cache()
    # ========small network quantization=====
    quantizer = ModelQuantizer(quant_config)
    graph_model = capture_pre_autograd_graph(float_model, example_inputs)
    quantized_model = quantizer.quantize_model(graph_model, [quant_inputs])

    count_quantizer = 0
    for module in quantized_model.modules():
        if isinstance(module, ScaledFakeQuantize):
            count_quantizer += 1
    assert count_quantizer == 18, "The total quantizer in this model should be 7"
    ops_mean_count, ops_adaptive_avg_pool_count = 0, 0
    for n in quantized_model.graph.nodes:
        if is_mean_node(n):
            ops_mean_count += 1
        if is_adaptive_avg_pool_node(n):
            ops_adaptive_avg_pool_count += 1
    assert ops_mean_count == 2
    assert ops_adaptive_avg_pool_count == 4
    opt_graph_module = quantizer.freeze(quantized_model.eval())
    torch.onnx.export(opt_graph_module, quant_inputs, tmpdir + "/mean2gap.onnx")
    torch.cuda.empty_cache()



if __name__ == "__main__":
    torch.cuda.empty_cache()
    test_torch_module_used_over_once_optim_strategy()
    test_torch_sg_bn2d_to_conv2d_optim_strategy()
    test_use_over_once_module_optim()
    test_torch_clip_2_relu_optim_strategy()
    test_mean_2_pooling_strategy()
    torch.cuda.empty_cache()
