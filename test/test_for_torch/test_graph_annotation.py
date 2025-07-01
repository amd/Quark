#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys

sys.path.append("..")
import quark.torch.kernel  # noqa
import torch
import onnx
import torch.nn as nn
from torch.fx import Node, GraphModule
from torch._export import capture_pre_autograd_graph
from quark.torch.quantization.graph.optimization.pre_quant.replace_linear_to_qtlinear import replace_linear_qtlinear
from quark.torch.quantization.graph.optimization.pre_quant.replace_conv2d_to_qtconv2d import replace_conv2d_qtconv2d
from quark.torch.quantization.graph.optimization.pre_quant.replace_conv_bn_to_qt_model import replace_conv2dbn_quantizedconv_module
from quark.torch.quantization.graph.torch_utils import is_max_pool2d_node, is_avg_pool2d_node, is_sum_node, \
    is_adaptive_avg_pool2d_node
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize
from quark.torch import ModelQuantizer
from quark.torch.quantization.graph.processor import insert_quantizer
from quark.torch.quantization.nn.modules.quantize_pool import QuantAdaptiveAvgPool2d, QuantAvgPool2d
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import QuantizedConvBatchNorm2d
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d
from quark.torch.quantization.config.config import QuantizationSpec, QuantizationConfig, Config
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType, QuantizationMode
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
from quark.shares.utils.testing_utils import torch_device, use_temporary_directory


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

# ------------- test model annotation------------

# This is a baseline quantizer group ID of the TinyModel's graph-based model,
# if reasonable, any modification of the processor_utils.py will not influence the result
# If the result changed, please check the reason
Edg_or_Node_to_Group_Id = {
    'arg0_1conv2d_bn_quantized_module': 0,
    'relu_': 1,
    'relu_conv2d_1_bn_quantized_module': 1,
    'conv2d_1_bn_quantized_module': 2,
    'conv2d_1_bn_quantized_moduleconv2d_2_quantized_module': 2,
    'relu__1': 3,
    'relu__1conv2d_4_quantized_module': 3,
    'conv2d_4_quantized_module': 4,
    'relu__1conv2d_3_quantized_module': 3,
    'conv2d_3_quantized_module': 5,
    'conv2d_3_quantized_moduleadd': 5,
    'conv2d_4_quantized_moduleadd': 4,
    'relu': 6,
    'reluadd_1': 6,
    'conv2d_4_quantized_moduleadd_1': 4,
    'add_1': 7,
    'add_1adaptive_avg_pool2d': 7,
    'adaptive_avg_pool2d': 8,
    'adaptive_avg_pool2dhardtanh': 8,
    'flatten': 9,
    'flattenlinear_quantized_module': 9,
    'linear_quantized_module': 10
}


class TinyModel(nn.Module):
    '''
  This model is particually designed to test the graph annotation function,
  each module's name is handily and particularly designed.
  If need to change, please particular care.

  for example
  1.no quantizer between  conv and relu, because insert quantizer after ReLu is enough;
  2. two tensors are quantized and then two tensors are added, and then fed to a ReLulayer,
    a quantizer is inserted after the ReLu, no need to insert befor ReLu.
  '''

    def __init__(self):
        super().__init__()
        # annoate -> quantized_convbn2d_act
        self.conv2d = nn.Conv2d(3, 32, 3, bias=False)
        self.bn = nn.BatchNorm2d(32)
        self.relu_ = nn.ReLU(inplace=True)
        # annoate -> quantized_convbn2d
        self.conv2d_1 = nn.Conv2d(32, 32, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        # annoate -> convlike_act
        self.conv2d_2 = nn.Conv2d(32, 32, 3, bias=False)
        self.relu__1 = nn.ReLU(inplace=True)
        # annoate -> convlike
        self.conv2d_3 = nn.Conv2d(32, 64, 3, bias=False)
        self.conv2d_4 = nn.Conv2d(32, 64, 3, bias=True)
        # annoate -> add_act
        self.relu = nn.ReLU()
        # annoate -> add
        # annoate -> adaptive_avg_pool2d
        self.adaptive_avg_pool2d = nn.AdaptiveAvgPool2d((1, 1))
        # annoate -> relu6
        self.hardtanh = nn.ReLU6()
        # annoate -> convlike
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)
        x = self.relu_(x)  # annoate -> quantized_convbn2d_act

        x = self.conv2d_1(x)
        x = self.bn1(x)  # annoate -> quantized_convbn2d

        x = self.conv2d_2(x)
        x = self.relu__1(x)  # annoate -> convlike_act

        x_1 = self.conv2d_3(x)
        x_2 = self.conv2d_4(x)  # annoate -> convlike

        x = self.relu(x_1 + x_2)  # annoate -> add_act
        x = x + x_2  # annoate -> add_act
        x = self.adaptive_avg_pool2d(x)  # adaptive_avg_pool2d
        x = self.hardtanh(x)  # annoate -> relu6
        x = torch.flatten(x, 1)
        x = self.linear(x)  # annoate -> convlike
        return x


def test_annotation():
    torch.cuda.empty_cache()
    from quark.torch.quantization.graph.processor import processor
    batch_shape = [4, 3, 56, 56]
    model = TinyModel().eval().to(torch_device)
    example_inputs = (torch.rand(batch_shape).to(torch_device), )
    # [[x.name, x.meta.get("nn_module_stack", None)] for x in graph_model.graph.nodes]
    graph_model = capture_pre_autograd_graph(model, example_inputs)
    # graph_model = processor._quant_optimize(graph_model)
    replace_conv2dbn_quantizedconv_module(graph_model)
    replace_linear_qtlinear(graph_model)
    replace_conv2d_qtconv2d(graph_model)
    processor.annotate(graph_model, quant_config)
    # graph_model =processor.insert_quantizer(graph_model)
    # graph_model.to_folder("./tinymodel_test")
    # check whether the group id modified
    edge_or_node_to_qspec = insert_quantizer._get_edge_or_node_to_qspec(graph_model)
    edge_or_node_to_group_id = insert_quantizer._get_edge_or_node_to_group_id(edge_or_node_to_qspec)
    group_id_dic = {}
    for node_or_edge, fk_id in edge_or_node_to_group_id.items():
        if isinstance(node_or_edge, Node):
            group_id_dic[node_or_edge.name] = fk_id
        else:
            group_id_dic[node_or_edge[0].name + node_or_edge[1].name] = fk_id

    assert group_id_dic.keys() == Edg_or_Node_to_Group_Id.keys()
    assert group_id_dic == Edg_or_Node_to_Group_Id, " you may modified the annotation function, and influence the quantizer insertation"

    graph_model = processor.allow_exported_model_train_eval(graph_model)

    replaced_convbn2d_num, replace_linear_num, replace_conv2d = 0, 0, 0
    model_out = model(example_inputs[0])
    for module in graph_model.modules():
        if isinstance(module, QuantizedConvBatchNorm2d):
            module.eval()
            replaced_convbn2d_num += 1
        if isinstance(module, QuantLinear):
            replace_linear_num += 1
        if isinstance(module, QuantConv2d):
            replace_conv2d += 1
    # 2 is depend on the TinyModel design, in this model, replacement will be 2.
    assert replaced_convbn2d_num == 2, "replace ops.conv2d + ops.bn2d -> QuantizedConvBatchNorm2d mismatch, please check"
    assert replace_linear_num == 1, "replace ops.linear -> QuantLinear mismatch, please check"
    assert replace_conv2d == 3, "replace ops.conv2d -> QuantConv2d mismatch, please check"
    graph_out = graph_model(example_inputs[0])
    assert torch.allclose(model_out, graph_out, atol=1e-5), "float model's out diffs vs graph model's out"
    print("Finish test basic graph annotation")
    torch.cuda.empty_cache()

# ------------------------------ test annotation elementary arithmetic (+, -, *, /) --------------------

class TinyMathArithmeticModel(nn.Module):
    '''
  This model is particually designed to test the graph annotation function,
  each module's name is handily and particularly designed.
  If need to change, please particular care.
  '''

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 16, 3)

    def forward(self, x):
        x = self.conv2d(x)  # input, weight, bias, output: 4
        # add   input2, output : 2 * 3
        x = x + 0.1
        x = torch.add(x, 0.2)
        x += 0.3
        # sub  input2, output : 2 * 4
        x = x - 0.1
        x = torch.sub(x, 0.2)
        x = torch.subtract(x, 0.3)
        x -= 0.4
        # mul  input2, output : 2 * 4
        x = x * 1.1
        x = torch.mul(x, 1.2)
        x = torch.multiply(x, 1.3)
        x *= 1.4
        # div input2, output : 2 * 3
        x = x / 1.1
        x = torch.div(x, 1.2)
        x /= 1.3
        return x

@use_temporary_directory
def test_annotation_element_arithmetic(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyMathArithmeticModel().to(torch_device).eval()
    example_inputs = (torch.ones(1, 3, 20, 20).to(torch_device), )
    fp_out = float_model(example_inputs[0])
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    # ========== test quant pipeline===============
    emp_config = QuantizationConfig()
    emp_quant_config = Config(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()
        quantized_model = quantizer.quantize_model(graph_model, [example_inputs[0]])
        out_2 = quantized_model.eval()(*example_inputs)
        if each_quant_config == emp_quant_config:
            assert torch.allclose(fp_out, out_2)
        assert len([x for x in quantized_model.graph.nodes]) in [61, 31]
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [32, 0]
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        out_onnx_path = tmpdir + "/max_pool_annotate.onnx"
        torch.onnx.export(opt_graph_module, example_inputs, out_onnx_path)
    torch.cuda.empty_cache()


# ------------ test partly quant model----------------------------
# Test1: no grad func
# Test2: partly nodes not to quant
# Test3: sigmoid, softmax annotation
# Test4: cat, ops.aten.(reshpe, permute, squeeze)
class TinyPartQuantModel(nn.Module):
    '''
  This model is particually designed to test the graph annotation function,
  If need to change, please particular care.
  '''

    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 32, 3, bias=False)
        self.bn = nn.BatchNorm2d(32)

        self.conv2d_0 = nn.Conv2d(32, 32, 3, bias=True)
        self.relu_0 = nn.ReLU()

        self.conv2d_1 = nn.Conv2d(32, 64, 3, bias=True)
        self.sigmoid_1 = nn.Sigmoid()

        self.conv2d_2 = nn.Conv2d(32, 64, 3, bias=True)
        self.softmax_2 = nn.Softmax(dim=1)

        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv2d(x)
        x = self.bn(x)  # in, w, b
        x = torch.nn.functional.sigmoid(x)   # o

        x_1 = self.sigmoid_1(self.conv2d_1(x))  # w, b, o
        x_2 = self.softmax_2(self.conv2d_2(x))  # w, b, o

        x_1 = x_1 + 10   # in2, out
        x_2 = x_2 + 10   # in2, out
        x = torch.cat([x_1, x_2], dim=1)  # out
        x = torch.permute(x.reshape([x.shape[0], 64, -1]), [0, 2, 1]).squeeze(0).unsqueeze(0)   # o1, o2, o3, o4
        with torch.no_grad():
            x = torch.permute(x, [0, 2, 1])
            sum_num = torch.sum(x)
            x = x + 0.5
            x = x + sum_num
            x = torch.nn.functional.hardtanh(x)
            x = torch.nn.functional.relu(x)
            # x = x.mean(dim=2)
            x = torch.mean(x, dim=2)
        x = self.linear(x)  # input, w, b, output
        return x


def test_annotation_without_grad_skip_quant():
    torch.cuda.empty_cache()
    float_model = TinyPartQuantModel().to(torch_device).eval()
    example_inputs = (torch.ones(1, 3, 32, 32).to(torch_device), )
    out_1 = float_model(example_inputs[0])
    graph_model = torch.export.export_for_training(float_model , example_inputs).module()
    out_2 = graph_model(example_inputs[0])
    assert torch.allclose(out_1, out_2)
    quantizer = ModelQuantizer(quant_config)
    quantized_model = quantizer.quantize_model(graph_model, [example_inputs[0]])
    quantized_model(example_inputs[0])
    assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) == 23
    print("Finish test: test_annotation_without_grad_skip_quant")
    torch.cuda.empty_cache()


'''
Test node annotate: avgpooling
'''
class Tiny_avg_pooling_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool1 = nn.AdaptiveAvgPool2d((1, 1))
        self.pool2 = nn.AdaptiveAvgPool2d((2, 2))
        self.pool3 = nn.AvgPool2d((5, 5), stride=5)
        self.conv2d = nn.Conv2d(3, 8, 3)

    def forward(self, x0, x1, x2, x3, x4, x5):
        # annotate avg pooling
        x0 = self.pool1(x0)   # ->QuantAdaptiveAvgPool2d
        x1 = self.pool2(x1)   # ops.adaptiveavgpool
        x2 = self.pool3(x2)   # QuantAvgpool
        x3 = torch.nn.functional.adaptive_avg_pool2d(x3, (1, 1))  # QuantAdaptiveAvgPool2d
        x4 = torch.nn.functional.adaptive_avg_pool2d(x4, (2, 2))  # ops.adaptiveavgpool
        x5 = self.conv2d(x5)
        x5 = torch.nn.functional.avg_pool2d(x5, (2, 2), stride=2)  # QuantAvgpool
        return x0, x1, x2, x3, x4, x5


@use_temporary_directory
def test_avg_pooling_annotation(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = Tiny_avg_pooling_Model().to(torch_device).eval()
    example_inputs = (torch.rand(2, 3, 10, 10).to(torch_device),
                      torch.rand(2, 3, 12, 12).to(torch_device),
                      torch.rand(2, 3, 14, 14).to(torch_device),
                      torch.rand(2, 3, 16, 16).to(torch_device),
                      torch.rand(2, 3, 18, 18).to(torch_device),
                      torch.rand(2, 3, 20, 20).to(torch_device),)
    quant_inputs = {"x" + str(index): value for index, value in enumerate(example_inputs)}

    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    assert fx_contains_op_num(graph_model, is_avg_pool2d_node) == 2
    assert fx_contains_op_num(graph_model, is_adaptive_avg_pool2d_node) == 4
    out1 = graph_model.eval()(*example_inputs)
    assert all([torch.allclose(x[0], x[1]) for x in zip(fp_out, out1)])

    # ========== test quant pipeline===============
    emp_config = QuantizationConfig()
    emp_quant_config = Config(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()
        quantized_model = quantizer.quantize_model(graph_model, [quant_inputs])
        out_2 = quantized_model.eval()(*example_inputs)
        # if each_quant_config == emp_quant_config:
        #     assert [torch.allclose(x[0], x[1]) for x in zip(fp_out, out_2)]
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [15, 0]
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        opt_graph_module(*example_inputs)
        assert fx_contain_module_num(opt_graph_module, QuantAvgPool2d) == 2
        assert fx_contains_op_num(opt_graph_module, is_adaptive_avg_pool2d_node) == 2
        assert fx_contain_module_num(opt_graph_module, QuantAdaptiveAvgPool2d) == 2
        out_onnx_path = tmpdir + "/avg_pool_annotate.onnx"
        torch.onnx.export(opt_graph_module, example_inputs, out_onnx_path)
        assert onnx_contains_op_num(out_onnx_path, 'GlobalAveragePool') == 2
        assert onnx_contains_op_num(out_onnx_path, 'Mul') == 4
        assert onnx_contains_op_num(out_onnx_path, 'AveragePool') == 4
    torch.cuda.empty_cache()


'''
Test node annotate: maxpooling
'''
class Tiny_max_pooling_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.maxpool1 = nn.MaxPool2d(3, 3)
        self.conv2d = nn.Conv2d(3, 8, 3)

    def forward(self, x0, x1):
        # annotate max pooling
        x0 = self.maxpool1(x0)
        x1 = self.conv2d(x1)
        x1 = torch.nn.functional.max_pool2d(x1, (2, 2), stride=2)
        return x0, x1


@use_temporary_directory
def test_max_pooling_annotation(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = Tiny_max_pooling_Model().to(torch_device).eval()
    example_inputs = (torch.rand(2, 3, 10, 10).to(torch_device),
                      torch.rand(2, 3, 12, 12).to(torch_device))
    quant_inputs = {"x" + str(index): value for index, value in enumerate(example_inputs)}

    fp_out = float_model(*example_inputs)
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    assert fx_contains_op_num(graph_model, is_max_pool2d_node) == 2
    out1 = graph_model.eval()(*example_inputs)
    assert all([torch.allclose(x[0], x[1]) for x in zip(fp_out, out1)])

    # ========== test quant pipeline===============
    emp_config = QuantizationConfig()
    emp_quant_config = Config(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()
        quantized_model = quantizer.quantize_model(graph_model, [quant_inputs])
        out_2 = quantized_model.eval()(*example_inputs)
        if each_quant_config == emp_quant_config:
            assert all([torch.allclose(x[0], x[1]) for x in zip(fp_out, out_2)])
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [7, 0]
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        opt_graph_module(*example_inputs)
        assert fx_contains_op_num(opt_graph_module, is_max_pool2d_node) == 2
        out_onnx_path = tmpdir + "/max_pool_annotate.onnx"
        torch.onnx.export(opt_graph_module, example_inputs, out_onnx_path)
        assert onnx_contains_op_num(out_onnx_path, 'MaxPool') == 2
    torch.cuda.empty_cache()

'''
Test node annotate: sum
'''
class Tiny_Sum_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 8, 3)


    def forward(self, x0):
        x0 = self.conv2d(x0)
        x1 = torch.sum(x0)
        return x1


@use_temporary_directory
def test_sum_annotation(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = Tiny_Sum_Model().to(torch_device).eval()
    example_inputs = (torch.rand(2, 3, 10, 10).to(torch_device),)

    fp_out = float_model(example_inputs[0])
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    assert fx_contains_op_num(graph_model, is_sum_node) == 1
    out1 = graph_model.eval()(*example_inputs)
    assert torch.allclose(fp_out, out1)

    # ========== test quant pipeline===============
    emp_config = QuantizationConfig()
    emp_quant_config = Config(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()
        quantized_model = quantizer.quantize_model(graph_model, [example_inputs[0]])
        out_2 = quantized_model.eval()(*example_inputs)
        if each_quant_config == emp_quant_config:
            assert torch.allclose(fp_out, out_2)
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [5, 0]
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        opt_graph_module(*example_inputs)
        assert fx_contains_op_num(opt_graph_module, is_sum_node) == 1
        out_onnx_path = tmpdir + "/max_pool_annotate.onnx"
        torch.onnx.export(opt_graph_module, example_inputs, out_onnx_path)
    torch.cuda.empty_cache()

'''
Test node annotate: hardtanh, add_act, sigmoid
'''
class Tiny_Hardtanh_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 8, 3)
        self.hardtanh = nn.Hardtanh()
        self.relu6 = nn.ReLU6()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(1)


    def forward(self, x0):
        x0 = self.conv2d(x0)   # input, weight, bias, output : 4
        x1 = self.hardtanh(x0)  # output: 1
        x2 = torch.nn.functional.hardtanh(x0)   # output: 1
        x2 = self.sigmoid(x2)  # output: 1
        x2 = torch.permute(x2, [1, 0, 2, 3])  # output: 1

        x3 = torch.nn.functional.hardtanh_(x0)  # output: 1
        x3 = torch.nn.functional.sigmoid(x3)  # output: 1
        x3 = torch.reshape(x3, (-1,))

        x4 = self.relu6(x0)   # output: 1
        x4 = self.softmax(x4)  # output: 1
        x4 = torch.unsqueeze(x4, 1)  # output: 1
        x4 = torch.squeeze(x4)  # output: 1

        x5 = torch.nn.functional.relu6(x0)    # output: 1
        x5 = torch.nn.functional.softmax(x5, 1)  # output: 1
        x6 = self.relu(x0)   # output: 1
        x7 = torch.nn.functional.relu(x0)  # output: 1
        x8 = torch.nn.functional.relu_(x0)  # output: 1
        x9 = torch.nn.functional.relu(x7 + x8)  # output: 1
        return x1, x2, x3, x4, x5, x6, x9


@use_temporary_directory
def test_hardtanh_annotation(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = Tiny_Hardtanh_Model().to(torch_device).eval()
    example_inputs = (torch.rand(2, 3, 10, 10).to(torch_device),)

    fp_out = float_model(example_inputs[0])
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    out1 = graph_model.eval()(*example_inputs)
    assert all([torch.allclose(x[0], x[1]) for x in zip(fp_out, out1)])

    # ========== test quant pipeline===============
    emp_config = QuantizationConfig()
    emp_quant_config = Config(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()
        quantized_model = quantizer.quantize_model(graph_model, [example_inputs[0]])
        out_2 = quantized_model.eval()(*example_inputs)
        if each_quant_config == emp_quant_config:
            test_fp = [fp_out[0]] + list(fp_out[3:])
            test_out2 = [out_2[0]] + list(out_2[3:])
            assert all([torch.allclose(x[0], x[1]) for x in zip(test_fp, test_out2)])
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [21, 0]
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        opt_graph_module(*example_inputs)
        out_onnx_path = tmpdir + "/hardtanh_relu_annotate.onnx"
        torch.onnx.export(opt_graph_module, example_inputs, out_onnx_path)
        assert onnx_contains_op_num(out_onnx_path, "Clip") == 5
        assert onnx_contains_op_num(out_onnx_path, "HardSigmoid") == 2
        assert onnx_contains_op_num(out_onnx_path, 'Relu') == 4
    torch.cuda.empty_cache()


'''
Test node annotate: pixel_shuffle
'''
class Tiny_Pixel_Shuffle_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(3, 64, 3)
        self.pixel_shuffle = nn.PixelShuffle(4)

    def forward(self, x):
        x = self.conv2d(x)   # input, weight, bias, output : 4
        x = self.pixel_shuffle(x)  # output: 1
        return x


@use_temporary_directory
def test_pixel_shuffle_annotation(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = Tiny_Pixel_Shuffle_Model().to(torch_device).eval()
    example_inputs = (torch.rand(2, 3, 10, 10).to(torch_device),)

    fp_out = float_model(example_inputs[0])
    graph_model = torch.export.export_for_training(float_model, example_inputs).module()
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    out1 = graph_model.eval()(*example_inputs)
    assert all([torch.allclose(x[0], x[1]) for x in zip(fp_out, out1)])

    # ========== test quant pipeline===============
    emp_config = QuantizationConfig()
    emp_quant_config = Config(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        quantizer = ModelQuantizer(each_quant_config)
        graph_model = torch.export.export_for_training(float_model, example_inputs).module()
        quantized_model = quantizer.quantize_model(graph_model, [example_inputs[0]])
        out_2 = quantized_model.eval()(*example_inputs)
        if each_quant_config == emp_quant_config:
            torch.allclose(out1, out_2)
        assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [5, 0]
        opt_graph_module = quantizer.freeze(quantized_model.eval())
        opt_graph_module(*example_inputs)
        out_onnx_path = tmpdir + "/pixel_shuffle_annotate.onnx"
        torch.onnx.export(opt_graph_module, example_inputs, out_onnx_path)
        assert onnx_contains_op_num(out_onnx_path, "DepthToSpace") == 1
        assert onnx_contains_op_num(out_onnx_path, "QuantizeLinear") in [5, 0]
    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    test_annotation()
    test_annotation_element_arithmetic()
    test_annotation_without_grad_skip_quant()
    test_avg_pooling_annotation()
    test_max_pooling_annotation()
    test_sum_annotation()
    test_hardtanh_annotation()
    test_pixel_shuffle_annotation()
    torch.cuda.empty_cache()
