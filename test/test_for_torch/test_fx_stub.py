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

from quark.common.utils.import_utils import export_for_training
from quark.common.utils.log import ScreenLogger
from quark.common.utils.testing_utils import torch_device, use_temporary_directory
from quark.torch import ModelQuantizer

# from quark.torch.quantization.graph.export.onnx import *
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, QuantizationMode, RoundType, ScaleType
from quark.torch.quantization.graph.graph_modelquantizer import FxGraphQuantizer
from quark.torch.quantization.graph.ops.quant_stubs import DeQuantStub, QuantStub
from quark.torch.quantization.graph.processor.processor import mark_exclude_nodes
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d, QuantConvTranspose2d
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import (
    QuantConvTransposeBatchNorm2d,
    QuantizedConvBatchNorm2d,
)
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

TEST_TOPIC = "torch FX graph mode quantization, partly quant model"

logger = ScreenLogger(__name__)
INT8_PER_TENSOR_SPEC = QTensorConfig(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)
quant_config = QLayerConfig(
    input_tensors=INT8_PER_TENSOR_SPEC,
    output_tensors=INT8_PER_TENSOR_SPEC,
    weight=INT8_PER_TENSOR_SPEC,
    bias=INT8_PER_TENSOR_SPEC,
)
quant_config = QConfig(global_quant_config=quant_config, quant_mode=QuantizationMode.fx_graph_mode)
"""
utils function
"""


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
Test user using QuantStub & DeQuantStub to specify the quant scope
"""


class Simply_Quant_Stub_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 16, 1)
        self.conv3 = nn.ConvTranspose2d(16, 32, 1)
        self.conv4 = nn.ConvTranspose2d(32, 32, 1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv5 = nn.Conv2d(32, 64, 3)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 1)
        self.conv7 = nn.ConvTranspose2d(64, 128, 1)
        self.conv8 = nn.ConvTranspose2d(128, 128, 1)
        self.bn4 = nn.BatchNorm2d(128)

        self.conv9 = nn.Conv2d(32, 64, 3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv10 = nn.Conv2d(64, 64, 1)
        self.conv11 = nn.ConvTranspose2d(64, 128, 1)
        self.conv12 = nn.ConvTranspose2d(128, 128, 1)
        self.bn6 = nn.BatchNorm2d(128)

        self.classifier = nn.Linear(256, 1000)
        self.dequant_stub = DeQuantStub
        self.quant_stub = QuantStub

    def forward(self, x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.quant_stub(x)
        x = self.bn2(self.conv4(x))  # input, w, b, o

        x1 = torch.nn.functional.relu(self.bn3(self.conv5(x)))  # w, b, o
        x1 = self.conv6(x1)  # w, b, output
        x1 = self.conv7(x1)  # w, b, output
        x1 = self.dequant_stub(x1)
        x1 = self.bn4(self.conv8(x1))

        x2 = torch.nn.functional.relu(self.bn5(self.conv9(x)))  # w, b, o
        x2 = self.conv10(x2)  # w, b, output
        x2 = self.conv11(x2)  # w, b, output
        x2 = self.dequant_stub(x2)
        x2 = self.bn6(self.conv12(x2))

        x = torch.cat([x1, x2], dim=1)
        x = x.mean([2, 3])
        x = self.quant_stub(x)
        x = self.classifier(x)  # input, w, b, output
        x = self.dequant_stub(x)
        return x


@use_temporary_directory
def test_torch_quant_stub(tmpdir: str):
    """
    test torch quant stub and dequantstub func
    """
    torch.cuda.empty_cache()
    float_model = Simply_Quant_Stub_Model().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 32, 32).to(torch_device),)
    out_fp32 = float_model.eval()(example_inputs[0])
    # session 1
    # =========================
    graph_model = export_for_training(float_model, example_inputs).module()
    _exclude_quant_node = mark_exclude_nodes(graph_model)
    out_opt_fx_graph = graph_model(example_inputs[0])
    assert torch.allclose(out_fp32, out_opt_fx_graph)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [emp_quant_config, quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [unify_quantizer, fx_quantizer]:
            graph_model_1 = export_for_training(float_model, example_inputs).module()
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
                out_2 = quantized_model.eval()(*example_inputs)
                if each_quant_config == emp_quant_config:
                    assert torch.allclose(out_fp32, out_2, atol=1e-5)
                assert fx_contain_module_num(quantized_model, QuantConv2d) == 3
                assert fx_contain_module_num(quantized_model, QuantConvTranspose2d) == 3
                assert fx_contain_module_num(quantized_model, QuantConvTransposeBatchNorm2d) == 3
                assert fx_contain_module_num(quantized_model, QuantizedConvBatchNorm2d) == 3
                assert fx_contain_module_num(quantized_model, QuantLinear) == 1
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [26, 0]
                freeze_graph_module = quantizer.freeze(quantized_model.eval())
                freeze_graph_module(*example_inputs)
                onnx_dir = tmpdir + "/module_part_quant.onnx"
                torch.onnx.export(freeze_graph_module, example_inputs, onnx_dir, dynamo=False)
                assert onnx_contains_op_num(onnx_dir, "Conv") == 6
                assert onnx_contains_op_num(onnx_dir, "ConvTranspose") == 6
                assert onnx_contains_op_num(onnx_dir, "Gemm") == 1
    logger.info(TEST_TOPIC + "split module that used over one to seperate module, Passed")
    torch.cuda.empty_cache()
    # from torch.autograd import gradcheck
    # gradcheck(torch.ops.my_library.quant_start, (x,))


if __name__ == "__main__":
    torch.cuda.empty_cache()
    test_torch_quant_stub()
    torch.cuda.empty_cache()
