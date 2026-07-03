#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import sys

sys.path.append("..")
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.fx import GraphModule

import quark.torch.quantization.graph.optimization.post_quant.opt_pass_after_quant_float_scale as opt_after_qt_fs
from quark.common.utils.import_utils import export_for_training
from quark.common.utils.log import ScreenLogger
from quark.common.utils.testing_utils import torch_device, use_temporary_directory
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, QuantizationMode, RoundType, ScaleType
from quark.torch.quantization.graph.graph_modelquantizer import FxGraphQuantizer
from quark.torch.quantization.graph.optimization.model_optimization import _apply_post_hw_fs_constrain_passes
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver, PerTensorPowOf2MinMaxObserver
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

logger = ScreenLogger(__name__)

TEST_TOPIC = "torch FX graph mode quantization, align with hw deploy need\n"

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


def fx_contain_module_num(model: GraphModule, target_module: torch.nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, target_module):
            count += 1
    return count


class TinyAlignConcatModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=0)
        self.conv_transpose1 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=0, bias=None)
        self.conv_transpose2 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=0)
        self.conv_transpose3 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=0)
        self.conv1 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x1, x2, x3 = torch.split(x, [16, 16, 16], 1)
        x1 = self.conv_transpose1(x1)
        x2 = self.conv_transpose2(x2)
        x3 = self.conv_transpose3(x3)
        x = torch.concatenate([x1, x2, x3], dim=1)
        return x


@use_temporary_directory
def test_torch_align_concat_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyAlignConcatModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_fs.AlignConcatQOPass()
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
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
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [0, 16]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    assert quantized_model.fake_quantizer_5.scale.item() != 0.25
                    assert quantized_model.fake_quantizer_6.scale.item() != 0.25
                    assert quantized_model.fake_quantizer_7.scale.item() != 0.25
                    assert quantized_model.fake_quantizer_8.scale.item() != 0.25

                    quantized_model.fake_quantizer_8.scale.fill_(0.25)
                    opt_graph = opt_module(quantized_model)

                    assert quantized_model.fake_quantizer_5.scale.item() == 0.25
                    assert quantized_model.fake_quantizer_6.scale.item() == 0.25
                    assert quantized_model.fake_quantizer_7.scale.item() == 0.25
                    assert quantized_model.fake_quantizer_8.scale.item() == 0.25

                    # Output quantizer scale is not a scale
                    opt_graph.fake_quantizer_8.scale = torch.tensor([1.0, 2.0])
                    opt_graph = opt_module(opt_graph)
                    # Input quantizer is missing
                    opt_graph.fake_quantizer_5 = nn.Identity()
                    opt_graph = opt_module(opt_graph)

    torch.cuda.empty_cache()


class TinyAlignPoolModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1, bias=True)
        self.avgpool = nn.AvgPool2d((3, 6), stride=(3, 5), ceil_mode=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.avgpool(x)
        return x


@use_temporary_directory
def test_torch_align_pool_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyAlignPoolModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_fs.AlignPoolQOPass()
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
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
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [0, 5]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    assert quantized_model.fake_quantizer_2.scale.item() != 0.5
                    quantized_model.fake_quantizer_1.scale.fill_(0.5)
                    opt_graph = opt_module(quantized_model)
                    assert quantized_model.fake_quantizer_2.scale.item() == 0.5
                    opt_graph.fake_quantizer_1.scale = torch.tensor([1.0, 2.0])
                    opt_graph = opt_module(opt_graph)

                    # Output quantizer scale is not a scale
                    opt_graph.fake_quantizer_2.scale = torch.tensor([1.0, 2.0])
                    opt_graph = opt_module(opt_graph)

                    # Input quantizer is missing
                    opt_graph.fake_quantizer_1 = nn.Identity()
                    opt_graph = opt_module(opt_graph)

                    # Output quantizer is missing
                    opt_graph.fake_quantizer_2 = nn.Identity()
                    opt_graph = opt_module(opt_graph)

    torch.cuda.empty_cache()


class TinyAlignPadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = x.transpose(2, 3)
        x = F.pad(x, pad=(0, 0, 2, 2))
        x = self.conv2(x)
        x = F.relu(x)
        return x


@use_temporary_directory
def test_torch_align_pad_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyAlignPadModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_fs.AlignPadQOPass()
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
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
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [0, 8]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    assert quantized_model.fake_quantizer_1.scale.item() != 0.5
                    quantized_model.fake_quantizer_2.scale.fill_(0.5)
                    opt_graph = opt_module(quantized_model)
                    assert quantized_model.fake_quantizer_1.scale.item() == 0.5

                    # Output quantizer scale is not a scale
                    opt_graph.fake_quantizer_1.scale = torch.tensor([1.0, 2.0])
                    opt_graph = opt_module(opt_graph)
                    # Input quantizer is missing
                    opt_graph.fake_quantizer_1 = nn.Identity()
                    opt_graph = opt_module(opt_graph)

    torch.cuda.empty_cache()


class TinyAlignSliceModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 48, kernel_size=3, stride=1, padding=0)
        self.conv_transpose1 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=0, bias=None)
        self.conv_transpose2 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=0)
        self.conv_transpose3 = nn.ConvTranspose2d(16, 16, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        x = self.conv(x)
        x1, x2, x3 = torch.split(x, [16, 16, 16], 1)
        return x1


@use_temporary_directory
def test_torch_align_slice_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyAlignSliceModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_fs.AlignSliceQOPass()
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
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
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [0, 5]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    assert quantized_model.fake_quantizer_2.scale.item() != 0.5
                    quantized_model.fake_quantizer_1.scale.fill_(0.5)
                    opt_graph = opt_module(quantized_model)
                    assert quantized_model.fake_quantizer_2.scale.item() == 0.5

                    # input quantizer scale is not a scale
                    opt_graph.fake_quantizer_1.scale = torch.tensor([1.0, 2.0])
                    opt_graph = opt_module(opt_graph)
                    # Output quantizer scale is not a scale
                    opt_graph.fake_quantizer_2.scale = torch.tensor([1.0, 2.0])
                    opt_graph = opt_module(opt_graph)
                    # Input quantizer is missing
                    opt_graph.fake_quantizer_1 = nn.Identity()
                    opt_graph = opt_module(opt_graph)
                    # Output quantizer is missing
                    opt_graph.fake_quantizer_2 = nn.Identity()
                    opt_graph = opt_module(opt_graph)

    torch.cuda.empty_cache()


class TinyAlignTransposeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.fc = nn.Linear(16 * 28 * 28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = x.permute(0, 2, 3, 1)
        return x


@use_temporary_directory
def test_torch_align_transpose_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyAlignTransposeModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_fs.AlignTransposeQOPass()
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
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
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [0, 8]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    assert quantized_model.fake_quantizer_2.scale.item() != 0.5
                    quantized_model.fake_quantizer_3.scale.fill_(0.5)
                    opt_graph = opt_module(quantized_model)
                    assert quantized_model.fake_quantizer_2.scale.item() == 0.5

                    # Input quantizer scale is not a scale
                    opt_graph.fake_quantizer_2.scale = torch.tensor([1.0, 2.0])
                    opt_graph = opt_module(opt_graph)
                    # Output quantizer scale is not a scale
                    opt_graph.fake_quantizer_3.scale = torch.tensor([1.0, 2.0])
                    opt_graph = opt_module(opt_graph)
                    # Input quantizer is missing
                    opt_graph.fake_quantizer_2 = nn.Identity()
                    opt_graph = opt_module(opt_graph)

    torch.cuda.empty_cache()


class TinyAlignReshapeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.pool = nn.AvgPool2d(3, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = self.relu(x)
        x = torch.reshape(x, [1, -1])
        return x


@use_temporary_directory
def test_torch_align_reshape_strategy(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyAlignReshapeModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    opt_module = opt_after_qt_fs.AlignReshapeQOPass()
    for each_quant_config in [quant_config, emp_quant_config]:
        unify_quantizer = ModelQuantizer(each_quant_config)
        fx_quantizer = FxGraphQuantizer(each_quant_config)
        for quantizer in [fx_quantizer, unify_quantizer]:
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
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [0, 7]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    assert quantized_model.fake_quantizer_3.scale.item() != 0.5
                    quantized_model.fake_quantizer_4.scale.fill_(0.5)
                    opt_graph = opt_module(quantized_model)
                    assert quantized_model.fake_quantizer_3.scale.item() == 0.5

                    # Output quantizer scale is not a scale
                    opt_graph.fake_quantizer_4.scale = torch.tensor([1.0, 2.0])
                    opt_graph = opt_module(opt_graph)
                    # Input quantizer is missing
                    opt_graph.fake_quantizer_2 = nn.Identity()
                    opt_graph = opt_module(opt_graph)

    torch.cuda.empty_cache()


INT8_PER_TENSOR_SPEC = QTensorConfig(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)

INT16_PER_TENSOR_SPEC = QTensorConfig(
    dtype=Dtype.int16,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)

INT32_PER_TENSOR_SPEC = QTensorConfig(
    dtype=Dtype.int32,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)

quant_tensor_A8W8B32_config = QLayerConfig(
    input_tensors=INT8_PER_TENSOR_SPEC,
    output_tensors=INT8_PER_TENSOR_SPEC,
    weight=INT8_PER_TENSOR_SPEC,
    bias=INT32_PER_TENSOR_SPEC,
)
quant_A8W8B32_config = QConfig(
    global_quant_config=quant_tensor_A8W8B32_config, quant_mode=QuantizationMode.fx_graph_mode
)


quant_tensor_A16W8B32_config = QLayerConfig(
    input_tensors=INT16_PER_TENSOR_SPEC,
    output_tensors=INT16_PER_TENSOR_SPEC,
    weight=INT8_PER_TENSOR_SPEC,
    bias=INT32_PER_TENSOR_SPEC,
)
quant_A16W8B32_config = QConfig(
    global_quant_config=quant_tensor_A16W8B32_config, quant_mode=QuantizationMode.fx_graph_mode
)


class TinyCONVModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


@use_temporary_directory
def test_torch_a8w8_a16w8(tmpdir: str):
    torch.cuda.empty_cache()
    float_model = TinyCONVModel().to(torch_device).eval()
    example_inputs = (torch.rand(1, 3, 28, 28).to(torch_device),)
    fp_out = float_model(*example_inputs)
    graph_model = export_for_training(float_model, example_inputs).module()
    gp_out = graph_model(*example_inputs)
    assert torch.allclose(fp_out, gp_out)
    # ========== test quant pipeline===============
    emp_config = QLayerConfig()
    emp_quant_config = QConfig(global_quant_config=emp_config, quant_mode=QuantizationMode.fx_graph_mode)
    for each_quant_config in [quant_A8W8B32_config, quant_A16W8B32_config, emp_quant_config]:
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
                quantized_model(example_inputs[0])
                copyed_quantized_model = copy.deepcopy(quantized_model)
                assert fx_contain_module_num(quantized_model, ScaledFakeQuantize) in [4, 0]
                if fx_contain_module_num(quantized_model, ScaledFakeQuantize):
                    act_quantizer = quantized_model.fake_quantizer_0
                    conv_w_quantizer = quantized_model.conv2d_quantized_module._weight_quantizer
                    conv_b_quantizer = quantized_model.conv2d_quantized_module._bias_quantizer
                    assert (
                        act_quantizer.scale.numel() == 1
                        and conv_w_quantizer.scale.numel() == 1
                        and conv_b_quantizer.scale.numel() == 1
                    )
                    assert act_quantizer.dtype in [Dtype.int8, Dtype.int16]
                    assert conv_w_quantizer.dtype == Dtype.int8
                    assert conv_b_quantizer.dtype == Dtype.int32
                    _apply_post_hw_fs_constrain_passes(quantized_model)
                    onnx_dir = tmpdir + "/a8w8_a16w16.onnx"
                    torch.onnx.export(
                        copyed_quantized_model.eval(),
                        example_inputs,
                        onnx_dir,
                        dynamo=False,
                    )

    torch.cuda.empty_cache()


if __name__ == "__main__":
    torch.cuda.empty_cache()
    test_torch_align_concat_strategy()
    test_torch_align_pool_strategy()
    test_torch_align_pad_strategy()
    test_torch_align_slice_strategy()
    test_torch_align_transpose_strategy()
    test_torch_align_reshape_strategy()
    test_torch_a8w8_a16w8()
    torch.cuda.empty_cache()
