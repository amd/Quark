#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#


import pytest
import torch
import torch.nn as nn
from torch import ops  # type: ignore[attr-defined]
from torch.utils.data import DataLoader, Dataset
from torch_testing_utils import run_torch_op_variants  # type: ignore[import-not-found]
from transformers import AutoModelForCausalLM

import quark.torch.kernel  # noqa
from quark.common.data_type import BaseFP8_E5M3
from quark.common.utils.testing_utils import FROM_PRETRAINED_KWARGS, torch_device
from quark.torch import LLMTemplate, ModelQuantizer
from quark.torch.kernel.float8_e5m3 import dequantize_float8_e5m3_func
from quark.torch.kernel.hw_emulation.hw_emulation_interface import (
    fake_quantize_mx,
    fake_quantize_non_mx,
    real_quantize_float8_e5m3,
)
from quark.torch.quantization.config.config import FP4PerGroupSpec, QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType
from quark.torch.quantization.observer.observer import PerBlockMXObserver, PerChannelMinMaxObserver
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase
from quark.torch.quantization.utils import calculate_qmin_qmax, get_dtype_params, reshape_to_blocks


class ToyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)

    def forward(self, x):
        x = self.fc(x)
        return x


input_tensor = torch.ones(1, 4096, 4096)


class MyDataset(Dataset):
    def __init__(self):
        return

    def __len__(self):
        return 2

    def __getitem__(self, index):
        return input_tensor


def is_power_of_two(tensor):
    log2_values = torch.log2(tensor)
    return torch.all(log2_values == log2_values.round())


@pytest.mark.parametrize(
    "dtype,dim,scale_format,scale_calculation_mode",
    [
        (Dtype.fp4, 0, "e8m0", "even"),
        (Dtype.fp4, 0, "e8m0", "floor"),
        (Dtype.fp4, 0, "e8m0", "ceil"),
        (Dtype.fp4, 0, "e4m3", ""),
        (Dtype.fp4, 0, "float32", ""),
        (Dtype.fp4, 1, "e8m0", "even"),
        (Dtype.fp4, 1, "e8m0", "floor"),
        (Dtype.fp4, 1, "e8m0", "ceil"),
        (Dtype.fp4, 1, "e4m3", ""),
        (Dtype.fp4, 1, "float32", ""),
        (Dtype.fp4, -1, "e8m0", "even"),
        (Dtype.fp4, -1, "e8m0", "floor"),
        (Dtype.fp4, -1, "e8m0", "ceil"),
        (Dtype.fp4, -1, "e4m3", ""),
        (Dtype.fp4, -1, "float32", ""),
        (Dtype.fp4, -2, "e8m0", "even"),
        (Dtype.fp4, -2, "e8m0", "floor"),
        (Dtype.fp4, -2, "e8m0", "ceil"),
        (Dtype.fp4, -2, "e4m3", ""),
        (Dtype.fp4, -2, "float32", ""),
    ],
)
def test_fp4_per_channel_scaled_fake_quantize(dtype, dim, scale_format, scale_calculation_mode):
    if len(scale_calculation_mode) == 0:
        scale_calculation_mode = None

    dim0 = 4096
    tensor_shape = (dim0, 4096)

    fp4_per_channel_spec = QTensorConfig(
        dtype=dtype,
        qscheme=QSchemeType.per_channel,
        observer_cls=PerChannelMinMaxObserver,
        ch_axis=dim,
        scale_format=scale_format,
        scale_calculation_mode=scale_calculation_mode,
        is_dynamic=False,
    )

    fp4_per_group_spec = QTensorConfig(
        dtype=dtype,
        qscheme=QSchemeType.per_group,
        observer_cls=PerBlockMXObserver,
        group_size=tensor_shape[-1],
        ch_axis=dim,
        scale_format=scale_format,
        scale_calculation_mode=scale_calculation_mode,
        is_dynamic=False,
    )

    test_device = torch.device(torch_device)
    if test_device.type == "cpu":
        n_runs = 1  # This test is relatively slow on CPU.
    else:
        n_runs = 30

    # NOTE: ``PerChannelMinMaxObserver`` accumulates running min/max across
    # every quantizer call, so the quantizers MUST be created inside the
    # pipeline; reusing them across stable/legacy runs would leave the legacy
    # run starting from the stable run's converged stats and break bit-exact
    # equivalence.
    def pipeline():
        quantizer_per_channel = FakeQuantizeBase.get_fake_quantize(fp4_per_channel_spec, device=torch_device)
        quantizer_per_group = FakeQuantizeBase.get_fake_quantize(fp4_per_group_spec)
        allclose_count = 0
        close_col_sum = 0
        for _ in range(n_runs):
            x = torch.randn(tensor_shape, dtype=torch.float32, device=torch_device)
            per_channel = quantizer_per_channel(x.clone())
            per_group = quantizer_per_group(x.transpose(0, 1).clone()).transpose(0, 1)

            if torch.allclose(per_channel, per_group):
                allclose_count += 1
            close_col = torch.all(torch.isclose(per_channel, per_group), dim=-1)
            close_col_sum += torch.sum(close_col).item()
            diff = (per_channel - per_group).abs()
            assert (torch.count_nonzero(diff).item() / diff.numel()) < 0.1
        return allclose_count, close_col_sum

    # TODO: Remove run_torch_op_variants once legacy pybind .so supported is dropped.
    allclose_count, close_col_sum = run_torch_op_variants(pipeline)

    print(f"Allclose: {allclose_count}/{n_runs} (columns average equal: {close_col_sum / n_runs} / {dim0})")


@pytest.mark.parametrize(
    "dtype,group_size",
    [
        (Dtype.fp4, 32),
        (Dtype.fp4, 16),
        (Dtype.fp8_e4m3, 32),
        (Dtype.fp8_e4m3, 16),
        (Dtype.fp8_e5m2, 32),
        (Dtype.fp8_e5m2, 16),
    ],
)
def test_fp_per_group_weight_quantization_qparams(dtype, group_size):
    FP_WEIGHT_PER_GROUP_SPEC = QTensorConfig(
        dtype=dtype,
        qscheme=QSchemeType.per_group,
        observer_cls=PerBlockMXObserver,
        ch_axis=-1,
        group_size=group_size,
        scale_format="e8m0",
        scale_calculation_mode="floor",
        is_dynamic=False,
    )
    FP_WEIGHT_PER_GROUP_CONFIG = QLayerConfig(weight=FP_WEIGHT_PER_GROUP_SPEC)

    def pipeline():
        model = ToyModel(in_features=4096, out_features=4096)
        model.fc.weight = torch.nn.Parameter(torch.randn([4096, 4096]))
        model(input_tensor)
        dataset = MyDataset()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
        quant_config = QConfig(global_quant_config=FP_WEIGHT_PER_GROUP_CONFIG)
        quantizer = ModelQuantizer(quant_config)
        quant_model = quantizer.quantize_model(model, dataloader)
        return quant_model.fc._weight_quantizer.scale

    scale = run_torch_op_variants(pipeline)
    assert scale.shape[1] == int(4096 / group_size)


@pytest.mark.parametrize(
    "dtype",
    [
        (Dtype.fp4),
    ],
)
def test_mxfp_per_group_scaled_fake_quantize(dtype):
    tensor_shape = (4096, 4096)
    x: torch.Tensor = torch.randn(tensor_shape, dtype=torch.float32)

    _, _, emax = get_dtype_params(dtype)
    block_size, axis = 32, 1
    quant_min, quant_max = calculate_qmin_qmax(dtype)

    block_x = reshape_to_blocks(x.clone(), block_size, axis)
    scale, _ = torch.max(torch.abs(block_x), dim=axis + 1, keepdim=True)
    scale = torch.pow(2, torch.floor(torch.log2(scale)) - emax)
    zero_point = torch.zeros_like(scale, dtype=torch.int32)

    def pipeline():
        qx_scaled = ops.quark.scaled_fake_quantize(
            dtype.value,
            x.clone(),
            scale,
            zero_point,
            -1,
            32,
            quant_min,
            quant_max,
            0,
            QSchemeType.per_group.value,
            "None",
        )
        # ``fake_quantize_mx`` discards the passed scale and recomputes it;
        # the (2, 3) shape only satisfies the parameter contract.
        qx_mx = fake_quantize_mx(
            input_tensor=x.clone(),
            scale=torch.ones((2, 3), dtype=torch.float32),
            mx_element_dtype=dtype,
            axis=-1,
            block_size=32,
            scale_calculation_mode="floor",
        )
        return qx_scaled, qx_mx

    qx_scaled_fake_quantize, qx_fake_quantize_mx = run_torch_op_variants(pipeline)
    assert torch.allclose(qx_scaled_fake_quantize, qx_fake_quantize_mx)


@pytest.mark.parametrize("dtype, block_size", [(Dtype.fp4, 8), (Dtype.fp4, 16), (Dtype.fp4, 32)])
def test_non_mxfp4_per_group_scaled_fake_quantize(dtype, block_size):
    tensor_shape = (4096, 4096)
    x: torch.Tensor = torch.randn(tensor_shape, dtype=torch.float32)

    quant_min, quant_max = calculate_qmin_qmax(dtype)
    block_x = reshape_to_blocks(x.clone(), block_size, 1)

    amax, _ = torch.max(torch.abs(block_x), dim=-1, keepdim=True)
    scale = torch.div(amax, quant_max)
    zero_point = torch.zeros_like(scale, dtype=torch.int32)

    def pipeline():
        qx_scaled = ops.quark.scaled_fake_quantize(
            dtype.value,
            x.clone().unsqueeze(0),
            scale,
            zero_point,
            -1,
            block_size,
            quant_min,
            quant_max,
            0,
            QSchemeType.per_group.value,
            "None",
        )
        qx_non_mx = fake_quantize_non_mx(input_tensor=x.clone(), element_dtype=dtype, axis=-1, block_size=block_size)
        return qx_scaled, qx_non_mx

    qx_scaled_fake_quantize, qx_fake_quantize_non_mx = run_torch_op_variants(pipeline)
    assert torch.allclose(qx_scaled_fake_quantize, qx_fake_quantize_non_mx)


@pytest.mark.parametrize(
    "dtype,scale_calculation_mode",
    [
        (Dtype.fp4, "even"),
        (Dtype.fp4, "floor"),
        (Dtype.fp4, "ceil"),
    ],
)
def test_mxfp4_per_group_scaled_and_non_scaled_fake_quantize(dtype, scale_calculation_mode):
    tensor_shape = (4096, 4096)
    x: torch.Tensor = torch.randn(tensor_shape, dtype=torch.float32)

    block_size, axis = 32, -1

    spec = FP4PerGroupSpec(
        ch_axis=axis,
        group_size=block_size,
        scale_format="e8m0",
        scale_calculation_mode=scale_calculation_mode,
        is_dynamic=False,
    ).to_quantization_spec()
    quantizer = FakeQuantizeBase.get_fake_quantize(spec)

    def pipeline():
        scaled = quantizer(x.clone())
        non_scaled = fake_quantize_mx(
            input_tensor=x.clone(),
            axis=axis,
            block_size=block_size,
            mx_element_dtype=dtype,
            scale_calculation_mode=scale_calculation_mode,
        )
        return scaled, non_scaled

    scaled_fake_quantize, non_scaled_fake_quantize = run_torch_op_variants(pipeline)
    assert torch.allclose(scaled_fake_quantize, non_scaled_fake_quantize)


@pytest.mark.parametrize("scale_format", [("e4m3"), ("float32")])
def test_fp4_per_group_scale(scale_format):
    tensor_shape = (4096, 4096)
    x: torch.Tensor = torch.randn(tensor_shape, dtype=torch.float32)

    block_size, axis = 32, -1
    spec = FP4PerGroupSpec(
        ch_axis=axis, group_size=block_size, scale_format=scale_format, is_dynamic=False
    ).to_quantization_spec()
    quantizer = FakeQuantizeBase.get_fake_quantize(spec)

    run_torch_op_variants(lambda: quantizer(x.clone()))


@pytest.mark.parametrize(
    "scheme,group_size",
    [
        ("amdfp4_global16", 16),
        ("amdfp4_global32", 32),
    ],
)
def test_amdfp4_global_scale_scheme(scheme, group_size):
    """Test amdfp4 global scale schemes with FP8 E5M3 scale quantization.

    This test verifies that the amdfp4 global scale schemes are correctly configured
    with FP4 per-group quantization and FP8 E5M3 per-tensor scale quantization.
    """
    # Use a tiny model for fast testing
    MODEL_NAME = "amd-quark/tiny-llama-fast-tokenizer"

    # Load model
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, **FROM_PRETRAINED_KWARGS).eval().to(torch_device)

    # Create simple dataloader
    input_ids = torch.ones(1, 16, dtype=torch.long, device=torch_device)
    dataloader = DataLoader([input_ids], batch_size=1)

    # Get config from template
    template = LLMTemplate.get(model_type=model.config.model_type)
    config = template.get_config(scheme=scheme)

    # Verify configuration
    assert config.global_quant_config.weight is not None
    assert config.global_quant_config.input_tensors is not None

    # Weight: Two-stage quantization (FP4 per-group with FP8 E5M3 scale, static)
    assert isinstance(config.global_quant_config.weight, list)
    assert len(config.global_quant_config.weight) == 2

    # First stage: FP4 per-group
    weight_first_stage = config.global_quant_config.weight[0]
    assert weight_first_stage.dtype == Dtype.fp4
    assert weight_first_stage.qscheme == QSchemeType.per_group
    assert weight_first_stage.group_size == group_size
    assert weight_first_stage.is_dynamic is False
    assert weight_first_stage.ch_axis == -1
    assert weight_first_stage.is_scale_quant is False

    # Second stage: FP8 E5M3 scale quantization
    weight_second_stage = config.global_quant_config.weight[1]
    assert weight_second_stage.dtype == Dtype.fp8_e5m3
    assert weight_second_stage.qscheme == QSchemeType.per_tensor
    assert weight_second_stage.scale_type == ScaleType.float32
    assert weight_second_stage.is_dynamic is False
    assert weight_second_stage.is_scale_quant is True

    # Activation: Two-stage quantization (FP4 per-group with FP8 E5M3 scale, dynamic)
    assert isinstance(config.global_quant_config.input_tensors, list)
    assert len(config.global_quant_config.input_tensors) == 2

    # First stage: FP4 per-group (dynamic)
    input_first_stage = config.global_quant_config.input_tensors[0]
    assert input_first_stage.dtype == Dtype.fp4
    assert input_first_stage.qscheme == QSchemeType.per_group
    assert input_first_stage.group_size == group_size
    assert input_first_stage.is_dynamic is True
    assert input_first_stage.ch_axis == -1
    assert input_first_stage.is_scale_quant is False

    # Second stage: FP8 E5M3 scale quantization (dynamic)
    input_second_stage = config.global_quant_config.input_tensors[1]
    assert input_second_stage.dtype == Dtype.fp8_e5m3
    assert input_second_stage.qscheme == QSchemeType.per_tensor
    assert input_second_stage.is_dynamic is False
    assert input_second_stage.is_scale_quant is True
    assert input_second_stage.scale_type == ScaleType.float32

    # Quantize model to ensure the scheme works end-to-end
    quantizer = ModelQuantizer(config)
    quant_model = quantizer.quantize_model(model, dataloader)

    # Verify quantized model is functional
    assert quant_model is not None

    # Verify model can perform forward pass
    with torch.no_grad():
        output = quant_model(input_ids)

    assert output.logits is not None
    assert output.logits.shape == (1, 16, model.config.vocab_size)


@pytest.mark.parametrize("group_size", [16, 32])
def test_real_quantize_float8_e5m3_global_scale(group_size):
    """amdfp4_global: real_quantize_float8_e5m3 stores round_e5m3(block_scale /
    global scale), so dequantizing and multiplying the global scale back recovers
    block_scale.
    """
    num_groups = 4096 // group_size
    block_scale = torch.rand(1, num_groups, dtype=torch.float32) * 0.02 + 1e-3
    # per-tensor global scale: map the largest block scale to the E5M3 max.
    global_scale = block_scale.max() / BaseFP8_E5M3.max_value

    stored = real_quantize_float8_e5m3(block_scale, scale=global_scale)
    recovered = dequantize_float8_e5m3_func(stored) * global_scale

    # Effective per-group scale is recovered up to a single E5M3 rounding step.
    assert torch.allclose(recovered, block_scale, rtol=2**-3)

    # Pre-fix control: without the global scale the block scale is cast directly
    # (no normalization), so the stored value differs from the two-stage one.
    stored_pre_fix = real_quantize_float8_e5m3(block_scale)
    assert not torch.equal(stored, stored_pre_fix)
