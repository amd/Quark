#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from tempfile import TemporaryDirectory

import torch
from transformers import AutoConfig, AutoModelForCausalLM

import quark
from quark.common.utils.testing_utils import skip_if_no_gpu, torch_device
from quark.torch import ModelQuantizer, export_safetensors, import_model_from_safetensors
from quark.torch.export.nn.modules.realquantizer import (
    DynamicScaledQuantizer,
    SequentialRealQuantizer,
    StaticScaledRealQuantizer,
    get_real_quantizer,
)
from quark.torch.quantization.config.config import (
    FP4PerGroupSpec,
    FP8E4M3PerTensorSpec,
    Int4PerTensorSpec,
    Int8PerChannelSpec,
    OCP_MXFP4Spec,
    QConfig,
    QLayerConfig,
    QTensorConfig,
    ScaleQuantSpec,
)
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import PerChannelMinMaxObserver, PerTensorMinMaxObserver
from quark.torch.quantization.post_calibration import realign_single_sequential_quantizer
from quark.torch.quantization.tensor_quantize import SequentialQuantize
from quark.torch.utils import getattr_recursive


@skip_if_no_gpu
def test_fp4_per_group_fp8_per_tensor_scale_real_quantize():
    fp4_real_quantizer = StaticScaledRealQuantizer(
        qspec=FP4PerGroupSpec(ch_axis=-1, group_size=5, is_dynamic=False).to_quantization_spec(),
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=torch.float32,
        device=torch.device("cuda"),
        scale_shape=[10, 2],
        zero_point_shape=None,
    )
    scale1 = torch.tensor(
        [
            [4.0000, 5.5000],
            [7.5000, 10.0000],
            [6.5000, 2.2500],
            [1.7500, 0.7500],
            [1.3750, 5.5000],
            [3.5000, 1.2500],
            [6.0000, 0.5625],
            [1.0000, 1.1250],
            [1.2500, 3.5000],
            [0.5625, 6.0000],
        ],
        device="cuda",
        dtype=torch.float8_e4m3fn,
    )
    fp4_real_quantizer.scale = scale1

    fp8_qspec = FP8E4M3PerTensorSpec(is_dynamic=False).to_quantization_spec()
    fp8_qspec.is_scale_quant = True
    fp8_real_quantizer = StaticScaledRealQuantizer(
        qspec=fp8_qspec,
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=torch.float32,
        device=torch.device("cuda"),
        scale_shape=[1],
        zero_point_shape=None,
    )
    scale2 = torch.tensor([13.3567], device="cuda")
    fp8_real_quantizer.scale = scale2

    fp4_fp8_quantizer = SequentialRealQuantizer(fp4_real_quantizer, fp8_real_quantizer)
    input_tensor = torch.tensor(
        [
            [23, 129, 202, 45, 88],
            [241, 175, 15, 193, 158],
            [92, 37, 240, 121, 50],
            [3, 212, 67, 142, 179],
            [234, 10, 189, 105, 246],
            [81, 152, 220, 53, 166],
            [7, 131, 28, 199, 74],
            [160, 115, 238, 39, 208],
            [96, 181, 62, 147, 224],
            [19, 173, 84, 227, 107],
        ],
        dtype=torch.uint8,
        device="cuda",
    )
    output_tensor = fp4_fp8_quantizer(input_tensor)

    x = fp4_real_quantizer.unpack_tensor(input_tensor)
    fp4_scale, fp4_zero_point = fp4_real_quantizer.unpack_params()
    fp8_scale, fp8_zero_point = fp8_real_quantizer.unpack_params()

    fp4_scale = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        fp8_real_quantizer.qspec.dtype.value,
        fp4_scale.to(fp8_real_quantizer.float_dtype),
        fp8_scale,
        fp8_zero_point,
        fp8_real_quantizer.qspec.ch_axis,
        fp8_real_quantizer.qspec.group_size,
        fp8_real_quantizer.qspec.qscheme.value,
    )

    golden_tensor = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        fp4_real_quantizer.qspec.dtype.value,
        x.to(fp4_real_quantizer.float_dtype),
        fp4_scale,
        fp4_zero_point,
        fp4_real_quantizer.qspec.ch_axis,
        fp4_real_quantizer.qspec.group_size,
        fp4_real_quantizer.qspec.qscheme.value,
    )

    assert torch.equal(output_tensor, golden_tensor)


@skip_if_no_gpu
def test_fp8_int4_perchannel_quantize():
    DEFAULT_FP8_PER_TENSOR_SYM_SPEC = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    DEFAULT_INT4_PER_CHANNEL_SYM_SPEC = QTensorConfig(
        dtype=Dtype.int4,
        qscheme=QSchemeType.per_channel,
        observer_cls=PerChannelMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        ch_axis=0,
        is_dynamic=False,
    )

    fp8_real_quantizer = StaticScaledRealQuantizer(
        qspec=DEFAULT_FP8_PER_TENSOR_SYM_SPEC,
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=torch.float32,
        device=torch.device("cuda"),
        scale_shape=[1],
        zero_point_shape=None,
    )
    scale1 = torch.tensor([13.3567], device="cuda")
    fp8_real_quantizer.scale = scale1

    int4_real_quantizer = StaticScaledRealQuantizer(
        qspec=DEFAULT_INT4_PER_CHANNEL_SYM_SPEC,
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=torch.float32,
        device=torch.device("cuda"),
        scale_shape=[10],
        zero_point_shape=[10],
    )
    scale2 = torch.tensor(
        [4.0624, 5.7138, 7.5172, 9.7354, 6.5853, 2.2768, 1.7770, 0.7387, 1.3623, 5.6237], device="cuda"
    )
    int4_real_quantizer.scale = scale2
    zero_point2 = torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0], device="cuda")
    int4_real_quantizer.zero_point = zero_point2

    fp8_int4_quantizer = SequentialRealQuantizer(fp8_real_quantizer, int4_real_quantizer)
    input_tensor = torch.tensor(
        [
            [-2147483648, 2147483647],
            [-123456789, 987654321],
            [-9999999, 88888888],
            [-42, 42],
            [20230101, -20230101],
            [10000000, -100000000],
            [7654321, -876543210],
            [-1234567, 123456789],
            [33333333, -444444444],
            [0, -2147483647],
        ],
        dtype=torch.int32,
        device="cuda",
    )
    output_tensor = fp8_int4_quantizer(input_tensor)

    x = int4_real_quantizer.unpack_tensor(input_tensor)
    int4_scale, int4_zero_point = int4_real_quantizer.unpack_params()
    int4_dequant = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        int4_real_quantizer.qspec.dtype.value,
        x.to(int4_real_quantizer.float_dtype),
        int4_scale,
        int4_zero_point,
        int4_real_quantizer.qspec.ch_axis,
        int4_real_quantizer.qspec.group_size,
        int4_real_quantizer.qspec.qscheme.value,
    )

    fp8_unpack = fp8_real_quantizer.unpack_tensor(int4_dequant)
    fp8_scale, fp8_zero_point = fp8_real_quantizer.unpack_params()
    golden_tensor = quark.torch.kernel.dequantize(  # type: ignore[attr-defined]
        fp8_real_quantizer.qspec.dtype.value,
        fp8_unpack.to(fp8_real_quantizer.float_dtype),
        fp8_scale,
        fp8_zero_point,
        fp8_real_quantizer.qspec.ch_axis,
        fp8_real_quantizer.qspec.group_size,
        fp8_real_quantizer.qspec.qscheme.value,
    )

    assert torch.equal(output_tensor, golden_tensor)


@skip_if_no_gpu
def test_e8m0_scale_pack_unpack():
    fp4_e8m0_quantizer = StaticScaledRealQuantizer(
        qspec=OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec(),
        quantizer=None,
        reorder=False,
        real_quantized=True,
        float_dtype=torch.float32,
        device=torch.device("cuda"),
        scale_shape=[2, 2],
        zero_point_shape=None,
    )

    scale_float = torch.tensor([[1.0, 0.5], [2.0, 0.25]], device="cuda", dtype=torch.float32)
    fp4_e8m0_quantizer.scale = scale_float
    fp4_e8m0_quantizer.maybe_convert_and_transpose_scale()

    golden_scale = torch.tensor([[127, 126], [128, 125]], device="cuda", dtype=torch.uint8)
    assert torch.equal(fp4_e8m0_quantizer.scale, golden_scale)

    scale, _ = fp4_e8m0_quantizer.unpack_params()
    assert torch.equal(scale, scale_float)


def test_freeze_export_reload():
    transformers_config = AutoConfig.from_pretrained("HuggingFaceTB/SmolLM-135M")

    weight_specs = [
        FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec(),
        Int4PerTensorSpec(
            observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
        ).to_quantization_spec(),
        Int8PerChannelSpec(
            symmetric=True, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
        ).to_quantization_spec(),
        OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False, scale_calculation_mode="even").to_quantization_spec(),
    ]

    for weight_spec in weight_specs:
        model = AutoModelForCausalLM.from_config(transformers_config)
        model = model.eval()
        model = model.to(torch_device)

        print("----- weight_spec:", weight_spec)
        global_quant_config = QLayerConfig(weight=weight_spec)
        quant_config = QConfig(global_quant_config=global_quant_config, exclude=["lm_head"])

        state_dict = model.state_dict()

        quantizer = ModelQuantizer(quant_config)
        quant_model = quantizer.quantize_model(model)
        quant_model = quantizer.freeze(quant_model)
        model = model.eval()
        model = model.to(torch.float32)

        model.generation_config.pad_token_id = 1  # just to bypass a bug in the model.

        state_dict_post_freeze = model.state_dict()

        for name, param in state_dict.items():
            if "lm_head" not in name and "embed_tokens" not in name and "norm" not in name:
                assert not torch.equal(param, state_dict_post_freeze[name])

        with TemporaryDirectory() as tmpdir:
            export_safetensors(
                model=quant_model, output_dir=tmpdir, weight_format="real_quantized", pack_method="reorder"
            )

            with torch.device(torch_device):
                original_model = AutoModelForCausalLM.from_config(transformers_config)

            q_model = import_model_from_safetensors(original_model, model_dir=tmpdir, multi_device=False)

        state_dict_reload = q_model.state_dict()

        for name, _ in state_dict.items():
            if "norm" in name or "lm_head" in name or "embed_tokens" in name:
                continue
            param_reload = state_dict_reload[name]

            param_frozen = state_dict_post_freeze[name]

            quantizer_name = name.replace(".weight", ".weight_quantizer")
            quantizer = getattr_recursive(q_model, quantizer_name)

            weight_dequantized = quantizer(param_reload)

            assert weight_dequantized.dtype == param_frozen.dtype
            absdiff = (weight_dequantized - param_frozen).abs()

            # TODO: should be torch.equal here! This is strictly equal for MXFP4, but not for FP8 or INT. There is likely a bug somewhere.
            assert absdiff.max() < 5e-4


def test_dynamic_scaled_quantizer_uses_input_device():
    """
    Verify that DynamicScaledQuantizer.update_dynamic_params creates the observer
    on the input tensor's device (X.device), not on self.device.
    This prevents device mismatch errors when the quantizer is constructed with
    one device but the input tensor is on a different device.
    """
    qspec = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=True,
    )

    quantizer = DynamicScaledQuantizer(
        qspec=qspec,
        device=torch.device("cpu"),
        float_dtype=torch.float32,
    )

    x_cpu = torch.randn(4, 8, device="cpu")
    output = quantizer(x_cpu)
    assert output.device == x_cpu.device, "Output should be on the same device as input"
    assert output.shape == x_cpu.shape


@skip_if_no_gpu
def test_dynamic_scaled_quantizer_cross_device():
    """
    Verify that a DynamicScaledQuantizer created with device='cpu' can correctly
    process an input tensor on CUDA. The observer should be created on the input's
    device (CUDA), not the quantizer's construction device (CPU).
    """
    qspec = QTensorConfig(
        dtype=Dtype.fp8_e4m3,
        qscheme=QSchemeType.per_tensor,
        observer_cls=PerTensorMinMaxObserver,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=True,
    )

    quantizer = DynamicScaledQuantizer(
        qspec=qspec,
        device=torch.device("cpu"),
        float_dtype=torch.float32,
    )

    x_cuda = torch.randn(4, 8, device="cuda")
    output = quantizer(x_cuda)
    assert output.device == x_cuda.device, "Output should be on CUDA when input is on CUDA"
    assert output.shape == x_cuda.shape


@skip_if_no_gpu
def test_combined_division_real_quantize_export():
    """Test that two-stage FP4+FP8 real-quantized export uses combined division pre-computation."""
    device = torch.device("cuda")
    weight_spec = ScaleQuantSpec(
        first_stage=FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=False),
        second_stage=FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False),
    ).to_quantization_spec()

    # Create and calibrate SequentialQuantize
    sequential_quantize = SequentialQuantize(weight_spec, device)
    sequential_quantize.enable_observer()
    weight_tensor = torch.randn(64, 64, dtype=torch.bfloat16, device=device)
    sequential_quantize(weight_tensor)
    # Align first-stage block scales after calibration (same step as ModelQuantizer.quantize_model).
    realign_single_sequential_quantizer(sequential_quantize)

    # Create real quantizer and consume the pre-computed combined-division cache.
    real_quantizer = get_real_quantizer(
        qspec=weight_spec,
        quantizer=sequential_quantize,
        reorder=True,
        real_quantized=True,
        float_dtype=torch.float32,
        device=device,
    )

    assert isinstance(real_quantizer, SequentialRealQuantizer)
    # Verify pre-computed attributes exist
    assert hasattr(real_quantizer[0], "_quantized_block_scale"), "Pre-computed FP8 block scale not found"
    assert real_quantizer[0]._quantized_block_scale.dtype == torch.float8_e4m3fn

    # Run to_real_quantize_params — should use pre-computed scale (skip to_fake_quantize_params)
    packed_weight = real_quantizer.to_real_quantize_params(weight_tensor)
    assert not packed_weight.isnan().any(), "Packed weight contains NaN"

    # Run maybe_convert_and_transpose_scale — should pack the pre-computed FP8 block scale
    real_quantizer.maybe_convert_and_transpose_scale()
    # After packing, scale should be in FP8 format
    assert real_quantizer[0].scale.dtype == torch.float8_e4m3fn


@skip_if_no_gpu
def test_combined_division_multi_stage_realign_adjacent_scale_only():
    """Test that multi-stage realign only uses the immediate following scale stage."""
    device = torch.device("cuda")
    quantization_spec_list = [
        FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=False).to_quantization_spec(),
        FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec(),
        FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=False).to_quantization_spec(),
        FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=False).to_quantization_spec(),
        FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec(),
    ]
    quantization_spec_list[1].is_scale_quant = True
    quantization_spec_list[4].is_scale_quant = True

    sequential_quantize = SequentialQuantize(quantization_spec_list, device)
    sequential_quantize.enable_observer()
    weight_tensor = torch.randn(64, 64, dtype=torch.bfloat16, device=device)
    sequential_quantize(weight_tensor)

    has_realign_action = realign_single_sequential_quantizer(sequential_quantize)
    assert has_realign_action is True
    assert hasattr(sequential_quantize[0], "_quantized_block_scale")
    assert not hasattr(sequential_quantize[2], "_quantized_block_scale")
    assert hasattr(sequential_quantize[3], "_quantized_block_scale")
    assert sequential_quantize[0].scale.dtype == torch.float32
    assert sequential_quantize[3].scale.dtype == torch.float32


@skip_if_no_gpu
def test_combined_division_zero_fill_after_fp8_cast():
    """Test that FP8 underflow zeros are correctly filled after dtype cast."""
    device = torch.device("cuda")
    weight_spec = ScaleQuantSpec(
        first_stage=FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=False),
        second_stage=FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False),
    ).to_quantization_spec()

    sequential_quantize = SequentialQuantize(weight_spec, device)
    sequential_quantize.enable_observer()
    # Use very small values to trigger FP8 underflow
    weight_tensor = torch.randn(32, 32, dtype=torch.bfloat16, device=device) * 1e-6
    sequential_quantize(weight_tensor)
    # Align first-stage block scales after calibration (same step as ModelQuantizer.quantize_model).
    realign_single_sequential_quantizer(sequential_quantize)

    real_quantizer = get_real_quantizer(
        qspec=weight_spec,
        quantizer=sequential_quantize,
        reorder=True,
        real_quantized=True,
        float_dtype=torch.float32,
        device=device,
    )

    # The effective scale should not contain zeros (which would cause NaN)
    assert (real_quantizer[0].scale != 0).all(), "Effective scale contains zeros after FP8 underflow"
    # The packed weight should not contain NaN
    packed_weight = real_quantizer.to_real_quantize_params(weight_tensor)
    assert not packed_weight.isnan().any(), "NaN in packed weight from FP8 underflow"


@skip_if_no_gpu
def test_combined_division_skipped_for_fake_quantized():
    """Test that combined division pre-computation is skipped for fake_quantized mode."""
    device = torch.device("cuda")
    weight_spec = ScaleQuantSpec(
        first_stage=FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=False),
        second_stage=FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False),
    ).to_quantization_spec()

    sequential_quantize = SequentialQuantize(weight_spec, device)
    sequential_quantize.enable_observer()
    weight_tensor = torch.randn(32, 32, dtype=torch.bfloat16, device=device)
    sequential_quantize(weight_tensor)

    # Create with real_quantized=False — should NOT pre-compute
    real_quantizer = get_real_quantizer(
        qspec=weight_spec,
        quantizer=sequential_quantize,
        reorder=True,
        real_quantized=False,
        float_dtype=torch.float32,
        device=device,
    )

    assert isinstance(real_quantizer, SequentialRealQuantizer)
    assert not hasattr(real_quantizer[0], "_quantized_block_scale"), (
        "Pre-computation should be skipped for fake_quantized"
    )


@skip_if_no_gpu
def test_sequential_quantize_observer_standard_path():
    """Test that non-FP8 scale quantizers use the standard observer path."""
    device = torch.device("cuda")
    # Use a config where the second stage does NOT have quant_max_first_level
    # (e.g., FP4 with float32 scale, no FP8 second stage)
    weight_spec = ScaleQuantSpec(
        first_stage=FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=False, scale_format="float32"),
        second_stage=FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False),
    ).to_quantization_spec()

    sequential_quantize = SequentialQuantize(weight_spec, device)
    sequential_quantize.enable_observer()
    weight_tensor = torch.randn(32, 32, dtype=torch.bfloat16, device=device)

    # Run forward — observer mode processes scale quantizer
    result = sequential_quantize(weight_tensor)
    assert not result.isnan().any(), "Observer mode produced NaN"
    # Verify the second-stage scale was computed
    assert sequential_quantize[1].scale.numel() > 0


@skip_if_no_gpu
def test_sequential_real_quantizer_mixed_dynamic():
    device = torch.device("cuda")
    activation_spec = ScaleQuantSpec(
        first_stage=FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=True, scale_type="float32"),
        second_stage=FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False, scale_type="float32"),
    ).to_quantization_spec()

    sequential_quantize = SequentialQuantize(activation_spec, device)
    sequential_quantize.enable_observer()
    activation_tensor = torch.randn(32, 32, dtype=torch.bfloat16, device=device)
    sequential_quantize(activation_tensor)

    real_quantizer = get_real_quantizer(
        qspec=activation_spec,
        quantizer=sequential_quantize,
        reorder=False,
        real_quantized=False,
        float_dtype=torch.float32,
        device=device,
    )

    assert isinstance(real_quantizer, SequentialRealQuantizer)
    assert real_quantizer.is_dynamic is True
    assert real_quantizer[0].is_dynamic is True
    assert real_quantizer[1].is_dynamic is False

    output_tensor = real_quantizer(activation_tensor)
    assert output_tensor.shape == activation_tensor.shape
    assert not output_tensor.isnan().any(), "Mixed dynamic SequentialRealQuantizer produced NaN"
    state_dict_keys = set(real_quantizer.state_dict().keys())
    assert "0.scale" not in state_dict_keys
    assert "0.zero_point" not in state_dict_keys
    assert "1.scale" in state_dict_keys
    assert "1.zero_point" not in state_dict_keys


@skip_if_no_gpu
def test_sequential_real_quantizer_dynamic_scale_stage():
    device = torch.device("cuda")
    activation_spec = ScaleQuantSpec(
        first_stage=FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=True, scale_type="float32"),
        second_stage=FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=True, scale_type="float32"),
    ).to_quantization_spec()

    sequential_quantize = SequentialQuantize(activation_spec, device)
    sequential_quantize.enable_observer()
    activation_tensor = torch.randn(32, 32, dtype=torch.bfloat16, device=device)
    sequential_quantize(activation_tensor)

    real_quantizer = get_real_quantizer(
        qspec=activation_spec,
        quantizer=sequential_quantize,
        reorder=False,
        real_quantized=False,
        float_dtype=torch.float32,
        device=device,
    )

    assert isinstance(real_quantizer, SequentialRealQuantizer)
    assert real_quantizer.is_dynamic is True
    assert real_quantizer[0].is_dynamic is True
    assert real_quantizer[1].is_dynamic is True

    output_tensor = real_quantizer(activation_tensor)
    assert output_tensor.shape == activation_tensor.shape
    assert not output_tensor.isnan().any(), "Dynamic scale SequentialRealQuantizer produced NaN"
    state_dict_keys = set(real_quantizer.state_dict().keys())
    assert "0.scale" not in state_dict_keys
    assert "0.zero_point" not in state_dict_keys
    assert "1.scale" not in state_dict_keys
    assert "1.zero_point" not in state_dict_keys
