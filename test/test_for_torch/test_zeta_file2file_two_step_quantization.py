#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Tests for file-to-file quantization (quantize_model_per_safetensor).

This module tests both single-stage and progressive (two-step) weight quantization
in the file2file pipeline, which processes safetensors files without loading
the full model into GPU memory.
"""

import json
import os
import tempfile

import torch

from quark.common.utils.import_utils import is_safetensors_available
from quark.common.utils.testing_utils import skip_if_no_gpu, torch_device
from quark.torch.quantization import (
    FP4PerGroupSpec,
    FP8E4M3PerTensorSpec,
    Int4PerChannelSpec,
    Int4PerTensorSpec,
    Int8PerChannelSpec,
    ProgressiveSpec,
    QConfig,
    QLayerConfig,
    ScaleQuantSpec,
)
from quark.torch.quantization.file2file_quantization import quantize_model_per_safetensor

if is_safetensors_available():
    from safetensors import safe_open
    from safetensors.torch import save_file

TEST_RANDOM_SEED = 42


def _create_fake_model_directory(
    model_directory: str,
    tensor_shape: tuple[int, int],
    num_layers: int,
    dtype: torch.dtype,
) -> None:
    """
    Create a fake model directory with safetensors files and config.json.

    This creates a minimal model directory structure with:
    - A single safetensors file containing weight tensors named ``layer_N.linear.weight``
      and a non-quantizable tensor ``embedding.weight``.
    - A ``config.json`` with ``torch_dtype`` and ``model_type`` fields.
    - A ``model.safetensors.index.json`` mapping tensor names to the safetensors file.

    :param str model_directory: Path to the directory to create the model in.
    :param tuple[int, int] tensor_shape: Shape of each weight tensor (rows, columns).
    :param int num_layers: Number of linear layers to create.
    :param torch.dtype dtype: Data type for the weight tensors.

    :return: None
    """
    safetensor_filename = "model-00001-of-00001.safetensors"
    random_generator = torch.Generator()
    random_generator.manual_seed(TEST_RANDOM_SEED)
    tensors = {}
    weight_map = {}

    for layer_index in range(num_layers):
        tensor_name = f"layer_{layer_index}.linear.weight"
        tensors[tensor_name] = torch.randn(tensor_shape, dtype=dtype, generator=random_generator)
        weight_map[tensor_name] = safetensor_filename

    # Add a non-quantizable tensor (embedding, which should be skipped)
    tensors["embedding.weight"] = torch.randn(100, tensor_shape[1], dtype=dtype, generator=random_generator)
    weight_map["embedding.weight"] = safetensor_filename

    # Add a norm tensor (should be skipped)
    tensors["layer_0.norm.weight"] = torch.randn(tensor_shape[1], dtype=dtype, generator=random_generator)
    weight_map["layer_0.norm.weight"] = safetensor_filename

    save_file(tensors, os.path.join(model_directory, safetensor_filename))

    # Determine the torch_dtype string from the dtype
    dtype_string_map = {
        torch.float32: "float32",
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
    }
    torch_dtype_string = dtype_string_map.get(dtype, "float32")

    config = {
        "model_type": "test",
        "torch_dtype": torch_dtype_string,
    }
    with open(os.path.join(model_directory, "config.json"), "w") as config_file:
        json.dump(config, config_file)

    index_data = {
        "metadata": {"total_size": 0},
        "weight_map": weight_map,
    }
    with open(os.path.join(model_directory, "model.safetensors.index.json"), "w") as index_file:
        json.dump(index_data, index_file)


@skip_if_no_gpu
def test_file2file_single_stage_int4_per_tensor():
    """
    Test single-stage INT4 per-tensor quantization via file-to-file pipeline.

    Verifies that:
    - Output safetensors files are created.
    - Quantized weight tensors have ``_scale`` suffix tensors.
    - Non-quantizable tensors (embedding, norm) are copied as-is.
    - Output model.safetensors.index.json is created with correct entries.
    """
    num_layers = 4
    # INT4 per-tensor requires column count divisible by 8 for packing
    tensor_shape = (32, 16)

    int4_per_tensor_spec = Int4PerTensorSpec(is_dynamic=False).to_quantization_spec()
    quantization_layer_config = QLayerConfig(weight=int4_per_tensor_spec)
    quantization_config = QConfig(global_quant_config=quantization_layer_config)

    with tempfile.TemporaryDirectory() as input_directory, tempfile.TemporaryDirectory() as output_directory:
        _create_fake_model_directory(
            model_directory=input_directory,
            tensor_shape=tensor_shape,
            num_layers=num_layers,
            dtype=torch.float16,
        )

        quantize_model_per_safetensor(
            pretrained_model_path=input_directory,
            quant_config=quantization_config,
            save_path=output_directory,
            keep_excluded_layers_as_original_model_state=False,
            device=torch_device,
        )

        # Verify output safetensors file exists
        output_safetensor_path = os.path.join(output_directory, "model-00001-of-00001.safetensors")
        assert os.path.exists(output_safetensor_path), "Output safetensors file should exist"

        # Verify output index file exists
        output_index_path = os.path.join(output_directory, "model.safetensors.index.json")
        assert os.path.exists(output_index_path), "Output index file should exist"

        # Check output tensors
        with safe_open(output_safetensor_path, framework="pt", device="cpu") as safetensor_file:
            output_keys = set(safetensor_file.keys())

            # Each linear weight should have a packed weight and a scale
            for layer_index in range(num_layers):
                weight_key = f"layer_{layer_index}.linear.weight"
                scale_key = f"layer_{layer_index}.linear.weight_scale"
                assert weight_key in output_keys, f"Packed weight {weight_key} should exist"
                assert scale_key in output_keys, f"Scale {scale_key} should exist"

            # Non-quantizable tensors should be copied as-is
            assert "embedding.weight" in output_keys, "Embedding weight should be copied as-is"
            assert "layer_0.norm.weight" in output_keys, "Norm weight should be copied as-is"

        # Verify index file has correct weight_map
        with open(output_index_path) as index_file:
            index_data = json.load(index_file)
        output_weight_map = index_data["weight_map"]
        for layer_index in range(num_layers):
            assert f"layer_{layer_index}.linear.weight" in output_weight_map
            assert f"layer_{layer_index}.linear.weight_scale" in output_weight_map


@skip_if_no_gpu
def test_file2file_single_stage_int8_per_channel():
    """
    Test single-stage INT8 per-channel quantization via file-to-file pipeline.

    Verifies that INT8 per-channel quantization produces valid packed weights and scales.
    """
    num_layers = 2
    tensor_shape = (32, 16)

    int8_per_channel_spec = Int8PerChannelSpec(ch_axis=0, is_dynamic=False).to_quantization_spec()
    quantization_layer_config = QLayerConfig(weight=int8_per_channel_spec)
    quantization_config = QConfig(global_quant_config=quantization_layer_config)

    with tempfile.TemporaryDirectory() as input_directory, tempfile.TemporaryDirectory() as output_directory:
        _create_fake_model_directory(
            model_directory=input_directory,
            tensor_shape=tensor_shape,
            num_layers=num_layers,
            dtype=torch.float16,
        )

        quantize_model_per_safetensor(
            pretrained_model_path=input_directory,
            quant_config=quantization_config,
            save_path=output_directory,
            keep_excluded_layers_as_original_model_state=False,
            device=torch_device,
        )

        output_safetensor_path = os.path.join(output_directory, "model-00001-of-00001.safetensors")
        with safe_open(output_safetensor_path, framework="pt", device="cpu") as safetensor_file:
            for layer_index in range(num_layers):
                weight_key = f"layer_{layer_index}.linear.weight"
                scale_key = f"layer_{layer_index}.linear.weight_scale"
                assert weight_key in set(safetensor_file.keys()), f"Packed weight {weight_key} should exist"
                assert scale_key in set(safetensor_file.keys()), f"Scale {scale_key} should exist"

                # For per-channel INT8, scale should have one value per output channel
                scale_tensor = safetensor_file.get_tensor(scale_key)
                assert scale_tensor.shape[0] == tensor_shape[0], (
                    f"Per-channel scale should have {tensor_shape[0]} values, got {scale_tensor.shape[0]}"
                )


@skip_if_no_gpu
def test_file2file_progressive_fp8_to_int4():
    """
    Test progressive (two-step) quantization via file-to-file pipeline.

    Uses ProgressiveSpec with FP8 per-tensor as first stage and INT4 per-channel
    as second stage. Verifies that:
    - Output contains packed weight, first stage scale (``_scale``), and second stage scale (``_scale_2``).
    - First stage scale is a scalar (per-tensor FP8).
    - Second stage scale has one value per output channel (per-channel INT4).
    - Non-quantizable tensors are copied as-is.
    """
    num_layers = 3
    # INT4 per-channel with ch_axis=0 requires column count divisible by 8 for packing
    tensor_shape = (32, 16)

    progressive_weight_spec = ProgressiveSpec(
        first_stage=FP8E4M3PerTensorSpec(observer_method="min_max", scale_type="float", is_dynamic=False),
        second_stage=Int4PerChannelSpec(
            symmetric=True,
            scale_type="float",
            round_method="half_even",
            is_dynamic=False,
            ch_axis=0,
        ),
    ).to_quantization_spec()

    quantization_layer_config = QLayerConfig(weight=progressive_weight_spec)
    quantization_config = QConfig(global_quant_config=quantization_layer_config)

    with tempfile.TemporaryDirectory() as input_directory, tempfile.TemporaryDirectory() as output_directory:
        _create_fake_model_directory(
            model_directory=input_directory,
            tensor_shape=tensor_shape,
            num_layers=num_layers,
            dtype=torch.float16,
        )

        quantize_model_per_safetensor(
            pretrained_model_path=input_directory,
            quant_config=quantization_config,
            save_path=output_directory,
            keep_excluded_layers_as_original_model_state=False,
            device=torch_device,
        )

        output_safetensor_path = os.path.join(output_directory, "model-00001-of-00001.safetensors")
        assert os.path.exists(output_safetensor_path), "Output safetensors file should exist"

        with safe_open(output_safetensor_path, framework="pt", device="cpu") as safetensor_file:
            output_keys = set(safetensor_file.keys())

            for layer_index in range(num_layers):
                weight_key = f"layer_{layer_index}.linear.weight"
                first_stage_scale_key = f"layer_{layer_index}.linear.weight_scale"
                second_stage_scale_key = f"layer_{layer_index}.linear.weight_scale_2"

                assert weight_key in output_keys, f"Packed weight {weight_key} should exist"
                assert first_stage_scale_key in output_keys, f"First stage scale {first_stage_scale_key} should exist"
                assert second_stage_scale_key in output_keys, (
                    f"Second stage scale {second_stage_scale_key} should exist"
                )

                # First stage is FP8 per-tensor, so scale should be a scalar (1 element)
                first_stage_scale = safetensor_file.get_tensor(first_stage_scale_key)
                assert first_stage_scale.numel() == 1, (
                    f"FP8 per-tensor scale should be scalar, got shape {first_stage_scale.shape}"
                )

                # Second stage is INT4 per-channel (ch_axis=0), so scale has one value per row
                second_stage_scale = safetensor_file.get_tensor(second_stage_scale_key)
                assert second_stage_scale.shape[0] == tensor_shape[0], (
                    f"INT4 per-channel scale should have {tensor_shape[0]} values, got shape {second_stage_scale.shape}"
                )

            # Non-quantizable tensors should be copied as-is
            assert "embedding.weight" in output_keys, "Embedding weight should be copied as-is"
            assert "layer_0.norm.weight" in output_keys, "Norm weight should be copied as-is"

        # Verify weight_map includes all scale keys
        output_index_path = os.path.join(output_directory, "model.safetensors.index.json")
        with open(output_index_path) as index_file:
            index_data = json.load(index_file)
        output_weight_map = index_data["weight_map"]
        for layer_index in range(num_layers):
            assert f"layer_{layer_index}.linear.weight_scale" in output_weight_map
            assert f"layer_{layer_index}.linear.weight_scale_2" in output_weight_map


@skip_if_no_gpu
def test_file2file_exclude_layers():
    """
    Test that excluded layers are not quantized in file-to-file pipeline.

    Verifies that layers matching exclude patterns are copied as-is (no scale tensors).
    """
    num_layers = 4
    tensor_shape = (32, 16)

    int4_per_tensor_spec = Int4PerTensorSpec(is_dynamic=False).to_quantization_spec()
    quantization_layer_config = QLayerConfig(weight=int4_per_tensor_spec)
    # Exclude layer_0 and layer_1
    quantization_config = QConfig(
        global_quant_config=quantization_layer_config,
        exclude=["layer_0.*", "layer_1.*"],
    )

    with tempfile.TemporaryDirectory() as input_directory, tempfile.TemporaryDirectory() as output_directory:
        _create_fake_model_directory(
            model_directory=input_directory,
            tensor_shape=tensor_shape,
            num_layers=num_layers,
            dtype=torch.float16,
        )

        quantize_model_per_safetensor(
            pretrained_model_path=input_directory,
            quant_config=quantization_config,
            save_path=output_directory,
            keep_excluded_layers_as_original_model_state=False,
            device=torch_device,
        )

        output_safetensor_path = os.path.join(output_directory, "model-00001-of-00001.safetensors")
        with safe_open(output_safetensor_path, framework="pt", device="cpu") as safetensor_file:
            output_keys = set(safetensor_file.keys())

            # Excluded layers should NOT have scale tensors
            assert "layer_0.linear.weight_scale" not in output_keys, "Excluded layer_0 should not have scale"
            assert "layer_1.linear.weight_scale" not in output_keys, "Excluded layer_1 should not have scale"

            # Non-excluded layers SHOULD have scale tensors
            assert "layer_2.linear.weight_scale" in output_keys, "layer_2 should be quantized with scale"
            assert "layer_3.linear.weight_scale" in output_keys, "layer_3 should be quantized with scale"


@skip_if_no_gpu
def test_file2file_progressive_with_exclude():
    """
    Test progressive quantization with layer exclusion in file-to-file pipeline.

    Verifies that excluded layers are not quantized while non-excluded layers
    get the full progressive two-step quantization with both scale and scale_2.
    """
    num_layers = 4
    tensor_shape = (32, 16)

    progressive_weight_spec = ProgressiveSpec(
        first_stage=FP8E4M3PerTensorSpec(observer_method="min_max", scale_type="float", is_dynamic=False),
        second_stage=Int4PerChannelSpec(
            symmetric=True,
            scale_type="float",
            round_method="half_even",
            is_dynamic=False,
            ch_axis=0,
        ),
    ).to_quantization_spec()

    quantization_layer_config = QLayerConfig(weight=progressive_weight_spec)
    # Exclude layer_0
    quantization_config = QConfig(
        global_quant_config=quantization_layer_config,
        exclude=["layer_0.*"],
    )

    with tempfile.TemporaryDirectory() as input_directory, tempfile.TemporaryDirectory() as output_directory:
        _create_fake_model_directory(
            model_directory=input_directory,
            tensor_shape=tensor_shape,
            num_layers=num_layers,
            dtype=torch.float16,
        )

        quantize_model_per_safetensor(
            pretrained_model_path=input_directory,
            quant_config=quantization_config,
            save_path=output_directory,
            keep_excluded_layers_as_original_model_state=False,
            device=torch_device,
        )

        output_safetensor_path = os.path.join(output_directory, "model-00001-of-00001.safetensors")
        with safe_open(output_safetensor_path, framework="pt", device="cpu") as safetensor_file:
            output_keys = set(safetensor_file.keys())

            # Excluded layer should NOT have any scale tensors
            assert "layer_0.linear.weight_scale" not in output_keys
            assert "layer_0.linear.weight_scale_2" not in output_keys

            # Non-excluded layers should have both scales from progressive quantization
            for layer_index in range(1, num_layers):
                assert f"layer_{layer_index}.linear.weight_scale" in output_keys, (
                    f"layer_{layer_index} should have first stage scale"
                )
                assert f"layer_{layer_index}.linear.weight_scale_2" in output_keys, (
                    f"layer_{layer_index} should have second stage scale"
                )


def _build_nvfp4_weight_spec():
    """
    Build the NVFP4 weight spec (a scale-quant two-stage spec): FP4 per-group
    (group_size 16) weight quantization whose per-group scale is itself quantized
    to FP8-E4M3 per-tensor. This is the spec used by the DeepSeek-V4-Pro pipeline.

    :return: A two-element ``list[QTensorConfig]`` with ``is_scale_quant=True`` on the
        second stage.
    """
    return ScaleQuantSpec(
        first_stage=FP4PerGroupSpec(ch_axis=-1, group_size=16, is_dynamic=False, scale_type="float32"),
        second_stage=FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False, scale_type="float32"),
    ).to_quantization_spec()


def _dequantize_nvfp4_packed(
    packed_weight: torch.Tensor,
    per_group_scale_fp8: torch.Tensor,
    global_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """
    Independent semantic dequantization of a packed NVFP4 weight back to float32.

    This does NOT reuse the file2file quantize code path. It reconstructs the
    effective per-group scale the way an inference engine would:

    1. unpack the U8 nibbles to their FP4 (e2m1) values,
    2. recover the effective per-group scale as ``fp8_block_scale * global_scale``,
    3. multiply each FP4 value by its group's effective scale.

    Comparing the result against the original source weight (within FP4 tolerance)
    validates that the exported tensors actually represent the source weight,
    rather than asserting the exporter equals a copy of itself.

    :param torch.Tensor packed_weight: U8 packed FP4 weight, shape ``[rows, cols / 2]``.
    :param torch.Tensor per_group_scale_fp8: F8_E4M3 per-group block scale, shape ``[rows, cols / group_size]``.
    :param torch.Tensor global_scale: F32 per-tensor global scale (scalar).
    :param int group_size: FP4 group size (16 for NVFP4).

    :return: Dequantized weight as float32, shape ``[rows, cols]``.
    """
    from quark.torch.utils.pack import Pack_fp4

    # Pack_fp4.unpack bit-reinterprets each nibble straight to its FP4 (e2m1)
    # float value -- no value lookup table, so this is independent of how the
    # weight was quantized. ``reorder`` is a required arg but Pack_fp4 ignores it
    # (it does not reorder nibbles), so the value passed is immaterial.
    fp4_values = Pack_fp4(qscheme="per_group", dtype="fp4").unpack(packed_weight, reorder=False).float()

    rows, cols = fp4_values.shape
    num_groups = cols // group_size
    effective_scale = per_group_scale_fp8.float() * global_scale.float()
    fp4_grouped = fp4_values.view(rows, num_groups, group_size)
    scale_grouped = effective_scale.view(rows, num_groups, 1)
    return (fp4_grouped * scale_grouped).view(rows, cols)


@skip_if_no_gpu
def test_file2file_scale_quant_nvfp4():
    """
    Test NVFP4 scale-quant (scale-of-scale) quantization via file-to-file pipeline.

    NVFP4 uses a ``ScaleQuantSpec`` whose second stage quantizes the first stage's
    per-group scale (``is_scale_quant=True``) rather than re-quantizing the weight.
    Verifies the output is the packed NVFP4 wire format:

    - ``weight``: U8-packed FP4 nibbles (inner dim halved).
    - ``weight_scale``: F8_E4M3 per-group block scale.
    - ``weight_scale_2``: F32 per-tensor global scale (scalar).
    """
    num_layers = 2
    # FP4 per-group with group_size 16 requires the inner dim divisible by 16.
    tensor_shape = (32, 32)

    nvfp4_weight_spec = _build_nvfp4_weight_spec()
    quantization_layer_config = QLayerConfig(weight=nvfp4_weight_spec)
    quantization_config = QConfig(global_quant_config=quantization_layer_config)

    with tempfile.TemporaryDirectory() as input_directory, tempfile.TemporaryDirectory() as output_directory:
        _create_fake_model_directory(
            model_directory=input_directory,
            tensor_shape=tensor_shape,
            num_layers=num_layers,
            dtype=torch.bfloat16,
        )

        quantize_model_per_safetensor(
            pretrained_model_path=input_directory,
            quant_config=quantization_config,
            save_path=output_directory,
            keep_excluded_layers_as_original_model_state=False,
            device=torch_device,
        )

        output_safetensor_path = os.path.join(output_directory, "model-00001-of-00001.safetensors")
        assert os.path.exists(output_safetensor_path), "Output safetensors file should exist"

        with safe_open(output_safetensor_path, framework="pt", device="cpu") as safetensor_file:
            output_keys = set(safetensor_file.keys())

            for layer_index in range(num_layers):
                weight_key = f"layer_{layer_index}.linear.weight"
                scale_key = f"layer_{layer_index}.linear.weight_scale"
                scale_2_key = f"layer_{layer_index}.linear.weight_scale_2"

                assert weight_key in output_keys, f"Packed weight {weight_key} should exist"
                assert scale_key in output_keys, f"Per-group scale {scale_key} should exist"
                assert scale_2_key in output_keys, f"Global scale {scale_2_key} should exist"

                # Packed FP4 weight is U8 with the inner dim halved (2 nibbles per byte).
                packed_weight = safetensor_file.get_tensor(weight_key)
                assert packed_weight.dtype == torch.uint8, (
                    f"Packed NVFP4 weight should be uint8, got {packed_weight.dtype}"
                )
                assert packed_weight.shape == (tensor_shape[0], tensor_shape[1] // 2), (
                    f"Packed weight should be {(tensor_shape[0], tensor_shape[1] // 2)}, got {tuple(packed_weight.shape)}"
                )

                # Per-group scale is FP8-E4M3, one value per group of 16.
                per_group_scale = safetensor_file.get_tensor(scale_key)
                assert per_group_scale.dtype == torch.float8_e4m3fn, (
                    f"Per-group scale should be float8_e4m3fn, got {per_group_scale.dtype}"
                )
                assert per_group_scale.shape[-1] == tensor_shape[1] // 16, (
                    f"Per-group scale inner dim should be {tensor_shape[1] // 16}, got {per_group_scale.shape[-1]}"
                )

                # Global scale-of-scale is an F32 scalar.
                global_scale = safetensor_file.get_tensor(scale_2_key)
                assert global_scale.dtype == torch.float32, f"Global scale should be float32, got {global_scale.dtype}"
                assert global_scale.numel() == 1, f"Global scale should be a scalar, got shape {global_scale.shape}"

            # Non-quantizable tensors should be copied as-is.
            assert "embedding.weight" in output_keys, "Embedding weight should be copied as-is"
            assert "layer_0.norm.weight" in output_keys, "Norm weight should be copied as-is"

        # Index weight_map should include both scale keys for every quantized layer.
        output_index_path = os.path.join(output_directory, "model.safetensors.index.json")
        with open(output_index_path) as index_file:
            index_data = json.load(index_file)
        output_weight_map = index_data["weight_map"]
        for layer_index in range(num_layers):
            assert f"layer_{layer_index}.linear.weight_scale" in output_weight_map
            assert f"layer_{layer_index}.linear.weight_scale_2" in output_weight_map


@skip_if_no_gpu
def test_file2file_scale_quant_dequantizes_to_source():
    """
    Acceptance test: the exported NVFP4 tensors actually represent the source
    weight. The packed FP4 weight is dequantized independently (unpack + apply
    ``fp8_block_scale * global_scale``) and compared against the original weight
    within FP4 tolerance. This validates the export against an independent oracle
    rather than against a copy of the exporter.
    """
    num_layers = 1
    # Larger tensor than the smoke test so the tolerance check is meaningful.
    tensor_shape = (64, 128)
    group_size = 16

    nvfp4_weight_spec = _build_nvfp4_weight_spec()
    quantization_layer_config = QLayerConfig(weight=nvfp4_weight_spec)
    quantization_config = QConfig(global_quant_config=quantization_layer_config)

    with tempfile.TemporaryDirectory() as input_directory, tempfile.TemporaryDirectory() as output_directory:
        _create_fake_model_directory(
            model_directory=input_directory,
            tensor_shape=tensor_shape,
            num_layers=num_layers,
            dtype=torch.bfloat16,
        )

        input_safetensor_path = os.path.join(input_directory, "model-00001-of-00001.safetensors")
        with safe_open(input_safetensor_path, framework="pt", device="cpu") as source_file:
            source_weight = source_file.get_tensor("layer_0.linear.weight").float()

        quantize_model_per_safetensor(
            pretrained_model_path=input_directory,
            quant_config=quantization_config,
            save_path=output_directory,
            keep_excluded_layers_as_original_model_state=False,
            device=torch_device,
        )

        output_safetensor_path = os.path.join(output_directory, "model-00001-of-00001.safetensors")
        with safe_open(output_safetensor_path, framework="pt", device="cpu") as safetensor_file:
            packed_weight = safetensor_file.get_tensor("layer_0.linear.weight")
            per_group_scale = safetensor_file.get_tensor("layer_0.linear.weight_scale")
            global_scale = safetensor_file.get_tensor("layer_0.linear.weight_scale_2")

        dequantized = _dequantize_nvfp4_packed(packed_weight, per_group_scale, global_scale, group_size)

        assert dequantized.shape == source_weight.shape, (
            f"Dequantized shape {tuple(dequantized.shape)} should match source {tuple(source_weight.shape)}"
        )
        # NVFP4 (4-bit) is lossy; assert the reconstruction tracks the source.
        # Check cosine similarity per row rather than over the flattened tensor: a
        # transposed scale or wrong group ordering can still score high globally on
        # an i.i.d.-normal weight, but breaks the per-row alignment.
        per_row_cosine = torch.nn.functional.cosine_similarity(dequantized, source_weight, dim=1)
        min_cosine = per_row_cosine.min()
        assert min_cosine > 0.97, (
            f"Every row of the NVFP4 dequantized weight should track the source, got min cos={min_cosine}"
        )


@skip_if_no_gpu
def test_file2file_scale_quant_with_exclude():
    """
    Test that excluded layers are not NVFP4-quantized in the file2file pipeline.

    The scale-quant path writes an extra ``weight_scale_2`` key, so verify an
    excluded layer gets neither ``weight_scale`` nor ``weight_scale_2`` while a
    non-excluded layer gets both.
    """
    num_layers = 3
    tensor_shape = (32, 32)

    nvfp4_weight_spec = _build_nvfp4_weight_spec()
    quantization_config = QConfig(
        global_quant_config=QLayerConfig(weight=nvfp4_weight_spec),
        exclude=["layer_0.*"],
    )

    with tempfile.TemporaryDirectory() as input_directory, tempfile.TemporaryDirectory() as output_directory:
        _create_fake_model_directory(
            model_directory=input_directory,
            tensor_shape=tensor_shape,
            num_layers=num_layers,
            dtype=torch.bfloat16,
        )

        quantize_model_per_safetensor(
            pretrained_model_path=input_directory,
            quant_config=quantization_config,
            save_path=output_directory,
            keep_excluded_layers_as_original_model_state=False,
            device=torch_device,
        )

        output_safetensor_path = os.path.join(output_directory, "model-00001-of-00001.safetensors")
        with safe_open(output_safetensor_path, framework="pt", device="cpu") as safetensor_file:
            output_keys = set(safetensor_file.keys())

            # Excluded layer should have neither scale tensor.
            assert "layer_0.linear.weight_scale" not in output_keys
            assert "layer_0.linear.weight_scale_2" not in output_keys

            # Non-excluded layers should have both NVFP4 scale tensors.
            for layer_index in range(1, num_layers):
                assert f"layer_{layer_index}.linear.weight_scale" in output_keys, (
                    f"layer_{layer_index} should have the per-group scale"
                )
                assert f"layer_{layer_index}.linear.weight_scale_2" in output_keys, (
                    f"layer_{layer_index} should have the global scale"
                )
