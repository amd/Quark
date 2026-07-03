#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Tests for quark.torch.quantization.inverse_quantizer module.

Covers:
- CompressedLinearInverseQuantizer  (compressed-tensors FP8 + INT4 models, >=0.15)
- FP8LinearInverseQuantizer         (transformers FP8Linear models)
- Public API functions: create_inverse_quantizer, is_prequantized_linear,
  dequantize_prequantized_to_linear
- Error handling and edge cases
"""

from __future__ import annotations

import copy
import gc
import shutil
import tempfile
import types
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import huggingface_hub
import pytest
import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.common.utils.import_utils import is_transformers_version_higher_or_equal
from quark.common.utils.testing_utils import TEST_WITH_EXTENSIVE, require_torch_cuda, torch_device
from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors
from quark.torch.export.safetensors import _load_weights_from_safetensors, patch_missing_weights
from quark.torch.quantization import OCP_MXFP4Spec
from quark.torch.quantization.config.config import QConfig, QLayerConfig
from quark.torch.quantization.inverse_quantizer import (
    CompressedLinearInverseQuantizer,
    FP8LinearInverseQuantizer,
    InverseWeightQuantizer,
    create_inverse_quantizer,
    dequantize_prequantized_to_linear,
    is_prequantized_linear,
)
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.utils.llm.model_preparation import get_model

FP8_LINEAR_MODEL_ID = "Qwen/Qwen3-0.6B-FP8"
COMPRESSED_FP8_MODEL_ID = "RedHatAI/Llama-3.2-1B-Instruct-FP8-dynamic"
COMPRESSED_INT4_MODEL_ID = "RedHatAI/Qwen2.5-0.5B-quantized.w4a16"


def _find_first_prequantized_module(model: nn.Module) -> nn.Module | None:
    """Walk the module tree and return the first pre-quantized module."""
    for module in model.modules():
        if is_prequantized_linear(module):
            return module
    return None


def _get_weight(module: nn.Module) -> torch.Tensor:
    """Extract weight (or weight_packed) from a quantized linear module."""
    if hasattr(module, "weight") and module.weight is not None:
        return module.weight
    if hasattr(module, "weight_packed") and module.weight_packed is not None:
        return module.weight_packed
    raise RuntimeError("Cannot find weight in module")


def _assert_valid_dequant(tensor: torch.Tensor, expected_shape: tuple[int, ...]) -> None:
    """Assert tensor has expected shape, float dtype, and all finite values."""
    assert tensor.shape == expected_shape
    assert tensor.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert torch.isfinite(tensor).all()


@pytest.fixture(scope="module")
def compressed_fp8_model() -> Iterator[nn.Module]:
    """Load the compressed-tensors FP8 model once per module."""
    model = AutoModelForCausalLM.from_pretrained(
        COMPRESSED_FP8_MODEL_ID,
        torch_dtype="auto",
        device_map="cpu",
    )
    yield model
    del model
    gc.collect()


@pytest.fixture(scope="module")
def compressed_int4_model() -> Iterator[nn.Module]:
    """Load the compressed-tensors INT4 model once per module."""
    model = AutoModelForCausalLM.from_pretrained(
        COMPRESSED_INT4_MODEL_ID,
        torch_dtype="auto",
        device_map="cpu",
    )
    yield model
    del model
    gc.collect()


@pytest.fixture(scope="module")
@require_torch_cuda
def fp8_linear_model() -> Iterator[nn.Module]:
    """Load the transformers FP8Linear model once per module (requires CUDA)."""
    model = AutoModelForCausalLM.from_pretrained(
        FP8_LINEAR_MODEL_ID,
        torch_dtype="auto",
        device_map="auto",
    )
    yield model
    del model
    gc.collect()
    torch.cuda.empty_cache()


def test_fp8_dequantize_and_to_linear(compressed_fp8_model: nn.Module) -> None:
    """Verify dequantize, dequantize_to_linear, and is_prequantized for compressed-tensors FP8."""
    module = _find_first_prequantized_module(compressed_fp8_model)
    assert module is not None
    assert is_prequantized_linear(module) is True

    inv = create_inverse_quantizer(module)
    assert isinstance(inv, CompressedLinearInverseQuantizer)
    repr_str = repr(inv)
    assert "quantization_scheme=" in repr_str

    expected_shape = (module.out_features, module.in_features)
    _assert_valid_dequant(inv.dequantize(_get_weight(module)), expected_shape)

    linear = dequantize_prequantized_to_linear(module)
    assert isinstance(linear, nn.Linear)
    _assert_valid_dequant(linear.weight, expected_shape)


def test_int4_dequantize_and_to_linear(compressed_int4_model: nn.Module) -> None:
    """Verify dequantize and dequantize_to_linear for compressed-tensors INT4."""
    module = _find_first_prequantized_module(compressed_int4_model)
    assert module is not None

    inv = create_inverse_quantizer(module)
    assert isinstance(inv, CompressedLinearInverseQuantizer)

    expected_shape = (module.out_features, module.in_features)
    _assert_valid_dequant(inv.dequantize(_get_weight(module)), expected_shape)

    linear = dequantize_prequantized_to_linear(module)
    assert isinstance(linear, nn.Linear)
    _assert_valid_dequant(linear.weight, expected_shape)


@require_torch_cuda
def test_dequantize_and_to_linear(fp8_linear_model: nn.Module) -> None:
    """Verify dequantize, dequantize_to_linear, and is_prequantized for FP8Linear."""
    module = _find_first_prequantized_module(fp8_linear_model)
    assert module is not None
    assert is_prequantized_linear(module) is True

    inv = create_inverse_quantizer(module)
    assert isinstance(inv, FP8LinearInverseQuantizer)
    _assert_valid_dequant(inv.dequantize(module.weight), module.weight.shape)

    linear = dequantize_prequantized_to_linear(module)
    assert isinstance(linear, nn.Linear)
    _assert_valid_dequant(linear.weight, (module.out_features, module.in_features))


def test_to_linear_uses_bias_dtype() -> None:
    """Tests for dtype alignment when converting to nn.Linear."""

    class _FakeQuantizationStatus:
        value = "compressed"

    class FakeCompressedModule(nn.Linear):
        def __init__(self) -> None:
            super().__init__(8, 4, bias=True)
            self.weight = nn.Parameter(torch.ones((4, 8), dtype=torch.float32), requires_grad=False)
            self.bias = nn.Parameter(torch.ones(4, dtype=torch.bfloat16), requires_grad=False)
            self.quantization_status = _FakeQuantizationStatus()

    class FakeInverseQuantizer(InverseWeightQuantizer):
        def dequantize(self, quantized_weight: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(quantized_weight, dtype=torch.float32)

    module = FakeCompressedModule()
    with patch(
        "quark.torch.quantization.inverse_quantizer.create_inverse_quantizer",
        return_value=FakeInverseQuantizer(),
    ):
        linear = dequantize_prequantized_to_linear(module)

    assert linear.weight.dtype == torch.bfloat16
    assert linear.bias is not None
    assert linear.bias.dtype == torch.bfloat16


def test_regular_module_rejected() -> None:
    """Regular nn.Linear: not prequantized, create/dequantize raise ValueError."""
    linear = nn.Linear(64, 64)
    assert is_prequantized_linear(linear) is False
    with pytest.raises(ValueError, match="Unsupported module type"):
        create_inverse_quantizer(linear)
    with pytest.raises(ValueError, match="is not a pre-quantized linear"):
        dequantize_prequantized_to_linear(linear)


def test_missing_scale_attributes() -> None:
    """Constructors raise when required scale attributes are missing."""

    class FakeNoScale(nn.Module):  # type: ignore[misc]
        def __init__(self) -> None:
            super().__init__()
            self.in_features = self.out_features = 64

            self.weight = nn.Parameter(torch.randn(64, 64))

    with pytest.raises(ValueError, match="weight_scale"):
        CompressedLinearInverseQuantizer(FakeNoScale())
    with pytest.raises(ValueError, match="weight_scale_inv"):
        FP8LinearInverseQuantizer(FakeNoScale())


def test_base_class_dequantize_not_implemented() -> None:
    """InverseWeightQuantizer.dequantize raises NotImplementedError."""
    with pytest.raises(NotImplementedError):
        InverseWeightQuantizer().dequantize(torch.randn(4, 4))


def _snapshot_download_or_skip(model_id: str) -> str:
    """Return the local snapshot directory for ``model_id``.

    Skips the test when the model is not available offline -- i.e. running with
    ``HF_HUB_OFFLINE=1`` (as the CI does) and the model is not in the local HF
    cache. Pre-populating the shared cache re-enables the test automatically.
    """
    try:
        return huggingface_hub.snapshot_download(model_id)
    except huggingface_hub.errors.LocalEntryNotFoundError:
        pytest.skip(
            f"Model {model_id} is not in the local HF cache and outgoing traffic is disabled "
            "(HF_HUB_OFFLINE). Pre-cache the model to run this test."
        )


def _dir_size_bytes(path: str) -> int:
    """Total size, in bytes, of all files under ``path`` (recursively)."""
    return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())


# TODO: verify whether Kimi-K2.5 / Kimi-K2.6 custom modeling code is compatible with Transformers v5.
@pytest.mark.skipif(
    is_transformers_version_higher_or_equal("5.0"),
    reason="requires transformers < 5.0",
)
@pytest.mark.parametrize(
    "model_id",
    [
        "amd-quark/Kimi-K2.5-2-layers-tiny",
        pytest.param(
            "moonshotai/Kimi-K2.6",
            marks=pytest.mark.skipif(not TEST_WITH_EXTENSIVE, reason="set QUARK_EXTENSIVE_TEST=1 to enable"),
        ),
    ],
)
def test_kimi_k25_quantize_export(model_id: str) -> None:
    """Full quantize_model + freeze + export_safetensors flow on Kimi models loaded via get_model."""
    model_dir = _snapshot_download_or_skip(model_id)

    required_disk_space = 2 * _dir_size_bytes(model_dir)
    available_disk_space = shutil.disk_usage(tempfile.gettempdir()).free

    if available_disk_space < required_disk_space:
        pytest.fail(
            f"Not enough disk space to save two quantized models for {model_id}. "
            f"Required: {required_disk_space / (1024**3):.1f} GB, "
            f"available: {available_disk_space / (1024**3):.1f} GB."
        )

    model, _ = get_model(model_dir, multi_gpu=True)

    # MXFP4 weight-only quantization config
    mxfp4_spec = OCP_MXFP4Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()
    quant_config = QConfig(
        global_quant_config=QLayerConfig(weight=mxfp4_spec),
        exclude=[
            "*lm_head*",
            "*vision_tower*",
            "*.mlp.gate",
            "*mm_projector*",
            "*shared_experts*",
            "*self_attn*",
            "*mlp.gate_proj*",
            "*mlp.up_proj*",
            "*mlp.gate_up_proj*",
            "*mlp.down_proj*",
        ],
    )

    original_get_normal_quant_weight = QuantLinear._get_normal_quant_weight
    get_normal_quant_weight_call_count = 0

    def tracking_get_normal_quant_weight(self: QuantLinear, *args: object, **kwargs: object) -> torch.Tensor:
        nonlocal get_normal_quant_weight_call_count
        get_normal_quant_weight_call_count += 1
        return original_get_normal_quant_weight(self, *args, **kwargs)

    with patch.object(QuantLinear, "_get_normal_quant_weight", tracking_get_normal_quant_weight):
        quantizer = ModelQuantizer(copy.deepcopy(quant_config))
        quant_model = quantizer.quantize_model(model)

        quant_model = quantizer.freeze(quant_model)

        # Export via standard flow.
        with tempfile.TemporaryDirectory() as in_memory_dir, tempfile.TemporaryDirectory() as file_to_file_dir:
            with torch.no_grad():
                export_safetensors(
                    model=quant_model,
                    output_dir=in_memory_dir,
                    weight_format="real_quantized",
                    pack_method="reorder",
                )

            del quant_model
            gc.collect()
            torch.cuda.empty_cache()

            # Export via file-to-file flow
            quantizer = ModelQuantizer(copy.deepcopy(quant_config))
            quantizer.direct_quantize_checkpoint(
                pretrained_model_path=model_dir,
                save_path=file_to_file_dir,
            )

            # Load and compare weights from both flows
            in_memory_weights: dict[str, torch.Tensor] = {}
            for safetensors_path in sorted(Path(in_memory_dir).glob("*.safetensors")):
                with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
                    for key in f.keys():  # noqa
                        in_memory_weights[key] = f.get_tensor(key)

            file_to_file_weights: dict[str, torch.Tensor] = {}
            for safetensors_path in sorted(Path(file_to_file_dir).glob("*.safetensors")):
                with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
                    for key in f.keys():  # noqa
                        file_to_file_weights[key] = f.get_tensor(key)

            assert "vision_tower.encoder.blocks.3.norm0.weight_scale" not in in_memory_weights
            assert "vision_tower.encoder.blocks.3.norm0.weight_scale" not in file_to_file_weights

            assert set(in_memory_weights.keys()) == set(file_to_file_weights.keys())

            # Every tensor must match exactly
            for key in sorted(in_memory_weights.keys()):
                assert in_memory_weights[key].dtype == file_to_file_weights[key].dtype
                assert in_memory_weights[key].shape == file_to_file_weights[key].shape
                assert torch.equal(in_memory_weights[key], file_to_file_weights[key])

    assert get_normal_quant_weight_call_count == 0, (
        f"QuantLinear._get_normal_quant_weight was called {get_normal_quant_weight_call_count} time(s), expected 0"
    )


# TODO: verify whether Kimi-K2.5 / Kimi-K2.6 custom modeling code is compatible with Transformers v5.
@require_torch_cuda
@pytest.mark.skipif(
    is_transformers_version_higher_or_equal("5.0"),
    reason="requires transformers < 5.0",
)
def test_kimi_k25_nvfp4_quantization_and_export() -> None:
    """NVFP4 quantize_model + freeze + export_safetensors flow on Kimi-K2.5 loaded via get_model."""
    model_id = "amd-quark/Kimi-K2.5-2-layers-tiny"

    model_dir = _snapshot_download_or_skip(model_id)
    model, _ = get_model(model_dir, device=torch_device)

    template = LLMTemplate.get(model.config.model_type)
    quant_config = template.get_config("nvfp4")

    text = "Hello, how are you?"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenized_outputs = tokenizer(text, return_tensors="pt")
    calib_dataloader = DataLoader(tokenized_outputs["input_ids"].to(torch_device))

    quantizer = ModelQuantizer(copy.deepcopy(quant_config))
    quant_model = quantizer.quantize_model(model, calib_dataloader)

    quant_model = quantizer.freeze(quant_model)

    with tempfile.TemporaryDirectory() as export_dir:
        with torch.no_grad():
            export_safetensors(
                model=quant_model,
                output_dir=export_dir,
                weight_format="real_quantized",
                pack_method="reorder",
            )

        exported_weights: dict[str, torch.Tensor] = {}
        for safetensors_path in sorted(Path(export_dir).glob("*.safetensors")):
            with safe_open(str(safetensors_path), framework="pt", device="cpu") as f:
                for key in f.keys():  # noqa
                    exported_weights[key] = f.get_tensor(key)

        assert "vision_tower.encoder.blocks.3.norm0.weight_scale" not in exported_weights

        # Check that weight_scale_2 is the same across projections within each (layer, expert),
        # and that input_scale_2 is the same across experts within each (layer, proj).
        n_layers = model.config.text_config.num_hidden_layers
        n_experts = model.config.text_config.n_routed_experts
        prefix = "language_model.model.layers"

        for layer_idx in range(n_layers):
            if layer_idx == 0:
                base = f"{prefix}.{layer_idx}.mlp"

                assert f"{base}.gate_proj.weight_scale_2" not in exported_weights
                assert f"{base}.up_proj.weight_scale_2" not in exported_weights
                assert f"{base}.down.weight_scale_2" not in exported_weights
            else:
                for expert_idx in range(n_experts):
                    base = f"{prefix}.{layer_idx}.mlp.experts.{expert_idx}"

                    # weight_scale_2 must be equal for gate_proj and up_proj (fused in vLLM).
                    assert torch.equal(
                        exported_weights[f"{base}.gate_proj.weight_scale_2"],
                        exported_weights[f"{base}.up_proj.weight_scale_2"],
                    )

                # input_scale_2 must be equal across all parallel layers.
                for expert_idx in range(1, n_experts):
                    assert torch.equal(
                        exported_weights[f"{prefix}.{layer_idx}.mlp.experts.0.gate_proj.input_scale_2"],
                        exported_weights[f"{prefix}.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.input_scale_2"],
                    )
                    assert torch.equal(
                        exported_weights[f"{prefix}.{layer_idx}.mlp.experts.0.up_proj.input_scale_2"],
                        exported_weights[f"{prefix}.{layer_idx}.mlp.experts.{expert_idx}.up_proj.input_scale_2"],
                    )
                    assert torch.equal(
                        exported_weights[f"{prefix}.{layer_idx}.mlp.experts.0.down_proj.input_scale_2"],
                        exported_weights[f"{prefix}.{layer_idx}.mlp.experts.{expert_idx}.down_proj.input_scale_2"],
                    )


def test_decompress_weight_calls_compressor_decompress() -> None:
    """Verify _decompress_weight delegates to the compressor's ``decompress`` and returns its weight."""
    fake_weight = torch.randn(4, 8)
    fake_scale = torch.ones(4, 1)
    compressed_data = {"weight": fake_weight, "weight_scale": fake_scale}

    class FakeQuantizationArgs:
        pass

    class FakeScheme:
        weights = FakeQuantizationArgs()

    fake_scheme = FakeScheme()
    sentinel = torch.randn(4, 8)
    with patch("quark.torch.quantization.inverse_quantizer.BaseCompressor.load_from_registry") as mock_load:
        mock_compressor = mock_load.return_value
        mock_compressor.decompress.return_value = {"weight": sentinel}

        result = CompressedLinearInverseQuantizer._decompress_weight(
            compression_format="float-quantized",
            compressed_data=compressed_data,
            quantization_scheme=fake_scheme,
        )

    mock_compressor.decompress.assert_called_once_with(state_dict=compressed_data, scheme=fake_scheme)
    assert torch.equal(result, sentinel)


def test_patch_missing_weights_does_not_restore_compressed_tensors_artifacts(tmp_path):
    """patch_missing_weights only copies back keys matching the model's ignore patterns. The
    compressed-tensors intermediate tensors (weight_scale_inv / weight_packed / weight_shape /
    weight_zero_point) do not match those patterns and must NOT be restored into the export — only
    genuine mtp.* weights should be. Regression guard for the earlier bug where weight_scale_inv
    was wrongly copied back and broke import.
    """
    # Source checkpoint mixes compressed-tensors artifacts with a real mtp weight.
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    source_sd = {
        "model.layers.0.self_attn.q_proj.weight_scale_inv": torch.ones(2, 2),
        "model.layers.0.self_attn.q_proj.weight_packed": torch.ones(2, 2),
        "model.layers.0.self_attn.q_proj.weight_shape": torch.ones(2, 2),
        "model.layers.0.mlp.down_proj.weight_zero_point": torch.ones(2, 2),
        "mtp.fc.weight": torch.ones(2, 2),
    }
    save_file(source_sd, str(source_dir / "model.safetensors"), metadata={"format": "pt"})

    export_dir = tmp_path / "export"
    export_dir.mkdir()
    save_file(
        {"model.embed.weight": torch.zeros(2, 2)}, str(export_dir / "model.safetensors"), metadata={"format": "pt"}
    )

    # None -> patch_missing_weights falls back to the default mtp.* pattern; artifacts never match it.
    model = types.SimpleNamespace()
    model.config = types.SimpleNamespace(_name_or_path=str(source_dir))
    model._keys_to_ignore_on_load_unexpected = None

    patch_missing_weights(export_dir, model)

    exported = _load_weights_from_safetensors(str(export_dir))
    # Only the genuine mtp weight is restored; compressed-tensors artifacts are not.
    assert "mtp.fc.weight" in exported
    for artifact in (
        "model.layers.0.self_attn.q_proj.weight_scale_inv",
        "model.layers.0.self_attn.q_proj.weight_packed",
        "model.layers.0.self_attn.q_proj.weight_shape",
        "model.layers.0.mlp.down_proj.weight_zero_point",
    ):
        assert artifact not in exported, f"{artifact} must not be restored into the export"
