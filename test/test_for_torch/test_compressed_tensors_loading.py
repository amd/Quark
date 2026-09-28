#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from quark.torch.integrations.compressed_tensors import loading as compressed_tensors_loading


def _patch_structural_loader(monkeypatch):
    calls = []

    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(2, 2, device="meta")

    model = DummyModel()

    class DummyAutoModelForCausalLM:
        @staticmethod
        def from_config(config, **kwargs):
            calls.append("from_config")
            return model

    compressor = SimpleNamespace(quantization_config=object())

    def compress_model(compressed_model):
        assert compressed_model is model
        calls.append("compress_model")

    compressor.compress_model = compress_model

    class DummyModelCompressor:
        @staticmethod
        def from_compression_config(compression_config):
            calls.append("from_compression_config")
            return compressor

    class DummyCompressedTensorsConfig:
        @staticmethod
        def from_dict(quantization_config):
            return object()

    def apply_quantization_config(compressed_model, quantization_config, *, run_compressed):
        assert compressed_model is model
        assert quantization_config is compressor.quantization_config
        assert run_compressed is True
        calls.append("apply_quantization_config")

    monkeypatch.setattr(compressed_tensors_loading, "is_compressed_tensors_available", lambda: True)
    monkeypatch.setattr(compressed_tensors_loading, "is_accelerate_available", lambda: True)
    monkeypatch.setattr(compressed_tensors_loading, "is_package_lower_or_equal", lambda *args: False)
    monkeypatch.setattr(compressed_tensors_loading, "no_init_weights", nullcontext)
    monkeypatch.setattr(compressed_tensors_loading, "init_empty_weights", nullcontext)
    monkeypatch.setattr(compressed_tensors_loading, "AutoModelForCausalLM", DummyAutoModelForCausalLM)
    monkeypatch.setattr(compressed_tensors_loading, "CompressedTensorsConfig", DummyCompressedTensorsConfig)
    monkeypatch.setattr(compressed_tensors_loading, "ModelCompressor", DummyModelCompressor)
    monkeypatch.setattr(compressed_tensors_loading, "apply_quantization_config", apply_quantization_config)

    config = SimpleNamespace(model_type="dummy", quantization_config={})
    return model, config, calls


@pytest.mark.parametrize("device_map", ["meta", torch.device("meta")])
def test_load_from_compressed_tensors_meta_returns_structure_without_checkpoint_io(monkeypatch, tmp_path, device_map):
    """Meta loading applies compressed structure but never inspects or opens safetensors."""
    model, config, calls = _patch_structural_loader(monkeypatch)
    (tmp_path / "model.safetensors.index.json").write_text("not valid JSON")

    def fail_safe_open(*args, **kwargs):
        pytest.fail("A meta-only structural load must not open safetensors")

    monkeypatch.setattr(compressed_tensors_loading, "safe_open", fail_safe_open)

    loaded = compressed_tensors_loading._load_from_compressed_tensors(
        model_dir=tmp_path,
        config=config,
        device_map=device_map,
        max_memory=None,
        trust_remote_code=True,
    )

    assert loaded is model
    assert calls == ["from_config", "from_compression_config", "apply_quantization_config", "compress_model"]
    assert all(parameter.device.type == "meta" for parameter in loaded.parameters())


def test_load_from_compressed_tensors_non_meta_still_loads_checkpoint(monkeypatch, tmp_path):
    """The existing non-meta path must continue materializing checkpoint tensors."""
    model, config, calls = _patch_structural_loader(monkeypatch)
    expected_weight = torch.arange(4, dtype=torch.float32).reshape(2, 2)
    expected_bias = torch.tensor([4.0, 5.0])
    save_file(
        {"proj.weight": expected_weight, "proj.bias": expected_bias},
        tmp_path / "model.safetensors",
    )

    loaded = compressed_tensors_loading._load_from_compressed_tensors(
        model_dir=tmp_path,
        config=config,
        device_map="cpu",
        max_memory=None,
        trust_remote_code=True,
    )

    assert loaded is model
    assert calls == ["from_config", "from_compression_config", "apply_quantization_config", "compress_model"]
    assert loaded.proj.weight.device.type == "cpu"
    assert torch.equal(loaded.proj.weight, expected_weight)
    assert torch.equal(loaded.proj.bias, expected_bias)
