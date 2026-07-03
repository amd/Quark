#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for per_block_runner — no CUDA or real model weights required."""

import json

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from quark.torch.utils.per_block_runner import finalize, get_module_weight_by_name, prepare
from quark.torch.utils.per_block_runner.runner import (
    _get_parent_and_attr_name,
    _make_block_hooks,
    _realign_rotary_emb,
    _recompute_rotary_caches_in_block,
    _swap_in_state_dict,
    _swap_out_to_meta,
)
from quark.torch.utils.per_block_runner.utils import infer_decoder_layers_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_index(tmp_path, weight_map: dict) -> None:
    index = {"weight_map": weight_map}
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)


# ---------------------------------------------------------------------------
# get_module_weight_by_name
# ---------------------------------------------------------------------------


def test_get_weight_empty_name(tmp_path):
    assert get_module_weight_by_name("", weight_map={}, model_dir=str(tmp_path)) is None


def test_get_weight_no_matching_keys(tmp_path):
    weight_map = {"other.weight": "shard.safetensors"}
    assert get_module_weight_by_name("my.module", weight_map=weight_map, model_dir=str(tmp_path)) is None


def test_get_weight_none_shard_file(tmp_path):
    weight_map = {"my.module.weight": None}
    result = get_module_weight_by_name("my.module", weight_map=weight_map, model_dir=str(tmp_path))
    assert result == {}


def test_get_weight_with_strip_prefix(tmp_path):
    w = torch.randn(4, 4)
    save_file({"my.module.weight": w}, str(tmp_path / "shard.safetensors"))
    weight_map = {"my.module.weight": "shard.safetensors"}
    result = get_module_weight_by_name(
        "my.module", weight_map=weight_map, model_dir=str(tmp_path), strip_prefix="my.module"
    )
    assert result is not None and "weight" in result
    assert result["weight"].shape == w.shape


def test_get_weight_no_strip_prefix(tmp_path):
    w = torch.randn(4, 4)
    save_file({"my.module.weight": w}, str(tmp_path / "shard.safetensors"))
    weight_map = {"my.module.weight": "shard.safetensors"}
    result = get_module_weight_by_name("my.module", weight_map=weight_map, model_dir=str(tmp_path))
    assert result is not None and "my.module.weight" in result


def test_get_weight_missing_shard_raises(tmp_path):
    weight_map = {"my.module.weight": "nonexistent.safetensors"}
    with pytest.raises(FileNotFoundError):
        get_module_weight_by_name("my.module", weight_map=weight_map, model_dir=str(tmp_path))


# ---------------------------------------------------------------------------
# _get_parent_and_attr_name
# ---------------------------------------------------------------------------


def test_get_parent_single_part():
    model = nn.Linear(4, 4)
    parent, attr = _get_parent_and_attr_name(model, "weight")
    assert parent is model and attr == "weight"


def test_get_parent_nested():
    class Outer(nn.Module):
        def __init__(self):
            super().__init__()
            self.inner = nn.Linear(4, 4)

    model = Outer()
    parent, attr = _get_parent_and_attr_name(model, "inner.weight")
    assert parent is model.inner and attr == "weight"


# ---------------------------------------------------------------------------
# _swap_in_state_dict / _swap_out_to_meta
# ---------------------------------------------------------------------------


def test_swap_in_parameter():
    model = nn.Linear(4, 4, bias=False)
    new_weight = torch.ones(4, 4)
    loaded = _swap_in_state_dict(model, {"weight": new_weight})
    assert "weight" in loaded
    assert torch.allclose(model.weight.data, new_weight)


def test_swap_in_buffer():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.zeros(8))

    model = M()
    loaded = _swap_in_state_dict(model, {"inv_freq": torch.ones(8)})
    assert "inv_freq" in loaded
    assert torch.allclose(model.inv_freq, torch.ones(8))


def test_swap_in_skips_rope_caches():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cos_cached", torch.zeros(8))
            self.register_buffer("sin_cached", torch.zeros(8))

    model = M()
    loaded = _swap_in_state_dict(model, {"cos_cached": torch.ones(8), "sin_cached": torch.ones(8)})
    assert "cos_cached" not in loaded and "sin_cached" not in loaded
    assert torch.all(model.cos_cached == 0)


def test_swap_out_parameter():
    model = nn.Linear(4, 4, bias=False)
    model.weight = nn.Parameter(torch.ones(4, 4))
    _swap_out_to_meta(model, ["weight"])
    assert model.weight.device.type == "meta"


def test_swap_out_buffer():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.ones(8))

    model = M()
    _swap_out_to_meta(model, ["inv_freq"])
    assert model.inv_freq.device.type == "meta"


# ---------------------------------------------------------------------------
# _make_block_hooks
# ---------------------------------------------------------------------------


def test_make_block_hooks_no_state_dict(tmp_path):
    pre_hook, post_hook = _make_block_hooks(
        "layers.0", weight_map={}, pretrained_model_name_or_path=str(tmp_path), target_device=torch.device("cpu")
    )
    model = nn.Linear(4, 4)
    pre_hook(model, ())
    assert model._pbr_loaded_keys == []


def test_make_block_hooks_load_and_offload(tmp_path):
    w = torch.randn(4, 4)
    save_file({"layers.0.weight": w}, str(tmp_path / "shard.safetensors"))
    weight_map = {"layers.0.weight": "shard.safetensors"}

    pre_hook, post_hook = _make_block_hooks(
        "layers.0",
        weight_map=weight_map,
        pretrained_model_name_or_path=str(tmp_path),
        target_device=torch.device("cpu"),
    )
    model = nn.Linear(4, 4, bias=False)
    pre_hook(model, ())
    assert len(model._pbr_loaded_keys) >= 1
    post_hook(model, (), None)
    assert model.weight.device.type == "meta"


def test_make_block_hooks_dtype_cast(tmp_path):
    w = torch.randn(4, 4).to(torch.float32)
    save_file({"layers.0.weight": w}, str(tmp_path / "shard.safetensors"))
    weight_map = {"layers.0.weight": "shard.safetensors"}

    pre_hook, _ = _make_block_hooks(
        "layers.0",
        weight_map=weight_map,
        pretrained_model_name_or_path=str(tmp_path),
        target_device=torch.device("cpu"),
        model_dtype=torch.float16,
    )
    model = nn.Linear(4, 4, bias=False)
    pre_hook(model, ())
    assert model.weight.dtype == torch.float16


# ---------------------------------------------------------------------------
# prepare edge cases
# ---------------------------------------------------------------------------


def test_prepare_no_index_file(tmp_path):
    model = nn.Linear(4, 4)
    prepare(model, str(tmp_path), target_device="cpu")
    assert not hasattr(model, "_pbr_hook_handles")


def test_prepare_non_modulelist_layers(tmp_path):
    _write_index(tmp_path, {})

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Linear(4, 4)

        def forward(self, x):
            return self.layers(x)

    model = FakeModel()
    prepare(model, str(tmp_path), target_device="cpu")
    assert not hasattr(model, "_pbr_hook_handles")


# ---------------------------------------------------------------------------
# finalize edge cases
# ---------------------------------------------------------------------------


def test_finalize_no_op_when_not_prepared():
    finalize(nn.Linear(4, 4))


def test_finalize_removes_hooks_and_attrs():
    model = nn.Linear(4, 4)
    handle = model.register_forward_hook(lambda m, i, o: None)
    model._pbr_hook_handles = [handle]
    model._pbr_prev_deterministic = False
    finalize(model)
    assert not hasattr(model, "_pbr_hook_handles")
    assert not hasattr(model, "_pbr_weight_map")


def test_finalize_reloads_block_weights(tmp_path):
    w = torch.randn(4, 4)
    save_file({"layers.0.weight": w}, str(tmp_path / "shard.safetensors"))
    weight_map = {"layers.0.weight": "shard.safetensors"}
    _write_index(tmp_path, weight_map)

    block = nn.Linear(4, 4, bias=False)
    block.weight = nn.Parameter(torch.empty_like(block.weight, device="meta"))

    class WrappedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([block])

    model = WrappedModel()
    handle = model.register_forward_hook(lambda m, i, o: None)
    model._pbr_hook_handles = [handle]
    model._pbr_weight_map = weight_map
    model._pbr_model_dir = str(tmp_path)
    model._pbr_target_device = torch.device("cpu")
    model._pbr_model_dtype = None
    model._pbr_decoder_layers_name = "layers"
    model._pbr_prev_deterministic = False

    finalize(model)

    assert not hasattr(model, "_pbr_hook_handles")
    assert block.weight.device.type == "cpu"
    assert torch.allclose(block.weight.data, w)


def test_finalize_with_model_dtype(tmp_path):
    w = torch.randn(4, 4)
    save_file({"layers.0.weight": w}, str(tmp_path / "shard.safetensors"))
    weight_map = {"layers.0.weight": "shard.safetensors"}

    block = nn.Linear(4, 4, bias=False)

    class WrappedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([block])

    model = WrappedModel()
    handle = model.register_forward_hook(lambda m, i, o: None)
    model._pbr_hook_handles = [handle]
    model._pbr_weight_map = weight_map
    model._pbr_model_dir = str(tmp_path)
    model._pbr_target_device = torch.device("cpu")
    model._pbr_model_dtype = torch.float16
    model._pbr_decoder_layers_name = "layers"
    model._pbr_prev_deterministic = False

    finalize(model)
    assert block.weight.dtype == torch.float16


# ---------------------------------------------------------------------------
# infer_decoder_layers_path
# ---------------------------------------------------------------------------


def test_infer_layers_path_simple():
    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4)])

    assert infer_decoder_layers_path(M()) == "layers"


def test_infer_layers_path_nested():
    class Inner(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = nn.ModuleList([nn.Linear(4, 4)])

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleDict({"layers": nn.ModuleList([Inner()])})

    # The root-level ModuleList is model.layers (the one containing Inner blocks)
    path = infer_decoder_layers_path(M())
    # model.model.layers contains Inner; model.model.layers.0.experts is nested
    assert not path.endswith("experts")


def test_infer_layers_path_no_modulelist():
    model = nn.Linear(4, 4)
    assert infer_decoder_layers_path(model) == ""


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------


def test_recompute_rotary_caches_called():
    class FakeRotary(nn.Module):
        def __init__(self):
            super().__init__()
            self.dim = 8
            self.base = 10000.0
            self.register_buffer("inv_freq", torch.ones(4))
            self.register_buffer("cos_cached", torch.zeros(16, 4))
            self._calls = []

        def _set_cos_sin_cache(self, seq_len, device, dtype):
            self._calls.append((seq_len, dtype))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary = FakeRotary()

    block = Block()
    _recompute_rotary_caches_in_block(block, model_dtype=torch.bfloat16)
    assert len(block.rotary._calls) == 1
    assert block.rotary._calls[0][1] == torch.bfloat16


def test_recompute_rotary_fallback_seq_len():
    class FakeRotary(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.ones(4))
            self._calls = []

        def _set_cos_sin_cache(self, seq_len, device, dtype):
            self._calls.append(seq_len)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.rotary = FakeRotary()

    block = Block()
    _recompute_rotary_caches_in_block(block, model_dtype=None)
    assert block.rotary._calls[0] == 4096


def test_realign_rotary_emb_skips_no_set_cos_sin_cache():
    class NoCache(nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("inv_freq", torch.ones(4))

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.r = NoCache()

    _realign_rotary_emb(nn.ModuleList([Block()]), target_device=torch.device("cpu"))


def test_realign_rotary_emb_skips_none_inv_freq():
    class NoneFreq(nn.Module):
        def __init__(self):
            super().__init__()
            self._buffers["inv_freq"] = None
            self._called = False

        def _set_cos_sin_cache(self, **_):
            self._called = True

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.r = NoneFreq()

    container = nn.ModuleList([Block()])
    _realign_rotary_emb(container, target_device=torch.device("cpu"))
    assert not container[0].r._called


# ---------------------------------------------------------------------------
# End-to-end smoke test (CPU only, tiny model, random weights)
# ---------------------------------------------------------------------------


class TinyBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class TinyModel(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 8, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([TinyBlock(dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def _build_tiny_checkpoint(tmp_path, model: TinyModel) -> None:
    """Save model weights as a sharded safetensors checkpoint with index."""
    sd = model.state_dict()
    # Shard 1: embed + layers.0
    shard1 = {k: v for k, v in sd.items() if k.startswith("embed") or k.startswith("layers.0")}
    # Shard 2: layers.1 + norm
    shard2 = {k: v for k, v in sd.items() if k.startswith("layers.1") or k.startswith("norm")}

    save_file(shard1, str(tmp_path / "shard-00001-of-00002.safetensors"))
    save_file(shard2, str(tmp_path / "shard-00002-of-00002.safetensors"))

    weight_map = {}
    for k in shard1:
        weight_map[k] = "shard-00001-of-00002.safetensors"
    for k in shard2:
        weight_map[k] = "shard-00002-of-00002.safetensors"

    index = {"weight_map": weight_map}
    with open(tmp_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f)


def test_end_to_end_smoke(tmp_path):
    """Verify that per-block prepare/forward/finalize produces the same output as a normal model."""
    torch.manual_seed(0)
    reference_model = TinyModel()
    reference_model.eval()

    _build_tiny_checkpoint(tmp_path, reference_model)

    # Build an empty model (simulate meta-device init by zeroing weights)
    torch.manual_seed(99)
    pbr_model = TinyModel()
    # Zero out all parameters to ensure we are actually loading from checkpoint
    for p in pbr_model.parameters():
        p.data.zero_()
    pbr_model.eval()

    prepare(pbr_model, str(tmp_path), target_device="cpu")
    assert hasattr(pbr_model, "_pbr_hook_handles")

    ids = torch.randint(0, 32, (1, 5))
    with torch.no_grad():
        out_pbr = pbr_model(ids)
        out_ref = reference_model(ids)

    assert torch.allclose(out_pbr, out_ref, atol=1e-5), f"max diff: {(out_pbr - out_ref).abs().max()}"

    finalize(pbr_model)
    assert not hasattr(pbr_model, "_pbr_hook_handles")

    # After finalize, the model should still produce correct output
    with torch.no_grad():
        out_final = pbr_model(ids)
    assert torch.allclose(out_final, out_ref, atol=1e-5)
