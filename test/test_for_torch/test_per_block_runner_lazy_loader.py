#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for per_block_runner.lazy_loader."""

import json

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from quark.torch.utils.per_block_runner.lazy_loader import (
    _offload_block_params_to_meta,
    _swap_in_params,
    _swap_out_params_to_meta,
    finalize,
    prepare,
)
from quark.torch.utils.per_block_runner.utils import infer_decoder_layers_path

# ---------------------------------------------------------------------------
# Tiny model + fake checkpoint fixture
# ---------------------------------------------------------------------------


class TinyBlock(nn.Module):
    def __init__(self, dim: int = 8):
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


def _save_fake_checkpoint(model: TinyModel, tmp_path) -> str:
    """Save model weights as a sharded safetensors checkpoint and return the directory."""
    # Clone: safetensors refuses to serialize tensors sharing storage, e.g. tied weights.
    tensors = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}

    # Split into two shards (shard-1: embed+norm, shard-2: layers)
    shard1 = {k: v for k, v in tensors.items() if not k.startswith("layers")}
    shard2 = {k: v for k, v in tensors.items() if k.startswith("layers")}

    save_file(shard1, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file(shard2, str(tmp_path / "model-00002-of-00002.safetensors"))

    weight_map = {}
    for k in shard1:
        weight_map[k] = "model-00001-of-00002.safetensors"
    for k in shard2:
        weight_map[k] = "model-00002-of-00002.safetensors"

    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
    return str(tmp_path)


# ---------------------------------------------------------------------------
# _offload_block_params_to_meta
# ---------------------------------------------------------------------------


def test_offload_puts_params_on_meta():
    block = TinyBlock()
    assert block.fc.weight.device.type == "cpu"

    _offload_block_params_to_meta(block)
    assert block.fc.weight.is_meta


def test_offload_preserves_shape_and_dtype():
    block = TinyBlock(dim=16)
    orig_shape = block.fc.weight.shape
    orig_dtype = block.fc.weight.dtype

    _offload_block_params_to_meta(block)

    assert block.fc.weight.shape == orig_shape
    assert block.fc.weight.dtype == orig_dtype


# ---------------------------------------------------------------------------
# _swap_in_params / _swap_out_params_to_meta
# ---------------------------------------------------------------------------


def test_swap_in_loads_into_params():
    block = TinyBlock()
    _offload_block_params_to_meta(block)
    assert block.fc.weight.is_meta

    real = torch.ones(8, 8)
    loaded = _swap_in_params(block, {"fc.weight": real})

    assert "fc.weight" in loaded
    assert not block.fc.weight.is_meta
    assert torch.all(block.fc.weight == 1.0)


def test_swap_in_skips_buffers():
    """_swap_in_params must not touch buffers — only _parameters."""
    block = TinyBlock()
    block.register_buffer("my_buf", torch.zeros(4))

    loaded = _swap_in_params(block, {"my_buf": torch.ones(4)})

    assert "my_buf" not in loaded
    assert torch.all(block.my_buf == 0.0)  # unchanged


def test_swap_out_returns_to_meta():
    block = TinyBlock()
    real = torch.ones(8, 8)
    _swap_in_params(block, {"fc.weight": real})
    assert not block.fc.weight.is_meta

    _swap_out_params_to_meta(block, ["fc.weight"])
    assert block.fc.weight.is_meta


# ---------------------------------------------------------------------------
# prepare edge cases
# ---------------------------------------------------------------------------


def test_prepare_without_index_uses_cpu_ram(tmp_path):
    """RAM offloading reads weights from the model, so a missing index is fine."""
    model = TinyModel()
    prepare(model, str(tmp_path), target_device="cpu")

    assert model._pbr_lazy_state.use_cpu_ram
    assert model._pbr_lazy_state.weight_map == {}
    for block in model.layers:
        assert all(p.is_meta for p in block.parameters())

    finalize(model)


def test_prepare_without_index_raises_in_disk_mode(tmp_path, monkeypatch):
    """Disk streaming needs the index, so say so instead of silently doing nothing."""
    from quark.torch.utils.per_block_runner import lazy_loader

    monkeypatch.setattr(lazy_loader.PerBlockLazyLoader, "_choose_backend", lambda *args, **kwargs: False)

    model = TinyModel()
    with pytest.raises(NotImplementedError, match="model.safetensors.index.json"):
        prepare(model, str(tmp_path), target_device="cpu")


def test_prepare_raises_when_checkpoint_keys_do_not_match_module_paths(tmp_path, monkeypatch):
    """A checkpoint whose keys use another prefix would load zero tensors per block."""
    from quark.torch.utils.per_block_runner import lazy_loader

    monkeypatch.setattr(lazy_loader.PerBlockLazyLoader, "_choose_backend", lambda *args, **kwargs: False)

    model = TinyModel()
    _save_fake_checkpoint(model, tmp_path)
    index_path = tmp_path / "model.safetensors.index.json"
    weight_map = json.loads(index_path.read_text())["weight_map"]
    renamed = {key.replace("layers.", "inner.layers.", 1): shard for key, shard in weight_map.items()}
    index_path.write_text(json.dumps({"weight_map": renamed}))

    with pytest.raises(NotImplementedError, match="No key in"):
        prepare(model, str(tmp_path), target_device="cpu")


def test_prepare_no_modulelist(tmp_path):
    """prepare() on a model with no ModuleList should be a no-op."""
    # Create a minimal index so the file-missing guard passes
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {}}))
    model = nn.Linear(4, 4)
    prepare(model, str(tmp_path))
    assert not hasattr(model, "_pbr_lazy_state")


def test_prepare_offloads_block_params(tmp_path):
    """After prepare(), all block parameters must be on meta device."""
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)

    prepare(model, model_dir, target_device="cpu")

    for block in model.layers:
        for p in block.parameters():
            assert p.is_meta, f"Expected meta, got {p.device}"


def test_prepare_leaves_non_block_params_on_cpu(tmp_path):
    """After prepare(), non-block parameters (embed, norm) must still be on CPU."""
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)

    prepare(model, model_dir, target_device="cpu")

    for p in model.embed.parameters():
        assert p.device.type == "cpu"
    for p in model.norm.parameters():
        assert p.device.type == "cpu"


def test_prepare_explicit_layers_path(tmp_path):
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu", decoder_layers_path="layers")
    assert hasattr(model, "_pbr_lazy_state")
    finalize(model)


# ---------------------------------------------------------------------------
# finalize
# ---------------------------------------------------------------------------


def test_finalize_no_op_when_not_prepared():
    model = TinyModel()
    finalize(model)  # must not raise


def test_finalize_removes_state(tmp_path):
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu")
    assert hasattr(model, "_pbr_lazy_state")
    finalize(model)
    assert not hasattr(model, "_pbr_lazy_state")


def test_finalize_restores_params(tmp_path):
    """After finalize(), block parameters must be real tensors again."""
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)

    prepare(model, model_dir, target_device="cpu")
    for block in model.layers:
        assert all(p.is_meta for p in block.parameters())

    finalize(model)
    for block in model.layers:
        assert all(not p.is_meta for p in block.parameters())


# ---------------------------------------------------------------------------
# End-to-end correctness
# ---------------------------------------------------------------------------


def test_forward_hooks_fire_during_forward(tmp_path):
    """Verify that pre/post hooks fire (params go real → meta → real → …)."""
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu")

    ids = torch.randint(0, 32, (1, 4))
    with torch.no_grad():
        _ = model(ids)

    # After forward, block params must be back on meta
    for block in model.layers:
        for p in block.parameters():
            assert p.is_meta

    finalize(model)


def test_end_to_end_correctness(tmp_path):
    """Output with lazy_loader must match output of the original model."""
    torch.manual_seed(0)
    ref = TinyModel()
    ref.eval()

    model_dir = _save_fake_checkpoint(ref, tmp_path)

    torch.manual_seed(0)
    pbr = TinyModel()
    pbr.load_state_dict(ref.state_dict())
    pbr.eval()

    prepare(pbr, model_dir, target_device="cpu")

    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        out_ref = ref(ids)
        out_pbr = pbr(ids)

    assert torch.allclose(out_ref, out_pbr, atol=1e-6), f"max diff: {(out_ref - out_pbr).abs().max()}"

    finalize(pbr)

    with torch.no_grad():
        out_final = pbr(ids)
    assert torch.allclose(out_ref, out_final, atol=1e-6)


def test_prepare_auto_detects_layers_path(tmp_path):
    model = TinyModel()
    assert infer_decoder_layers_path(model) == "layers"
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu")
    assert hasattr(model, "_pbr_lazy_state")
    finalize(model)


# ---------------------------------------------------------------------------
# CPU RAM offload mode
# ---------------------------------------------------------------------------


def test_cpu_offload_params_go_to_meta(tmp_path):
    """After prepare(), block params must be on meta device (RAM offload selected)."""
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu")
    for block in model.layers:
        for p in block.parameters():
            assert p.is_meta, f"Expected meta, got {p.device}"
    finalize(model)


def test_cpu_offload_non_block_params_unchanged(tmp_path):
    """After prepare(), non-block params must remain on CPU."""
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu")
    for p in model.embed.parameters():
        assert p.device.type == "cpu"
    for p in model.norm.parameters():
        assert p.device.type == "cpu"
    finalize(model)


def test_prepare_moves_non_decoder_params_and_buffers_to_target(tmp_path):
    """prepare() should leave non-decoder modules ready on the target device."""
    model = TinyModel()
    model.register_buffer("global_buffer", torch.ones(1))
    model_dir = _save_fake_checkpoint(model, tmp_path)

    prepare(model, model_dir, target_device="meta")

    assert model.embed.weight.is_meta
    assert model.norm.weight.is_meta
    assert model.norm.bias.is_meta
    assert model.global_buffer.is_meta


def test_prepare_keeps_tied_weights_tied(tmp_path):
    """prepare() must not untie `lm_head.weight` from `embed.weight`."""

    class TiedModel(TinyModel):
        def __init__(self, vocab: int = 32, dim: int = 8):
            super().__init__(vocab=vocab, dim=dim)
            self.lm_head = nn.Linear(dim, vocab, bias=False)
            self.lm_head.weight = self.embed.weight

    model = TiedModel()
    assert model.lm_head.weight is model.embed.weight
    model_dir = _save_fake_checkpoint(model, tmp_path)

    prepare(model, model_dir, target_device="cpu")

    assert model.lm_head.weight is model.embed.weight
    assert model.lm_head.weight.data_ptr() == model.embed.weight.data_ptr()


def test_prepare_preserves_attributes_attached_to_non_decoder_params(tmp_path):
    """Attributes hung on a parameter object must survive the move to the target device."""
    model = TinyModel()
    model.embed.weight.scale = torch.ones(1)
    model_dir = _save_fake_checkpoint(model, tmp_path)

    prepare(model, model_dir, target_device="cpu")

    assert hasattr(model.embed.weight, "scale")


def test_cpu_offload_end_to_end_correctness(tmp_path):
    """RAM-offload output must match the original model output."""
    torch.manual_seed(42)
    ref = TinyModel()
    ref.eval()
    model_dir = _save_fake_checkpoint(ref, tmp_path)

    torch.manual_seed(42)
    pbr = TinyModel()
    pbr.load_state_dict(ref.state_dict())
    pbr.eval()

    prepare(pbr, model_dir, target_device="cpu")

    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        out_ref = ref(ids)
        out_pbr = pbr(ids)

    assert torch.allclose(out_ref, out_pbr, atol=1e-6), f"max diff: {(out_ref - out_pbr).abs().max()}"

    finalize(pbr)

    with torch.no_grad():
        out_final = pbr(ids)
    assert torch.allclose(out_ref, out_final, atol=1e-6)


def test_cpu_offload_hooks_fire(tmp_path):
    """After each forward block params must be back on meta."""
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu")

    ids = torch.randint(0, 32, (1, 4))
    with torch.no_grad():
        _ = model(ids)

    for block in model.layers:
        for p in block.parameters():
            assert p.is_meta

    finalize(model)


def test_cpu_offload_finalize_restores(tmp_path):
    """After finalize(), block params must be real tensors on target device."""
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu")
    finalize(model)
    for block in model.layers:
        for p in block.parameters():
            assert not p.is_meta
            assert p.device.type == "cpu"


def test_ram_offload_selected_when_ram_available(tmp_path):
    """cpu_caches must be populated when free RAM is sufficient (always true in CI)."""
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu")
    # TinyModel is tiny — RAM offload should always be selected.
    assert len(model._pbr_lazy_state.cpu_caches) == len(model.layers)
    finalize(model)


# ---------------------------------------------------------------------------
# Disk-streaming offload mode (forced by making the RAM probe fail)
# ---------------------------------------------------------------------------


def _force_disk_mode(monkeypatch) -> None:
    """Select the disk-streaming branch regardless of how the backend is chosen.

    The selection rule itself is covered by `test_backend_selection_*`; these tests are
    about the streaming behaviour, so they pin the branch rather than the rule.
    """
    from quark.torch.utils.per_block_runner import lazy_loader

    monkeypatch.setattr(lazy_loader.PerBlockLazyLoader, "_choose_backend", lambda *args, **kwargs: False)


def test_backend_selection_prefers_cpu_ram_for_materialised_weights(tmp_path):
    """Caching resident weights costs no extra RAM, so it must always win."""
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)

    prepare(model, model_dir, target_device="cpu")

    assert model._pbr_lazy_state.use_cpu_ram
    assert any(cache is not None for cache in model._pbr_lazy_state.cpu_caches)
    finalize(model)


def test_backend_selection_falls_back_to_disk_for_meta_weights(tmp_path):
    """A block with nothing materialised has nothing to cache; it has to come from disk."""
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)
    for block in model.layers:
        _offload_block_params_to_meta(block)

    prepare(model, model_dir, target_device="cpu")

    assert not model._pbr_lazy_state.use_cpu_ram
    assert all(cache is None for cache in model._pbr_lazy_state.cpu_caches)
    finalize(model)


def test_disk_mode_no_cpu_cache(tmp_path, monkeypatch):
    """When RAM is reported unavailable, prepare() must select disk streaming
    (cpu_caches entries are all None)."""
    _force_disk_mode(monkeypatch)
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu")
    assert all(cache is None for cache in model._pbr_lazy_state.cpu_caches)
    finalize(model)


def test_disk_mode_end_to_end_correctness(tmp_path, monkeypatch):
    """Disk-streaming output must match the original model output, including
    after finalize()."""
    torch.manual_seed(7)
    ref = TinyModel()
    ref.eval()
    model_dir = _save_fake_checkpoint(ref, tmp_path)

    torch.manual_seed(7)
    pbr = TinyModel()
    pbr.load_state_dict(ref.state_dict())
    pbr.eval()

    _force_disk_mode(monkeypatch)
    prepare(pbr, model_dir, target_device="cpu")

    ids = torch.randint(0, 32, (2, 6))
    with torch.no_grad():
        out_ref = ref(ids)
        out_pbr = pbr(ids)

    assert torch.allclose(out_ref, out_pbr, atol=1e-6), f"max diff: {(out_ref - out_pbr).abs().max()}"

    finalize(pbr)
    with torch.no_grad():
        out_final = pbr(ids)
    assert torch.allclose(out_ref, out_final, atol=1e-6)


def test_disk_mode_state_dict_transform(tmp_path, monkeypatch):
    """The state_dict_transform callback must be applied to each block's state
    dict on the disk-streaming path."""
    _force_disk_mode(monkeypatch)
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)

    seen_keys: list[str] = []

    def transform(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        seen_keys.extend(state_dict.keys())
        return {key: tensor * 0.0 for key, tensor in state_dict.items()}

    prepare(model, model_dir, target_device="cpu", state_dict_transform=transform)

    ids = torch.randint(0, 32, (1, 4))
    with torch.no_grad():
        _ = model(ids)

    assert any(key.endswith("fc.weight") for key in seen_keys)
    finalize(model)


def test_ram_offload_state_dict_transform(tmp_path):
    """The state_dict_transform callback must also be applied on the CPU-RAM path,
    so behavior is identical regardless of which backend is auto-selected."""
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)

    seen_keys: list[str] = []

    def transform(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        seen_keys.extend(state_dict.keys())
        return {key: torch.zeros_like(tensor) for key, tensor in state_dict.items()}

    prepare(model, model_dir, target_device="cpu", state_dict_transform=transform)
    # TinyModel is tiny, so the CPU-RAM backend is selected (cpu_caches populated).
    assert any(cache is not None for cache in model._pbr_lazy_state.cpu_caches)

    ids = torch.randint(0, 32, (1, 4))
    with torch.no_grad():
        output = model(ids)

    assert any(key.endswith("fc.weight") for key in seen_keys)
    # Weights were zeroed by the transform, so every block output is the bias-free
    # zero map; the final LayerNorm of an all-zero input is all zeros.
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)
    finalize(model)


def test_prepare_moves_live_buffers_to_target(tmp_path):
    """prepare() should move live buffers to the target device with the runnable model."""

    class BlockWithBuffer(nn.Module):
        def __init__(self, dim: int = 8):
            super().__init__()
            self.fc = nn.Linear(dim, dim, bias=False)
            self.register_buffer("inv_freq", torch.ones(dim))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.fc(x)

    class ModelWithBuffer(nn.Module):
        def __init__(self, vocab: int = 32, dim: int = 8, n_layers: int = 2):
            super().__init__()
            self.embed = nn.Embedding(vocab, dim)
            self.layers = nn.ModuleList([BlockWithBuffer(dim) for _ in range(n_layers)])
            self.norm = nn.LayerNorm(dim)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            x = self.embed(ids)
            for layer in self.layers:
                x = layer(x)
            return self.norm(x)

    model = ModelWithBuffer()
    model_dir = _save_fake_checkpoint(model, tmp_path)

    prepare(model, model_dir, target_device="meta")

    for block in model.layers:
        assert block.inv_freq.is_meta


# ---------------------------------------------------------------------------
# GPU-resident leading blocks (n_gpu_blocks > 0)
# ---------------------------------------------------------------------------


def test_n_gpu_blocks_keeps_leading_block_resident(tmp_path):
    """With n_gpu_blocks=1 the first block stays resident (no swap hook, cache
    entry is None) while later blocks are offloaded to meta."""
    model = TinyModel(n_layers=2)
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu", n_gpu_blocks=1)

    cpu_caches = model._pbr_lazy_state.cpu_caches
    assert cpu_caches[0] is None  # resident block has no cache entry
    for parameter in model.layers[0].parameters():
        assert not parameter.is_meta
    for parameter in model.layers[1].parameters():
        assert parameter.is_meta
    finalize(model)


def test_n_gpu_blocks_end_to_end_correctness(tmp_path):
    """Output with a GPU-resident leading block must match the original model."""
    torch.manual_seed(11)
    ref = TinyModel(n_layers=2)
    ref.eval()
    model_dir = _save_fake_checkpoint(ref, tmp_path)

    torch.manual_seed(11)
    pbr = TinyModel(n_layers=2)
    pbr.load_state_dict(ref.state_dict())
    pbr.eval()

    prepare(pbr, model_dir, target_device="cpu", n_gpu_blocks=1)

    ids = torch.randint(0, 32, (2, 5))
    with torch.no_grad():
        out_ref = ref(ids)
        out_pbr = pbr(ids)

    assert torch.allclose(out_ref, out_pbr, atol=1e-6), f"max diff: {(out_ref - out_pbr).abs().max()}"
    finalize(pbr)


# ---------------------------------------------------------------------------
# weight.scale reattachment (quantized-style layers carrying a `scale` param)
# ---------------------------------------------------------------------------


class ScaledLinearBlock(nn.Module):
    """A block whose linear carries a sibling ``scale`` parameter that must be
    re-linked to ``weight.scale`` after each swap-in."""

    def __init__(self, dim: int = 8):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=False)
        self.fc.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ScaledModel(nn.Module):
    def __init__(self, vocab: int = 32, dim: int = 8, n_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = nn.ModuleList([ScaledLinearBlock(dim) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


def test_ram_offload_reattaches_weight_scale(tmp_path):
    """On the RAM-offload path, weight.scale must be re-linked during forward."""
    model = ScaledModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu")

    ids = torch.randint(0, 32, (1, 4))
    with torch.no_grad():
        _ = model(ids)

    finalize(model)
    for block in model.layers:
        assert hasattr(block.fc.weight, "scale")


def test_n_gpu_blocks_reattaches_weight_scale(tmp_path):
    """A GPU-resident block carrying a scale param must have weight.scale linked
    immediately after prepare()."""
    model = ScaledModel(n_layers=2)
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu", n_gpu_blocks=1)
    assert hasattr(model.layers[0].fc.weight, "scale")
    finalize(model)


def test_disk_mode_refuses_weights_modified_in_memory(tmp_path, monkeypatch):
    """Algorithms edit weights before the loader exists; disk streaming would undo that."""
    _force_disk_mode(monkeypatch)
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)

    with pytest.raises(NotImplementedError, match="weight modifications"):
        prepare(model, model_dir, target_device="cpu", weights_modified_in_memory=True)


def test_cpu_ram_mode_allows_weights_modified_in_memory(tmp_path):
    """RAM caching snapshots the live weights, so edited weights survive."""
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)
    with torch.no_grad():
        model.layers[0].fc.weight.mul_(3.0)
    modified = model.layers[0].fc.weight.detach().clone()

    prepare(model, model_dir, target_device="cpu", weights_modified_in_memory=True)
    finalize(model)

    assert torch.equal(model.layers[0].fc.weight, modified)


# ---------------------------------------------------------------------------
# Progress label
# ---------------------------------------------------------------------------


def test_progress_label_without_batch_count(tmp_path):
    """A single pass over the blocks is labelled ``idx/n`` with no batch suffix."""
    model = TinyModel(n_layers=2)
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu")
    loader = model._pbr_lazy_state

    assert loader._progress_label(0, 2) == "1/2"
    assert loader._progress_label(1, 2) == "2/2"
    finalize(model)


def test_progress_label_counts_batches_when_total_unknown(tmp_path):
    """With n_batches unknown, replaying the stack bumps an open-ended batch counter."""
    model = TinyModel(n_layers=2)
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu")
    loader = model._pbr_lazy_state

    loader._progress_label(0, 2)
    loader._progress_label(1, 2)
    # Second pass: block 0 is the first hooked block, so it opens batch 2.
    assert loader._progress_label(0, 2) == "1/2 (batch 2)"
    assert loader._progress_label(1, 2) == "2/2 (batch 2)"
    finalize(model)


def test_progress_label_with_known_batch_count(tmp_path):
    """A known n_batches shows ``batch i/n`` from the very first block."""
    model = TinyModel(n_layers=2)
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu", n_batches=3)
    loader = model._pbr_lazy_state

    assert loader._progress_label(0, 2) == "1/2 (batch 1/3)"
    assert loader._progress_label(1, 2) == "2/2 (batch 1/3)"
    assert loader._progress_label(0, 2) == "1/2 (batch 2/3)"
    finalize(model)


def test_progress_label_counts_batches_off_first_hooked_block(tmp_path):
    """With n_gpu_blocks=1 the resident block 0 is never hooked, so block 1 opens the batch."""
    model = TinyModel(n_layers=2)
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu", n_gpu_blocks=1, n_batches=2)
    loader = model._pbr_lazy_state

    assert loader._progress_label(1, 2) == "2/2 (batch 1/2)"
    assert loader._progress_label(1, 2) == "2/2 (batch 2/2)"
    finalize(model)
