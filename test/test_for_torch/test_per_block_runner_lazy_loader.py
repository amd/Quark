#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for per_block_runner.lazy_loader."""

import json

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
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}

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


def test_prepare_missing_index(tmp_path):
    """prepare() with a directory that has no index.json should be a no-op."""
    model = TinyModel()
    prepare(model, str(tmp_path))
    assert not hasattr(model, "_pbr_lazy_state")


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
    assert len(model._pbr_lazy_state["cpu_caches"]) == len(model.layers)
    finalize(model)


# ---------------------------------------------------------------------------
# Disk-streaming offload mode (forced by making the RAM probe fail)
# ---------------------------------------------------------------------------


def _force_disk_mode(monkeypatch) -> None:
    """Make the ``/proc/meminfo`` probe raise so ``free_ram_bytes`` falls back to 0,
    which forces the disk-streaming branch regardless of the host's real RAM."""
    real_open = open

    def fake_open(file, *args, **kwargs):
        if file == "/proc/meminfo":
            raise OSError("forced meminfo failure for test")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)


def test_disk_mode_no_cpu_cache(tmp_path, monkeypatch):
    """When RAM is reported unavailable, prepare() must select disk streaming
    (cpu_caches entries are all None)."""
    _force_disk_mode(monkeypatch)
    model = TinyModel()
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu")
    assert all(cache is None for cache in model._pbr_lazy_state["cpu_caches"])
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
    assert any(cache is not None for cache in model._pbr_lazy_state["cpu_caches"])

    ids = torch.randint(0, 32, (1, 4))
    with torch.no_grad():
        output = model(ids)

    assert any(key.endswith("fc.weight") for key in seen_keys)
    # Weights were zeroed by the transform, so every block output is the bias-free
    # zero map; the final LayerNorm of an all-zero input is all zeros.
    assert torch.allclose(output, torch.zeros_like(output), atol=1e-6)
    finalize(model)


def test_prepare_warns_on_live_buffers(tmp_path, monkeypatch):
    """prepare() must warn when an offloaded block carries a live (non-meta) buffer,
    since buffers are not moved with the block and can cause a runtime device mismatch."""

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

    # ScreenLogger sets propagate=False, so caplog cannot see it; capture the
    # warning message directly from the module logger instead.
    from quark.torch.utils.per_block_runner import lazy_loader

    warnings_seen: list[str] = []
    original_warning = lazy_loader.logger.warning

    def record_warning(message, *args, **kwargs):
        warnings_seen.append(message % args if args else message)
        return original_warning(message, *args, **kwargs)

    monkeypatch.setattr(lazy_loader.logger, "warning", record_warning)
    prepare(model, model_dir, target_device="cpu")

    assert any("live buffers" in message for message in warnings_seen)
    finalize(model)


# ---------------------------------------------------------------------------
# GPU-resident leading blocks (n_gpu_blocks > 0)
# ---------------------------------------------------------------------------


def test_n_gpu_blocks_keeps_leading_block_resident(tmp_path):
    """With n_gpu_blocks=1 the first block stays resident (no swap hook, cache
    entry is None) while later blocks are offloaded to meta."""
    model = TinyModel(n_layers=2)
    model_dir = _save_fake_checkpoint(model, tmp_path)
    prepare(model, model_dir, target_device="cpu", n_gpu_blocks=1)

    cpu_caches = model._pbr_lazy_state["cpu_caches"]
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
