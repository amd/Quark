# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Regression tests for the presharded FP8 dequantization fix.

The bug: ``_weight_dequant_fp8`` and the corresponding triton kernel
treat the full ``[M, N]`` tensor as one block-128-scaled tensor. When
the input is the row-wise concatenation of ``n_chunks`` independently-
quantized presharded TP chunks whose ``chunk_rows`` is not a multiple
of ``block_size``, every chunk-boundary scale block straddles two
chunks and applies the wrong chunk's scale to the next chunk's first
rows.

For example MiMo-V2.5-Pro fused QKV (``chunk_rows=3392``,
``block_size=128``, ``3392 % 128 = 64``) loses ~12 % of every QKV
weight to silent corruption after a ``--file2file_quantization``
recovery, which kills downstream wikitext PPL by 4 orders of magnitude
in vLLM.

The fix adds a ``chunk_rows`` parameter to ``_weight_dequant_fp8`` (and
a ``presharded_weights`` ``{glob: chunk_rows}`` hint to
``_recover_fp8_weights``); when set, dequantization is performed per
chunk so the wrong-scale corruption no longer happens.
"""

from __future__ import annotations

import importlib
import sys
import types
from typing import Any

import pytest
import torch

from quark.torch.quantization.config.config import QConfig, QLayerConfig
from quark.torch.quantization.file2file_quantization import (
    _recover_fp8_weights,
    _resolve_presharded_chunk_rows,
)

# ---------- Helpers reused from test_zeta_file2file_quantization ----------


class _FakeSafeOpen:
    def __init__(self, tensor_map: dict[str, torch.Tensor]) -> None:
        self.tensor_map = tensor_map

    def __enter__(self) -> _FakeSafeOpen:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    def keys(self) -> list[str]:
        return list(self.tensor_map.keys())

    def get_tensor(self, tensor_name: str) -> torch.Tensor:
        return self.tensor_map[tensor_name]


def _build_minimal_quant_config() -> QConfig:
    return QConfig(global_quant_config=QLayerConfig(), exclude=[])


def _quantize_per_chunk_block_fp8(
    real: torch.Tensor,
    block_size: int,
    chunk_rows: int,
    n_chunks: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Round-trip a real BF16 tensor through per-chunk per-block FP8 e4m3
    quantization, mimicking how the source-side checkpoint stores a
    presharded TP fused-QKV weight. Returns ``(fp8_stored, scale_inv)``
    in the concatenated layout.
    """
    rows, cols = real.shape
    assert rows == n_chunks * chunk_rows
    chunk_scale_rows = (chunk_rows + block_size - 1) // block_size
    cols_blocks = (cols + block_size - 1) // block_size
    fp8_chunks: list[torch.Tensor] = []
    scale_chunks: list[torch.Tensor] = []
    for ci in range(n_chunks):
        chunk = real[ci * chunk_rows : (ci + 1) * chunk_rows]
        fp8 = torch.empty(chunk_rows, cols, dtype=torch.float8_e4m3fn)
        scale = torch.empty(chunk_scale_rows, cols_blocks, dtype=torch.float32)
        for bi in range(chunk_scale_rows):
            for bj in range(cols_blocks):
                r0 = bi * block_size
                r1 = min((bi + 1) * block_size, chunk_rows)
                c0 = bj * block_size
                c1 = min((bj + 1) * block_size, cols)
                block = chunk[r0:r1, c0:c1]
                amax = block.abs().max().clamp_min(1e-12)
                block_scale = amax / 448.0
                fp8[r0:r1, c0:c1] = (block.float() / block_scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                scale[bi, bj] = block_scale
        fp8_chunks.append(fp8)
        scale_chunks.append(scale)
    return torch.cat(fp8_chunks, dim=0), torch.cat(scale_chunks, dim=0)


# ---------- _resolve_presharded_chunk_rows ----------


@pytest.mark.parametrize(
    "weight_name,table,expected",
    [
        ("model.layers.1.self_attn.qkv_proj.weight", {"*.self_attn.qkv_proj.weight": 3392}, 3392),
        ("model.layers.1.mlp.gate_proj.weight", {"*.self_attn.qkv_proj.weight": 3392}, None),
        ("anything", None, None),
        ("anything", {}, None),
        # First-match wins.
        ("model.layers.1.self_attn.qkv_proj.weight", {"*qkv*": 3000, "*.self_attn.qkv_proj.weight": 3392}, 3000),
    ],
)
def test_resolve_presharded_chunk_rows(weight_name: str, table: dict[str, int] | None, expected: int | None) -> None:
    assert _resolve_presharded_chunk_rows(weight_name, table) == expected


# ---------- _weight_dequant_fp8 chunk_rows behavior ----------


def _make_pure_python_dequant(
    block_size: int, chunk_rows: int, n_chunks: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build a tiny presharded FP8 + scale_inv fixture and the correct
    per-chunk-dequanted reference. Designed so the chunk_rows is NOT a
    multiple of block_size to exercise the bug path."""
    torch.manual_seed(0)
    cols = 4
    rows = n_chunks * chunk_rows
    chunks = []
    for ci in range(n_chunks):
        # Distinct magnitude per chunk so a misapplied scale is detectable.
        magnitude = 0.01 if ci == 0 else 1.0
        chunks.append(magnitude * torch.randn(chunk_rows, cols))
    real = torch.cat(chunks, dim=0)
    fp8, scale = _quantize_per_chunk_block_fp8(real, block_size, chunk_rows, n_chunks)
    # Reference: per-chunk dequant manually here (independent of the unit
    # under test) so we don't compare the function against itself.
    correct = torch.empty(rows, cols, dtype=torch.bfloat16)
    chunk_scale_rows = (chunk_rows + block_size - 1) // block_size
    for ci in range(n_chunks):
        cw = fp8[ci * chunk_rows : (ci + 1) * chunk_rows]
        cs = scale[ci * chunk_scale_rows : (ci + 1) * chunk_scale_rows]
        cs_full = cs.repeat_interleave(block_size, dim=0)[: cw.shape[0]].repeat_interleave(block_size, dim=1)[:, :cols]
        correct[ci * chunk_rows : (ci + 1) * chunk_rows] = (cw.float() * cs_full.float()).to(torch.bfloat16)
    return fp8, scale, correct


def _import_with_fake_triton(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Reload ``file2file_quantization`` with a CPU-only fake triton + a
    fake ``_weight_dequant_kernel`` that does the actual dequant in
    PyTorch. Returns the reloaded module."""
    import quark.common.utils.import_utils as import_utils
    import quark.torch.quantization.file2file_quantization as file2file_quantization

    fake_triton = types.ModuleType("triton")
    fake_tl = types.ModuleType("triton.language")
    fake_triton.__path__ = []
    fake_triton.cdiv = lambda dividend, divisor: (dividend + divisor - 1) // divisor
    fake_triton.jit = lambda function: function
    fake_triton.language = fake_tl
    fake_tl.constexpr = object()

    class _FakeKernel:
        def __getitem__(self, grid):  # type: ignore[no-untyped-def]
            def launcher(
                x: torch.Tensor,
                s: torch.Tensor,
                y: torch.Tensor,
                m_dim: int,
                n_dim: int,
                *,
                BLOCK_SIZE: int,
            ) -> None:
                # Reproduce the triton kernel's broadcast in pure PyTorch so the
                # tests are GPU-free. ``s_full`` is built at the same granularity
                # as the kernel's broadcast (``s_ptr + pid_m * n + pid_n``).
                s_full = (
                    s.float()
                    .repeat_interleave(BLOCK_SIZE, dim=0)[:m_dim]
                    .repeat_interleave(BLOCK_SIZE, dim=1)[:, :n_dim]
                )
                y.copy_((x.float() * s_full).to(y.dtype))

            return launcher

    monkeypatch.setattr(import_utils, "is_triton_available", lambda: True)
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.language", fake_tl)
    importlib.reload(file2file_quantization)
    monkeypatch.setattr(file2file_quantization, "_weight_dequant_kernel", _FakeKernel())
    return file2file_quantization


def test_weight_dequant_fp8_default_path_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """When ``chunk_rows`` is not provided, the function must produce the
    same result as before the fix (full-tensor block-broadcast). This
    pins down the absence of side effects for non-presharded weights."""
    file2file_quantization = _import_with_fake_triton(monkeypatch)
    try:
        torch.manual_seed(0)
        # Standard (non-presharded) input: rows = 4 (multiple of block_size 2)
        x = torch.randn(4, 4).to(torch.float8_e4m3fn)
        s = torch.full((2, 2), 0.1, dtype=torch.float32)
        # Reference = naive full-tensor dequant
        s_full = s.repeat_interleave(2, dim=0).repeat_interleave(2, dim=1)
        expected = (x.float() * s_full.float()).to(torch.bfloat16)
        # Default call (no chunk_rows): must equal the naive dequant.
        actual = file2file_quantization._weight_dequant_fp8(
            x.contiguous(),
            s.contiguous(),
            block_size=2,
            model_dtype=torch.bfloat16,
        )
        assert torch.equal(actual, expected)
    finally:
        importlib.reload(file2file_quantization)


def test_weight_dequant_fp8_chunk_rows_corrects_presharded_layout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``chunk_rows=N`` must dequantize each chunk with its own scale slice,
    matching the per-chunk reference (and DIFFERING from the buggy
    full-tensor broadcast at chunk boundaries)."""
    file2file_quantization = _import_with_fake_triton(monkeypatch)
    try:
        block_size = 2
        chunk_rows = 5  # not a multiple of block_size -> bug triggers
        n_chunks = 2
        fp8, scale, correct = _make_pure_python_dequant(block_size, chunk_rows, n_chunks)

        # Buggy (no chunk_rows) — must NOT match the per-chunk reference,
        # so we can be sure our fix actually changes behavior on this input.
        buggy = file2file_quantization._weight_dequant_fp8(
            fp8.contiguous(),
            scale.contiguous(),
            block_size=block_size,
            model_dtype=torch.bfloat16,
        )
        assert not torch.equal(buggy, correct), (
            "Pre-fix path silently equals the per-chunk reference on this "
            "fixture; the test fixture is no longer exercising the bug."
        )

        # Fixed (with chunk_rows) — must match the per-chunk reference.
        fixed = file2file_quantization._weight_dequant_fp8(
            fp8.contiguous(),
            scale.contiguous(),
            block_size=block_size,
            model_dtype=torch.bfloat16,
            chunk_rows=chunk_rows,
        )
        assert torch.equal(fixed, correct)
    finally:
        importlib.reload(file2file_quantization)


def test_weight_dequant_fp8_chunk_rows_validates_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity: assert the input shape constraints raise cleanly when violated."""
    file2file_quantization = _import_with_fake_triton(monkeypatch)
    try:
        x = torch.zeros(7, 2).to(torch.float8_e4m3fn).contiguous()
        s = torch.zeros(4, 1, dtype=torch.float32).contiguous()
        with pytest.raises(AssertionError, match="multiple of chunk_rows"):
            file2file_quantization._weight_dequant_fp8(
                x,
                s,
                block_size=2,
                model_dtype=torch.bfloat16,
                chunk_rows=3,
            )
        with pytest.raises(AssertionError, match="scale rows"):
            file2file_quantization._weight_dequant_fp8(
                x[:6],
                torch.zeros(5, 1, dtype=torch.float32).contiguous(),
                block_size=2,
                model_dtype=torch.bfloat16,
                chunk_rows=3,
            )
    finally:
        importlib.reload(file2file_quantization)


# ---------- _recover_fp8_weights opt-in plumbing ----------


def test_recover_fp8_weights_threads_chunk_rows_from_kwarg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Passing ``presharded_weights`` to ``_recover_fp8_weights`` must reach
    ``_weight_dequant_fp8`` as ``chunk_rows`` for matching tensor names."""
    weight_name = "model.layers.1.self_attn.qkv_proj.weight"
    other_name = "model.layers.1.mlp.gate_proj.weight"
    tensor_map = {
        weight_name: torch.ones((2, 2), dtype=torch.float8_e4m3fn),
        f"{weight_name}_scale_inv": torch.ones((1, 1), dtype=torch.float32),
        other_name: torch.ones((2, 2), dtype=torch.float8_e4m3fn),
        f"{other_name}_scale_inv": torch.ones((1, 1), dtype=torch.float32),
    }
    seen: list[tuple[str, int | None]] = []

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        return _FakeSafeOpen(tensor_map=tensor_map)

    def fake_weight_dequant_fp8(
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        block_size: int = 128,
        *,
        model_dtype: torch.dtype,
        chunk_rows: int | None = None,
    ) -> torch.Tensor:
        # The recoverer iterates an unordered set; we match on identity by
        # weight tensor pointer rather than name.
        for k, v in tensor_map.items():
            if v is weight:
                seen.append((k, chunk_rows))
                break
        return torch.zeros(weight.shape, dtype=model_dtype)

    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._weight_dequant_fp8",
        fake_weight_dequant_fp8,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._empty_cache_if_cuda",
        lambda device: None,
    )

    _recover_fp8_weights(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict={"quant_method": "fp8"},
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.float16,
        presharded_weights={"*.self_attn.qkv_proj.weight": 1024},
    )

    seen_dict = dict(seen)
    assert seen_dict[weight_name] == 1024, f"expected chunk_rows=1024 on qkv_proj, got {seen_dict[weight_name]}"
    assert seen_dict[other_name] is None, f"expected chunk_rows=None on non-qkv weight, got {seen_dict[other_name]}"


def test_recover_fp8_weights_picks_up_presharded_from_hf_quant_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``presharded_weights`` is not passed explicitly, the recoverer
    must fall back to ``hf_quant_config_dict['presharded_weights']``."""
    weight_name = "model.layers.1.self_attn.qkv_proj.weight"
    tensor_map = {
        weight_name: torch.ones((2, 2), dtype=torch.float8_e4m3fn),
        f"{weight_name}_scale_inv": torch.ones((1, 1), dtype=torch.float32),
    }
    seen: list[int | None] = []

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        return _FakeSafeOpen(tensor_map=tensor_map)

    def fake_weight_dequant_fp8(
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        block_size: int = 128,
        *,
        model_dtype: torch.dtype,
        chunk_rows: int | None = None,
    ) -> torch.Tensor:
        seen.append(chunk_rows)
        return torch.zeros(weight.shape, dtype=model_dtype)

    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._weight_dequant_fp8",
        fake_weight_dequant_fp8,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._empty_cache_if_cuda",
        lambda device: None,
    )

    _recover_fp8_weights(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict={
            "quant_method": "fp8",
            "presharded_weights": {"*.self_attn.qkv_proj.weight": 2048},
        },
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.float16,
    )
    assert seen == [2048]


def test_recover_fp8_weights_does_not_pass_chunk_rows_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``presharded_weights`` is not configured, the recoverer must
    NOT pass ``chunk_rows`` as a kwarg — preserving compatibility with
    third-party monkeypatches whose ``_weight_dequant_fp8`` signature
    predates this fix."""
    weight_name = "model.layers.1.mlp.gate_proj.weight"
    tensor_map = {
        weight_name: torch.ones((2, 2), dtype=torch.float8_e4m3fn),
        f"{weight_name}_scale_inv": torch.ones((1, 1), dtype=torch.float32),
    }
    received_kwargs: list[set[str]] = []

    def fake_safe_open(_path: str, framework: str, device: str) -> _FakeSafeOpen:
        return _FakeSafeOpen(tensor_map=tensor_map)

    def fake_old_signature_dequant(
        weight: torch.Tensor,
        scale_inv: torch.Tensor,
        block_size: int = 128,
        *,
        model_dtype: torch.dtype,
        **kwargs: Any,
    ) -> torch.Tensor:
        received_kwargs.append(set(kwargs.keys()))
        return torch.zeros(weight.shape, dtype=model_dtype)

    monkeypatch.setattr("quark.torch.quantization.file2file_quantization.safe_open", fake_safe_open)
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._weight_dequant_fp8",
        fake_old_signature_dequant,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._empty_cache_if_cuda",
        lambda device: None,
    )

    _recover_fp8_weights(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        hf_quant_config_dict={"quant_method": "fp8"},
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.float16,
    )
    assert received_kwargs == [set()], (
        f"Expected no extra kwargs when presharded_weights is unset, got {received_kwargs}"
    )


# ---------- presharded_weights threading through the call layers above _recover_fp8_weights ----------


class _ThreadingComplete(Exception):
    """Raised by stubs after capturing kwargs to short-circuit the parent.

    Lets the threading tests verify the kwarg propagated without running
    the rest of the parent function (which would do real work).
    """


def test_load_safetensor_with_recover_threads_presharded_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_load_safetensor_with_recover(..., presharded_weights=...)`` must
    forward the kwarg to ``_recover_fp8_weights``."""
    from quark.torch.quantization.file2file_quantization import _load_safetensor_with_recover

    captured: dict[str, Any] = {}

    def fake_recover_fp8_weights(**kwargs: Any) -> dict[str, torch.Tensor]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._recover_fp8_weights",
        fake_recover_fp8_weights,
    )

    presharded = {"*.self_attn.qkv_proj.weight": 1024}
    _load_safetensor_with_recover(
        safetensor_path="dummy.safetensors",
        quant_config=_build_minimal_quant_config(),
        device="cpu",
        keep_excluded_layers_as_original_model_state=False,
        model_dtype=torch.float16,
        hf_model_config={"quantization_config": {"quant_method": "fp8"}},
        presharded_weights=presharded,
    )
    assert captured.get("presharded_weights") == presharded


def test_quantize_and_save_safetensor_shard_threads_presharded_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``_quantize_and_save_safetensor_shard(..., presharded_weights=...)`` must
    forward the kwarg to ``_load_safetensor_with_recover``."""
    from quark.torch.quantization.file2file_quantization import _quantize_and_save_safetensor_shard

    captured: dict[str, Any] = {}

    def fake_load(**kwargs: Any) -> dict[str, torch.Tensor]:
        captured.update(kwargs)
        raise _ThreadingComplete

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._load_safetensor_with_recover",
        fake_load,
    )

    presharded = {"*.self_attn.qkv_proj.weight": 1024}
    with pytest.raises(_ThreadingComplete):
        _quantize_and_save_safetensor_shard(
            safetensor_path="/tmp/dummy.safetensors",
            export_path="/tmp/out",
            quant_config=_build_minimal_quant_config(),
            device="cpu",
            keep_excluded_layers_as_original_model_state=False,
            model_dtype=torch.float16,
            presharded_weights=presharded,
        )
    assert captured.get("presharded_weights") == presharded


def test_quantize_model_per_safetensor_threads_presharded_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``quantize_model_per_safetensor(..., presharded_weights=...)`` must
    forward the kwarg to ``_quantize_and_save_safetensor_shard``."""
    from quark.torch.quantization.file2file_quantization import quantize_model_per_safetensor

    captured: dict[str, Any] = {}

    def fake_shard(**kwargs: Any) -> None:
        captured.update(kwargs)
        raise _ThreadingComplete

    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._quantize_and_save_safetensor_shard",
        fake_shard,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_safetensor_files",
        lambda _path: ["/tmp/dummy.safetensors"],
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_hf_model_config",
        lambda _path: {"torch_dtype": "bfloat16"},
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization._get_model_dtype_from_hf_model_config",
        lambda _cfg: torch.bfloat16,
    )
    monkeypatch.setattr(
        "quark.torch.quantization.file2file_quantization.get_quantization_config",
        lambda _cfg: None,
    )
    monkeypatch.setattr("os.makedirs", lambda *a, **k: None)

    presharded = {"*.self_attn.qkv_proj.weight": 1024}
    with pytest.raises(_ThreadingComplete):
        quantize_model_per_safetensor(
            pretrained_model_path="/tmp/model",
            quant_config=_build_minimal_quant_config(),
            save_path="/tmp/out",
            device="cpu",
            presharded_weights=presharded,
        )
    assert captured.get("presharded_weights") == presharded


def test_direct_quantize_checkpoint_threads_presharded_weights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ModelQuantizer.direct_quantize_checkpoint(..., presharded_weights=...)``
    must forward the kwarg to ``quantize_model_per_safetensor``."""
    from quark.torch.quantization.api import ModelQuantizer

    captured: dict[str, Any] = {}

    def fake_quantize(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(
        "quark.torch.quantization.api.quantize_model_per_safetensor",
        fake_quantize,
    )

    presharded = {"*.self_attn.qkv_proj.weight": 1024}
    quantizer = ModelQuantizer(_build_minimal_quant_config())
    quantizer.direct_quantize_checkpoint(
        pretrained_model_path="/tmp/model",
        save_path="/tmp/out",
        device="cpu",
        presharded_weights=presharded,
    )
    assert captured.get("presharded_weights") == presharded
