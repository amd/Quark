#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from quark.common.utils.testing_utils import require_torch_higher_or_equal
from quark.torch.export.llama_cpp_export.formats import (
    LLAMA_CPP_EXPORT_FORMATS,
    get_export_format,
)
from quark.torch.export.llama_cpp_export.ggml_quantizer import GgmlQuantizer


def _libggml_path() -> Path:
    llama_cpp_dir = Path(os.environ.get("LLAMA_CPP_DIR", "/home/l/work/llama.cpp"))
    libggml = Path(os.environ.get("LIBGGML", llama_cpp_dir / "build-hip" / "bin" / "libggml.so"))
    if not libggml.exists():
        pytest.skip(f"libggml not found: {libggml}")
    return libggml


def _sample_shape(qtype_name: str) -> tuple[int, int]:
    if qtype_name in {"MXFP4"}:
        return (4, 32)
    if qtype_name in {"NVFP4"}:
        return (2, 64)
    return (4, 1024)


@pytest.mark.parametrize("format_name", sorted(LLAMA_CPP_EXPORT_FORMATS))
@require_torch_higher_or_equal("2.4.0")
def test_libggml_roundtrip(format_name: str) -> None:
    fmt = get_export_format(format_name)
    if not fmt.use_libggml:
        pytest.skip(f"{format_name} does not use libggml")

    quantizer = GgmlQuantizer(_libggml_path())
    rng = np.random.default_rng(0)
    data = rng.standard_normal(_sample_shape(fmt.weight_qtype.name), dtype=np.float32)

    try:
        q = quantizer.quantize(data, fmt.weight_qtype)
        restored = quantizer.dequantize(q, fmt.weight_qtype)
    except (RuntimeError, ValueError, OSError) as exc:
        pytest.skip(f"{format_name} unsupported by local libggml: {exc}")

    assert restored.shape == data.shape
    assert np.isfinite(restored).all()


@pytest.mark.parametrize("format_name", ["f32", "f16", "bf16"])
@require_torch_higher_or_equal("2.4.0")
def test_gguf_python_roundtrip(format_name: str) -> None:
    """Non-libggml formats use gguf.quants directly."""
    import gguf

    fmt = get_export_format(format_name)
    assert not fmt.use_libggml

    rng = np.random.default_rng(0)
    data = rng.standard_normal((4, 1024), dtype=np.float32)
    q = gguf.quants.quantize(data, fmt.weight_qtype)
    restored = gguf.quants.dequantize(q, fmt.weight_qtype)
    assert restored.shape == data.shape
    assert np.isfinite(restored).all()


def test_list_export_formats_contains_q4_k_m() -> None:
    assert "q4_k_m" in LLAMA_CPP_EXPORT_FORMATS
    assert "q8_0" in LLAMA_CPP_EXPORT_FORMATS
    assert len(LLAMA_CPP_EXPORT_FORMATS) == 32
