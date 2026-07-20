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
    assert "q4_1" in LLAMA_CPP_EXPORT_FORMATS
    assert "q4_0" in LLAMA_CPP_EXPORT_FORMATS
    assert len(LLAMA_CPP_EXPORT_FORMATS) == 32


@require_torch_higher_or_equal("2.4.0")
def test_native_q4_1_pack_roundtrip() -> None:
    import gguf
    import torch
    from pathlib import Path
    from safetensors import safe_open

    from quark.torch.export.llama_cpp_export.quark_gguf_pack import pack_affine_uint4_to_q4_1
    from quark.torch.export.llama_cpp_export.quark_awq_unpack import dequantize_uint4_weight

    model_dir = Path(
        os.environ.get(
            "UINT4_TEST_MODEL",
            "/home/l/work/quantization_work/qwen36_uint4_q4_1_gguf/output/uint4-wo32",
        )
    )
    weights = model_dir / "model.safetensors"
    if not weights.exists():
        pytest.skip(f"uint4 checkpoint not found: {weights}")

    name = "model.language_model.layers.0.linear_attn.in_proj_a.weight"
    with safe_open(str(weights), framework="pt", device="cpu") as f:
        if name not in f.keys():
            pytest.skip(f"{name} not in checkpoint")
        w = f.get_tensor(name)
        s = f.get_tensor(name.replace(".weight", ".weight_scale"))
        z = f.get_tensor(name.replace(".weight", ".weight_zero_point"))

    ref = dequantize_uint4_weight(
        w, s, z, group_size=32, pack_reorder=True, out_dtype=torch.float32
    ).numpy()
    packed = pack_affine_uint4_to_q4_1(
        w, s, z, group_size=32, pack_reorder=True
    ).numpy()
    restored = gguf.quants.dequantize(packed, gguf.GGMLQuantizationType.Q4_1)
    assert restored.shape == ref.shape
    assert np.abs(restored - ref).max() < 1e-3


def test_validate_native_passthrough_uint4() -> None:
    from quark.torch.export.llama_cpp_export.scheme_compat import validate_native_passthrough
    import json
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        cfg = {
            "config": {"architectures": ["Qwen3_5MoeForConditionalGeneration"]},
            "quantization_config": {
                "global_quant_config": {
                    "weight": {
                        "dtype": "uint4",
                        "symmetric": False,
                        "group_size": 32,
                        "qscheme": "per_group",
                    }
                },
                "export": {"pack_method": "reorder"},
            },
        }
        path = Path(tmp) / "config.json"
        path.write_text(json.dumps(cfg), encoding="utf-8")
        scheme = validate_native_passthrough(Path(tmp), "q4_1")
        assert scheme.gguf_format == "q4_1"
        with pytest.raises(ValueError):
            validate_native_passthrough(Path(tmp), "q4_0")
