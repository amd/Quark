#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Public API for llama.cpp GGUF export."""

from __future__ import annotations

from pathlib import Path

from quark.common.utils.import_utils import is_gguf_available_and_minimum_version
from quark.torch.export.llama_cpp_export.converter import (
    LlamaCppConvertConfig,
    convert_quark_checkpoint_to_llama_cpp_gguf,
)
from quark.torch.export.llama_cpp_export.formats import list_export_formats


def export_llama_cpp_gguf(
    quark_model_dir: str | Path,
    output_dir: str | Path,
    export_format: str,
    *,
    name: str | None = None,
    llama_cpp_dir: str | Path = "/home/l/work/llama.cpp",
    libggml: str | Path | None = None,
    tokenizer_source: str | Path | None = None,
    group_size: int | None = None,
    pack_method: str | None = None,
    native_passthrough: bool | None = None,
    split_max_size: str = "8G",
    max_tensors: int | None = None,
    dry_run: bool = False,
    keep_staging: bool = False,
) -> Path:
    """Export a Quark checkpoint to a llama.cpp-compatible GGUF file.

    This path is independent from ``export_gguf`` and ``export_safetensors``.

    When the checkpoint uses a native scheme that maps 1:1 to a public GGUF
    quant type, export avoids libggml re-quantization:

    * ``uint4_wo_32`` (asymmetric, group size 32) -> ``q4_1``
    * ``int4_wo_32`` (symmetric, group size 32) -> ``q4_0``

    Architectures such as ``Qwen3_5MoeForConditionalGeneration`` are handled
    via llama.cpp ``conversion/`` tensor mapping; only the quant payload is
    produced by Quark.

    Args:
        quark_model_dir: Directory containing Quark AWQ safetensors.
        output_dir: Directory for generated GGUF shards.
        export_format: Target llama.cpp format name, e.g. ``"q4_k_m"`` or
            ``"q8_0"``. See ``list_llama_cpp_export_formats()``.
        name: Output name stem. Defaults to ``quark_model_dir.name``.
        llama_cpp_dir: llama.cpp source tree with ``gguf-py`` and ``conversion``.
        libggml: Path to ``libggml.so``. Defaults to
            ``$llama_cpp_dir/build-hip/bin/libggml.so``.
        tokenizer_source: Optional tokenizer directory when the Quark repo uses
            TokenizersBackend and lacks a standard HF tokenizer export.
        group_size: Quark per-group size. Auto-detected from ``config.json`` when omitted.
        pack_method: Quark pack method, ``"order"`` or ``"reorder"``. Auto-detected when omitted.
        native_passthrough: When ``True``, require a native scheme/format match and pack
            Quark blocks directly (no libggml re-quant). When ``None`` (default), enable
            passthrough automatically for ``q4_0`` / ``q4_1`` when the checkpoint matches.
        split_max_size: GGUF shard size limit.
        max_tensors: Debug option to stop after N tensors.
        dry_run: If True, do not write GGUF payload bytes.
        keep_staging: Keep the temporary metadata staging directory.

    Returns:
        Path to the first GGUF shard, or the output file when unsplit.
    """
    if not is_gguf_available_and_minimum_version():
        raise ImportError(
            "export_llama_cpp_gguf requires gguf>=0.10.0. "
            "Install with: pip install 'gguf>=0.10.0'"
        )

    quark_model_dir = Path(quark_model_dir).resolve()
    llama_cpp_dir = Path(llama_cpp_dir).resolve()
    if libggml is None:
        libggml = llama_cpp_dir / "build-hip" / "bin" / "libggml.so"

    config = LlamaCppConvertConfig(
        quark_model_dir=quark_model_dir,
        output_dir=Path(output_dir).resolve(),
        export_format=export_format,
        name=name or quark_model_dir.name,
        llama_cpp_dir=llama_cpp_dir,
        libggml=Path(libggml),
        tokenizer_source=Path(tokenizer_source).resolve() if tokenizer_source else None,
        group_size=group_size,
        pack_method=pack_method,
        native_passthrough=native_passthrough,
        split_max_size=split_max_size,
        max_tensors=max_tensors,
        dry_run=dry_run,
        keep_staging=keep_staging,
    )
    return convert_quark_checkpoint_to_llama_cpp_gguf(config)


def list_llama_cpp_export_formats() -> list[str]:
    """Return supported llama.cpp export format names."""
    return list_export_formats()
