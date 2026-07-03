#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Section helpers + per-artifact source builders for ``hw_emulation``.

Stable-ABI artifact is shared by AOT and JIT via :func:`torch_ops_sources`;
the legacy pybind11 surface is JIT-only and ships through
:func:`legacy_hw_emulation_sources`.

JIT/AOT call sites consume the per-artifact builders below, never the
section helpers directly (see :mod:`quark.common.torch_cpp_build_specs`).
"""

from __future__ import annotations

from pathlib import Path

# Absolute paths so AOT (setup.py cwd) and JIT (site-packages cwd) agree.
_QUARK_ROOT = Path(__file__).resolve().parent.parent.parent
_HW = _QUARK_ROOT / "torch" / "kernel" / "hw_emulation"
_CSRC = _HW / "csrc"
# pyinit_stub exports the PyInit_ symbol Windows setuptools requires; lives
# under quark/common/csrc so future native extensions can share the shim.
_PYINIT_STUB = _QUARK_ROOT / "common" / "csrc" / "pyinit_stub.cpp"
_TORCH_INCLUDE = _QUARK_ROOT / "torch" / "include"


# --- Section helpers (composed by the per-artifact builders below). ---


def stable_abi_kernel_sources_cpu() -> list[str]:
    """Stable-ABI CPU sources: pybind glue stays in ``stable_ops_bindings.cpp``."""
    return [
        str(_CSRC / "python_function_export.cpp"),
        str(_CSRC / "mx" / "cpu" / "funcs.cpp"),
        str(_CSRC / "tqt" / "tqt_op.cpp"),
        str(_CSRC / "stable_ops_bindings.cpp"),
    ]


def stable_abi_kernel_sources_cuda() -> list[str]:
    """Stable-ABI CUDA sources (``.cu``; torch hipifies on ROCm)."""
    return [
        str(_CSRC / "fake_tensor_cuda_hip.cu"),
        str(_CSRC / "mx" / "cuda" / "funcs.cu"),
        str(_CSRC / "mxfp4" / "dequantize.cu"),
        str(_CSRC / "mxfp4" / "fake.cu"),
        str(_CSRC / "tqt" / "tqt.cu"),
    ]


def legacy_kernel_sources_cpu() -> list[str]:
    """Legacy CPU sources for the pre-stable-ABI hw_emulation pybind11 module."""
    return [
        str(_CSRC / "legacy" / "python_function_export.cpp"),
        str(_CSRC / "legacy" / "mx" / "cpu" / "funcs.cpp"),
        str(_CSRC / "legacy" / "tqt" / "tqt_op.cpp"),
    ]


def legacy_kernel_sources_cuda() -> list[str]:
    """Legacy CUDA sources for the pre-stable-ABI hw_emulation pybind11 module.

    Some files (``mx/cuda/funcs.cu``, ``tqt/tqt.cu``) are shared with the
    stable-ABI tree; ``tqt/cu_utils.cc`` is legacy-only on the link side
    because the stable-ABI build inlines its symbols via the bindings TU.
    """
    return [
        str(_CSRC / "legacy" / "fake_tensor_cuda_hip.cu"),
        str(_CSRC / "mx" / "cuda" / "funcs.cu"),
        str(_CSRC / "tqt" / "tqt.cu"),
        str(_CSRC / "tqt" / "cu_utils.cc"),
        str(_CSRC / "legacy" / "mxfp4" / "dequantize.cu"),
        str(_CSRC / "legacy" / "mxfp4" / "fake.cu"),
    ]


def pyinit_stub_source() -> str:
    """Path to the stable-ABI ``pyinit_stub.cpp`` shim (Windows setuptools)."""
    return str(_PYINIT_STUB)


def torch_include_paths() -> list[str]:
    """Include dirs for hw_emulation sources.

    ``csrc/`` resolves ``mx/*`` / ``mxfp4/*`` prefixed quoted includes; sibling
    includes resolve via the source file's own dir without a ``-I``.
    ``<quark>/torch/include`` carries the shared stable_ops / device_guard /
    gpu_stream headers.
    """
    return [str(_CSRC), str(_TORCH_INCLUDE)]


# --- Per-artifact source builders. ---


def torch_ops_sources(*, use_cuda: bool) -> list[str]:
    """Stable-ABI ``hw_emulation`` source list (shared by AOT and JIT).

    CPU set always linked: ``torch_ops`` dispatches on runtime tensor device,
    so a CUDA build still routes CPU inputs to CPU kernels.
    """
    sources = [*stable_abi_kernel_sources_cpu(), pyinit_stub_source()]
    if use_cuda:
        sources.extend(stable_abi_kernel_sources_cuda())
    return sources


def legacy_hw_emulation_sources(*, use_cuda: bool) -> list[str]:
    """Legacy pybind11 ``hw_emulation`` source list (JIT-only).

    No AOT mirror: the pre-stable-ABI surface is built on-demand for older
    torch and for dual-variant tests; wheel ships only the stable-ABI ``_C``.
    """
    sources = [*legacy_kernel_sources_cpu()]
    if use_cuda:
        sources.extend(legacy_kernel_sources_cuda())
    return sources
