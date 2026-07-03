#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""ONNX custom-ops source builders (section helpers + per-artifact lists).

torch >= 2.10: wheel and JIT both use :func:`onnx_ops_sources` for a symbol-identical ``_C``.
torch < 2.10: :func:`ort_lib_jit_sources` (legacy ORT lib, optional ``-DTORCH_OP`` pybind11).
"""

from __future__ import annotations

from pathlib import Path

_QUARK_ROOT = Path(__file__).resolve().parent.parent.parent
_BASE = _QUARK_ROOT / "onnx" / "operators" / "custom_ops"
_SRC = _BASE / "src"
_INC = _BASE / "include"
# pyinit_stub is shared across stable-ABI extensions; lives under common/csrc.
_PYINIT_STUB = _QUARK_ROOT / "common" / "csrc" / "pyinit_stub.cpp"


# --- Section helpers (composed by the per-artifact builders below). ---


def ort_glue_sources() -> list[str]:
    """ORT custom-op registration glue."""
    return [
        str(_SRC / "custom_op_library.cc"),
        str(_SRC / "custom_op_qdq.cc"),
        str(_SRC / "custom_op_in.cc"),
        str(_SRC / "custom_op_bfp.cc"),
        str(_SRC / "custom_op_mx.cc"),
        str(_SRC / "custom_op_lstm.cc"),
    ]


def shared_kernel_sources_cpu() -> list[str]:
    """BFP/MX CPU kernels (shared between ORT and stable-ABI torch surfaces)."""
    return [
        str(_SRC / "bfp" / "cpu" / "bfp_kernel.cc"),
        str(_SRC / "mx" / "cpu" / "mx_kernel.cc"),
    ]


def shared_kernel_sources_cuda() -> list[str]:
    """BFP/MX GPU kernels (``.cu``; torch hipifies on ROCm)."""
    return [
        str(_SRC / "bfp" / "cuda" / "bfp_kernel.cu"),
        str(_SRC / "mx" / "cuda" / "mx_kernel.cu"),
    ]


def ort_bfp_mx_wrappers_cpu() -> list[str]:
    """ORT-side CPU BFP/MX wrappers; same symbols as the CUDA set, so link only one."""
    return [
        str(_SRC / "bfp" / "cpu" / "bfp.cc"),
        str(_SRC / "mx" / "cpu" / "mx.cc"),
    ]


def ort_bfp_mx_wrappers_cuda() -> list[str]:
    """ORT-side CUDA BFP/MX wrappers + qdq quantize_linear."""
    return [
        str(_SRC / "qdq" / "cuda" / "quantize_linear.cu"),
        str(_SRC / "bfp" / "cuda" / "bfp.cc"),
        str(_SRC / "mx" / "cuda" / "mx.cc"),
    ]


def torch_ops_source() -> list[str]:
    """Stable-ABI ``torch_ops.cc``."""
    return [str(_SRC / "torch_ops.cc")]


def legacy_torch_ops_source() -> list[str]:
    """Pre-stable-ABI pybind11 ``legacy/torch_ops.cc``."""
    return [str(_SRC / "legacy" / "torch_ops.cc")]


def pyinit_stub_source() -> str:
    """Path to the stable-ABI ``pyinit_stub.cpp`` shim (Windows setuptools)."""
    return str(_PYINIT_STUB)


def stable_abi_include_paths() -> list[str]:
    """Include dirs for ONNX custom-op sources (``include/`` + ``src/``)."""
    return [str(_INC), str(_SRC)]


def ort_include_paths() -> list[str]:
    """ORT headers + GSL."""
    return [
        str(_INC / "onnxruntime-1.17.0" / "onnxruntime"),
        str(_INC / "onnxruntime-1.17.0" / "onnxruntime" / "core" / "session"),
        str(_INC / "gsl-4.0.0"),
    ]


# --- Per-artifact source builders. ---


def onnx_ops_sources(*, use_cuda: bool) -> list[str]:
    """``_C`` sources (torch >= 2.10): wheel and JIT share this list for symbol parity.

    Excludes legacy pybind11 ``src/legacy/torch_ops.cc`` (lives in
    :func:`ort_lib_jit_sources` and :func:`torch_legacy_jit_sources` only).
    Link exactly one CPU/CUDA bfp/mx wrapper set per build.
    """
    sources = [
        *ort_glue_sources(),
        *shared_kernel_sources_cpu(),
        *torch_ops_source(),
        pyinit_stub_source(),
    ]
    if use_cuda:
        sources.extend(shared_kernel_sources_cuda())
        sources.extend(ort_bfp_mx_wrappers_cuda())
    else:
        sources.extend(ort_bfp_mx_wrappers_cpu())
    return sources


def onnx_ort_cpu_sources() -> list[str]:
    """Sources for the torch-free CPU-EP ORT companion (``_C_cpu``).

    Excludes ``torch_ops.cc`` -- its ``STABLE_TORCH_LIBRARY`` would
    double-register the namespace against ``_C`` and crash when both load -- and
    all ``.cu`` sources (CPU-EP ``-DNO_GPU`` build). ``pyinit_stub`` satisfies
    the Windows ``/EXPORT:PyInit__C_cpu`` link.
    """
    return [
        *ort_glue_sources(),
        *ort_bfp_mx_wrappers_cpu(),
        *shared_kernel_sources_cpu(),
        pyinit_stub_source(),
    ]


def ort_lib_jit_sources(*, use_cuda: bool, include_legacy_torch_ops: bool = False) -> list[str]:
    """Legacy ORT lib JIT (``libcustom_ops{,_gpu}``); torch < 2.10 only.

    ``include_legacy_torch_ops``: inline pybind11 via ``-DTORCH_OP`` when stable-ABI
    is unavailable. Link exactly one CPU/CUDA bfp/mx wrapper set per build.
    """
    sources = [*ort_glue_sources()]
    if use_cuda:
        sources.extend(ort_bfp_mx_wrappers_cuda())
        sources.extend(shared_kernel_sources_cuda())
    else:
        sources.extend(ort_bfp_mx_wrappers_cpu())
        sources.extend(shared_kernel_sources_cpu())
    if include_legacy_torch_ops:
        sources.extend(legacy_torch_ops_source())
    return sources


def torch_legacy_jit_sources(*, use_cuda: bool) -> list[str]:
    """JIT source list for the test-only ``libcustom_ops_torch_legacy{,_gpu}``.

    Pre-stable-ABI pybind11 + raw kernels only; no ORT glue. Built on-demand
    by the test conftest on torch >= 2.10.
    """
    sources = [*legacy_torch_ops_source()]
    sources.extend(shared_kernel_sources_cuda() if use_cuda else shared_kernel_sources_cpu())
    return sources
