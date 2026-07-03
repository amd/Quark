#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for the JIT-compilation helpers in
:mod:`quark.torch.kernel.hw_emulation.extensions`.

These tests exercise the conditional branches that are otherwise only
reachable at *import* time (version gating, load-failure fallback, CUDA
vs ROCm compile-flag selection), so they exist purely to keep the
extension loader self-tested as the variant strategy evolves.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from quark.torch.kernel.hw_emulation import extensions


@pytest.mark.parametrize("helper_return", [True, False])
def test_jit_compile_stable_abi_propagates_helper_bool(helper_return):
    """Wrapper just pins hw_emulation sources/defines and forwards the helper's bool;
    version gating and ops-packet resolution live in the caller."""
    with mock.patch.object(extensions, "jit_compile_stable_abi_library", return_value=helper_return) as helper_mock:
        assert extensions._jit_compile_stable_abi() is helper_return
    helper_mock.assert_called_once()


def test_compile_torch_legacy_library_cuda_flags():
    """The CUDA runtime branch must append ``-O2`` and ``--extended-lambda``
    to ``extra_cuda_cflags`` (the ROCm branch is exercised at import time
    on ROCm CI)."""
    captured: dict[str, list[str]] = {}

    def _capture(name, _compile_dir, extra_cuda_cflags, extra_cflags):
        captured["extra_cuda_cflags"] = list(extra_cuda_cflags)
        return mock.sentinel.legacy_module

    with (
        mock.patch("torch.version.cuda", "12.8"),
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch.object(extensions, "compile_kernel_legacy", side_effect=_capture),
    ):
        result = extensions._compile_torch_legacy_library()

    assert result is mock.sentinel.legacy_module
    assert "-O2" in captured["extra_cuda_cflags"]
    assert "--extended-lambda" in captured["extra_cuda_cflags"]


def test_initialize_kernels_propagates_stable_abi_failure():
    """On torch >= 2.10, a stable-ABI load+JIT failure surfaces as a ``RuntimeError`` from
    the shared helper. Falling through to the legacy build there would mask the real
    packaging/compile regression behind a working-but-different code path; the helper
    itself owns the message so the torch and onnx loaders stay aligned (see
    ``test_for_common/test_torch_cpp_ext_helpers.py::test_load_or_jit_stable_abi_both_fail_raises``)."""
    with (
        mock.patch.object(extensions, "torch_supports_stable_abi", return_value=True),
        mock.patch.object(extensions, "load_or_jit_stable_abi", side_effect=RuntimeError("stable-abi miss")),
        mock.patch.object(extensions, "_compile_torch_legacy_library") as build_mock,
        pytest.raises(RuntimeError, match="stable-abi miss"),
    ):
        extensions._initialize_kernels()

    build_mock.assert_not_called()


def test_initialize_kernels_surfaces_stable_abi_namespace():
    """On stable-ABI hit, return the ops packet and forward the JIT callable to the
    orchestrator (pre-compiled-vs-JIT semantics covered in
    ``test_for_common/test_torch_cpp_ext_helpers.py``)."""
    with (
        mock.patch.object(extensions, "load_or_jit_stable_abi", return_value=None) as orchestrator,
        mock.patch.object(extensions, "_compile_torch_legacy_library") as legacy_mock,
        mock.patch.object(extensions, "torch_supports_stable_abi", return_value=True),
    ):
        kernel_ext = extensions._initialize_kernels()

    assert kernel_ext is torch.ops.quark_hw_emulation
    legacy_mock.assert_not_called()
    assert orchestrator.call_args.kwargs["jit_compile_fn"] is extensions._jit_compile_stable_abi


def test_initialize_kernels_skips_stable_abi_on_old_torch():
    """torch < 2.10 must bypass the stable-ABI orchestrator entirely and go straight
    to the legacy build (stable-ABI is meaningless without runtime support)."""
    with (
        mock.patch.object(extensions, "load_or_jit_stable_abi") as orchestrator,
        mock.patch.object(extensions, "_compile_torch_legacy_library", return_value=mock.sentinel.legacy),
        mock.patch.object(extensions, "torch_supports_stable_abi", return_value=False),
    ):
        kernel_ext = extensions._initialize_kernels()

    orchestrator.assert_not_called()
    assert kernel_ext is mock.sentinel.legacy
