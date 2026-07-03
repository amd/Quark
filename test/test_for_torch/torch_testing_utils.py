#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Torch-side wrapper around :func:`quark.common.utils.testing_utils.run_op_variants`,
mirroring ``test/test_for_onnx/testing_utils.py``.
"""

from __future__ import annotations

import functools
import sys
import warnings
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from typing import Any

from quark.common.utils.testing_utils import run_op_variants as _run_op_variants
from quark.common.utils.torch_utils import torch_supports_stable_abi
from quark.torch.kernel.hw_emulation import extensions


def _kernel_ext_ctx(target: Any) -> AbstractContextManager[None] | None:
    """Context manager that swaps ``kernel_ext`` to ``target`` on every
    consumer module in ``sys.modules`` (identity scan) and restores on exit.
    ``None`` target short-circuits to ``None`` so callers can pass through
    a missing variant without a separate guard.
    """
    if target is None:
        return None

    @contextmanager
    def _ctx() -> Iterator[None]:
        current = extensions.kernel_ext
        swapped = [
            mod for mod in list(sys.modules.values()) if mod is not None and getattr(mod, "kernel_ext", None) is current
        ]
        for mod in swapped:
            mod.kernel_ext = target
        try:
            yield
        finally:
            for mod in swapped:
                mod.kernel_ext = current

    return _ctx()


def get_stable_torch_ctx() -> AbstractContextManager[None] | None:
    """Swap-to-stable-ABI context, or ``None`` on torch < 2.10.

    A stable-ABI compile miss that fell through to legacy makes the runner degenerate
    to a no-op legacy-vs-legacy assertion — wasted compute, never a false signal.
    """
    if not torch_supports_stable_abi():
        return None
    return _kernel_ext_ctx(extensions.kernel_ext)


@functools.cache
def _get_legacy_torch_kernel() -> Any:
    """Memoised legacy pybind11 module: reused from ``extensions`` on torch < 2.10,
    lazy-JIT-built on torch >= 2.10. ``None`` (one-shot warning) on build failure so
    dual-variant tests degrade to stable-ABI-only instead of crashing.
    """
    if not torch_supports_stable_abi():
        return extensions.kernel_ext
    kernel = extensions._compile_torch_legacy_library()
    if kernel is None:
        warnings.warn(
            "Legacy hw_emulation kernel JIT build failed on torch >= 2.10; "
            "dual-variant tests will run stable-ABI only without bit-exact "
            "equivalence assertion. See logs for the underlying compile error.",
            RuntimeWarning,
            stacklevel=2,
        )
    return kernel


def get_legacy_torch_ctx() -> AbstractContextManager[None] | None:
    """Swap-to-legacy context, or ``None`` if the module isn't loadable.
    Build is lazy on torch >= 2.10 via :func:`_get_legacy_torch_kernel`;
    """
    return _kernel_ext_ctx(_get_legacy_torch_kernel())


def run_torch_op_variants(pipeline_fn: Any, *, seed: int = 42) -> Any:
    """Run ``pipeline_fn`` under each loaded variant and assert bit-exact
    equivalence; see :func:`quark.common.utils.testing_utils.run_op_variants`.
    A failed lazy legacy build on torch >= 2.10 degrades to a stable-ABI-only
    run (no equivalence assertion) rather than failing the test.
    """
    return _run_op_variants(
        pipeline_fn,
        legacy_ctx=get_legacy_torch_ctx(),
        stable_ctx=get_stable_torch_ctx(),
        seed=seed,
    )
