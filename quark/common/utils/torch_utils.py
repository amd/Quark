#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""PyTorch capability probes shared across the Quark codebase.

This module is intentionally tiny and dependency-light so it can be imported
from both the build-time JIT machinery (``build_custom_ops``) and the runtime
op-resolution path (``base_fn_quantizers``) without pulling in heavier modules.
"""

import torch
from packaging import version


def is_torch_higher_or_equal(min_version: str) -> bool:
    """Return ``True`` when the running PyTorch's base release is ``>= min_version``.

    Compares ``.base_version`` (e.g. ``"2.10.0"`` from ``"2.10.0.dev20260101"``)
    so PEP 440 pre-release/dev/post segments don't gate below their final
    release — torch nightly and ``rcN`` builds carry the stable-ABI headers
    from the same branch as ``2.10.0``, so they must gate identically.

    Local-version suffixes (``+cu124``, ``+rocm7.1``) are stripped before
    parsing; any parse failure degrades to ``False``. Shared by the runtime
    capability probes and the pytest skip decorator
    (``require_torch_higher_or_equal`` in ``quark.common.utils.testing_utils``).
    """
    try:
        running_base = version.parse(torch.__version__.split("+", 1)[0]).base_version
        return version.parse(running_base) >= version.parse(min_version)
    except (version.InvalidVersion, AttributeError, TypeError):
        return False


def torch_supports_stable_abi() -> bool:
    """Return ``True`` when the running PyTorch has the ``torch::stable`` headers.

    The stable-ABI custom-ops surface (``torch.ops.quark_custom_ops``) is only
    registered on PyTorch >= 2.10. On older releases, the legacy pybind11
    modules are the only viable backend. This is a thin wrapper around
    :func:`is_torch_higher_or_equal` so both build-time JIT machinery
    (``build_custom_ops``) and runtime op-resolution paths
    (``base_fn_quantizers``) can share one capability probe with one version
    string to keep in sync.
    """
    return is_torch_higher_or_equal("2.10")
