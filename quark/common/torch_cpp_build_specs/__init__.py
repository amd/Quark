#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Build specs for Quark's torch C++ extensions.

The submodules under this package are the *single source of truth* for what
goes into each torch C++ extension Quark ships (``torch_ops`` for the
``hw_emulation`` artifact, ``onnx_ops`` for the ONNX custom-ops artifacts).
Both the wheel-time AOT build (``setup.py``) and the JIT fallback paths
(``quark/torch/kernel/hw_emulation/extensions.py``,
``quark/onnx/operators/custom_ops/build_custom_ops.py``) consume the same
helpers here so flags / source sets / include paths can't drift between
the two surfaces.

Policy for callers
==================
JIT and AOT call sites must NOT inline source paths, include paths, or
compose source lists out of section helpers. Each artifact has exactly one
``*_sources(*, use_cuda: bool)`` builder and one ``*_include_paths()``
helper in the per-target submodule; call sites use those and nothing
else. Section helpers (``ort_glue_sources``, ``shared_kernel_sources_*``,
``legacy_kernel_sources_*``, …) exist only so the per-artifact builders
can be expressed without duplication; they are not part of the call-site
contract.

Build-time constants (artifact names, compiler defines) live in
:mod:`quark.common.torch_cpp_build_specs.constants` and are re-exported
from this package root for back-compat.

Lives in ``quark.common`` so ``setup.py`` can import without triggering
``quark.torch.__init__`` / ``quark.onnx.__init__`` (torch + transformers + …).
"""

from quark.common.torch_cpp_build_specs.constants import (
    HW_EMULATION_C_MODULE,
    ONNX_CUSTOM_OPS_C_MODULE,
    ONNX_CUSTOM_OPS_CPU_C_MODULE,
    ONNX_CUSTOM_OPS_JIT_BASENAME,
    ORT_WINDOWS_DEFINE,
    TORCH_TARGET_VERSION_DEFINE,
)

__all__ = [
    "HW_EMULATION_C_MODULE",
    "ONNX_CUSTOM_OPS_C_MODULE",
    "ONNX_CUSTOM_OPS_CPU_C_MODULE",
    "ONNX_CUSTOM_OPS_JIT_BASENAME",
    "ORT_WINDOWS_DEFINE",
    "TORCH_TARGET_VERSION_DEFINE",
]
