#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Build-time constants for Quark's torch C++ extensions.

Public surface consumed by ``setup.py`` (AOT) and the JIT loaders. Grouped
in one file matching the in-tree convention (``quark/torch/*/constants.py``)
so a future project-wide constants module is a single-file migration.

These are build-time strings (artifact names, compiler defines); they are
*not* user-tunable. Runtime feature flags live in ``quark/torch/utils/constants.py``.
"""

# Shared by AOT and JIT so the stable-ABI floor can't drift between paths.
TORCH_TARGET_VERSION_DEFINE = "-DTORCH_TARGET_VERSION=TORCH_VERSION_2_10_0"

# Identifiers for the stable-ABI ``_C`` artifacts (torch >= 2.10):
#   ``*_C_MODULE``       - dotted import path of the pre-compiled wheel extension;
#                          read by the runtime loader before falling through to JIT.
#   ``*_JIT_BASENAME``   - bare name used as the JIT cache subdir and ``.so`` basename;
#                          consumed by both ``torch.ops.load_library`` and ORT
#                          ``register_custom_ops_library``. Legacy ``custom_ops{,_gpu}``
#                          remains the torch < 2.10 ONNX surface.
HW_EMULATION_C_MODULE = "quark.torch.kernel.hw_emulation._C"
ONNX_CUSTOM_OPS_C_MODULE = "quark.onnx.operators.custom_ops._C"
ONNX_CUSTOM_OPS_JIT_BASENAME = "quark_custom_ops_stable_abi"

ONNX_CUSTOM_OPS_CPU_C_MODULE = "quark.onnx.operators.custom_ops._C_cpu"

# Windows ``dllimport`` for the ORT C-API; shared by every artifact that
# links ORT headers (legacy ORT lib + ONNX ``_C``).
ORT_WINDOWS_DEFINE = "-DORT_DLL_IMPORT"
