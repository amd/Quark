#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Pins each artifact's composition (which section helpers it must include) so wheel/JIT can't drift.

Contracts are expressed in terms of the section helpers themselves, never as
literal path-suffix strings, so file moves in ``quark.common.torch_cpp_build_specs``
ripple through these tests automatically without a parallel hardcoded mirror.
On-disk path existence is left to ``tools/ci/test_kernel_build.sh`` and the
JIT-exercising unit tests, both of which compile every artifact this PR ships.
"""

import pytest

from quark.common.torch_cpp_build_specs import onnx_ops, torch_ops


def test_torch_legacy_and_stable_abi_kernel_sources_disjoint_on_cpu():
    # Misclassifying a path between the two sets would silently duplicate or
    # drop symbols in the legacy JIT artifact.
    assert set(torch_ops.stable_abi_kernel_sources_cpu()).isdisjoint(set(torch_ops.legacy_kernel_sources_cpu()))


@pytest.mark.parametrize(
    "cpu_section,cuda_section",
    [
        pytest.param(onnx_ops.shared_kernel_sources_cpu, onnx_ops.shared_kernel_sources_cuda, id="shared_kernels"),
        pytest.param(onnx_ops.ort_bfp_mx_wrappers_cpu, onnx_ops.ort_bfp_mx_wrappers_cuda, id="ort_bfp_mx_wrappers"),
    ],
)
def test_cpu_cuda_section_pairs_are_disjoint(cpu_section, cuda_section):
    # CPU and CUDA variants of these sections export the same symbols, so
    # any artifact must pick exactly one flavour — the per-artifact
    # composition tests below pin which flavour each picks.
    assert set(cpu_section()).isdisjoint(set(cuda_section()))


# Per-artifact: (builder, expected CPU section composition, expected CUDA section composition).
# Each section entry is a callable returning either a ``list[str]`` or a ``str``;
# the builder's output is asserted equal to the union of its sections, so a
# refactor that adds/drops/reclassifies a section trips the matching artifact row.
_ARTIFACT_COMPOSITIONS = [
    pytest.param(
        torch_ops.torch_ops_sources,
        [torch_ops.stable_abi_kernel_sources_cpu, torch_ops.pyinit_stub_source],
        [
            torch_ops.stable_abi_kernel_sources_cpu,
            torch_ops.pyinit_stub_source,
            torch_ops.stable_abi_kernel_sources_cuda,
        ],
        id="torch_ops",
    ),
    pytest.param(
        torch_ops.legacy_hw_emulation_sources,
        [torch_ops.legacy_kernel_sources_cpu],
        [torch_ops.legacy_kernel_sources_cpu, torch_ops.legacy_kernel_sources_cuda],
        id="legacy_hw_emulation",
    ),
    pytest.param(
        onnx_ops.onnx_ops_sources,
        [
            onnx_ops.ort_glue_sources,
            onnx_ops.shared_kernel_sources_cpu,
            onnx_ops.torch_ops_source,
            onnx_ops.pyinit_stub_source,
            onnx_ops.ort_bfp_mx_wrappers_cpu,
        ],
        [
            onnx_ops.ort_glue_sources,
            onnx_ops.shared_kernel_sources_cpu,
            onnx_ops.torch_ops_source,
            onnx_ops.pyinit_stub_source,
            onnx_ops.shared_kernel_sources_cuda,
            onnx_ops.ort_bfp_mx_wrappers_cuda,
        ],
        id="onnx_ops",
    ),
    pytest.param(
        onnx_ops.ort_lib_jit_sources,
        [onnx_ops.ort_glue_sources, onnx_ops.ort_bfp_mx_wrappers_cpu, onnx_ops.shared_kernel_sources_cpu],
        [onnx_ops.ort_glue_sources, onnx_ops.ort_bfp_mx_wrappers_cuda, onnx_ops.shared_kernel_sources_cuda],
        id="ort_lib_jit",
    ),
    pytest.param(
        onnx_ops.torch_legacy_jit_sources,
        [onnx_ops.legacy_torch_ops_source, onnx_ops.shared_kernel_sources_cpu],
        [onnx_ops.legacy_torch_ops_source, onnx_ops.shared_kernel_sources_cuda],
        id="torch_legacy_jit",
    ),
]


def _union(sections):
    out: set[str] = set()
    for fn in sections:
        result = fn()
        out.update([result] if isinstance(result, str) else result)
    return out


@pytest.mark.parametrize("sources_fn,cpu_sections,cuda_sections", _ARTIFACT_COMPOSITIONS)
def test_artifact_composition(sources_fn, cpu_sections, cuda_sections):
    assert set(sources_fn(use_cuda=False)) == _union(cpu_sections)
    assert set(sources_fn(use_cuda=True)) == _union(cuda_sections)


def test_ort_lib_jit_include_legacy_torch_ops_toggle():
    legacy = set(onnx_ops.legacy_torch_ops_source())
    plain = set(onnx_ops.ort_lib_jit_sources(use_cuda=False))
    with_torch_op = set(onnx_ops.ort_lib_jit_sources(use_cuda=False, include_legacy_torch_ops=True))
    assert legacy.isdisjoint(plain)
    assert legacy <= with_torch_op


def test_onnx_ort_cpu_sources_composition():
    sources = set(onnx_ops.onnx_ort_cpu_sources())
    assert sources == _union(
        [
            onnx_ops.ort_glue_sources,
            onnx_ops.ort_bfp_mx_wrappers_cpu,
            onnx_ops.shared_kernel_sources_cpu,
            onnx_ops.pyinit_stub_source,
        ]
    )
    # torch_ops.cc and .cu sources must stay out: torch_ops.cc would
    # double-register the STABLE_TORCH_LIBRARY namespace against _C, and the
    # CPU-EP build is -DNO_GPU.
    assert set(onnx_ops.torch_ops_source()).isdisjoint(sources)
    assert not any(src.endswith(".cu") for src in sources)
