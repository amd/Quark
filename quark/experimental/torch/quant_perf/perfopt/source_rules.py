#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Small, version-controlled operator-to-source rules for stable call seams."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceRule:
    kernel_pattern: str
    relative_source: str
    source_symbol: str
    repo_role: str = ""
    builder_symbol: str = ""
    build_module: str = ""
    relative_launcher: str = ""
    live_call_seam: str = ""
    compiler: str = ""
    parent_op_pattern: str = ""
    context_pattern: str = ""
    label: str = ""

    def matches(
        self,
        kernel_names: list[str],
        parent_ops: list[str],
        context_tags: tuple[str, ...] = (),
    ) -> bool:
        if not any(re.search(self.kernel_pattern, name) for name in kernel_names):
            return False
        if not self.parent_op_pattern and not self.context_pattern:
            return True
        parent_match = bool(self.parent_op_pattern) and any(
            re.search(self.parent_op_pattern, parent) for parent in parent_ops
        )
        context_match = bool(self.context_pattern) and any(re.search(self.context_pattern, tag) for tag in context_tags)
        return parent_match or context_match


SOURCE_RULES: tuple[SourceRule, ...] = (
    SourceRule(
        kernel_pattern=r"dynamic_per_group_scaled_quant_kernel",
        relative_source="csrc/kernels/quant_kernels.cu",
        source_symbol="dynamic_per_group_scaled_quant_kernel",
        repo_role="kernel",
        builder_symbol="dynamic_per_group_scaled_quant",
        build_module="module_quant",
        relative_launcher="aiter/ops/quant.py",
        live_call_seam=("aiter::dynamic_per_group_scaled_quant via module_quant"),
        compiler="hip_cpp",
        label="AITER dynamic per-group scaled quant",
    ),
    SourceRule(
        parent_op_pattern=(
            r"^vllm::rocm_aiter_flydsl_gemm_"
            r"a(?:4w4_dynamic|8w4_per_tensor)$"
        ),
        context_pattern=r"^flydsl_dense_mxfp4$",
        kernel_pattern=r"^kernel_gemm_\d+(?:\.kd)?$",
        relative_source=("aiter/ops/flydsl/kernels/mxfp4_preshuffle.py"),
        source_symbol="kernel_gemm",
        repo_role="kernel",
        builder_symbol="launch_gemm",
        relative_launcher="aiter/ops/flydsl/batched_gemm_mxfp4.py",
        live_call_seam="vLLM dense MXFP4 FlyDSL custom ops",
        compiler="flydsl",
        label="FlyDSL dense MXFP4 preshuffle GEMM",
    ),
    SourceRule(
        parent_op_pattern=r"^aiter::fused_moe_$",
        kernel_pattern=r"^moe_reduction_kernel_(?:plain|masked)_",
        relative_source=("aiter/ops/flydsl/kernels/moe_gemm_2stage.py"),
        source_symbol="moe_reduction_kernel",
        repo_role="kernel",
        builder_symbol="compile_moe_reduction",
        relative_launcher="aiter/ops/flydsl/moe_kernels.py",
        live_call_seam="aiter::fused_moe_",
        compiler="flydsl",
        label="FlyDSL MoE reduction",
    ),
    SourceRule(
        kernel_pattern=(r"(?:flydsl_moe[12]_|moe_flydsl_|mfma_moe[12]_)"),
        relative_source=("aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py"),
        source_symbol="",
        repo_role="kernel",
        builder_symbol="compile_mixed_moe_gemm",
        relative_launcher="aiter/ops/flydsl/moe_kernels.py",
        live_call_seam="aiter::fused_moe_",
        compiler="flydsl",
        label="FlyDSL mixed MoE GEMM",
    ),
)
