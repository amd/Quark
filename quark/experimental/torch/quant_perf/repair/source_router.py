#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from .evidence import extract_failure_evidence, is_missing_flydsl_backend_failure
from .types import RepairRequest, RepairTarget


def resolve_repair_target(request: RepairRequest) -> RepairTarget:
    if request.target_role == "kernel" and request.kernel_repo:
        return RepairTarget(
            role="kernel",
            repo=request.kernel_repo,
            reason="failure_diagnosis",
        )
    if request.target_role == "framework" and request.framework_repo:
        return RepairTarget(
            role="framework",
            repo=request.framework_repo,
            reason="failure_diagnosis",
        )
    evidence = extract_failure_evidence(request)
    error = evidence.full_error.lower()
    kernel_repo = request.kernel_repo
    if kernel_repo and is_missing_flydsl_backend_failure(error):
        return RepairTarget(
            role="kernel",
            repo=kernel_repo,
            reason="backend_capability",
        )

    root_file = evidence.root_file.lower()
    if evidence.exception_type and root_file:
        kernel_repo = request.kernel_repo
        if kernel_repo and (
            root_file.startswith(str(kernel_repo).lower().rstrip("/") + "/")
            or "/aiter/aiter/" in root_file
            or "/aiter/ops/" in root_file
        ):
            return RepairTarget(
                role="kernel",
                repo=kernel_repo,
                reason="root_file",
            )
        if root_file.startswith(str(request.framework_repo).lower().rstrip("/") + "/") or "/vllm/" in root_file:
            return RepairTarget(
                role="framework",
                repo=request.framework_repo,
                reason="root_file",
            )

    if kernel_repo:
        kernel_markers = (
            str(kernel_repo).lower(),
            "/aiter/",
            "aiter/ops/",
            "flydsl",
            "triton kernel",
        )
        if any(marker and marker in error for marker in kernel_markers):
            return RepairTarget(
                role="kernel",
                repo=kernel_repo,
                reason="error_path",
            )
    return RepairTarget(
        role="framework",
        repo=request.framework_repo,
        reason="default_framework",
    )
