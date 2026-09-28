#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class FailureEvidence:
    exception_type: str
    exception_message: str
    root_file: str
    root_function: str
    root_line: int
    root_source: str
    full_error: str
    signature: str
    evidence_paths: tuple[str, ...] = ()


RepairKnowledgeQuery = Callable[[int, FailureEvidence], tuple[str, list[str]]]


@dataclass(frozen=True)
class VerificationResult:
    verifier: str
    passed: bool
    metrics: dict[str, Any] = field(default_factory=dict)
    evidence_paths: tuple[str, ...] = ()
    failure: str = ""
    evidence: FailureEvidence | None = None


@dataclass(frozen=True)
class RepairTarget:
    role: str
    repo: str
    reason: str


@dataclass
class RepairRequest:
    failure_class: str
    error: str
    model_dir: str
    quant_ckpt_dir: str
    framework: str
    framework_repo: str
    kernel_repo: str
    stack_fingerprint: dict[str, Any]
    quant_signature: str
    workload: dict[str, Any]
    immutable_constraints: dict[str, Any]
    verifier_profile: str
    failure_code: str = ""
    target_role: str = ""
    evidence_signature: str = ""
    session_dir: str = ""
    metrics: dict[str, Any] = field(default_factory=dict)
    verifier: Callable[[], tuple[bool, str]] | None = None
    evidence: FailureEvidence | None = None

    def __post_init__(self) -> None:
        from .evidence import extract_failure_evidence

        if self.evidence is None:
            self.evidence = extract_failure_evidence(self.error)


@dataclass
class RepairResult:
    status: str
    target_repos: list[str] = field(default_factory=list)
    attempts: list[dict[str, Any]] = field(default_factory=list)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    verifier_results: list[VerificationResult] = field(default_factory=list)
    knowledge_ids: list[str] = field(default_factory=list)
