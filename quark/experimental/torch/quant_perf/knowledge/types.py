#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class KnowledgeContext:
    domain: str
    stage: str = ""
    model_arch: str = ""
    arch_fingerprint: str = ""
    framework: str = ""
    framework_version: str = ""
    framework_commit: str = ""
    kernel_commit: str = ""
    quark_version: str = ""
    rocm_version: str = ""
    gpu_type: str = ""
    quant_signature: str = ""
    failure_class: str = ""
    error_signature: str = ""
    error_text: str = ""
    workload: dict[str, Any] = field(default_factory=dict)
    kernel_context: dict[str, Any] = field(default_factory=dict)

    def for_domain(self, domain: str) -> KnowledgeContext:
        return replace(self, domain=domain)

    def value(self, key: str) -> Any:
        if hasattr(self, key):
            return getattr(self, key)
        if key in self.workload:
            return self.workload[key]
        return self.kernel_context.get(key)


@dataclass(frozen=True)
class KnowledgeRecord:
    id: str
    domain: str
    kind: str
    status: str
    applicability: dict[str, Any]
    match: dict[str, Any]
    summary: str
    guidance: tuple[str, ...]
    required_checks: tuple[str, ...]
    evidence: dict[str, Any]
    provenance: dict[str, Any]

    @property
    def symptoms(self) -> tuple[str, ...]:
        value = self.match.get("symptoms") or self.match.get("error_contains")
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in (value or ()))

    @property
    def keywords(self) -> tuple[str, ...]:
        value = self.match.get("keywords")
        if isinstance(value, str):
            return (value,)
        return tuple(str(item) for item in (value or ()))

    @property
    def evidence_level(self) -> str:
        return str(self.evidence.get("level") or "E1")

    @property
    def source(self) -> dict[str, str]:
        return {str(key): str(value) for key, value in self.provenance.items()}


@dataclass(frozen=True)
class KnowledgeMatch:
    item: KnowledgeRecord
    match_type: str
    applicability_score: float
    stack_distance: int = 0
    workload_distance: int = 0

    @property
    def record(self) -> KnowledgeRecord:
        return self.item


@dataclass
class KnowledgeBundle:
    exact_matches: list[KnowledgeMatch] = field(default_factory=list)
    playbooks: list[KnowledgeMatch] = field(default_factory=list)
    priors: list[KnowledgeMatch] = field(default_factory=list)
    session_dead_ends: list[dict[str, Any]] = field(default_factory=list)
    required_checks: list[str] = field(default_factory=list)
    source_ids: list[str] = field(default_factory=list)

    @property
    def empty(self) -> bool:
        return not (self.exact_matches or self.playbooks or self.priors or self.session_dead_ends)
