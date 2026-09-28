#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from typing import TYPE_CHECKING, Any

from quark.experimental.torch.quant_perf.knowledge import (
    KnowledgeBundle,
    KnowledgeContext,
    build_knowledge_router,
    query_and_render,
)
from quark.experimental.torch.quant_perf.session.state import SessionState

if TYPE_CHECKING:
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore


def build_repair_knowledge(
    *,
    experience_store: ExperienceStore | None,
    state: SessionState | None = None,
    session_dir: str,
    consumer: str,
    stage: str,
    framework: str,
    framework_version: str,
    arch_fingerprint: str,
    quant_signature: str,
    failure_class: str,
    error_signature: str,
    workload: dict[str, Any],
    error_text: str = "",
    round_id: int | None = None,
) -> tuple[KnowledgeBundle, str]:
    context = KnowledgeContext(
        domain="repair",
        stage=stage,
        arch_fingerprint=arch_fingerprint,
        framework=framework,
        framework_version=framework_version,
        quant_signature=quant_signature,
        failure_class=failure_class,
        error_signature=error_signature,
        error_text=error_text,
        workload=workload,
    )
    return query_and_render(
        router=build_knowledge_router(
            experience_store,
            state=state,
        ),
        context=context,
        session_dir=session_dir,
        consumer=consumer,
        max_chars=8000,
        round_id=round_id,
    )
