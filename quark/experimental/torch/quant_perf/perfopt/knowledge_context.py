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
from quark.experimental.torch.quant_perf.session.spec import Spec
from quark.experimental.torch.quant_perf.session.state import SessionState

if TYPE_CHECKING:
    from quark.experimental.torch.quant_perf.knowledge.store import ExperienceStore


def build_kernel_knowledge(
    *,
    experience_store: ExperienceStore | None,
    state: SessionState,
    spec: Spec,
    bottleneck: dict[str, Any],
) -> tuple[KnowledgeBundle, str]:
    context = KnowledgeContext(
        domain="kernel_optimization",
        stage="optimize",
        model_arch=spec.model_arch,
        arch_fingerprint=spec.arch_fingerprint,
        framework=spec.framework,
        framework_version=spec.framework_version,
        framework_commit=spec.framework_version,
        kernel_commit=spec.kernel_version,
        gpu_type=spec.gpu_type,
        quant_signature=spec.quant_strategy or "",
        workload={
            "tp": spec.tp,
            "isl": spec.isl,
            "osl": spec.osl,
            "concurrency": spec.bench_concurrency,
        },
        kernel_context={
            **bottleneck,
            "bound_type": bottleneck.get("roofline_bound", ""),
        },
    )
    return query_and_render(
        router=build_knowledge_router(
            experience_store,
            state=state,
        ),
        context=context,
        session_dir=spec.session_dir,
        consumer="kernel_optimization",
        max_chars=6000,
    )
