#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Typed, advisory knowledge retrieval for Quark Quant-Perf LLM call sites."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from quark.experimental.torch.quant_perf.session.state import SessionState

from .experience import ExperienceKnowledgeProvider
from .provider import CuratedKnowledgeProvider, KnowledgeProvider
from .renderer import KnowledgeRenderer
from .router import KnowledgeRouter
from .service import query_and_render
from .session import SessionKnowledgeProvider
from .types import (
    KnowledgeBundle,
    KnowledgeContext,
    KnowledgeMatch,
    KnowledgeRecord,
)

if TYPE_CHECKING:
    from .store import ExperienceStore


def build_knowledge_router(
    experience_store: ExperienceStore | None = None,
    state: SessionState | None = None,
) -> KnowledgeRouter:
    """Build a router from the available knowledge providers.

    :param experience_store: Optional cross-session experience store.
    :param state: Optional current-session state.
    :return: Configured knowledge router.
    """
    providers: list[KnowledgeProvider] = [CuratedKnowledgeProvider(Path(__file__).resolve().parent / "records")]
    if experience_store is not None:
        providers.append(ExperienceKnowledgeProvider(experience_store))
    if state is not None:
        providers.append(SessionKnowledgeProvider(state))
    return KnowledgeRouter(providers)


__all__ = [
    "KnowledgeBundle",
    "KnowledgeContext",
    "KnowledgeMatch",
    "KnowledgeRecord",
    "KnowledgeRenderer",
    "KnowledgeRouter",
    "CuratedKnowledgeProvider",
    "build_knowledge_router",
    "query_and_render",
]
