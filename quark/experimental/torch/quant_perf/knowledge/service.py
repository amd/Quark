#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import hashlib

from .audit import append_query_audit
from .renderer import KnowledgeRenderer
from .router import KnowledgeRouter
from .types import KnowledgeBundle, KnowledgeContext


def query_and_render(
    *,
    router: KnowledgeRouter,
    context: KnowledgeContext,
    session_dir: str,
    consumer: str,
    max_chars: int,
    round_id: int | None = None,
) -> tuple[KnowledgeBundle, str]:
    bundle = router.query(context)
    if session_dir:
        append_query_audit(
            session_dir,
            consumer=consumer,
            context_hash=hashlib.sha256(repr(context).encode()).hexdigest(),
            bundle=bundle,
            round_id=round_id,
            error_signature=context.error_signature,
        )
    return bundle, KnowledgeRenderer(max_chars=max_chars).render(bundle)
