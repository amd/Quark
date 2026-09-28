#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from typing import Any

from quark.experimental.torch.quant_perf.session.state import SessionState

from .types import KnowledgeContext, KnowledgeMatch


class SessionKnowledgeProvider:
    """Expose run-scoped rejected attempts without promoting them to history."""

    def __init__(self, state: SessionState) -> None:
        self.state = state

    def query(self, context: KnowledgeContext) -> list[KnowledgeMatch]:
        return []

    def dead_ends(self, context: KnowledgeContext) -> list[dict[str, Any]]:
        if context.domain == "repair":
            out: list[dict[str, Any]] = []
            for journey in self.state.get("repair_journey") or []:
                if context.failure_class and journey.get("failure_class") != context.failure_class:
                    continue
                for attempt in journey.get("attempts") or []:
                    if attempt.get("failed_because"):
                        out.append(
                            {
                                "tried": str(attempt.get("tried") or ""),
                                "failed_because": str(attempt.get("failed_because") or ""),
                            }
                        )
            return out
        if context.domain == "kernel_optimization":
            return [
                {
                    "kernel": str(row.get("name") or row.get("kernel_id") or ""),
                    "failed_because": str(row.get("skip_reason") or row.get("outcome") or ""),
                }
                for row in self.state.get("kernel_journey") or []
                if row.get("outcome") in {"rejected", "skipped"}
            ]
        return []
