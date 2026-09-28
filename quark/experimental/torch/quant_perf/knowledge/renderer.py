#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from .types import KnowledgeBundle, KnowledgeMatch


def _render_match(match: KnowledgeMatch) -> list[str]:
    item = match.item
    lines = [
        f"- [{item.id}] {item.summary}",
        f"  evidence={item.evidence_level}; match={match.match_type}",
    ]
    lines.extend(f"  guidance: {step}" for step in item.guidance)
    return lines


class KnowledgeRenderer:
    def __init__(self, max_chars: int):
        self.max_chars = max(0, int(max_chars))

    def render(self, bundle: KnowledgeBundle) -> str:
        if bundle.empty or self.max_chars == 0:
            return ""
        lines = ["## Retrieved knowledge (advisory; verifiers remain authoritative)"]
        sections = (
            ("Exact matches", bundle.exact_matches[:3]),
            ("Applicable playbooks", bundle.playbooks[:3]),
            ("Related priors", bundle.priors[:5]),
        )
        for title, matches in sections:
            if not matches:
                continue
            lines.append(f"### {title}")
            for match in matches:
                lines.extend(_render_match(match))
        if bundle.session_dead_ends:
            lines.append("### Previously rejected in this session")
            for row in bundle.session_dead_ends[:5]:
                lines.append(f"- {row}")
        if bundle.required_checks:
            lines.append("### Required independent checks")
            lines.extend(f"- {check}" for check in bundle.required_checks)
        return "\n".join(lines)[: self.max_chars]
