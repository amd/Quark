#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

from .provider import KnowledgeProvider
from .types import KnowledgeBundle, KnowledgeContext, KnowledgeMatch


class KnowledgeRouter:
    def __init__(self, providers: list[KnowledgeProvider] | None = None):
        self.providers = list(providers or [])

    def _bundle(self, context: KnowledgeContext) -> KnowledgeBundle:
        by_id: dict[str, KnowledgeMatch] = {}
        for provider in self.providers:
            for match in provider.query(context):
                current = by_id.get(match.item.id)
                if current is None or match.applicability_score > current.applicability_score:
                    by_id[match.item.id] = match
        matches = sorted(
            by_id.values(),
            key=lambda match: (
                match.match_type == "exact",
                match.applicability_score,
                match.item.evidence_level,
                match.item.id,
            ),
            reverse=True,
        )
        bundle = KnowledgeBundle()
        for provider in self.providers:
            dead_ends = getattr(provider, "dead_ends", None)
            if callable(dead_ends):
                bundle.session_dead_ends.extend(dead_ends(context))
        for match in matches:
            if match.match_type == "exact":
                bundle.exact_matches.append(match)
            elif match.item.kind in {
                "diagnostic_playbook",
                "validated_playbook",
                "verified_repair_recipe",
                "policy",
            }:
                bundle.playbooks.append(match)
            else:
                bundle.priors.append(match)
            for check in match.item.required_checks:
                if check not in bundle.required_checks:
                    bundle.required_checks.append(check)
            bundle.source_ids.append(match.item.id)
        return bundle

    def query(self, context: KnowledgeContext) -> KnowledgeBundle:
        return self._bundle(context)
