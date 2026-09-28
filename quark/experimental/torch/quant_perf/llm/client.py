#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""The sole entry point for direct single-turn LLM decision calls.
Agentic repair and GEAK subprocesses do not go through this client.

Secrets are never read directly here -- config.resolve_api_key()/
resolve_user()/base_url() are the single entry point (project convention).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from quark.experimental.torch.quant_perf import config

logger = logging.getLogger(__name__)

_client: Any | None = None
_client_identity: tuple[str, str, str] | None = None


def _get_client() -> Any:
    """Return a cached gateway client, refreshing it when its identity changes."""
    global _client, _client_identity
    identity = (config.base_url(), config.resolve_api_key(), config.resolve_user())
    if _client is None or _client_identity != identity:
        import anthropic

        _client = anthropic.Anthropic(
            api_key="dummy",  # the AMD gateway authenticates via the subscription header below
            base_url=identity[0],
            default_headers={
                "Ocp-Apim-Subscription-Key": identity[1],
                "user": identity[2],
                "anthropic-version": "2023-10-16",
            },
        )
        _client_identity = identity
    return _client


@dataclass
class _Throttle:
    """Per-tag rate limit, ported from Hyperloom's rca_engine.py pattern
    (concept only, reimplemented -- no Hyperloom runtime dependency)."""

    cooldown_s: float = 60.0
    max_per_round: int = 3
    _last: dict[str, float] = field(default_factory=dict)
    _counts: dict[tuple[str, int], int] = field(default_factory=dict)

    def allow(self, tag: str, round_id: int) -> bool:
        if self._counts.get((tag, round_id), 0) >= self.max_per_round:
            return False
        if time.time() - self._last.get(tag, 0) < self.cooldown_s:
            return False
        return True

    def record(self, tag: str, round_id: int) -> None:
        self._last[tag] = time.time()
        key = (tag, round_id)
        self._counts[key] = self._counts.get(key, 0) + 1


_throttle = _Throttle()


def direct_api_call(
    *,
    tag: str,
    system: str,
    user: str,
    model: str | None = None,
    max_tokens: int = 512,
    round_id: int = 0,
    default: str = "",
) -> str:
    """A single-turn (C) direct API call. Fail-open on any error (throttled,
    network, auth, malformed response) -- callers must treat `default` as a
    normal, expected outcome to fall back on, not an exception to catch."""
    if not _throttle.allow(tag, round_id):
        logger.debug("[%s] throttled", tag)
        return default
    try:
        client = _get_client()
        resp = client.messages.create(
            model=model or config.decision_model(),
            max_tokens=max_tokens,
            system=[{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user}],
        )
        _throttle.record(tag, round_id)
        return resp.content[0].text if resp.content else default
    except Exception as e:
        logger.warning("[%s] LLM call failed: %s", tag, e)
        return default
