#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Extraction-engine watchdog.

Long streaming runs can hit a dead vLLM extraction engine (``EngineDeadError``).
This helper wraps a training launch so that, on such a failure, the run
auto-resumes from the last checkpoint (with reduced extraction memory headroom)
instead of losing the whole job.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from quark.experimental.speculative_decoding.utils.logging import get_logger

logger = get_logger(__name__)

_ENGINE_DEAD_MARKERS = ("EngineDeadError", "EngineCore", "engine core", "NCCL", "RCCL", "HIP error")

T = TypeVar("T")


def run_with_watchdog(launch_fn: Callable[[], T], max_restarts: int = 3, backoff_s: float = 30.0) -> T:
    """Call ``launch_fn`` and auto-restart on an extraction-engine death.

    ``launch_fn`` must itself resume from the latest checkpoint (the trainer does
    this when ``watchdog=True``). :func:`..run.run_from_config` wraps the training
    launch with this when the recipe sets ``training.watchdog``.
    """
    attempt = 0
    while True:
        try:
            return launch_fn()
        except Exception as e:  # noqa: BLE001 - watchdog intentionally broad
            msg = str(e)
            is_engine_death = any(m in msg for m in _ENGINE_DEAD_MARKERS)
            attempt += 1
            if not is_engine_death or attempt > max_restarts:
                logger.error("watchdog: giving up after %d attempt(s): %s", attempt, msg)
                raise
            logger.warning(
                "watchdog: extraction engine died (%s); restart %d/%d in %.0fs", msg, attempt, max_restarts, backoff_s
            )
            time.sleep(backoff_s)
