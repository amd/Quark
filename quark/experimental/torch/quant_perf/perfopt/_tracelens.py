#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Optional TraceLens integration for PerfOpt."""

from __future__ import annotations

import importlib.util
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TraceLensBackend:
    tree_perf_analyzer: Any
    event_replayer: Any
    resolve_gpu_arch: Callable[..., Any]
    get_pseudo_op_mappings: Callable[[], dict[Any, Any]] | None


def is_tracelens_available() -> bool:
    try:
        return importlib.util.find_spec("TraceLens") is not None
    except (ImportError, ValueError):
        return False


def load_tracelens() -> TraceLensBackend | None:
    if not is_tracelens_available():
        return None

    try:
        from TraceLens import TreePerfAnalyzer
        from TraceLens.EventReplay.event_replay import EventReplayer
        from TraceLens.Reporting.reporting_utils import resolve_gpu_arch
    except ImportError:
        return None

    try:
        from TraceLens.PerfModel.extensions.pseudo_ops_perf_utils import get_pseudo_op_mappings
    except Exception as exc:
        logger.warning("TraceLens pseudo-op extension load failed (%s); continuing without it", exc)
        get_pseudo_op_mappings = None

    return TraceLensBackend(
        tree_perf_analyzer=TreePerfAnalyzer,
        event_replayer=EventReplayer,
        resolve_gpu_arch=resolve_gpu_arch,
        get_pseudo_op_mappings=get_pseudo_op_mappings,
    )
