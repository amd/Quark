#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Deterministic guardrail before spending a GEAK optimization round.

llm/client.py's `_Throttle` already covers the third guardrail (LLM call-rate
skeleton: severity gate + per-round cap + per-key cooldown + fail-open) --
not duplicated here.
"""

from __future__ import annotations

import logging
import re

from quark.experimental.torch.quant_perf.perfopt.kernel_source import _demangle

logger = logging.getLogger(__name__)


def geak_execution_skip_reason(kernel_name: str, source_symbol: str = "") -> str:
    """Check device function names against GEAK's single-GPU harness."""
    # Template arguments and parent operators do not establish a collective.
    # The source resolver's best-effort demangler leaves unknown encodings as _Z...
    for symbol in (kernel_name, source_symbol):
        name = _demangle(symbol)
        if not name.startswith("_Z") and re.search(r"all_?reduce|reduce_scatter|all_?gather|all_?to_?all", name, re.I):
            return (
                f"Collective kernel {kernel_name} requires a multi-GPU harness; "
                "GEAK kernel_workflow supports one GPU per lane."
            )
    return ""


def amdahl_ceiling(p: float, s: float) -> float:
    """Amdahl's law: the end-to-end speedup ceiling if the optimizable
    fraction `p` of the program were sped up by factor `s`. `1/((1-p)+p/s)`."""
    if s <= 0:
        return 1.0
    return 1.0 / ((1.0 - p) + p / s)


def should_attempt_geak(
    kernel_time_us: float,
    total_time_us: float,
    assumed_kernel_speedup: float = 1.3,
    min_e2e_gain: float = 0.001,
) -> bool:
    """Preflight veto (IMPL_SPEC §3): before spending a GEAK round on this
    kernel, check whether even a generously-assumed per-kernel speedup could
    move the end-to-end needle at all. `p` is this kernel's share of the
    profiled window (a proxy for "the optimizable fraction"); if the Amdahl
    ceiling implies less than `min_e2e_gain` end-to-end improvement, GEAK is
    not worth running on this kernel."""
    if total_time_us <= 0:
        return True  # fail-open: no way to judge, don't veto
    p = min(kernel_time_us / total_time_us, 1.0)
    ceiling = amdahl_ceiling(p, assumed_kernel_speedup)
    return (ceiling - 1.0) >= min_e2e_gain
