#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Landing: dispatch to the selected framework adapter.

Repair policy belongs to the Orchestrator/RepairService boundary; this module
only attempts the requested load and surfaces a StageError on failure.
"""

from __future__ import annotations

from quark.experimental.torch.quant_perf.session.spec import ServerHandle, Spec, StageError


def _serve(quant_ckpt_dir: str, spec: Spec, profiler_dir: str | None = None) -> ServerHandle:
    if spec.framework == "atom":
        from quark.experimental.torch.quant_perf.landing import atom_adapter

        return atom_adapter.serve(quant_ckpt_dir, spec, profiler_dir=profiler_dir)
    if spec.framework == "vllm":
        from quark.experimental.torch.quant_perf.landing import vllm_adapter

        return vllm_adapter.serve(quant_ckpt_dir, spec, profiler_dir=profiler_dir)
    raise StageError("land", f"framework {spec.framework!r} is not implemented yet (only 'atom'/'vllm' in V1)")


def load(quant_ckpt_dir: str, spec: Spec, profiler_dir: str | None = None) -> ServerHandle:
    """Load the model once; callers own repair and retry decisions."""
    return _serve(quant_ckpt_dir, spec, profiler_dir=profiler_dir)
