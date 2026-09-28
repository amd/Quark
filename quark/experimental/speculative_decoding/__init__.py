#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
AMD Quark - Speculative Decoding (EAGLE-3)
==========================================

An end-to-end, AMD/ROCm-native pipeline to train EAGLE-3 draft models from
scratch, synthesize on-policy training data, export to Hugging Face format,
deploy on vLLM, and evaluate acceptance length + per-GPU throughput.

The public surface mirrors the design document and is intentionally small::

    import quark.experimental.speculative_decoding as qsd

    spec_model = qsd.convert(target, spec_cfg={...})   # target -> EAGLE-3 spec model
    qsd.train(spec_model, data_cfg=..., train_cfg=...)  # cold-start training
    qsd.export_hf(spec_model, "release/draft_hf")       # vLLM-loadable EAGLE-3 draft

Status
------
``export_hf`` emits a draft in vLLM's ``LlamaForCausalLMEagle3`` format
(architecture + weight key names + shapes validated byte-for-byte against a
known-good draft), so exports load directly into a vLLM speculative serve. The
native ``qsd.train`` path (single-GPU, online extraction) is format-validated;
the fully GPU-validated, reproducible end-to-end recipe (data -> train -> serve
-> measured speedup) is the runnable ``examples/experimental/speculative_decoding``
example, which drives the proven TorchSpec streaming trainer and is where
multi-GPU training lives.

    qsd.data.synthesize(...)          # on-policy data generation
    qsd.eval.acceptance(...)          # AL@NST + per-position acceptance
    qsd.eval.throughput_sweep(...)    # per-GPU tokens/s sweep

Everything is also drivable from YAML recipes + CLI dotlist overrides through
:func:`quark.experimental.speculative_decoding.run.run_from_config`.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from quark.experimental.speculative_decoding import data, eval, export
    from quark.experimental.speculative_decoding.config import (
        DataConfig,
        EagleArchitectureConfig,
        SpecConfig,
        TorchSpecRunConfig,
        TrainConfig,
    )
    from quark.experimental.speculative_decoding.convert import SpecModel, convert
    from quark.experimental.speculative_decoding.export.export_hf import export_hf
    from quark.experimental.speculative_decoding.training.trainer import train

_LAZY_MODULES = {
    "data": "quark.experimental.speculative_decoding.data",
    "eval": "quark.experimental.speculative_decoding.eval",
    "export": "quark.experimental.speculative_decoding.export",
}
_LAZY_ATTRS = {
    "convert": ("quark.experimental.speculative_decoding.convert", "convert"),
    "SpecModel": ("quark.experimental.speculative_decoding.convert", "SpecModel"),
    "train": ("quark.experimental.speculative_decoding.training.trainer", "train"),
    "export_hf": ("quark.experimental.speculative_decoding.export.export_hf", "export_hf"),
    "SpecConfig": ("quark.experimental.speculative_decoding.config", "SpecConfig"),
    "EagleArchitectureConfig": ("quark.experimental.speculative_decoding.config", "EagleArchitectureConfig"),
    "DataConfig": ("quark.experimental.speculative_decoding.config", "DataConfig"),
    "TrainConfig": ("quark.experimental.speculative_decoding.config", "TrainConfig"),
    "TorchSpecRunConfig": ("quark.experimental.speculative_decoding.config", "TorchSpecRunConfig"),
}


def __getattr__(name: str) -> Any:
    """Load torch-backed APIs only when callers actually request them."""
    if name in _LAZY_MODULES:
        value = importlib.import_module(_LAZY_MODULES[name])
    elif name in _LAZY_ATTRS:
        module_name, attr = _LAZY_ATTRS[name]
        value = getattr(importlib.import_module(module_name), attr)
    else:
        raise AttributeError(name)
    globals()[name] = value
    return value


__all__ = [
    "convert",
    "SpecModel",
    "train",
    "export_hf",
    "SpecConfig",
    "EagleArchitectureConfig",
    "DataConfig",
    "TrainConfig",
    "TorchSpecRunConfig",
    "data",
    "eval",
    "export",
]

__version__ = "0.1.0"
