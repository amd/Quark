#
# Copyright (C) 2025 - 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for ``quark.torch.utils.accelerate_helper.clone_align_devices_hook``.

The helper centralizes the 8-attribute ``AlignDevicesHook`` copy block that
was previously inlined at five call sites.  These tests exercise its three
distinct contracts: default (no override), explicit ``weights_map`` override
(used by the Llama4/Qwen3VLMoE expert paths in
``quark/torch/utils/llm/module_replacement/replacement_utils.py``), and the
``ImportError`` path when ``accelerate`` is missing.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from quark.torch.utils.accelerate_helper import clone_align_devices_hook


def test_clone_align_devices_hook_no_override() -> None:
    """When ``weights_map`` is omitted, every attribute is copied verbatim."""
    pytest.importorskip("accelerate")
    from accelerate.hooks import AlignDevicesHook

    src_weights_map = {"layer.weight": "data"}
    src = AlignDevicesHook(
        execution_device="cpu",
        offload=True,
        io_same_device=True,
        weights_map=src_weights_map,
        offload_buffers=True,
        place_submodules=True,
        skip_keys="output",
        tied_params_map={},
    )

    cloned = clone_align_devices_hook(src)

    assert cloned is not src
    assert cloned.execution_device == src.execution_device
    assert cloned.offload == src.offload
    assert cloned.io_same_device == src.io_same_device
    assert cloned.weights_map is src.weights_map  # default: copied by reference
    assert cloned.offload_buffers == src.offload_buffers
    assert cloned.place_submodules == src.place_submodules
    assert cloned.skip_keys == src.skip_keys
    assert cloned.tied_params_map == src.tied_params_map


def test_clone_align_devices_hook_weights_map_override() -> None:
    """Passing ``weights_map`` substitutes only that attribute; the rest are copied."""
    pytest.importorskip("accelerate")
    from accelerate.hooks import AlignDevicesHook

    src = AlignDevicesHook(execution_device="cpu", offload=False, weights_map={})
    new_weights_map = {"custom.layer.weight": "override"}

    cloned = clone_align_devices_hook(src, weights_map=new_weights_map)

    assert cloned.weights_map is new_weights_map
    assert cloned.weights_map is not src.weights_map
    # Other attributes are unchanged.
    assert cloned.execution_device == "cpu"
    assert cloned.offload is False


def test_clone_align_devices_hook_requires_accelerate() -> None:
    """``ImportError`` surfaces when ``accelerate`` is unavailable, with an actionable message."""
    with (
        patch("quark.torch.utils.accelerate_helper.is_accelerate_available", return_value=False),
        pytest.raises(ImportError, match="requires the package `accelerate`"),
    ):
        clone_align_devices_hook(hook=object())  # type: ignore[arg-type]
