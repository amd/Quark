#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from quark.torch.algorithm.osscar.osscar import OsscarProcessor


def _config(use_cache: bool, nested: bool) -> SimpleNamespace:
    if nested:
        return SimpleNamespace(text_config=SimpleNamespace(use_cache=use_cache))
    return SimpleNamespace(use_cache=use_cache)


def _get_use_cache(config: SimpleNamespace, nested: bool) -> bool:
    return config.text_config.use_cache if nested else config.use_cache


@pytest.mark.parametrize("nested_config", [False, True])
def test_osscar_disables_and_restores_cache_for_flat_and_nested_configs(nested_config):
    model = SimpleNamespace(config=_config(use_cache=True, nested=nested_config))

    processor = OsscarProcessor.__new__(OsscarProcessor)
    processor.model = model
    processor.inps = []
    processor.modules = []
    # Set by __init__; this test bypasses it. False keeps the offload path live.
    processor.using_accelerate = False

    observed_cache_values = []
    with patch(
        "quark.torch.algorithm.osscar.osscar.clear_memory",
        side_effect=lambda: observed_cache_values.append(_get_use_cache(model.config, nested_config)),
    ):
        processor.apply()

    assert observed_cache_values[0] is False
    assert _get_use_cache(model.config, nested_config) is True
