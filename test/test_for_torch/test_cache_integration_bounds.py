#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Test for QuarkQuantizedCache.load_from_state_dict bounds checking.

The method indexes ``config[0]``, ``config[1]``, ``config[2]`` from the
``state_dict["kv_cache.config"]`` tensor without checking its length.
A truncated or malformed checkpoint with fewer than 3 entries triggers
``IndexError`` instead of a typed error explaining the problem.
"""

import pytest
import torch
import torch.nn as nn

from quark.torch.quantization.cache_integration import QuarkQuantizedCache


def _make_cache():
    """Build a minimal QuarkQuantizedCache instance for direct method testing."""
    return QuarkQuantizedCache(cache_config={})


def test_load_from_state_dict_short_config_raises_value_error():
    """A config tensor with fewer than 3 elements must raise a typed
    ValueError (not a low-level IndexError) so callers can react."""
    cache = _make_cache()
    state_dict = {"kv_cache.config": torch.tensor([5, 3])}  # only 2 elements

    with pytest.raises(ValueError, match="kv_cache.config"):
        cache.load_from_state_dict(state_dict, nn.Linear(1, 1))


def test_load_from_state_dict_empty_config_raises_value_error():
    """An empty config tensor should also raise a typed ValueError."""
    cache = _make_cache()
    state_dict = {"kv_cache.config": torch.tensor([], dtype=torch.long)}

    with pytest.raises(ValueError, match="kv_cache.config"):
        cache.load_from_state_dict(state_dict, nn.Linear(1, 1))


def test_load_from_state_dict_well_formed_config_succeeds():
    """Regression: a 3-element config tensor must continue to load cleanly."""
    cache = _make_cache()
    state_dict = {"kv_cache.config": torch.tensor([4, 2, 2])}

    # Should not raise.
    cache.load_from_state_dict(state_dict, nn.Linear(1, 1))


def test_load_from_state_dict_missing_config_key_is_a_no_op():
    """Regression: an absent config key should leave the method silent
    (the existing ``if "kv_cache.config" in state_dict:`` guard already
    handles this)."""
    cache = _make_cache()
    cache.load_from_state_dict({}, nn.Linear(1, 1))
