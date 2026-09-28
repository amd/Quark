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

from quark.torch.quantization.cache_integration import QuarkCacheLayer, QuarkQuantizedCache


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


def test_cache_layer_reports_unbounded_length():
    """Both max-length accessors must report -1 so HF treats the layer as dynamic.

    Which of the two a given transformers release calls depends on its Cache API vintage, so they
    have to agree; a fixed length here would make HF pre-allocate and truncate the KV cache.
    """
    layer = QuarkCacheLayer()

    assert layer.get_max_cache_shape() == -1
    assert layer.get_max_length() == -1


@pytest.mark.parametrize(
    "query_length",
    [3, torch.arange(3)],
    ids=["int query_length", "cache_position tensor"],
)
def test_cache_layer_mask_sizes_count_new_tokens_for_both_hf_signatures(query_length):
    """New tokens must extend the reported KV length under either HF argument shape.

    transformers 5.x hands the mask builder a plain int where releases up to 5.2 handed it a
    ``cache_position`` tensor. Sizing off ``len()`` alone counts zero new tokens for the int, which
    silently understates ``kv_length`` by the whole query and mis-sizes the attention mask.
    """
    layer = QuarkCacheLayer()
    layer.update_lengths(5)

    assert layer.get_mask_sizes(query_length) == (8, 5)


def test_cache_layer_mask_sizes_without_query_length():
    """No query length means nothing new to attend to, so KV length stays at the cached prefix."""
    layer = QuarkCacheLayer()
    layer.update_lengths(5)

    assert layer.get_mask_sizes(None) == (5, 5)
