#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""
Test for EasyGraph.find_nearest_module_name's handling of an empty
``nn_module_stack`` entry on the FX node metadata.

The function currently does ``[...][-1][0]`` over a list comprehension on
``node.meta["nn_module_stack"].items()`` without checking whether the dict
is empty. When the metadata key is present but the stack is empty
(e.g. constants, dead code, or fresh nodes), ``[-1]`` raises IndexError.
The function should fall through to the default empty-string return.
"""

from types import SimpleNamespace

from quark.torch.algorithm.utils.auto_config import EasyGraph


def _make_node(nn_module_stack):
    """Build a stand-in FX node exposing only the .meta attribute the
    function under test reads from."""
    return SimpleNamespace(meta={"nn_module_stack": nn_module_stack})


def test_find_nearest_module_name_empty_stack_returns_empty_string():
    """Empty nn_module_stack must not raise IndexError."""
    node = _make_node({})
    assert EasyGraph.find_nearest_module_name(node) == ""


def test_find_nearest_module_name_missing_key_returns_empty_string():
    """Sanity check: when nn_module_stack key is absent, also returns ''."""
    node = SimpleNamespace(meta={})
    assert EasyGraph.find_nearest_module_name(node) == ""


def test_find_nearest_module_name_populated_stack_still_works():
    """Regression: a populated stack must continue to produce the cleaned name."""
    nn_module_stack = {
        "L__self__": ("L['self']", type("Root", (), {})),
        "leaf": (
            "L['self']._modules['model']._modules['layers']._modules['0'].input_layernorm",
            type("Layer", (), {}),
        ),
    }
    node = _make_node(nn_module_stack)
    assert EasyGraph.find_nearest_module_name(node) == "model.layers.0.input_layernorm"
