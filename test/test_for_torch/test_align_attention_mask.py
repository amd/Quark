#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest
import torch
from torch import nn

from quark.torch.algorithm.awq.utils import align_attention_mask_with_input


class DummyModel(nn.Module):
    """A dummy model to simulate forward signature."""

    def forward(self, x, attention_mask=None, unused_arg=None):
        return x


@pytest.fixture
def model():
    return DummyModel()


def test_filtering_logic(model):
    """Test if the function correctly filters out kwargs not in forward signature."""
    kwargs = {"attention_mask": torch.ones(1, 5), "invalid_arg": 100}
    filtered = align_attention_mask_with_input(model, kwargs)

    assert "attention_mask" in filtered
    assert "invalid_arg" not in filtered


def test_perfect_multiple_repeat(model):
    """Test Case: input_batch (4) is a perfect multiple of mask_batch (2)."""
    inp = torch.randn(4, 10)
    # Mask batch size is 2
    mask = torch.tensor([[1, 1], [0, 0]])
    kwargs = {"attention_mask": mask}

    result = align_attention_mask_with_input(model, kwargs, inp)
    updated_mask = result["attention_mask"]

    assert updated_mask.shape[0] == 4
    # Check if it's repeated: [1,1, 0,0, 1,1, 0,0]
    expected = torch.tensor([[1, 1], [0, 0], [1, 1], [0, 0]])
    assert torch.equal(updated_mask, expected)


def test_remainder_patching(model):
    """Test Case: input_batch (5) needs repeat (2*2) + patching (1)."""
    inp = torch.randn(5, 10)
    # Mask batch size is 2: [M0, M1]
    mask = torch.tensor([[1, 0], [0, 1]])
    kwargs = {"attention_mask": mask}

    result = align_attention_mask_with_input(model, kwargs, inp)
    updated_mask = result["attention_mask"]

    assert updated_mask.shape[0] == 5
    # Expected: [M0, M1] (repeat) + [M0, M1] (repeat) + [M0] (patch from index 0)
    expected = torch.tensor(
        [
            [1, 0],
            [0, 1],  # First repeat
            [1, 0],
            [0, 1],  # Second repeat
            [1, 0],  # Patch using mask[0]
        ]
    )
    assert torch.equal(updated_mask, expected)


def test_high_dimension_mask(model):
    """Test if it works with 4D attention masks (e.g., [batch, heads, seq, seq])."""
    inp = torch.randn(3, 10)
    # Mask: [2, 1, 4, 4]
    mask = torch.ones(2, 1, 4, 4)
    mask[0] = mask[0] * 0  # Make first sample distinct
    kwargs = {"attention_mask": mask}

    result = align_attention_mask_with_input(model, kwargs, inp)
    updated_mask = result["attention_mask"]

    assert updated_mask.shape == (3, 1, 4, 4)
    # The 3rd sample should be a copy of mask[0]
    assert torch.equal(updated_mask[2], mask[0])


def test_value_error_on_smaller_input(model):
    """Test if it raises ValueError when input batch < mask batch."""
    inp = torch.randn(2, 10)
    mask = torch.ones(4, 10)
    kwargs = {"attention_mask": mask}

    with pytest.raises(ValueError, match="indicates an abnormal shape mismatch"):
        align_attention_mask_with_input(model, kwargs, inp)


def test_no_input_provided(model):
    """Test if it returns filtered kwargs unchanged when inp is None."""
    mask = torch.ones(2, 10)
    kwargs = {"attention_mask": mask}
    result = align_attention_mask_with_input(model, kwargs, inp=None)

    assert torch.equal(result["attention_mask"], mask)
