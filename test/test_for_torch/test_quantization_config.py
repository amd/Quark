#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from quark.torch.quantization.config.config import Int4PerGroupSpec

INT4_PER_GROUP_SYM_SPEC = Int4PerGroupSpec(ch_axis=1, is_dynamic=False, group_size=128).to_quantization_spec()


def test_set_group_size():
    # Create an instance of QTensorConfig

    # Set group size
    new_group_size = 8
    INT4_PER_GROUP_SYM_SPEC.set_group_size(new_group_size)

    # Assert the group size was set correctly
    assert INT4_PER_GROUP_SYM_SPEC.group_size == new_group_size, "The group size should be updated to the new value"

    # Test with group_size = -1 (valid case)
    INT4_PER_GROUP_SYM_SPEC.set_group_size(-1)
    assert INT4_PER_GROUP_SYM_SPEC.group_size == -1, "The group size should be set to -1"

    # Test with group_size = 0.1 (invalid case)
    try:
        INT4_PER_GROUP_SYM_SPEC.set_group_size(0.1)
    except AssertionError as e:
        assert (
            str(e) == "Group size must be a positive integer or -1 (which means group size equals to dimension size)."
        ), "Expected AssertionError for invalid group size"
