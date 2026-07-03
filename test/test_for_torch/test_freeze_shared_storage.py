#
# Copyright (C) 2023 - 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Tests for ``uses_shared_storage`` semantics used by ModelQuantizer.freeze.

Background (commit f8b60ba): MoE experts produced by QuarkExperts hold
``nn.Linear`` weights that are zero-copy slices of a single stacked
``(num_experts, ...)`` tensor — they all share one underlying storage. Before
this commit, ``freeze`` did ``module.weight.data = quant_weight``, which
rebound the Parameter but left the original (huge) storage alive until every
sibling expert was rebound — temporarily doubling MoE weight memory. The fix:
when a weight is a view into a larger storage, use ``copy_()`` (in place) so
the shared storage is updated and never duplicated.

The predicate that selects the in-place branch is::

    parameter.untyped_storage().nbytes() > parameter.numel() * parameter.element_size()

These tests pin down its truth table and the resulting in-place semantics, so
a future refactor can't silently regress the MoE memory fix.
"""

import torch


def _uses_shared_storage(parameter: torch.Tensor) -> bool:
    """Mirror of the local helper inside ModelQuantizer.freeze (api.py)."""
    return parameter.untyped_storage().nbytes() > parameter.numel() * parameter.element_size()


class TestUsesSharedStoragePredicate:
    def test_standalone_tensor_is_not_shared(self):
        t = torch.empty(4, 4)
        assert _uses_shared_storage(t) is False

    def test_view_into_larger_storage_is_shared(self):
        # Simulate a QuarkExperts-style slice from a stacked (num_experts, ...) buffer.
        stacked = torch.empty(8, 4, 4)  # 8 "experts"
        sliced = stacked[0]  # numel=16, storage holds 128 elements
        assert _uses_shared_storage(sliced) is True

    def test_transposed_view_is_shared(self):
        # The MoE path transposes per-expert weights before wrapping into nn.Linear.
        stacked = torch.empty(8, 4, 4)
        sliced_t = stacked[0].transpose(0, 1)
        assert _uses_shared_storage(sliced_t) is True

    def test_clone_drops_sharing(self):
        stacked = torch.empty(8, 4, 4)
        cloned = stacked[0].clone()
        assert _uses_shared_storage(cloned) is False


class TestInPlaceVsRebindSemantics:
    """Verify the two branches behave as freeze() expects them to."""

    @torch.no_grad()
    def test_copy_in_shared_case_preserves_storage_pointer(self):
        # ModelQuantizer.freeze runs under @torch.no_grad() — mirror that here.
        # When weights share storage, copy_() keeps the storage alive (no rebind).
        stacked = torch.zeros(2, 4, 4)
        weight_a = torch.nn.Parameter(stacked[0])
        weight_b = torch.nn.Parameter(stacked[1])
        original_storage_ptr = stacked.untyped_storage().data_ptr()

        new_value = torch.ones(4, 4)
        assert _uses_shared_storage(weight_a)
        weight_a.copy_(new_value)

        # weight_a's storage is unchanged (still the stacked buffer) and weight_b is
        # untouched — exactly the property MoE freeze relies on.
        assert weight_a.untyped_storage().data_ptr() == original_storage_ptr
        assert weight_b.untyped_storage().data_ptr() == original_storage_ptr
        torch.testing.assert_close(weight_a.data, new_value)
        torch.testing.assert_close(weight_b.data, torch.zeros(4, 4))

    def test_data_assignment_in_non_shared_case_rebinds_storage(self):
        # When weights don't share storage, ``module.weight.data = ...`` is cheap
        # and the new value owns its own storage.
        weight = torch.nn.Parameter(torch.zeros(4, 4))
        assert not _uses_shared_storage(weight)

        new_value = torch.ones(4, 4)
        weight.data = new_value
        assert weight.untyped_storage().data_ptr() == new_value.untyped_storage().data_ptr()
        torch.testing.assert_close(weight.data, new_value)
