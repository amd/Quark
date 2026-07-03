#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Regression tests for NonScaledFakeQuantizeFunction.backward gradient count.

Background: NonScaledFakeQuantizeFunction wraps the custom op
``ops.quark.non_scaled_fake_quantize``. PyTorch requires
``Function.backward`` to return exactly one gradient per ``forward`` input
(excluding ``ctx``); a mismatch raises RuntimeError during backprop. Because
``backward`` is hand-written and decoupled from the op signature, adding or
removing a ``forward`` parameter can silently desynchronize the two.

These tests exercise the real ``NonScaledFakeQuantizeFunction`` and lock in
the forward/backward arity coupling so future signature changes fail loudly.
"""

import inspect

import torch

from quark.torch.kernel import NonScaledFakeQuantizeFunction, non_scaled_fake_quantize


def _forward_param_count() -> int:
    """Number of forward inputs excluding ``ctx``."""
    return sum(1 for name in inspect.signature(NonScaledFakeQuantizeFunction.forward).parameters if name != "ctx")


class TestNonScaledFakeQuantizeGradient:
    def test_backward_runs_and_propagates_gradient(self):
        """End-to-end: a wrong backward arity would raise here. Pre-fix this
        raised ``function NonScaledFakeQuantizeFunctionBackward returned an
        incorrect number of gradients``."""
        x = torch.randn(2, 32, dtype=torch.float32, requires_grad=True)

        y = non_scaled_fake_quantize(x, "mx", "fp4", -1, 32, "even")
        y.sum().backward()

        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_backward_returns_one_grad_per_forward_input(self):
        """Structural guard: ``backward`` return arity must match ``forward``
        input arity. Catches future drift if a ``forward`` parameter is added
        or removed without updating ``backward``."""
        grad_out = torch.ones(2, 32, dtype=torch.float32)
        grads = NonScaledFakeQuantizeFunction.backward(None, grad_out)

        assert len(grads) == _forward_param_count(), (
            f"backward returned {len(grads)} gradients but forward has {_forward_param_count()} inputs (excluding ctx)"
        )
