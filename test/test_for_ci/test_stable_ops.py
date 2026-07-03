#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Direct unit tests for every stable ABI wrapper in stable_ops.h.

stable_ops_bindings.cpp registers each ``quark::torch_stable`` wrapper as a
custom op under the ``quark_test_stable_ops`` namespace. Tests compare each
op's output against the equivalent native PyTorch call.

The whole module is gated by ``pytestmark = pytest.mark.skipif(...)`` on
PyTorch < 2.10 (no stable ABI). On supported versions, the
``quark_test_stable_ops`` namespace is registered as a side effect of the
prod stable-ABI JIT build, which is triggered by importing
``quark.torch.kernel.hw_emulation.extensions`` below.
"""

import pytest
import torch

import quark.torch.kernel.hw_emulation.extensions  # noqa: F401  # triggers prod stable-ABI JIT build that registers quark_test_stable_ops
from quark.common.utils.torch_utils import torch_supports_stable_abi

pytestmark = pytest.mark.skipif(
    not torch_supports_stable_abi(),
    reason="test_stable_ops.py requires PyTorch >= 2.10 (stable ABI).",
)


_ops = torch.ops.quark_test_stable_ops


def _assert_matches_torch(op_name, *args, equal_nan=False):
    """Assert ``_ops.<op_name>(*args) == torch.<op_name>(*args)``."""
    stable_result = getattr(_ops, op_name)(*args)
    native_result = getattr(torch, op_name)(*args)
    torch.testing.assert_close(stable_result, native_result, equal_nan=equal_nan)


_UNARY_ROUNDING_DATA = {
    "floor": {
        "basic_1d": [1.7, -1.3, 2.0, -0.5, 0.0],
        "scalar_0d": 1.7,
        "float64_1d": [1.7, -1.3],
    },
    "ceil": {
        "basic_1d": [1.3, -1.7, 2.0, -0.5, 0.0],
        "scalar_0d": -1.7,
        "float64_1d": [1.3, -1.7],
    },
    "round": {
        "basic_1d": [1.4, 1.5, 2.5, 3.5, -0.5, -1.5],
        "scalar_0d": 2.6,
        "float64_1d": [1.4, 2.5, -0.5],
    },
}


@pytest.mark.parametrize("op_name", list(_UNARY_ROUNDING_DATA))
class TestUnaryRounding:
    def test_basic(self, op_name):
        _assert_matches_torch(op_name, torch.tensor(_UNARY_ROUNDING_DATA[op_name]["basic_1d"]))

    def test_2d(self, op_name):
        _assert_matches_torch(op_name, torch.randn(4, 8))

    def test_special_floats(self, op_name):
        x = torch.tensor([float("inf"), float("-inf"), float("nan")])
        _assert_matches_torch(op_name, x, equal_nan=True)

    def test_already_integer(self, op_name):
        x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])
        torch.testing.assert_close(getattr(_ops, op_name)(x), x)

    def test_empty(self, op_name):
        _assert_matches_torch(op_name, torch.tensor([], dtype=torch.float32))

    def test_scalar_0d(self, op_name):
        _assert_matches_torch(op_name, torch.tensor(_UNARY_ROUNDING_DATA[op_name]["scalar_0d"]))

    def test_float64(self, op_name):
        x = torch.tensor(_UNARY_ROUNDING_DATA[op_name]["float64_1d"], dtype=torch.float64)
        _assert_matches_torch(op_name, x)


class TestRoundOnly:
    def test_bankers_rounding(self):
        """Round-half-to-even: 0.5 -> 0, 1.5 -> 2, 2.5 -> 2."""
        x = torch.tensor([0.5, 1.5, 2.5, 3.5, 4.5])
        expected = torch.tensor([0.0, 2.0, 2.0, 4.0, 4.0])
        torch.testing.assert_close(_ops.round(x), expected)

    def test_negative_half_integers(self):
        _assert_matches_torch("round", torch.tensor([-0.5, -1.5, -2.5, -3.5, -4.5]))


_BINARY_ARITH_BASIC = {
    "mul": ([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]),
    "div": ([10.0, 20.0, 30.0], [2.0, 4.0, 5.0]),
    "sub": ([5.0, 10.0, 15.0], [1.0, 2.0, 3.0]),
}


@pytest.mark.parametrize("op_name", list(_BINARY_ARITH_BASIC))
class TestBinaryArithmeticShared:
    def test_basic(self, op_name):
        a_vals, b_vals = _BINARY_ARITH_BASIC[op_name]
        _assert_matches_torch(op_name, torch.tensor(a_vals), torch.tensor(b_vals))

    def test_empty(self, op_name):
        empty = torch.tensor([], dtype=torch.float32)
        _assert_matches_torch(op_name, empty, empty)

    def test_scalar_0d(self, op_name):
        _assert_matches_torch(op_name, torch.tensor(10.0), torch.tensor(3.0))

    def test_float64(self, op_name):
        a = torch.randn(4, dtype=torch.float64)
        b = torch.randn(4, dtype=torch.float64)
        if op_name == "div":
            b = b.abs() + 0.1
        _assert_matches_torch(op_name, a, b)


class TestMul:
    def test_broadcast(self):
        _assert_matches_torch("mul", torch.randn(3, 4), torch.randn(1, 4))

    def test_by_zero(self):
        _assert_matches_torch("mul", torch.tensor([1.0, -1.0, 0.0]), torch.zeros(3), equal_nan=True)

    def test_special_floats(self):
        a = torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0])
        b = torch.tensor([2.0, 2.0, 2.0, float("inf")])
        _assert_matches_torch("mul", a, b, equal_nan=True)


class TestDiv:
    def test_broadcast(self):
        _assert_matches_torch("div", torch.randn(3, 4), torch.tensor([2.0]))

    def test_div_by_zero(self):
        _assert_matches_torch("div", torch.tensor([1.0, -1.0, 0.0]), torch.zeros(3), equal_nan=True)

    def test_negative_division(self):
        _assert_matches_torch("div", torch.tensor([-6.0, 6.0, -6.0]), torch.tensor([3.0, -3.0, -3.0]))


class TestSub:
    def test_broadcast(self):
        _assert_matches_torch("sub", torch.randn(3, 4), torch.randn(1, 4))

    def test_self_subtract(self):
        a = torch.tensor([1.0, -1.0, 1e30, -1e30])
        _assert_matches_torch("sub", a, a)


@pytest.mark.parametrize("op_name", ["lt", "gt"])
class TestComparisonLtGt:
    def test_basic(self, op_name):
        _assert_matches_torch(op_name, torch.tensor([1.0, 5.0, 3.0]), torch.tensor([2.0, 4.0, 3.0]))

    def test_negative(self, op_name):
        _assert_matches_torch(op_name, torch.tensor([-1.0, -5.0, 0.0]), torch.tensor([0.0, -3.0, 0.0]))

    def test_equal_values(self, op_name):
        a = torch.tensor([1.0, 2.0, 3.0])
        _assert_matches_torch(op_name, a, a)

    def test_special_floats(self, op_name):
        a = torch.tensor([float("nan"), float("inf"), float("-inf"), 0.0])
        b = torch.tensor([0.0, 0.0, 0.0, float("nan")])
        _assert_matches_torch(op_name, a, b)

    def test_broadcast(self, op_name):
        _assert_matches_torch(op_name, torch.randn(3, 4), torch.randn(1, 4))

    def test_empty(self, op_name):
        empty = torch.tensor([], dtype=torch.float32)
        _assert_matches_torch(op_name, empty, empty)

    def test_scalar_0d(self, op_name):
        _assert_matches_torch(op_name, torch.tensor(2.0), torch.tensor(1.0))


class TestEq:
    """Eq-specific IEEE 754 semantics (NaN != NaN, inf == inf, +0 == -0)."""

    @pytest.mark.parametrize(
        "a, b",
        [
            pytest.param(torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1.0, 9.0, 3.0]), id="basic"),
            pytest.param(torch.ones(4), torch.ones(4), id="all_equal"),
            pytest.param(torch.tensor([float("nan")]), torch.tensor([float("nan")]), id="nan_ne_nan"),
            pytest.param(
                torch.tensor([float("inf"), float("-inf")]),
                torch.tensor([float("inf"), float("-inf")]),
                id="inf_eq_inf",
            ),
            pytest.param(torch.randn(3, 4), torch.randn(1, 4), id="broadcast"),
            pytest.param(
                torch.tensor([], dtype=torch.float32),
                torch.tensor([], dtype=torch.float32),
                id="empty",
            ),
            pytest.param(torch.tensor([0.0]), torch.tensor([-0.0]), id="zero_signs"),
        ],
    )
    def test_matches_torch(self, a, b):
        _assert_matches_torch("eq", a, b)


@pytest.mark.parametrize(
    "a, b",
    [
        pytest.param(
            torch.tensor([True, True, False, False]),
            torch.tensor([True, False, True, False]),
            id="bool",
        ),
        pytest.param(
            torch.tensor([1.0, 0.0, 1.0, 0.0]),
            torch.tensor([1.0, 1.0, 0.0, 0.0]),
            id="float",
        ),
        pytest.param(torch.tensor([0, 1, 2, 0]), torch.tensor([1, 1, 0, 0]), id="int"),
        pytest.param(torch.ones(4, dtype=torch.bool), torch.ones(4, dtype=torch.bool), id="all_true"),
        pytest.param(torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool), id="all_false"),
        pytest.param(
            torch.tensor([], dtype=torch.bool),
            torch.tensor([], dtype=torch.bool),
            id="empty",
        ),
        pytest.param(
            torch.tensor([[True], [False]]),
            torch.tensor([True, False, True]),
            id="broadcast",
        ),
    ],
)
def test_logical_and(a, b):
    _assert_matches_torch("logical_and", a, b)


@pytest.mark.parametrize(
    "x",
    [
        pytest.param(torch.tensor([True, False, True, False]), id="bool"),
        pytest.param(torch.tensor([0.0, 1.0, -1.0, 0.0]), id="float"),
        pytest.param(torch.tensor([0, 1, -1, 2, 0]), id="int"),
        pytest.param(torch.ones(5, dtype=torch.bool), id="all_true"),
        pytest.param(torch.zeros(5, dtype=torch.bool), id="all_false"),
        pytest.param(torch.tensor([], dtype=torch.bool), id="empty"),
        pytest.param(torch.tensor(True), id="scalar_0d"),
    ],
)
def test_logical_not(x):
    _assert_matches_torch("logical_not", x)


@pytest.mark.parametrize(
    "a, b",
    [
        pytest.param(torch.tensor([42.0]), torch.randn(5), id="scalar_to_1d"),
        pytest.param(torch.tensor([[1.0, 2.0, 3.0]]), torch.randn(4, 3), id="row_to_matrix"),
        pytest.param(torch.tensor(7.0), torch.randn(3), id="0d_to_1d"),
        pytest.param(torch.randn(3, 4), torch.randn(3, 4), id="same_shape_noop"),
        pytest.param(torch.randn(1, 1, 4), torch.randn(2, 3, 4), id="3d_broadcast"),
        pytest.param(torch.tensor([[1.0], [2.0], [3.0]]), torch.randn(3, 5), id="column_to_matrix"),
    ],
)
def test_expand_as(a, b):
    torch.testing.assert_close(_ops.expand_as(a, b), a.expand_as(b))


@pytest.mark.parametrize(
    "cond, a, b, equal_nan",
    [
        pytest.param(
            torch.tensor([True, False, True, False]),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([10.0, 20.0, 30.0, 40.0]),
            False,
            id="basic",
        ),
        pytest.param(torch.randn(3, 4) > 0, torch.randn(3, 4), torch.randn(3, 4), False, id="2d"),
        pytest.param(
            torch.ones(4, dtype=torch.bool),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([10.0, 20.0, 30.0, 40.0]),
            False,
            id="all_true",
        ),
        pytest.param(
            torch.zeros(4, dtype=torch.bool),
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            torch.tensor([10.0, 20.0, 30.0, 40.0]),
            False,
            id="all_false",
        ),
        pytest.param(
            torch.tensor([[True], [False], [True]]),
            torch.randn(3, 4),
            torch.randn(3, 4),
            False,
            id="broadcast",
        ),
        pytest.param(
            torch.tensor([True, False, True, False]),
            torch.tensor([float("nan"), float("inf"), 0.0, float("-inf")]),
            torch.tensor([1.0, 2.0, float("nan"), float("inf")]),
            True,
            id="special_floats",
        ),
        pytest.param(
            torch.tensor([], dtype=torch.bool),
            torch.tensor([], dtype=torch.float32),
            torch.tensor([], dtype=torch.float32),
            False,
            id="empty",
        ),
        pytest.param(
            torch.tensor([True, False, True]),
            torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
            torch.tensor([4.0, 5.0, 6.0], dtype=torch.float64),
            False,
            id="float64",
        ),
    ],
)
def test_where(cond, a, b, equal_nan):
    _assert_matches_torch("where", cond, a, b, equal_nan=equal_nan)


_LIKE_SHAPES = [
    pytest.param(torch.randn(3, 5), id="2d"),
    pytest.param(torch.randn(10), id="1d"),
    pytest.param(torch.randn(2, 3, 4), id="3d"),
    pytest.param(torch.tensor([], dtype=torch.float32), id="empty"),
    pytest.param(torch.tensor(42.0), id="scalar_0d"),
]


class TestZerosLike:
    @pytest.mark.parametrize("x", _LIKE_SHAPES)
    def test_matches_torch(self, x):
        result = _ops.zeros_like(x)
        assert result.shape == x.shape
        assert result.dtype == x.dtype
        torch.testing.assert_close(result, torch.zeros_like(x))

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["float32", "float64"])
    def test_preserves_dtype(self, dtype):
        x = torch.randn(4, dtype=dtype)
        result = _ops.zeros_like(x)
        assert result.dtype == dtype
        torch.testing.assert_close(result, torch.zeros(4, dtype=dtype))


class TestFullLike:
    @pytest.mark.parametrize(
        "fill_value, equal_nan",
        [
            pytest.param(3.14, False, id="positive"),
            pytest.param(-2.5, False, id="negative"),
            pytest.param(0.0, False, id="zero"),
            pytest.param(99.0, False, id="medium"),
            pytest.param(1e38, False, id="large"),
            pytest.param(float("inf"), False, id="inf"),
            pytest.param(float("nan"), True, id="nan"),
        ],
    )
    @pytest.mark.parametrize("x", _LIKE_SHAPES)
    def test_matches_torch(self, x, fill_value, equal_nan):
        result = _ops.full_like(x, fill_value)
        assert result.shape == x.shape
        assert result.dtype == x.dtype
        torch.testing.assert_close(result, torch.full_like(x, fill_value), equal_nan=equal_nan)

    def test_preserves_dtype(self):
        x = torch.randn(3, dtype=torch.float64)
        result = _ops.full_like(x, 2.0)
        assert result.dtype == torch.float64
        torch.testing.assert_close(result, torch.full_like(x, 2.0))
