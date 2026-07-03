#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for ``quark.onnx.operators.quant_ops.concat.QDQConcat``.

The class is not currently registered in
``quark/onnx/quantizers/registry.py``, so no integration test exercises
it.  These tests construct it directly with mocked ``quantizer`` and
``node`` objects and drive every branch of ``quantize()``.
"""

import itertools
import unittest
from typing import Any
from unittest.mock import MagicMock

import onnx

from quark.onnx.operators.quant_ops.concat import QDQConcat


def _make_concat_node(name: str = "concat0", n_inputs: int = 3) -> Any:
    """Construct a minimal Concat NodeProto with ``n_inputs`` inputs."""
    inputs = [f"in_{i}" for i in range(n_inputs)]
    return onnx.helper.make_node("Concat", inputs=inputs, outputs=[f"{name}_out"], name=name, axis=0)


def _make_concat_op(
    *,
    force_quantize: bool,
    disable_qdq: bool,
    all_inputs_quantized: bool = True,
    n_inputs: int = 3,
) -> tuple[QDQConcat, MagicMock]:
    """Construct a QDQConcat via its real ``__init__`` (covers line 14).

    The base class ``__init__`` just sets ``self.quantizer`` and
    ``self.node`` and reads ``onnx_quantizer.disable_qdq_for_node_output``
    (we expose that as a MagicMock attribute so the base init is happy).
    """
    quantizer = MagicMock()
    quantizer.force_quantize_no_input_check = force_quantize
    quantizer.is_tensor_quantized = MagicMock(return_value=all_inputs_quantized)
    # The base __init__ reads this attribute; the value we put into
    # `op.disable_qdq_for_node_output` later overrides it for clarity.
    quantizer.disable_qdq_for_node_output = []
    node = _make_concat_node(n_inputs=n_inputs)
    op = QDQConcat(quantizer, node)  # exercises line 14: super().__init__
    op.disable_qdq_for_node_output = disable_qdq
    return op, quantizer


class TestQDQConcatQuantize(unittest.TestCase):
    """Drive every branch of :meth:`QDQConcat.quantize`."""

    def test_force_quantize_with_outputs(self) -> None:
        """force_quantize=True + disable_qdq=False -> inputs AND outputs."""
        op, quantizer = _make_concat_op(force_quantize=True, disable_qdq=False, n_inputs=3)
        op.quantize()
        # All inputs + the single output should have been quantized.
        called = [c.args[0] for c in quantizer.quantize_activation_tensor.call_args_list]
        self.assertEqual(called, ["in_0", "in_1", "in_2", "concat0_out"])

    def test_force_quantize_without_outputs(self) -> None:
        """force_quantize=True + disable_qdq=True -> inputs ONLY."""
        op, quantizer = _make_concat_op(force_quantize=True, disable_qdq=True, n_inputs=2)
        op.quantize()
        called = [c.args[0] for c in quantizer.quantize_activation_tensor.call_args_list]
        self.assertEqual(called, ["in_0", "in_1"])

    def test_default_path_all_inputs_quantized(self) -> None:
        """force_quantize=False + all inputs already quantized + disable_qdq=False
        -> inputs AND outputs are quantized."""
        op, quantizer = _make_concat_op(force_quantize=False, disable_qdq=False, all_inputs_quantized=True, n_inputs=2)
        op.quantize()
        called = [c.args[0] for c in quantizer.quantize_activation_tensor.call_args_list]
        self.assertEqual(called, ["in_0", "in_1", "concat0_out"])

    def test_default_path_some_inputs_unquantized(self) -> None:
        """force_quantize=False + not all inputs quantized -> nothing happens."""
        op, quantizer = _make_concat_op(force_quantize=False, disable_qdq=False, all_inputs_quantized=False, n_inputs=3)
        op.quantize()
        quantizer.quantize_activation_tensor.assert_not_called()

    def test_default_path_disable_qdq_short_circuits(self) -> None:
        """force_quantize=False + all inputs quantized BUT disable_qdq=True
        -> the ``and not self.disable_qdq_for_node_output`` guard short-circuits
        and nothing is quantized."""
        op, quantizer = _make_concat_op(force_quantize=False, disable_qdq=True, all_inputs_quantized=True, n_inputs=2)
        op.quantize()
        quantizer.quantize_activation_tensor.assert_not_called()

    def test_quantize_asserts_op_type(self) -> None:
        """``quantize`` asserts the node op_type is 'Concat'."""
        op, _ = _make_concat_op(force_quantize=True, disable_qdq=False)
        op.node = onnx.helper.make_node("MatMul", inputs=["a", "b"], outputs=["y"], name="not_concat")
        with self.assertRaises(AssertionError):
            op.quantize()


class TestQDQConcatExhaustive(unittest.TestCase):
    """Exhaustively iterate every (force_quantize, disable_qdq,
    all_inputs_quantized) combination to make sure no branch raises."""

    def test_all_8_combinations_do_not_raise(self) -> None:
        for force, disable, all_q in itertools.product([True, False], repeat=3):
            with self.subTest(force=force, disable=disable, all_q=all_q):
                op, _ = _make_concat_op(
                    force_quantize=force,
                    disable_qdq=disable,
                    all_inputs_quantized=all_q,
                )
                op.quantize()  # must not raise


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
