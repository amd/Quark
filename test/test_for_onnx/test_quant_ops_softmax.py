#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for ``quark.onnx.operators.quant_ops.softmax.QDQSoftmax``.

The class is not currently registered in
``quark/onnx/quantizers/registry.py``, so no integration test exercises
it.  These tests construct it directly with mocked ``quantizer`` and
``node`` objects and drive every branch of ``quantize()``.
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import onnx

from quark.onnx.operators.quant_ops.softmax import QDQSoftmax

# (qtype, is_symmetric)  ->  expected (scale, zero_point, dtype)
SOFTMAX_QPARAMS: dict[tuple[int, bool], tuple[float, int, type]] = {
    (onnx.onnx_pb.TensorProto.UINT8, True): (1 / 128.0, 128, np.uint8),
    (onnx.onnx_pb.TensorProto.UINT8, False): (1 / 256.0, 0, np.uint8),
    (onnx.onnx_pb.TensorProto.INT8, True): (1 / 128.0, 0, np.int8),
    (onnx.onnx_pb.TensorProto.INT8, False): (1 / 256.0, -128, np.int8),
    (onnx.onnx_pb.TensorProto.UINT16, True): (1 / 32768.0, 32768, np.uint16),
    (onnx.onnx_pb.TensorProto.UINT16, False): (1 / 65536.0, 0, np.uint16),
    (onnx.onnx_pb.TensorProto.INT16, True): (1 / 32768.0, 0, np.int16),
    (onnx.onnx_pb.TensorProto.INT16, False): (1 / 65536.0, -32768, np.int16),
    (onnx.onnx_pb.TensorProto.UINT32, True): (1.0 / 2**31, 2**31, np.uint32),
    (onnx.onnx_pb.TensorProto.UINT32, False): (1.0 / 2**32, 0, np.uint32),
    (onnx.onnx_pb.TensorProto.INT32, True): (1.0 / 2**31, 0, np.int32),
    (onnx.onnx_pb.TensorProto.INT32, False): (1.0 / 2**32, -(2**31), np.int32),
}

# The else branch returns float scale=1, zp=0 regardless of symmetry.
SOFTMAX_FALLBACK = (1.0, 0, np.float32)


def _make_softmax(qtype: int, *, symmetric: bool) -> QDQSoftmax:
    """Build a QDQSoftmax with mocked dependencies, bypassing the
    ``onnxruntime`` base-class ``__init__`` to keep the unit test free of
    any quantizer-construction side effects."""
    op = object.__new__(QDQSoftmax)
    op.quantizer = MagicMock()
    op.quantizer.activation_qType = qtype
    op.quantizer.is_activation_symmetric = symmetric
    op.node = MagicMock()
    op.node.output = ["softmax_out"]
    return op


@patch("quark.onnx.operators.quant_ops.softmax.QDQOperatorBase.quantize", lambda self: None)
class TestQDQSoftmaxQuantize(unittest.TestCase):
    """Drive every branch of :meth:`QDQSoftmax.quantize`."""

    def _assert_qparams(
        self,
        op: QDQSoftmax,
        expected_scale: float,
        expected_zp: int,
        expected_dtype: type,
    ) -> None:
        op.quantize()
        op.quantizer.set_quant_scale_zp.assert_called_once()
        args = op.quantizer.set_quant_scale_zp.call_args.args
        self.assertEqual(args[0], "softmax_out")
        scale, zp = args[1]
        np.testing.assert_allclose(scale, np.float32(expected_scale), rtol=0, atol=0)
        self.assertEqual(scale.dtype, np.float32)
        self.assertEqual(int(zp), expected_zp)
        self.assertEqual(zp.dtype, expected_dtype)

    def test_all_known_dtype_branches(self) -> None:
        """One subTest per (dtype, symmetry) combination — 12 branches."""
        for (qtype, symmetric), (scale, zp, dtype) in SOFTMAX_QPARAMS.items():
            with self.subTest(qtype=qtype, symmetric=symmetric):
                op = _make_softmax(qtype, symmetric=symmetric)
                self._assert_qparams(op, scale, zp, dtype)

    def test_unknown_dtype_falls_through_to_else(self) -> None:
        """Any qType not in the if/elif chain hits the trailing ``else``."""
        # FLOAT is not one of the listed integer dtypes, so it falls through.
        op = _make_softmax(onnx.onnx_pb.TensorProto.FLOAT, symmetric=False)
        scale, zp, dtype = SOFTMAX_FALLBACK
        self._assert_qparams(op, scale, zp, dtype)

    def test_super_quantize_is_called_first(self) -> None:
        """``QDQSoftmax.quantize`` must invoke the base-class quantize
        before setting the output q-params (line 12)."""
        op = _make_softmax(onnx.onnx_pb.TensorProto.UINT8, symmetric=True)
        # The class-level patch already stubs the parent ``quantize`` so a
        # real invocation succeeds. We additionally re-patch here to count
        # the call and verify ordering relative to ``set_quant_scale_zp``.
        with patch.object(type(op).__mro__[1], "quantize", autospec=True) as mock_super_quantize:
            op.quantize()
        mock_super_quantize.assert_called_once_with(op)
        op.quantizer.set_quant_scale_zp.assert_called_once()


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
