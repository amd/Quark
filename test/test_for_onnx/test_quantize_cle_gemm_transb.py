#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Regression tests for CLE reading Gemm ``transB`` by name instead of by index.

Context (QUARK-1049): Cross Layer Equalization decided whether to transpose a
Gemm weight by reading ``node.attribute[1]`` and assuming that slot held
``transB``. ONNX does not define an attribute order, so this only worked for one
specific producer. onnxsim >= 0.7.1 fuses MatMul+Add into Gemm with attribute
order ``[alpha, beta, transA, transB]`` -> ``attribute[1]`` is ``beta``, the
check fails, the transpose is skipped, and CLE either crashes on a broadcast or
silently scales along the wrong axis.

CLE is a functionally invariant transform (ReLU is positive-homogeneous), so a
correct implementation leaves the model's outputs unchanged. Each test below
asserts pre-CLE and post-CLE inference agree; wrong-axis scaling or a skipped
transpose would break that equality (or raise).
"""

import unittest

import numpy as np
import onnx
import onnxruntime
from onnx import TensorProto, helper, numpy_helper

from quark.onnx.algorithm.cle.equalization import _get_trans_b, cle_transforms


def _make_gemm_attributes(trans_b: int) -> list[onnx.AttributeProto]:
    """Build Gemm attributes in onnxsim order so attribute[1] is NOT transB."""
    # onnxsim >= 0.7.1 emits Gemm attributes in this insertion order, so
    # attribute[1] is "beta" -- reading transB by index would pick up beta.
    attrs = {"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": trans_b}
    return [helper.make_attribute(name, value) for name, value in attrs.items()]


def _weight_for(trans_b: int, ic: int, oc: int, seed: int) -> np.ndarray:
    """Return a Gemm weight of the storage shape implied by trans_b.

    trans_b == 0 -> Y = X @ B, B stored as (ic, oc).
    trans_b == 1 -> Y = X @ B.T, B stored as (oc, ic).

    Values are scaled per output channel so channel ranges differ enough that
    CLE produces a non-trivial scale (otherwise a wrong axis would go unnoticed).
    """
    rng = np.random.RandomState(seed)
    base = rng.uniform(0.5, 1.5, size=(oc, ic)).astype(np.float32)
    per_oc = (np.arange(1, oc + 1, dtype=np.float32) * 2.0).reshape(oc, 1)
    weight_oc_ic = base * per_oc  # (oc, ic), oc on axis 0
    return weight_oc_ic if trans_b == 1 else weight_oc_ic.T.copy()


def _build_gemm_relu_gemm_model(head_trans_b: int, tail_trans_b: int) -> onnx.ModelProto:
    """Gemm -> Relu -> Gemm, a CLE head/tail pair, with the given transB flags."""
    batch, ic, hidden, oc = 1, 4, 8, 2

    head_w = _weight_for(head_trans_b, ic=ic, oc=hidden, seed=1)
    head_b = np.random.RandomState(2).uniform(-0.5, 0.5, size=(hidden,)).astype(np.float32)
    tail_w = _weight_for(tail_trans_b, ic=hidden, oc=oc, seed=3)
    tail_b = np.random.RandomState(4).uniform(-0.5, 0.5, size=(oc,)).astype(np.float32)

    initializers = [
        numpy_helper.from_array(head_w, "head_w"),
        numpy_helper.from_array(head_b, "head_b"),
        numpy_helper.from_array(tail_w, "tail_w"),
        numpy_helper.from_array(tail_b, "tail_b"),
    ]

    head_gemm = helper.make_node("Gemm", ["input", "head_w", "head_b"], ["head_out"], name="head_gemm")
    head_gemm.attribute.extend(_make_gemm_attributes(head_trans_b))
    relu = helper.make_node("Relu", ["head_out"], ["relu_out"], name="relu")
    tail_gemm = helper.make_node("Gemm", ["relu_out", "tail_w", "tail_b"], ["output"], name="tail_gemm")
    tail_gemm.attribute.extend(_make_gemm_attributes(tail_trans_b))

    graph = helper.make_graph(
        [head_gemm, relu, tail_gemm],
        "gemm_relu_gemm",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [batch, ic])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [batch, oc])],
        initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_operatorsetid("", 17)])
    model.ir_version = 10
    onnx.checker.check_model(model)
    return model


def _run(model: onnx.ModelProto, x: np.ndarray) -> np.ndarray:
    sess = onnxruntime.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return sess.run(None, {"input": x})[0]


class TestCleGemmTransB(unittest.TestCase):
    def test_get_trans_b_lookup_by_name(self) -> None:
        # attribute[1] is "beta", not "transB"; lookup must still find transB.
        node = helper.make_node("Gemm", ["x", "w"], ["y"], name="g")
        node.attribute.extend(_make_gemm_attributes(1))
        self.assertEqual(node.attribute[1].name, "beta")
        self.assertEqual(_get_trans_b(node), 1)

    def test_get_trans_b_defaults_to_zero_when_absent(self) -> None:
        node = helper.make_node("Gemm", ["x", "w"], ["y"], name="g")
        self.assertEqual(_get_trans_b(node), 0)

    def test_onnxsim_order_transb0_does_not_crash_and_is_invariant(self) -> None:
        # Reproduces the JIRA failure: onnxsim attribute order with transB=0.
        # Pre-fix this raised a broadcast ValueError inside _combine_weight_and_bias.
        model = _build_gemm_relu_gemm_model(head_trans_b=0, tail_trans_b=0)
        x = np.random.RandomState(7).uniform(-1.0, 1.0, size=(1, 4)).astype(np.float32)

        golden = _run(model, x)
        equalized = cle_transforms(model, ["Gemm"], [], [], cle_steps=1)
        after = _run(equalized, x)

        np.testing.assert_allclose(after, golden, rtol=1e-4, atol=1e-4)

    def test_head_tail_different_transb_scales_correct_axis(self) -> None:
        # head transB=0 (weight (ic, oc)) vs tail transB=1 (weight (oc, ic)).
        # The old code scaled the head weight using the *tail* node's transB,
        # which picks the wrong reshape axis when the two flags differ.
        model = _build_gemm_relu_gemm_model(head_trans_b=0, tail_trans_b=1)
        x = np.random.RandomState(8).uniform(-1.0, 1.0, size=(1, 4)).astype(np.float32)

        golden = _run(model, x)
        equalized = cle_transforms(model, ["Gemm"], [], [], cle_steps=1)
        after = _run(equalized, x)

        np.testing.assert_allclose(after, golden, rtol=1e-4, atol=1e-4)

    def test_head_transb1_tail_transb0_scales_correct_axis(self) -> None:
        # Mirror of the previous case, flags swapped.
        model = _build_gemm_relu_gemm_model(head_trans_b=1, tail_trans_b=0)
        x = np.random.RandomState(9).uniform(-1.0, 1.0, size=(1, 4)).astype(np.float32)

        golden = _run(model, x)
        equalized = cle_transforms(model, ["Gemm"], [], [], cle_steps=1)
        after = _run(equalized, x)

        np.testing.assert_allclose(after, golden, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
