#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import numpy as np
import onnx
import onnx.helper as oh
from onnx import TensorProto, numpy_helper
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.onnx.algorithm.mprecision.mixing_strategy import MixingStrategy as QDQMixingStrategy
from quark.onnx.quantization.config.spec import Int8Spec, Int16Spec, QLayerConfig


def _int8_qdq_model() -> onnx.ModelProto:
    """Conv with INT8 Q/DQ; a Transpose sits between the input Q/DQ and Conv.

    Both the pre-Transpose and post-Transpose Q/DQ pairs share the same
    ``act_scale`` / ``act_zp`` initializers, which lets tests exercise the
    ``_unshare_init`` path in :class:`MixingStrategy`.

    Graph::

        X → Q(act_scale, act_zp) → X_q → DQ(act_scale, act_zp) → X_dq
          → Transpose → X_t
          → Q(act_scale, act_zp) → X_t_q → DQ(act_scale, act_zp) → X_t_dq
          → Conv(X_t_dq, W_dq) → Y
        W_quantized → DQ(w_scale, w_zp) → W_dq
    """
    X = oh.make_tensor_value_info("X", TensorProto.FLOAT, [1, 1, 4, 4])
    Y = oh.make_tensor_value_info("Y", TensorProto.FLOAT, None)

    weight_data = np.zeros((1, 1, 3, 3), dtype=np.int8)
    scale_data = np.array(0.1, dtype=np.float32)
    zp_data = np.array(0, dtype=np.int8)

    weight_init = numpy_helper.from_array(weight_data, "W_quantized")
    act_scale = numpy_helper.from_array(scale_data, "act_scale")
    act_zp = numpy_helper.from_array(np.array(0, dtype=np.int8), "act_zp")
    w_scale = numpy_helper.from_array(scale_data, "w_scale")
    w_zp = numpy_helper.from_array(zp_data, "w_zp")

    # Pre-Transpose Q/DQ — share act_scale / act_zp
    q_act = oh.make_node("QuantizeLinear", ["X", "act_scale", "act_zp"], ["X_q"], name="X_QuantizeLinear")
    dq_act = oh.make_node("DequantizeLinear", ["X_q", "act_scale", "act_zp"], ["X_dq"], name="X_DequantizeLinear")
    # Transpose (identity permutation keeps shapes compatible with Conv)
    transpose = oh.make_node("Transpose", ["X_dq"], ["X_t"], name="Transpose", perm=[0, 1, 2, 3])
    # Post-Transpose Q/DQ — share the same act_scale / act_zp
    q_t = oh.make_node("QuantizeLinear", ["X_t", "act_scale", "act_zp"], ["X_t_q"], name="X_t_QuantizeLinear")
    dq_t = oh.make_node("DequantizeLinear", ["X_t_q", "act_scale", "act_zp"], ["X_t_dq"], name="X_t_DequantizeLinear")
    # Weight DQ
    dq_w = oh.make_node("DequantizeLinear", ["W_quantized", "w_scale", "w_zp"], ["W_dq"], name="W_DequantizeLinear")
    # Conv consumes the post-Transpose DQ output
    conv = oh.make_node("Conv", ["X_t_dq", "W_dq"], ["Y"], name="/conv/Conv")

    graph = oh.make_graph(
        [q_act, dq_act, transpose, q_t, dq_t, dq_w, conv],
        "test_graph",
        [X],
        [Y],
        initializer=[weight_init, act_scale, act_zp, w_scale, w_zp],
    )
    model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])
    model = onnx.shape_inference.infer_shapes(model)
    return model


def test_qdq_strategy_promote_changes_weight_zero_point_dtype():
    model = _int8_qdq_model()
    q_model = ONNXModel(model)

    target_config = QLayerConfig(activation=Int16Spec(), weight=Int16Spec())
    strategy = QDQMixingStrategy(target_config)
    strategy.promote(q_model, ["/conv/Conv"], tensors_range=None, layer_config=target_config)

    w_zp = q_model.get_initializer("w_zp")
    assert w_zp is not None
    assert w_zp.data_type == TensorProto.INT16


def test_qdq_strategy_demote_restores_original():
    model = _int8_qdq_model()
    ref_proto = onnx.ModelProto()
    ref_proto.CopyFrom(model)
    q_model = ONNXModel(model)

    target_config = QLayerConfig(activation=Int16Spec(), weight=Int16Spec())
    strategy = QDQMixingStrategy(target_config)
    strategy.promote(q_model, ["/conv/Conv"], tensors_range=None, layer_config=target_config)
    strategy.demote(q_model, ref_proto)

    w_zp = q_model.get_initializer("w_zp")
    assert w_zp.data_type == TensorProto.INT8


def test_qdq_strategy_promote_unshares_shared_initializers():
    """With shared_param_mode='unshare', promoting Conv creates private act_scale_mp /
    act_zp_mp copies for the post-Transpose Q/DQ pair and leaves the pre-Transpose DQ
    still referencing the original act_scale initializer (covers _unshare_init())."""

    class _RangeData:
        def __init__(self, rmin: np.ndarray, rmax: np.ndarray) -> None:
            self.range_value = (rmin, rmax)

    tensors_range = {
        "X_t": _RangeData(
            np.array(-1.0, dtype=np.float32),
            np.array(1.0, dtype=np.float32),
        )
    }

    model = _int8_qdq_model()
    q_model = ONNXModel(model)

    target_config = QLayerConfig(activation=Int8Spec(), weight=Int8Spec())
    strategy = QDQMixingStrategy(target_config, shared_param_mode="unshare")
    strategy.promote(q_model, ["/conv/Conv"], tensors_range=tensors_range, layer_config=target_config)

    init_names = {i.name for i in q_model.model.graph.initializer}
    assert any(n.startswith("act_scale_mp") for n in init_names), "act_scale_mp initializer not created"

    dq_pre = next(n for n in q_model.model.graph.node if n.name == "X_DequantizeLinear")
    assert dq_pre.input[1] == "act_scale", "pre-Transpose DQ should still reference the original act_scale"


def test_qdq_strategy_promote_warns_when_no_target_formats():
    """promote() logs a warning and skips a candidate when the layer config has no
    target formats (all specs None)."""
    import unittest

    model = _int8_qdq_model()
    q_model = ONNXModel(model)

    empty_config = QLayerConfig()
    strategy = QDQMixingStrategy(empty_config)
    with unittest.TestCase().assertLogs(
        "quark.onnx.algorithm.mprecision.mixing_strategy_screen", level="WARNING"
    ) as cm:
        strategy.promote(q_model, ["/conv/Conv"], tensors_range=None, layer_config=empty_config)

    assert any("No target layer types" in m for m in cm.output)


def test_qdq_strategy_promote_propagates_domain_to_shared_nodes():
    """With shared_param_mode='propagate' (default), promoting Conv to INT16 updates the
    op domain of the pre-Transpose Q/DQ pair that shares act_scale/act_zp."""
    from onnxruntime.quantization.quant_utils import ms_domain

    class _RangeData:
        def __init__(self, rmin: np.ndarray, rmax: np.ndarray) -> None:
            self.range_value = (rmin, rmax)

    tensors_range = {
        "X_t": _RangeData(
            np.array(-1.0, dtype=np.float32),
            np.array(1.0, dtype=np.float32),
        )
    }

    model = _int8_qdq_model()
    q_model = ONNXModel(model)

    target_config = QLayerConfig(activation=Int16Spec(), weight=Int16Spec())
    strategy = QDQMixingStrategy(target_config)  # shared_param_mode="propagate" by default
    strategy.promote(q_model, ["/conv/Conv"], tensors_range=tensors_range, layer_config=target_config)

    # _propagate_domain must have updated the pre-Transpose Q node to use ms_domain
    q_pre = next(n for n in q_model.model.graph.node if n.name == "X_QuantizeLinear")
    assert q_pre.domain == ms_domain, "pre-Transpose Q domain should be propagated to ms_domain"
