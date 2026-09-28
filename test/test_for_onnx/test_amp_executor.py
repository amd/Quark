#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import numpy as np
import onnx

from quark.onnx.algorithm.mprecision.mixing_executor import MixingExecutor
from quark.onnx.algorithm.mprecision.mprecision_config import AutoMixprecisionConfig
from quark.onnx.algorithm.mprecision.sensitivity_analyzer import SensitivityResult
from quark.onnx.quantization.config.spec import Int8Spec, QLayerConfig


def _dummy_ranked(n: int) -> list[SensitivityResult]:
    return [
        SensitivityResult(
            name=f"node_{i}",
            score=float(n - i),
            candidate_nodes=[f"node_{i}"],
            all_config_scores=[float(n - i)],
            best_config_index=0,
        )
        for i in range(n)
    ]


def test_executor_promotes_all_under_threshold():
    from unittest.mock import patch

    cfg = AutoMixprecisionConfig(
        target_layer_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec()),
        metric_threshold=10.0,
    )
    promoted = []

    class CountingStrategy:
        target_config_list = [QLayerConfig(activation=Int8Spec(), weight=Int8Spec())]

        def promote(self, work_model, candidate_nodes, tensors_range, layer_config):
            promoted.extend(candidate_nodes)
            return set()

        def demote(self, work_model, prev_proto):
            pass

    float_out = [np.zeros((1,), dtype=np.float32)]
    executor = MixingExecutor(cfg, metric_fn=lambda f, q: 0.0, float_out=float_out, strategy=CountingStrategy())
    ranked = _dummy_ranked(5)

    dummy_out = [np.zeros((1,), dtype=np.float32)]
    with patch("quark.onnx.algorithm.mprecision.mixing_executor.inference_model", return_value=dummy_out):
        executor.execute(
            quant_model=onnx.ModelProto(),
            data_reader=None,
            tensors_range=None,
            ranked=ranked,
        )
    assert len(promoted) == 5


def test_executor_promotes_all_when_threshold_is_zero():
    """metric_threshold=0 disables the threshold; all candidates are promoted even with a high score."""
    from unittest.mock import patch

    cfg = AutoMixprecisionConfig(
        target_layer_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec()),
        metric_threshold=0,
    )
    promoted = []

    class CountingStrategy:
        target_config_list = [QLayerConfig(activation=Int8Spec(), weight=Int8Spec())]

        def promote(self, work_model, candidate_nodes, tensors_range, layer_config):
            promoted.extend(candidate_nodes)
            return set()

        def demote(self, work_model, prev_proto):
            pass

    float_out = [np.zeros((1,), dtype=np.float32)]
    # Return a very high score — with threshold=0 it should still promote all candidates.
    executor = MixingExecutor(cfg, metric_fn=lambda f, q: 9999.0, float_out=float_out, strategy=CountingStrategy())
    ranked = _dummy_ranked(5)

    dummy_out = [np.zeros((1,), dtype=np.float32)]
    with patch("quark.onnx.algorithm.mprecision.mixing_executor.inference_model", return_value=dummy_out):
        executor.execute(
            quant_model=onnx.ModelProto(),
            data_reader=None,
            tensors_range=None,
            ranked=ranked,
        )
    assert len(promoted) == 5


def test_promoted_nodes_and_tensors_properties():
    """promoted_nodes and promoted_tensors properties are accessible after execute()
    (covers mixing_executor.py lines 60 and 65)."""
    from unittest.mock import patch

    cfg = AutoMixprecisionConfig(
        target_layer_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec()),
        metric_threshold=0,
    )

    class TrackingStrategy:
        target_config_list = [QLayerConfig(activation=Int8Spec(), weight=Int8Spec())]

        def promote(self, work_model, candidate_nodes, tensors_range, layer_config):
            return {"tensor_a"}

        def demote(self, work_model, prev_proto):
            pass

    float_out = [np.zeros((1,), dtype=np.float32)]
    executor = MixingExecutor(cfg, metric_fn=lambda f, q: 0.0, float_out=float_out, strategy=TrackingStrategy())
    ranked = _dummy_ranked(2)

    with patch("quark.onnx.algorithm.mprecision.mixing_executor.inference_model", return_value=float_out):
        executor.execute(onnx.ModelProto(), data_reader=None, tensors_range=None, ranked=ranked)

    assert isinstance(executor.promoted_nodes, set)
    assert isinstance(executor.promoted_tensors, set)


def test_disabled_candidate_is_skipped():
    """Candidates with enabled=False must be skipped and logged
    (covers mixing_executor.py lines 152-153)."""
    import unittest
    from unittest.mock import patch

    cfg = AutoMixprecisionConfig(
        target_layer_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec()),
        metric_threshold=0,
    )
    promoted = []

    class CountingStrategy:
        target_config_list = [QLayerConfig(activation=Int8Spec(), weight=Int8Spec())]

        def promote(self, work_model, candidate_nodes, tensors_range, layer_config):
            promoted.extend(candidate_nodes)
            return set()

        def demote(self, work_model, prev_proto):
            pass

    float_out = [np.zeros((1,), dtype=np.float32)]
    executor = MixingExecutor(cfg, metric_fn=lambda f, q: 0.0, float_out=float_out, strategy=CountingStrategy())

    ranked = [
        SensitivityResult(
            name="node_0", score=1.0, candidate_nodes=["node_0"], all_config_scores=[1.0], best_config_index=0
        ),
        SensitivityResult(
            name="node_1",
            score=2.0,
            candidate_nodes=["node_1"],
            all_config_scores=[2.0],
            best_config_index=0,
            enabled=False,
        ),
    ]

    with (
        patch("quark.onnx.algorithm.mprecision.mixing_executor.inference_model", return_value=float_out),
        unittest.TestCase().assertLogs("quark.onnx.algorithm.mprecision.mixing_executor_screen", level="INFO") as cm,
    ):
        executor.execute(onnx.ModelProto(), data_reader=None, tensors_range=None, ranked=ranked)

    assert "node_1" not in promoted
    assert "node_0" in promoted
    assert any("disabled" in m for m in cm.output)


def test_executor_strategy_selection_qdq_for_int_types():
    from quark.onnx.algorithm.mprecision.mixing_strategy import MixingStrategy as QDQMixingStrategy

    target_config = QLayerConfig(activation=Int8Spec(), weight=Int8Spec())
    cfg = AutoMixprecisionConfig(target_layer_config=target_config)
    float_out = [np.zeros((1,), dtype=np.float32)]
    executor = MixingExecutor(
        cfg,
        metric_fn=lambda f, q: 0.0,
        float_out=float_out,
        strategy=QDQMixingStrategy(target_config),
    )
    assert isinstance(executor._strategy, QDQMixingStrategy)
