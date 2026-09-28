#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import numpy as np
import onnx
import pytest

from quark.onnx.algorithm.mprecision.metric_funcs import l2_metric
from quark.onnx.algorithm.mprecision.mixing_strategy import MixingStrategy as QDQMixingStrategy
from quark.onnx.algorithm.mprecision.mprecision_config import AutoMixprecisionConfig
from quark.onnx.algorithm.mprecision.sensitivity_analyzer import (
    SensitivityAnalyzer,
    SensitivityResult,
    load_sensitivity_results,
    save_sensitivity_results,
)
from quark.onnx.quantization.config.spec import Int8Spec, QLayerConfig


def _mock_data_reader():
    class DR:
        def get_next(self):
            return {"input": np.ones((1, 1, 4, 4), dtype=np.float32)}

        def rewind(self):
            pass

    return DR()


def test_sensitivity_result_dataclass():
    r = SensitivityResult(
        name="Conv_0", score=0.5, candidate_nodes=["Conv_0"], all_config_scores=[0.5], best_config_index=0
    )
    assert r.name == "Conv_0"
    assert r.score == 0.5


def test_no_subgraph_specs_uses_layer_analysis():
    cfg = AutoMixprecisionConfig(
        target_layer_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec()),
    )
    target_config = QLayerConfig(activation=Int8Spec(), weight=Int8Spec())
    strategy = QDQMixingStrategy(target_config)
    float_out = [np.zeros((1,), dtype=np.float32)]
    analyzer = SensitivityAnalyzer(cfg, l2_metric, float_out, strategy)
    result = analyzer.analyze(onnx.ModelProto(), _mock_data_reader(), tensors_range=None, subgraph_specs=None)
    assert isinstance(result, list)


def test_save_and_load_sensitivity_results(tmp_path):
    original = [
        SensitivityResult(
            name="Conv_0", score=0.1, candidate_nodes=["Conv_0"], all_config_scores=[0.1], best_config_index=0
        ),
        SensitivityResult(
            name="block_1",
            score=0.5,
            candidate_nodes=["Conv_1", "Conv_2"],
            all_config_scores=[0.5],
            best_config_index=0,
        ),
    ]
    path = tmp_path / "cache.json"
    key = "test_key_abc123"
    save_sensitivity_results(original, path, cache_key=key)
    loaded = load_sensitivity_results(path, valid_key=key)
    assert loaded is not None
    assert len(loaded) == len(original)
    for orig, load in zip(original, loaded, strict=False):
        assert orig.name == load.name
        assert orig.score == load.score
        assert orig.candidate_nodes == load.candidate_nodes


def test_load_sensitivity_results_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_sensitivity_results(tmp_path / "nonexistent.json", valid_key="any")


def test_load_sensitivity_results_stale_key_returns_none(tmp_path):
    import unittest

    # Save with one key, load with a different key — should warn and return None
    # (covers sensitivity_analyzer.py lines 339 and 343).
    results = [
        SensitivityResult(
            name="Conv_0", score=0.1, candidate_nodes=["Conv_0"], all_config_scores=[0.1], best_config_index=0
        )
    ]
    path = tmp_path / "cache.json"
    save_sensitivity_results(results, path, cache_key="original_key")
    with unittest.TestCase().assertLogs(
        "quark.onnx.algorithm.mprecision.sensitivity_analyzer_screen", level="WARNING"
    ) as cm:
        loaded = load_sensitivity_results(path, valid_key="different_key")
    assert loaded is None
    assert any("stale" in m for m in cm.output)


def test_print_sensitivity_table_empty_results():
    # print_sensitivity_table should return None immediately without raising
    # when given an empty list (covers sensitivity_analyzer.py line 218).
    result = SensitivityAnalyzer.print_sensitivity_table([])
    assert result is None
