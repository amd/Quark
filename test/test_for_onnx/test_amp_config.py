#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import pytest

from quark.onnx.algorithm.mprecision.mprecision_config import AutoMixprecisionConfig
from quark.onnx.quantization.config.spec import Int8Spec, QLayerConfig


def _make_config(**kwargs) -> AutoMixprecisionConfig:
    return AutoMixprecisionConfig(
        target_layer_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec()),
        **kwargs,
    )


def test_default_target_op_types():
    cfg = _make_config()
    assert "Conv" in cfg.target_op_type
    assert "MatMul" in cfg.target_op_type


def test_get_config_contains_required_keys():
    cfg = _make_config(metric_threshold=0.3, data_size=10)
    result = cfg._get_config({})
    amp = result["AutoMixprecision"]
    assert amp["MetricThreshold"] == 0.3
    assert amp["DataSize"] == 10


def test_metric_threshold_none_stored():
    cfg = _make_config(metric_threshold=None)
    assert cfg.metric_threshold is None
    result = cfg._get_config({})
    assert result["AutoMixprecision"]["MetricThreshold"] is None


def test_get_config_skips_keys_already_in_extra_options():
    # _get_config only emits keys that are NOT already present in extra_options.
    # A key supplied by the caller is left in extra_options untouched and is
    # absent from the returned dict — it is not overwritten.
    cfg = _make_config(data_size=10)
    extra = {"AutoMixprecision": {"DataSize": 999}}
    result = cfg._get_config(extra)
    assert "DataSize" not in result["AutoMixprecision"]
    # The caller's value in extra_options is preserved unchanged.
    assert extra["AutoMixprecision"]["DataSize"] == 999


def test_target_layer_config_none_is_stored():
    # AutoMixprecisionConfig does not validate target_layer_config at
    # construction or in _get_config; None is stored and emitted as-is.
    cfg = AutoMixprecisionConfig(target_layer_config=None)
    result = cfg._get_config({})
    assert result["AutoMixprecision"]["TargetLayerConfig"] is None


def test_name_attribute():
    cfg = _make_config()
    assert cfg.name == "auto_mixprecision"


def test_worker_num_default_is_one():
    cfg = _make_config()
    assert cfg.worker_num == 1


def test_worker_num_custom():
    cfg = _make_config(worker_num=4)
    assert cfg.worker_num == 4
    result = cfg._get_config({})
    assert result["AutoMixprecision"]["WorkerNum"] == 4


def test_metric_optimize_object_invalid_raises():
    with pytest.raises(ValueError, match="metric_optimize_object"):
        _make_config(metric_optimize_object="invalid")


def test_metric_optimize_object_valid_values():
    for direction in ("speed", "quality"):
        cfg = _make_config(metric_optimize_object=direction)
        assert cfg.metric_optimize_object == direction


def test_shared_param_mode_invalid_raises():
    with pytest.raises(ValueError, match="shared_param_mode"):
        _make_config(shared_param_mode="invalid")


def test_shared_param_mode_valid_values():
    for mode in ("propagate", "unshare"):
        cfg = _make_config(shared_param_mode=mode)
        assert cfg.shared_param_mode == mode


def test_from_extra_options_missing_target_layer_config_warns():
    import unittest

    # _from_extra_options with no TargetLayerConfig should log a warning and still
    # return a config with target_layer_config=None.
    with unittest.TestCase().assertLogs("quark.onnx.quantization.config.algorithm_screen", level="WARNING") as cm:
        cfg = AutoMixprecisionConfig._from_extra_options({"AutoMixprecision": {}})
    assert cfg.target_layer_config is None
    assert any("target_layer_config is required" in m for m in cm.output)


def test_auto_mixprecision_returns_quant_model_when_no_data_reader():
    """auto_mixprecision warns and returns the quantized model unchanged when
    data_reader is None."""
    import unittest

    import onnx

    from quark.onnx.algorithm.mprecision.auto_mixprecision import auto_mixprecision

    quant_proto = onnx.ModelProto()
    with unittest.TestCase().assertLogs(
        "quark.onnx.algorithm.mprecision.auto_mixprecision_screen", level="WARNING"
    ) as cm:
        result = auto_mixprecision(
            float_model=onnx.ModelProto(),
            quant_model=quant_proto,
            use_external_data_format=False,
            data_reader=None,
            tensors_range=None,
            extra_options={},
        )
    assert result is quant_proto
    assert any("No data reader" in m for m in cm.output)


def test_auto_mixprecision_returns_quant_model_when_no_target_layer_config():
    """auto_mixprecision warns and returns the quantized model unchanged when
    target_layer_config is absent from extra_options."""
    import unittest

    import onnx

    from quark.onnx.algorithm.mprecision.auto_mixprecision import auto_mixprecision

    quant_proto = onnx.ModelProto()
    with unittest.TestCase().assertLogs(
        "quark.onnx.algorithm.mprecision.auto_mixprecision_screen", level="WARNING"
    ) as cm:
        result = auto_mixprecision(
            float_model=onnx.ModelProto(),
            quant_model=quant_proto,
            use_external_data_format=False,
            data_reader=object(),  # non-None, passes the data_reader check
            tensors_range=None,
            extra_options={"AutoMixprecision": {}},  # no TargetLayerConfig → target_layer_config=None
        )
    assert result is quant_proto
    assert any("No target_layer_config" in m for m in cm.output)
