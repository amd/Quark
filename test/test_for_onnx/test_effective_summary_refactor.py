#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Pytest suite for the QUARK-536 effective-summary refactor."""

import logging

import pytest
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantFormat, QuantType

from quark.onnx.calibration import LayerWiseMethod, PowerOfTwoMethod
from quark.onnx.calibration.calibrators import resolve_calibrator_extra_defaults
from quark.onnx.quantization.quant_utils import ExtendedQuantFormat, ExtendedQuantType
from quark.onnx.utils.print_utils import (
    _CALIB_KEY_TO_LOWER,
    _QCFG_CATEGORIES,
    _resolve_active_quantizer,
    print_effective_quantization_summary,
    print_user_supplied_configuration,
)


def _capture(func, **call_kwargs) -> str:
    from quark.common.utils.log import ScreenLogger
    from quark.onnx.utils import print_utils

    target = ScreenLogger(print_utils.__name__).logger
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[assignment]
    handler.setLevel(logging.DEBUG)
    target.addHandler(handler)
    prev_level = target.level
    target.setLevel(logging.DEBUG)
    try:
        func(**call_kwargs)
    finally:
        target.removeHandler(handler)
        target.setLevel(prev_level)
    return "\n".join(r.getMessage() for r in records)


# ---------------------------------------------------------------------------
# resolve_calibrator_extra_defaults
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, expected",
    [
        (
            CalibrationMethod.MinMax,
            {"symmetric": False, "moving_average": False, "averaging_constant": 0.01, "optimize_mem": True},
        ),
        (CalibrationMethod.Percentile, {"symmetric": True, "percentile": 99.999}),
        (PowerOfTwoMethod.MinMSE, {"symmetric": True, "percentile": 99.999, "minmse_mode": "All"}),
        (
            LayerWiseMethod.LayerWisePercentile,
            {
                "symmetric": True,
                "percentile": 99.999,
                "optimize_disk": True,
                "optimize_mem": False,
                "percentile_candidates": [99.99, 99.999, 99.99999],
                "lwp_metric": "mae",
            },
        ),
    ],
)
def test_calibrator_defaults(method, expected):
    overlay = resolve_calibrator_extra_defaults(method, {}, emit_warnings=False)
    for k, v in expected.items():
        assert overlay[k] == v


def test_user_override_wins():
    overlay = resolve_calibrator_extra_defaults(CalibrationMethod.Percentile, {"percentile": 99.99, "symmetric": False})
    assert overlay["percentile"] == 99.99
    assert overlay["symmetric"] is False


def test_unknown_method_returns_empty():
    assert resolve_calibrator_extra_defaults("not-a-real-method", {}) == {}


def test_lwp_disk_mem_mutex_warns():
    # Exercise the mutex + warning branch.
    # ScreenLogger doesn't propagate to root, so attach a handler directly.
    target = logging.getLogger("quark.onnx.calibration.calibrators_screen")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[assignment]
    target.addHandler(handler)
    try:
        overlay = resolve_calibrator_extra_defaults(
            LayerWiseMethod.LayerWisePercentile,
            {"optimize_disk": True, "optimize_mem": True},
            emit_warnings=True,
        )
    finally:
        target.removeHandler(handler)
    assert overlay["optimize_mem"] is False
    assert overlay["optimize_disk"] is True
    assert any("CalibOptimizeMem is forced to be False" in r.getMessage() for r in records)


# ---------------------------------------------------------------------------
# _resolve_active_quantizer
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "quant_format, calibrate_method, enable_npu_cnn, expected",
    [
        (QuantFormat.QDQ, CalibrationMethod.MinMax, False, "BaseExtendedQDQQuantizer"),
        (ExtendedQuantFormat.QDQ, CalibrationMethod.MinMax, False, "ExtendedQDQQuantizer"),
        (QuantFormat.QDQ, PowerOfTwoMethod.MinMSE, True, "XINT8QDQQuantizer"),
        (QuantFormat.QOperator, CalibrationMethod.MinMax, False, "ONNXQuantizer"),
        (QuantFormat.QOperator, PowerOfTwoMethod.MinMSE, False, "ExtendedONNXQuantizer"),
        # No matching dispatch → empty string (mirrors create_static_quantizer
        # raising in this case; summary degrades gracefully without annotations).
        ("not-a-quant-format", CalibrationMethod.MinMax, False, ""),
    ],
)
def test_resolve_active_quantizer(quant_format, calibrate_method, enable_npu_cnn, expected):
    assert _resolve_active_quantizer(quant_format, calibrate_method, enable_npu_cnn) == expected


# ---------------------------------------------------------------------------
# print_user_supplied_configuration
# ---------------------------------------------------------------------------


def _user_kwargs(**overrides):
    base = dict(
        q_config_source=None,
        user_extra={"EnableNPUCnn": True, "SimplifyModel": False, "Int32Bias": False},
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=PowerOfTwoMethod.MinMSE,
        optimize_model=False,
        model_input="dummy.onnx",
        model_output="out.onnx",
        calibration_data_reader=None,
    )
    base.update(overrides)
    return base


def test_user_supplied_header_and_raw_extra():
    out = _capture(print_user_supplied_configuration, **_user_kwargs())
    assert "Quantized Configuration information:" in out
    assert "'SimplifyModel': False" in out
    assert "'Int32Bias': False" in out
    assert "active_quantizer --- XINT8QDQQuantizer" in out
    # The categorized effective summary banner must NOT appear here — this
    # helper prints the pre-quantization user-supplied block only.
    assert "categories below reflect" not in out


@pytest.mark.parametrize("user_extra", [{"CryptoMode": True}, {"PrintSummary": False}])
def test_user_supplied_skips(user_extra):
    out = _capture(print_user_supplied_configuration, **_user_kwargs(user_extra=user_extra))
    assert "Quantized Configuration information" not in out


def test_q_config_source_dedupe_skips_overlap():
    # Dataclass fields that overlap with main_rows or 'extra_options' must not
    # produce duplicate rows when q_config_source is supplied.
    from quark.onnx.quantization.config.legacy import QuantizationConfig

    out = _capture(print_user_supplied_configuration, **_user_kwargs(q_config_source=QuantizationConfig()))
    assert "Quantized Configuration information:" in out
    assert "per_channel" in out
    assert out.count("optimize_model ---") == 1


# ---------------------------------------------------------------------------
# print_effective_quantization_summary
# ---------------------------------------------------------------------------


def _eff_kwargs(**overrides):
    base = dict(
        user_extra={},
        effective_extra={},
        effective_quant_format=QuantFormat.QDQ,
        effective_calibrate_method=CalibrationMethod.MinMax,
        effective_activation_type=QuantType.QInt8,
        effective_weight_type=QuantType.QInt8,
    )
    base.update(overrides)
    return base


def test_effective_success_banner():
    out = _capture(print_effective_quantization_summary, **_eff_kwargs())
    assert "effective extra_options consumed by the active quantizer" in out
    assert "at the point of failure" not in out


def test_effective_failure_banner_tags_exception():
    out = _capture(print_effective_quantization_summary, **_eff_kwargs(exception_context="RuntimeError"))
    assert "at the point of failure" in out
    assert "exception: RuntimeError" in out
    assert "1. Preprocessing & Graph Optimization" in out
    assert "11. Debug, Logging & Evaluation" in out


@pytest.mark.parametrize(
    "method, expect_substrings",
    [
        (CalibrationMethod.MinMax, ["CalibTensorRangeSymmetric --- False"]),
        (CalibrationMethod.Percentile, ["CalibTensorRangeSymmetric --- True", "NumBins --- 2048"]),
        (CalibrationMethod.Entropy, ["NumBins --- 128", "NumQuantizedBins --- 128"]),
        (CalibrationMethod.Distribution, ["NumBins --- 2048", "Scenario --- same"]),
    ],
)
def test_calibrator_defaults_propagate_to_summary(method, expect_substrings):
    out = _capture(print_effective_quantization_summary, **_eff_kwargs(effective_calibrate_method=method))
    for s in expect_substrings:
        assert s in out


def test_lwp_defaults_show_in_summary():
    out = _capture(
        print_effective_quantization_summary,
        **_eff_kwargs(
            effective_quant_format=ExtendedQuantFormat.QDQ,
            effective_calibrate_method=LayerWiseMethod.LayerWisePercentile,
        ),
    )
    for s in [
        "CalibTensorRangeSymmetric --- True",
        "CalibOptimizeDisk --- True",
        "CalibOptimizeMem --- False",
        "Percentile --- 99.999",
        "PercentileCandidates --- [99.99, 99.999, 99.99999]",
    ]:
        assert s in out


def test_npu_only_options_quiet_when_default():
    out = _capture(print_effective_quantization_summary, **_eff_kwargs())
    assert "only takes effect under" not in out
    # User did not set it AND quantizer does not consume it → row is omitted.
    assert "SimulateDPU" not in out


def test_npu_only_options_loud_when_user_set():
    keys = {"SimulateDPU": True, "AdjustShiftBias": True, "AdjustBiasScale": True}
    out = _capture(print_effective_quantization_summary, **_eff_kwargs(user_extra=keys, effective_extra=keys))
    expected = {
        "SimulateDPU": "NPU-CNN (XINT8 QDQ) or Extended QDQ",
        "AdjustShiftBias": "NPU-CNN (XINT8 QDQ)",
        "AdjustBiasScale": "Extended QDQ",
    }
    for k, label in expected.items():
        assert f"{k} --- True  (only takes effect under {label} config)" in out


def test_copy_bias_init_active_on_int8_minmax():
    out = _capture(print_effective_quantization_summary, **_eff_kwargs())
    assert "CopyBiasInit --- ['Conv', 'ConvTranspose', 'Gemm']" in out
    assert "CopyBiasInit --- ['Conv', 'ConvTranspose', 'Gemm']  (no-op" not in out


@pytest.mark.parametrize(
    "override",
    [
        {"effective_weight_type": ExtendedQuantType.QBFP},
        {"effective_calibrate_method": LayerWiseMethod.LayerWisePercentile},
    ],
    ids=["non-int-dtype", "non-native-calibrate-method"],
)
def test_copy_bias_init_no_op_skipped_when_default(override):
    # User did not set CopyBiasInit AND it would be a no-op → row omitted entirely.
    out = _capture(print_effective_quantization_summary, **_eff_kwargs(**override))
    assert "CopyBiasInit" not in out


@pytest.mark.parametrize(
    "override",
    [
        {"effective_weight_type": ExtendedQuantType.QBFP},
        {"effective_calibrate_method": LayerWiseMethod.LayerWisePercentile},
    ],
    ids=["non-int-dtype", "non-native-calibrate-method"],
)
def test_copy_bias_init_no_op_loud_when_user_set(override):
    out = _capture(
        print_effective_quantization_summary,
        **_eff_kwargs(user_extra={"CopyBiasInit": ["Conv"]}, effective_extra={"CopyBiasInit": ["Conv"]}, **override),
    )
    assert (
        "CopyBiasInit --- ['Conv']  (only takes effect under int8/int16 dtypes with "
        "calibrate_method in {MinMax, Entropy, Percentile, Distribution})"
    ) in out


def test_save_and_restore_from_deprecated_alias():
    out = _capture(
        print_effective_quantization_summary,
        **_eff_kwargs(
            user_extra={"TensorsRangeFile": "foo.pkl"},
            effective_extra={"SaveAndRestore": "foo.pkl", "TensorsRangeFile": "foo.pkl"},
        ),
    )
    assert "SaveAndRestore --- foo.pkl  (from deprecated TensorsRangeFile alias)" in out


def test_save_and_restore_direct_no_alias_tag():
    out = _capture(
        print_effective_quantization_summary,
        **_eff_kwargs(
            user_extra={"SaveAndRestore": "bar.pkl"},
            effective_extra={"SaveAndRestore": "bar.pkl"},
        ),
    )
    assert "SaveAndRestore --- bar.pkl" in out
    assert "(from deprecated TensorsRangeFile alias)" not in out


def test_matmul_nbits_params_not_expanded_when_gate_off():
    out = _capture(
        print_effective_quantization_summary,
        **_eff_kwargs(effective_extra={"UseMatMulNBits": False, "MatMulNBitsParams": {}}),
    )
    assert "MatMulNBitsParams --- {}" in out


@pytest.mark.parametrize("effective_extra", [{"CryptoMode": True}, {"PrintSummary": False}])
def test_effective_summary_skips(effective_extra):
    out = _capture(print_effective_quantization_summary, **_eff_kwargs(effective_extra=effective_extra))
    assert "categories below reflect" not in out


# ---------------------------------------------------------------------------
# Schema invariants
# ---------------------------------------------------------------------------


def test_format_param_value_dumps_algo_config():
    # An algorithm config object must be expanded via _dump_algo_config_full.
    from quark.onnx.quantization.config.algorithm import CLEConfig
    from quark.onnx.utils.print_utils import _format_param_value

    text = _format_param_value(CLEConfig())
    assert "'name': 'cle'" in text


def test_format_param_value_falls_back_to_repr_on_pformat_failure():
    # When config_to_dict raises, _format_param_value must fall back to repr.
    # A self-referencing list makes config_to_dict recurse past Python's
    # recursion limit (RecursionError is an Exception subclass).
    from quark.onnx.utils.print_utils import _format_param_value

    cyclic: list = []
    cyclic.append(cyclic)
    text = _format_param_value(cyclic)
    assert text == repr(cyclic)


def test_categories_use_3_tuple_schema():
    for _title, _description, items in _QCFG_CATEGORIES:
        for item in items:
            assert len(item) == 3, f"Category item must be 3-tuple, got {item!r}"
            _key, _default, consumed_by = item
            if consumed_by is not None:
                assert isinstance(consumed_by, frozenset)


def test_calib_key_to_lower_mapping_subset_of_known_keys():
    all_keys = {k for _t, _d, items in _QCFG_CATEGORIES for (k, _, _) in items}
    for pascal_key, _lower in _CALIB_KEY_TO_LOWER:
        assert pascal_key in all_keys


# ---------------------------------------------------------------------------
# quantize_static failure-path summary
# ---------------------------------------------------------------------------


def _build_tiny_onnx_model(path: str) -> None:
    import numpy as np
    import torch
    import torch.nn as nn

    class _M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 1, 3, padding=1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.conv(x.to(torch.float))

    torch.manual_seed(0)
    torch.onnx.export(
        _M(),
        torch.randn(1, 3, 4, 4),
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        dynamo=False,
    )
    # Touch numpy so the import is not flagged unused on systems where the
    # data reader below is the only consumer.
    _ = np.float32


class _OneShotReader:
    def __init__(self) -> None:
        import numpy as np

        self._data = [{"input": np.random.rand(1, 3, 4, 4).astype("float32")}]
        self._i = 0

    def get_next(self):
        if self._i >= len(self._data):
            return None
        item = self._data[self._i]
        self._i += 1
        return item

    def rewind(self) -> None:
        self._i = 0


def test_quantize_static_failure_prints_summary_with_exception_context(tmp_path, monkeypatch):
    """The failure path must tag the effective summary with the exception
    class name and re-raise the original exception unchanged."""
    from quark.onnx import Int8Spec, ModelQuantizer, QConfig, QLayerConfig, UInt8Spec
    from quark.onnx.quantization import quantize as quantize_mod
    from quark.onnx.utils import print_utils

    model_in = str(tmp_path / "tiny.onnx")
    model_out = str(tmp_path / "tiny_quant.onnx")
    _build_tiny_onnx_model(model_in)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("simulated quantizer crash")

    monkeypatch.setattr(quantize_mod, "run_static_quantization", _boom)

    seen: list[dict] = []

    real_printer = print_utils.print_effective_quantization_summary

    def _spy(**kwargs):
        seen.append(kwargs)
        return real_printer(**kwargs)

    monkeypatch.setattr(print_utils, "print_effective_quantization_summary", _spy)
    monkeypatch.setattr(quantize_mod, "print_effective_quantization_summary", _spy)

    quantizer = ModelQuantizer(QConfig(global_config=QLayerConfig(activation=UInt8Spec(), weight=Int8Spec())))

    with pytest.raises(RuntimeError, match="simulated quantizer crash"):
        quantizer.quantize_model(model_in, model_out, _OneShotReader())

    assert seen, "effective summary printer must be invoked on the failure path"
    assert seen[-1].get("exception_context") == "RuntimeError"
