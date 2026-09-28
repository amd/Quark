#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Unit tests for the AdjustWeightScaleForInt32Bias extra_option.

The option is gated inside BaseExtendedQDQQuantizer.quantize_bias_static()
(quark/onnx/quantizers/qdq_quantizer.py) and targets A16W8 + Int32Bias configs.

Test strategy
-------------
Build a minimal Conv model whose bias is deliberately large (1e5) while the
Conv weights are tiny (~0.01).  Under A16W8 calibration the resulting
bias_scale = input_scale * weight_scale will be on the order of 1e-9, causing
bias / bias_scale >> INT32_MAX (~2.1e9).  Without the option those channels
saturate at INT32_MIN / INT32_MAX; with the option the weight scale is inflated
just enough to keep every channel within INT32 range.

The additional TestAdjustWeightScaleUnit and TestQuantizeBiasStaticOverride
classes call the internal methods directly via MagicMock to cover every guard
branch and the full re-quantization path.
"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import onnx
import onnx.numpy_helper
from onnx import TensorProto, helper
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.qdq_quantizer import QDQBiasQuantInfo, QDQTensorQuantizedValue
from onnxruntime.quantization.quant_utils import QuantizedValue, QuantizedValueType

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import Int8Spec, Int16Spec, ModelQuantizer, QConfig, QLayerConfig
from quark.onnx.quantization.config.spec import CalibMethod, Int32Spec
from quark.onnx.quantizers.qdq_quantizer import BaseExtendedQDQQuantizer

# -----------------------------------------------------------------------
# Model geometry
# -----------------------------------------------------------------------
INPUT_SHAPE = [1, 1, 8, 8]
WEIGHT_SHAPE = [4, 1, 3, 3]
BIAS_SHAPE = [4]
LARGE_BIAS_VALUE = 1e5  # large enough to overflow INT32 with tiny weight_scale


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
class _SingleInputReader(CalibrationDataReader):
    def __init__(self, data: dict):
        self._pool = [data]
        self._iter = iter(self._pool)

    def get_next(self):
        return next(self._iter, None)

    def rewind(self):
        self._iter = iter(self._pool)


def _build_model(output_dir: str) -> str:
    """Build a one-Conv ONNX model with a deliberately large bias."""
    rng = np.random.default_rng(seed=7)
    # Very small weights -> very small weight_scale -> bias easily overflows INT32
    weight = (rng.random(WEIGHT_SHAPE) * 0.01).astype(np.float32)
    bias = np.full(BIAS_SHAPE, LARGE_BIAS_VALUE, dtype=np.float32)

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, INPUT_SHAPE)
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8, 8])
    w_init = helper.make_tensor("w", TensorProto.FLOAT, WEIGHT_SHAPE, weight.flatten().tolist())
    b_init = helper.make_tensor("b", TensorProto.FLOAT, BIAS_SHAPE, bias.tolist())
    conv = helper.make_node("Conv", inputs=["x", "w", "b"], outputs=["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])

    graph = helper.make_graph([conv], "overflow_test", [x], [y], initializer=[w_init, b_init])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], ir_version=10)

    path = Path(output_dir, "float_model.onnx").as_posix()
    onnx.save(model, path)
    return path


def _make_calib_reader() -> _SingleInputReader:
    rng = np.random.default_rng(seed=42)
    return _SingleInputReader({"x": rng.random(INPUT_SHAPE).astype(np.float32)})


def _make_config(adjust: bool) -> QConfig:
    # Use MinMax calibration to avoid histogram-bin issues when bias dominates
    # the activation range (output ~= LARGE_BIAS_VALUE with tiny conv weights).
    return QConfig(
        global_config=QLayerConfig(
            activation=Int16Spec(calibration_method=CalibMethod.MinMax),
            weight=Int8Spec(calibration_method=CalibMethod.MinMax),
            bias=Int32Spec(),
        ),
        extra_options={
            "ForceQuantizeNoInputCheck": True,
            "Int32Bias": True,
            "QuantizeBias": True,
            "AdjustWeightScaleForInt32Bias": adjust,
        },
    )


def _count_int32_overflows(model_path: str) -> int:
    """Count INT32 bias initializer channels saturated at INT32_MIN or INT32_MAX."""
    model = onnx.load(model_path)
    int32_min = np.iinfo(np.int32).min
    int32_max = np.iinfo(np.int32).max
    total = 0
    for init in model.graph.initializer:
        if init.data_type != onnx.TensorProto.INT32:
            continue
        arr = onnx.numpy_helper.to_array(init)
        total += int(np.sum(arr == int32_min) + np.sum(arr == int32_max))
    return total


def _quantize(model_path: str, out_path: str, adjust: bool) -> str:
    ModelQuantizer(_make_config(adjust)).quantize_model(model_path, out_path, _make_calib_reader())
    return out_path


# -----------------------------------------------------------------------
# Integration tests
# -----------------------------------------------------------------------
class TestAdjustWeightScaleForInt32Bias(unittest.TestCase):
    @use_temporary_directory
    def test_overflow_present_without_adjustment(self, tmpdir: str):
        """Without the option, the large bias saturates at least one INT32 channel."""
        model_path = _build_model(tmpdir)
        out_path = Path(tmpdir, "quantized_no_adjust.onnx").as_posix()
        _quantize(model_path, out_path, adjust=False)
        overflows = _count_int32_overflows(out_path)
        self.assertGreater(
            overflows,
            0,
            "Expected >=1 saturated INT32 channel when AdjustWeightScaleForInt32Bias=False",
        )

    @use_temporary_directory
    def test_no_overflow_with_adjustment(self, tmpdir: str):
        """With the option enabled, all INT32 bias channels must be within representable range."""
        model_path = _build_model(tmpdir)
        out_path = Path(tmpdir, "quantized_adjusted.onnx").as_posix()
        _quantize(model_path, out_path, adjust=True)
        overflows = _count_int32_overflows(out_path)
        self.assertEqual(
            overflows,
            0,
            "Expected zero saturated INT32 channels when AdjustWeightScaleForInt32Bias=True",
        )


# -----------------------------------------------------------------------
# Unit helpers
# -----------------------------------------------------------------------
def _f32(name: str, arr) -> onnx.TensorProto:
    return onnx.numpy_helper.from_array(np.asarray(arr, dtype=np.float32), name)


def _i8(name: str, arr) -> onnx.TensorProto:
    return onnx.numpy_helper.from_array(np.asarray(arr, dtype=np.int8), name)


def _make_qv(
    orig_name: str,
    q_name: str = "w_q",
    scale_name: str = "w_scale",
    zp_name: str = "w_zp",
    axis: int = 0,
) -> QDQTensorQuantizedValue:
    qval = QuantizedValue(orig_name, q_name, scale_name, zp_name, QuantizedValueType.Initializer, axis)
    return QDQTensorQuantizedValue(qval, None, None)


def _mock_quantizer(initializers, qv_map=None, weight_qtype=onnx.TensorProto.INT8, ort_result=None):
    mock_q = MagicMock()
    mock_q.model.initializer.return_value = list(initializers)
    mock_q.quantized_value_map = dict(qv_map) if qv_map is not None else {}
    mock_q.weight_qType = weight_qtype
    if ort_result is not None:
        mock_q._adjust_weight_scale_for_int32_bias.return_value = ort_result
    return mock_q


def _call_adjust(mock_q, bias_name, weight_name, inp_s, ws, beta=1.0):
    return BaseExtendedQDQQuantizer._adjust_weight_scale_for_int32bias_overflow(
        mock_q,
        bias_name,
        weight_name,
        np.asarray(inp_s, dtype=np.float32),
        np.asarray(ws, dtype=np.float32),
        beta,
    )


# -----------------------------------------------------------------------
# Unit tests: _adjust_weight_scale_for_int32bias_overflow
# -----------------------------------------------------------------------
class TestAdjustWeightScaleUnit(unittest.TestCase):
    """Direct unit tests for each guard branch and the full re-quantization path."""

    def test_bias_not_found_returns_original_scale(self):
        """Lines 879-881: unknown bias name -> weight_scale returned unchanged (ORT not called)."""
        ws = np.array([0.01], dtype=np.float32)
        mock_q = _mock_quantizer([])
        result = _call_adjust(mock_q, "missing", "w", [1.0], ws)
        np.testing.assert_array_equal(result, ws)
        mock_q._adjust_weight_scale_for_int32_bias.assert_not_called()

    def test_ort_no_update_returns_original_scale(self):
        """Lines 892-893: ORT reports no overflow -> weight_scale returned unchanged."""
        ws = np.array([1.0], dtype=np.float32)
        result = _call_adjust(
            _mock_quantizer([_f32("b", [1.0])], ort_result=(False, None)),
            "b",
            "w",
            [1.0],
            ws,
        )
        np.testing.assert_array_equal(result, ws)

    def test_grouped_conv_scale_size_mismatch_returns_original(self):
        """Grouped conv: weight_scale.size > 1 and != bias channels -> original scale, ORT not called."""
        ws = np.array([0.01] * 16, dtype=np.float32)  # 16 elements
        mock_q = _mock_quantizer([_f32("b", [1.0, 2.0, 3.0, 4.0])])  # bias has 4 channels
        result = _call_adjust(mock_q, "b", "w", [1.0], ws)
        np.testing.assert_array_equal(result, ws)
        mock_q._adjust_weight_scale_for_int32_bias.assert_not_called()

    def test_new_scale_size_mismatch_with_model_returns_original(self):
        """ORT's new scale size != model scale-init dims -> original scale, no model update.

        Guards the case where the adjusted scale cannot be written back consistently
        (e.g. an unexpected scale initializer shape), so re-quantizing the weight would
        broadcast incorrectly. The original scale is returned and the model is untouched.
        """
        ws = np.array([1e-8] * 4, dtype=np.float32)
        new_ws = np.array([1e-4] * 4, dtype=np.float32)
        bias_init = onnx.numpy_helper.from_array(np.full(4, 1e5, dtype=np.float32), "b")
        orig_w_init = onnx.numpy_helper.from_array(np.zeros((4, 1, 3, 3), dtype=np.float32), "w")
        # scale initializer deliberately sized 8 (!= new_ws.size of 4)
        scale_init = onnx.numpy_helper.from_array(np.ones(8, dtype=np.float32), "w_scale")
        q_w_init = onnx.numpy_helper.from_array(np.zeros((4, 1, 3, 3), dtype=np.int8), "w_q")
        zp_init = onnx.numpy_helper.from_array(np.zeros(8, dtype=np.int8), "w_zp")
        all_inits = [bias_init, orig_w_init, scale_init, q_w_init, zp_init]
        qv_map = {"w": _make_qv("w")}

        removed = []
        mock_q = MagicMock()
        mock_q.model.initializer.return_value = all_inits
        mock_q.model.remove_initializer.side_effect = lambda t: removed.append(t.name)
        mock_q.quantized_value_map = qv_map
        mock_q.weight_qType = onnx.TensorProto.INT8
        mock_q._adjust_weight_scale_for_int32_bias.return_value = (True, new_ws)

        result = BaseExtendedQDQQuantizer._adjust_weight_scale_for_int32bias_overflow(
            mock_q, "b", "w", np.array([1e-4], dtype=np.float32), ws
        )

        np.testing.assert_array_equal(result, ws)
        self.assertEqual(removed, [])  # model must be left untouched

    def test_overflow_weight_not_in_qv_map_returns_orig_scale(self):
        """ORT reports overflow + weight absent from quantized_value_map -> original scale.

        When the model cannot be updated (no QV entry), the original weight_scale is
        returned so that the bias_scale computed by quantize_bias_static_impl stays
        consistent with the weight DQ node already in the graph.
        """
        ws = np.array([1e-8], dtype=np.float32)
        new_ws = np.array([1e-4], dtype=np.float32)
        result = _call_adjust(
            _mock_quantizer([_f32("b", [1e5])], ort_result=(True, new_ws)),
            "b",
            "w",
            [1e-4],
            ws,
        )
        np.testing.assert_array_equal(result, ws)

    def test_overflow_initializers_missing_returns_orig_scale(self):
        """Weight in qv_map but orig/scale/q_weight initializers not found -> original scale.

        When the model cannot be updated (missing initializers), the original weight_scale is
        returned so that the bias_scale stays consistent with the model's weight DQ node.
        """
        inits = [_f32("b", [1e5])]  # only bias; no weight initializers
        ws = np.array([1e-8], dtype=np.float32)
        new_ws = np.array([1e-4], dtype=np.float32)
        qv_map = {"w": _make_qv("w")}
        result = _call_adjust(
            _mock_quantizer(inits, qv_map, ort_result=(True, new_ws)),
            "b",
            "w",
            [1e-4],
            ws,
        )
        np.testing.assert_array_equal(result, ws)

    def test_overflow_full_path_updates_model_initializers(self):
        """Full path: ORT returns updated scale + all initializers present -> re-quantizes weight."""
        rng = np.random.default_rng(0)
        w_data = (rng.random((4, 1, 3, 3)) * 0.01).astype(np.float32)
        ws_orig = np.array([1e-8] * 4, dtype=np.float32)
        ws_new = np.array([1e-4] * 4, dtype=np.float32)

        q_w_data = np.clip(np.round(w_data / ws_orig[:, None, None, None]), -128, 127).astype(np.int8)

        bias_init = onnx.numpy_helper.from_array(np.full(4, 1e5, dtype=np.float32), "b")
        orig_w_init = onnx.numpy_helper.from_array(w_data, "w")
        scale_init = onnx.numpy_helper.from_array(ws_orig, "w_scale")
        q_w_init = onnx.numpy_helper.from_array(q_w_data, "w_q")
        zp_init = onnx.numpy_helper.from_array(np.zeros(4, dtype=np.int8), "w_zp")

        all_inits = [bias_init, orig_w_init, scale_init, q_w_init, zp_init]
        qv_map = {"w": _make_qv("w")}

        removed, added_names = [], []
        mock_q = MagicMock()
        mock_q.model.initializer.return_value = all_inits
        mock_q.model.remove_initializer.side_effect = lambda t: removed.append(t.name)
        mock_q.model.add_initializer.side_effect = lambda t: added_names.append(t.name)
        mock_q.quantized_value_map = qv_map
        mock_q.weight_qType = onnx.TensorProto.INT8
        mock_q._adjust_weight_scale_for_int32_bias.return_value = (True, ws_new)

        result = BaseExtendedQDQQuantizer._adjust_weight_scale_for_int32bias_overflow(
            mock_q, "b", "w", np.array([1e-4], dtype=np.float32), ws_orig
        )

        np.testing.assert_array_equal(result, ws_new)
        # Both old scale and quantized weight must be replaced
        self.assertEqual(removed, ["w_scale", "w_q"])
        self.assertEqual(sorted(added_names), ["w_q", "w_scale"])

    def test_qv_axis_none_but_scale_per_channel_updates_model(self):
        """SE-block fc2 case: qv.original.axis=None but scale initializer has C elements.

        Previously this triggered a size-mismatch warning and skipped the model update,
        leaving the bias_scale inconsistent with the model's weight DQ and causing
        adjust_bias_scale to re-quantize the INT32 bias incorrectly (wrap-around overflow).

        After the fix, the actual axis is derived from the scale/weight dims and the
        model update proceeds normally.
        """
        rng = np.random.default_rng(1)
        w_data = (rng.random((4, 1, 3, 3)) * 0.01).astype(np.float32)
        ws_orig = np.array([1e-8] * 4, dtype=np.float32)
        ws_new = np.array([1e-4] * 4, dtype=np.float32)

        q_w_data = np.clip(np.round(w_data / ws_orig[:, None, None, None]), -128, 127).astype(np.int8)

        bias_init = onnx.numpy_helper.from_array(np.full(4, 1e5, dtype=np.float32), "b")
        orig_w_init = onnx.numpy_helper.from_array(w_data, "w")
        scale_init = onnx.numpy_helper.from_array(ws_orig, "w_scale")  # shape (4,) — per-channel
        q_w_init = onnx.numpy_helper.from_array(q_w_data, "w_q")
        zp_init = onnx.numpy_helper.from_array(np.zeros(4, dtype=np.int8), "w_zp")

        all_inits = [bias_init, orig_w_init, scale_init, q_w_init, zp_init]
        # Simulate SE-block fc2: axis=None in qv despite scale being per-channel
        qv_map = {"w": _make_qv("w", axis=None)}

        removed, added_names = [], []
        mock_q = MagicMock()
        mock_q.model.initializer.return_value = all_inits
        mock_q.model.remove_initializer.side_effect = lambda t: removed.append(t.name)
        mock_q.model.add_initializer.side_effect = lambda t: added_names.append(t.name)
        mock_q.quantized_value_map = qv_map
        mock_q.weight_qType = onnx.TensorProto.INT8
        mock_q._adjust_weight_scale_for_int32_bias.return_value = (True, ws_new)

        result = BaseExtendedQDQQuantizer._adjust_weight_scale_for_int32bias_overflow(
            mock_q, "b", "w", np.array([1e-4], dtype=np.float32), ws_orig
        )

        np.testing.assert_array_equal(result, ws_new)
        # Model update must have proceeded despite qv.original.axis=None
        self.assertEqual(removed, ["w_scale", "w_q"])
        self.assertEqual(sorted(added_names), ["w_q", "w_scale"])


# -----------------------------------------------------------------------
# Unit tests: quantize_bias_static
# -----------------------------------------------------------------------
class TestQuantizeBiasStaticOverride(unittest.TestCase):
    """Unit tests for BaseExtendedQDQQuantizer.quantize_bias_static."""

    @staticmethod
    def _bias_info(node="n", inp="x", weight="w", beta=1.0) -> QDQBiasQuantInfo:
        return QDQBiasQuantInfo(node, inp, weight, beta)

    def test_early_return_when_bias_already_quantized(self):
        """Line 956: bias already in quantized_value_map -> returns q_name immediately."""
        qval = QuantizedValue("b", "b_q", "b_scale", "b_zp", QuantizedValueType.Initializer)
        mock_q = MagicMock()
        mock_q.quantized_value_map = {"b": QDQTensorQuantizedValue(qval, None, None)}

        result = BaseExtendedQDQQuantizer.quantize_bias_static(mock_q, "b", self._bias_info())

        self.assertEqual(result, "b_q")
        mock_q._get_tensor_quantization_scale.assert_not_called()

    def test_adjust_called_when_option_true(self):
        """Line 973: AdjustWeightScaleForInt32Bias=True triggers the adjustment helper."""
        ws = np.array([0.01], dtype=np.float32)

        mock_q = MagicMock()
        mock_q.quantized_value_map = {}
        mock_q.extra_options = {"AdjustWeightScaleForInt32Bias": True}
        mock_q._get_tensor_quantization_scale.return_value = ws
        mock_q._adjust_weight_scale_for_int32bias_overflow.return_value = ws
        mock_q.quantize_bias_static_impl.return_value = (
            "b_q",
            "b_scale",
            "b_zp",
            np.array([0.01], dtype=np.float32),
            None,
            onnx.TensorProto.INT32,
        )

        result = BaseExtendedQDQQuantizer.quantize_bias_static(mock_q, "b", self._bias_info())

        self.assertEqual(result, "b_q")
        mock_q._adjust_weight_scale_for_int32bias_overflow.assert_called_once()


if __name__ == "__main__":
    unittest.main()
