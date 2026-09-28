#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Unit tests for standard (non-microscaling) FP8 support in Quark ONNX.

Covers the code added for FLOAT8E4M3FN / FLOAT8E5M2:
  * quant_utils.py
      - ONNX_FP_QTYPES_LIST / ONNX_TYPE_TO_NP_TYPE registration of both FP8 types
      - get_qmin_qmax_for_qType FP8 ranges (+/-448, +/-57344)
      - compute_scale_zp_fp FP8 branch (scale = absmax / fp8_max, zp in FP8 dtype)
      - is_fp8_qtype detection helper and FP8_MIN_OPSET constant
  * quantize.py
      - automatic opset conversion to FP8_MIN_OPSET when an FP8 type is requested
        on a low-opset input model, and the no-op path when the opset is already
        high enough.
"""

import importlib
import importlib.util
import types
import unittest
from unittest import mock

import ml_dtypes
import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantFormat, QuantType

import quark.onnx.quantization.quantize as quantize_mod
from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import Config, ModelQuantizer
from quark.onnx.quantization import quant_utils as quant_utils_mod
from quark.onnx.quantization.config.legacy import QuantizationConfig
from quark.onnx.quantization.quant_utils import (
    FP8_MIN_OPSET,
    ONNX_FP_QTYPES_LIST,
    ONNX_TYPE_TO_NP_TYPE,
    compute_scale_zp_fp,
    get_opset_version,
    get_qmin_qmax_for_qType,
    is_fp8_qtype,
)

E4M3 = onnx.TensorProto.FLOAT8E4M3FN
E5M2 = onnx.TensorProto.FLOAT8E5M2


def _e5m2_qtype() -> types.SimpleNamespace:
    """QuantType-compatible stub for E5M2 (ORT's QuantType has no QFLOAT8E5M2)."""
    return types.SimpleNamespace(tensor_type=E5M2)


# ---------------------------------------------------------------------------
# quant_utils.py helper tests (no model / no I/O required)
# ---------------------------------------------------------------------------


class TestFP8QuantUtils(unittest.TestCase):
    def test_fp8_types_registered_in_fp_list(self):
        self.assertIn(E4M3, ONNX_FP_QTYPES_LIST)
        self.assertIn(E5M2, ONNX_FP_QTYPES_LIST)

    def test_fp8_types_mapped_to_numpy_dtype(self):
        self.assertIs(ONNX_TYPE_TO_NP_TYPE[E4M3], ml_dtypes.float8_e4m3fn)
        self.assertIs(ONNX_TYPE_TO_NP_TYPE[E5M2], ml_dtypes.float8_e5m2)

    def test_qmin_qmax_e4m3(self):
        qmin, qmax = get_qmin_qmax_for_qType(E4M3)
        self.assertEqual(float(qmin), -448.0)
        self.assertEqual(float(qmax), 448.0)

    def test_qmin_qmax_e5m2(self):
        qmin, qmax = get_qmin_qmax_for_qType(E5M2)
        self.assertEqual(float(qmin), -57344.0)
        self.assertEqual(float(qmax), 57344.0)

    def test_compute_scale_zp_e4m3(self):
        # scale = max(|rmin|, |rmax|) / 448 ; zero-point 0 in FP8 dtype
        rmin = np.array(-4.0, dtype=np.float32)
        rmax = np.array(8.0, dtype=np.float32)
        zero, scale = compute_scale_zp_fp(rmin, rmax, np.array(-448.0), np.array(448.0), E4M3, CalibrationMethod.MinMax)
        self.assertAlmostEqual(float(scale), 8.0 / 448.0, places=5)
        self.assertEqual(zero.dtype, ml_dtypes.float8_e4m3fn)
        self.assertEqual(float(zero), 0.0)

    def test_compute_scale_zp_e5m2(self):
        rmin = np.array(-100.0, dtype=np.float32)
        rmax = np.array(50.0, dtype=np.float32)
        zero, scale = compute_scale_zp_fp(
            rmin, rmax, np.array(-57344.0), np.array(57344.0), E5M2, CalibrationMethod.MinMax
        )
        self.assertAlmostEqual(float(scale), 100.0 / 57344.0, places=7)
        self.assertEqual(zero.dtype, ml_dtypes.float8_e5m2)
        self.assertEqual(float(zero), 0.0)

    def test_compute_scale_zp_fp16_scale_dtype(self):
        # scale dtype follows rmax.dtype -> FP16 input yields an FP16 scale
        rmin = np.array(-2.0, dtype=np.float16)
        rmax = np.array(2.0, dtype=np.float16)
        _, scale = compute_scale_zp_fp(rmin, rmax, np.array(-448.0), np.array(448.0), E4M3, CalibrationMethod.MinMax)
        self.assertEqual(scale.dtype, np.float16)

    def test_compute_scale_zp_zero_range_falls_back_to_one(self):
        # An all-zero tensor must not produce a zero/NaN scale.
        rmin = np.array(0.0, dtype=np.float32)
        rmax = np.array(0.0, dtype=np.float32)
        _, scale = compute_scale_zp_fp(rmin, rmax, np.array(-448.0), np.array(448.0), E4M3, CalibrationMethod.MinMax)
        self.assertEqual(float(scale), 1.0)

    def test_is_fp8_qtype(self):
        self.assertTrue(is_fp8_qtype(QuantType.QFLOAT8E4M3FN))
        self.assertTrue(is_fp8_qtype(_e5m2_qtype()))
        # Non-FP8 types / objects without a tensor_type return False
        self.assertFalse(is_fp8_qtype(QuantType.QInt8))
        self.assertFalse(is_fp8_qtype(QuantType.QUInt8))
        self.assertFalse(is_fp8_qtype(object()))
        self.assertFalse(is_fp8_qtype(None))

    def test_fp8_min_opset_constant(self):
        self.assertEqual(FP8_MIN_OPSET, 21)


# ---------------------------------------------------------------------------
# quantize.py end-to-end tests (auto opset conversion + FP8 QDQ insertion)
# ---------------------------------------------------------------------------


class TinyMatMulModel(nn.Module):
    """Minimal model whose only quantizable op is a MatMul (Linear)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.fc(x)


class DataReader(CalibrationDataReader):
    def __init__(self, input_name: str, shape) -> None:
        self.input_name = input_name
        self.data = [{input_name: np.random.rand(*shape).astype(np.float32)} for _ in range(2)]
        self.index = 0

    def get_next(self):
        if self.index < len(self.data):
            item = self.data[self.index]
            self.index += 1
            return item
        return None

    def rewind(self):
        self.index = 0

    def reset(self):
        self.index = 0


def _export_model(path: str, opset: int) -> None:
    torch.manual_seed(0)
    dummy = torch.randn(1, 8)
    torch.onnx.export(
        TinyMatMulModel(),
        dummy,
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=opset,
        dynamo=False,
    )


def _fp8_config(fp8_type: str, per_channel: bool = False, weights_only: bool = True) -> QuantizationConfig:
    quant_type = QuantType.QFLOAT8E4M3FN if fp8_type == "e4m3" else _e5m2_qtype()
    return QuantizationConfig(
        calibrate_method=CalibrationMethod.MinMax,
        quant_format=QuantFormat.QDQ,
        activation_type=quant_type,
        weight_type=quant_type,
        op_types_to_quantize=["MatMul", "Gemm"],
        per_channel=per_channel,
        include_cle=False,
        extra_options={
            "WeightsOnly": weights_only,
            "ActivationSymmetric": True,
            "QuantizeBias": False,
        },
    )


def _quantize(in_path: str, out_path: str, cfg: QuantizationConfig) -> None:
    dr = DataReader("input", (1, 8))
    ModelQuantizer(Config(global_quant_config=cfg)).quantize_model(in_path, out_path, dr)


class TestFP8AutoOpsetConversion(unittest.TestCase):
    @use_temporary_directory
    def test_low_opset_input_is_bumped_to_fp8_min_opset(self, tmpdir: str):
        """An opset-14 input must be auto-converted to FP8_MIN_OPSET for E4M3."""
        src = f"{tmpdir}/tiny_op14.onnx"
        dst = f"{tmpdir}/tiny_op14_fp8.onnx"
        _export_model(src, opset=14)
        self.assertLess(get_opset_version(onnx.load(src)), FP8_MIN_OPSET)

        _quantize(src, dst, _fp8_config("e4m3"))

        out = onnx.load(dst)
        self.assertGreaterEqual(get_opset_version(out), FP8_MIN_OPSET)
        # Weight was quantized to FP8: a DequantizeLinear with an FP8 initializer exists.
        inits = {t.name: t for t in out.graph.initializer}
        fp8_dq = [
            n
            for n in out.graph.node
            if n.op_type == "DequantizeLinear" and n.input[0] in inits and inits[n.input[0]].data_type == E4M3
        ]
        self.assertGreaterEqual(len(fp8_dq), 1)

    @use_temporary_directory
    def test_high_opset_input_is_not_downgraded(self, tmpdir: str):
        """An input already at opset 21 must be left at >= FP8_MIN_OPSET (no-op path)."""
        src = f"{tmpdir}/tiny_op21.onnx"
        dst = f"{tmpdir}/tiny_op21_fp8.onnx"
        _export_model(src, opset=21)
        self.assertGreaterEqual(get_opset_version(onnx.load(src)), FP8_MIN_OPSET)

        _quantize(src, dst, _fp8_config("e4m3"))

        self.assertGreaterEqual(get_opset_version(onnx.load(dst)), FP8_MIN_OPSET)

    @use_temporary_directory
    def test_e4m3_output_loads_in_ort(self, tmpdir: str):
        """The weights-only FP8 E4M3 output must load and run in ONNX Runtime."""
        src = f"{tmpdir}/tiny.onnx"
        dst = f"{tmpdir}/tiny_e4m3.onnx"
        _export_model(src, opset=17)
        _quantize(src, dst, _fp8_config("e4m3", weights_only=True))

        sess = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
        out = sess.run(None, {"input": np.random.rand(1, 8).astype(np.float32)})
        self.assertEqual(out[0].shape, (1, 8))

    @use_temporary_directory
    def test_weights_only_has_no_activation_quantize(self, tmpdir: str):
        """WeightsOnly=True must not insert any QuantizeLinear (activations stay float)."""
        src = f"{tmpdir}/tiny.onnx"
        dst = f"{tmpdir}/tiny_wo.onnx"
        _export_model(src, opset=17)
        _quantize(src, dst, _fp8_config("e4m3", weights_only=True))

        out = onnx.load(dst)
        q_nodes = [n for n in out.graph.node if n.op_type == "QuantizeLinear"]
        self.assertEqual(len(q_nodes), 0)
        dq_nodes = [n for n in out.graph.node if n.op_type == "DequantizeLinear"]
        self.assertGreaterEqual(len(dq_nodes), 1)

    @use_temporary_directory
    def test_e5m2_weight_initializer_dtype(self, tmpdir: str):
        """E5M2 weight-only quantization must store weights as FLOAT8E5M2."""
        src = f"{tmpdir}/tiny.onnx"
        dst = f"{tmpdir}/tiny_e5m2.onnx"
        _export_model(src, opset=17)
        _quantize(src, dst, _fp8_config("e5m2", weights_only=True))

        out = onnx.load(dst)
        self.assertGreaterEqual(get_opset_version(out), FP8_MIN_OPSET)
        fp8_inits = [t for t in out.graph.initializer if t.data_type == E5M2]
        self.assertGreaterEqual(len(fp8_inits), 1)

    @use_temporary_directory
    def test_opset_bump_fallback_when_convert_fails(self, tmpdir: str):
        """If convert_opset_version raises, fall back to a direct opset_import bump.

        Covers the ``except`` branch in quantize_static that manually rewrites the
        ai.onnx opset and ir_version when the ONNX version converter cannot handle
        the model.
        """
        src = f"{tmpdir}/tiny_op14.onnx"
        dst = f"{tmpdir}/tiny_op14_fallback.onnx"
        _export_model(src, opset=14)

        # Force the primary conversion path to fail so the fallback executes.
        with mock.patch.object(
            quantize_mod, "convert_opset_version", side_effect=RuntimeError("forced failure")
        ) as patched:
            _quantize(src, dst, _fp8_config("e4m3"))
        self.assertTrue(patched.called)

        out = onnx.load(dst)
        self.assertGreaterEqual(get_opset_version(out), FP8_MIN_OPSET)
        self.assertGreaterEqual(out.ir_version, 9)


class TestFP8LegacyOnnxImport(unittest.TestCase):
    """Cover the onnx<1.19 import branch for the FP8 numpy dtype aliases.

    In the running environment onnx >= 1.19, so the legacy branch (which imports
    from onnx.reference.custom_element_types, falling back to None on ImportError)
    is normally never executed. We exercise it by executing a *fresh, isolated*
    copy of quant_utils (under a throwaway module name) with a spoofed onnx
    version.

    Note: we deliberately do NOT ``importlib.reload`` the canonical
    ``quant_utils`` module. Reloading re-creates ``ExtendedQuantType`` and
    ``get_tensor_type_from_qType``; other already-imported modules keep the
    original objects, and because enum equality is by identity this permanently
    breaks qtype comparisons (``Unexpected value qtype=...``) for every test that
    runs afterwards in the same process. Loading an isolated copy leaves
    ``sys.modules`` untouched, so no other test is affected.
    """

    @staticmethod
    def _load_isolated_quant_utils():
        """Execute a fresh copy of quant_utils without touching sys.modules."""
        spec = importlib.util.spec_from_file_location("quark_quant_utils_isolated_fp8test", quant_utils_mod.__file__)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_legacy_import_branch_sets_fp8_aliases(self):
        with mock.patch.object(onnx, "__version__", "1.18.0"):
            isolated = self._load_isolated_quant_utils()
        # onnx.reference.custom_element_types is unavailable in modern onnx,
        # so the ImportError fallback assigns None to both aliases.
        self.assertIsNone(isolated.float8e4m3fn)
        self.assertIsNone(isolated.float8e5m2)

    def test_modern_onnx_uses_ml_dtypes_aliases(self):
        # The canonical module (onnx >= 1.19) uses the ml_dtypes aliases.
        self.assertIs(quant_utils_mod.float8e4m3fn, ml_dtypes.float8_e4m3fn)
        self.assertIs(quant_utils_mod.float8e5m2, ml_dtypes.float8_e5m2)


if __name__ == "__main__":
    unittest.main()
