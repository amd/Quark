#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import Config, ModelQuantizer
from quark.onnx.algorithm.cle.equalization import Equalization, _detect_stem_bn_groups, stem_equalize_transforms
from quark.onnx.quantization.config.custom_config import U8S8_AAWS_CONFIG


class StemBNModel(nn.Module):
    """Conv(stem) -> BN -> ReLU -> MaxPool -> Conv -> GAP -> FC.

    The stem conv is given a large per-output-channel magnitude spread so that
    stem equalization has something to fix.
    """

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(8, 4, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 1)

        with torch.no_grad():
            # Make per-output-channel weight magnitudes wildly uneven.
            w = self.conv0.weight
            scales = torch.tensor([1.0, 0.01, 0.5, 0.02, 1.0, 0.005, 0.3, 0.01]).reshape(8, 1, 1, 1)
            w.mul_(scales)
            # Give BN non-trivial affine so folding is observable.
            self.bn.weight.copy_(torch.linspace(0.5, 2.0, 8))
            self.bn.bias.copy_(torch.linspace(-1.0, 1.0, 8))
            self.bn.running_mean.copy_(torch.linspace(-0.5, 0.5, 8))
            self.bn.running_var.copy_(torch.linspace(0.5, 1.5, 8))

    def forward(self, x):
        x = self.pool(self.relu(self.bn(self.conv0(x))))
        x = self.gap(self.conv1(x))
        x = torch.flatten(x, 1)
        return self.fc(x)


class StemConcatBNModel(nn.Module):
    """Conv(stem) -> ReLU -> Concat([branch, stem]) -> BN. The stem lands at a
    non-zero channel offset in the Concat, exercising offset accumulation."""

    def __init__(self):
        super().__init__()
        self.conv0 = nn.Conv2d(3, 8, 3, padding=1)  # stem
        self.branch = nn.Conv2d(3, 4, 3, padding=1)  # precedes stem in Concat
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(12)
        self.conv1 = nn.Conv2d(12, 4, 3, padding=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(4, 1)
        with torch.no_grad():
            self.conv0.weight.mul_(torch.tensor([1.0, 0.01, 0.5, 0.02, 1.0, 0.005, 0.3, 0.01]).reshape(8, 1, 1, 1))
            self.bn.weight.copy_(torch.linspace(0.5, 2.0, 12))
            self.bn.bias.copy_(torch.linspace(-1.0, 1.0, 12))
            self.bn.running_mean.copy_(torch.linspace(-0.5, 0.5, 12))
            self.bn.running_var.copy_(torch.linspace(0.5, 1.5, 12))

    def forward(self, x):
        s = self.relu(self.conv0(x))  # conv0 exported first -> chosen as stem
        b = self.branch(x)
        x = self.bn(torch.cat([b, s], dim=1))  # stem occupies channels [4:12]
        x = self.gap(self.conv1(x))
        return self.fc(torch.flatten(x, 1))


class DataReader(CalibrationDataReader):
    def __init__(self, input_tensor):
        self.data = [input_tensor]
        self.input_name = "input"
        self.index = 0

    def get_next(self):
        """Get the next calibration data sample.
        This method is part of the CalibrationDataReader interface and provides
        input data for model calibration during quantization.
        Returns:
            dict or None: A dictionary mapping input name to tensor data if available,
                         or None when all calibration data has been exhausted.
        """
        if self.index < len(self.data):
            d = {self.input_name: self.data[self.index]}
            self.index += 1
            return d
        return None

    def rewind(self):
        """Reset the data reader index to the beginning."""
        self.index = 0


def _export(model, path, shape=(1, 3, 8, 8)):
    """Export a PyTorch model to ONNX format.
    Args:
        model: PyTorch model to export.
        path: File path where the ONNX model will be saved.
        shape: Input tensor shape for the model. Defaults to (1, 3, 8, 8).
    Returns:
        The path where the ONNX model was saved.
    """
    model.eval()
    torch.onnx.export(
        model,
        torch.randn(*shape),
        path,
        input_names=["input"],
        output_names=["output"],
        opset_version=17,
        do_constant_folding=False,  # keep BatchNormalization as a standalone node
        dynamo=False,
    )
    return path


def _fp32_max_diff(path_a, path_b, shape=(1, 3, 8, 8), n=8, seed=0):
    """Compute the maximum FP32 output difference between two ONNX models.
    Args:
        path_a: Path to the first ONNX model.
        path_b: Path to the second ONNX model.
        shape: Input tensor shape for testing. Defaults to (1, 3, 8, 8).
        n: Number of random inputs to test. Defaults to 8.
        seed: Random seed for reproducibility. Defaults to 0.
    Returns:
        float: Maximum absolute difference across all outputs and test runs.
    """
    rng = np.random.default_rng(seed)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sa = ort.InferenceSession(path_a, so, providers=["CPUExecutionProvider"])
    sb = ort.InferenceSession(path_b, so, providers=["CPUExecutionProvider"])
    oa = [o.name for o in sa.get_outputs()]
    ob = [o.name for o in sb.get_outputs()]
    ia = sa.get_inputs()[0].name
    mx = 0.0
    for _ in range(n):
        x = rng.standard_normal(shape).astype(np.float32)
        ra = sa.run(oa, {ia: x})[0]
        rb = sb.run(ob, {ia: x})[0]
        mx = max(mx, float(np.abs(ra - rb).max()))
    return mx


def _get_init(model, name):
    """Retrieve an initializer by name from an ONNX model.
    Args:
        model: An ONNX model.
        name: The name of the initializer to retrieve.
    Returns:
        A numpy array containing the initializer's data.
    Raises:
        KeyError: If no initializer with the given name is found.
    """
    for init in model.graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    raise KeyError(name)


def _conv(name, w, out, x="input", bias=None):
    inits = [numpy_helper.from_array(w, name + ".w")]
    ins = [x, name + ".w"]
    if bias is not None:
        inits.append(numpy_helper.from_array(bias, name + ".b"))
        ins.append(name + ".b")
    return helper.make_node("Conv", ins, [out], name=name, group=1), inits


def _bn(name, c, x, out):
    inits = [
        numpy_helper.from_array(np.ones(c, np.float32), name + ".g"),
        numpy_helper.from_array(np.zeros(c, np.float32), name + ".b"),
        numpy_helper.from_array(np.zeros(c, np.float32), name + ".m"),
        numpy_helper.from_array(np.ones(c, np.float32), name + ".v"),
    ]
    node = helper.make_node(
        "BatchNormalization", [x, name + ".g", name + ".b", name + ".m", name + ".v"], [out], name=name
    )
    return node, inits


def _make_model(nodes, inits, out_name, out_c=8):
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, [1, out_c, 8, 8])],
        inits,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_eq(model):
    return Equalization(model, ["Conv", "Gemm"], [], [])


class TestStemEqualize(unittest.TestCase):
    def _assert_noop(self, model):
        """stem_equalize_transforms on `model` must leave every initializer untouched."""
        before = {i.name: numpy_helper.to_array(i).copy() for i in model.graph.initializer}
        after = stem_equalize_transforms(model, ["Conv", "Gemm"], [], [])
        self.assertEqual(set(before), {i.name for i in after.graph.initializer})
        for i in after.graph.initializer:
            np.testing.assert_array_equal(numpy_helper.to_array(i), before[i.name])

    @use_temporary_directory
    def test_math_equivalence(self, tmpdir: str):
        # Core positive case: equalizing StemBNModel must (1) keep the FP32 output
        # bit-for-bit (weight up-scale and BN fold cancel), and (2) actually shrink
        # the per-output-channel weight spread (so it is not a silent no-op).
        torch.manual_seed(42)
        src = Path(tmpdir, "stem.onnx").as_posix()
        out = Path(tmpdir, "stem_eq.onnx").as_posix()
        _export(StemBNModel(), src)

        model = onnx.load(src)
        stem, cout, bn_groups = _detect_stem_bn_groups(model, _make_eq(model))
        self.assertIsNotNone(stem)
        self.assertGreaterEqual(len(bn_groups), 1)
        w_name = next(inp for inp in stem.input if inp in {i.name for i in model.graph.initializer})
        absmax_before = np.abs(_get_init(model, w_name).reshape(cout, -1)).max(axis=1)

        eq_model = stem_equalize_transforms(onnx.load(src), ["Conv", "Gemm"], [], [])
        onnx.save(eq_model, out)

        # (1) FP32 output preserved.
        self.assertLess(_fp32_max_diff(src, out), 1e-3)

        # (2) Per-channel magnitude spread shrinks.
        absmax_after = np.abs(_get_init(eq_model, w_name).reshape(cout, -1)).max(axis=1)
        live = absmax_before > 1e-6
        self.assertLess(
            absmax_after[live].max() / absmax_after[live].min(),
            absmax_before[live].max() / absmax_before[live].min(),
        )

    @use_temporary_directory
    def test_concat_offset_equivalence(self, tmpdir: str):
        # Stem reaches the BN through a Concat where it is the *second* branch, so it
        # sits at channel offset 4. Exercises offset accumulation + _tensor_channels,
        # and must stay FP32-equivalent.
        torch.manual_seed(42)
        src = Path(tmpdir, "concat.onnx").as_posix()
        out = Path(tmpdir, "concat_eq.onnx").as_posix()
        _export(StemConcatBNModel(), src)
        # Concat branch channel counts come from static shapes.
        model = onnx.shape_inference.infer_shapes(onnx.load(src))
        onnx.save(model, src)

        stem, cout, bn_groups = _detect_stem_bn_groups(model, _make_eq(model))
        self.assertIsNotNone(stem)
        self.assertEqual(cout, 8)
        self.assertEqual(len(bn_groups), 1)
        self.assertEqual(bn_groups[0][1], 4)  # stem at offset 4 (after the 4-ch branch)

        eq_model = stem_equalize_transforms(onnx.load(src), ["Conv", "Gemm"], [], [])
        onnx.save(eq_model, out)
        self.assertLess(_fp32_max_diff(src, out), 1e-3)

    def test_pad_passthrough(self):
        # Conv -> Pad(constant, zero) -> BN: a zero constant pad is homogeneous, so the
        # stem is detected and the offset stays 0 (the pad does not shift channels).
        w = np.random.default_rng(0).standard_normal((8, 3, 3, 3)).astype(np.float32)
        pads = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "pads")
        conv, ci = _conv("stem", w, "c")
        pad = helper.make_node("Pad", ["c", "pads"], ["p"], name="pad", mode="constant")
        bn, bi = _bn("bn", 8, "p", "out")
        model = _make_model([conv, pad, bn], ci + [pads] + bi, "out")
        stem, _, bn_groups = _detect_stem_bn_groups(model, _make_eq(model))
        self.assertIsNotNone(stem)
        self.assertEqual(bn_groups[0][1], 0)

    def test_pad_variants_abort(self):
        # Every Pad shape that is NOT a statically-zero constant pad breaks homogeneity,
        # so detection must bail out (stem is None).
        w = np.random.default_rng(0).standard_normal((8, 3, 3, 3)).astype(np.float32)
        pads = numpy_helper.from_array(np.array([0, 0, 1, 1, 0, 0, 1, 1], np.int64), "pads")

        # reflect: non-constant mode.
        conv, ci = _conv("stem", w, "c")
        pad = helper.make_node("Pad", ["c", "pads"], ["p"], name="pad", mode="reflect")
        bn, bi = _bn("bn", 8, "p", "out")
        reflect = _make_model([conv, pad, bn], ci + [pads] + bi, "out")

        # nonzero constant: pad value 5.0 fills regions the stem scale never reaches.
        conv, ci = _conv("stem", w, "c")
        cval = numpy_helper.from_array(np.array(5.0, np.float32), "cval")
        pad = helper.make_node("Pad", ["c", "pads", "cval"], ["p"], name="pad", mode="constant")
        bn, bi = _bn("bn", 8, "p", "out")
        nonzero = _make_model([conv, pad, bn], ci + [pads, cval] + bi, "out")

        # dynamic constant: pad value is a runtime input, not provably zero.
        conv, ci = _conv("stem", w, "c")
        pad = helper.make_node("Pad", ["c", "pads", "cval"], ["p"], name="pad", mode="constant")
        bn, bi = _bn("bn", 8, "p", "out")
        graph = helper.make_graph(
            [conv, pad, bn],
            "g",
            [
                helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8]),
                helper.make_tensor_value_info("cval", TensorProto.FLOAT, []),
            ],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 8, 8, 8])],
            ci + [pads] + bi,
        )
        dynamic = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

        # legacy opset<11 constant pad: pads/value are node attributes (no input[2]);
        # a nonzero attribute value still breaks homogeneity -> abort.
        conv, ci = _conv("stem", w, "c")
        pad = helper.make_node(
            "Pad", ["c"], ["p"], name="pad", mode="constant", pads=[0, 0, 1, 1, 0, 0, 1, 1], value=5.0
        )
        bn, bi = _bn("bn", 8, "p", "out")
        legacy = _make_model([conv, pad, bn], ci + bi, "out")

        for name, model in [
            ("reflect", reflect),
            ("nonzero-const", nonzero),
            ("dynamic-const", dynamic),
            ("legacy-attr-nonzero", legacy),
        ]:
            with self.subTest(pad=name):
                stem, _, _ = _detect_stem_bn_groups(model, _make_eq(model))
                self.assertIsNone(stem)

    def test_concat_variants_abort(self):
        # Concats whose channel offset cannot be tracked statically -> detection aborts.
        w = np.random.default_rng(0).standard_normal((8, 3, 3, 3)).astype(np.float32)

        # bad axis: Concat on a non-channel axis.
        conv, ci = _conv("stem", w, "c")
        concat = helper.make_node("Concat", ["c", "c"], ["cc"], name="cat", axis=2)
        bn, bi = _bn("bn", 8, "cc", "out")
        bad_axis = _make_model([conv, concat, bn], ci + bi, "out")

        # unknown channels: a preceding branch has no static shape, so its channel
        # count (needed for the offset) is unknown.
        wb = np.random.default_rng(1).standard_normal((8, 3, 3, 3)).astype(np.float32)
        conv, ci = _conv("stem", w, "s")
        branch, bri = _conv("branch", wb, "b")  # 'b' has no value_info
        concat = helper.make_node("Concat", ["b", "s"], ["cc"], name="cat", axis=1)
        bn, bi = _bn("bn", 16, "cc", "out")
        unknown = _make_model([conv, branch, concat, bn], ci + bri + bi, "out", out_c=16)

        for name, model in [("bad-axis", bad_axis), ("unknown-channels", unknown)]:
            with self.subTest(concat=name):
                stem, _, _ = _detect_stem_bn_groups(model, _make_eq(model))
                self.assertIsNone(stem)

    def test_stem_search_aborts(self):
        # Structural reasons the stem search / BFS finds nothing foldable -> stem is None.
        w = np.random.default_rng(0).standard_normal((8, 3, 3, 3)).astype(np.float32)

        # no BN downstream: stem found, but the BFS reaches no BatchNorm.
        conv, ci = _conv("stem", w, "c")
        relu = helper.make_node("Relu", ["c"], ["out"], name="relu")
        no_bn = (_make_model([conv, relu], ci, "out"), _make_eq)

        # non-homogeneous op: Sigmoid between stem and BN hits the else branch.
        conv, ci = _conv("stem", w, "c")
        sig = helper.make_node("Sigmoid", ["c"], ["sg"], name="sig")
        bn, bi = _bn("bn", 8, "sg", "out")
        sigmoid = (_make_model([conv, sig, bn], ci + bi, "out"), _make_eq)

        # excluded / non-input: the only input-fed Conv is excluded from quantization,
        # and the other Conv is not fed by a graph input.
        w2 = np.random.default_rng(1).standard_normal((8, 8, 3, 3)).astype(np.float32)
        c0, ci0 = _conv("stem", w, "c0")
        c1, ci1 = _conv("mid", w2, "out", x="c0")
        excluded_model = _make_model([c0, c1], ci0 + ci1, "out")
        excluded = (excluded_model, lambda m: Equalization(m, ["Conv", "Gemm"], [], ["stem"]))

        # weight not an initializer: stem Conv weight is a graph input.
        w_vi = helper.make_tensor_value_info("w_in", TensorProto.FLOAT, [8, 3, 3, 3])
        conv = helper.make_node("Conv", ["input", "w_in"], ["c"], name="stem", group=1)
        bn, bi = _bn("bn", 8, "c", "out")
        graph = helper.make_graph(
            [conv, bn],
            "g",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 8, 8]), w_vi],
            [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, 8, 8, 8])],
            bi,
        )
        no_weight_init = (helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)]), _make_eq)

        cases = [
            ("no-bn-downstream", no_bn),
            ("non-homogeneous", sigmoid),
            ("excluded/non-input", excluded),
            ("weight-not-initializer", no_weight_init),
        ]
        for name, (model, make_eq) in cases:
            with self.subTest(reason=name):
                stem, _, _ = _detect_stem_bn_groups(model, make_eq(model))
                self.assertIsNone(stem)

    def test_noop_leaves_model_unchanged(self):
        # Cases where the stem is reached but equalization cannot proceed: the model
        # must be returned bit-for-bit unchanged (never half-scaled).
        w = np.random.default_rng(0).standard_normal((8, 3, 3, 3)).astype(np.float32)

        # all channels near-zero: no live channel to equalize.
        dead = np.full((8, 3, 3, 3), 1e-9, np.float32)
        conv, ci = _conv("stem", dead, "c")
        bn, bi = _bn("bn", 8, "c", "out")
        all_dead = _make_model([conv, bn], ci + bi, "out")

        # BN too small: its 4 scale channels cannot hold the stem's 8-channel span, so
        # detection rejects it and stem-eq is a no-op.
        conv, ci = _conv("stem", w, "c")
        bn, bi = _bn("bn", 4, "c", "out")
        bn_too_small = _make_model([conv, bn], ci + bi, "out", out_c=4)

        for name, model in [("all-dead-channels", all_dead), ("bn-too-small", bn_too_small)]:
            with self.subTest(case=name):
                self._assert_noop(model)

    @use_temporary_directory
    def test_e2e_standalone_quantization(self, tmpdir: str):
        # Standalone use: with include_cle=False, stem_equalize_transforms is called
        # directly (no CLE state needed), and the equalized model still quantizes and
        # runs end to end.
        torch.manual_seed(42)
        src = Path(tmpdir, "stem.onnx").as_posix()
        eq = Path(tmpdir, "stem_eq.onnx").as_posix()
        out = Path(tmpdir, "stem_quant.onnx").as_posix()
        _export(StemBNModel(), src)

        eq_model = stem_equalize_transforms(onnx.load(src), ["Conv", "Gemm"], [], [])
        onnx.save(eq_model, eq)

        cfg = copy.deepcopy(U8S8_AAWS_CONFIG)
        cfg.include_cle = False
        quant_config = Config(global_quant_config=cfg)

        x = np.random.default_rng(0).standard_normal((1, 3, 8, 8)).astype(np.float32)
        ModelQuantizer(quant_config).quantize_model(eq, out, DataReader(x))
        sess = ort.InferenceSession(out, providers=["CPUExecutionProvider"])
        y = sess.run(None, {sess.get_inputs()[0].name: x})
        self.assertEqual(len(y), 1)


if __name__ == "__main__":
    unittest.main()
