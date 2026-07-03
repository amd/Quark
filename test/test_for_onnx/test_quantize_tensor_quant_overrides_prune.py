#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import copy
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import CalibrationDataReader
from onnxruntime.quantization.quant_utils import QuantType

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import Config, ModelQuantizer
from quark.onnx.quantization.config.custom_config import A16W8_CONFIG
from quark.onnx.quantization.quant_utils import ExtendedQuantFormat
from quark.onnx.quantization.quantize import (
    apply_align_eltwise_quant_type,
    prune_stale_tensor_quant_overrides,
)

QUANTIZE_LOGGER_NAME = "quark.onnx.quantization.quantize_screen"


class _SingleBatchDataReader(CalibrationDataReader):
    """Yields a single calibration batch keyed by ``input``."""

    def __init__(self, input_tensor: np.ndarray) -> None:
        self._batches = iter([{"input": input_tensor}])

    def get_next(self):
        return next(self._batches, None)


def _prepare_conv_model(output_dir: str) -> tuple[str, str]:
    """Build a minimal Conv-only model. Returns (float_path, quantized_path)."""
    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])

    weight = numpy_helper.from_array(np.ones((1, 3, 3, 3), dtype=np.float32) * 0.1, name="conv_w")
    bias = numpy_helper.from_array(np.zeros(1, dtype=np.float32), name="conv_b")

    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "conv_w", "conv_b"],
        outputs=["output"],
        name="conv_node",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    )

    graph = helper.make_graph(
        nodes=[conv_node],
        name="ConvGraph",
        inputs=[input_vi],
        outputs=[output_vi],
        initializer=[weight, bias],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    float_path = Path(output_dir, "float_model.onnx").as_posix()
    quantized_path = Path(output_dir, "quantized_model.onnx").as_posix()
    onnx.save(model, float_path)
    return float_path, quantized_path


def _prepare_foldable_eltwise_model(output_dir: str) -> tuple[str, str]:
    """Build a model whose Shape->Gather subgraph is constant-folded by ORT pre-process.

    Conv keeps the model quantizable; the Shape->Gather subgraph (constant axis,
    fixed input shape) is folded away during pre-process so ``gather_out``
    disappears from the graph. Mirrors the YOLO12 head failure pattern that
    motivated the fix in ``apply_align_eltwise_quant_type``.
    """
    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])

    weight = numpy_helper.from_array(np.ones((1, 3, 3, 3), dtype=np.float32) * 0.1, name="conv_w")
    bias = numpy_helper.from_array(np.zeros(1, dtype=np.float32), name="conv_b")
    gather_idx = numpy_helper.from_array(np.array(0, dtype=np.int64), name="gather_idx")

    conv_node = helper.make_node(
        "Conv",
        inputs=["input", "conv_w", "conv_b"],
        outputs=["conv_out"],
        name="conv_node",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    cast_to_int = helper.make_node(
        "Cast", inputs=["conv_out"], outputs=["conv_int"], name="cast_to_int", to=TensorProto.INT64
    )
    shape_node = helper.make_node("Shape", inputs=["input"], outputs=["shape_out"], name="shape_node")
    gather_node = helper.make_node(
        "Gather", inputs=["shape_out", "gather_idx"], outputs=["gather_out"], name="gather_node", axis=0
    )
    add_node = helper.make_node("Add", inputs=["conv_int", "gather_out"], outputs=["add_out"], name="add_node")
    cast_back = helper.make_node("Cast", inputs=["add_out"], outputs=["output"], name="cast_back", to=TensorProto.FLOAT)

    graph = helper.make_graph(
        nodes=[conv_node, cast_to_int, shape_node, gather_node, add_node, cast_back],
        name="EltwiseFoldableGraph",
        inputs=[input_vi],
        outputs=[output_vi],
        initializer=[weight, bias, gather_idx],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8

    float_path = Path(output_dir, "float_model.onnx").as_posix()
    quantized_path = Path(output_dir, "quantized_model.onnx").as_posix()
    onnx.save(model, float_path)
    return float_path, quantized_path


def _build_two_branch_eltwise_model() -> onnx.ModelProto:
    """Build a Conv1, Conv2 -> Add model in-memory for direct unit tests."""
    input_vi = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    output_vi = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 4, 4])

    w1 = numpy_helper.from_array(np.ones((1, 3, 3, 3), dtype=np.float32) * 0.1, name="conv1_w")
    b1 = numpy_helper.from_array(np.zeros(1, dtype=np.float32), name="conv1_b")
    w2 = numpy_helper.from_array(np.ones((1, 3, 3, 3), dtype=np.float32) * 0.2, name="conv2_w")
    b2 = numpy_helper.from_array(np.zeros(1, dtype=np.float32), name="conv2_b")

    conv1 = helper.make_node(
        "Conv",
        inputs=["input", "conv1_w", "conv1_b"],
        outputs=["conv1_out"],
        name="conv1",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    conv2 = helper.make_node(
        "Conv",
        inputs=["input", "conv2_w", "conv2_b"],
        outputs=["conv2_out"],
        name="conv2",
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
    )
    add = helper.make_node("Add", inputs=["conv1_out", "conv2_out"], outputs=["output"], name="add_node")

    graph = helper.make_graph(
        nodes=[conv1, conv2, add],
        name="TwoBranchEltwiseGraph",
        inputs=[input_vi],
        outputs=[output_vi],
        initializer=[w1, b1, w2, b2],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _build_config(tensor_quant_overrides: dict | None = None) -> Config:
    """Return an A16W8 Config with optional ``TensorQuantOverrides`` injected."""
    inner = copy.deepcopy(A16W8_CONFIG)
    if tensor_quant_overrides is not None:
        inner.extra_options["TensorQuantOverrides"] = tensor_quant_overrides
    return Config(global_quant_config=inner)


class TestPruneStaleTensorQuantOverrides(unittest.TestCase):
    """End-to-end + unit coverage for ``prune_stale_tensor_quant_overrides``."""

    INPUT_TENSOR = np.random.RandomState(0).rand(1, 3, 4, 4).astype(np.float32)

    def _data_reader(self) -> _SingleBatchDataReader:
        return _SingleBatchDataReader(self.INPUT_TENSOR)

    @use_temporary_directory
    def test_stale_overrides_are_pruned_with_warning(self, tmpdir: str) -> None:
        """Stale override keys are dropped and listed in a single WARNING log."""
        float_path, quantized_path = _prepare_conv_model(tmpdir)
        config = _build_config(
            {
                "missing_tensor_one": [{"quant_type": "QInt16"}],
                "missing_tensor_two": [{"quant_type": "QInt16"}],
            }
        )

        with self.assertLogs(QUANTIZE_LOGGER_NAME, level="WARNING") as cm:
            ModelQuantizer(config).quantize_model(float_path, quantized_path, self._data_reader())

        self.assertTrue(Path(quantized_path).is_file())
        log_text = "\n".join(cm.output)
        self.assertIn("Dropped 2 TensorQuantOverrides entries", log_text)
        self.assertIn("not present in the model", log_text)
        self.assertIn("missing_tensor_one", log_text)
        self.assertIn("missing_tensor_two", log_text)

    def test_prune_returns_empty_when_no_overrides(self) -> None:
        """No overrides set → return [] without inspecting the graph."""
        extra: dict = {}
        dropped = prune_stale_tensor_quant_overrides(_build_two_branch_eltwise_model(), extra)
        self.assertEqual(dropped, [])
        self.assertNotIn("TensorQuantOverrides", extra)

    def test_prune_keeps_valid_overrides_without_warning(self) -> None:
        """All keys reference real tensors → no entry dropped, no log emitted."""
        extra = {"TensorQuantOverrides": {"input": [{"quant_type": "QInt16"}]}}
        dropped = prune_stale_tensor_quant_overrides(_build_two_branch_eltwise_model(), extra)
        self.assertEqual(dropped, [])
        self.assertEqual(extra["TensorQuantOverrides"], {"input": [{"quant_type": "QInt16"}]})

    def test_prune_singular_warning_for_one_stale_entry(self) -> None:
        """Single stale entry triggers the singular ``entry`` form of the warning."""
        extra = {"TensorQuantOverrides": {"definitely_not_a_tensor": [{"quant_type": "QInt16"}]}}
        with self.assertLogs(QUANTIZE_LOGGER_NAME, level="WARNING") as cm:
            dropped = prune_stale_tensor_quant_overrides(_build_two_branch_eltwise_model(), extra)
        self.assertEqual(dropped, ["definitely_not_a_tensor"])
        self.assertEqual(extra["TensorQuantOverrides"], {})
        self.assertIn("Dropped 1 TensorQuantOverrides entry", "\n".join(cm.output))


class TestApplyAlignEltwiseQuantType(unittest.TestCase):
    """End-to-end + unit coverage for ``apply_align_eltwise_quant_type``."""

    INPUT_TENSOR = np.random.RandomState(0).rand(1, 3, 4, 4).astype(np.float32)

    def _data_reader(self) -> _SingleBatchDataReader:
        return _SingleBatchDataReader(self.INPUT_TENSOR)

    @use_temporary_directory
    def test_does_not_raise_on_foldable_input(self, tmpdir: str) -> None:
        """Regression for QUAKR-475 (YOLO12 head failure).

        ``AlignEltwiseQuantType`` (enabled by ``A16W8_CONFIG``) used to inject
        ``gather_out`` into ``TensorQuantOverrides`` *before* pre-process. ORT
        then constant-folded ``gather_out`` away and rejected the override as
        "not present in the model". After the fix the injection runs *after*
        pre-process, so quantization completes end-to-end.
        """
        float_path, quantized_path = _prepare_foldable_eltwise_model(tmpdir)
        ModelQuantizer(_build_config()).quantize_model(float_path, quantized_path, self._data_reader())
        self.assertTrue(Path(quantized_path).is_file())

    def test_is_noop_when_flag_disabled(self) -> None:
        """``AlignEltwiseQuantType`` not set → return immediately, no mutation."""
        extra: dict = {}
        apply_align_eltwise_quant_type(
            _build_two_branch_eltwise_model(),
            extra,
            enable_npu_cnn=False,
            enable_npu_transformer=False,
            enable_dpu=False,
            quant_format=ExtendedQuantFormat.QDQ,
            activation_type=QuantType.QInt16,
        )
        self.assertEqual(extra, {})

    def test_warns_on_unsupported_context(self) -> None:
        """Flag enabled but NPU/DPU active or non-QDQ format → warn and skip injection."""
        extra: dict = {"AlignEltwiseQuantType": True}
        with self.assertLogs(QUANTIZE_LOGGER_NAME, level="WARNING") as cm:
            apply_align_eltwise_quant_type(
                _build_two_branch_eltwise_model(),
                extra,
                enable_npu_cnn=True,
                enable_npu_transformer=False,
                enable_dpu=False,
                quant_format=ExtendedQuantFormat.QDQ,
                activation_type=QuantType.QInt16,
            )
        self.assertNotIn("TensorQuantOverrides", extra)
        self.assertIn("AlignEltwiseQuantType only takes effect", "\n".join(cm.output))

    def test_updates_existing_user_override(self) -> None:
        """Eltwise input that already has a user override → its quant_type is rewritten."""
        extra = {
            "AlignEltwiseQuantType": True,
            "TensorQuantOverrides": {"conv1_out": [{"quant_type": QuantType.QInt8}]},
        }
        apply_align_eltwise_quant_type(
            _build_two_branch_eltwise_model(),
            extra,
            enable_npu_cnn=False,
            enable_npu_transformer=False,
            enable_dpu=False,
            quant_format=ExtendedQuantFormat.QDQ,
            activation_type=QuantType.QInt16,
        )
        overrides = extra["TensorQuantOverrides"]
        # Existing entry is *updated* (not replaced) to the activation type.
        self.assertEqual(overrides["conv1_out"], [{"quant_type": QuantType.QInt16}])
        # The other eltwise input is *inserted* with a fresh entry.
        self.assertEqual(overrides["conv2_out"], [{"quant_type": QuantType.QInt16}])


if __name__ == "__main__":
    unittest.main()
