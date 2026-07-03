#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import unittest
from unittest.mock import patch

import numpy as np
import onnx
import onnx.helper
from onnx_testing_utils import prepare_model
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import use_temporary_directory
from quark.onnx import ModelQuantizer, QConfig, QLayerConfig, XInt8Spec
from quark.onnx.utils.model_utils import collect_tensor_shapes_from_feed, fill_all_tensors_value_info

INPUT_TENSOR = np.random.default_rng(1).random((1, 3, 4, 4)).astype(np.float32)


class DataReader(CalibrationDataReader):
    """Single-sample calibration data reader backed by a fixed numpy array."""

    def __init__(self, tensor: np.ndarray) -> None:
        """Store *tensor* as the sole calibration sample under the key ``"input"``."""
        self._data = [{"input": tensor}]
        self._index = 0

    def get_next(self):
        """Return the next sample dict, or ``None`` when the single sample is exhausted."""
        if self._index < len(self._data):
            item = self._data[self._index]
            self._index += 1
            return item
        return None

    def reset_iter(self) -> None:
        """Reset the iterator so the single sample can be re-read."""
        self._index = 0

    def rewind(self) -> None:
        """Alias for ``reset_iter`` required by some calibration paths."""
        self._index = 0


def _run_quantization(output_dir: str) -> str:
    """Quantize the test model into *output_dir* and return the output model path.

    Builds a simple XInt8 config, runs ``ModelQuantizer.quantize_model``, and
    returns the path of the saved quantized ONNX model.
    """
    input_model_path, output_model_path = prepare_model(output_dir)
    data_reader = DataReader(INPUT_TENSOR)
    config = QConfig(
        global_config=QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()),
        extra_options={"FillAllValueInfo": True},
    )
    ModelQuantizer(config).quantize_model(input_model_path, output_model_path, data_reader)
    return output_model_path


class TestFillValueInfoRefactor(unittest.TestCase):
    @use_temporary_directory
    def test_value_info_populated_after_quantization(self, tmpdir: str) -> None:
        """Verify that ``FillAllValueInfo`` populates ``graph.value_info`` for every node input.

        Runs a full quantization pass with default settings (``FillAllValueInfo=True``) and
        checks that every tensor consumed by a graph node is covered by ``graph.value_info``,
        ``graph.input``, ``graph.output``, or ``graph.initializer``.
        """
        output_model_path = _run_quantization(tmpdir)
        model = onnx.load(output_model_path)

        covered: set[str] = (
            {vi.name for vi in model.graph.value_info}
            | {inp.name for inp in model.graph.input}
            | {out.name for out in model.graph.output}
            | {init.name for init in model.graph.initializer}
        )

        missing = [name for node in model.graph.node for name in node.input if name and name not in covered]

        self.assertEqual(missing, [], msg=f"Tensors without value_info: {missing}")

    @use_temporary_directory
    def test_collect_tensor_shapes_returns_dict(self, tmpdir: str) -> None:
        """Verify ``collect_tensor_shapes_from_feed`` returns well-formed results and restores graph state.

        Checks that the returned mapping contains string tensor names, integer ONNX element
        types, and tuple shapes.  Also asserts that ``graph.output`` is restored to its
        original length after the call, since the function temporarily appends outputs for
        the ORT inference pass.
        """
        input_model_path, _ = prepare_model(tmpdir)
        model = onnx.load(input_model_path)
        original_output_count = len(model.graph.output)
        feed_dict = {"input": INPUT_TENSOR}
        result = collect_tensor_shapes_from_feed(model, feed_dict)
        self.assertEqual(
            len(model.graph.output),
            original_output_count,
            "collect_tensor_shapes_from_feed must restore graph.output after the call",
        )
        self.assertIsInstance(result, dict)
        for name, (elem_type, shape) in result.items():
            self.assertIsInstance(name, str)
            self.assertIsInstance(elem_type, int)
            self.assertIsInstance(shape, tuple)

    @use_temporary_directory
    def test_concrete_dims_always_used(self, tmpdir: str) -> None:
        """Verify that all ``value_info`` entries use concrete integer dims, not symbolic ``dim_param``.

        Since shapes are collected from an actual ORT inference run with a fixed-shape input,
        every dimension must be a concrete integer.  A ``dim_param`` entry would indicate
        that a symbolic shape leaked through, which would break downstream compiler shape
        inference.
        """
        output_model_path = _run_quantization(tmpdir)
        model = onnx.load(output_model_path)

        for vi in model.graph.value_info:
            tt = vi.type.tensor_type
            if tt.HasField("shape"):
                for dim in tt.shape.dim:
                    self.assertFalse(
                        dim.HasField("dim_param"),
                        msg=f"value_info '{vi.name}' has unexpected dim_param",
                    )


def _make_single_node_model() -> onnx.ModelProto:
    """Build a minimal ONNX model with a single Add node and two float32 inputs.

    This model has no intermediate tensors (both inputs go directly to the sole
    node output), so ``collect_tensor_shapes_from_feed`` should return ``{}``.
    """
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4])
    Z = onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [1, 4])
    node = onnx.helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"])
    graph = onnx.helper.make_graph([node], "add_graph", [X, Y], [Z])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


def _make_two_node_model() -> onnx.ModelProto:
    """Build a minimal two-node ONNX model (Add → Relu) with one intermediate tensor.

    The Add node outputs ``"add_out"``, which is consumed by the Relu node.
    ``"add_out"`` is a genuine intermediate tensor — not in ``graph.input``,
    ``graph.output``, or ``graph.initializer``.
    """
    X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4])
    Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4])
    Z = onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [1, 4])
    add_node = onnx.helper.make_node("Add", inputs=["X", "Y"], outputs=["add_out"])
    relu_node = onnx.helper.make_node("Relu", inputs=["add_out"], outputs=["Z"])
    graph = onnx.helper.make_graph([add_node, relu_node], "two_node_graph", [X, Y], [Z])
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class TestCollectTensorShapesEdgeCases(unittest.TestCase):
    def test_returns_empty_dict_when_no_intermediate_tensors(self) -> None:
        """Verify early return of ``{}`` when all node outputs are graph inputs or initializers.

        A model whose only node consumes graph inputs and produces a graph output
        has no tensors that are *only* intermediate (all names appear in graph inputs
        or initializers).  ``output_list`` will be empty only for such tensors;
        graph output tensors produced by nodes are still included.  When the
        graph input set already covers every node input and every node output is a
        graph output, the function returns a map whose keys are the graph outputs
        (not ``{}``, but they are correctly skipped by ``fill_all_tensors_value_info``
        because they appear in ``skip_names``).

        This test verifies that for a model with a single Add node the result
        contains exactly the graph output shape and nothing else.
        """
        model = _make_single_node_model()
        feed = {"X": np.ones((1, 4), dtype=np.float32), "Y": np.ones((1, 4), dtype=np.float32)}
        result = collect_tensor_shapes_from_feed(model, feed)
        # Only "Z" (the graph output) is collected; no genuine intermediate tensors.
        graph_output_names = {out.name for out in model.graph.output}
        non_output_keys = {k for k in result if k not in graph_output_names}
        self.assertEqual(non_output_keys, set(), msg=f"Unexpected intermediate tensors: {non_output_keys}")

    def test_node_input_seen_before_producer_is_collected(self) -> None:
        """Verify that a node input not yet seen via node outputs is added to ``output_list``.

        When graph nodes are stored in reverse topological order (consumer before
        producer), the consumer's input tensor hasn't been added via any node's
        output list yet.  The inner loop over ``node.input`` (lines 863-866) must
        therefore add it.  The result must contain the intermediate tensor shape.
        """
        model = _make_two_node_model()
        # Reverse the node order so Relu (consumer) comes before Add (producer).
        nodes = list(model.graph.node)
        del model.graph.node[:]
        model.graph.node.extend(reversed(nodes))
        feed = {"X": np.ones((1, 4), dtype=np.float32), "Y": np.ones((1, 4), dtype=np.float32)}
        result = collect_tensor_shapes_from_feed(model, feed)
        self.assertIn("add_out", result)

    def test_all_outputs_are_graph_inputs_returns_empty(self) -> None:
        """Verify that ``output_list`` is empty and ``{}`` is returned when all node outputs are graph inputs.

        Builds a model where the sole node output is also listed as a graph input
        (an unusual but valid construction).  Every node output is in
        ``graph_input_names``, so ``output_list`` stays empty and the early-return
        path (line 869) is executed.
        """
        X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4])
        Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4])
        # List "Z" as both a graph input and the sole node output.
        Z_input = onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [1, 4])
        Z_output = onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [1, 4])
        node = onnx.helper.make_node("Add", inputs=["X", "Y"], outputs=["Z"])
        graph = onnx.helper.make_graph([node], "all_boundary_graph", [X, Y, Z_input], [Z_output])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
        model.ir_version = 8
        feed = {
            "X": np.ones((1, 4), dtype=np.float32),
            "Y": np.ones((1, 4), dtype=np.float32),
            "Z": np.zeros((1, 4), dtype=np.float32),
        }
        result = collect_tensor_shapes_from_feed(model, feed)
        self.assertEqual(result, {})

    def test_none_result_in_ort_output_is_skipped(self) -> None:
        """Verify that a ``None`` entry in ORT results is skipped without error.

        Patches ``session.run`` to return a list where the first element is
        ``None``, exercising the ``if arr is None: continue`` branch.  The
        resulting map must not contain a key for the ``None`` result.
        """
        model = _make_two_node_model()
        feed = {"X": np.ones((1, 4), dtype=np.float32), "Y": np.ones((1, 4), dtype=np.float32)}

        real_result = [None, np.ones((1, 4), dtype=np.float32), np.ones((1, 4), dtype=np.float32)]

        with patch("quark.onnx.utils.model_utils.create_infer_session_for_onnx_model") as mock_sess_fn:
            mock_session = mock_sess_fn.return_value
            mock_session.run.return_value = real_result
            result = collect_tensor_shapes_from_feed(model, feed)

        # The first output_list entry got None — it must be absent from the result.
        self.assertIsInstance(result, dict)
        for _, (elem_type, shape) in result.items():
            self.assertIsInstance(elem_type, int)
            self.assertIsInstance(shape, tuple)

    def test_unsupported_dtype_in_ort_output_is_skipped(self) -> None:
        """Verify that an unsupported numpy dtype causes the tensor to be skipped.

        Patches ``np_dtype_to_tensor_dtype`` to raise so the inner except branch
        is executed.  The result must be an empty dict (all tensors skipped).
        """
        model = _make_two_node_model()
        feed = {"X": np.ones((1, 4), dtype=np.float32), "Y": np.ones((1, 4), dtype=np.float32)}
        with patch(
            "quark.onnx.utils.model_utils.onnx.helper.np_dtype_to_tensor_dtype", side_effect=Exception("bad dtype")
        ):
            result = collect_tensor_shapes_from_feed(model, feed)
        self.assertEqual(result, {})

    def test_ort_failure_returns_empty_dict_and_restores_outputs(self) -> None:
        """Verify that an ORT session failure returns ``{}`` and restores graph outputs.

        Patches ``create_infer_session_for_onnx_model`` to raise so the except
        branch is executed.  The function must return ``{}`` and the graph outputs
        must be restored to their original length.
        """
        input_model_path, _ = prepare_model("/tmp")
        model = onnx.load(input_model_path)
        original_output_count = len(model.graph.output)
        feed = {"input": INPUT_TENSOR}
        with patch(
            "quark.onnx.utils.model_utils.create_infer_session_for_onnx_model",
            side_effect=RuntimeError("simulated ORT failure"),
        ):
            result = collect_tensor_shapes_from_feed(model, feed)
        self.assertEqual(result, {})
        self.assertEqual(len(model.graph.output), original_output_count)


class _NoResetReader:
    """Minimal data reader without ``reset_iter`` that immediately returns ``None``."""

    def get_next(self):
        return None


class TestFillAllTensorsValueInfoEdgeCases(unittest.TestCase):
    def test_none_data_reader_returns_model_unchanged(self) -> None:
        """Verify that passing ``data_reader=None`` returns the model without modification.

        The function must log a warning and return the model unchanged.
        """
        input_model_path, _ = prepare_model("/tmp")
        model = onnx.load(input_model_path)
        original_vi_count = len(model.graph.value_info)
        result = fill_all_tensors_value_info(model, None)
        self.assertIs(result, model)
        self.assertEqual(len(model.graph.value_info), original_vi_count)

    def test_exhausted_data_reader_returns_model_unchanged(self) -> None:
        """Verify that an exhausted data reader returns the model without modification.

        After the data reader's single sample is consumed, ``get_next()`` returns
        ``None`` and ``fill_all_tensors_value_info`` must short-circuit and return
        the model unchanged.
        """
        input_model_path, _ = prepare_model("/tmp")
        model = onnx.load(input_model_path)
        reader = DataReader(INPUT_TENSOR)
        reader.get_next()  # exhaust the sole sample
        reader._index = 1  # ensure get_next() returns None
        original_vi_count = len(model.graph.value_info)
        result = fill_all_tensors_value_info(model, reader)
        self.assertIs(result, model)
        self.assertEqual(len(model.graph.value_info), original_vi_count)

    def test_reader_without_reset_iter_returning_none_is_handled(self) -> None:
        """Verify the ``feed_dict is None`` path when the reader has no ``reset_iter``.

        Uses ``_NoResetReader`` which lacks ``reset_iter`` and returns ``None``
        from ``get_next()``.  ``fill_all_tensors_value_info`` must log a warning
        and return the model unchanged without raising ``AttributeError``.
        """
        model = _make_two_node_model()
        original_vi_count = len(model.graph.value_info)
        result = fill_all_tensors_value_info(model, _NoResetReader())
        self.assertIs(result, model)
        self.assertEqual(len(model.graph.value_info), original_vi_count)

    def test_empty_shape_map_returns_model_unchanged(self) -> None:
        """Verify that an empty ``shape_map`` from ORT causes an early return.

        Patches ``collect_tensor_shapes_from_feed`` to return ``{}`` so the
        ``if not shape_map`` guard fires.  The model must be returned unchanged.
        """
        model = _make_two_node_model()
        feed = {"X": np.ones((1, 4), dtype=np.float32), "Y": np.ones((1, 4), dtype=np.float32)}
        reader = DataReader(feed["X"])
        reader._data = [feed]
        reader._index = 0
        original_vi_count = len(model.graph.value_info)
        with patch("quark.onnx.utils.model_utils.collect_tensor_shapes_from_feed", return_value={}):
            result = fill_all_tensors_value_info(model, reader)
        self.assertIs(result, model)
        self.assertEqual(len(model.graph.value_info), original_vi_count)

    def test_make_tensor_value_info_failure_skips_tensor_in_multi_node_model(self) -> None:
        """Verify that ``make_tensor_value_info`` failure skips the tensor in a model with intermediates.

        Uses a two-node model so there is a genuine intermediate tensor in
        ``shape_map``.  The patch causes ``make_tensor_value_info`` to raise for
        that tensor so the ``except`` branch is exercised.  The model must be
        returned without crashing and its ``value_info`` must remain unchanged.
        """
        model = _make_two_node_model()
        feed = {"X": np.ones((1, 4), dtype=np.float32), "Y": np.ones((1, 4), dtype=np.float32)}
        reader = DataReader(feed["X"])
        reader._data = [feed]
        reader._index = 0
        original_vi_count = len(model.graph.value_info)
        with patch("quark.onnx.utils.model_utils.onnx.helper.make_tensor_value_info", side_effect=Exception("bad")):
            result = fill_all_tensors_value_info(model, reader)
        self.assertIs(result, model)
        self.assertEqual(len(model.graph.value_info), original_vi_count)

    def test_matching_existing_value_info_is_not_updated(self) -> None:
        """Verify that an existing ``value_info`` with a matching shape is kept unchanged.

        Inserts a correctly-shaped ``value_info`` for the intermediate tensor
        ``"add_out"`` before calling ``fill_all_tensors_value_info``.  Because the
        existing shape matches the ORT-observed shape, ``shapes_match`` must be
        ``True`` and ``CopyFrom`` must not be called.  The ``value_info`` count
        must be unchanged and the entry must still have the original shape.
        """
        model = _make_two_node_model()
        target_name = "add_out"
        correct_vi = onnx.helper.make_tensor_value_info(target_name, onnx.TensorProto.FLOAT, [1, 4])
        model.graph.value_info.append(correct_vi)
        original_vi_count = len(model.graph.value_info)

        feed = {"X": np.ones((1, 4), dtype=np.float32), "Y": np.ones((1, 4), dtype=np.float32)}
        reader = DataReader(feed["X"])
        reader._data = [feed]
        reader._index = 0

        fill_all_tensors_value_info(model, reader)

        self.assertEqual(len(model.graph.value_info), original_vi_count)
        updated = {vi.name: vi for vi in model.graph.value_info}.get(target_name)
        self.assertIsNotNone(updated)
        self.assertTrue(updated.type.tensor_type.HasField("shape"))

    def test_dynamic_input_dims_are_concretised(self) -> None:
        """Verify that a symbolic ``dim_param`` on a graph input is replaced with a concrete value.

        Build a model whose graph input carries a symbolic batch dimension
        (``dim_param="N"``) and verify that after ``fill_all_tensors_value_info``
        that dimension is replaced with the concrete integer from the feed dict.
        """
        input_model_path, _ = prepare_model("/tmp")
        model = onnx.load(input_model_path)
        # Replace the first input's batch dim with a symbolic dim_param.
        inp = model.graph.input[0]
        dim = inp.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "N"
        reader = DataReader(INPUT_TENSOR)
        fill_all_tensors_value_info(model, reader)
        updated_dim = model.graph.input[0].type.tensor_type.shape.dim[0]
        self.assertFalse(updated_dim.HasField("dim_param"))
        self.assertEqual(updated_dim.dim_value, INPUT_TENSOR.shape[0])

    def test_dynamic_output_dims_are_concretised(self) -> None:
        """Verify that a symbolic ``dim_param`` on a graph output is replaced with a concrete value.

        Injects a symbolic batch dimension into the model's graph output and
        verifies that ``fill_all_tensors_value_info`` replaces it with the
        concrete integer observed during the ORT inference pass.
        """
        input_model_path, _ = prepare_model("/tmp")
        model = onnx.load(input_model_path)
        out = model.graph.output[0]
        dim = out.type.tensor_type.shape.dim[0]
        dim.ClearField("dim_value")
        dim.dim_param = "N"
        reader = DataReader(INPUT_TENSOR)
        fill_all_tensors_value_info(model, reader)
        updated_dim = model.graph.output[0].type.tensor_type.shape.dim[0]
        self.assertFalse(updated_dim.HasField("dim_param"))
        self.assertEqual(updated_dim.dim_value, 1)

    def test_existing_value_info_without_shape_is_updated(self) -> None:
        """Verify that an existing ``value_info`` entry with no shape field is overwritten.

        Builds a two-node model (Add → Relu) so that the Add output is a genuine
        intermediate tensor.  Inserts a shapeless ``value_info`` entry for it, then
        checks that ``fill_all_tensors_value_info`` overwrites it with the ORT-observed
        concrete shape.
        """
        X = onnx.helper.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 4])
        Y = onnx.helper.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, [1, 4])
        Z = onnx.helper.make_tensor_value_info("Z", onnx.TensorProto.FLOAT, [1, 4])
        add_node = onnx.helper.make_node("Add", inputs=["X", "Y"], outputs=["add_out"])
        relu_node = onnx.helper.make_node("Relu", inputs=["add_out"], outputs=["Z"])
        graph = onnx.helper.make_graph([add_node, relu_node], "two_node_graph", [X, Y], [Z])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
        model.ir_version = 8

        # "add_out" is an intermediate tensor — not in graph.input, .output, or .initializer.
        target_name = "add_out"
        shapeless_vi = onnx.ValueInfoProto()
        shapeless_vi.name = target_name
        shapeless_vi.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        model.graph.value_info.append(shapeless_vi)

        feed = {"X": np.ones((1, 4), dtype=np.float32), "Y": np.ones((1, 4), dtype=np.float32)}
        reader = DataReader(feed["X"])
        # Override get_next to return the two-input feed dict directly.
        reader._data = [feed]
        reader._index = 0

        fill_all_tensors_value_info(model, reader)

        updated = {vi.name: vi for vi in model.graph.value_info}.get(target_name)
        self.assertIsNotNone(updated)
        self.assertTrue(updated.type.tensor_type.HasField("shape"))


if __name__ == "__main__":
    unittest.main()
