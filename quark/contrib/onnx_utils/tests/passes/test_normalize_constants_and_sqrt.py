# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for the normalize_constants_and_sqrt pass.

This pass replaces Constant and Sqrt nodes with initializers when:
1. The tensor size is larger than 1024 bytes, OR
2. The node has fewer than 4 children

The tests verify:
1. Constant nodes are converted to initializers based on size/usage
2. Sqrt nodes with constant inputs are pre-computed and converted to initializers
3. Nodes that don't meet criteria are left unchanged
"""

import logging
from pathlib import Path

import numpy as np
import onnx
import onnx.helper
import pytest

import ryzenai_onnx_utils
import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.partitioner


@pytest.fixture(autouse=True)
def hide_ryzenai_logs(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, "ryzenai_onnx_utils"):
        yield


def run_pass(model: onnx.ModelProto, tmp_path: Path) -> tuple[onnx.ModelProto, int]:
    """Helper function to run the normalize_constants_and_sqrt pass."""
    extractor = ryzenai_onnx_utils.matcher.get_extractor(model)
    params = ryzenai_onnx_utils.ReplaceParams({}, Path(), Path(), tmp_path)
    model, replaced_num = ryzenai_onnx_utils.partitioner.partition(
        extractor,
        ["normalize_constants_and_sqrt"],
        params,
        {},
    )
    ryzenai_onnx_utils.matcher.save_initializers_with_extractor(extractor, Path.cwd(), None)
    return model, replaced_num


def create_constant_node(name: str, value: np.ndarray, output_name: str) -> onnx.NodeProto:
    """Helper to create a Constant node."""
    tensor = onnx.numpy_helper.from_array(value, name=output_name)
    return onnx.helper.make_node(
        "Constant",
        inputs=[],
        outputs=[output_name],
        name=name,
        value=tensor,
    )


def test_constant_large_size_converted(tmp_path: Path) -> None:
    """Test that a large constant (>1024 bytes) is converted to an initializer."""
    # Create a constant with > 1024 bytes (257 float32 values = 1028 bytes)
    large_value = np.random.randn(257).astype(np.float32)
    const_node = create_constant_node("large_const", large_value, "large_const_out")

    # Simple graph: Constant -> Add
    add_node = onnx.helper.make_node(
        "Add",
        inputs=["input", "large_const_out"],
        outputs=["output"],
        name="add",
    )

    graph = onnx.helper.make_graph(
        nodes=[const_node, add_node],
        name="test_graph",
        inputs=[onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [257])],
        outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [257])],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])

    # Run the pass
    new_model, replaced_num = run_pass(model, tmp_path)

    # Verify Constant node was removed
    const_nodes = [n for n in new_model.graph.node if n.op_type == "Constant"]
    assert len(const_nodes) == 0, "Constant node should be removed"

    # Verify initializer was added
    initializer_names = [init.name for init in new_model.graph.initializer]
    assert "large_const_out" in initializer_names, "Initializer should be added"


def test_constant_few_children_converted(tmp_path: Path) -> None:
    """Test that a constant with < 4 children is converted to an initializer."""
    # Small constant (< 1024 bytes) but with only 2 children
    small_value = np.array([1.0, 2.0, 3.0], dtype=np.float32)  # 12 bytes
    const_node = create_constant_node("small_const", small_value, "small_const_out")

    # Graph with 2 consumers of the constant
    add1 = onnx.helper.make_node("Add", inputs=["input", "small_const_out"], outputs=["out1"], name="add1")
    add2 = onnx.helper.make_node("Add", inputs=["input", "small_const_out"], outputs=["out2"], name="add2")
    concat = onnx.helper.make_node("Concat", inputs=["out1", "out2"], outputs=["output"], name="concat", axis=0)

    graph = onnx.helper.make_graph(
        nodes=[const_node, add1, add2, concat],
        name="test_graph",
        inputs=[onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [3])],
        outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [6])],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])

    # Run the pass
    new_model, replaced_num = run_pass(model, tmp_path)

    # Verify Constant node was removed
    const_nodes = [n for n in new_model.graph.node if n.op_type == "Constant"]
    assert len(const_nodes) == 0, "Constant node should be removed"

    # Verify initializer was added
    initializer_names = [init.name for init in new_model.graph.initializer]
    assert "small_const_out" in initializer_names, "Initializer should be added"


def test_constant_many_children_not_converted(tmp_path: Path) -> None:
    """Test that a small constant with >= 4 children is NOT converted."""
    # Small constant (< 1024 bytes) with 4 children
    small_value = np.array([1.0, 2.0], dtype=np.float32)  # 8 bytes
    const_node = create_constant_node("small_const", small_value, "small_const_out")

    # Graph with 4 consumers of the constant
    add1 = onnx.helper.make_node("Add", inputs=["input", "small_const_out"], outputs=["out1"], name="add1")
    add2 = onnx.helper.make_node("Add", inputs=["input", "small_const_out"], outputs=["out2"], name="add2")
    add3 = onnx.helper.make_node("Add", inputs=["input", "small_const_out"], outputs=["out3"], name="add3")
    add4 = onnx.helper.make_node("Add", inputs=["input", "small_const_out"], outputs=["out4"], name="add4")
    concat = onnx.helper.make_node(
        "Concat", inputs=["out1", "out2", "out3", "out4"], outputs=["output"], name="concat", axis=0
    )

    graph = onnx.helper.make_graph(
        nodes=[const_node, add1, add2, add3, add4, concat],
        name="test_graph",
        inputs=[onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2])],
        outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [8])],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])

    new_model, replaced_num = run_pass(model, tmp_path)

    # Verify Constant node is still present
    const_nodes = [n for n in new_model.graph.node if n.op_type == "Constant"]
    assert len(const_nodes) == 1, "Constant node should still be present"

    # Verify initializer was NOT added
    initializer_names = [init.name for init in new_model.graph.initializer]
    assert "small_const_out" not in initializer_names, "Initializer should not be added"


def test_sqrt_with_constant_input_converted(tmp_path: Path) -> None:
    """Test that Sqrt with constant input is pre-computed and converted."""
    # Create initializer for Sqrt input
    input_value = np.array([4.0, 9.0, 16.0], dtype=np.float32)
    initializer = onnx.numpy_helper.from_array(input_value, name="sqrt_input")

    # Sqrt node with 2 children (< 4)
    sqrt_node = onnx.helper.make_node("Sqrt", inputs=["sqrt_input"], outputs=["sqrt_out"], name="sqrt")
    mul1 = onnx.helper.make_node("Mul", inputs=["input", "sqrt_out"], outputs=["out1"], name="mul1")
    mul2 = onnx.helper.make_node("Mul", inputs=["input", "sqrt_out"], outputs=["output"], name="mul2")

    graph = onnx.helper.make_graph(
        nodes=[sqrt_node, mul1, mul2],
        name="test_graph",
        inputs=[onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [3])],
        outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [3])],
        initializer=[initializer],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])

    new_model, replaced_num = run_pass(model, tmp_path)

    # Verify Sqrt node was removed
    sqrt_nodes = [n for n in new_model.graph.node if n.op_type == "Sqrt"]
    assert len(sqrt_nodes) == 0, "Sqrt node should be removed"

    # Verify initializer was added with correct sqrt values
    sqrt_init = None
    for init in new_model.graph.initializer:
        if init.name == "sqrt_input":
            sqrt_init = init
            break
    assert sqrt_init is not None, "Sqrt output initializer should be added"

    # Verify the values are correct
    sqrt_values = onnx.numpy_helper.to_array(sqrt_init)
    expected_values = np.sqrt(input_value)
    np.testing.assert_array_almost_equal(sqrt_values, expected_values)


def test_sqrt_large_output_converted(tmp_path: Path) -> None:
    """Test that Sqrt with large output (>1024 bytes) is converted."""
    # Create large initializer (257 float32 values = 1028 bytes)
    input_value = np.random.randn(257).astype(np.float32) ** 2  # Ensure positive for sqrt
    initializer = onnx.numpy_helper.from_array(input_value, name="large_sqrt_input")

    # Sqrt node with many children (would normally not be converted)
    sqrt_node = onnx.helper.make_node("Sqrt", inputs=["large_sqrt_input"], outputs=["sqrt_out"], name="sqrt")
    mul1 = onnx.helper.make_node("Mul", inputs=["input", "sqrt_out"], outputs=["out1"], name="mul1")
    mul2 = onnx.helper.make_node("Mul", inputs=["input", "sqrt_out"], outputs=["out2"], name="mul2")
    mul3 = onnx.helper.make_node("Mul", inputs=["input", "sqrt_out"], outputs=["out3"], name="mul3")
    mul4 = onnx.helper.make_node("Mul", inputs=["input", "sqrt_out"], outputs=["out4"], name="mul4")
    mul5 = onnx.helper.make_node("Mul", inputs=["input", "sqrt_out"], outputs=["out5"], name="mul5")
    add = onnx.helper.make_node("Add", inputs=["out1", "out2"], outputs=["output"], name="add")

    graph = onnx.helper.make_graph(
        nodes=[sqrt_node, mul1, mul2, mul3, mul4, mul5, add],
        name="test_graph",
        inputs=[onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [257])],
        outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [257])],
        initializer=[initializer],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])

    new_model, replaced_num = run_pass(model, tmp_path)

    # Verify Sqrt node was removed (because output > 1024 bytes)
    sqrt_nodes = [n for n in new_model.graph.node if n.op_type == "Sqrt"]
    assert len(sqrt_nodes) == 0, "Sqrt node should be removed due to large output"


def test_sqrt_non_constant_input_not_converted(tmp_path: Path) -> None:
    """Test that Sqrt with non-constant input is NOT converted."""
    # Sqrt node with dynamic input
    sqrt_node = onnx.helper.make_node("Sqrt", inputs=["input"], outputs=["output"], name="sqrt")
    mul1 = onnx.helper.make_node("Mul", inputs=["input", "output"], outputs=["out1"], name="mul1")

    graph = onnx.helper.make_graph(
        nodes=[sqrt_node, mul1],
        name="test_graph",
        inputs=[onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [3])],
        outputs=[onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [3])],
    )

    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 14)])

    new_model, replaced_num = run_pass(model, tmp_path)

    # Verify Sqrt node is still present
    sqrt_nodes = [n for n in new_model.graph.node if n.op_type == "Sqrt"]
    assert len(sqrt_nodes) == 1, "Sqrt node should still be present"
