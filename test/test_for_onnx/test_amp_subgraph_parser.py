#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json
from pathlib import Path

import onnx
import onnx.helper as oh
import pytest

from quark.onnx.algorithm.mprecision.subgraph_parser import parse_subgraph_json


def _make_linear_model() -> onnx.ModelProto:
    """Conv_0 -> Relu_1 -> Conv_2 -> MatMul_3"""
    X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 4, 4])
    W0 = oh.make_tensor_value_info("W0", onnx.TensorProto.FLOAT, [8, 3, 1, 1])
    W2 = oh.make_tensor_value_info("W2", onnx.TensorProto.FLOAT, [8, 8, 1, 1])
    W3 = oh.make_tensor_value_info("W3", onnx.TensorProto.FLOAT, [8, 4])
    Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)

    conv0 = oh.make_node("Conv", ["X", "W0"], ["conv0_out"], name="Conv_0")
    relu1 = oh.make_node("Relu", ["conv0_out"], ["relu1_out"], name="Relu_1")
    conv2 = oh.make_node("Conv", ["relu1_out", "W2"], ["conv2_out"], name="Conv_2")
    matmul3 = oh.make_node("MatMul", ["conv2_out", "W3"], ["Y"], name="MatMul_3")

    graph = oh.make_graph([conv0, relu1, conv2, matmul3], "g", [X, W0, W2, W3], [Y])
    return oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])


def _write_json(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "subgraphs.json"
    p.write_text(json.dumps(data))
    return p


def test_parse_valid_subgraph(tmp_path):
    model = _make_linear_model()
    data = {
        "num_subgraphs": 2,
        "subgraphs": [
            {"name": "first_block", "start_nodes": ["Conv_0"], "end_nodes": ["Relu_1"]},
            {"name": "second_block", "start_nodes": ["Conv_2"], "end_nodes": ["MatMul_3"]},
        ],
    }
    specs = parse_subgraph_json(_write_json(tmp_path, data), model, model)
    assert len(specs) == 2
    assert specs[0].name == "first_block"
    assert "Conv_0" in specs[0].resolved_nodes
    assert "Relu_1" in specs[0].resolved_nodes
    assert specs[1].name == "second_block"
    assert "Conv_2" in specs[1].resolved_nodes
    assert "MatMul_3" in specs[1].resolved_nodes


def test_num_subgraphs_mismatch_raises(tmp_path):
    """num_subgraphs must match the actual subgraph count when present."""
    model = _make_linear_model()
    data = {
        "num_subgraphs": 99,
        "subgraphs": [
            {"name": "first_block", "start_nodes": ["Conv_0"], "end_nodes": ["Relu_1"]},
        ],
    }
    with pytest.raises(ValueError, match="num_subgraphs"):
        parse_subgraph_json(_write_json(tmp_path, data), model, model)


def test_parse_missing_node_raises(tmp_path):
    """A node missing from the *float* model raises ValueError."""
    model = _make_linear_model()
    data = {
        "subgraphs": [
            {"name": "bad", "start_nodes": ["NonExistentNode"], "end_nodes": ["Relu_1"]},
        ]
    }
    with pytest.raises(ValueError, match="NonExistentNode"):
        parse_subgraph_json(_write_json(tmp_path, data), model, model)


def test_parse_quant_missing_nodes_excluded(tmp_path):
    """Nodes absent in the *quant* model (graph optimisation) are excluded from the
    subgraph; the surviving nodes still form the subgraph."""
    float_model = _make_linear_model()
    # Quant model has Conv_2 optimised away; MatMul_3 still exists.
    X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 4, 4])
    W0 = oh.make_tensor_value_info("W0", onnx.TensorProto.FLOAT, [8, 3, 1, 1])
    W3 = oh.make_tensor_value_info("W3", onnx.TensorProto.FLOAT, [8, 4])
    Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)
    conv0 = oh.make_node("Conv", ["X", "W0"], ["conv0_out"], name="Conv_0")
    matmul3 = oh.make_node("MatMul", ["conv0_out", "W3"], ["Y"], name="MatMul_3")
    graph = oh.make_graph([conv0, matmul3], "g", [X, W0, W3], [Y])
    quant_model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])

    data = {
        "subgraphs": [
            # Conv_2 is missing from quant_model; MatMul_3 survives → kept in subgraph
            {"name": "second_block", "start_nodes": ["Conv_2"], "end_nodes": ["MatMul_3"]},
        ]
    }
    specs = parse_subgraph_json(_write_json(tmp_path, data), float_model, quant_model)
    second = next(s for s in specs if s.name == "second_block")
    assert "Conv_2" not in second.resolved_nodes
    assert "MatMul_3" in second.resolved_nodes


def test_parse_quantized_flag_uses_quant_model(tmp_path):
    """Top-level quantized=true: boundary nodes validated against the quant model."""
    float_model = _make_linear_model()
    data = {
        "quantized": True,
        "num_subgraphs": 1,
        "subgraphs": [
            {"name": "first_block", "start_nodes": ["Conv_0"], "end_nodes": ["Relu_1"]},
        ],
    }
    specs = parse_subgraph_json(_write_json(tmp_path, data), float_model, float_model)
    first = next(s for s in specs if s.name == "first_block")
    assert "Conv_0" in first.resolved_nodes
    assert "Relu_1" in first.resolved_nodes


def test_parse_quantized_flag_missing_raises(tmp_path):
    """Top-level quantized=true: missing quant-model boundary node raises ValueError."""
    float_model = _make_linear_model()
    X = oh.make_tensor_value_info("X", onnx.TensorProto.FLOAT, [1, 3, 4, 4])
    W0 = oh.make_tensor_value_info("W0", onnx.TensorProto.FLOAT, [8, 3, 1, 1])
    W3 = oh.make_tensor_value_info("W3", onnx.TensorProto.FLOAT, [8, 4])
    Y = oh.make_tensor_value_info("Y", onnx.TensorProto.FLOAT, None)
    conv0 = oh.make_node("Conv", ["X", "W0"], ["conv0_out"], name="Conv_0")
    matmul3 = oh.make_node("MatMul", ["conv0_out", "W3"], ["Y"], name="MatMul_3")
    graph = oh.make_graph([conv0, matmul3], "g", [X, W0, W3], [Y])
    quant_model = oh.make_model(graph, opset_imports=[oh.make_opsetid("", 17)])

    data = {
        "quantized": True,
        "subgraphs": [
            {"name": "second_block", "start_nodes": ["Conv_2"], "end_nodes": ["MatMul_3"]},
        ],
    }
    with pytest.raises(ValueError, match="Conv_2"):
        parse_subgraph_json(_write_json(tmp_path, data), float_model, quant_model)


def test_parse_overlapping_nodes_warns_and_removes(tmp_path):
    model = _make_linear_model()
    data = {
        "subgraphs": [
            {"name": "a", "start_nodes": ["Conv_0"], "end_nodes": ["Conv_2"]},
            {"name": "b", "start_nodes": ["Relu_1"], "end_nodes": ["MatMul_3"]},
        ]
    }
    specs = parse_subgraph_json(_write_json(tmp_path, data), model, model)
    b_spec = next(s for s in specs if s.name == "b")
    assert "Relu_1" not in b_spec.resolved_nodes
    assert "Conv_2" not in b_spec.resolved_nodes


def test_ungrouped_subgraph_created(tmp_path):
    model = _make_linear_model()
    data = {
        "subgraphs": [
            {"name": "first_block", "start_nodes": ["Conv_0"], "end_nodes": ["Relu_1"]},
        ]
    }
    specs = parse_subgraph_json(_write_json(tmp_path, data), model, model)
    ungrouped = [s for s in specs if s.name == "__ungrouped__"]
    assert len(ungrouped) == 1
    assert "Conv_2" in ungrouped[0].resolved_nodes
    assert "MatMul_3" in ungrouped[0].resolved_nodes
