#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

import onnx
import yaml

from quark.experimental.cli.main import main as cli


class Builder:
    @staticmethod
    def q_linear_matmul(input_name: str, output_name: str, idx: int = 0) -> onnx.NodeProto:
        return onnx.helper.make_node(
            "QLinearMatMul",
            inputs=[input_name, "a_scale", "a_zero_point", "b_scale", "b_zero_point", "b", "y_scale", "y_zero_point"],
            outputs=[output_name],
            name=f"QLinearMatMul_{idx}",
        )

    @staticmethod
    def dequant(input_name: str, output_name: str, idx: int = 0) -> onnx.NodeProto:
        return onnx.helper.make_node(
            "DequantizeLinear",
            inputs=[input_name, "scale", "zero_point"],
            outputs=[output_name],
            name=f"DequantizeLinear_{idx}",
        )

    @staticmethod
    def quant(input_name: str, output_name: str, idx: int = 0) -> onnx.NodeProto:
        return onnx.helper.make_node(
            "QuantizeLinear",
            inputs=[input_name, "scale", "zero_point"],
            outputs=[output_name],
            name=f"QuantizeLinear_{idx}",
        )


def get_simple_nodes():
    """
    Builds nodes for a simple graph with a back-to-back QDQ pair:

    input -> QLinearMatMul (0) -> DequantizeLinear -> QuantizeLinear -> QLinearMatMul (1) -> output
    """
    qmatmul_0 = Builder.q_linear_matmul("input", "qmatmul_0_out", 0)
    dequant = Builder.dequant("qmatmul_0_out", "dquant_out", 0)
    quant = Builder.quant("dquant_out", "quant_out", 0)
    qmatmul_1 = Builder.q_linear_matmul("quant_out", "output", 1)
    return [qmatmul_0, dequant, quant, qmatmul_1]


def get_advanced_nodes():
    """
    Builds nodes for a more complex graph with a back-to-back QDQ pair:

    input -> QLinearMatMul (0) -> DequantizeLinear -> QuantizeLinear -> QLinearMatMul (1) -> output
                                                    \
                                                     -> QLinearMatMul (2) -> output_2
    """
    qmatmul_0 = Builder.q_linear_matmul("input", "qmatmul_0_out", 0)
    dequant = Builder.dequant("qmatmul_0_out", "dquant_out", 0)
    quant = Builder.quant("dquant_out", "quant_out", 0)
    qmatmul_1 = Builder.q_linear_matmul("quant_out", "output", 1)
    qmatmul_2 = Builder.q_linear_matmul("dquant_out", "output_2", 2)
    return [qmatmul_0, dequant, quant, qmatmul_1, qmatmul_2]


def prepare_model(output_dir, simple: bool):
    input_tvis = [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.UINT16, [1, 1, 3072])]
    output_tvis = [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.UINT16, [1, 1, 3072])]

    if simple:
        nodes = get_simple_nodes()
    else:
        nodes = get_advanced_nodes()
        output_tvis.append(onnx.helper.make_tensor_value_info("output_2", onnx.TensorProto.FLOAT16, [1, 1, 3072]))

    graph = onnx.helper.make_graph(nodes=nodes, name="QDQGraph", inputs=input_tvis, outputs=output_tvis)
    model = onnx.helper.make_model(graph, producer_name="quark_test")

    onnx_model_path = Path(output_dir) / "model.onnx"
    onnx_optimized_model_path = Path(output_dir) / "output_model.onnx"
    onnx.save_model(model, onnx_model_path)
    return onnx_model_path, onnx_optimized_model_path


def prepare_yaml(output_dir, onnx_model_path: Path, onnx_optimized_model_path: Path):
    yaml_path = Path(output_dir) / "remove_qdq.yaml"
    config = {
        "input_model_path": onnx_model_path.as_posix(),
        "passes": {"remove_qdq": {}},
        "output_model_path": onnx_optimized_model_path.as_posix(),
    }

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


def test_remove_qdq_pass_simple(tmp_path):
    """
    This tests whether a simple back-to-back QDQ pair is removed correctly.
    """
    onnx_model_path, onnx_optimized_model_path = prepare_model(tmp_path, True)
    yaml_path = prepare_yaml(tmp_path, onnx_model_path, onnx_optimized_model_path)
    cli(["onnx-adapter", yaml_path.as_posix()])

    output_model = onnx.load_model(onnx_optimized_model_path)
    assert len(output_model.graph.node) == 2
    for idx, node in enumerate(output_model.graph.node):
        assert node.op_type == "QLinearMatMul"
        assert node.name == f"QLinearMatMul_{idx}"
    assert output_model.graph.node[0].output[0] == output_model.graph.node[1].input[0]


def test_remove_qdq_pass_advanced(tmp_path):
    """
    This tests whether a a QDQ pair is removed correctly when the DequantizeLinear node has multiple successors.
    """
    onnx_model_path, onnx_optimized_model_path = prepare_model(tmp_path, False)
    yaml_path = prepare_yaml(tmp_path, onnx_model_path, onnx_optimized_model_path)
    cli(["onnx-adapter", yaml_path.as_posix()])

    output_model = onnx.load_model(onnx_optimized_model_path)
    output_model_nodes = output_model.graph.node

    assert len(output_model_nodes) == 4
    qmatmul_0 = output_model_nodes[0]
    dequant_0 = output_model_nodes[1]
    qmatmul_1 = output_model_nodes[2]
    qmatmul_2 = output_model_nodes[3]

    assert qmatmul_0.name == "QLinearMatMul_0"
    assert dequant_0.name == "DequantizeLinear_0"
    assert qmatmul_1.name == "QLinearMatMul_1"
    assert qmatmul_2.name == "QLinearMatMul_2"

    assert qmatmul_0.output[0] == dequant_0.input[0]
    assert qmatmul_0.output[0] == qmatmul_1.input[0]
    assert dequant_0.output[0] == qmatmul_2.input[0]
