# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
Test suite for the transfer_pow_mul_add_tanh_to_gelu pass.

This pass replaces the mathematical approximation of GELU activation function
with a single FastGelu operator. The GELU approximation formula is:
    GELU(x) ≈ 0.5 * x * (1 + Tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

The tests verify:
1. Pattern matching for Add-based and Sum-based graph patterns
2. Correct replacement with FastGelu operator
3. Rejection of invalid patterns (wrong constants, wrong exponent)
"""

import logging
import math
from pathlib import Path

import numpy as np
import onnx
import onnx.helper
import onnx.numpy_helper
import pytest

import ryzenai_onnx_utils
import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.partitioner


@pytest.fixture(autouse=True)
def hide_ryzenai_logs(caplog: pytest.LogCaptureFixture):
    with caplog.at_level(logging.WARNING, "ryzenai_onnx_utils"):
        yield


def verify_fastgelu_replacement(
    model: onnx.ModelProto, expected_input: str = "input", expected_output: str = "output"
) -> None:
    """
    Verify that the model has been correctly transformed to use FastGelu.

    Args:
        model: The transformed ONNX model
        expected_input: Expected input name for the FastGelu node
        expected_output: Expected output name for the FastGelu node

    Raises:
        AssertionError: If verification fails
    """
    # Check that FastGelu node was created
    fastgelu_nodes = [node for node in model.graph.node if node.op_type == "FastGelu"]
    assert len(fastgelu_nodes) == 1, f"Expected 1 FastGelu node, found {len(fastgelu_nodes)}"

    # Verify FastGelu has correct input and output
    fastgelu_node = fastgelu_nodes[0]
    assert fastgelu_node.input[0] == expected_input, f"Expected input '{expected_input}', got {fastgelu_node.input[0]}"
    assert fastgelu_node.output[0] == expected_output, (
        f"Expected output '{expected_output}', got {fastgelu_node.output[0]}"
    )
    assert fastgelu_node.domain == "com.microsoft", f"Expected domain 'com.microsoft', got {fastgelu_node.domain}"

    # Verify old nodes are removed
    old_node_types = {"Pow", "Tanh"}
    remaining_old_nodes = [node for node in model.graph.node if node.op_type in old_node_types]
    assert len(remaining_old_nodes) == 0, f"Expected no old nodes, found {len(remaining_old_nodes)}"


def verify_pattern_not_replaced(model: onnx.ModelProto, test_name: str) -> None:
    """
    Verify that the pattern was NOT replaced (original nodes still present).

    Args:
        model: The ONNX model
        test_name: Name of the test case for error messages

    Raises:
        AssertionError: If verification fails
    """
    # Original nodes should still be present
    tanh_nodes = [node for node in model.graph.node if node.op_type == "Tanh"]
    assert len(tanh_nodes) == 1, f"Tanh node should still be present for {test_name}"

    # FastGelu should NOT be present
    fastgelu_nodes = [node for node in model.graph.node if node.op_type == "FastGelu"]
    assert len(fastgelu_nodes) == 0, f"FastGelu node should not be present for {test_name}, found {len(fastgelu_nodes)}"


def verify_graph(should_replace, replaced_num, new_model, test_name):
    if should_replace:
        # Should replace the pattern once
        assert replaced_num >= 1, f"Expected at least 1 replacement, got {replaced_num}"

        verify_fastgelu_replacement(new_model)
    else:
        # can't verify replacement count since other passes may have replaced things

        verify_pattern_not_replaced(new_model, test_name)


def run_model(tmp_path, graph: onnx.GraphProto) -> tuple[onnx.ModelProto, int]:
    opsets = [
        onnx.OperatorSetIdProto(domain="ai.onnx", version=14),
        onnx.OperatorSetIdProto(domain="com.microsoft", version=1),
    ]

    model = onnx.helper.make_model(graph, opset_imports=opsets)
    model.ir_version = onnx.IR_VERSION_2023_5_5

    # Create extractor
    extractor = ryzenai_onnx_utils.matcher.get_extractor(model)

    # Apply the pass
    passes = ["normalize_binary_ops", "normalize_constants", "sd3.transfer_pow_mul_add_tanh_to_gelu"]

    params = ryzenai_onnx_utils.ReplaceParams(
        {},
        Path(),
        Path(),
        tmp_path,
    )

    model, replaced_num = ryzenai_onnx_utils.partitioner.partition(extractor, passes, params, {})
    ryzenai_onnx_utils.matcher.save_initializers_with_extractor(extractor, Path.cwd(), None)

    return extractor.model, replaced_num


def build_gelu_approximation_graph_add_pattern(
    exponent: float = 3.0,
    mul_const: float = 0.044715,
    sqrt_2_pi: float | None = None,
    add_const: float = 1.0,
    final_mul_const: float = 0.5,
) -> onnx.GraphProto:
    """
    Build a graph that represents the GELU approximation using Add operations:
    GELU(x) ≈ 0.5 * x * (1 + Tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    Pattern: Pow -> Mul -> Add -> Mul -> Tanh -> Add -> Mul -> Mul

    Args:
        exponent: Power exponent (default: 3.0 for x^3)
        mul_const: Multiplier for x^exponent (default: 0.044715)
        sqrt_2_pi: Square root of 2/π (default: computed value)
        add_const: Constant added to tanh output (default: 1.0)
        final_mul_const: Final multiplier (default: 0.5)
    """
    if sqrt_2_pi is None:
        sqrt_2_pi = math.sqrt(2.0 / math.pi)

    # Input
    input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10, 768])

    # Output
    output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 10, 768])

    # Constants
    const_exp = onnx.numpy_helper.from_array(np.array(exponent, dtype=np.float32), name="const_exp")
    const_mul = onnx.numpy_helper.from_array(np.array(mul_const, dtype=np.float32), name="const_mul")
    const_sqrt = onnx.numpy_helper.from_array(np.array(sqrt_2_pi, dtype=np.float32), name="const_sqrt")
    const_add = onnx.numpy_helper.from_array(np.array(add_const, dtype=np.float32), name="const_add")
    const_final = onnx.numpy_helper.from_array(np.array(final_mul_const, dtype=np.float32), name="const_final")

    # Nodes: x^exponent
    pow_node = onnx.helper.make_node("Pow", inputs=["input", "const_exp"], outputs=["pow_out"], name="pow_node")

    # mul_const * x^exponent
    mul2_node = onnx.helper.make_node("Mul", inputs=["pow_out", "const_mul"], outputs=["mul2_out"], name="mul2_node")

    # x + mul_const * x^exponent
    add0_node = onnx.helper.make_node("Add", inputs=["input", "mul2_out"], outputs=["add0_out"], name="add0_node")

    # sqrt(2/π) * (x + mul_const * x^exponent)
    mul3_node = onnx.helper.make_node("Mul", inputs=["add0_out", "const_sqrt"], outputs=["mul3_out"], name="mul3_node")

    # Tanh(sqrt(2/π) * (x + mul_const * x^exponent))
    tanh_node = onnx.helper.make_node("Tanh", inputs=["mul3_out"], outputs=["tanh_out"], name="tanh_node")

    # add_const + Tanh(...)
    add1_node = onnx.helper.make_node("Add", inputs=["tanh_out", "const_add"], outputs=["add1_out"], name="add1_node")

    # x * (add_const + Tanh(...))
    mul4_node = onnx.helper.make_node("Mul", inputs=["input", "add1_out"], outputs=["mul4_out"], name="mul4_node")

    # final_mul_const * x * (add_const + Tanh(...))
    mul5_node = onnx.helper.make_node("Mul", inputs=["const_final", "mul4_out"], outputs=["output"], name="mul5_node")

    graph = onnx.helper.make_graph(
        [pow_node, mul2_node, add0_node, mul3_node, tanh_node, add1_node, mul4_node, mul5_node],
        "gelu_approximation_add",
        [input_tensor],
        [output_tensor],
        [const_exp, const_mul, const_sqrt, const_add, const_final],
    )

    return graph


def build_gelu_approximation_graph_add_pattern_commutative_variant(
    variant: int = 1,
) -> onnx.GraphProto:
    """
    Build commutative variations of the GELU approximation graph.
    Tests that the pattern matcher handles different orderings of commutative operations.

    GELU(x) ≈ 0.5 * x * (1 + Tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    Args:
        variant: Which commutative variant to build (1-4)
            1: Swap order in final multiply: (1 + Tanh(...)) * x * 0.5
            2: Swap order in Add operations: const + x, const + Tanh(...)
            3: Swap order in Mul operations: const * pow, const * add_result
            4: Multiple swaps combined
    """
    # Input and output
    input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10, 768])
    output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 10, 768])

    # Constants
    const_3 = onnx.numpy_helper.from_array(np.array(3.0, dtype=np.float32), name="const_3")
    const_0_044715 = onnx.numpy_helper.from_array(np.array(0.044715, dtype=np.float32), name="const_0.044715")
    const_sqrt_2_pi = onnx.numpy_helper.from_array(
        np.array(math.sqrt(2.0 / math.pi), dtype=np.float32), name="const_sqrt_2_pi"
    )
    const_1 = onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="const_1")
    const_0_5 = onnx.numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="const_0.5")

    # x^3
    pow_node = onnx.helper.make_node("Pow", inputs=["input", "const_3"], outputs=["pow_out"], name="pow_node")

    if variant in [1, 2]:
        # 0.044715 * x^3 (original order)
        mul2_node = onnx.helper.make_node(
            "Mul", inputs=["pow_out", "const_0.044715"], outputs=["mul2_out"], name="mul2_node"
        )
    else:
        # const * x^3 (swapped order)
        mul2_node = onnx.helper.make_node(
            "Mul", inputs=["const_0.044715", "pow_out"], outputs=["mul2_out"], name="mul2_node"
        )

    if variant in [2, 4]:
        # mul2_out + x (swapped order)
        add0_node = onnx.helper.make_node("Add", inputs=["mul2_out", "input"], outputs=["add0_out"], name="add0_node")
    else:
        # x + mul2_out (original order)
        add0_node = onnx.helper.make_node("Add", inputs=["input", "mul2_out"], outputs=["add0_out"], name="add0_node")

    if variant == 4:
        # (x + ...) * sqrt(2/π) (swapped order)
        mul3_node = onnx.helper.make_node(
            "Mul", inputs=["add0_out", "const_sqrt_2_pi"], outputs=["mul3_out"], name="mul3_node"
        )
    else:
        # sqrt(2/π) * (x + ...) (original order)
        mul3_node = onnx.helper.make_node(
            "Mul", inputs=["const_sqrt_2_pi", "add0_out"], outputs=["mul3_out"], name="mul3_node"
        )

    # Tanh
    tanh_node = onnx.helper.make_node("Tanh", inputs=["mul3_out"], outputs=["tanh_out"], name="tanh_node")

    if variant in [2, 4]:
        # const + Tanh(...) (swapped order)
        add1_node = onnx.helper.make_node("Add", inputs=["const_1", "tanh_out"], outputs=["add1_out"], name="add1_node")
    else:
        # Tanh(...) + const (original order)
        add1_node = onnx.helper.make_node("Add", inputs=["tanh_out", "const_1"], outputs=["add1_out"], name="add1_node")

    if variant == 1:
        # (1 + Tanh(...)) * x (swapped order)
        mul4_node = onnx.helper.make_node("Mul", inputs=["add1_out", "input"], outputs=["mul4_out"], name="mul4_node")
        # mul4_out * 0.5
        mul5_node = onnx.helper.make_node("Mul", inputs=["mul4_out", "const_0.5"], outputs=["output"], name="mul5_node")
    elif variant == 3:
        # x * (1 + Tanh(...))
        mul4_node = onnx.helper.make_node("Mul", inputs=["input", "add1_out"], outputs=["mul4_out"], name="mul4_node")
        # mul4_out * 0.5 (swapped from original 0.5 * mul4_out)
        mul5_node = onnx.helper.make_node("Mul", inputs=["mul4_out", "const_0.5"], outputs=["output"], name="mul5_node")
    else:
        # x * (1 + Tanh(...)) (original order)
        mul4_node = onnx.helper.make_node("Mul", inputs=["input", "add1_out"], outputs=["mul4_out"], name="mul4_node")
        # 0.5 * mul4_out (original order)
        mul5_node = onnx.helper.make_node("Mul", inputs=["const_0.5", "mul4_out"], outputs=["output"], name="mul5_node")

    graph = onnx.helper.make_graph(
        [pow_node, mul2_node, add0_node, mul3_node, tanh_node, add1_node, mul4_node, mul5_node],
        f"gelu_approximation_add_variant_{variant}",
        [input_tensor],
        [output_tensor],
        [const_3, const_0_044715, const_sqrt_2_pi, const_1, const_0_5],
    )

    return graph


def build_gelu_approximation_graph_sum_pattern(
    exponent: float = 3.0,
    mul_const: float = 0.044715,
    two_over_pi: float | None = None,
    add_const: float = 1.0,
    final_mul_const: float = 0.5,
) -> onnx.GraphProto:
    """
    Build a graph that represents the GELU approximation using Sum operations:
    GELU(x) ≈ 0.5 * x * (1 + Tanh(sqrt(2/π) * (x + 0.044715 * x^3)))

    Pattern: Pow -> Mul -> Sum -> Sqrt -> Mul -> Tanh -> Sum -> Mul -> Mul

    Args:
        exponent: Power exponent (default: 3.0 for x^3)
        mul_const: Multiplier for x^exponent (default: 0.044715)
        two_over_pi: Value of 2/π (default: computed value)
        add_const: Constant added to tanh output (default: 1.0)
        final_mul_const: Final multiplier (default: 0.5)
    """
    if two_over_pi is None:
        two_over_pi = 2.0 / math.pi

    # Input
    input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10, 768])

    # Output
    output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 10, 768])

    # Constants
    const_exp = onnx.numpy_helper.from_array(np.array(exponent, dtype=np.float32), name="const_exp")
    const_mul = onnx.numpy_helper.from_array(np.array(mul_const, dtype=np.float32), name="const_mul")
    const_two_pi = onnx.numpy_helper.from_array(np.array(two_over_pi, dtype=np.float32), name="const_two_pi")
    const_add = onnx.numpy_helper.from_array(np.array(add_const, dtype=np.float32), name="const_add")
    const_final = onnx.numpy_helper.from_array(np.array(final_mul_const, dtype=np.float32), name="const_final")

    # Nodes: x^exponent
    pow_node = onnx.helper.make_node("Pow", inputs=["input", "const_exp"], outputs=["pow_out"], name="pow_node")

    # mul_const * x^exponent
    mul2_node = onnx.helper.make_node("Mul", inputs=["pow_out", "const_mul"], outputs=["mul2_out"], name="mul2_node")

    # x + mul_const * x^exponent
    sum0_node = onnx.helper.make_node("Sum", inputs=["input", "mul2_out"], outputs=["sum0_out"], name="sum0_node")

    # sqrt(2/π)
    sqrt_node = onnx.helper.make_node("Sqrt", inputs=["const_two_pi"], outputs=["sqrt_out"], name="sqrt_node")

    # sqrt(2/π) * (x + mul_const * x^exponent)
    mul3_node = onnx.helper.make_node("Mul", inputs=["sqrt_out", "sum0_out"], outputs=["mul3_out"], name="mul3_node")

    # Tanh(sqrt(2/π) * (x + mul_const * x^exponent))
    tanh_node = onnx.helper.make_node("Tanh", inputs=["mul3_out"], outputs=["tanh_out"], name="tanh_node")

    # add_const + Tanh(...)
    sum1_node = onnx.helper.make_node("Sum", inputs=["tanh_out", "const_add"], outputs=["sum1_out"], name="sum1_node")

    # final_mul_const * x
    mul4_node = onnx.helper.make_node("Mul", inputs=["const_final", "input"], outputs=["mul4_out"], name="mul4_node")

    # final_mul_const * x * (add_const + Tanh(...))
    mul5_node = onnx.helper.make_node("Mul", inputs=["mul4_out", "sum1_out"], outputs=["output"], name="mul5_node")

    graph = onnx.helper.make_graph(
        [pow_node, mul2_node, sum0_node, sqrt_node, mul3_node, tanh_node, sum1_node, mul4_node, mul5_node],
        "gelu_approximation_sum",
        [input_tensor],
        [output_tensor],
        [const_exp, const_mul, const_two_pi, const_add, const_final],
    )

    return graph


def build_gelu_approximation_graph_for_subpass_pattern(subpass_name: str) -> onnx.GraphProto:
    """Build a graph that is intended to match a specific SubPass pattern.

    This test helper exists to guarantee that every SubPass in
    `sd3.transfer_pow_mul_add_tanh_to_gelu.PATTERN` is covered by at least one
    unit test.

    Notes:
        The test pipeline runs `normalize_binary_ops` and `normalize_constants`
        before the actual pass, so minor commutative differences should still
        match, but we construct outputs that directly resemble each pattern.
    """
    input_tensor = onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 10, 768])
    output_tensor = onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 10, 768])

    const_3 = onnx.numpy_helper.from_array(np.array(3.0, dtype=np.float32), name="const_3")
    const_0_044715 = onnx.numpy_helper.from_array(np.array(0.044715, dtype=np.float32), name="const_0.044715")
    const_sqrt_2_pi = onnx.numpy_helper.from_array(
        np.array(math.sqrt(2.0 / math.pi), dtype=np.float32), name="const_sqrt_2_pi"
    )
    const_two_over_pi = onnx.numpy_helper.from_array(
        np.array(2.0 / math.pi, dtype=np.float32), name="const_two_over_pi"
    )
    const_1 = onnx.numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="const_1")
    const_0_5 = onnx.numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="const_0.5")

    pow_node = onnx.helper.make_node("Pow", inputs=["input", "const_3"], outputs=["pow_out"], name="pow_node")
    mul2_node = onnx.helper.make_node(
        "Mul", inputs=["pow_out", "const_0.044715"], outputs=["mul2_out"], name="mul2_node"
    )

    def _add_or_sum0(op_type: str, swapped: bool):
        if swapped:
            return onnx.helper.make_node(op_type, inputs=["input", "mul2_out"], outputs=["add0_out"], name="add0_node")
        return onnx.helper.make_node(op_type, inputs=["mul2_out", "input"], outputs=["add0_out"], name="add0_node")

    def _add_or_sum1(op_type: str, swapped: bool):
        if swapped:
            return onnx.helper.make_node(
                op_type, inputs=["const_1", "tanh_out"], outputs=["add1_out"], name="add1_node"
            )
        return onnx.helper.make_node(op_type, inputs=["tanh_out", "const_1"], outputs=["add1_out"], name="add1_node")

    if subpass_name.startswith("Add"):
        # Add-based patterns use sqrt(2/pi) directly.
        if subpass_name in {"Add", "Add3"}:
            add0_node = _add_or_sum0("Add", swapped=True)
        else:
            add0_node = _add_or_sum0("Add", swapped=False)

        mul3_node = onnx.helper.make_node(
            "Mul", inputs=["add0_out", "const_sqrt_2_pi"], outputs=["mul3_out"], name="mul3_node"
        )
        tanh_node = onnx.helper.make_node("Tanh", inputs=["mul3_out"], outputs=["tanh_out"], name="tanh_node")

        add1_node = _add_or_sum1("Add", swapped=False)

        if subpass_name in {"Add2", "Add3"}:
            mul4_node = onnx.helper.make_node(
                "Mul", inputs=["input", "add1_out"], outputs=["mul4_out"], name="mul4_node"
            )
        else:
            mul4_node = onnx.helper.make_node(
                "Mul", inputs=["add1_out", "input"], outputs=["mul4_out"], name="mul4_node"
            )

        mul5_node = onnx.helper.make_node("Mul", inputs=["mul4_out", "const_0.5"], outputs=["output"], name="mul5_node")

        nodes = [pow_node, mul2_node, add0_node, mul3_node, tanh_node, add1_node, mul4_node, mul5_node]
        initializers = [const_3, const_0_044715, const_sqrt_2_pi, const_1, const_0_5]
        return onnx.helper.make_graph(
            nodes, f"gelu_subpass_{subpass_name}", [input_tensor], [output_tensor], initializers
        )

    if subpass_name.startswith("Sum"):
        # Sum-based patterns compute sqrt(2/pi) via Sqrt(2/pi) in the test file's
        # existing builder, but the pass' PATTERN list for Sum variants doesn't
        # include Sqrt. We therefore build the direct-mul form that those
        # patterns describe.
        if subpass_name in {"Sum", "Sum3"}:
            sum0_node = onnx.helper.make_node(
                "Sum", inputs=["mul2_out", "input"], outputs=["sum0_out"], name="sum0_node"
            )
        else:
            sum0_node = onnx.helper.make_node(
                "Sum", inputs=["input", "mul2_out"], outputs=["sum0_out"], name="sum0_node"
            )

        mul3_node = onnx.helper.make_node(
            "Mul", inputs=["sum0_out", "const_sqrt_2_pi"], outputs=["mul3_out"], name="mul3_node"
        )
        tanh_node = onnx.helper.make_node("Tanh", inputs=["mul3_out"], outputs=["tanh_out"], name="tanh_node")

        if subpass_name in {"Sum", "Sum3"}:
            sum1_node = onnx.helper.make_node(
                "Sum", inputs=["tanh_out", "const_1"], outputs=["sum1_out"], name="sum1_node"
            )
        else:
            sum1_node = onnx.helper.make_node(
                "Sum", inputs=["const_1", "tanh_out"], outputs=["sum1_out"], name="sum1_node"
            )

        if subpass_name in {"Sum2", "Sum3"}:
            mul4_node = onnx.helper.make_node(
                "Mul", inputs=["input", "sum1_out"], outputs=["mul4_out"], name="mul4_node"
            )
            mul5_node = onnx.helper.make_node(
                "Mul", inputs=["mul4_out", "const_0.5"], outputs=["output"], name="mul5_node"
            )
        elif subpass_name == "Sum4":
            # Mul([?,?], b6) then Mul([b6, b5], ?) where b5 is sum1_out.
            mul4_node = onnx.helper.make_node(
                "Mul", inputs=["input", "const_0.5"], outputs=["mul4_out"], name="mul4_node"
            )
            mul5_node = onnx.helper.make_node(
                "Mul", inputs=["mul4_out", "sum1_out"], outputs=["output"], name="mul5_node"
            )
        else:
            # Sum / Sum1: Mul([b5,?], b6) then Mul([b6,?], ?)
            mul4_node = onnx.helper.make_node(
                "Mul", inputs=["sum1_out", "input"], outputs=["mul4_out"], name="mul4_node"
            )
            mul5_node = onnx.helper.make_node(
                "Mul", inputs=["mul4_out", "const_0.5"], outputs=["output"], name="mul5_node"
            )

        nodes = [pow_node, mul2_node, sum0_node, mul3_node, tanh_node, sum1_node, mul4_node, mul5_node]
        # Keep const_two_over_pi around to avoid name collisions with other
        # builders and also validate normalize_constants doesn't break.
        initializers = [const_3, const_0_044715, const_sqrt_2_pi, const_two_over_pi, const_1, const_0_5]
        return onnx.helper.make_graph(
            nodes, f"gelu_subpass_{subpass_name}", [input_tensor], [output_tensor], initializers
        )

    raise ValueError(f"Unknown subpass name: {subpass_name}")


@pytest.mark.parametrize(
    "exponent,mul_const,sqrt_2_pi,add_const,final_mul_const,should_replace,test_name",
    [
        # Valid pattern - should replace
        (3.0, 0.044715, None, 1.0, 0.5, True, "valid_pattern"),
        # Invalid constant - wrong mul_const
        (3.0, 0.1, None, 1.0, 0.5, False, "invalid_mul_const"),
        # Invalid exponent - should be 3, but using 2
        (2.0, 0.044715, None, 1.0, 0.5, False, "invalid_exponent"),
        # Invalid sqrt constant
        (3.0, 0.044715, 0.5, 1.0, 0.5, False, "invalid_sqrt"),
        # Invalid add constant
        (3.0, 0.044715, None, 2.0, 0.5, False, "invalid_add_const"),
        # Invalid final mul constant
        (3.0, 0.044715, None, 1.0, 0.7, False, "invalid_final_mul"),
    ],
)
def test_transfer_pow_mul_add_tanh_to_gelu_add_pattern(
    tmp_path: Path,
    exponent: float,
    mul_const: float,
    sqrt_2_pi: float | None,
    add_const: float,
    final_mul_const: float,
    should_replace: bool,
    test_name: str,
) -> None:
    """Test the Add pattern replacement with various constant configurations."""
    graph = build_gelu_approximation_graph_add_pattern(
        exponent=exponent,
        mul_const=mul_const,
        sqrt_2_pi=sqrt_2_pi,
        add_const=add_const,
        final_mul_const=final_mul_const,
    )

    new_model, replaced_num = run_model(tmp_path, graph)

    verify_graph(should_replace, replaced_num, new_model, test_name)


@pytest.mark.parametrize(
    "exponent,mul_const,two_over_pi,add_const,final_mul_const,should_replace,test_name",
    [
        # Valid pattern - should replace
        (3.0, 0.044715, None, 1.0, 0.5, True, "valid_pattern"),
        # Invalid constant - wrong mul_const
        (3.0, 0.1, None, 1.0, 0.5, False, "invalid_mul_const"),
        # Invalid exponent
        (2.0, 0.044715, None, 1.0, 0.5, False, "invalid_exponent"),
    ],
)
def test_transfer_pow_mul_add_tanh_to_gelu_sum_pattern(
    tmp_path: Path,
    exponent: float,
    mul_const: float,
    two_over_pi: float | None,
    add_const: float,
    final_mul_const: float,
    should_replace: bool,
    test_name: str,
) -> None:
    """Test the Sum pattern replacement with various constant configurations."""
    graph = build_gelu_approximation_graph_sum_pattern(
        exponent=exponent,
        mul_const=mul_const,
        two_over_pi=two_over_pi,
        add_const=add_const,
        final_mul_const=final_mul_const,
    )

    new_model, replaced_num = run_model(tmp_path, graph)

    onnx.save_model(new_model, tmp_path / f"{test_name}_model.onnx")

    verify_graph(should_replace, replaced_num, new_model, test_name)


@pytest.mark.parametrize(
    "subpass_name",
    [
        pytest.param("Add", id="pattern_Add"),
        pytest.param("Add1", id="pattern_Add1"),
        pytest.param("Add2", id="pattern_Add2"),
        pytest.param("Add3", id="pattern_Add3"),
        pytest.param("Sum", id="pattern_Sum"),
        pytest.param("Sum1", id="pattern_Sum1"),
        pytest.param("Sum2", id="pattern_Sum2"),
        pytest.param("Sum3", id="pattern_Sum3"),
        pytest.param("Sum4", id="pattern_Sum4"),
    ],
)
def test_transfer_pow_mul_add_tanh_to_gelu_all_subpass_patterns(tmp_path: Path, subpass_name: str) -> None:
    """Ensure every SubPass pattern in the pass file is covered by a unit test."""
    graph = build_gelu_approximation_graph_for_subpass_pattern(subpass_name)
    new_model, replaced_num = run_model(tmp_path, graph)

    assert replaced_num >= 1, f"Expected replacement for subpass {subpass_name}, got {replaced_num}"
    verify_fastgelu_replacement(new_model, "input", "output")
