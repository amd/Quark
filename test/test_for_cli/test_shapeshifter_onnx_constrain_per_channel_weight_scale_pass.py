#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

"""Tests for the onnx_constrain_per_channel_weight_scale adapter pass.

The tests build small hand-crafted QDQ graphs (Conv / ConvTranspose / Gemm whose
weight comes from a DequantizeLinear with a per-channel scale initializer) and run
the pass directly as well as through the Shapeshifter engine and CLI. They cover:

- config defaults and user overrides,
- the master switch and the config validation errors,
- the fixed and the adaptive scale floor,
- every "leave the tensor alone" early exit (per-tensor scale, non-weight op,
  weight not produced by a DequantizeLinear, missing/non-initializer scale,
  ratio within budget, non-positive minimum scale, floor below the minimum scale),
- the quantized-bias adjustment and all of its early exits.
"""

import numpy as np
import onnx
import pytest
import yaml
from onnx import TensorProto, helper, numpy_helper

from quark.shapeshifter import Engine
from quark.shapeshifter.pass_base import REGISTRY
from quark.shapeshifter.passes.onnx_constrain_per_channel_weight_scale import (
    ONNXConstrainPerChannelWeightScalePass,
)

PASS_NAME = "onnx_constrain_per_channel_weight_scale"

# Activation scale used to build a consistent quantized bias
# (bias_scale = input_scale * weight_scale).
INPUT_SCALE = 0.5


def _input_shape(op_type: str) -> list[int]:
    """Return the graph input shape used for a given weight op type.

    Args:
        op_type: One of ``Conv``, ``ConvTranspose`` or ``Gemm``.

    Returns:
        The shape of the float graph input feeding the op.
    """
    return [1, 1] if op_type == "Gemm" else [1, 1, 1, 1]


def _output_shape(op_type: str, channels: int) -> list[int]:
    """Return the graph output shape for a given weight op type.

    Args:
        op_type: One of ``Conv``, ``ConvTranspose`` or ``Gemm``.
        channels: Number of output channels of the weight tensor.

    Returns:
        The shape of the float graph output.
    """
    return [1, channels] if op_type == "Gemm" else [1, channels, 1, 1]


def _weight_shape(op_type: str, channels: int) -> list[int]:
    """Return the quantized weight shape for a given weight op type.

    Args:
        op_type: One of ``Conv``, ``ConvTranspose`` or ``Gemm``.
        channels: Number of output channels of the weight tensor.

    Returns:
        The shape of the quantized weight initializer.
    """
    if op_type == "Conv":
        return [channels, 1, 1, 1]
    if op_type == "ConvTranspose":
        return [1, channels, 1, 1]
    return [1, channels]  # Gemm: B is [K, N].


def build_qdq_model(
    weight_scales,
    *,
    op_type: str = "Conv",
    channels: int | None = None,
    with_bias: bool = True,
    bias_scales=None,
    quantized_bias=None,
    bias_zero_point=None,
    with_weight_q: bool = False,
) -> onnx.ModelProto:
    """Build a minimal QDQ model whose weight carries a per-channel scale.

    The produced graph is ``[QuantizeLinear ->] DequantizeLinear -> <op_type>``, with an
    optional quantized bias branch ``DequantizeLinear -> <op_type>``, followed by a Relu
    (which also exercises the "not a weight op" branch of the pass).

    Args:
        weight_scales: Scalar or 1-D array used as the weight scale initializer.
        op_type: Weight op to build; ``Conv``, ``ConvTranspose`` or ``Gemm``.
        channels: Number of output channels. Defaults to the size of ``weight_scales``.
        with_bias: When True, add a quantized (int32) bias input to the op.
        bias_scales: Bias scale initializer value. Defaults to
            ``INPUT_SCALE * weight_scales``.
        quantized_bias: int32 bias values. Defaults to ``[10000, 200, 300, ...]``.
        bias_zero_point: Optional int32 bias zero point initializer.
        with_weight_q: When True, prepend a QuantizeLinear that shares the weight
            scale initializer with the DequantizeLinear.

    Returns:
        The constructed ONNX model.
    """
    weight_scales = np.asarray(weight_scales, dtype=np.float32)
    if channels is None:
        channels = int(weight_scales.size)

    w_shape = _weight_shape(op_type, channels)
    axis = 1 if op_type in ("ConvTranspose", "Gemm") else 0

    initializers = [
        numpy_helper.from_array(np.ones(w_shape, dtype=np.int8), name="w_q"),
        numpy_helper.from_array(weight_scales, name="w_scale"),
        numpy_helper.from_array(np.zeros(channels, dtype=np.int8), name="w_zp"),
    ]
    nodes = []

    if with_weight_q:
        initializers.append(numpy_helper.from_array(np.ones(w_shape, dtype=np.float32), name="w_float"))
        nodes.append(
            helper.make_node("QuantizeLinear", ["w_float", "w_scale", "w_zp"], ["w_q_out"], name="weight_q", axis=axis)
        )
        dq_weight_input = "w_q_out"
    else:
        dq_weight_input = "w_q"

    nodes.append(
        helper.make_node(
            "DequantizeLinear", [dq_weight_input, "w_scale", "w_zp"], ["w_dq"], name="weight_dq", axis=axis
        )
    )

    op_inputs = ["input", "w_dq"]
    if with_bias:
        if quantized_bias is None:
            quantized_bias = np.array([10000, 200, 300, 400][:channels], dtype=np.int32)
        quantized_bias = np.asarray(quantized_bias, dtype=np.int32)
        if bias_scales is None:
            bias_scales = INPUT_SCALE * weight_scales
        bias_scales = np.asarray(bias_scales, dtype=np.float32)

        initializers.append(numpy_helper.from_array(quantized_bias, name="b_q"))
        initializers.append(numpy_helper.from_array(bias_scales, name="b_scale"))
        bias_dq_inputs = ["b_q", "b_scale"]
        if bias_zero_point is not None:
            initializers.append(numpy_helper.from_array(np.asarray(bias_zero_point, dtype=np.int32), name="b_zp"))
            bias_dq_inputs.append("b_zp")
        nodes.append(helper.make_node("DequantizeLinear", bias_dq_inputs, ["b_dq"], name="bias_dq", axis=0))
        op_inputs.append("b_dq")

    nodes.append(helper.make_node(op_type, op_inputs, ["op_out"], name="weight_op"))
    nodes.append(helper.make_node("Relu", ["op_out"], ["output"], name="relu"))

    graph = helper.make_graph(
        nodes,
        "constrain_per_channel_weight_scale_test",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, _input_shape(op_type))],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, _output_shape(op_type, channels))],
        initializer=initializers,
    )
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def get_init(model: onnx.ModelProto, name: str) -> np.ndarray:
    """Return the value of a model initializer as a numpy array.

    Args:
        model: The model to read from.
        name: Name of the initializer.

    Returns:
        The initializer value.
    """
    init = next(i for i in model.graph.initializer if i.name == name)
    return numpy_helper.to_array(init)


def get_node(model: onnx.ModelProto, name: str):
    """Return a node of the model by name.

    Args:
        model: The model to read from.
        name: Name of the node.

    Returns:
        The matching NodeProto.
    """
    return next(n for n in model.graph.node if n.name == name)


def run_pass(model: onnx.ModelProto, **config) -> onnx.ModelProto:
    """Run the pass on a model with the master switch enabled by default.

    Args:
        model: The model to transform.
        **config: Pass configuration overrides.

    Returns:
        The transformed model.
    """
    full_config = {"constrain_per_channel_weight_scale": True}
    full_config.update(config)
    return ONNXConstrainPerChannelWeightScalePass(full_config).run(model)


# ---------------------------------------------------------------------------
# Registration and configuration
# ---------------------------------------------------------------------------


def test_pass_is_registered():
    """The pass is discoverable in the Shapeshifter registry under its file name."""
    assert REGISTRY[PASS_NAME] is ONNXConstrainPerChannelWeightScalePass


def test_default_config():
    """The default config exposes the documented parameters and default values."""
    config = ONNXConstrainPerChannelWeightScalePass({})._default_config()

    assert config["constrain_per_channel_weight_scale"].type_ is bool
    assert config["constrain_per_channel_weight_scale"].default_value is True
    assert config["constrain_per_channel_weight_scale"].required is True
    assert config["min_w_scale"].default_value == 1e-7
    assert config["min_w_scale"].required is False
    assert config["adaptive_min_w_scale"].default_value is False
    assert config["maxmin_scale_ratio"].default_value == 1e6
    assert config["adjust_bias"].default_value is True


def test_default_config_merges_user_config():
    """User-supplied config entries override the defaults in the merged dict."""
    config = ONNXConstrainPerChannelWeightScalePass({"min_w_scale": 1e-3, "extra": 5})._default_config()

    assert config["min_w_scale"] == 1e-3
    assert config["extra"] == 5
    assert config["maxmin_scale_ratio"].default_value == 1e6


def test_pass_disabled_leaves_model_untouched():
    """A missing or falsy master switch turns the pass into a no-op."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)
    model = build_qdq_model(scales)

    model = ONNXConstrainPerChannelWeightScalePass({"constrain_per_channel_weight_scale": False}).run(model)

    np.testing.assert_array_equal(get_init(model, "w_scale"), scales)


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"min_w_scale": 0.0}, "min_w_scale must be positive"),
        ({"min_w_scale": -1e-7}, "min_w_scale must be positive"),
        ({"maxmin_scale_ratio": 0.5}, "maxmin_scale_ratio must be >= 1.0"),
    ],
)
def test_invalid_config_raises(config, message):
    """Non-positive floors and ratios below 1.0 are rejected."""
    model = build_qdq_model([1e-9, 1e-3, 1.0])

    with pytest.raises(ValueError, match=message):
        run_pass(model, **config)


# ---------------------------------------------------------------------------
# Scale constraining
# ---------------------------------------------------------------------------


def test_fixed_floor_clamps_small_scales():
    """With a fixed floor, only the smallest scales are lifted; the max is kept."""
    model = build_qdq_model([1e-9, 1e-3, 1.0])

    model = run_pass(model, min_w_scale=1e-6, maxmin_scale_ratio=1e6, adjust_bias=False)

    np.testing.assert_allclose(get_init(model, "w_scale"), [1e-6, 1e-3, 1.0], rtol=1e-6)


def test_adaptive_floor_enforces_ratio():
    """The adaptive floor is raised to ``max_scale / maxmin_scale_ratio``."""
    model = build_qdq_model([1e-9, 1e-3, 1.0])

    model = run_pass(model, min_w_scale=1e-7, adaptive_min_w_scale=True, maxmin_scale_ratio=100.0, adjust_bias=False)

    new_scales = get_init(model, "w_scale")
    np.testing.assert_allclose(new_scales, [1e-2, 1e-2, 1.0], rtol=1e-6)
    # float32 rounding of the stored floor allows a negligible overshoot.
    assert float(new_scales.max()) / float(new_scales.min()) == pytest.approx(100.0, rel=1e-6)


def test_adaptive_floor_keeps_min_w_scale_when_larger():
    """The adaptive floor never drops below the configured ``min_w_scale``."""
    model = build_qdq_model([1e-9, 1e-3, 1.0])

    model = run_pass(model, min_w_scale=0.5, adaptive_min_w_scale=True, maxmin_scale_ratio=1e6, adjust_bias=False)

    np.testing.assert_allclose(get_init(model, "w_scale"), [0.5, 0.5, 1.0], rtol=1e-6)


def test_shared_quantize_linear_scale_is_updated():
    """A QuantizeLinear sharing the weight scale initializer sees the new values."""
    model = build_qdq_model([1e-9, 1e-3, 1.0], with_weight_q=True)

    model = run_pass(model, min_w_scale=1e-6, adjust_bias=False)

    assert get_node(model, "weight_q").input[1] == get_node(model, "weight_dq").input[1] == "w_scale"
    np.testing.assert_allclose(get_init(model, "w_scale"), [1e-6, 1e-3, 1.0], rtol=1e-6)


@pytest.mark.parametrize("op_type", ["Conv", "ConvTranspose", "Gemm"])
def test_all_weight_op_types_are_handled(op_type):
    """Conv, ConvTranspose and Gemm weights are all constrained."""
    model = build_qdq_model([1e-9, 1e-3, 1.0], op_type=op_type)

    model = run_pass(model, min_w_scale=1e-6, adjust_bias=False)

    np.testing.assert_allclose(get_init(model, "w_scale"), [1e-6, 1e-3, 1.0], rtol=1e-6)


def test_ratio_within_budget_is_untouched():
    """A weight tensor whose max/min ratio is within budget is not modified."""
    scales = np.array([0.5, 0.75, 1.0], dtype=np.float32)
    model = build_qdq_model(scales)

    model = run_pass(model, min_w_scale=0.9, maxmin_scale_ratio=1e6)

    np.testing.assert_array_equal(get_init(model, "w_scale"), scales)


def test_non_positive_min_scale_is_untouched():
    """A tensor containing a zero (or negative) scale is skipped."""
    scales = np.array([0.0, 1e-3, 1.0], dtype=np.float32)
    model = build_qdq_model(scales)

    model = run_pass(model, min_w_scale=1e-6)

    np.testing.assert_array_equal(get_init(model, "w_scale"), scales)


def test_floor_below_min_scale_is_a_no_op():
    """A violating tensor is left as-is when the floor is below its smallest scale."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)
    model = build_qdq_model(scales)

    model = run_pass(model, min_w_scale=1e-12, maxmin_scale_ratio=1e6)

    np.testing.assert_array_equal(get_init(model, "w_scale"), scales)
    np.testing.assert_array_equal(get_init(model, "b_q"), [10000, 200, 300])


@pytest.mark.parametrize("scales", [np.float32(1e-9), np.array([1e-9], dtype=np.float32)])
def test_per_tensor_scale_is_untouched(scales):
    """Scalar and single-element (per-tensor) scales are ignored by the pass."""
    model = build_qdq_model(scales, channels=3, with_bias=False)

    model = run_pass(model, min_w_scale=1e-6)

    np.testing.assert_array_equal(get_init(model, "w_scale"), scales)


def test_non_weight_op_is_untouched():
    """Per-channel scales feeding a non Conv/ConvTranspose/Gemm op are ignored."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)
    model = build_qdq_model(scales, with_bias=False)
    get_node(model, "weight_op").op_type = "Add"

    model = run_pass(model, min_w_scale=1e-6)

    np.testing.assert_array_equal(get_init(model, "w_scale"), scales)


def test_op_without_weight_input_is_skipped():
    """An op with fewer than two inputs, or an empty weight input, is skipped."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)

    model = build_qdq_model(scales, with_bias=False)
    del get_node(model, "weight_op").input[1:]
    model = run_pass(model, min_w_scale=1e-6)
    np.testing.assert_array_equal(get_init(model, "w_scale"), scales)

    model = build_qdq_model(scales, with_bias=False)
    get_node(model, "weight_op").input[1] = ""
    model = run_pass(model, min_w_scale=1e-6)
    np.testing.assert_array_equal(get_init(model, "w_scale"), scales)


def test_weight_not_produced_by_dequantize_is_skipped():
    """A weight that is a plain initializer or comes from another op is skipped."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)

    # Float weight initializer, no producing node at all.
    model = build_qdq_model(scales, with_bias=False)
    get_node(model, "weight_op").input[1] = "w_q"
    model = run_pass(model, min_w_scale=1e-6)
    np.testing.assert_array_equal(get_init(model, "w_scale"), scales)

    # Weight produced by a non-DequantizeLinear node.
    model = build_qdq_model(scales, with_bias=False)
    get_node(model, "weight_dq").op_type = "Identity"
    model = run_pass(model, min_w_scale=1e-6)
    np.testing.assert_array_equal(get_init(model, "w_scale"), scales)


def test_dequantize_without_scale_input_is_skipped():
    """A DequantizeLinear that carries no scale input is skipped."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)
    model = build_qdq_model(scales, with_bias=False)
    del get_node(model, "weight_dq").input[1:]

    model = run_pass(model, min_w_scale=1e-6)

    np.testing.assert_array_equal(get_init(model, "w_scale"), scales)


def test_non_initializer_scale_is_skipped():
    """A scale that is not an initializer (e.g. a graph input) is skipped."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)
    model = build_qdq_model(scales, with_bias=False)
    get_node(model, "weight_dq").input[1] = "dynamic_scale"

    model = run_pass(model, min_w_scale=1e-6)

    np.testing.assert_array_equal(get_init(model, "w_scale"), scales)


# ---------------------------------------------------------------------------
# Quantized bias adjustment
# ---------------------------------------------------------------------------


def test_bias_is_adjusted_for_modified_channels():
    """The bias scale tracks the new weight scale and the dequantized bias is kept."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)
    old_bias_scale = (INPUT_SCALE * scales).astype(np.float32)
    old_q_bias = np.array([10000, 200, 300], dtype=np.int32)
    model = build_qdq_model(scales)

    model = run_pass(model, min_w_scale=1e-6, maxmin_scale_ratio=1e6, adjust_bias=True)

    new_w_scale = get_init(model, "w_scale")
    new_bias_scale = get_init(model, "b_scale")
    new_q_bias = get_init(model, "b_q")

    # bias_scale = input_scale * weight_scale is restored on the modified channel.
    np.testing.assert_allclose(new_bias_scale, INPUT_SCALE * new_w_scale, rtol=1e-5)
    # Only the first channel's weight scale changed, so only its int bias moves.
    np.testing.assert_array_equal(new_q_bias, [10, 200, 300])
    assert new_q_bias.dtype == np.int32
    # The dequantized bias value is preserved.
    np.testing.assert_allclose(
        new_q_bias.astype(np.float64) * new_bias_scale.astype(np.float64),
        old_q_bias.astype(np.float64) * old_bias_scale.astype(np.float64),
        rtol=1e-5,
    )


def test_bias_zero_point_is_honoured():
    """A non-zero bias zero point is used for both dequantization and re-quantization."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)
    model = build_qdq_model(scales, quantized_bias=[10001, 201, 301], bias_zero_point=[1, 1, 1])

    model = run_pass(model, min_w_scale=1e-6, adjust_bias=True)

    np.testing.assert_array_equal(get_init(model, "b_q"), [11, 201, 301])


def test_bias_zero_point_name_without_initializer_defaults_to_zero():
    """A zero point input that is not an initializer falls back to a zero offset."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)
    model = build_qdq_model(scales)
    get_node(model, "bias_dq").input.append("missing_zp")

    model = run_pass(model, min_w_scale=1e-6, adjust_bias=True)

    np.testing.assert_array_equal(get_init(model, "b_q"), [10, 200, 300])


def test_bias_not_adjusted_when_disabled():
    """With ``adjust_bias`` disabled, the weight scale changes but the bias does not."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)
    model = build_qdq_model(scales)

    model = run_pass(model, min_w_scale=1e-6, adjust_bias=False)

    np.testing.assert_allclose(get_init(model, "w_scale"), [1e-6, 1e-3, 1.0], rtol=1e-6)
    np.testing.assert_allclose(get_init(model, "b_scale"), INPUT_SCALE * scales, rtol=1e-6)
    np.testing.assert_array_equal(get_init(model, "b_q"), [10000, 200, 300])


def test_missing_bias_is_skipped():
    """Ops without a bias input, or with an empty bias input, are handled safely."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)

    model = run_pass(build_qdq_model(scales, with_bias=False), min_w_scale=1e-6, adjust_bias=True)
    np.testing.assert_allclose(get_init(model, "w_scale"), [1e-6, 1e-3, 1.0], rtol=1e-6)

    model = build_qdq_model(scales)
    get_node(model, "weight_op").input[2] = ""
    model = run_pass(model, min_w_scale=1e-6, adjust_bias=True)
    np.testing.assert_allclose(get_init(model, "w_scale"), [1e-6, 1e-3, 1.0], rtol=1e-6)
    np.testing.assert_array_equal(get_init(model, "b_q"), [10000, 200, 300])


def test_bias_not_produced_by_dequantize_is_skipped():
    """A bias that is not produced by a DequantizeLinear is left untouched."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)

    # Bias is a plain initializer (no producing node).
    model = build_qdq_model(scales)
    get_node(model, "weight_op").input[2] = "b_q"
    model = run_pass(model, min_w_scale=1e-6, adjust_bias=True)
    np.testing.assert_array_equal(get_init(model, "b_q"), [10000, 200, 300])

    # Bias produced by a non-DequantizeLinear node.
    model = build_qdq_model(scales)
    get_node(model, "bias_dq").op_type = "Identity"
    model = run_pass(model, min_w_scale=1e-6, adjust_bias=True)
    np.testing.assert_array_equal(get_init(model, "b_q"), [10000, 200, 300])

    # Bias DequantizeLinear without a scale input.
    model = build_qdq_model(scales)
    del get_node(model, "bias_dq").input[1:]
    model = run_pass(model, min_w_scale=1e-6, adjust_bias=True)
    np.testing.assert_array_equal(get_init(model, "b_q"), [10000, 200, 300])


def test_bias_with_non_initializer_inputs_is_skipped():
    """A bias whose int values or scale are not initializers is left untouched."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)

    model = build_qdq_model(scales)
    get_node(model, "bias_dq").input[0] = "dynamic_bias"
    model = run_pass(model, min_w_scale=1e-6, adjust_bias=True)
    np.testing.assert_allclose(get_init(model, "b_scale"), INPUT_SCALE * scales, rtol=1e-6)

    model = build_qdq_model(scales)
    get_node(model, "bias_dq").input[1] = "dynamic_bias_scale"
    model = run_pass(model, min_w_scale=1e-6, adjust_bias=True)
    np.testing.assert_array_equal(get_init(model, "b_q"), [10000, 200, 300])


@pytest.mark.parametrize(
    ("bias_scales", "quantized_bias"),
    [
        (np.float32(0.5), np.array([10000, 200, 300], dtype=np.int32)),  # per-tensor bias scale
        (np.array([0.5, 0.5], dtype=np.float32), np.array([10000, 200], dtype=np.int32)),  # misaligned
    ],
)
def test_bias_not_aligned_with_weight_channels_is_skipped(bias_scales, quantized_bias):
    """A per-tensor or misaligned bias is reported and left untouched."""
    scales = np.array([1e-9, 1e-3, 1.0], dtype=np.float32)
    model = build_qdq_model(scales, bias_scales=bias_scales, quantized_bias=quantized_bias)

    model = run_pass(model, min_w_scale=1e-6, adjust_bias=True)

    np.testing.assert_allclose(get_init(model, "w_scale"), [1e-6, 1e-3, 1.0], rtol=1e-6)
    np.testing.assert_array_equal(get_init(model, "b_scale"), bias_scales)
    np.testing.assert_array_equal(get_init(model, "b_q"), quantized_bias)


def test_bias_requantization_is_clipped_to_int_range():
    """Re-quantized bias values are clipped to the int bias dtype range."""
    scales = np.array([1.0, 1.0, 1e9], dtype=np.float32)
    # A tiny bias scale on the clamped channels forces a huge re-quantized value.
    model = build_qdq_model(
        scales,
        bias_scales=np.array([1e-30, 1e-30, 1.0], dtype=np.float32),
        quantized_bias=np.array([2000000000, 1, 1], dtype=np.int32),
    )

    model = run_pass(model, min_w_scale=1e-38, adaptive_min_w_scale=True, maxmin_scale_ratio=10.0, adjust_bias=True)

    new_q_bias = get_init(model, "b_q")
    assert new_q_bias.dtype == np.int32
    assert new_q_bias.max() <= np.iinfo(np.int32).max


# ---------------------------------------------------------------------------
# Engine / CLI integration
# ---------------------------------------------------------------------------


def write_run_config(tmp_path) -> tuple[str, str]:
    """Write a QDQ model and a Shapeshifter YAML config that runs this pass on it.

    Args:
        tmp_path: Directory to write the model and the config into.

    Returns:
        A tuple of (yaml config path, output model path).
    """
    input_model_path = (tmp_path / "constrain_per_channel_weight_scale.onnx").as_posix()
    output_model_path = (tmp_path / "constrain_per_channel_weight_scale_out.onnx").as_posix()
    yaml_path = (tmp_path / "constrain_per_channel_weight_scale.yaml").as_posix()

    onnx.save(build_qdq_model([1e-9, 1e-3, 1.0]), input_model_path)

    config = {
        "input_model_path": input_model_path,
        "passes": {
            PASS_NAME: {
                "constrain_per_channel_weight_scale": True,
                "min_w_scale": 1e-6,
                "adaptive_min_w_scale": False,
                "maxmin_scale_ratio": 1e6,
                "adjust_bias": True,
            }
        },
        "output_model_path": output_model_path,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False, default_flow_style=False)

    return yaml_path, output_model_path


def check_output_model(output_model_path: str) -> None:
    """Assert that the saved model has constrained scales and an adjusted bias.

    Args:
        output_model_path: Path of the model produced by the pass.
    """
    model = onnx.load(output_model_path)
    onnx.checker.check_model(model)
    np.testing.assert_allclose(get_init(model, "w_scale"), [1e-6, 1e-3, 1.0], rtol=1e-6)
    np.testing.assert_array_equal(get_init(model, "b_q"), [10, 200, 300])


def test_pass_runs_through_engine(tmp_path):
    """The pass is reachable from the Shapeshifter engine through a YAML config."""
    yaml_path, output_model_path = write_run_config(tmp_path)

    with open(yaml_path, encoding="utf-8") as f:
        engine = Engine(config=yaml.safe_load(f))
    engine.initialize()
    engine.run()

    check_output_model(output_model_path)


def test_pass_runs_through_cli(tmp_path):
    """The pass is reachable from the `quark shapeshifter` CLI subcommand."""
    pytest.importorskip("transformers", reason="quark.experimental.cli.main requires the CLI extra dependencies")
    from quark.experimental.cli.main import main as cli

    yaml_path, output_model_path = write_run_config(tmp_path)

    cli(["shapeshifter", yaml_path])

    check_output_model(output_model_path)


if __name__ == "__main__":
    pytest.main([__file__])
