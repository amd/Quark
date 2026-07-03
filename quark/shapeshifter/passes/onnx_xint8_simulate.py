#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""XINT8 DPU simulation pass: convert selected ops to DPU-compatible forms.

Transforms LeakyRelu, Sigmoid, HardSigmoid, AveragePool, ReduceMean, Softmax,
InstanceNormalization, and Clip into DPU-simulated equivalents (e.g. Sigmoid->HardSigmoid,
Softmax->polynomial approximation, Clip bounds to [-128, 127]).
"""

import math
from typing import Any

import numpy as np
import onnx
from onnx import ModelProto
from onnx import onnx_pb as onnx_proto
from onnxruntime.quantization.onnx_model import ONNXModel

from quark.common.utils.log import ScreenLogger
from quark.onnx.quantization.quant_utils import (
    COP_DOMAIN,
    COP_IN_OP_NAME,
    HARD_SIGMOID_SCALE,
    check_hard_sigmoid_condition,
    dpu_leaky_relu_alpha,
    get_clip_min_max,
    get_opset_version,
)
from quark.shapeshifter.pass_base import ONNXPass, register_pass
from quark.shapeshifter.pass_config import PassConfigParam

logger = ScreenLogger(__name__)


@register_pass
class ONNXXInt8SimulatePass(ONNXPass):
    """Adapter pass: convert LeakyRelu, Sigmoid, HardSigmoid, Pool, ReduceMean, Softmax, InstanceNorm, Clip to DPU-simulated forms."""

    def _default_config(self) -> dict[str, PassConfigParam]:
        """Default config: xint8_simulate (bool) to enable/disable the pass."""
        config = {
            "xint8_simulate": PassConfigParam(
                type_=bool,
                default_value=True,
                required=True,
                description="Enable XINT8 DPU simulation (LeakyRelu, Sigmoid, HardSigmoid, AvgPool, ReduceMean, Softmax, InstanceNorm, Clip conversions).",
            )
        }
        config.update(self.config)
        return config

    def _onnx_xint8_simulate(
        self,
        model: ModelProto,
        nodes_to_exclude: list[str],
    ) -> ModelProto:
        """Run DPU simulation conversions and return topologically sorted model."""
        simulate_dpu = SimulateDPU(model, nodes_to_exclude)
        simulate_dpu._convert_leaky_relu_to_dpu_version()
        simulate_dpu._convert_sigmoid_to_hard_sigmoid()
        simulate_dpu._convert_hard_sigmoid_to_dpu_version()
        simulate_dpu._convert_avg_pool_to_dpu_version()
        simulate_dpu._convert_reduce_mean_to_dpu_version()
        simulate_dpu._convert_softmax_to_dpu_version()
        simulate_dpu._convert_instance_norm_to_dpu_version()
        simulate_dpu._convert_clip_to_dpu_version()

        onnx_model = ONNXModel(simulate_dpu.model)
        onnx_model.topological_sort()
        return onnx_model.model

    def _run_for_config(self, model: ModelProto, config: dict[str, PassConfigParam]) -> ModelProto:
        """Execute pass when xint8_simulate is enabled; otherwise return model unchanged."""
        if "xint8_simulate" in config and config["xint8_simulate"]:
            model = self._onnx_xint8_simulate(
                model,
                nodes_to_exclude=[],
            )
        else:
            logger.warning("xint8_simulate is missing or falsy; enable xint8_simulate to run this pass.")
        return model


class SimulateDPUSoftmax:
    """Replaces Softmax with a DPU subgraph: exp (polynomial), ReduceSum, Div; uses BF16."""

    def __init__(self, opset_version: float = 21) -> None:
        """Ctor; opset_version used for ReduceSum (axes attribute vs second input)."""
        self._opset_version = opset_version

    def _simulate(self, node: onnx.NodeProto) -> list[onnx.NodeProto]:
        """Return replacement nodes for this Softmax (opset >= 7); empty list if opset < 7."""
        new_nodes: list[onnx.NodeProto] = []

        if self._opset_version < 7:
            return new_nodes

        reduce_axes = -1
        for attr in node.attribute:
            if attr.name == "axis":
                reduce_axes = attr.i

        softmax_name = node.name

        data_type = onnx_proto.TensorProto.BFLOAT16

        def _exp_poly_approximation(input_node_name: str) -> str:
            nonlocal new_nodes

            exp_poly_namescope = softmax_name + "/exp_poly"

            # Round: x * rcp_ln2, then Floor
            round_namescope = exp_poly_namescope + "/round"

            cast_0_node_name = round_namescope + "/cast"
            cast_0_node = onnx.helper.make_node(
                "Cast", [input_node_name], [cast_0_node_name + "_output"], name=cast_0_node_name, to=data_type
            )
            new_nodes.append(cast_0_node)

            rcp_ln2_node_name = round_namescope + "/rcp_ln2"
            rcp_ln2_node = onnx.helper.make_node(
                "Constant",
                [],
                [rcp_ln2_node_name + "_output"],
                name=rcp_ln2_node_name,
                value=onnx.helper.make_tensor(rcp_ln2_node_name + "_value", data_type, [], [1.4426950408889634]),
            )
            new_nodes.append(rcp_ln2_node)

            mul_0_node_name = round_namescope + "/mul"
            mul_0_node = onnx.helper.make_node(
                "Mul",
                [cast_0_node_name + "_output", rcp_ln2_node_name + "_output"],
                [mul_0_node_name + "_output"],
                name=mul_0_node_name,
            )
            new_nodes.append(mul_0_node)

            round_0_node_name = round_namescope + "/round"
            round_0_node = onnx.helper.make_node(
                "Floor", [mul_0_node_name + "_output"], [round_0_node_name + "_output"], name=round_0_node_name
            )
            new_nodes.append(round_0_node)

            # Modulo: x - floor(x*rcp_ln2)*ln2
            modulo_namescope = exp_poly_namescope + "/modulo"

            ln2_node_name = modulo_namescope + "/ln2"
            ln2_node = onnx.helper.make_node(
                "Constant",
                [],
                [ln2_node_name + "_output"],
                name=ln2_node_name,
                value=onnx.helper.make_tensor(ln2_node_name + "_value", data_type, [], [0.6931471805599453]),
            )
            new_nodes.append(ln2_node)

            mul_1_node_name = modulo_namescope + "/mul"
            mul_1_node = onnx.helper.make_node(
                "Mul",
                [round_0_node_name + "_output", ln2_node_name + "_output"],
                [mul_1_node_name + "_output"],
                name=mul_1_node_name,
            )
            new_nodes.append(mul_1_node)

            sub_1_node_name = modulo_namescope + "/sub"
            sub_1_node = onnx.helper.make_node(
                "Sub",
                [cast_0_node_name + "_output", mul_1_node_name + "_output"],
                [sub_1_node_name + "_output"],
                name=sub_1_node_name,
            )
            new_nodes.append(sub_1_node)

            # Polynomial approximation for exp(x) in [-1, 0]
            poly_approx_namescope = exp_poly_namescope + "/poly_approx"

            cast_1_node_name = poly_approx_namescope + "/cast_1"
            cast_1_node = onnx.helper.make_node(
                "Cast",
                [sub_1_node_name + "_output"],
                [cast_1_node_name + "_output"],
                name=cast_1_node_name,
                to=onnx_proto.TensorProto.FLOAT,
            )
            new_nodes.append(cast_1_node)

            alpha_3_node_name = poly_approx_namescope + "/alpha_3"
            alpha_3_node = onnx.helper.make_node(
                "Constant",
                [],
                [alpha_3_node_name + "_output"],
                name=alpha_3_node_name,
                value=onnx.helper.make_tensor(alpha_3_node_name + "_value", data_type, [], [0.21875]),
            )
            new_nodes.append(alpha_3_node)

            cast_2_node_name = poly_approx_namescope + "/cast_2"
            cast_2_node = onnx.helper.make_node(
                "Cast",
                [alpha_3_node_name + "_output"],
                [cast_2_node_name + "_output"],
                name=cast_2_node_name,
                to=onnx_proto.TensorProto.FLOAT,
            )
            new_nodes.append(cast_2_node)

            mul_2_node_name = poly_approx_namescope + "/mul_2"
            mul_2_node = onnx.helper.make_node(
                "Mul",
                [cast_1_node_name + "_output", cast_2_node_name + "_output"],
                [mul_2_node_name + "_output"],
                name=mul_2_node_name,
            )
            new_nodes.append(mul_2_node)

            alpha_2_node_name = poly_approx_namescope + "/alpha_2"
            alpha_2_node = onnx.helper.make_node(
                "Constant",
                [],
                [alpha_2_node_name + "_output"],
                name=alpha_2_node_name,
                value=onnx.helper.make_tensor(alpha_2_node_name + "_value", data_type, [], [0.486328125]),
            )
            new_nodes.append(alpha_2_node)

            cast_3_node_name = poly_approx_namescope + "/cast_3"
            cast_3_node = onnx.helper.make_node(
                "Cast",
                [alpha_2_node_name + "_output"],
                [cast_3_node_name + "_output"],
                name=cast_3_node_name,
                to=onnx_proto.TensorProto.FLOAT,
            )
            new_nodes.append(cast_3_node)

            add_0_node_name = poly_approx_namescope + "/add"
            add_0_node = onnx.helper.make_node(
                "Add",
                [mul_2_node_name + "_output", cast_3_node_name + "_output"],
                [add_0_node_name + "_output"],
                name=add_0_node_name,
            )
            new_nodes.append(add_0_node)

            cast_4_node_name = poly_approx_namescope + "/cast_4"
            cast_4_node = onnx.helper.make_node(
                "Cast",
                [add_0_node_name + "_output"],
                [cast_4_node_name + "_output"],
                name=cast_4_node_name,
                to=data_type,
            )
            new_nodes.append(cast_4_node)

            cast_5_node_name = poly_approx_namescope + "/cast_5"
            cast_5_node = onnx.helper.make_node(
                "Cast",
                [cast_4_node_name + "_output"],
                [cast_5_node_name + "_output"],
                name=cast_5_node_name,
                to=onnx_proto.TensorProto.FLOAT,
            )
            new_nodes.append(cast_5_node)

            mul_3_node_name = poly_approx_namescope + "/mul_3"
            mul_3_node = onnx.helper.make_node(
                "Mul",
                [cast_1_node_name + "_output", cast_5_node_name + "_output"],
                [mul_3_node_name + "_output"],
                name=mul_3_node_name,
            )
            new_nodes.append(mul_3_node)

            alpha_1_node_name = poly_approx_namescope + "/alpha_1"
            alpha_1_node = onnx.helper.make_node(
                "Constant",
                [],
                [alpha_1_node_name + "_output"],
                name=alpha_1_node_name,
                value=onnx.helper.make_tensor(alpha_1_node_name + "_value", data_type, [], [1.0]),
            )
            new_nodes.append(alpha_1_node)

            cast_6_node_name = poly_approx_namescope + "/cast_6"
            cast_6_node = onnx.helper.make_node(
                "Cast",
                [alpha_1_node_name + "_output"],
                [cast_6_node_name + "_output"],
                name=cast_6_node_name,
                to=onnx_proto.TensorProto.FLOAT,
            )
            new_nodes.append(cast_6_node)

            add_1_node_name = poly_approx_namescope + "/add_1"
            add_1_node = onnx.helper.make_node(
                "Add",
                [mul_3_node_name + "_output", cast_6_node_name + "_output"],
                [add_1_node_name + "_output"],
                name=add_1_node_name,
            )
            new_nodes.append(add_1_node)

            cast_7_node_name = poly_approx_namescope + "/cast_7"
            cast_7_node = onnx.helper.make_node(
                "Cast",
                [add_1_node_name + "_output"],
                [cast_7_node_name + "_output"],
                name=cast_7_node_name,
                to=data_type,
            )
            new_nodes.append(cast_7_node)

            cast_8_node_name = poly_approx_namescope + "/cast_8"
            cast_8_node = onnx.helper.make_node(
                "Cast",
                [cast_7_node_name + "_output"],
                [cast_8_node_name + "_output"],
                name=cast_8_node_name,
                to=onnx_proto.TensorProto.FLOAT,
            )
            new_nodes.append(cast_8_node)

            mul_4_node_name = poly_approx_namescope + "/mul_4"
            mul_4_node = onnx.helper.make_node(
                "Mul",
                [cast_1_node_name + "_output", cast_8_node_name + "_output"],
                [mul_4_node_name + "_output"],
                name=mul_4_node_name,
            )
            new_nodes.append(mul_4_node)

            alpha_0_node_name = poly_approx_namescope + "/alpha_0"
            alpha_0_node = onnx.helper.make_node(
                "Constant",
                [],
                [alpha_0_node_name + "_output"],
                name=alpha_0_node_name,
                value=onnx.helper.make_tensor(alpha_0_node_name + "_value", data_type, [], [1.0]),
            )
            new_nodes.append(alpha_0_node)

            cast_9_node_name = poly_approx_namescope + "/cast_9"
            cast_9_node = onnx.helper.make_node(
                "Cast",
                [alpha_0_node_name + "_output"],
                [cast_9_node_name + "_output"],
                name=cast_9_node_name,
                to=onnx_proto.TensorProto.FLOAT,
            )
            new_nodes.append(cast_9_node)

            add_2_node_name = poly_approx_namescope + "/add_2"
            add_2_node = onnx.helper.make_node(
                "Add",
                [mul_4_node_name + "_output", cast_9_node_name + "_output"],
                [add_2_node_name + "_output"],
                name=add_2_node_name,
            )
            new_nodes.append(add_2_node)

            cast_10_node_name = poly_approx_namescope + "/cast_10"
            cast_10_node = onnx.helper.make_node(
                "Cast",
                [add_2_node_name + "_output"],
                [cast_10_node_name + "_output"],
                name=cast_10_node_name,
                to=data_type,
            )
            new_nodes.append(cast_10_node)

            # Power: 2^floor(x*rcp_ln2) * poly(x - floor*ln2)
            pow_namescope = exp_poly_namescope + "/pow"

            pow_x_node_name = pow_namescope + "/pow/x"
            pow_x_node = onnx.helper.make_node(
                "Constant",
                [],
                [pow_x_node_name + "_output"],
                name=pow_x_node_name,
                value=onnx.helper.make_tensor(pow_x_node_name + "_value", data_type, [], [2.0]),
            )
            new_nodes.append(pow_x_node)

            pow_node_name = pow_namescope + "/pow"
            pow_node = onnx.helper.make_node(
                "Pow",
                [pow_x_node_name + "_output", round_0_node_name + "_output"],
                [pow_node_name + "_output"],
                name=pow_node_name,
            )
            new_nodes.append(pow_node)

            # exp(x) = 2^int * poly(frac)
            exp_x_node_name = exp_poly_namescope + "/exp_x"
            exp_x_node = onnx.helper.make_node(
                "Mul",
                [pow_node_name + "_output", cast_10_node_name + "_output"],
                [exp_x_node_name + "_output"],
                name=exp_x_node_name,
            )
            new_nodes.append(exp_x_node)

            return exp_x_node_name

        def _exp_sum(exp_x_node_name: str) -> str:
            nonlocal new_nodes

            exp_sum_namescope = softmax_name + "/exp_sum"

            sum_node_name = exp_sum_namescope + "/sum"
            if self._opset_version <= 12:
                sum_node = onnx.helper.make_node(
                    "ReduceSum",
                    [exp_x_node_name + "_output"],
                    [sum_node_name + "_output"],
                    name=sum_node_name,
                    axes=[reduce_axes],
                    keepdims=1,
                )
                new_nodes.append(sum_node)
            else:
                sum_axis_node_name = exp_sum_namescope + "/sum/reduction_indices"
                sum_axis_node = onnx.helper.make_node(
                    "Constant",
                    [],
                    [sum_axis_node_name + "_output"],
                    name=sum_axis_node_name,
                    value=onnx.helper.make_tensor(
                        sum_axis_node_name + "_value", onnx_proto.TensorProto.INT64, [1], [reduce_axes]
                    ),
                )
                new_nodes.append(sum_axis_node)

                sum_node = onnx.helper.make_node(
                    "ReduceSum",
                    [exp_x_node_name + "_output", sum_axis_node_name + "_output"],
                    [sum_node_name + "_output"],
                    name=sum_node_name,
                    keepdims=1,
                )
                new_nodes.append(sum_node)

            cast_sum_out_16_name = exp_sum_namescope + "/cast_reduce_sum_out_16"
            cast_sum_out_16 = onnx.helper.make_node(
                "Cast",
                [sum_node_name + "_output"],
                [cast_sum_out_16_name + "_output"],
                name=cast_sum_out_16_name,
                to=data_type,
            )
            new_nodes.append(cast_sum_out_16)

            return cast_sum_out_16_name

        def _reciprocal_approximation(exp_x_node_name: str, cast_sum_out_16_name: str) -> str:
            nonlocal new_nodes

            reciprocal_namescope = softmax_name + "/reciprocal"

            to_int_node_name = reciprocal_namescope + "/to_int"
            to_int_node = onnx.helper.make_node(
                "Bitcast",
                [cast_sum_out_16_name + "_output"],
                [to_int_node_name + "_output"],
                name=to_int_node_name,
                type=onnx_proto.TensorProto.INT16,
            )
            new_nodes.append(to_int_node)

            complement_node_name = reciprocal_namescope + "/complement"
            complement_node = onnx.helper.make_node(
                "Constant",
                [],
                [complement_node_name + "_output"],
                name=complement_node_name,
                value=onnx.helper.make_tensor(
                    complement_node_name + "_value", onnx_proto.TensorProto.INT16, [], [0x7EB5]
                ),
            )
            new_nodes.append(complement_node)

            sub_2_node_name = reciprocal_namescope + "/sub_2"
            sub_2_node = onnx.helper.make_node(
                "Sub",
                [complement_node_name + "_output", to_int_node_name + "_output"],
                [sub_2_node_name + "_output"],
                name=sub_2_node_name,
            )
            new_nodes.append(sub_2_node)

            y0_node_name = reciprocal_namescope + "/y0"
            y0_node = onnx.helper.make_node(
                "Bitcast", [sub_2_node_name + "_output"], [y0_node_name + "_output"], name=y0_node_name, type=data_type
            )
            new_nodes.append(y0_node)

            newton_k1_name = reciprocal_namescope + "/mul_6/k1"
            newton_k1 = onnx.helper.make_node(
                "Constant",
                [],
                [newton_k1_name + "_output"],
                name=newton_k1_name,
                value=onnx.helper.make_tensor(newton_k1_name + "_value", data_type, [], [1.9395974]),
            )
            new_nodes.append(newton_k1)

            mul_6_node_name = reciprocal_namescope + "/mul_6"
            mul_6_node = onnx.helper.make_node(
                "Mul",
                [y0_node_name + "_output", newton_k1_name + "_output"],
                [mul_6_node_name + "_output"],
                name=mul_6_node_name,
            )
            new_nodes.append(mul_6_node)

            cast_11_node_name = reciprocal_namescope + "/cast_11"
            cast_11_node = onnx.helper.make_node(
                "Cast",
                [cast_sum_out_16_name + "_output"],
                [cast_11_node_name + "_output"],
                name=cast_11_node_name,
                to=onnx_proto.TensorProto.FLOAT,
            )
            new_nodes.append(cast_11_node)

            cast_12_node_name = reciprocal_namescope + "/cast_12"
            cast_12_node = onnx.helper.make_node(
                "Cast",
                [y0_node_name + "_output"],
                [cast_12_node_name + "_output"],
                name=cast_12_node_name,
                to=onnx_proto.TensorProto.FLOAT,
            )
            new_nodes.append(cast_12_node)

            mul_7_node_name = reciprocal_namescope + "/mul_7"
            mul_7_node = onnx.helper.make_node(
                "Mul",
                [cast_12_node_name + "_output", cast_11_node_name + "_output"],
                [mul_7_node_name + "_output"],
                name=mul_7_node_name,
            )
            new_nodes.append(mul_7_node)

            newton_k2_name = reciprocal_namescope + "/sub_3/k2"
            newton_k2 = onnx.helper.make_node(
                "Constant",
                [],
                [newton_k2_name + "_output"],
                name=newton_k2_name,
                value=onnx.helper.make_tensor(newton_k2_name + "_value", data_type, [], [1.436142]),
            )
            new_nodes.append(newton_k2)

            cast_13_node_name = reciprocal_namescope + "/cast_13"
            cast_13_node = onnx.helper.make_node(
                "Cast",
                [newton_k2_name + "_output"],
                [cast_13_node_name + "_output"],
                name=cast_13_node_name,
                to=onnx_proto.TensorProto.FLOAT,
            )
            new_nodes.append(cast_13_node)

            sub_3_node_name = reciprocal_namescope + "/sub_3"
            sub_3_node = onnx.helper.make_node(
                "Sub",
                [cast_13_node_name + "_output", mul_7_node_name + "_output"],
                [sub_3_node_name + "_output"],
                name=sub_3_node_name,
            )
            new_nodes.append(sub_3_node)

            cast_14_node_name = reciprocal_namescope + "/cast_14"
            cast_14_node = onnx.helper.make_node(
                "Cast",
                [sub_3_node_name + "_output"],
                [cast_14_node_name + "_output"],
                name=cast_14_node_name,
                to=data_type,
            )
            new_nodes.append(cast_14_node)

            y1_node_name = reciprocal_namescope + "/y1"
            y1_node = onnx.helper.make_node(
                "Mul",
                [mul_6_node_name + "_output", cast_14_node_name + "_output"],
                [y1_node_name + "_output"],
                name=y1_node_name,
            )
            new_nodes.append(y1_node)

            cast_y1_node_name = reciprocal_namescope + "/cast_15"
            cast_y1_node = onnx.helper.make_node(
                "Cast",
                [y1_node_name + "_output"],
                [cast_y1_node_name + "_output"],
                name=cast_y1_node_name,
                to=onnx_proto.TensorProto.FLOAT,
            )
            new_nodes.append(cast_y1_node)

            mul_9_node_name = reciprocal_namescope + "/mul_9"
            mul_9_node = onnx.helper.make_node(
                "Mul",
                [cast_y1_node_name + "_output", cast_11_node_name + "_output"],
                [mul_9_node_name + "_output"],
                name=mul_9_node_name,
            )
            new_nodes.append(mul_9_node)

            newton_ones_name = reciprocal_namescope + "/add/ones"
            newton_ones = onnx.helper.make_node(
                "Constant",
                [],
                [newton_ones_name + "_output"],
                name=newton_ones_name,
                value=onnx.helper.make_tensor(newton_ones_name + "_value", data_type, [], [1.0]),
            )
            new_nodes.append(newton_ones)

            cast_16_node_name = reciprocal_namescope + "/cast_16"
            cast_16_node = onnx.helper.make_node(
                "Cast",
                [newton_ones_name + "_output"],
                [cast_16_node_name + "_output"],
                name=cast_16_node_name,
                to=onnx_proto.TensorProto.FLOAT,
            )
            new_nodes.append(cast_16_node)

            sub_4_node_name = reciprocal_namescope + "/sub_4"
            sub_4_node = onnx.helper.make_node(
                "Sub",
                [cast_16_node_name + "_output", mul_9_node_name + "_output"],
                [sub_4_node_name + "_output"],
                name=sub_4_node_name,
            )
            new_nodes.append(sub_4_node)

            cast_17_node_name = reciprocal_namescope + "/cast_17"
            cast_17_node = onnx.helper.make_node(
                "Cast",
                [sub_4_node_name + "_output"],
                [cast_17_node_name + "_output"],
                name=cast_17_node_name,
                to=data_type,
            )
            new_nodes.append(cast_17_node)

            cast_18_node_name = reciprocal_namescope + "/cast_18"
            cast_18_node = onnx.helper.make_node(
                "Cast",
                [cast_17_node_name + "_output"],
                [cast_18_node_name + "_output"],
                name=cast_18_node_name,
                to=onnx_proto.TensorProto.FLOAT,
            )
            new_nodes.append(cast_18_node)

            mul_10_node_name = reciprocal_namescope + "/mul_10"
            mul_10_node = onnx.helper.make_node(
                "Mul",
                [cast_18_node_name + "_output", cast_y1_node_name + "_output"],
                [mul_10_node_name + "_output"],
                name=mul_10_node_name,
            )
            new_nodes.append(mul_10_node)

            add_4_node_name = reciprocal_namescope + "/add_4"
            add_4_node = onnx.helper.make_node(
                "Add",
                [mul_10_node_name + "_output", cast_y1_node_name + "_output"],
                [add_4_node_name + "_output"],
                name=add_4_node_name,
            )
            new_nodes.append(add_4_node)

            cast_19_node_name = reciprocal_namescope + "/cast_19"
            cast_19_node = onnx.helper.make_node(
                "Cast",
                [add_4_node_name + "_output"],
                [cast_19_node_name + "_output"],
                name=cast_19_node_name,
                to=data_type,
            )
            new_nodes.append(cast_19_node)

            y2_node_name = softmax_name + "/y2"
            y2_node = onnx.helper.make_node(
                "Mul",
                [exp_x_node_name + "_output", cast_19_node_name + "_output"],
                [y2_node_name + "_output"],
                name=y2_node_name,
            )
            new_nodes.append(y2_node)

            return y2_node_name

        def _exp_div(exp_x_node_name: str, cast_sum_out_16_name: str) -> str:
            nonlocal new_nodes

            y2_node_name = softmax_name + "/div"
            y2_node = onnx.helper.make_node(
                "Div",
                [exp_x_node_name + "_output", cast_sum_out_16_name + "_output"],
                [y2_node_name + "_output"],
                name=y2_node_name,
            )
            new_nodes.append(y2_node)

            return y2_node_name

        input_node_name = node.input[0]
        output_node_name = node.output[0]

        # Step 1: exp(x) via polynomial (BF16)
        exp_x_node_name = _exp_poly_approximation(input_node_name)

        # Step 2: ReduceSum over softmax axis
        cast_sum_out_16_name = _exp_sum(exp_x_node_name)

        # Step 3: exp(x) / sum (reciprocal approx not used; plain Div)
        y2_node_name = _exp_div(exp_x_node_name, cast_sum_out_16_name)

        # Final cast to float
        cast_y2_node_name = softmax_name + "/output_cast"
        cast_y2_node = onnx.helper.make_node(
            "Cast",
            [y2_node_name + "_output"],
            [output_node_name],
            name=cast_y2_node_name,
            to=onnx_proto.TensorProto.FLOAT,
        )
        new_nodes.append(cast_y2_node)

        return new_nodes


class SimulateDPU:
    """Apply DPU simulation: LeakyRelu, Sigmoid->HardSigmoid, HardSigmoid, AvgPool, ReduceMean, Softmax, InstanceNorm, Clip."""

    def __init__(
        self,
        model: onnx.ModelProto,
        nodes_to_exclude: list[str],
    ) -> None:
        self.model = model
        self.nodes_to_exclude = nodes_to_exclude

    def _insert_mul(self, node: onnx.NodeProto, scale: float) -> None:
        """Insert Constant + Mul after node output to scale by factor (in-place)."""
        constant_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=[node.output[0] + "_Scale"],
            value=onnx.helper.make_tensor("scale", onnx.TensorProto.FLOAT, [], [scale]),
        )
        mul_tensor = node.output[0] + "_Mul"
        mul_node = onnx.helper.make_node(
            "Mul", inputs=[mul_tensor, node.output[0] + "_Scale"], outputs=[node.output[0]], name=mul_tensor
        )
        self.model.graph.node.extend([constant_node, mul_node])
        if not node.name:
            node.name = node.output[0]
        node.output[0] = mul_tensor

    def _convert_leaky_relu_to_dpu_version(self) -> None:
        """Convert LeakyRelu alpha to DPU format: round(alpha*256)/256."""
        for node in self.model.graph.node:
            if node.op_type == "LeakyRelu":
                alpha_attr = next((attr for attr in node.attribute if attr.name == "alpha"), None)
                if alpha_attr:
                    ori_alpha = alpha_attr.f
                    dpu_alpha = dpu_leaky_relu_alpha(alpha_attr.f)
                    alpha_attr.f = dpu_alpha
                    logger.info(
                        f"Found Leaky ReLU node {node.name} with alpha={ori_alpha}. "
                        f"Replacing with new alpha={dpu_alpha}."
                    )

    def _convert_sigmoid_to_hard_sigmoid(self) -> None:
        """Replace Sigmoid with HardSigmoid(alpha=1/6) in-place to keep topological order."""
        nodes = self.model.graph.node
        i = 0
        while i < len(nodes):
            node = nodes[i]
            if node.op_type == "Sigmoid":
                hard_sigmoid_node = onnx.helper.make_node(
                    "HardSigmoid", inputs=node.input, outputs=node.output, name=node.name
                )
                hard_sigmoid_alpha = onnx.helper.make_attribute("alpha", 1.0 / 6.0)
                hard_sigmoid_node.attribute.append(hard_sigmoid_alpha)
                nodes.remove(node)
                nodes.insert(i, hard_sigmoid_node)
                logger.info(f"Found Sigmoid node {node.name}. Replacing with HardSigmoid.")
            i += 1

    def _convert_hard_sigmoid_to_dpu_version(self) -> None:
        """Insert scale Mul after HardSigmoid (HARD_SIGMOID_SCALE) for DPU match."""
        for node in self.model.graph.node:
            if node.op_type == "HardSigmoid" and check_hard_sigmoid_condition(node):
                self._insert_mul(node, HARD_SIGMOID_SCALE)
                logger.info(f"Found HardSigmoid node {node.name} with alpha={1.0 / 6.0}. Convert to DPU version.")

    def _convert_avg_pool_to_dpu_version(self) -> None:
        """Insert kernel-dependent scale Mul after AveragePool/GlobalAveragePool for DPU rescale."""

        def _get_avgpool_scale(kh: int, kw: int) -> Any:
            """DPU rescale factor for kernel (kh, kw); fixed values for common kernels."""
            if kh > 255 or kw > 255:
                return 1.0
            elif kh == 3 and kw == 3:
                return 9.0 * 7.0 / 64.0
            elif kh == 5 and kw == 5:
                return 25.0 * 10.0 / 256.0
            elif kh == 6 and kw == 6:
                return 36.0 * 7.0 / 256.0
            elif kh == 7 and kw == 7:
                return 49.0 * 21.0 / 1024.0
            elif kh == 14 and kw == 14:
                return 196.0 * 21.0 / 4096.0
            else:
                rec = kw * kh
                n_max = 7 + math.ceil(math.log2(rec))
                ns = range(0, n_max)
                ns_pow = [2**n for n in ns]
                ks = [round(ns_p / rec) for ns_p in ns_pow]
                diffs = [abs(k / ns_p - 1 / rec) for k, ns_p in zip(ks, ns_pow, strict=False)]
                n = diffs.index(min(diffs))
                k = ks[n]
                scale = k / 2**n
                scale *= rec
                return scale

        for node in self.model.graph.node:
            if node.op_type in ["AveragePool", "GlobalAveragePool"]:
                is_global_avg_pool = node.op_type == "GlobalAveragePool"
                input_name = node.input[0]
                for n1 in self.model.graph.node:
                    if n1.output[0] == input_name:
                        if n1.op_type == "DequantizeLinear":
                            input_name = n1.input[0]
                            for n2 in self.model.graph.node:
                                if n2.output[0] == input_name:
                                    input_name = n2.input[0]
                                    break
                        else:
                            break
                input_shape = None
                shape_to_check = False
                kh = 0
                kw = 0
                if is_global_avg_pool:
                    for input_info in self.model.graph.value_info:
                        if input_info.name == input_name:
                            input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
                            if len(input_shape) == 4 and input_shape[2] == input_shape[3]:
                                shape_to_check = True
                                kh = input_shape[2]
                                kw = input_shape[3]
                            break
                    if not input_shape:
                        logger.warning(
                            f"Failed to get the input shape of GlobalAveragePool {node.name}, skip simulating DPU behavior."
                        )
                        continue
                else:
                    kernel_shape_attr = next((attr for attr in node.attribute if attr.name == "kernel_shape"), None)
                    if kernel_shape_attr:
                        kernel_shape = kernel_shape_attr.ints
                        if len(kernel_shape) == 2 and kernel_shape[0] == kernel_shape[1]:
                            shape_to_check = True
                            kh = kernel_shape[0]
                            kw = kernel_shape[1]

                if shape_to_check and (kh * kw > 0):
                    scale = _get_avgpool_scale(kh, kw)
                    self._insert_mul(node, scale)
                    logger.info(f"Rescale {node.op_type} {node.name} with factor {scale} to simulate DPU behavior.")
                else:
                    logger.warning(f"Do not support rescale {node.op_type} {node.name} to simulate DPU behavior.")

    def _convert_reduce_mean_to_dpu_version(self) -> None:
        """Insert scale Mul after ReduceMean (reduction-size-dependent) for DPU rescale."""

        def _get_reduce_mean_scale(rec: int) -> Any:
            """DPU rescale factor for reduction size rec (rational approximation)."""
            n_max = 7 + math.ceil(math.log2(rec))
            ns = range(0, n_max)
            ns_pow = [2**n for n in ns]
            ks = [round(ns_p / rec) for ns_p in ns_pow]
            diffs = [abs(k / ns_p - 1 / rec) for k, ns_p in zip(ks, ns_pow, strict=False)]
            n = diffs.index(min(diffs))
            k = ks[n]
            scale = k / 2**n
            scale *= rec
            return scale

        for node in self.model.graph.node:
            if node.op_type in ["ReduceMean"]:
                input_name = node.input[0]
                input_shape = None
                for n1 in self.model.graph.node:
                    if n1.output[0] == input_name:
                        if n1.op_type == "DequantizeLinear":
                            input_name = n1.input[0]
                            for n2 in self.model.graph.node:
                                if n2.output[0] == input_name:
                                    input_name = n2.input[0]
                                    break
                        else:
                            break
                for input_info in self.model.graph.value_info:
                    if input_info.name == input_name:
                        input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
                axes = None
                if len(node.input) == 1:
                    for attr in node.attribute:
                        if attr.name == "axes":
                            axes = attr.ints
                elif len(node.input) == 2:
                    for init in self.model.graph.initializer:
                        if init.name == node.input[1]:
                            axes = onnx.numpy_helper.to_array(init).tolist()

                if axes is not None and input_shape is not None and len(input_shape) > 0:
                    rec = 1
                    for i in axes:
                        rec *= input_shape[i]
                    if isinstance(rec, int) and rec > 0:
                        scale = _get_reduce_mean_scale(rec)
                        self._insert_mul(node, scale)
                        logger.info(f"Rescale {node.op_type} {node.name} with factor {scale} to simulate DPU behavior.")
                    else:
                        logger.warning(
                            f"Do not support rescale {node.op_type} {node.name} to simulate DPU behavior."
                            f"Please check axes: {axes} and input shape: {input_shape}."
                        )
                else:
                    logger.warning(
                        f"Do not support rescale {node.op_type} {node.name} to simulate DPU behavior. Please check axes and input shape."
                    )

    def _convert_softmax_to_dpu_version(self) -> None:
        """Replace each Softmax with SimulateDPUSoftmax subgraph."""

        nodes = []

        for node in self.model.graph.node:
            if node.op_type == "Softmax":
                nodes.append(node)

        opset_version = get_opset_version(self.model)
        for node in nodes:
            new_nodes = SimulateDPUSoftmax(opset_version=opset_version)._simulate(node)

            if len(new_nodes):
                self.model.graph.node.remove(node)
                self.model.graph.node.extend(new_nodes)

                self.nodes_to_exclude.extend(new_nodes[:-1])
                logger.info(f"Softmax {node.name} to simulate DPU behavior under opset {opset_version}.")
            else:
                logger.warning(f"Softmax {node.name} to simulate DPU behavior under opset {opset_version} failed.")

    def _convert_instance_norm_to_dpu_version(self) -> None:
        """Replace InstanceNormalization with custom op (COP_IN_OP_NAME) in-place for DPU."""
        nodes = self.model.graph.node
        i = 0
        while i < len(nodes):
            node = nodes[i]
            if node.op_type == "InstanceNormalization":
                epsilon = next((attr.f for attr in node.attribute if attr.name == "epsilon"), 1e-05)
                new_node = onnx.helper.make_node(
                    COP_IN_OP_NAME,
                    node.input,
                    node.output,
                    domain=COP_DOMAIN,
                    name=node.name,
                    epsilon=epsilon,
                )
                nodes.remove(node)
                nodes.insert(i, new_node)
                logger.info(f"InstanceNormalization node {node.name} to simulate DPU behavior by {new_node.op_type}.")
            i += 1

    def _convert_clip_to_dpu_version(self) -> None:
        """Clamp Clip min/max to [-128, 127] and ensure min/max are initializers for DPU."""
        for node in self.model.graph.node:
            if node.op_type == "Clip":
                min_value, max_value, para_type = get_clip_min_max(self.model, node)

                if para_type != 1:  # Require initializer-backed min/max
                    logger.warning(
                        f"The min and max of Clip node '{node.name}' are not initializers, conversion to the DPU version is not supported yet."
                    )
                    continue

                if min_value is not None:
                    min_value = max(-128, min(127, round(min_value)))
                    new_min_value = np.array(min_value, dtype=np.float32)
                    for initializer in self.model.graph.initializer:
                        if initializer.name == node.input[1]:
                            new_tensor = onnx.numpy_helper.from_array(new_min_value, name=node.input[1])
                            initializer.CopyFrom(new_tensor)
                            break
                elif min_value is None:
                    assert node.input[1] == "" and node.attribute == []
                    min_value = -128
                    new_min_value = np.array(min_value, dtype=np.float32)
                    new_weight_name = node.name + "_dpu_min"
                    node.input[1] = new_weight_name
                    new_weight_tensor = onnx.numpy_helper.from_array(new_min_value, name=new_weight_name)
                    self.model.graph.initializer.append(new_weight_tensor)

                if max_value is not None:
                    max_value = max(-128, min(127, round(max_value)))
                    new_max_value = np.array(max_value, dtype=np.float32)
                    for initializer in self.model.graph.initializer:
                        if initializer.name == node.input[2]:
                            new_tensor = onnx.numpy_helper.from_array(new_max_value, name=node.input[2])
                            initializer.CopyFrom(new_tensor)
                            break
                elif max_value is None:
                    assert node.input[2] == "" and node.attribute == []
                    max_value = 127
                    new_max_value = np.array(max_value, dtype=np.float32)
                    new_weight_name = node.name + "_dpu_max"
                    node.input[2] = new_weight_name
                    new_weight_tensor = onnx.numpy_helper.from_array(new_max_value, name=new_weight_name)
                    self.model.graph.initializer.append(new_weight_tensor)

                logger.info(
                    f"Clip node '{node.name}' is converted to DPU version, min is {new_min_value}, max is {new_max_value}."
                )
