#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Regression tests for GHE AMDNeuralOpt/Quark#6143: align the quark.torch QAT
# FX-graph ONNX export with the quark.onnx PTQ graph for the MS Frame-Generation
# model. Each class below covers one alignment item:
#
#   TestHardSigmoidInputQuantized  - HardSigmoid input must be Q/DQ-wrapped (scoped
#                                    to HardSigmoid; Conv+ReLU fusion untouched)
#   TestConstantDivFold            - all-constant Div folded before calibration
#   TestGridConstChainFold         - constant linspace/meshgrid/stack/unsqueeze grid
#                                    chain folded before calibration
#   TestConvScalarMulAbsorbed      - Conv-output scalar Mul folded into the Conv
#                                    weight (plain Conv only; Conv+BN left alone)
#   TestPReluSlopeQuant            - PRelu treated as a fusable activation; slope
#                                    quantized as a weight; no Conv->PRelu Q/DQ
#   TestConsecutiveSliceMerge      - two per-axis Slices merged into one multi-axis
#                                    Slice by a post-export ONNX pass
#   TestWeightQDQFold              - constant float->Q->DQ folded to int->DQ, and the
#                                    1-D PRelu-slope Unsqueeze folded, post-export
#
# The pre-calibration folds and the annotation changes run before quantizer
# insertion; the Slice merge and the weight/reshape folds are post-export ONNX
# passes that do not change QAT training.

import operator
import sys
from contextlib import contextmanager

sys.path.append("..")

import numpy as np
import onnx
import onnxruntime
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper
from torch.fx import Graph, GraphModule

import quark.torch.export.onnx as export_onnx
import quark.torch.export.qat_export_passes as qat_passes
import quark.torch.quantization.graph.processor.processor_utils as processor_utils
from quark.common.utils.import_utils import export_for_training
from quark.common.utils.testing_utils import torch_device, use_temporary_directory
from quark.torch import ModelQuantizer
from quark.torch.export.qat_export_passes import (
    _get_axes,
    _get_slice_params,
    fold_constant_reshape_after_dequant,
    fold_quantizers_for_weight,
    merge_consecutive_slices,
    merge_equivalent_constant_dequantizers,
)
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import (
    Dtype,
    QSchemeType,
    QuantizationMode,
    RoundType,
    ScaleType,
)
from quark.torch.quantization.graph.optimization.pre_quant.opt_pass_before_quant import (
    FoldConstantDivQOPass,
    FoldConvScalarMulQOPass,
)
from quark.torch.quantization.nn.modules import QuantConv2d
from quark.torch.quantization.observer.observer import PerTensorMinMaxObserver
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

# ---------------------------------------------------------------------------
# Shared quantization config (INT8 per-tensor symmetric, fx_graph_mode)
# ---------------------------------------------------------------------------

INT8_PER_TENSOR_SPEC = QTensorConfig(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)
_layer_cfg = QLayerConfig(
    input_tensors=INT8_PER_TENSOR_SPEC,
    output_tensors=INT8_PER_TENSOR_SPEC,
    weight=INT8_PER_TENSOR_SPEC,
    bias=INT8_PER_TENSOR_SPEC,
)
quant_config = QConfig(global_quant_config=_layer_cfg, quant_mode=QuantizationMode.fx_graph_mode)

_INT_ZP_DTYPES = {np.int8, np.uint8}


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def onnx_contains_op_num(model_path: str, target_op_type: str) -> int:
    return sum(1 for n in onnx.load(model_path).graph.node if n.op_type == target_op_type)


def prepare_quant_model(float_model: nn.Module, example_inputs: tuple):
    """Run export_for_training + quantize_model, returning the prepared model."""
    float_model = float_model.to(torch_device).eval()
    example_inputs = tuple(t.to(torch_device) for t in example_inputs)
    graph_model = export_for_training(float_model, example_inputs).module()
    quantizer = ModelQuantizer(quant_config)
    return quantizer.quantize_model(graph_model, list(example_inputs))


def export_qat_onnx(float_model: nn.Module, example_inputs: tuple, onnx_path: str) -> None:
    """Standard Quark QAT pipeline (quantize -> freeze) then ONNX export."""
    float_model = float_model.to(torch_device).eval()
    example_inputs = tuple(t.to(torch_device) for t in example_inputs)
    graph_model = export_for_training(float_model, example_inputs).module()
    quantizer = ModelQuantizer(quant_config)
    quantized_model = quantizer.quantize_model(graph_model, list(example_inputs))
    frozen_model = quantizer.freeze(quantized_model.eval())
    frozen_model(*example_inputs)  # warm-up
    torch.onnx.export(frozen_model, example_inputs, onnx_path, dynamo=False)


def disable_fake_quant(prepared) -> None:
    for module in prepared.modules():
        if isinstance(module, ScaledFakeQuantize):
            module.disable_fake_quant()
            module.disable_observer()


# ===========================================================================
# Item 1: HardSigmoid input must be Q/DQ-wrapped (scoped to HardSigmoid)
# ===========================================================================


class _ConvHardsigmoid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.hardsigmoid(self.conv(x))


class _AddHardsigmoid(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.hardsigmoid(self.conv1(x) + self.conv2(x))


class _ConvRelu(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv(x))


def _hardsigmoid_input_is_dq_wrapped(model_path: str) -> bool:
    model = onnx.load(model_path)
    out2node = {o: n for n in model.graph.node for o in n.output}
    for node in model.graph.node:
        if node.op_type != "HardSigmoid":
            continue
        producer = out2node.get(node.input[0])
        if producer is None or producer.op_type != "DequantizeLinear":
            return False
    return True


class TestHardSigmoidInputQuantized:
    """HardSigmoid input gets a Q/DQ; the fix must not touch Conv+ReLU fusion."""

    @use_temporary_directory
    def test_conv_hardsigmoid_input_has_dq(self, tmpdir: str) -> None:
        torch.cuda.empty_cache()
        onnx_path = tmpdir + "/conv_hardsigmoid.onnx"
        export_qat_onnx(_ConvHardsigmoid(), (torch.rand(2, 3, 8, 8),), onnx_path)

        assert onnx_contains_op_num(onnx_path, "HardSigmoid") == 1, (
            "Expected exactly one HardSigmoid node in the exported ONNX"
        )
        assert _hardsigmoid_input_is_dq_wrapped(onnx_path), (
            "HardSigmoid input is NOT produced by a DequantizeLinear node "
            "(unpatched _annotate_conv_act / _annotate_quantized_convbn_2d_act)."
        )
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_add_hardsigmoid_input_has_dq(self, tmpdir: str) -> None:
        torch.cuda.empty_cache()
        onnx_path = tmpdir + "/add_hardsigmoid.onnx"
        export_qat_onnx(_AddHardsigmoid(), (torch.rand(2, 3, 8, 8),), onnx_path)

        assert onnx_contains_op_num(onnx_path, "HardSigmoid") == 1, (
            "Expected exactly one HardSigmoid node in the exported ONNX"
        )
        assert _hardsigmoid_input_is_dq_wrapped(onnx_path), (
            "HardSigmoid input is NOT produced by a DequantizeLinear node (unpatched _annotate_add_relu)."
        )
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_conv_relu_no_extra_dq(self, tmpdir: str) -> None:
        """Conv -> ReLU must NOT gain an extra DQ between Conv and ReLU."""
        torch.cuda.empty_cache()
        onnx_path = tmpdir + "/conv_relu.onnx"
        export_qat_onnx(_ConvRelu(), (torch.rand(2, 3, 8, 8),), onnx_path)

        m = onnx.load(onnx_path)
        out2node = {o: n for n in m.graph.node for o in n.output}
        relu_nodes = [n for n in m.graph.node if n.op_type == "Relu"]
        assert relu_nodes, "Expected at least one Relu node in the ONNX export"
        for relu in relu_nodes:
            producer = out2node.get(relu.input[0])
            assert producer is None or producer.op_type != "DequantizeLinear", (
                "Relu input is produced by DequantizeLinear — the fix must be "
                "restricted to HardSigmoid only, not Conv->ReLU fusion."
            )
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_conv_hardsigmoid_numerically_close(self, tmpdir: str) -> None:
        torch.manual_seed(0)
        float_model = _ConvHardsigmoid().to(torch_device).eval()
        example_inputs = (torch.rand(2, 3, 8, 8).to(torch_device),)
        with torch.no_grad():
            float_out = float_model(*example_inputs)

        prepared = prepare_quant_model(float_model, example_inputs)
        with torch.no_grad():
            quant_out = prepared(*example_inputs)
        assert (float_out - quant_out).abs().max().item() < 0.1, (
            "Quantized Conv->HardSigmoid output deviates too far from float reference"
        )

        disable_fake_quant(prepared)
        with torch.no_grad():
            struct_out = prepared(*example_inputs)
        torch.testing.assert_close(struct_out, float_out, rtol=1e-4, atol=1e-4)
        torch.cuda.empty_cache()


# ===========================================================================
# Item 4: all-constant Div folded before calibration
# ===========================================================================


class _ConvGridNorm(nn.Module):
    """Conv followed by grid normalization with an all-constant Div."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        size = x.new_tensor([w, h]).view(-1, 1, 1) / 2  # constant given fixed shape
        return self.conv(x) / size


class _ConvRuntimeDiv(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) / x  # both operands runtime


class TestConstantDivFold:
    """FoldConstantDivQOPass folds an all-constant Div; runtime Div is left alone."""

    @use_temporary_directory
    def test_constant_div_folded_before_calibration(self, tmpdir: str) -> None:
        torch.cuda.empty_cache()
        onnx_path = tmpdir + "/grid_norm.onnx"
        export_qat_onnx(_ConvGridNorm(), (torch.rand(1, 2, 4, 4),), onnx_path)
        div_count = onnx_contains_op_num(onnx_path, "Div")
        assert div_count == 1, f"Expected 1 Div (runtime x/size; constant [w,h]/2 folded), got {div_count}."
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_runtime_div_not_folded(self, tmpdir: str) -> None:
        torch.cuda.empty_cache()
        onnx_path = tmpdir + "/runtime_div.onnx"
        export_qat_onnx(_ConvRuntimeDiv(), (torch.rand(1, 2, 4, 4),), onnx_path)
        div_count = onnx_contains_op_num(onnx_path, "Div")
        assert div_count == 1, f"A runtime-runtime Div must NOT be folded. Expected 1, got {div_count}."
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_constant_div_fold_numerically_close(self, tmpdir: str) -> None:
        torch.manual_seed(0)
        float_model = _ConvGridNorm().to(torch_device).eval()
        example_inputs = (torch.rand(1, 2, 4, 4).to(torch_device),)
        with torch.no_grad():
            float_out = float_model(*example_inputs)

        prepared = prepare_quant_model(float_model, example_inputs)
        with torch.no_grad():
            quant_out = prepared(*example_inputs)
        assert (float_out - quant_out).abs().max().item() < 0.1, (
            "Quantized Conv->Div output deviates too far from float reference"
        )

        disable_fake_quant(prepared)
        with torch.no_grad():
            struct_out = prepared(*example_inputs)
        torch.testing.assert_close(struct_out, float_out, rtol=1e-4, atol=1e-4)
        torch.cuda.empty_cache()


# ===========================================================================
# Item 4b: constant grid-construction chain folded before calibration
# ===========================================================================


class _GridConstModel(nn.Module):
    """Conv output added to a constant coordinate grid (linspace/meshgrid/stack)."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        xs = torch.linspace(-w / 2, w / 2, w, device=x.device, dtype=x.dtype)
        ys = torch.linspace(-h / 2, h / 2, h, device=x.device, dtype=x.dtype)
        gx, gy = torch.meshgrid(xs, ys, indexing="xy")
        grid = torch.stack((gx, gy), dim=0).unsqueeze(0)  # (1, 2, h, w), constant
        return self.conv(x) + grid


class TestGridConstChainFold:
    """FoldConstantDivQOPass collapses the whole constant grid chain."""

    @use_temporary_directory
    def test_grid_const_chain_folded(self, tmpdir: str) -> None:
        torch.cuda.empty_cache()
        onnx_path = tmpdir + "/grid_const.onnx"
        export_qat_onnx(_GridConstModel(), (torch.rand(1, 2, 8, 8),), onnx_path)
        for op in ("Range", "Unsqueeze", "Concat", "Expand"):
            count = onnx_contains_op_num(onnx_path, op)
            assert count == 0, (
                f"Constant grid chain not fully folded: found {count} {op} node(s); "
                f"the linspace/meshgrid/stack/unsqueeze chain should collapse to a constant."
            )
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_grid_const_fold_numerically_close(self, tmpdir: str) -> None:
        torch.manual_seed(0)
        float_model = _GridConstModel().to(torch_device).eval()
        example_inputs = (torch.rand(1, 2, 8, 8).to(torch_device),)
        with torch.no_grad():
            float_out = float_model(*example_inputs)

        prepared = prepare_quant_model(float_model, example_inputs)
        disable_fake_quant(prepared)
        with torch.no_grad():
            struct_out = prepared(*example_inputs)
        torch.testing.assert_close(struct_out, float_out, rtol=1e-4, atol=1e-4)
        torch.cuda.empty_cache()


# ===========================================================================
# Item 3: Conv-output scalar Mul absorbed into the Conv weight (plain Conv only)
# ===========================================================================


class _ConvScalarMul(nn.Module):
    """Conv followed by a constant scalar multiply."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x) * 32.0


class _ConvRuntimeMul(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 1)
        self.conv2 = nn.Conv2d(3, 8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv1(x) * self.conv2(x)


class _ConvBNScalarMul(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)
        self.bn = nn.BatchNorm2d(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x)) * 32.0


class TestConvScalarMulAbsorbed:
    """FoldConvScalarMulQOPass folds a scalar into a plain Conv weight only."""

    @use_temporary_directory
    def test_conv_scalar_mul_absorbed_into_weight(self, tmpdir: str) -> None:
        torch.cuda.empty_cache()
        onnx_path = tmpdir + "/conv_scalar_mul.onnx"
        export_qat_onnx(_ConvScalarMul(), (torch.rand(1, 3, 4, 4),), onnx_path)
        mul_count = onnx_contains_op_num(onnx_path, "Mul")
        assert mul_count == 0, f"Expected 0 Mul nodes (scalar folded into Conv weight), got {mul_count}."
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_runtime_mul_not_absorbed(self, tmpdir: str) -> None:
        torch.cuda.empty_cache()
        onnx_path = tmpdir + "/conv_runtime_mul.onnx"
        export_qat_onnx(_ConvRuntimeMul(), (torch.rand(1, 3, 4, 4),), onnx_path)
        mul_count = onnx_contains_op_num(onnx_path, "Mul")
        assert mul_count == 1, f"A runtime-runtime Mul must NOT be absorbed. Expected 1, got {mul_count}."
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_conv_bn_scalar_mul_not_absorbed(self, tmpdir: str) -> None:
        """Conv+BN -> Mul(scalar): the Mul must NOT be absorbed (BN not yet merged)."""
        torch.manual_seed(0)
        float_model = _ConvBNScalarMul().to(torch_device).eval()
        float_model.bn.running_mean.fill_(0.5)
        float_model.bn.running_var.fill_(2.0)
        float_model.bn.weight.data.fill_(1.5)
        float_model.bn.bias.data.fill_(0.3)
        float_model.eval()

        example_inputs = (torch.rand(1, 2, 4, 4).to(torch_device),)
        with torch.no_grad():
            float_out = float_model(*example_inputs)

        prepared = prepare_quant_model(float_model, example_inputs)
        mul_nodes = [n for n in prepared.graph.nodes if "mul" in str(n.target).lower()]
        assert len(mul_nodes) == 1, (
            f"Conv+BN scalar Mul must NOT be absorbed (BN not yet merged). Expected 1 Mul node, got {len(mul_nodes)}."
        )

        disable_fake_quant(prepared)
        with torch.no_grad():
            prepared_out = prepared(*example_inputs)
        torch.testing.assert_close(prepared_out, float_out, rtol=1e-3, atol=1e-3)
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_conv_scalar_mul_numerically_close(self, tmpdir: str) -> None:
        torch.manual_seed(0)
        float_model = _ConvScalarMul().to(torch_device).eval()
        example_inputs = (torch.rand(1, 3, 4, 4).to(torch_device),)
        with torch.no_grad():
            float_out = float_model(*example_inputs)

        prepared = prepare_quant_model(float_model, example_inputs)
        disable_fake_quant(prepared)
        with torch.no_grad():
            struct_out = prepared(*example_inputs)
        torch.testing.assert_close(struct_out, float_out, rtol=1e-4, atol=1e-4)
        torch.cuda.empty_cache()


# ===========================================================================
# Item 6: PRelu slope quantized as weight; no Q/DQ between Conv and PRelu
# ===========================================================================


class _ConvPRelu(nn.Module):
    """Conv immediately followed by PReLU, mirroring the FG model's blocks."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)
        self.act = nn.PReLU(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class TestPReluSlopeQuant:
    """PRelu is a fusable activation: slope weight-quantized, Conv->PRelu fused."""

    @use_temporary_directory
    def test_conv_prelu_no_intermediate_qdq(self, tmpdir: str) -> None:
        torch.cuda.empty_cache()
        onnx_path = tmpdir + "/conv_prelu.onnx"
        export_qat_onnx(_ConvPRelu(), (torch.rand(1, 2, 8, 8),), onnx_path)
        graph = onnx.load(onnx_path).graph
        producer = {out: node for node in graph.node for out in node.output}

        prelu_nodes = [node for node in graph.node if node.op_type == "PRelu"]
        assert prelu_nodes, "no PRelu node found in exported ONNX"
        for prelu in prelu_nodes:
            activation_source = producer.get(prelu.input[0])
            assert activation_source is not None, "PRelu activation input has no producer"
            assert activation_source.op_type != "DequantizeLinear", (
                "found a DequantizeLinear on the Conv->PRelu tensor; the intermediate "
                "quantizer was not removed (Conv and PRelu should fuse like PTQ)."
            )
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_prelu_slope_quantized(self, tmpdir: str) -> None:
        torch.cuda.empty_cache()
        onnx_path = tmpdir + "/conv_prelu.onnx"
        export_qat_onnx(_ConvPRelu(), (torch.rand(1, 2, 8, 8),), onnx_path)
        graph = onnx.load(onnx_path).graph
        producer = {out: node for node in graph.node for out in node.output}

        prelu_nodes = [node for node in graph.node if node.op_type == "PRelu"]
        assert prelu_nodes, "no PRelu node found in exported ONNX"

        # The slope may be reshaped (Unsqueeze) between the quantizer and PRelu,
        # so trace back through shape-only ops to find the DequantizeLinear.
        shape_only_ops = {"Unsqueeze", "Reshape", "Squeeze", "Identity"}

        def has_quantized_slope(slope_tensor: str) -> bool:
            current = producer.get(slope_tensor)
            while current is not None and current.op_type in shape_only_ops:
                current = producer.get(current.input[0])
            return current is not None and current.op_type == "DequantizeLinear"

        for prelu in prelu_nodes:
            assert has_quantized_slope(prelu.input[1]), (
                "PRelu slope is not quantized: expected a DequantizeLinear feeding "
                "input[1] (possibly through a reshape), matching PTQ."
            )
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_conv_prelu_numerically_close(self, tmpdir: str) -> None:
        torch.manual_seed(0)
        float_model = _ConvPRelu().to(torch_device).eval()
        example_inputs = (torch.rand(1, 2, 8, 8).to(torch_device),)
        with torch.no_grad():
            float_out = float_model(*example_inputs)

        prepared = prepare_quant_model(float_model, example_inputs)
        disable_fake_quant(prepared)
        with torch.no_grad():
            struct_out = prepared(*example_inputs)
        torch.testing.assert_close(struct_out, float_out, rtol=1e-4, atol=1e-4)
        torch.cuda.empty_cache()


# ===========================================================================
# Item 2: consecutive per-axis Slices merged into one multi-axis Slice
# (post-export ONNX pass; does not change QAT training)
# ===========================================================================


def _build_consecutive_slice_model(onnx_path: str, with_fanout: bool) -> None:
    """Hand-build input -> Slice(A, axis 2) -> Q -> DQ -> Slice(B, axis 3) -> Q -> DQ
    with a transparent intermediate Q/DQ. If with_fanout, Slice(A) also feeds an Add
    (a second real consumer) that the pass must not break.
    """
    scale = numpy_helper.from_array(np.float32(0.5), name="scale")
    zp = numpy_helper.from_array(np.int8(0), name="zp")

    def const(name: str, values: list[int]):
        return numpy_helper.from_array(np.array(values, dtype=np.int64), name=name)

    initializers = [
        scale,
        zp,
        const("a_starts", [0]),
        const("a_ends", [8]),
        const("a_axes", [2]),
        const("a_steps", [2]),
        const("b_starts", [0]),
        const("b_ends", [8]),
        const("b_axes", [3]),
        const("b_steps", [2]),
    ]
    # Feed Slice A from an upstream DequantizeLinear sharing the same scalar scale/zp, so the
    # tensor entering Slice A is already on the quant grid (as in real QAT exports). This makes
    # the intermediate Q/DQ a lossless round-trip and satisfies merge_consecutive_slices' guard.
    nodes = [
        helper.make_node("QuantizeLinear", ["x", "scale", "zp"], ["xq"], name="InQ"),
        helper.make_node("DequantizeLinear", ["xq", "scale", "zp"], ["xdq"], name="InDQ"),
        helper.make_node("Slice", ["xdq", "a_starts", "a_ends", "a_axes", "a_steps"], ["a_out"], name="SliceA"),
        helper.make_node("QuantizeLinear", ["a_out", "scale", "zp"], ["q_out"], name="Q"),
        helper.make_node("DequantizeLinear", ["q_out", "scale", "zp"], ["dq_out"], name="DQ"),
        helper.make_node("Slice", ["dq_out", "b_starts", "b_ends", "b_axes", "b_steps"], ["b_out"], name="SliceB"),
        helper.make_node("QuantizeLinear", ["b_out", "scale", "zp"], ["q2_out"], name="Q2"),
        helper.make_node("DequantizeLinear", ["q2_out", "scale", "zp"], ["y"], name="DQ2"),
    ]
    outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT, None)]
    if with_fanout:
        initializers.append(numpy_helper.from_array(np.float32(1.0), name="one"))
        nodes.append(helper.make_node("Add", ["a_out", "one"], ["fanout"], name="Fanout"))
        outputs.append(helper.make_tensor_value_info("fanout", TensorProto.FLOAT, None))

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8, 8])
    graph = helper.make_graph(nodes, "consecutive_slices", [x], outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10
    onnx.save(model, onnx_path)


class _ConvSliceModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)[..., ::3, ::3]


class TestConsecutiveSliceMerge:
    """merge_consecutive_slices merges Slice->Q->DQ->Slice into one multi-axis Slice."""

    @use_temporary_directory
    def test_consecutive_slices_merged(self, tmpdir: str) -> None:
        from quark.torch.export.qat_export_passes import merge_consecutive_slices

        onnx_path = tmpdir + "/slices.onnx"
        _build_consecutive_slice_model(onnx_path, with_fanout=False)
        assert onnx_contains_op_num(onnx_path, "Slice") == 2
        sample = np.random.RandomState(0).rand(1, 4, 8, 8).astype(np.float32)

        def run(path: str) -> np.ndarray:
            session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
            return session.run(None, {"x": sample})[0]

        before = run(onnx_path)
        merge_consecutive_slices(onnx_path)
        after = run(onnx_path)

        assert onnx_contains_op_num(onnx_path, "Slice") == 1, "the two Slices should merge into one"
        np.testing.assert_allclose(before, after, rtol=0, atol=0)

    @use_temporary_directory
    def test_fanout_branch_preserved(self, tmpdir: str) -> None:
        from quark.torch.export.qat_export_passes import merge_consecutive_slices

        onnx_path = tmpdir + "/slices_fanout.onnx"
        _build_consecutive_slice_model(onnx_path, with_fanout=True)
        assert onnx_contains_op_num(onnx_path, "Slice") == 2
        merge_consecutive_slices(onnx_path)
        assert onnx_contains_op_num(onnx_path, "Slice") == 2, "fan-out Slice must be preserved, not merged"

    @use_temporary_directory
    def test_end_to_end_qat_slices_match_ptq(self, tmpdir: str) -> None:
        from quark.torch.export.qat_export_passes import merge_consecutive_slices

        torch.cuda.empty_cache()
        onnx_path = tmpdir + "/conv_slice.onnx"
        export_qat_onnx(_ConvSliceModel(), (torch.rand(1, 3, 9, 9),), onnx_path)
        assert onnx_contains_op_num(onnx_path, "Slice") == 2
        merge_consecutive_slices(onnx_path)
        assert onnx_contains_op_num(onnx_path, "Slice") == 1, "consecutive Slices should merge to one"
        torch.cuda.empty_cache()


# ===========================================================================
# Item: constant weight float->Q->DQ folded to int->DQ, and the 1-D PRelu-slope
# Unsqueeze folded (post-export ONNX passes; do not change QAT training)
# ===========================================================================


class _ConstQuantModel(nn.Module):
    """Conv weight + PReLU slope + a constant Add operand — three quantized consts."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)
        self.act = nn.PReLU(2)
        self.register_buffer("shift", torch.rand(1, 2, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x)) + self.shift


def _export_const_quant_onnx(tmpdir: str) -> str:
    onnx_path = tmpdir + "/const_quant.onnx"
    export_qat_onnx(_ConstQuantModel(), (torch.rand(1, 2, 8, 8),), onnx_path)
    return onnx_path


def _build_dq_unsqueeze_model(onnx_path: str) -> np.ndarray:
    """Hand-build int8_const -> DequantizeLinear(per-tensor) -> Unsqueeze -> Add,
    the pattern torch.onnx leaves for a 1-D PRelu slope. Returns dequantized values.
    """
    slope_int8 = np.array([3, -5, 7, -9], dtype=np.int8)
    scale = np.float32(0.5)
    zero_point = np.int8(0)
    axes = np.array([0, 2, 3], dtype=np.int64)  # (C,) -> (1, C, 1, 1)

    initializers = [
        numpy_helper.from_array(slope_int8, name="slope_int8"),
        numpy_helper.from_array(np.array(scale, dtype=np.float32), name="slope_scale"),
        numpy_helper.from_array(np.array(zero_point, dtype=np.int8), name="slope_zp"),
        numpy_helper.from_array(axes, name="unsqueeze_axes"),
    ]
    nodes = [
        helper.make_node("DequantizeLinear", ["slope_int8", "slope_scale", "slope_zp"], ["slope_f"]),
        helper.make_node("Unsqueeze", ["slope_f", "unsqueeze_axes"], ["slope_4d"]),
        helper.make_node("Add", ["x", "slope_4d"], ["y"]),
    ]
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 1, 1])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 1, 1])
    graph = helper.make_graph(nodes, "dq_unsqueeze", [x], [y], initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10  # match the onnxruntime build used in CI
    onnx.save(model, onnx_path)
    return (slope_int8.astype(np.float32) - float(zero_point)) * float(scale)


class TestWeightQDQFold:
    """fold_quantizers_for_weight + fold_constant_reshape_after_dequant match PTQ packaging."""

    @use_temporary_directory
    def test_no_quantizelinear_on_constants(self, tmpdir: str) -> None:
        torch.cuda.empty_cache()
        onnx_path = _export_const_quant_onnx(tmpdir)
        fold_quantizers_for_weight(onnx_path)

        graph = onnx.load(onnx_path).graph
        initializer_names = {init.name for init in graph.initializer}
        offending = [
            node.name for node in graph.node if node.op_type == "QuantizeLinear" and node.input[0] in initializer_names
        ]
        assert not offending, (
            f"{len(offending)} QuantizeLinear node(s) still sit on constant initializers "
            f"after folding; constants should be pre-quantized int -> DequantizeLinear."
        )
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_weight_dq_reads_int_initializer(self, tmpdir: str) -> None:
        torch.cuda.empty_cache()
        onnx_path = _export_const_quant_onnx(tmpdir)
        fold_quantizers_for_weight(onnx_path)

        graph = onnx.load(onnx_path).graph
        name_to_initializer = {init.name: init for init in graph.initializer}
        producer = {out: node for node in graph.node for out in node.output}

        conv_nodes = [node for node in graph.node if node.op_type in ("Conv", "ConvTranspose")]
        assert conv_nodes, "no Conv node found in exported ONNX"
        for conv in conv_nodes:
            weight_producer = producer.get(conv.input[1])
            assert weight_producer is not None and weight_producer.op_type == "DequantizeLinear", (
                "Conv weight is not fed by a DequantizeLinear after folding"
            )
            int_const = weight_producer.input[0]
            assert int_const in name_to_initializer, "weight DQ input[0] is not an initializer"
            dtype = numpy_helper.to_array(name_to_initializer[int_const]).dtype.type
            assert dtype in _INT_ZP_DTYPES, f"folded weight initializer dtype {dtype} is not int8/uint8"
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_folded_onnx_numerically_equivalent(self, tmpdir: str) -> None:
        torch.cuda.empty_cache()
        onnx_path = _export_const_quant_onnx(tmpdir)
        sample = np.random.RandomState(0).rand(1, 2, 8, 8).astype(np.float32)

        def run(path: str) -> np.ndarray:
            session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
            input_name = session.get_inputs()[0].name
            return session.run(None, {input_name: sample})[0]

        before = run(onnx_path)
        fold_quantizers_for_weight(onnx_path)
        after = run(onnx_path)
        np.testing.assert_allclose(before, after, rtol=0, atol=0)
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_prelu_slope_reshape_folded(self, tmpdir: str) -> None:
        """The residual constant Unsqueeze broadcasting a 1-D slope must be folded
        into the constant (deterministic hand-built graph)."""
        torch.cuda.empty_cache()
        onnx_path = tmpdir + "/dq_unsqueeze.onnx"
        _build_dq_unsqueeze_model(onnx_path)

        graph_before = onnx.load(onnx_path).graph
        assert sum(1 for node in graph_before.node if node.op_type == "Unsqueeze") == 1

        sample = np.random.RandomState(0).rand(1, 4, 1, 1).astype(np.float32)

        def run(path: str) -> np.ndarray:
            session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
            return session.run(None, {"x": sample})[0]

        before = run(onnx_path)
        fold_constant_reshape_after_dequant(onnx_path)
        after = run(onnx_path)

        graph_after = onnx.load(onnx_path).graph
        assert sum(1 for node in graph_after.node if node.op_type == "Unsqueeze") == 0, (
            "the constant Unsqueeze should be folded into the reshaped constant"
        )
        np.testing.assert_allclose(before, after, rtol=0, atol=0)
        torch.cuda.empty_cache()


# ===========================================================================
# Branch coverage for the post-export ONNX passes: every early-exit / guard in
# qat_export_passes.py exercised on tiny hand-built graphs. These are pure ONNX
# manipulations (no torch export, no GPU), so they are fast and deterministic.
# ===========================================================================


def _fv(name: str, shape: list[int] | None = None):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, shape)


def _i64(name: str, values: list[int]):
    return numpy_helper.from_array(np.array(values, dtype=np.int64), name=name)


def _save_onnx(nodes, initializers, inputs, outputs, path: str, opset: int = 13) -> None:
    graph = helper.make_graph(nodes, "g", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = 10
    onnx.save(model, path)


def _op_count(path: str, op_type: str) -> int:
    return sum(1 for n in onnx.load(path).graph.node if n.op_type == op_type)


@contextmanager
def _patched_slim(func):
    """Temporarily replace onnxslim.slim as seen by qat_export_passes."""
    original = qat_passes.slim
    qat_passes.slim = func
    try:
        yield
    finally:
        qat_passes.slim = original


class _SlimFailsAfterN:
    """Callable that returns the model for the first ``n`` calls, then raises.

    Used to hit the *post-transform* ``slim`` failure branches: the pass must
    still save the (unsimplified) transformed model instead of aborting.
    """

    def __init__(self, n: int = 1) -> None:
        self.calls = 0
        self.n = n

    def __call__(self, model):
        self.calls += 1
        if self.calls > self.n:
            raise RuntimeError("slim boom")
        return model


class TestGetAxesHelper:
    """_get_axes: attribute (opset < 13), axes-input initializer, and dynamic."""

    def test_axes_from_attribute(self) -> None:
        node = helper.make_node("Unsqueeze", ["x"], ["y"], axes=[0, 2])
        assert _get_axes(node, {}) == [0, 2]

    def test_axes_from_initializer(self) -> None:
        axes = numpy_helper.from_array(np.array([1, 3], dtype=np.int64), name="ax")
        node = helper.make_node("Unsqueeze", ["x", "ax"], ["y"])
        assert _get_axes(node, {"ax": axes}) == [1, 3]

    def test_axes_dynamic_returns_none(self) -> None:
        node = helper.make_node("Unsqueeze", ["x", "dyn"], ["y"])
        assert _get_axes(node, {}) is None


class TestGetSliceParamsHelper:
    """_get_slice_params: every dynamic/missing early-exit plus the default paths."""

    def test_fewer_than_three_inputs(self) -> None:
        node = helper.make_node("Slice", ["x"], ["y"])
        assert _get_slice_params(node, {}) is None

    def test_starts_ends_not_initializer(self) -> None:
        node = helper.make_node("Slice", ["x", "st", "en"], ["y"])
        assert _get_slice_params(node, {}) is None

    def test_default_axes_and_steps(self) -> None:
        inits = {"st": _i64("st", [0]), "en": _i64("en", [4])}
        node = helper.make_node("Slice", ["x", "st", "en"], ["y"])
        assert _get_slice_params(node, inits) == {0: (0, 4, 1)}

    def test_axes_input_not_initializer(self) -> None:
        inits = {"st": _i64("st", [0]), "en": _i64("en", [4])}
        node = helper.make_node("Slice", ["x", "st", "en", "axdyn"], ["y"])
        assert _get_slice_params(node, inits) is None

    def test_steps_input_not_initializer(self) -> None:
        inits = {"st": _i64("st", [0]), "en": _i64("en", [4]), "ax": _i64("ax", [2])}
        node = helper.make_node("Slice", ["x", "st", "en", "ax", "stpdyn"], ["y"])
        assert _get_slice_params(node, inits) is None


class TestFoldQuantizersForWeightBranches:
    """Guards inside fold_quantizers_for_weight (per-channel, non-int8 zp, no DQ, slim)."""

    @use_temporary_directory
    def test_per_channel_axis_reshape(self, tmpdir: str) -> None:
        """A per-channel (axis, 1-D scale) constant Q/DQ is folded, exercising the
        broadcast-shape reshape path."""
        onnx_path = tmpdir + "/per_channel.onnx"
        weight = numpy_helper.from_array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32), name="w")
        scale = numpy_helper.from_array(np.array([0.1, 0.2], dtype=np.float32), name="scale")
        zero_point = numpy_helper.from_array(np.array([0, 0], dtype=np.int8), name="zp")
        nodes = [
            helper.make_node("QuantizeLinear", ["w", "scale", "zp"], ["qo"], axis=0),
            helper.make_node("DequantizeLinear", ["qo", "scale", "zp"], ["y"], axis=0),
        ]
        _save_onnx(nodes, [weight, scale, zero_point], [], [_fv("y")], onnx_path)
        fold_quantizers_for_weight(onnx_path)
        assert _op_count(onnx_path, "QuantizeLinear") == 0

    @use_temporary_directory
    def test_int32_zero_point_not_folded(self, tmpdir: str) -> None:
        """int32 zero-point (bias) is left for fold_quantizers_for_bias."""
        onnx_path = tmpdir + "/int32.onnx"
        weight = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), name="w")
        scale = numpy_helper.from_array(np.float32(0.1), name="scale")
        zero_point = numpy_helper.from_array(np.array([0, 0], dtype=np.int32), name="zp")
        nodes = [
            helper.make_node("QuantizeLinear", ["w", "scale", "zp"], ["qo"]),
            helper.make_node("DequantizeLinear", ["qo", "scale", "zp"], ["y"]),
        ]
        _save_onnx(nodes, [weight, scale, zero_point], [], [_fv("y")], onnx_path)
        with _patched_slim(lambda m: m):
            fold_quantizers_for_weight(onnx_path)
        assert _op_count(onnx_path, "QuantizeLinear") == 1

    @use_temporary_directory
    def test_zero_point_not_initializer(self, tmpdir: str) -> None:
        """A QuantizeLinear whose zero-point is produced by a node (not an
        initializer) is skipped."""
        onnx_path = tmpdir + "/zp_node.onnx"
        weight = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), name="w")
        scale = numpy_helper.from_array(np.float32(0.1), name="scale")
        zp_const = numpy_helper.from_array(np.array([0], dtype=np.int8), name="zc")
        nodes = [
            helper.make_node("Identity", ["zc"], ["z"]),
            helper.make_node("QuantizeLinear", ["w", "scale", "z"], ["qo"]),
            helper.make_node("DequantizeLinear", ["qo", "scale", "z"], ["y"]),
        ]
        _save_onnx(nodes, [weight, scale, zp_const], [], [_fv("y")], onnx_path)
        with _patched_slim(lambda m: m):
            fold_quantizers_for_weight(onnx_path)
        assert _op_count(onnx_path, "QuantizeLinear") == 1

    @use_temporary_directory
    def test_quantize_without_dequant_consumer(self, tmpdir: str) -> None:
        """A constant QuantizeLinear whose output does not feed a DequantizeLinear
        is not folded."""
        onnx_path = tmpdir + "/no_dq.onnx"
        weight = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), name="w")
        scale = numpy_helper.from_array(np.float32(0.1), name="scale")
        zero_point = numpy_helper.from_array(np.int8(0), name="zp")
        nodes = [
            helper.make_node("QuantizeLinear", ["w", "scale", "zp"], ["qo"]),
            helper.make_node("Identity", ["qo"], ["y"]),
        ]
        _save_onnx(nodes, [weight, scale, zero_point], [], [_fv("y")], onnx_path)
        with _patched_slim(lambda m: m):
            fold_quantizers_for_weight(onnx_path)
        assert _op_count(onnx_path, "QuantizeLinear") == 1

    @use_temporary_directory
    def test_initial_slim_failure_aborts(self, tmpdir: str) -> None:
        """If the first slim() raises, the pass logs and returns without changes."""
        onnx_path = tmpdir + "/slim_fail.onnx"
        weight = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="w")
        _save_onnx([helper.make_node("Identity", ["w"], ["y"])], [weight], [], [_fv("y")], onnx_path)

        def boom(_model):
            raise RuntimeError("slim boom")

        with _patched_slim(boom):
            fold_quantizers_for_weight(onnx_path)
        assert _op_count(onnx_path, "Identity") == 1

    @use_temporary_directory
    def test_post_transform_slim_failure_still_saves(self, tmpdir: str) -> None:
        """When the second slim() (after folding) raises, the folded model is still saved."""
        onnx_path = tmpdir + "/slim_fail2.onnx"
        weight = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), name="w")
        scale = numpy_helper.from_array(np.float32(0.1), name="scale")
        zero_point = numpy_helper.from_array(np.int8(0), name="zp")
        nodes = [
            helper.make_node("QuantizeLinear", ["w", "scale", "zp"], ["qo"]),
            helper.make_node("DequantizeLinear", ["qo", "scale", "zp"], ["y"]),
        ]
        _save_onnx(nodes, [weight, scale, zero_point], [], [_fv("y")], onnx_path)
        with _patched_slim(_SlimFailsAfterN(1)):
            fold_quantizers_for_weight(onnx_path)
        assert _op_count(onnx_path, "QuantizeLinear") == 0


class TestFoldConstantReshapeBranches:
    """Guards inside fold_constant_reshape_after_dequant."""

    @use_temporary_directory
    def test_producer_not_dequant(self, tmpdir: str) -> None:
        onnx_path = tmpdir + "/non_dq.onnx"
        const = numpy_helper.from_array(np.array([1.0, 2.0], dtype=np.float32), name="c")
        axes = _i64("ax", [0])
        nodes = [
            helper.make_node("Identity", ["c"], ["f"]),
            helper.make_node("Unsqueeze", ["f", "ax"], ["y"]),
        ]
        _save_onnx(nodes, [const, axes], [], [_fv("y")], onnx_path)
        with _patched_slim(lambda m: m):
            fold_constant_reshape_after_dequant(onnx_path)
        assert _op_count(onnx_path, "Unsqueeze") == 1

    @use_temporary_directory
    def test_dequant_input_not_initializer(self, tmpdir: str) -> None:
        onnx_path = tmpdir + "/dq_in_node.onnx"
        const = numpy_helper.from_array(np.array([1, 2], dtype=np.int8), name="c")
        scale = numpy_helper.from_array(np.float32(0.1), name="s")
        zero_point = numpy_helper.from_array(np.int8(0), name="z")
        axes = _i64("ax", [0])
        nodes = [
            helper.make_node("Identity", ["c"], ["ci"]),
            helper.make_node("DequantizeLinear", ["ci", "s", "z"], ["f"]),
            helper.make_node("Unsqueeze", ["f", "ax"], ["y"]),
        ]
        _save_onnx(nodes, [const, scale, zero_point, axes], [], [_fv("y")], onnx_path)
        with _patched_slim(lambda m: m):
            fold_constant_reshape_after_dequant(onnx_path)
        assert _op_count(onnx_path, "Unsqueeze") == 1

    @use_temporary_directory
    def test_non_scalar_scale_not_folded(self, tmpdir: str) -> None:
        onnx_path = tmpdir + "/nonscalar.onnx"
        const = numpy_helper.from_array(np.array([1, 2], dtype=np.int8), name="c")
        scale = numpy_helper.from_array(np.array([0.1, 0.2], dtype=np.float32), name="s")
        zero_point = numpy_helper.from_array(np.array([0, 0], dtype=np.int8), name="z")
        axes = _i64("ax", [0])
        nodes = [
            helper.make_node("DequantizeLinear", ["c", "s", "z"], ["f"], axis=0),
            helper.make_node("Unsqueeze", ["f", "ax"], ["y"]),
        ]
        _save_onnx(nodes, [const, scale, zero_point, axes], [], [_fv("y")], onnx_path)
        with _patched_slim(lambda m: m):
            fold_constant_reshape_after_dequant(onnx_path)
        assert _op_count(onnx_path, "Unsqueeze") == 1

    @use_temporary_directory
    def test_dynamic_axes_not_folded(self, tmpdir: str) -> None:
        onnx_path = tmpdir + "/dyn_axes.onnx"
        const = numpy_helper.from_array(np.array([1, 2], dtype=np.int8), name="c")
        scale = numpy_helper.from_array(np.float32(0.1), name="s")
        zero_point = numpy_helper.from_array(np.int8(0), name="z")
        nodes = [
            helper.make_node("DequantizeLinear", ["c", "s", "z"], ["f"]),
            helper.make_node("Identity", ["c"], ["axd"]),
            helper.make_node("Unsqueeze", ["f", "axd"], ["y"]),
        ]
        _save_onnx(nodes, [const, scale, zero_point], [], [_fv("y")], onnx_path)
        with _patched_slim(lambda m: m):
            fold_constant_reshape_after_dequant(onnx_path)
        assert _op_count(onnx_path, "Unsqueeze") == 1

    @use_temporary_directory
    def test_initial_slim_failure_aborts(self, tmpdir: str) -> None:
        onnx_path = tmpdir + "/slim_fail.onnx"
        const = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="w")
        _save_onnx([helper.make_node("Identity", ["w"], ["y"])], [const], [], [_fv("y")], onnx_path)

        def boom(_model):
            raise RuntimeError("slim boom")

        with _patched_slim(boom):
            fold_constant_reshape_after_dequant(onnx_path)
        assert _op_count(onnx_path, "Identity") == 1

    @use_temporary_directory
    def test_post_transform_slim_failure_still_saves(self, tmpdir: str) -> None:
        onnx_path = tmpdir + "/slim_fail2.onnx"
        const = numpy_helper.from_array(np.array([1, 2], dtype=np.int8), name="c")
        scale = numpy_helper.from_array(np.float32(0.1), name="s")
        zero_point = numpy_helper.from_array(np.int8(0), name="z")
        axes = _i64("ax", [0])
        # The Unsqueeze output must not be a graph output (that path is guarded and skipped),
        # so route it through an Identity whose output is the graph output.
        nodes = [
            helper.make_node("DequantizeLinear", ["c", "s", "z"], ["f"]),
            helper.make_node("Unsqueeze", ["f", "ax"], ["u"]),
            helper.make_node("Identity", ["u"], ["y"]),
        ]
        _save_onnx(nodes, [const, scale, zero_point, axes], [], [_fv("y", [2, 1])], onnx_path)
        with _patched_slim(_SlimFailsAfterN(1)):
            fold_constant_reshape_after_dequant(onnx_path)
        assert _op_count(onnx_path, "Unsqueeze") == 0
        # Even when the post-fold slim fails, the saved graph must stay valid: the cloned
        # DequantizeLinear nodes are appended after their consumers, so the pass must
        # topologically re-sort before saving (onnx.checker rejects an unsorted graph).
        onnx.checker.check_model(onnx.load(onnx_path))

    @use_temporary_directory
    def test_different_axes_not_shape_aliased(self, tmpdir: str) -> None:
        """One shared int constant feeding two Unsqueeze with DIFFERENT axes must produce
        two differently-shaped reshaped constants, not reuse the first branch's shape.
        """
        onnx_path = tmpdir + "/diff_axes.onnx"
        const = numpy_helper.from_array(np.array([3, 5], dtype=np.int8), name="c")
        scale = numpy_helper.from_array(np.float32(0.5), name="s")
        zero_point = numpy_helper.from_array(np.int8(0), name="z")
        ax1 = _i64("ax1", [0, 2, 3])  # -> (1, 2, 1, 1)
        ax2 = _i64("ax2", [1])  # -> (2, 1)
        nodes = [
            helper.make_node("DequantizeLinear", ["c", "s", "z"], ["dq1"]),
            helper.make_node("DequantizeLinear", ["c", "s", "z"], ["dq2"]),
            helper.make_node("Unsqueeze", ["dq1", "ax1"], ["u1"]),
            helper.make_node("Unsqueeze", ["dq2", "ax2"], ["u2"]),
            helper.make_node("Relu", ["u1"], ["out1"]),
            helper.make_node("Relu", ["u2"], ["out2"]),
        ]
        _save_onnx(nodes, [const, scale, zero_point, ax1, ax2], [], [_fv("out1"), _fv("out2")], onnx_path)

        def shapes(path: str) -> dict[str, tuple[int, ...]]:
            session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
            return {o.name: session.run([o.name], {})[0].shape for o in session.get_outputs()}

        before = shapes(onnx_path)
        with _patched_slim(lambda m: m):
            fold_constant_reshape_after_dequant(onnx_path)
        after = shapes(onnx_path)
        assert after["out1"] == before["out1"], "branch 1 shape must be preserved"
        assert after["out2"] == before["out2"], "branch 2 shape must not alias branch 1"


def _slice_qdq_chain(
    q_scale: float,
    dq_scale: float,
    q_zp: int,
    dq_zp: int,
    axes_a: list[int],
    axes_b: list[int],
    per_channel: bool = False,
):
    """Build Slice(A) -> Q -> DQ -> Slice(B) with configurable Q/DQ scale/zp/axes."""
    if per_channel:
        q_scale_init = numpy_helper.from_array(np.array([q_scale, q_scale], dtype=np.float32), name="qs")
        dq_scale_init = numpy_helper.from_array(np.array([dq_scale, dq_scale], dtype=np.float32), name="ds")
        q_zp_init = numpy_helper.from_array(np.array([q_zp, q_zp], dtype=np.int8), name="qz")
        dq_zp_init = numpy_helper.from_array(np.array([dq_zp, dq_zp], dtype=np.int8), name="dz")
        quant = helper.make_node("QuantizeLinear", ["ao", "qs", "qz"], ["qo"], axis=1)
        dequant = helper.make_node("DequantizeLinear", ["qo", "ds", "dz"], ["do"], axis=1)
    else:
        q_scale_init = numpy_helper.from_array(np.float32(q_scale), name="qs")
        dq_scale_init = numpy_helper.from_array(np.float32(dq_scale), name="ds")
        q_zp_init = numpy_helper.from_array(np.int8(q_zp), name="qz")
        dq_zp_init = numpy_helper.from_array(np.int8(dq_zp), name="dz")
        quant = helper.make_node("QuantizeLinear", ["ao", "qs", "qz"], ["qo"])
        dequant = helper.make_node("DequantizeLinear", ["qo", "ds", "dz"], ["do"])
    # Feed Slice A from an upstream DequantizeLinear whose scalar scale/zp equal the
    # intermediate quantizer's, so the tensor entering Slice A is already on the quant
    # grid. This mirrors real QAT exports (every Slice is fed by a DQ) and satisfies the
    # lossless guard in merge_consecutive_slices.
    in_q_init = numpy_helper.from_array(np.float32(q_scale), name="iqs")
    in_zp_init = numpy_helper.from_array(np.int8(q_zp), name="iqz")
    in_quant = helper.make_node("QuantizeLinear", ["x", "iqs", "iqz"], ["xq"], name="InQ")
    in_dequant = helper.make_node("DequantizeLinear", ["xq", "iqs", "iqz"], ["xdq"], name="InDQ")
    initializers = [
        q_scale_init,
        dq_scale_init,
        q_zp_init,
        dq_zp_init,
        in_q_init,
        in_zp_init,
        _i64("as", [0]),
        _i64("ae", [8]),
        _i64("aa", axes_a),
        _i64("ast", [2]),
        _i64("bs", [0]),
        _i64("be", [8]),
        _i64("ba", axes_b),
        _i64("bst", [2]),
    ]
    nodes = [
        in_quant,
        in_dequant,
        helper.make_node("Slice", ["xdq", "as", "ae", "aa", "ast"], ["ao"], name="SA"),
        quant,
        dequant,
        helper.make_node("Slice", ["do", "bs", "be", "ba", "bst"], ["y"], name="SB"),
    ]
    return nodes, initializers


class TestMergeConsecutiveSlicesBranches:
    """Guards inside merge_consecutive_slices (fan-out, scales, params, axes, slim)."""

    @use_temporary_directory
    def test_unequal_scales_not_merged(self, tmpdir: str) -> None:
        onnx_path = tmpdir + "/unequal_scale.onnx"
        nodes, inits = _slice_qdq_chain(0.5, 0.25, 0, 0, [2], [3])
        _save_onnx(nodes, inits, [_fv("x", [1, 4, 8, 8])], [_fv("y")], onnx_path)
        with _patched_slim(lambda m: m):
            merge_consecutive_slices(onnx_path)
        assert _op_count(onnx_path, "Slice") == 2

    @use_temporary_directory
    def test_unequal_zero_point_not_merged(self, tmpdir: str) -> None:
        onnx_path = tmpdir + "/unequal_zp.onnx"
        nodes, inits = _slice_qdq_chain(0.5, 0.5, 0, 1, [2], [3])
        _save_onnx(nodes, inits, [_fv("x", [1, 4, 8, 8])], [_fv("y")], onnx_path)
        with _patched_slim(lambda m: m):
            merge_consecutive_slices(onnx_path)
        assert _op_count(onnx_path, "Slice") == 2

    @use_temporary_directory
    def test_non_scalar_scale_not_merged(self, tmpdir: str) -> None:
        onnx_path = tmpdir + "/nonscalar_scale.onnx"
        nodes, inits = _slice_qdq_chain(0.5, 0.5, 0, 0, [2], [3], per_channel=True)
        _save_onnx(nodes, inits, [_fv("x", [1, 4, 8, 8])], [_fv("y")], onnx_path)
        with _patched_slim(lambda m: m):
            merge_consecutive_slices(onnx_path)
        assert _op_count(onnx_path, "Slice") == 2

    @use_temporary_directory
    def test_scale_not_initializer_not_merged(self, tmpdir: str) -> None:
        """Q scale produced by a node (not an initializer) blocks the merge."""
        onnx_path = tmpdir + "/scale_node.onnx"
        nodes, inits = _slice_qdq_chain(0.5, 0.5, 0, 0, [2], [3])
        inits = [i for i in inits if i.name != "qs"]
        inits.append(numpy_helper.from_array(np.float32(0.5), name="qs_src"))
        nodes.insert(0, helper.make_node("Identity", ["qs_src"], ["qs"]))
        _save_onnx(nodes, inits, [_fv("x", [1, 4, 8, 8])], [_fv("y")], onnx_path)
        with _patched_slim(lambda m: m):
            merge_consecutive_slices(onnx_path)
        assert _op_count(onnx_path, "Slice") == 2

    @use_temporary_directory
    def test_overlapping_axes_not_merged(self, tmpdir: str) -> None:
        onnx_path = tmpdir + "/overlap.onnx"
        nodes, inits = _slice_qdq_chain(0.5, 0.5, 0, 0, [2], [2])
        _save_onnx(nodes, inits, [_fv("x", [1, 4, 8, 8])], [_fv("y")], onnx_path)
        with _patched_slim(lambda m: m):
            merge_consecutive_slices(onnx_path)
        assert _op_count(onnx_path, "Slice") == 2

    @use_temporary_directory
    def test_dynamic_slice_params_not_merged(self, tmpdir: str) -> None:
        onnx_path = tmpdir + "/dyn_params.onnx"
        nodes, inits = _slice_qdq_chain(0.5, 0.5, 0, 0, [2], [3])
        inits = [i for i in inits if i.name != "as"]
        inits.append(_i64("as_src", [0]))
        nodes.insert(0, helper.make_node("Identity", ["as_src"], ["as"]))
        _save_onnx(nodes, inits, [_fv("x", [1, 4, 8, 8])], [_fv("y")], onnx_path)
        with _patched_slim(lambda m: m):
            merge_consecutive_slices(onnx_path)
        assert _op_count(onnx_path, "Slice") == 2

    @use_temporary_directory
    def test_no_quantize_before_dequant_not_merged(self, tmpdir: str) -> None:
        """Slice(B) whose DequantizeLinear is not fed by a QuantizeLinear is skipped."""
        onnx_path = tmpdir + "/no_q.onnx"
        initializers = [
            numpy_helper.from_array(np.float32(0.5), name="ds"),
            numpy_helper.from_array(np.int8(0), name="dz"),
            _i64("bs", [0]),
            _i64("be", [8]),
            _i64("ba", [3]),
            _i64("bst", [2]),
        ]
        nodes = [
            helper.make_node("Identity", ["x"], ["ii"]),
            helper.make_node("DequantizeLinear", ["ii", "ds", "dz"], ["do"]),
            helper.make_node("Slice", ["do", "bs", "be", "ba", "bst"], ["y"]),
        ]
        _save_onnx(nodes, initializers, [_fv("x", [1, 4, 8, 8])], [_fv("y")], onnx_path)
        with _patched_slim(lambda m: m):
            merge_consecutive_slices(onnx_path)
        assert _op_count(onnx_path, "Slice") == 1

    @use_temporary_directory
    def test_initial_slim_failure_aborts(self, tmpdir: str) -> None:
        onnx_path = tmpdir + "/slim_fail.onnx"
        const = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="w")
        _save_onnx([helper.make_node("Identity", ["w"], ["y"])], [const], [], [_fv("y")], onnx_path)

        def boom(_model):
            raise RuntimeError("slim boom")

        with _patched_slim(boom):
            merge_consecutive_slices(onnx_path)
        assert _op_count(onnx_path, "Identity") == 1

    @use_temporary_directory
    def test_post_transform_slim_failure_still_saves(self, tmpdir: str) -> None:
        onnx_path = tmpdir + "/slim_fail2.onnx"
        nodes, inits = _slice_qdq_chain(0.5, 0.5, 0, 0, [2], [3])
        _save_onnx(nodes, inits, [_fv("x", [1, 4, 8, 8])], [_fv("y")], onnx_path)
        with _patched_slim(_SlimFailsAfterN(1)):
            merge_consecutive_slices(onnx_path)
        assert _op_count(onnx_path, "Slice") == 1

    @use_temporary_directory
    def test_offgrid_input_not_merged_lossless(self, tmpdir: str) -> None:
        """Guard: if the tensor entering Slice A is NOT on the quant grid, the intermediate
        Q/DQ is lossy and must not be dropped. Here Slice A is fed directly by a graph input
        (no upstream DequantizeLinear), so merging would change values. Equal scale/zp alone
        must not be treated as proof of losslessness.
        """
        onnx_path = tmpdir + "/offgrid.onnx"
        # Slice A reads graph input x (off-grid); intermediate Q/DQ share scale/zp.
        inits = [
            numpy_helper.from_array(np.float32(0.5), name="qs"),
            numpy_helper.from_array(np.float32(0.5), name="ds"),
            numpy_helper.from_array(np.int8(0), name="qz"),
            numpy_helper.from_array(np.int8(0), name="dz"),
            _i64("as", [0]),
            _i64("ae", [8]),
            _i64("aa", [2]),
            _i64("ast", [2]),
            _i64("bs", [0]),
            _i64("be", [8]),
            _i64("ba", [3]),
            _i64("bst", [2]),
        ]
        nodes = [
            helper.make_node("Slice", ["x", "as", "ae", "aa", "ast"], ["ao"], name="SA"),
            helper.make_node("QuantizeLinear", ["ao", "qs", "qz"], ["qo"]),
            helper.make_node("DequantizeLinear", ["qo", "ds", "dz"], ["do"]),
            helper.make_node("Slice", ["do", "bs", "be", "ba", "bst"], ["y"], name="SB"),
        ]
        _save_onnx(nodes, inits, [_fv("x", [1, 4, 8, 8])], [_fv("y")], onnx_path)
        sample = np.random.RandomState(0).rand(1, 4, 8, 8).astype(np.float32)

        def run(path: str) -> np.ndarray:
            session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
            return session.run(None, {"x": sample})[0]

        before = run(onnx_path)
        with _patched_slim(lambda m: m):
            merge_consecutive_slices(onnx_path)
        # Off-grid input -> guard blocks the merge, so both Slices (and the Q/DQ) survive
        # and the result is unchanged.
        assert _op_count(onnx_path, "Slice") == 2, "off-grid input must not merge (lossy Q/DQ)"
        np.testing.assert_allclose(before, run(onnx_path), rtol=0, atol=0)


class TestExportOnnxModelOptimizationSlim:
    """onnx.py: the trailing onnxslim failure is swallowed and the model is saved."""

    @use_temporary_directory
    def test_trailing_slim_failure_swallowed(self, tmpdir: str) -> None:
        onnx_path = tmpdir + "/opt.onnx"
        const = numpy_helper.from_array(np.array([1.0], dtype=np.float32), name="w")
        _save_onnx([helper.make_node("Identity", ["w"], ["y"])], [const], [], [_fv("y")], onnx_path)

        original = export_onnx.slim
        export_onnx.slim = lambda _m: (_ for _ in ()).throw(RuntimeError("slim boom"))
        try:
            export_onnx.export_onnx_model_optimization(onnx_path)
        finally:
            export_onnx.slim = original
        # The Identity model is untouched but successfully re-saved.
        assert _op_count(onnx_path, "Identity") == 1


# ===========================================================================
# Branch coverage for the pre-quant FX passes in opt_pass_before_quant.py:
# FoldConstantDivQOPass._trace_const / _trace_const_tuple and
# FoldConvScalarMulQOPass._try_scalar / call() guards, exercised directly on
# hand-built FX nodes (fast, no export, no GPU).
# ===========================================================================


def _empty_gm() -> tuple[Graph, GraphModule]:
    graph = Graph()
    return graph, GraphModule(nn.Module(), graph)


class TestTraceConstBranches:
    """FoldConstantDivQOPass._trace_const covers every node-type / failure branch."""

    def setup_method(self) -> None:
        self.fold = FoldConstantDivQOPass()

    def test_depth_limit(self) -> None:
        _graph, gm = _empty_gm()
        assert self.fold._trace_const(None, gm, depth=11) is None

    def test_int_and_float_literals(self) -> None:
        _graph, gm = _empty_gm()
        assert torch.equal(self.fold._trace_const(5, gm), torch.tensor(5.0))
        assert torch.equal(self.fold._trace_const(2.5, gm), torch.tensor(2.5))

    def test_bool_literal_returns_none(self) -> None:
        _graph, gm = _empty_gm()
        assert self.fold._trace_const(True, gm) is None

    def test_non_node_returns_none(self) -> None:
        _graph, gm = _empty_gm()
        assert self.fold._trace_const([1, 2], gm) is None

    def test_get_attr_buffer_success(self) -> None:
        graph, gm = _empty_gm()
        gm.register_buffer("buf", torch.tensor([2.0, 4.0]))
        node = graph.get_attr("buf")
        assert torch.equal(self.fold._trace_const(node, gm), torch.tensor([2.0, 4.0]))

    def test_get_attr_failure_returns_none(self) -> None:
        graph, gm = _empty_gm()
        node = graph.get_attr("missing")
        assert self.fold._trace_const(node, gm) is None

    def test_passthrough_ops(self) -> None:
        graph, gm = _empty_gm()
        gm.register_buffer("b", torch.tensor([1.0, 2.0, 3.0]))
        attr = graph.get_attr("b")
        node = graph.call_function(torch.ops.aten.contiguous.default, (attr,))
        assert torch.equal(self.fold._trace_const(node, gm), torch.tensor([1.0, 2.0, 3.0]))

    def test_view_of_non_constant(self) -> None:
        graph, gm = _empty_gm()
        placeholder = graph.placeholder("x")
        node = graph.call_function(torch.ops.aten.view.default, (placeholder, [2, 1]))
        assert self.fold._trace_const(node, gm) is None

    def test_view_of_constant(self) -> None:
        graph, gm = _empty_gm()
        gm.register_buffer("b", torch.tensor([1.0, 2.0, 3.0, 4.0]))
        attr = graph.get_attr("b")
        node = graph.call_function(torch.ops.aten.view.default, (attr, [2, 2]))
        assert torch.equal(self.fold._trace_const(node, gm), torch.tensor([[1.0, 2.0], [3.0, 4.0]]))

    def test_unsqueeze_of_constant(self) -> None:
        graph, gm = _empty_gm()
        gm.register_buffer("b", torch.tensor([1.0, 2.0]))
        attr = graph.get_attr("b")
        node = graph.call_function(torch.ops.aten.unsqueeze.default, (attr, 0))
        assert torch.equal(self.fold._trace_const(node, gm), torch.tensor([[1.0, 2.0]]))

    def test_linspace(self) -> None:
        graph, gm = _empty_gm()
        node = graph.call_function(torch.ops.aten.linspace.default, (0.0, 1.0, 3))
        assert torch.equal(self.fold._trace_const(node, gm), torch.linspace(0.0, 1.0, 3))

    def test_stack_with_non_constant(self) -> None:
        graph, gm = _empty_gm()
        placeholder = graph.placeholder("x")
        node = graph.call_function(torch.ops.aten.stack.default, ([placeholder],))
        assert self.fold._trace_const(node, gm) is None

    def test_stack_of_constants(self) -> None:
        graph, gm = _empty_gm()
        gm.register_buffer("b", torch.tensor([1.0, 2.0]))
        attr = graph.get_attr("b")
        node = graph.call_function(torch.ops.aten.stack.default, ([attr, attr], 0))
        assert torch.equal(self.fold._trace_const(node, gm), torch.stack([torch.tensor([1.0, 2.0])] * 2, 0))

    def test_getitem_with_non_constant_container(self) -> None:
        graph, gm = _empty_gm()
        placeholder = graph.placeholder("x")
        meshgrid = graph.call_function(torch.ops.aten.meshgrid.indexing, ([placeholder],))
        node = graph.call_function(operator.getitem, (meshgrid, 0))
        assert self.fold._trace_const(node, gm) is None

    def test_div_with_non_constant(self) -> None:
        graph, gm = _empty_gm()
        placeholder = graph.placeholder("x")
        node = graph.call_function(torch.ops.aten.div.Tensor, (placeholder, 2.0))
        assert self.fold._trace_const(node, gm) is None

    def test_mul_with_non_constant(self) -> None:
        graph, gm = _empty_gm()
        placeholder = graph.placeholder("x")
        node = graph.call_function(torch.ops.aten.mul.Tensor, (placeholder, 2.0))
        assert self.fold._trace_const(node, gm) is None

    def test_mul_of_constants(self) -> None:
        graph, gm = _empty_gm()
        gm.register_buffer("a", torch.tensor([2.0, 3.0]))
        gm.register_buffer("b", torch.tensor([4.0, 5.0]))
        node = graph.call_function(torch.ops.aten.mul.Tensor, (graph.get_attr("a"), graph.get_attr("b")))
        assert torch.equal(self.fold._trace_const(node, gm), torch.tensor([8.0, 15.0]))

    def test_unknown_call_function(self) -> None:
        graph, gm = _empty_gm()
        placeholder = graph.placeholder("x")
        node = graph.call_function(torch.ops.aten.relu.default, (placeholder,))
        assert self.fold._trace_const(node, gm) is None


class TestTraceConstTupleBranches:
    """FoldConstantDivQOPass._trace_const_tuple guards."""

    def setup_method(self) -> None:
        self.fold = FoldConstantDivQOPass()

    def test_depth_or_non_node(self) -> None:
        _graph, gm = _empty_gm()
        assert self.fold._trace_const_tuple(None, gm, depth=11) is None

    def test_non_meshgrid_call(self) -> None:
        graph, gm = _empty_gm()
        placeholder = graph.placeholder("x")
        node = graph.call_function(torch.ops.aten.relu.default, (placeholder,))
        assert self.fold._trace_const_tuple(node, gm) is None

    def test_meshgrid_with_non_constant(self) -> None:
        graph, gm = _empty_gm()
        placeholder = graph.placeholder("x")
        node = graph.call_function(torch.ops.aten.meshgrid.indexing, ([placeholder],))
        assert self.fold._trace_const_tuple(node, gm) is None


class TestConvScalarMulQOPassBranches:
    """FoldConvScalarMulQOPass.call()/_try_scalar guards on hand-built graphs."""

    def setup_method(self) -> None:
        self.pass_ = FoldConvScalarMulQOPass()

    def _run(self, gm: GraphModule) -> GraphModule:
        gm.recompile()
        return self.pass_.call(gm)

    @staticmethod
    def _has_mul(gm: GraphModule) -> bool:
        return any(
            n.op == "call_function" and n.target in (torch.ops.aten.mul.Tensor, torch.ops.aten.mul.Scalar)
            for n in gm.graph.nodes
        )

    def test_mul_with_wrong_arity_skipped(self) -> None:
        graph, gm = _empty_gm()
        placeholder = graph.placeholder("x")
        mul = graph.call_function(torch.ops.aten.mul.Tensor, (placeholder,))
        mul.meta = {}
        graph.output(mul)
        self._run(gm)
        assert self._has_mul(gm)

    def test_scalar_one_not_folded(self) -> None:
        graph, gm = _empty_gm()
        placeholder = graph.placeholder("x")
        mul = graph.call_function(torch.ops.aten.mul.Tensor, (placeholder, 1.0))
        mul.meta = {}
        graph.output(mul)
        self._run(gm)
        assert self._has_mul(gm)

    def test_bad_get_attr_scalar_skipped(self) -> None:
        graph, gm = _empty_gm()
        placeholder = graph.placeholder("x")
        bad_attr = graph.get_attr("missing")
        mul = graph.call_function(torch.ops.aten.mul.Tensor, (placeholder, bad_attr))
        mul.meta = {}
        graph.output(mul)
        self._run(gm)
        assert self._has_mul(gm)

    def test_non_module_operand_not_folded(self) -> None:
        """A scalar Mul on a non-call_module operand is left alone."""
        graph, gm = _empty_gm()
        placeholder = graph.placeholder("x")
        relu = graph.call_function(torch.ops.aten.relu.default, (placeholder,))
        mul = graph.call_function(torch.ops.aten.mul.Tensor, (relu, 2.0))
        mul.meta = {}
        graph.output(mul)
        self._run(gm)
        assert self._has_mul(gm)

    def test_scalar_first_arg_and_multiuser_conv_not_folded(self) -> None:
        """scalar * conv (scalar as arg0) with a Conv that has >1 user: the extra
        user guard blocks folding, and the scalar-as-arg0 branch is exercised."""
        graph, gm = _empty_gm()
        gm.add_module("conv", QuantConv2d(2, 2, 1))
        placeholder = graph.placeholder("x")
        conv = graph.call_module("conv", (placeholder,))
        mul = graph.call_function(torch.ops.aten.mul.Tensor, (2.0, conv))
        extra = graph.call_function(torch.ops.aten.relu.default, (conv,))
        graph.output((mul, extra))
        self._run(gm)
        assert self._has_mul(gm)


# ===========================================================================
# Branch coverage for processor_utils.py qspec-map helpers and the annotator
# lines that call them (PRelu slope / HardSigmoid input), on FX graphs.
# ===========================================================================

_ANNOTATE_SPEC = QTensorConfig(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=PerTensorMinMaxObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)
_ANNOTATE_CFG = QLayerConfig(
    input_tensors=_ANNOTATE_SPEC,
    output_tensors=_ANNOTATE_SPEC,
    weight=_ANNOTATE_SPEC,
    bias=_ANNOTATE_SPEC,
)


class _CallFunctionConvPRelu(nn.Module):
    """Conv (kept as an aten call_function) directly followed by PReLU."""

    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 2, 1)
        self.act = nn.PReLU(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class _StandalonePRelu(nn.Module):
    """PReLU whose parent is an Add (not a Conv), so it hits the activation_op annotator."""

    def __init__(self) -> None:
        super().__init__()
        self.act = nn.PReLU(4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + 1.0)


class TestPReluSlopeQspecMap:
    """_prelu_slope_qspec_map: non-PRelu, missing weight qspec, and slope-not-Node."""

    def test_non_prelu_returns_empty(self) -> None:
        gm = export_for_training(_StandalonePRelu().eval(), (torch.rand(1, 4, 4, 4),)).module()
        add_nodes = [n for n in gm.graph.nodes if n.op == "call_function" and "add" in str(n.target)]
        assert add_nodes
        assert processor_utils._prelu_slope_qspec_map(add_nodes[0], _ANNOTATE_CFG) == {}

    def test_prelu_without_weight_config_returns_empty(self) -> None:
        gm = export_for_training(_StandalonePRelu().eval(), (torch.rand(1, 4, 4, 4),)).module()
        prelu_nodes = [n for n in gm.graph.nodes if processor_utils.is_prelu_node(n)]
        assert prelu_nodes
        # QLayerConfig() has no weight qspec -> empty map (line 312 branch).
        assert processor_utils._prelu_slope_qspec_map(prelu_nodes[0], QLayerConfig()) == {}


class TestHardsigmoidInputQspecMap:
    """_hardsigmoid_input_qspec_map: non-HardSigmoid and missing-input-config early exits."""

    def test_non_hardsigmoid_returns_empty(self) -> None:
        gm = export_for_training(_StandalonePRelu().eval(), (torch.rand(1, 4, 4, 4),)).module()
        prelu_nodes = [n for n in gm.graph.nodes if processor_utils.is_prelu_node(n)]
        assert prelu_nodes
        result = processor_utils._hardsigmoid_input_qspec_map(prelu_nodes[0], prelu_nodes[0], _ANNOTATE_CFG)
        assert result == {}

    def test_hardsigmoid_without_input_config_returns_empty(self) -> None:
        class _Hardsigmoid(nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.nn.functional.hardsigmoid(x)

        gm = export_for_training(_Hardsigmoid().eval(), (torch.rand(1, 4),)).module()
        hs_nodes = [n for n in gm.graph.nodes if processor_utils.is_hardsigmoid_node(n)]
        assert hs_nodes
        input_node = hs_nodes[0].args[0]
        assert processor_utils._hardsigmoid_input_qspec_map(hs_nodes[0], input_node, QLayerConfig()) == {}


class TestConvActAndActivationAnnotators:
    """Exercise the qspec-map calls inside _annotate_conv_act (588-589) and
    _annotate_activation (747) via the registered annotators."""

    def test_conv_act_annotates_prelu_slope(self) -> None:
        annotator = processor_utils.OP_TO_ANNOTATOR["convlike_act"]
        gm = export_for_training(_CallFunctionConvPRelu().eval(), (torch.rand(1, 2, 4, 4),)).module()
        partitions = annotator(gm, _ANNOTATE_CFG)
        assert partitions
        prelu_nodes = [
            n for n in gm.graph.nodes if processor_utils.is_prelu_node(n) and "quantization_annotation" in n.meta
        ]
        assert prelu_nodes
        # slope node registered in the PRelu activation's input_qspec_map.
        assert len(prelu_nodes[0].meta["quantization_annotation"].input_qspec_map) >= 1

    def test_activation_op_annotates_standalone_prelu_slope(self) -> None:
        annotator = processor_utils.OP_TO_ANNOTATOR["activation_op"]
        gm = export_for_training(_StandalonePRelu().eval(), (torch.rand(1, 4, 4, 4),)).module()
        partitions = annotator(gm, _ANNOTATE_CFG)
        assert partitions
        prelu_nodes = [
            n for n in gm.graph.nodes if processor_utils.is_prelu_node(n) and "quantization_annotation" in n.meta
        ]
        assert prelu_nodes
        qspec_map = prelu_nodes[0].meta["quantization_annotation"].input_qspec_map
        assert len(qspec_map) >= 1


def _build_shared_slope_unsqueeze_model(onnx_path: str, num_prelu: int = 3) -> None:
    """Hand-build one int8 slope constant shared by ``num_prelu`` PRelu, each via its own
    ``DequantizeLinear -> Unsqueeze -> PRelu``. This is what onnxslim leaves when many PRelu
    share a slope. fold_constant_reshape_after_dequant folds every branch into a private cloned
    ``reshaped_int -> DequantizeLinear`` (never mutating a shared node), then
    merge_equivalent_constant_dequantizers collapses those identical clones back onto one shared
    DequantizeLinear, matching PTQ.
    """
    channels = 4
    slope_int8 = np.array([3, -5, 7, -9], dtype=np.int8)
    initializers = [
        numpy_helper.from_array(slope_int8, name="slope_int8"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="slope_scale"),
        numpy_helper.from_array(np.array(0, dtype=np.int8), name="slope_zp"),
        numpy_helper.from_array(np.array([0, 2, 3], dtype=np.int64), name="unsqueeze_axes"),
    ]
    nodes = []
    inputs = []
    outputs = []
    for i in range(num_prelu):
        nodes.append(
            helper.make_node(
                "DequantizeLinear", ["slope_int8", "slope_scale", "slope_zp"], [f"slope_f{i}"], name=f"DQ{i}"
            )
        )
        nodes.append(helper.make_node("Unsqueeze", [f"slope_f{i}", "unsqueeze_axes"], [f"slope_4d{i}"], name=f"UQ{i}"))
        nodes.append(helper.make_node("PRelu", [f"x{i}", f"slope_4d{i}"], [f"y{i}"], name=f"PRelu{i}"))
        inputs.append(helper.make_tensor_value_info(f"x{i}", TensorProto.FLOAT, [1, channels, 4, 4]))
        outputs.append(helper.make_tensor_value_info(f"y{i}", TensorProto.FLOAT, [1, channels, 4, 4]))
    graph = helper.make_graph(nodes, "shared_slope", inputs, outputs, initializer=initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 10  # match the onnxruntime build used in CI
    onnx.save(model, onnx_path)


class TestSharedPReluSlopeFoldAndMerge:
    """fold_constant_reshape_after_dequant clones a private DequantizeLinear per shared-slope
    branch; merge_equivalent_constant_dequantizers collapses the identical clones back onto one,
    matching PTQ (single shared slope DQ, no Unsqueeze)."""

    @use_temporary_directory
    def test_clone_fold_then_merge_to_single_dq(self, tmpdir: str) -> None:
        torch.cuda.empty_cache()
        onnx_path = tmpdir + "/shared_slope.onnx"
        _build_shared_slope_unsqueeze_model(onnx_path, num_prelu=3)

        # The reshape fold removes every Unsqueeze and clones a private DQ per branch, so no
        # shared DequantizeLinear is mutated. Result: 0 Unsqueeze, 3 cloned slope DQ.
        fold_constant_reshape_after_dequant(onnx_path)
        assert onnx_contains_op_num(onnx_path, "Unsqueeze") == 0, "every PRelu-slope Unsqueeze should be folded"
        assert onnx_contains_op_num(onnx_path, "DequantizeLinear") == 3, "each branch should get its own cloned DQ"

        # The merge collapses the 3 identical slope DQ onto one shared DQ, matching PTQ.
        merge_equivalent_constant_dequantizers(onnx_path)
        assert onnx_contains_op_num(onnx_path, "DequantizeLinear") == 1, (
            "all PRelu should share a single slope DQ, like PTQ"
        )
        torch.cuda.empty_cache()

    @use_temporary_directory
    def test_fold_and_merge_numerically_correct(self, tmpdir: str) -> None:
        # Validate the folded+merged graph against a numpy PRelu reference: the slope dequantizes
        # to [3,-5,7,-9]*0.5 broadcast across the channel axis.
        torch.cuda.empty_cache()
        onnx_path = tmpdir + "/shared_slope.onnx"
        _build_shared_slope_unsqueeze_model(onnx_path, num_prelu=3)
        fold_constant_reshape_after_dequant(onnx_path)
        merge_equivalent_constant_dequantizers(onnx_path)

        slope = (np.array([3, -5, 7, -9], dtype=np.float32) * 0.5).reshape(1, 4, 1, 1)
        rng = np.random.RandomState(0)
        feeds = {f"x{i}": rng.rand(1, 4, 4, 4).astype(np.float32) for i in range(3)}

        session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        outputs = session.run(None, feeds)
        for i, produced in enumerate(outputs):
            x = feeds[f"x{i}"]
            expected = np.where(x >= 0, x, x * slope)
            np.testing.assert_allclose(produced, expected, rtol=0, atol=1e-6)
        torch.cuda.empty_cache()


class TestFoldOwnershipGuards:
    """The folds must not corrupt a graph where a QuantizeLinear output is also a graph output,
    or where a DequantizeLinear is shared by a non-fold consumer (ownership guards)."""

    @use_temporary_directory
    def test_weight_fold_keeps_quantizelinear_that_is_graph_output(self, tmpdir: str) -> None:
        # float_const -> Q -> {DQ, graph output}. Removing Q would invalidate the graph output,
        # so fold_quantizers_for_weight must skip this QuantizeLinear.
        onnx_path = tmpdir + "/q_is_graph_output.onnx"
        inits = [
            numpy_helper.from_array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), name="w"),
            numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="q_scale"),
            numpy_helper.from_array(np.array(0, dtype=np.int8), name="q_zp"),
        ]
        nodes = [
            helper.make_node("QuantizeLinear", ["w", "q_scale", "q_zp"], ["w_q"], name="Q"),
            helper.make_node("DequantizeLinear", ["w_q", "q_scale", "q_zp"], ["w_dq"], name="DQ"),
        ]
        # w_q is both consumed by DQ and exposed as a graph output.
        outputs = [
            helper.make_tensor_value_info("w_dq", TensorProto.FLOAT, [4]),
            helper.make_tensor_value_info("w_q", TensorProto.INT8, [4]),
        ]
        graph = helper.make_graph(nodes, "q_go", [], outputs, initializer=inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 10
        onnx.save(model, onnx_path)

        fold_quantizers_for_weight(onnx_path)
        assert onnx_contains_op_num(onnx_path, "QuantizeLinear") == 1, (
            "QuantizeLinear whose output is a graph output must not be folded away"
        )

    @use_temporary_directory
    def test_reshape_fold_does_not_disturb_shared_dq_sibling(self, tmpdir: str) -> None:
        # int_const -> DQ -> {Unsqueeze -> PRelu, Add}. Folding the Unsqueeze branch must clone a
        # private DQ and leave the original DQ (and its Add sibling) intact.
        onnx_path = tmpdir + "/shared_dq_sibling.onnx"
        inits = [
            numpy_helper.from_array(np.array([3, -5, 7, -9], dtype=np.int8), name="slope_int8"),
            numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="slope_scale"),
            numpy_helper.from_array(np.array(0, dtype=np.int8), name="slope_zp"),
            numpy_helper.from_array(np.array([0, 2, 3], dtype=np.int64), name="uq_axes"),
        ]
        nodes = [
            helper.make_node("DequantizeLinear", ["slope_int8", "slope_scale", "slope_zp"], ["slope_f"], name="DQ"),
            helper.make_node("Unsqueeze", ["slope_f", "uq_axes"], ["slope_4d"], name="UQ"),
            helper.make_node("PRelu", ["x", "slope_4d"], ["y"], name="PRelu"),
            # Sibling consumer of the same DQ output (1-D add), must stay valid.
            helper.make_node("Add", ["z", "slope_f"], ["w"], name="Add"),
        ]
        inputs = [
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 4, 4]),
            helper.make_tensor_value_info("z", TensorProto.FLOAT, [4]),
        ]
        outputs = [
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 4, 4]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [4]),
        ]
        graph = helper.make_graph(nodes, "shared_dq", inputs, outputs, initializer=inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 10
        onnx.save(model, onnx_path)

        fold_constant_reshape_after_dequant(onnx_path)
        graph_after = onnx.load(onnx_path).graph
        producer = {o: n for n in graph_after.node for o in n.output}
        # The Add sibling must still read a valid DequantizeLinear output (original DQ intact).
        add_node = next(n for n in graph_after.node if n.op_type == "Add")
        assert add_node.input[1] in producer and producer[add_node.input[1]].op_type == "DequantizeLinear", (
            "the shared DequantizeLinear feeding the Add sibling must be left intact"
        )
        assert onnx_contains_op_num(onnx_path, "Unsqueeze") == 0, "the private PRelu-slope Unsqueeze should be folded"

    @use_temporary_directory
    def test_merge_keeps_dequant_that_is_graph_output(self, tmpdir: str) -> None:
        # Two equivalent slope DQ where the second is also a graph output. Merging must not
        # remove the graph-output DQ, or the graph output would become dangling.
        onnx_path = tmpdir + "/merge_graph_output.onnx"
        inits = [
            numpy_helper.from_array(np.array([3, -5, 7, -9], dtype=np.int8).reshape(1, 4, 1, 1), name="slope_q"),
            numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="slope_scale"),
            numpy_helper.from_array(np.array(0, dtype=np.int8), name="slope_zp"),
        ]
        nodes = [
            helper.make_node("DequantizeLinear", ["slope_q", "slope_scale", "slope_zp"], ["d1"], name="DQ1"),
            helper.make_node("PRelu", ["x1", "d1"], ["y1"], name="P1"),
            helper.make_node("DequantizeLinear", ["slope_q", "slope_scale", "slope_zp"], ["d2"], name="DQ2"),
            helper.make_node("PRelu", ["x2", "d2"], ["y2"], name="P2"),
        ]
        inputs = [
            helper.make_tensor_value_info("x1", TensorProto.FLOAT, [1, 4, 4, 4]),
            helper.make_tensor_value_info("x2", TensorProto.FLOAT, [1, 4, 4, 4]),
        ]
        # d2 is both a PRelu slope and a graph output.
        outputs = [
            helper.make_tensor_value_info("y1", TensorProto.FLOAT, [1, 4, 4, 4]),
            helper.make_tensor_value_info("y2", TensorProto.FLOAT, [1, 4, 4, 4]),
            helper.make_tensor_value_info("d2", TensorProto.FLOAT, [1, 4, 1, 1]),
        ]
        graph = helper.make_graph(nodes, "merge_go", inputs, outputs, initializer=inits)
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
        model.ir_version = 10
        onnx.save(model, onnx_path)

        merge_equivalent_constant_dequantizers(onnx_path)
        graph_after = onnx.load(onnx_path).graph
        produced = {out for node in graph_after.node for out in node.output}
        assert "d2" in produced, "a DequantizeLinear whose output is a graph output must not be merged away"
        # The graph must remain loadable by onnxruntime (no dangling output).
        onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
