#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Tests for the onnx_xint8_adjust Shapeshifter pass.

This module tests the XINT8 quantize-position adjustment pass by:
- Exporting a Torch model (Conv with bias, Add, HardSigmoid, Swish) to float ONNX.
- Quantizing with XINT8 (power-of-two scale) and running shape inference for a valid graph.
- Optionally perturbing Q/DQ scales so that adjust_shift_read, adjust_shift_write,
  adjust_shift_cut, adjust_shift_bias, adjust_hard_sigmoid, and adjust_shift_swish
  are triggered.
- Running the pass via the shapeshifter CLI and asserting log messages and output
  model validity (with fallback to loadable model + expected ops when the ONNX checker fails).
"""

import contextlib
import unittest
from pathlib import Path

import numpy as np
import onnx
import onnx.numpy_helper
import torch
import torch.nn as nn
import yaml
from onnxruntime.quantization import CalibrationDataReader

from quark.common.utils.testing_utils import use_temporary_directory
from quark.experimental.cli.main import main as cli
from quark.onnx import ModelQuantizer, QConfig, QLayerConfig, XInt8Spec
from quark.onnx.quantization.quant_utils import pos2scale

np.random.seed(123456)
# Single batch used for XINT8 calibration; shape (1, 3, 16, 16) matches model input.
CALIBRATION_INPUT = np.random.randn(1, 3, 16, 16).astype(np.float32) * 0.1

# Alias for perturbing Q/DQ scales via position (pos2scale) to trigger pass adjustments.
POS2SCALE = pos2scale


class DataReader(CalibrationDataReader):
    """Single-batch calibration data reader for ONNX static quantization (XINT8)."""

    def __init__(self, input_tensor: np.ndarray, input_name: str = "input"):
        """Initialize the calibration data reader.

        Args:
            input_tensor: NumPy array containing the calibration input data.
            input_name: Name of the input tensor (default: "input").
        """
        self.data = [input_tensor]
        self.input_name = input_name
        self._index = 0

    def get_next(self):
        """Return the next batch of calibration data.

        Returns:
            dict: A dictionary mapping input name to the next data batch, or None if all data has been consumed.
        """
        if self._index < len(self.data):
            self._index += 1
            return {self.input_name: self.data[self._index - 1]}
        return None

    def rewind(self):
        """Reset the calibration data reader to the beginning."""
        self._index = 0


class XInt8AdjustOpsModel(nn.Module):
    """
    Model that can trigger onnx_xint8_adjust sub-routines:
    - Conv (with bias) -> shift_cut, shift_bias
    - Add (two branches) -> shift_read, shift_write
    - HardSigmoid (alpha=1/6) -> adjust_hard_sigmoid
    - Swish (x * hardsigmoid(x)) -> adjust_shift_swish
    """

    def __init__(self):
        """Initialize the XInt8AdjustOpsModel with convolutional and activation layers.

        This model architecture consists of:
        - Two parallel Conv2d layers (conv1, conv2) with 3 input channels, 8 output channels,
          kernel size 3, padding 1, and bias enabled
        - A HardSigmoid activation layer used in the Swish computation
        """
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(3, 8, 3, padding=1, bias=True)
        self.hardsigmoid = nn.Hardsigmoid()

    def forward(self, x):
        """Forward pass computation for the XInt8AdjustOpsModel.
        Args:
            x: Input tensor of shape (batch_size, 3, height, width).
        Returns:
            Output tensor after applying two parallel Conv2d branches, adding them,
            and computing Swish activation (add_out * hardsigmoid(add_out)).
        """
        a = self.conv1(x)
        b = self.conv2(x)
        add_out = a + b
        # Swish: x * hardsigmoid(x)
        hs_out = self.hardsigmoid(add_out)
        out = add_out * hs_out
        return out


class ConcatModel(nn.Module):
    """Model with Concat operation to test align_concat."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(3, 4, 3, padding=1, bias=False)

    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        return torch.cat([a, b], dim=1)


class PoolModel(nn.Module):
    """Model with AveragePool operation to test align_pool.

    Note: Uses AveragePool because _get_ipos_name has special handling for avg_pool_op_type.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        return x


class PadModel(nn.Module):
    """Model with Pad operation to test align_pad.

    Note: Uses ReplicationPad2d which exports as ONNX Pad with 'edge' mode.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.pad = nn.ReplicationPad2d((2, 2, 2, 2))
        self.conv2 = nn.Conv2d(8, 8, 3, padding=0, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pad(x)
        x = self.conv2(x)
        return x


class SliceModel(nn.Module):
    """Model with Slice operation to test align_slice."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(8, 8, 3, padding=1, bias=False)

    def forward(self, x):
        x = self.conv1(x)
        x = x[:, :, 2:14, 2:14]
        x = self.conv2(x)
        return x


def export_float_onnx(output_dir: str, model_class=None, model_name: str = "xint8_adjust_ops_float") -> str:
    """Export model to float ONNX. Returns path to the saved model."""
    torch.manual_seed(42)
    if model_class is None:
        model_class = XInt8AdjustOpsModel
    model = model_class()
    model.eval()
    out_path = Path(output_dir, f"{model_name}.onnx").as_posix()
    dummy_input = torch.randn(1, 3, 16, 16)
    torch.onnx.export(
        model,
        dummy_input,
        out_path,
        input_names=["input"],
        output_names=["output"],
        keep_initializers_as_inputs=False,
        do_constant_folding=False,
        opset_version=17,
        dynamo=False,
    )
    return out_path


def quantize_model_xint8(
    output_dir: str, calibration_data: np.ndarray, model_class=None, model_name: str = "xint8_adjust_ops"
) -> str:
    """Quantize the float ONNX model with XINT8 (power-of-two scale). Returns path to the quantized model."""
    float_model_path = export_float_onnx(output_dir, model_class, f"{model_name}_float")
    quant_model_path = Path(output_dir, f"{model_name}_quant.onnx").as_posix()
    data_reader = DataReader(calibration_data)
    qconfig = QConfig(global_config=QLayerConfig(activation=XInt8Spec(), weight=XInt8Spec()))
    quantizer = ModelQuantizer(qconfig)
    quantizer.quantize_model(float_model_path, quant_model_path, data_reader)
    return quant_model_path


def ensure_model_valid_with_shape_inference(model_path: str, output_path: str | None = None) -> str:
    """Run shape inference on the model and save. Returns path to the saved model (output_path or model_path)."""
    model = onnx.load(model_path)
    with contextlib.suppress(Exception):
        model = onnx.shape_inference.infer_shapes(model)
    out = output_path or model_path
    onnx.save(model, out)
    return out


def _build_output_to_node(model: onnx.ModelProto) -> dict[str, onnx.NodeProto]:
    """Build a map from tensor name to the node that produces it (for Q/DQ scale lookup)."""
    out_to_node = {}
    for node in model.graph.node:
        for out in node.output:
            if out:
                out_to_node[out] = node
    return out_to_node


def _get_scale_initializer_name_for_tensor(model: onnx.ModelProto, tensor_name: str) -> str | None:
    """Return the scale initializer name (input[1]) of the Q/DQ node that produces tensor_name."""
    out_to_node = _build_output_to_node(model)
    if tensor_name not in out_to_node:
        return None
    node = out_to_node[tensor_name]
    if node.op_type not in ("QuantizeLinear", "DequantizeLinear") or len(node.input) < 2:
        return None
    return node.input[1]


def _get_scale_initializer_name_for_consumer(model: onnx.ModelProto, tensor_name: str) -> str | None:
    """Return the scale initializer name (input[1]) of the Q/DQ node that consumes tensor_name (input[0])."""
    for node in model.graph.node:
        if (
            len(node.input) >= 2
            and node.input[0] == tensor_name
            and node.op_type in ("QuantizeLinear", "DequantizeLinear")
        ):
            return node.input[1]
    return None


def perturb_scales_to_trigger_align_concat(model_path: str, output_path: str) -> None:
    """Modify Q/DQ scales so that align_concat is triggered by setting different pos for Concat inputs/output."""
    model = onnx.load(model_path)
    name_to_init = {init.name: init for init in model.graph.initializer}

    def set_scale_by_name(scale_init_name: str, new_scale: float) -> None:
        if scale_init_name not in name_to_init:
            return
        init = name_to_init[scale_init_name]
        arr = onnx.numpy_helper.to_array(init)
        new_val = np.array(new_scale, dtype=arr.dtype)
        new_tensor = onnx.numpy_helper.from_array(new_val, name=init.name)
        for i, inits in enumerate(model.graph.initializer):
            if inits.name == scale_init_name:
                model.graph.initializer[i].CopyFrom(new_tensor)
                return

    for node in model.graph.node:
        if node.op_type == "Concat" and len(node.input) >= 2:
            in0, in1 = node.input[0], node.input[1]
            ipos0_node_name = _find_qdq_node_name_for_tensor(model, in0)
            ipos1_node_name = _find_qdq_node_name_for_tensor(model, in1)
            if ipos0_node_name:
                scale0 = _get_scale_init_name_by_node_name(model, ipos0_node_name)
                if scale0:
                    set_scale_by_name(scale0, POS2SCALE(5))
            if ipos1_node_name:
                scale1 = _get_scale_init_name_by_node_name(model, ipos1_node_name)
                if scale1:
                    set_scale_by_name(scale1, POS2SCALE(10))

            concat_out = node.output[0] if node.output else None
            if concat_out:
                opos_node_name = _find_qdq_node_name_for_consumer(model, concat_out)
                if opos_node_name:
                    scale_out = _get_scale_init_name_by_node_name(model, opos_node_name)
                    if scale_out:
                        set_scale_by_name(scale_out, POS2SCALE(8))
            break
    onnx.save(model, output_path)


REFINE_OP_TYPES = ["QuantizeLinear", "DequantizeLinear", "ExtendedQuantizeLinear", "ExtendedDequantizeLinear"]


def _find_qdq_node_name_for_tensor(model: onnx.ModelProto, tensor_name: str) -> str | None:
    """Find the name of the Q/DQ node whose output is the given tensor (mirrors QuantPosManager._find_node_name)."""
    for node in model.graph.node:
        if len(node.output) > 0 and node.output[0] == tensor_name and node.op_type in REFINE_OP_TYPES:
            return node.name
    return None


def _find_qdq_node_name_for_consumer(model: onnx.ModelProto, tensor_name: str) -> str | None:
    """Find the name of the Q/DQ node that consumes the given tensor (mirrors QuantPosManager._find_o_name)."""
    for node in model.graph.node:
        if len(node.input) >= 1 and node.input[0] == tensor_name and node.op_type in REFINE_OP_TYPES:
            return node.name
    return None


def _get_scale_init_name_by_node_name(model: onnx.ModelProto, node_name: str) -> str | None:
    """Get the scale initializer name for a Q/DQ node by its node name."""
    for node in model.graph.node:
        if node.name == node_name and node.op_type in REFINE_OP_TYPES and len(node.input) >= 2:
            return node.input[1]
    return None


def _find_upstream_qdq_scale(model: onnx.ModelProto, tensor_name: str) -> str | None:
    """Find the scale initializer of a Q/DQ node upstream from the given tensor."""
    out_to_node = _build_output_to_node(model)
    visited = set()
    queue = [tensor_name]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        if current not in out_to_node:
            continue
        node = out_to_node[current]
        if node.op_type in REFINE_OP_TYPES and len(node.input) >= 2:
            return node.input[1]
        for inp in node.input[:1]:
            if inp and inp not in visited:
                queue.append(inp)
    return None


def _find_downstream_qdq_scale(model: onnx.ModelProto, tensor_name: str) -> str | None:
    """Find the scale initializer of a Q/DQ node downstream from the given tensor."""
    visited = set()
    queue = [tensor_name]
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        for node in model.graph.node:
            if current in node.input:
                if node.op_type in REFINE_OP_TYPES and len(node.input) >= 2:
                    return node.input[1]
                for out in node.output:
                    if out and out not in visited:
                        queue.append(out)
                break
    return None


def perturb_scales_to_trigger_align_pool(model_path: str, output_path: str) -> None:
    """Modify Q/DQ scales so that align_pool is triggered by setting different pos for Pool input/output."""
    model = onnx.load(model_path)
    name_to_init = {init.name: init for init in model.graph.initializer}

    def set_scale_by_name(scale_init_name: str, new_scale: float) -> None:
        if scale_init_name not in name_to_init:
            return
        init = name_to_init[scale_init_name]
        arr = onnx.numpy_helper.to_array(init)
        new_val = np.array(new_scale, dtype=arr.dtype)
        new_tensor = onnx.numpy_helper.from_array(new_val, name=init.name)
        for i, inits in enumerate(model.graph.initializer):
            if inits.name == scale_init_name:
                model.graph.initializer[i].CopyFrom(new_tensor)
                return

    for node in model.graph.node:
        if node.op_type in ["AveragePool", "GlobalAveragePool"] and node.input:
            ipos_node_name = _find_qdq_node_name_for_tensor(model, node.input[0])
            if ipos_node_name:
                scale_in = _get_scale_init_name_by_node_name(model, ipos_node_name)
                if scale_in:
                    set_scale_by_name(scale_in, POS2SCALE(5))

            pool_out = node.output[0] if node.output else None
            if pool_out:
                opos_node_name = _find_qdq_node_name_for_consumer(model, pool_out)
                if opos_node_name:
                    scale_out = _get_scale_init_name_by_node_name(model, opos_node_name)
                    if scale_out:
                        set_scale_by_name(scale_out, POS2SCALE(10))
            break
    onnx.save(model, output_path)


def _ensure_qdq_around_node_with_different_scales(
    model: onnx.ModelProto, target_op_type: str, ipos: int, opos: int
) -> onnx.ModelProto:
    """Ensure Q/DQ nodes exist around target node with different scales to trigger alignment."""
    from onnx import TensorProto, helper

    name_to_init = {init.name: init for init in model.graph.initializer}

    def set_scale(scale_name: str, new_scale: float) -> None:
        if scale_name in name_to_init:
            init = name_to_init[scale_name]
            arr = onnx.numpy_helper.to_array(init)
            new_val = np.array(new_scale, dtype=arr.dtype)
            new_tensor = onnx.numpy_helper.from_array(new_val, name=init.name)
            for i, inits in enumerate(model.graph.initializer):
                if inits.name == scale_name:
                    model.graph.initializer[i].CopyFrom(new_tensor)
                    return

    for idx, node in enumerate(model.graph.node):
        if node.op_type != target_op_type:
            continue

        original_input = node.input[0]
        original_output = node.output[0]

        ipos_node_name = _find_qdq_node_name_for_tensor(model, original_input)
        opos_node_name = _find_qdq_node_name_for_consumer(model, original_output)

        if ipos_node_name:
            scale_in = _get_scale_init_name_by_node_name(model, ipos_node_name)
            if scale_in:
                set_scale(scale_in, POS2SCALE(ipos))
        else:
            scale_in_name = f"{node.name}_input_scale"
            zp_in_name = f"{node.name}_input_zp"
            q_in_name = f"{node.name}_QuantizeLinear_Input"
            dq_in_name = f"{node.name}_DequantizeLinear_Input"
            q_in_out = f"{q_in_name}_Output"
            dq_in_out = f"{dq_in_name}_Output"

            scale_in = helper.make_tensor(scale_in_name, TensorProto.FLOAT, [], [POS2SCALE(ipos)])
            zp_in = helper.make_tensor(zp_in_name, TensorProto.INT8, [], [0])
            model.graph.initializer.append(scale_in)
            model.graph.initializer.append(zp_in)

            q_in_node = helper.make_node(
                "QuantizeLinear", [original_input, scale_in_name, zp_in_name], [q_in_out], name=q_in_name
            )
            dq_in_node = helper.make_node(
                "DequantizeLinear", [q_in_out, scale_in_name, zp_in_name], [dq_in_out], name=dq_in_name
            )
            model.graph.node.insert(idx, q_in_node)
            model.graph.node.insert(idx + 1, dq_in_node)
            node.input[0] = dq_in_out
            idx += 2

        if opos_node_name:
            scale_out = _get_scale_init_name_by_node_name(model, opos_node_name)
            if scale_out:
                set_scale(scale_out, POS2SCALE(opos))
        else:
            scale_out_name = f"{node.name}_output_scale"
            zp_out_name = f"{node.name}_output_zp"
            q_out_name = f"{node.name}_QuantizeLinear_Output"
            dq_out_name = f"{node.name}_DequantizeLinear_Output"
            q_out_out = f"{q_out_name}_Output"
            new_output = f"{original_output}_orig"

            scale_out = helper.make_tensor(scale_out_name, TensorProto.FLOAT, [], [POS2SCALE(opos)])
            zp_out = helper.make_tensor(zp_out_name, TensorProto.INT8, [], [0])
            model.graph.initializer.append(scale_out)
            model.graph.initializer.append(zp_out)

            for other_node in model.graph.node:
                for i, inp in enumerate(other_node.input):
                    if inp == original_output:
                        other_node.input[i] = f"{dq_out_name}_Output"

            node.output[0] = new_output
            q_out_node = helper.make_node(
                "QuantizeLinear", [new_output, scale_out_name, zp_out_name], [q_out_out], name=q_out_name
            )
            dq_out_node = helper.make_node(
                "DequantizeLinear",
                [q_out_out, scale_out_name, zp_out_name],
                [f"{dq_out_name}_Output"],
                name=dq_out_name,
            )
            model.graph.node.append(q_out_node)
            model.graph.node.append(dq_out_node)

        break

    return model


def _create_model_with_pad_and_qdq(output_path: str, ipos: int = 5, opos: int = 10) -> None:
    """Create a simple ONNX model with Pad node surrounded by Q/DQ nodes."""
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 16, 16])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 20, 20])

    scale_in = helper.make_tensor("scale_in", TensorProto.FLOAT, [], [POS2SCALE(ipos)])
    zp_in = helper.make_tensor("zp_in", TensorProto.INT8, [], [0])
    scale_out = helper.make_tensor("scale_out", TensorProto.FLOAT, [], [POS2SCALE(opos)])
    zp_out = helper.make_tensor("zp_out", TensorProto.INT8, [], [0])
    pads = helper.make_tensor("pads", TensorProto.INT64, [8], [0, 0, 2, 2, 0, 0, 2, 2])

    q_in = helper.make_node("QuantizeLinear", ["input", "scale_in", "zp_in"], ["q_in_out"], name="QuantizeLinear_Input")
    dq_in = helper.make_node(
        "DequantizeLinear", ["q_in_out", "scale_in", "zp_in"], ["dq_in_out"], name="DequantizeLinear_Input"
    )
    pad = helper.make_node("Pad", ["dq_in_out", "pads"], ["pad_out"], name="Pad_0", mode="constant")
    q_out = helper.make_node(
        "QuantizeLinear", ["pad_out", "scale_out", "zp_out"], ["q_out_out"], name="QuantizeLinear_Output"
    )
    dq_out = helper.make_node(
        "DequantizeLinear", ["q_out_out", "scale_out", "zp_out"], ["output"], name="DequantizeLinear_Output"
    )

    graph = helper.make_graph(
        [q_in, dq_in, pad, q_out, dq_out], "pad_model", [X], [Y], [scale_in, zp_in, scale_out, zp_out, pads]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)


def perturb_scales_to_trigger_align_pad(model_path: str, output_path: str) -> None:
    """Create a model with Pad and Q/DQ nodes with different scales to trigger align_pad."""
    _create_model_with_pad_and_qdq(output_path, ipos=5, opos=10)


def _create_model_with_slice_and_qdq(output_path: str, ipos: int = 5, opos: int = 10) -> None:
    """Create a simple ONNX model with Slice node surrounded by Q/DQ nodes."""
    from onnx import TensorProto, helper

    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 8, 16, 16])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 8, 12, 12])

    scale_in = helper.make_tensor("scale_in", TensorProto.FLOAT, [], [POS2SCALE(ipos)])
    zp_in = helper.make_tensor("zp_in", TensorProto.INT8, [], [0])
    scale_out = helper.make_tensor("scale_out", TensorProto.FLOAT, [], [POS2SCALE(opos)])
    zp_out = helper.make_tensor("zp_out", TensorProto.INT8, [], [0])
    starts = helper.make_tensor("starts", TensorProto.INT64, [4], [0, 0, 2, 2])
    ends = helper.make_tensor("ends", TensorProto.INT64, [4], [1, 8, 14, 14])
    axes = helper.make_tensor("axes", TensorProto.INT64, [4], [0, 1, 2, 3])

    q_in = helper.make_node("QuantizeLinear", ["input", "scale_in", "zp_in"], ["q_in_out"], name="QuantizeLinear_Input")
    dq_in = helper.make_node(
        "DequantizeLinear", ["q_in_out", "scale_in", "zp_in"], ["dq_in_out"], name="DequantizeLinear_Input"
    )
    slice_node = helper.make_node("Slice", ["dq_in_out", "starts", "ends", "axes"], ["slice_out"], name="Slice_0")
    q_out = helper.make_node(
        "QuantizeLinear", ["slice_out", "scale_out", "zp_out"], ["q_out_out"], name="QuantizeLinear_Output"
    )
    dq_out = helper.make_node(
        "DequantizeLinear", ["q_out_out", "scale_out", "zp_out"], ["output"], name="DequantizeLinear_Output"
    )

    graph = helper.make_graph(
        [q_in, dq_in, slice_node, q_out, dq_out],
        "slice_model",
        [X],
        [Y],
        [scale_in, zp_in, scale_out, zp_out, starts, ends, axes],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, output_path)


def perturb_scales_to_trigger_align_slice(model_path: str, output_path: str) -> None:
    """Create a model with Slice and Q/DQ nodes with different scales to trigger align_slice."""
    _create_model_with_slice_and_qdq(output_path, ipos=5, opos=10)


def perturb_scales_to_trigger_adjustments(model_path: str, output_path: str) -> None:
    """
    Modify Q/DQ scales so that all six onnx_xint8_adjust routines are triggered:
    _adjust_shift_read, _adjust_shift_write, _adjust_shift_cut, _adjust_shift_bias,
    _adjust_hard_sigmoid, _adjust_shift_swish.
    """
    model = onnx.load(model_path)
    name_to_init = {init.name: init for init in model.graph.initializer}

    def set_scale_by_name(scale_init_name: str, new_scale: float) -> None:
        if scale_init_name not in name_to_init:
            return
        init = name_to_init[scale_init_name]
        arr = onnx.numpy_helper.to_array(init)
        new_val = np.array(new_scale, dtype=arr.dtype)
        new_tensor = onnx.numpy_helper.from_array(new_val, name=init.name)
        for i, inits in enumerate(model.graph.initializer):
            if inits.name == scale_init_name:
                model.graph.initializer[i].CopyFrom(new_tensor)
                return

    # --- adjust_shift_cut: Conv shift_cut = wpos+ipos-opos, clamp [0,16]. Set opos=0 and wpos+ipos>16.
    for node in model.graph.node:
        if node.op_type == "Conv" and len(node.input) >= 2:
            conv_out = node.output[0] if node.output else None
            scale_conv_out = _get_scale_initializer_name_for_consumer(model, conv_out) if conv_out else None
            if scale_conv_out:
                set_scale_by_name(scale_conv_out, POS2SCALE(0))
            scale_in = _get_scale_initializer_name_for_tensor(model, node.input[0])
            scale_w = _get_scale_initializer_name_for_tensor(model, node.input[1])
            if scale_in and scale_w:
                set_scale_by_name(scale_in, POS2SCALE(9))
                set_scale_by_name(scale_w, POS2SCALE(9))
            break

    # --- adjust_shift_bias: shift_bias = wpos+ipos-bpos, clamp to [min_sb,15]. Set bpos=0.
    for node in model.graph.node:
        if node.op_type == "Conv" and len(node.input) > 2:
            scale_bias = _get_scale_initializer_name_for_tensor(model, node.input[2])
            if scale_bias:
                set_scale_by_name(scale_bias, POS2SCALE(0))
            break

    # --- adjust_shift_read: Add shift_read = max(ipos)-min(ipos), clamp [0,7]. Set 0 and 10.
    for node in model.graph.node:
        if node.op_type == "Add" and len(node.input) >= 2:
            in0, in1 = node.input[0], node.input[1]
            scale0 = _get_scale_initializer_name_for_tensor(model, in0)
            scale1 = _get_scale_initializer_name_for_tensor(model, in1)
            if scale0 and scale1:
                set_scale_by_name(scale0, POS2SCALE(0))
                set_scale_by_name(scale1, POS2SCALE(10))
            break

    # --- adjust_shift_write: Add shift_write = min(ipos)-opos, clamp [-7,25]. Set opos=30.
    for node in model.graph.node:
        if node.op_type == "Add" and node.output:
            add_out = node.output[0]
            scale_add_out = _get_scale_initializer_name_for_consumer(model, add_out)
            if scale_add_out:
                set_scale_by_name(scale_add_out, POS2SCALE(30))
            break

    # --- adjust_hard_sigmoid: input pos in [0,15], output>=7, shift_sigmoid in [0,31]. Set input pos 20.
    for node in model.graph.node:
        if node.op_type == "HardSigmoid" and node.input:
            scale_in = _get_scale_initializer_name_for_tensor(model, node.input[0])
            if scale_in:
                set_scale_by_name(scale_in, POS2SCALE(20))
            break

    # --- adjust_shift_swish: Mul shift_swish = ipos0+ipos1-opos, clamp [0,15]. Set opos=0.
    for node in model.graph.node:
        if node.op_type == "Mul" and node.output:
            mul_out = node.output[0]
            scale_mul_out = _get_scale_initializer_name_for_consumer(model, mul_out)
            if scale_mul_out:
                set_scale_by_name(scale_mul_out, POS2SCALE(0))
            break

    onnx.save(model, output_path)


def write_xint8_adjust_yaml(
    output_dir: str,
    input_model_path: str,
    output_model_path: str,
) -> str:
    """Write shapeshifter YAML config with onnx_xint8_adjust pass. Returns path to the YAML file."""
    yaml_path = Path(output_dir, "xint8_adjust.yaml").as_posix()
    adapter_config = {
        "input_model_path": input_model_path,
        "passes": {
            "onnx_xint8_adjust": {
                "xint8_adjust": True,
            }
        },
        "output_model_path": output_model_path,
    }
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(adapter_config, f, allow_unicode=True, sort_keys=False)
    return yaml_path


# Log substrings emitted by each adjust_* in onnx_xint8_adjust (one per routine).
_ADJUST_LOG_MARKERS = (
    "Shift read",  # _adjust_shift_read
    "Shift write",  # _adjust_shift_write
    "Shift cut",  # _adjust_shift_cut
    "Shift bias",  # _adjust_shift_bias
    "HardSigmoid",  # _adjust_hard_sigmoid
    "Shift Swish",  # _adjust_shift_swish
)


class TestONNXAdapterONNXXInt8AdjustPass(unittest.TestCase):
    """Test the onnx_xint8_adjust adapter pass on XINT8 quantized models via the shapeshifter CLI."""

    @use_temporary_directory
    def test_xint8_adjust_pass_all_adjustments_triggered(self, tmpdir: str) -> None:
        """Perturb scales so all six adjust_* routines run; assert each logs and output model is valid or loadable."""
        quant_model_path = quantize_model_xint8(tmpdir, CALIBRATION_INPUT)
        valid_quant_path = Path(tmpdir, "xint8_adjust_ops_quant_valid.onnx").as_posix()
        ensure_model_valid_with_shape_inference(quant_model_path, valid_quant_path)
        perturbed_path = Path(tmpdir, "xint8_adjust_ops_quant_perturbed.onnx").as_posix()
        perturb_scales_to_trigger_adjustments(valid_quant_path, perturbed_path)
        adjusted_model_path = Path(tmpdir, "xint8_adjust_ops_adjusted.onnx").as_posix()
        with self.assertLogs("quark.shapeshifter.passes.onnx_xint8_adjust_screen", level="INFO") as log_ctx:
            yaml_path = write_xint8_adjust_yaml(tmpdir, perturbed_path, adjusted_model_path)
            cli(["shapeshifter", yaml_path])
            logs = " ".join(log_ctx.output)
            self.assertIn("Adjust the quantize info", logs)
            for marker in _ADJUST_LOG_MARKERS:
                self.assertIn(
                    marker,
                    logs,
                    f"Expected '{marker}' (each adjust_* should be triggered); got: {log_ctx.output}",
                )

    @use_temporary_directory
    def test_xint8_adjust_pass_align_concat(self, tmpdir: str) -> None:
        """Test align_concat: Concat inputs and output pos should be aligned to min pos."""
        quant_model_path = quantize_model_xint8(tmpdir, CALIBRATION_INPUT, ConcatModel, "concat_model")
        valid_quant_path = Path(tmpdir, "concat_model_quant_valid.onnx").as_posix()
        ensure_model_valid_with_shape_inference(quant_model_path, valid_quant_path)
        perturbed_path = Path(tmpdir, "concat_model_quant_perturbed.onnx").as_posix()
        perturb_scales_to_trigger_align_concat(valid_quant_path, perturbed_path)
        adjusted_model_path = Path(tmpdir, "concat_model_adjusted.onnx").as_posix()
        with self.assertLogs("quark.shapeshifter.passes.onnx_xint8_adjust_screen", level="INFO") as log_ctx:
            yaml_path = write_xint8_adjust_yaml(tmpdir, perturbed_path, adjusted_model_path)
            cli(["shapeshifter", yaml_path])
            logs = " ".join(log_ctx.output)
            self.assertIn("Input pos of concat node", logs)
            self.assertTrue(
                "concat node" in logs.lower() or "concat" in logs.lower(),
                f"Expected align_concat to be triggered; got: {log_ctx.output}",
            )

    @use_temporary_directory
    def test_xint8_adjust_pass_align_pool(self, tmpdir: str) -> None:
        """Test align_pool: MaxPool/AveragePool input and output pos should be aligned."""
        quant_model_path = quantize_model_xint8(tmpdir, CALIBRATION_INPUT, PoolModel, "pool_model")
        valid_quant_path = Path(tmpdir, "pool_model_quant_valid.onnx").as_posix()
        ensure_model_valid_with_shape_inference(quant_model_path, valid_quant_path)
        perturbed_path = Path(tmpdir, "pool_model_quant_perturbed.onnx").as_posix()
        perturb_scales_to_trigger_align_pool(valid_quant_path, perturbed_path)
        adjusted_model_path = Path(tmpdir, "pool_model_adjusted.onnx").as_posix()
        with self.assertLogs("quark.shapeshifter.passes.onnx_xint8_adjust_screen", level="INFO") as log_ctx:
            yaml_path = write_xint8_adjust_yaml(tmpdir, perturbed_path, adjusted_model_path)
            cli(["shapeshifter", yaml_path])
            logs = " ".join(log_ctx.output)
            self.assertIn("Input pos of pooling layer", logs)
        self.assertTrue(Path(adjusted_model_path).exists(), "Output model should be created")
        adjusted_model = onnx.load(adjusted_model_path)
        has_pool = any(n.op_type in ["MaxPool", "AveragePool", "GlobalAveragePool"] for n in adjusted_model.graph.node)
        self.assertTrue(has_pool, "Model should contain Pool operation")

    @use_temporary_directory
    def test_xint8_adjust_pass_align_pad(self, tmpdir: str) -> None:
        """Test align_pad: Pad input and output pos should be aligned."""
        quant_model_path = quantize_model_xint8(tmpdir, CALIBRATION_INPUT, PadModel, "pad_model")
        valid_quant_path = Path(tmpdir, "pad_model_quant_valid.onnx").as_posix()
        ensure_model_valid_with_shape_inference(quant_model_path, valid_quant_path)

        input_model = onnx.load(valid_quant_path)
        has_pad_in_input = any(n.op_type == "Pad" for n in input_model.graph.node)

        perturbed_path = Path(tmpdir, "pad_model_quant_perturbed.onnx").as_posix()
        perturb_scales_to_trigger_align_pad(valid_quant_path, perturbed_path)
        adjusted_model_path = Path(tmpdir, "pad_model_adjusted.onnx").as_posix()
        with self.assertLogs("quark.shapeshifter.passes.onnx_xint8_adjust_screen", level="INFO") as log_ctx:
            yaml_path = write_xint8_adjust_yaml(tmpdir, perturbed_path, adjusted_model_path)
            cli(["shapeshifter", yaml_path])
            logs = " ".join(log_ctx.output)
            self.assertIn("Input pos of pad layer", logs)
        self.assertTrue(Path(adjusted_model_path).exists(), "Output model should be created")

        if has_pad_in_input:
            adjusted_model = onnx.load(adjusted_model_path)
            has_pad = any(n.op_type == "Pad" for n in adjusted_model.graph.node)
            self.assertTrue(has_pad, "Model should contain Pad operation")

    @use_temporary_directory
    def test_xint8_adjust_pass_align_slice(self, tmpdir: str) -> None:
        """Test align_slice: Slice input and output pos should be aligned."""
        quant_model_path = quantize_model_xint8(tmpdir, CALIBRATION_INPUT, SliceModel, "slice_model")
        valid_quant_path = Path(tmpdir, "slice_model_quant_valid.onnx").as_posix()
        ensure_model_valid_with_shape_inference(quant_model_path, valid_quant_path)
        perturbed_path = Path(tmpdir, "slice_model_quant_perturbed.onnx").as_posix()
        perturb_scales_to_trigger_align_slice(valid_quant_path, perturbed_path)
        adjusted_model_path = Path(tmpdir, "slice_model_adjusted.onnx").as_posix()
        with self.assertLogs("quark.shapeshifter.passes.onnx_xint8_adjust_screen", level="INFO") as log_ctx:
            yaml_path = write_xint8_adjust_yaml(tmpdir, perturbed_path, adjusted_model_path)
            cli(["shapeshifter", yaml_path])
            logs = " ".join(log_ctx.output)
            self.assertIn("Input pos of Slice layer", logs)
        self.assertTrue(Path(adjusted_model_path).exists(), "Output model should be created")
        adjusted_model = onnx.load(adjusted_model_path)
        has_slice = any(n.op_type == "Slice" for n in adjusted_model.graph.node)
        self.assertTrue(has_slice, "Model should contain Slice operation")


if __name__ == "__main__":
    unittest.main()
