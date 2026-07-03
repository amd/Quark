#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""End-to-end unittest for apply_layer_quant_config using original model layer names (no need to inspect FX graph)."""

import logging
import logging.handlers

import torch
import torch.nn as nn
from torch.fx import GraphModule

from quark.common.utils.import_utils import export_for_training
from quark.common.utils.testing_utils import torch_device
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, QuantizationMode, RoundType, ScaleType
from quark.torch.quantization.graph.processor.insert_quantizer import apply_layer_quant_config
from quark.torch.quantization.graph.torch_utils import QuantConv2d, is_relu_act_node
from quark.torch.quantization.observer.observer import (
    PerChannelPowOf2MinMaxObserver,
    PerTensorPowOf2MinMaxObserver,
)
from quark.torch.quantization.tensor_quantize import FrozenScaledFakeQuantize, ScaledFakeQuantize


def _count_per_channel_quantizers(model: torch.nn.Module) -> int:
    """Count modules that are ScaledFakeQuantize with qscheme == per_channel."""
    count = 0
    for m in model.modules():
        if isinstance(m, ScaledFakeQuantize) and getattr(m, "qscheme", None) == QSchemeType.per_channel:
            count += 1
    return count


def _count_int8_quantizers(model: torch.nn.Module) -> int:
    """Count quantizer modules (ScaledFakeQuantize / FrozenScaledFakeQuantize) with dtype == int8."""
    count = 0
    for m in model.modules():
        if isinstance(m, ScaledFakeQuantize | FrozenScaledFakeQuantize):
            if getattr(m, "dtype", None) == Dtype.int8:
                count += 1
    return count


# ----- Simple CNN: 3 conv layers (layer_quant_config test targets conv only) -----
class SimpleCNN(nn.Module):
    """Minimal CNN for testing: 3 conv layers; ReLU in forward is not under layer_quant_config in this test."""

    def __init__(self, in_ch=3, mid_ch=16, out_ch=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, mid_ch, 3, padding=1, bias=True)
        self.conv2 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(mid_ch, mid_ch, 3, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(x)
        x = self.relu(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        return x


def _get_xint8_global_config():
    """Build global XINT8-style config (per-tensor int8 weight, uint8 act), as in run_quant_perchannel.py."""
    calib_observer = PerTensorPowOf2MinMaxObserver
    weight_spec = QTensorConfig(
        dtype=Dtype.uint8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=calib_observer,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    act_spec = QTensorConfig(
        dtype=Dtype.uint8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=calib_observer,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    return QLayerConfig(
        input_tensors=act_spec,
        output_tensors=act_spec,
        weight=weight_spec,
        bias=weight_spec,
    )


def _get_per_channel_layer_config():
    """Per-channel weight spec for layer_quant_config (as in run_quant_perchannel.py)."""
    calib_observer = PerTensorPowOf2MinMaxObserver
    calib_observer_pc = PerChannelPowOf2MinMaxObserver
    per_channel_weight = QTensorConfig(
        dtype=Dtype.uint8,
        qscheme=QSchemeType.per_channel,
        observer_cls=calib_observer_pc,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
        ch_axis=0,
    )
    per_tensor_act = QTensorConfig(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=calib_observer,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    return QLayerConfig(
        input_tensors=per_tensor_act,
        output_tensors=per_tensor_act,
        weight=per_channel_weight,
        bias=per_channel_weight,
    )


def _get_relu_layer_config():
    """Layer config for ReLU: int8 input/output only (no weight/bias). Used for layerwise override to int8."""
    calib_observer = PerTensorPowOf2MinMaxObserver
    int8_act_spec = QTensorConfig(
        dtype=Dtype.int8,
        qscheme=QSchemeType.per_tensor,
        observer_cls=calib_observer,
        symmetric=True,
        scale_type=ScaleType.float,
        round_method=RoundType.half_even,
        is_dynamic=False,
    )
    return QLayerConfig(
        input_tensors=int8_act_spec,
        output_tensors=int8_act_spec,
        weight=None,
        bias=None,
    )


def _collect_relu_node_names(graph_model: GraphModule) -> list[str]:
    """Collect ReLU call_function node names (for layer_quant_config keys)."""
    return [node.name for node in graph_model.graph.nodes if is_relu_act_node(node)]


# Intentional wrong name to trigger "unmatched layer name" warning from apply_layer_quant_config.
_UNMATCHED_LAYER_NAME = "nonexistent_or_typo_layer"


def test_layer_quant_config_e2e_quantize_forward():
    """
    Run PTQ without layer_quant_config, then apply_layer_quant_config using original model
    conv layer names (conv1, conv2, conv3) only. No need to export FX graph to discover names.
    Includes one invalid name in config to verify the unmatched-name warning is emitted.
    """
    device = torch_device
    model = SimpleCNN(in_ch=3, mid_ch=16, out_ch=8).to(device).eval()
    example_inputs = (torch.rand(2, 3, 16, 16).to(device),)
    out_fp = model(*example_inputs)

    # ---- Step 1: Quantize WITHOUT layer_quant_config (global only, all per-tensor) ----
    global_config = _get_xint8_global_config()
    config_no_layer = QConfig(
        global_quant_config=global_config,
        quant_mode=QuantizationMode.fx_graph_mode,
        layer_quant_config=None,
    )
    quantizer = ModelQuantizer(config_no_layer)
    graph_model = export_for_training(model, example_inputs).module()
    graph_model = torch.fx.GraphModule(graph_model, graph_model.graph)
    calib_data = [example_inputs[0] for _ in range(4)]
    quantized_model = quantizer.quantize_model(graph_model, calib_data)

    # replace_conv2d_qtconv2d already sets org_module_name on conv nodes during quantize_model.
    # Sanity: we expect 3 QuantConv2d in the graph
    conv_count = sum(
        1
        for n in quantized_model.graph.nodes
        if n.op == "call_module" and isinstance(getattr(quantized_model, n.target, None), QuantConv2d)
    )
    assert conv_count == 3, f"Graph should have 3 QuantConv2d modules, got {conv_count}"

    _ = quantized_model.eval()(*example_inputs)

    num_per_channel_before = _count_per_channel_quantizers(quantized_model)
    assert num_per_channel_before == 0, (
        f"Without layer_quant_config, model should have no per-channel quantizers, got {num_per_channel_before}"
    )
    num_int8_before = _count_int8_quantizers(quantized_model)
    assert num_int8_before == 0, (
        f"Without layer_quant_config, model should have no int8 quantizers, got {num_int8_before}"
    )

    # ---- Step 2: layer_quant_config by original conv layer names only (no graph inspection) ----
    # collect the layer config for the conv layers
    layer_config = _get_per_channel_layer_config()
    layer_quant_config = {
        "conv1": layer_config,
        "conv2": layer_config,
        "conv3": layer_config,
        _UNMATCHED_LAYER_NAME: layer_config,  # Intentionally wrong name to trigger warning
    }

    # collect the layer config for the relu layers
    relu_layer_config = _get_relu_layer_config()
    relu_node_names = _collect_relu_node_names(quantized_model)
    for name in relu_node_names:
        layer_quant_config[name] = relu_layer_config

    config_with_layer = QConfig(
        global_quant_config=global_config,
        quant_mode=QuantizationMode.fx_graph_mode,
        layer_quant_config=layer_quant_config,
    )

    # Call apply_layer_quant_config and assert the unmatched-name warning is logged
    # ScreenLogger uses logger name "<module>_screen"
    insert_quantizer_logger = logging.getLogger("quark.torch.quantization.graph.processor.insert_quantizer_screen")
    log_capture = logging.handlers.MemoryHandler(capacity=100, flushLevel=logging.CRITICAL)
    log_capture.setLevel(logging.WARNING)
    insert_quantizer_logger.addHandler(log_capture)
    insert_quantizer_logger.setLevel(logging.WARNING)
    try:
        apply_layer_quant_config(quantized_model, config_with_layer)
    finally:
        insert_quantizer_logger.removeHandler(log_capture)
    warning_messages = [r.getMessage() for r in log_capture.buffer]
    assert any("did not match" in msg and _UNMATCHED_LAYER_NAME in msg for msg in warning_messages), (
        f"Expected a warning about unmatched layer name {_UNMATCHED_LAYER_NAME!r}; got: {warning_messages}"
    )

    num_per_channel_after = _count_per_channel_quantizers(quantized_model)
    assert num_per_channel_after == 6, (
        f"After apply_layer_quant_config with per-channel layer config (3 conv: weight+bias each), expected 6 per-channel quantizers, got {num_per_channel_after}"
    )
    num_int8_after = _count_int8_quantizers(quantized_model)
    assert num_int8_after == 5, f"After apply_layer_quant_config, expected 5 int8 quantizers, got {num_int8_after}"

    out_after = quantized_model.eval()(*example_inputs)
    assert out_after.shape == out_fp.shape, "Output shape should still match after apply_layer_quant_config"
    assert out_after.isfinite().all(), "Output should still be finite after apply_layer_quant_config"


if __name__ == "__main__":
    test_layer_quant_config_e2e_quantize_forward()
    print("test_layer_quant_config_e2e_quantize_forward passed")
