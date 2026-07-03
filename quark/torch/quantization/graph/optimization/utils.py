#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
from torch import nn
from torch.fx import GraphModule, Node

from quark.common.utils.import_utils import _get_tensor_constant_from_node
from quark.common.utils.log import ScreenLogger
from quark.torch.quantization.graph.torch_utils import is_call_module_node
from quark.torch.quantization.nn.modules.quantize_conv import QuantConv2d, QuantConvTranspose2d
from quark.torch.quantization.nn.modules.quantize_conv_bn_fused import (
    QuantConvTransposeBatchNorm2d,
    QuantizedConvBatchNorm2d,
)
from quark.torch.quantization.nn.modules.quantize_leakyrelu import QuantLeakyReLU
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear
from quark.torch.quantization.nn.modules.quantize_pool import QuantAdaptiveAvgPool2d, QuantAvgPool2d
from quark.torch.quantization.tensor_quantize import (
    FrozenScaledFakeQuantize,
    ScaledFakeQuantize,
    StaticScaledFakeQuantize,
)

logger = ScreenLogger(__name__)

# Mapping from quantized module class to the suffix used in replacement module names.
replace_ops_module_name_suffix = {
    # conv + batch_norm -> QuantizedConvBatchNorm2d
    QuantizedConvBatchNorm2d: "_bn_quantized_module",
    # conv_transpose2d + batch_norm -> QuantConvTransposeBatchNorm2d
    QuantConvTransposeBatchNorm2d: "_bn_quantized_module",
    # linear -> QuantLinear
    QuantLinear: "_quantized_module",
    # conv2d -> QuantConv2d
    QuantConv2d: "_quantized_module",
    # conv_transpose2d -> QuantConvTranspose2d
    QuantConvTranspose2d: "_quantized_model",
    # adaptive_avg_pool2d -> QuantAdaptiveAvgPool2d
    QuantAdaptiveAvgPool2d: "_quantized_module",
    QuantAvgPool2d: "_quantized_module",
    QuantLeakyReLU: "_quantized_module",
}


def get_node_original_module_name(node: Node) -> str | None:
    """
    Resolve the original module name in the source model from the node's meta.

    Reads ``node.meta["nn_module_stack"]``, which has the form
    ``{'L__self__': ('', '...ResNet'), 'L__self__bn1': ('bn1', '...BatchNorm2d')}``.
    The last entry in the stack (the leaf module for this node) is used;
    its first element is the module name (e.g. ``'bn1'``).

    Args:
        node: An FX graph node; must have ``meta`` with ``nn_module_stack``
            for a non-None result.

    Returns:
        The original module name (e.g. ``'bn1'``), or None if ``meta`` has no
        ``nn_module_stack``, the stack is empty, or the name is empty.
    """
    if not hasattr(node, "meta"):
        return None  # pragma: no cover
    stack = node.meta.get("nn_module_stack")
    if not stack or not isinstance(stack, dict):
        return None  # pragma: no cover
    # Use the last stack entry (key -> (name, class_path)) and return name.
    try:
        last_entry = next(reversed(stack.values()))
        if isinstance(last_entry, list | tuple) and len(last_entry) >= 1:
            name = last_entry[0]
            return name if name else None
    except (StopIteration, TypeError):
        pass  # pragma: no cover
    return None  # pragma: no cover


def _copy_node_meta_info(org_node: Node, target_node: Node) -> None:
    """Copy meta from org_node to target_node: fake tensor (val) and skip_quant."""
    assert hasattr(org_node, "meta") and "val" in org_node.meta
    assert hasattr(target_node, "meta")
    fake_mode = org_node.meta["val"].fake_mode
    tensor_device = org_node.meta["val"].device
    fake_tensor = fake_mode.from_tensor(
        torch.randn(org_node.meta["val"].shape, device=tensor_device), static_shapes=True
    )
    target_node.meta["val"] = fake_tensor
    if "skip_quant" in org_node.meta:
        target_node.meta["skip_quant"] = org_node.meta["skip_quant"]
    return


def is_all_nodes_save_parameters(m: GraphModule, nodes: list[Node]) -> bool:
    """Return True iff every node is a get_attr and resolves to a nn.Parameter in m."""
    is_parameters = True
    for node in nodes:
        if node.op != "get_attr":
            is_parameters = False
            break
        if not isinstance(
            _get_tensor_constant_from_node(node, m),  # type: ignore [no-untyped-call]
            torch.nn.Parameter,
        ):
            is_parameters = False
            return is_parameters
    return is_parameters


def is_quantizer_node(m: GraphModule, n: Node) -> bool:
    """Return True iff n is a call_module node whose target is a quantizer (ScaledFakeQuantize or FrozenScaledFakeQuantize)."""
    if (
        (not isinstance(n, Node))
        or (not is_call_module_node(n))
        or (not isinstance(n.target, str))
        or (not isinstance(getattr(m, n.target), FrozenScaledFakeQuantize | ScaledFakeQuantize))
    ):
        return False
    return True


def is_quantizer(module: nn.Module) -> bool:
    """Return True iff module is a ScaledFakeQuantize or FrozenScaledFakeQuantize."""
    return isinstance(module, FrozenScaledFakeQuantize | ScaledFakeQuantize)


def get_quantizer_scale_pos(quantizer: ScaledFakeQuantize | FrozenScaledFakeQuantize) -> float:
    """
    Return scale position for a quantizer: pos = log2(1 / scale).

    Applicable to ScaledFakeQuantize and FrozenScaledFakeQuantize.
    Examples: scale 0.5 -> pos 1 (1 / 2**1 = 0.5);
              scale 0.0625 -> pos 4 (1 / 2**4 = 0.0625).
    """
    scale = quantizer.scale.detach().clone()
    pos = torch.log2(1 / scale).item()
    return pos


def get_quantizer_powof2_scale_pos(quantizer: ScaledFakeQuantize | FrozenScaledFakeQuantize) -> int:
    """
    Return integer scale position for a power-of-two quantizer: pos = log2(1 / scale).

    Applicable to StaticScaledFakeQuantize and FrozenScaledFakeQuantize.
    Examples: scale 0.5 -> pos 1; scale 0.0625 -> pos 4.
    """
    assert isinstance(quantizer, StaticScaledFakeQuantize | FrozenScaledFakeQuantize)
    scale = quantizer.scale.detach().clone()
    pos = get_quantizer_scale_pos(quantizer)
    if pos % 1 != 0:
        logger.warning(
            "Quantizer scale %s is not power-of-two; verify it matches the intended config.",
            scale.item(),
        )
    return int(pos)
