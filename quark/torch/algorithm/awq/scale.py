#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2023 MIT HAN Lab
# SPDX-License-Identifier: MIT
#

from typing import Dict, List, Optional, Tuple, cast

import torch
import torch.nn as nn

from quark.shares.utils.log import ScreenLogger
from quark.torch.algorithm.awq.modules.act import ScaledActivation
from quark.torch.algorithm.utils.utils import is_attention_module
from quark.torch.quantization.utils import assert_no_nan
from quark.torch.utils import getattr_recursive, setattr_recursive
from quark.torch.utils.accelerate_helper import OffloadParameter, update_offload_parameter

logger = ScreenLogger(__name__)

allowed_norms = ("RMSNorm", "CohereLayerNorm")
allowed_act_fns = ("BloomGelu", "NewGELUActivation", "PytorchGELUTanh", "GELUActivation")


@torch.no_grad()
def apply_clip(module: nn.Module, clip_list: list[tuple[str, torch.Tensor]], device: torch.device) -> None:
    for name, max_val in clip_list:
        layer: nn.Linear = cast(nn.Linear, getattr_recursive(module, name))
        # When using accelerate, there will be meta data. accelerate will manage the meta data and does not need to be actively scheduled.
        if "meta" not in str(device):
            layer.to(device)
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        update_offload_parameter(layer, "weight", layer.weight)
        if "meta" not in str(device):
            layer.cpu()


def apply_scale(
    module: nn.Module,
    scales_list: list[tuple[str, tuple[str, ...], torch.Tensor]],
    input_feat_dict: dict[str, torch.Tensor] | None = None,
    device: torch.device | None = torch.device("cuda"),
    num_attention_heads: int = 1,
    num_key_value_heads: int = 1,
) -> None:
    def get_prev_op_parent_op(module: nn.Module, prev_op_name: str) -> nn.Module:
        """
        This function is an encapsulation of the internal lengthy code to improve readability
        Our goal is to identify the Attention module, because there are only two cases: either the input itself is an Attention module, or it is a linear submodule of an Attention module.

        We obtain the module name of the current layer, split it by "." to identify the name of the preceding layer, and then use that name to locate the corresponding parent module.
        If this module cannot be split, then its parent module is the module itself.
        """

        if "." not in prev_op_name:
            # module is Attention Class
            prev_op_parent_op = module
        else:
            # Parent module is Attention Class
            prev_op_parent_op = getattr_recursive(module, ".".join(prev_op_name.split(".")[:-1]))
        return prev_op_parent_op

    for prev_op_name, layer_names, scales in scales_list:
        prev_op = getattr_recursive(module, prev_op_name)
        layers = [getattr_recursive(module, name) for name in layer_names]

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == len(layer_names) == 1
            try:
                prev_op_parent_op = get_prev_op_parent_op(module, prev_op_name)
                is_prev_op_in_attention_module = is_attention_module(prev_op_parent_op)
                scale_fc_fc(
                    prev_op, layers[0], scales, num_attention_heads, num_key_value_heads, is_prev_op_in_attention_module
                )
            except RuntimeError as e:
                logger.warning(
                    f"\nUnknown fc1-scales-fc2 pair to support scaling between them, the scale (smooth) computation will not be implemented in fact. This may impact the quantization accuracy."
                    f"\n\tfc1 is {prev_op_name}, shape is {prev_op.weight.shape}."
                    f"\n\tscales shape is {scales.shape}."
                    f"\n\tfc2 is {layer_names[0]}, shape is {layers[0].weight.shape}."
                    f"\nPlease check your model and/or algorithm configuration(s) or report your case to us."
                )

        elif isinstance(prev_op, nn.LayerNorm) or any(
            t.lower() in str(prev_op.__class__).lower() for t in allowed_norms
        ):
            scale_ln_fcs(prev_op, layers, scales)

        elif isinstance(prev_op, nn.GELU) or any(t.lower() in str(prev_op.__class__).lower() for t in allowed_act_fns):
            new_module = ScaledActivation(prev_op, scales)
            setattr_recursive(module, prev_op_name, new_module)
            assert len(layers) == 1
            scale_gelu_fc(prev_op, layers[0], scales)

        else:
            raise NotImplementedError(f"prev_op {type(prev_op)} not supported yet!")

        # apply the scaling to input feat if given; prepare it for clipping
        if input_feat_dict is not None:
            for layer_name in layer_names:
                # Skip the modules that are not quantized
                if layer_name in input_feat_dict:
                    inp = input_feat_dict[layer_name]
                    inp.div_(scales.view(1, -1).to(inp.device))


@torch.no_grad()
def scale_ln_fcs(ln: nn.Module, fcs: list[nn.Module], scales: torch.Tensor) -> None:
    if not isinstance(fcs, list):
        fcs = [fcs]
    with OffloadParameter(ln):
        if hasattr(ln, "weight") and ln.weight is not None:
            scales = scales.to(ln.weight.device)
            if "gemma" in str(ln.__class__).lower():
                ln.weight.data = (ln.weight.data + 1.0) / scales.to(ln.weight.device) - 1.0
            else:
                ln.weight.div_(scales.to(ln.weight.device))
            update_offload_parameter(ln, "weight", ln.weight)
        else:  # for grok, the scale of RMSnorm is named by "scale"
            scales = scales.to(ln.scale.device)
            ln.scale.div_(scales.to(ln.scale.device))
            update_offload_parameter(ln, "scale", ln.scale)

        if hasattr(ln, "bias") and ln.bias is not None:
            ln.bias.div_(scales.to(ln.bias.device))
            update_offload_parameter(ln, "bias", ln.bias)

        for fc in fcs:
            with OffloadParameter(fc):
                fc.weight.mul_(scales.to(fc.weight.device).view(1, -1))
                update_offload_parameter(fc, "weight", fc.weight)

        for p in ln.parameters():
            assert_no_nan(p, "LN parameters should not contain NaN")
        for fc in fcs:
            with OffloadParameter(fc):
                for name, p in fc.named_parameters():
                    assert_no_nan(p, "FC parameters should not contain NaN")


@torch.no_grad()
def scale_fc_fc(
    fc1: nn.Module,
    fc2: nn.Module,
    scales: torch.Tensor,
    num_attention_heads: int = 1,
    num_key_value_heads: int = 1,
    is_prev_op_in_attention_module: bool = False,
) -> None:
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    def get_group_query_attention_scales(
        scales: torch.Tensor, num_attention_heads: int, num_key_value_heads: int
    ) -> tuple[torch.Tensor, ...]:
        num_head_repeats = num_attention_heads // num_key_value_heads
        head_dims = scales.numel() // num_attention_heads
        scales_tmp = scales.view(num_key_value_heads, num_head_repeats, head_dims).max(dim=1, keepdim=True)[
            0
        ]  # (num_key_value_heads, 1, head_dims)
        prev_scales = scales_tmp.reshape(-1)
        scales = scales_tmp.expand(num_key_value_heads, num_head_repeats, head_dims).reshape(-1)
        return prev_scales, scales

    # Group Query Attention
    if (
        is_prev_op_in_attention_module
        and fc1.weight.shape[0] != scales.size(0)
        and ((num_attention_heads // num_key_value_heads) != 1)
    ):
        prev_scales, scales = get_group_query_attention_scales(scales, num_attention_heads, num_key_value_heads)
        fc1.weight[-prev_scales.size(0) :].div_(prev_scales.to(fc1.weight.device).view(-1, 1))
        if fc1.bias is not None:
            fc1.bias.div_(prev_scales.to(fc1.bias.device).view(-1))
        fc2.weight.mul_(scales.to(fc2.weight.device).view(1, -1))

    # Multi-head Attention
    # TODO: can unify with Group Query Attention using same scale (smooth) computations.
    elif scales.size(0) == fc2.weight.shape[1]:
        if fc1.weight.shape[0] > scales.size(0):
            # For layer which merge qkv, need to seperate out the v_proj from fc1.weight to do scale (smooth) with o_proj
            # An example model: microsoft/Phi-3-mini-4k-instruct
            fc1.weight[-scales.size(0) :].div_(scales.to(fc1.weight.device).view(-1, 1))
        elif fc1.weight.shape[0] == scales.size(0):
            # For layer which has seperate qkv or mlp, can directly do scale (smooth) between fc1 and fc2 (e.g., v_proj and o_proj)
            # An example model: google/gemma-7b
            fc1.weight.div_(scales.to(fc1.weight.device).view(-1, 1))
        else:
            raise RuntimeError("Unable to perform scale (smooth) since fc2's shape is larger than fc1's shape.")

        if fc1.bias is not None:
            fc1.bias.div_(scales.to(fc1.bias.device).view(-1))
        fc2.weight.mul_(scales.to(fc2.weight.device).view(1, -1))

    else:
        raise RuntimeError("Unable to perform scale (smooth) since the fc1-scale-fc2 pair has a mismatch tensor shape.")

    update_offload_parameter(fc1, "weight", fc1.weight)
    update_offload_parameter(fc2, "weight", fc2.weight)

    for p in fc1.parameters():
        assert_no_nan(p, "FC1 parameters should not contain NaN")
    for p in fc2.parameters():
        assert_no_nan(p, "FC2 parameters should not contain NaN")


@torch.no_grad()
def scale_gelu_fc(gelu: nn.Module, fc: nn.Module, scales: torch.Tensor) -> None:
    assert isinstance(gelu, nn.GELU) or any(t.lower() in str(gelu.__class__).lower() for t in allowed_act_fns)
    assert isinstance(fc, nn.Linear)
    with OffloadParameter(fc):
        fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))
        update_offload_parameter(fc, "weight", fc.weight)
        for p in fc.parameters():
            assert_no_nan(p, "FC parameters should not contain NaN")
