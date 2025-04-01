#
# Modifications copyright(c) 2024 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Copyright (c) 2023 MIT HAN Lab
# SPDX-License-Identifier: MIT
#

import torch
import torch.nn as nn
from typing import Tuple, List, Optional, Dict, cast
from quark.torch.algorithm.awq.modules.act import ScaledActivation
from quark.torch.algorithm.utils.module import get_op_by_name, set_op_by_name
from quark.shares.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

allowed_norms = ("RMSNorm", "CohereLayerNorm")
allowed_act_fns = ("BloomGelu", "NewGELUActivation", "PytorchGELUTanh", "GELUActivation")


@torch.no_grad()
def apply_clip(module: nn.Module, clip_list: List[Tuple[str, torch.Tensor]], device: torch.device) -> None:
    for name, max_val in clip_list:
        layer: nn.Linear = cast(nn.Linear, get_op_by_name(module, name))
        layer.to(device)
        max_val = max_val.to(layer.weight.device)
        org_shape = layer.weight.shape
        layer.weight.data = layer.weight.data.reshape(*max_val.shape[:2], -1)
        layer.weight.data = torch.clamp(layer.weight.data, -max_val, max_val)
        layer.weight.data = layer.weight.data.reshape(org_shape)
        layer.cpu()


def apply_scale(module: nn.Module,
                scales_list: List[Tuple[str, Tuple[str, ...], torch.Tensor]],
                input_feat_dict: Optional[Dict[str, torch.Tensor]] = None,
                device: Optional[torch.device] = torch.device('cuda'),
                num_attention_heads: int = 1,
                num_key_value_heads: int = 1) -> None:
    for prev_op_name, layer_names, scales in scales_list:
        prev_op = get_op_by_name(module, prev_op_name)
        layers = [get_op_by_name(module, name) for name in layer_names]

        if isinstance(prev_op, nn.Linear):
            assert len(layers) == 1
            scale_fc_fc(prev_op, layers[0], scales, num_attention_heads, num_key_value_heads)

        elif isinstance(prev_op, nn.LayerNorm) or any(t.lower() in str(prev_op.__class__).lower()
                                                      for t in allowed_norms):
            scale_ln_fcs(prev_op, layers, scales)

        elif isinstance(prev_op, nn.GELU) or any(t.lower() in str(prev_op.__class__).lower() for t in allowed_act_fns):
            new_module = ScaledActivation(prev_op, scales)
            set_op_by_name(module, prev_op_name, new_module)
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
def scale_ln_fcs(ln: nn.Module, fcs: List[nn.Module], scales: torch.Tensor) -> None:
    if not isinstance(fcs, list):
        fcs = [fcs]

    if hasattr(ln, 'weight') and ln.weight is not None:
        scales = scales.to(ln.weight.device)
        if "gemma" in str(ln.__class__).lower():
            ln.weight.data = (ln.weight.data + 1.0) / scales.to(ln.weight.device) - 1.0
        else:
            ln.weight.div_(scales.to(ln.weight.device))
    else:  # for grok, the scale of RMSnorm is named by "scale"
        scales = scales.to(ln.scale.device)
        ln.scale.div_(scales.to(ln.scale.device))

    if hasattr(ln, 'bias') and ln.bias is not None:
        ln.bias.div_(scales.to(ln.bias.device))

    for fc in fcs:
        fc.weight.mul_(scales.to(fc.weight.device).view(1, -1))

    for p in ln.parameters():
        assert torch.isnan(p).sum() == 0
    for fc in fcs:
        for p in fc.parameters():
            assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_fc_fc(fc1: nn.Module,
                fc2: nn.Module,
                scales: torch.Tensor,
                num_attention_heads: int = 1,
                num_key_value_heads: int = 1) -> None:
    assert isinstance(fc1, nn.Linear)
    assert isinstance(fc2, nn.Linear)

    def get_group_query_attention_scales(scales: torch.Tensor, num_attention_heads: int,
                                         num_key_value_heads: int) -> Tuple[torch.Tensor, ...]:
        num_head_repeats = num_attention_heads // num_key_value_heads
        head_dims = scales.numel() // num_attention_heads
        scales_tmp = scales.view(num_key_value_heads, num_head_repeats,
                                 head_dims).max(dim=1, keepdim=True)[0]  # (num_key_value_heads, 1, head_dims)
        prev_scales = scales_tmp.reshape(-1)
        scales = scales_tmp.expand(num_key_value_heads, num_head_repeats, head_dims).reshape(-1)
        return prev_scales, scales

    # Group Query Attention
    if fc1.weight.shape[0] != scales.size(0) and ((num_attention_heads // num_key_value_heads) != 1):
        prev_scales, scales = get_group_query_attention_scales(scales, num_attention_heads, num_key_value_heads)
        fc1.weight.div_(prev_scales.to(fc1.weight.device).view(-1, 1))
        if fc1.bias is not None:
            fc1.bias.div_(prev_scales.to(fc1.bias.device).view(-1))
        fc2.weight.mul_(scales.to(fc2.weight.device).view(1, -1))
    elif fc1.weight.shape[0] == scales.size(0) == fc2.weight.shape[1]:
        # Multi-head Attention
        fc1.weight[-scales.size(0):].div_(scales.to(fc1.weight.device).view(-1, 1))  # For layer which merge qkv
        if fc1.bias is not None:
            fc1.bias.div_(scales.to(fc1.bias.device).view(-1))
        fc2.weight.mul_(scales.to(fc2.weight.device).view(1, -1))

    for p in fc1.parameters():
        assert torch.isnan(p).sum() == 0
    for p in fc2.parameters():
        assert torch.isnan(p).sum() == 0


@torch.no_grad()
def scale_gelu_fc(gelu: nn.Module, fc: nn.Module, scales: torch.Tensor) -> None:
    assert (isinstance(gelu, nn.GELU) or any(t.lower() in str(gelu.__class__).lower() for t in allowed_act_fns))
    assert isinstance(fc, nn.Linear)

    fc.weight.mul_(scales.view(1, -1).to(fc.weight.device))

    for p in fc.parameters():
        assert torch.isnan(p).sum() == 0
