#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import numpy as np
import torch
from typing import Tuple, List, Dict, Union
try:
    from gguf import GGMLQuantizationType  # type: ignore
except ImportError as e:
    raise ImportError("please install gguf==0.6.0") from e


def quantize_row_q4_1(inpt: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    block_size = 32
    assert inpt.size(-1) % block_size == 0
    assert inpt.dtype == torch.float32 and scale.dtype == torch.float32 \
        and zero_point.dtype == torch.float32
    origin_shape = inpt.shape
    inpt = inpt.reshape(-1, block_size)
    scale = scale.reshape(-1, 1)
    zero_point = zero_point.reshape(-1, 1)
    assert inpt.size(0) == scale.size(0) and inpt.size(0) == zero_point.size(0)

    min_val = -scale * zero_point
    scale_inverse = (1 / scale).masked_fill(scale == 0.0, 0)

    quant_inpt = torch.round((inpt - min_val) * scale_inverse).to(torch.uint8).clamp(0, 15)
    quant_inpt_left_part = quant_inpt[:, :block_size // 2]
    quant_inpt_right_part = quant_inpt[:, block_size // 2:]
    data_part = quant_inpt_left_part + (quant_inpt_right_part << 4)

    scale = scale.to(torch.float16)
    scale_part = torch.frombuffer(scale.numpy(), dtype=torch.uint8).reshape(scale.size(0), 2)
    min_val = min_val.to(torch.float16)
    min_val_part = torch.frombuffer(min_val.numpy(), dtype=torch.uint8).reshape(min_val.size(0), 2)

    return torch.cat([scale_part, min_val_part, data_part], dim=-1).reshape(*origin_shape[:-1], -1)


def dequantize_row_q4_1(quantized: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    block_size = 32
    assert quantized.size(-1) % (block_size // 2 + 4) == 0

    origin_shape = quantized.shape
    quantized = quantized.reshape(-1, block_size // 2 + 4)

    # Extract scale and min_val from the quantized tensor
    scale_part = quantized[:, :2].contiguous()
    min_val_part = quantized[:, 2:4].contiguous()
    data_part = quantized[:, 4:].contiguous()

    # Convert back to float16 and then to float32
    scale = torch.frombuffer(scale_part.numpy().astype(np.uint8), dtype=torch.float16).reshape(-1, 1).float()
    min_val = torch.frombuffer(min_val_part.numpy().astype(np.uint8), dtype=torch.float16).reshape(-1, 1).float()

    # Calculate zero_point from min_val and scale
    zero_point = -min_val.to(torch.float32) / scale.to(torch.float32)

    # Extract the quantized data
    quant_inpt = data_part.view(-1, block_size // 2)
    quant_inpt_left_part = quant_inpt & 0x0F
    quant_inpt_right_part = quant_inpt >> 4

    # Concatenate left and right parts
    quant_inpt = torch.cat([quant_inpt_left_part, quant_inpt_right_part], dim=-1).float()

    # Dequantize
    inpt = quant_inpt.to(torch.float32) * scale.to(torch.float32) + min_val.to(torch.float32)
    return inpt.reshape(*origin_shape[:-1], -1), scale.reshape(*origin_shape[:-1],
                                                               -1), zero_point.reshape(*origin_shape[:-1], -1)


def convert_to_gguf(inpt: torch.Tensor,
                    scale: torch.Tensor,
                    zero_point: torch.Tensor,
                    gguf_type: GGMLQuantizationType = GGMLQuantizationType.Q4_1) -> torch.Tensor:
    inpt = inpt.to(torch.float32)
    scale = scale.to(torch.float32)
    zero_point = zero_point.to(torch.float32)
    if gguf_type == GGMLQuantizationType.Q4_1:
        return quantize_row_q4_1(inpt=inpt, scale=scale, zero_point=zero_point)
    else:
        raise TypeError(f"gguf_type {gguf_type} is not supported yet")


def convert_from_gguf(
        inpt: torch.Tensor,
        gguf_type: GGMLQuantizationType = GGMLQuantizationType.Q4_1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if gguf_type == GGMLQuantizationType.Q4_1:
        return dequantize_row_q4_1(quantized=inpt)
    else:
        raise TypeError(f"gguf_type {gguf_type} is not supported yet")


def build_quant_cfg(tensor_name: str,
                    gguf_type: GGMLQuantizationType = GGMLQuantizationType.Q4_1) -> Dict[str, Union[str, int]]:
    quant_cfg: Dict[str, Union[str, int]] = {}
    if gguf_type == GGMLQuantizationType.Q4_1:
        quant_cfg["scale"] = tensor_name + "_scale"
        quant_cfg["zero_point"] = tensor_name + "_zero_point"
        quant_cfg["dtype"] = "uint4"
        quant_cfg["qscheme"] = "per_group"
        quant_cfg["ch_axis"] = 0
        quant_cfg["group_size"] = 32
        quant_cfg["round_method"] = "half_even"
        quant_cfg["scale_type"] = "float"
        return quant_cfg
    else:
        raise TypeError(f"gguf_type {gguf_type} is not supported yet")


def gguf_shape(tensor_shape: List[int], gguf_type: GGMLQuantizationType = GGMLQuantizationType.Q4_1) -> List[int]:
    if gguf_type in [
            GGMLQuantizationType.F32, GGMLQuantizationType.F16, GGMLQuantizationType.F64, GGMLQuantizationType.BF16
    ]:
        return tensor_shape
    elif gguf_type == GGMLQuantizationType.Q4_1:
        output_dim = int((tensor_shape[-1] / 32) * 20)
        return [*tensor_shape[:-1], output_dim]
    else:
        raise TypeError(f"gguf_type {gguf_type} is not supported yet")
