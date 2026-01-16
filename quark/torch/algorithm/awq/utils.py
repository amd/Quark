#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import inspect
from typing import Any

import torch
from torch import nn


def align_attention_mask_with_input(
    module: nn.Module, kwargs: dict[str, Any], inp: torch.Tensor | None = None
) -> dict[str, Any]:
    # Extract expected parameters from the module's forward method signature
    forward_params = inspect.signature(module.forward).parameters
    # Filter kwargs to only include those accepted by the forward method
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in forward_params}

    if inp is not None:
        attn_mask = filtered_kwargs.get("attention_mask")

        # Check if attention_mask exists and its shape differs from the input tensor
        if attn_mask is not None and inp.shape != attn_mask.shape:
            input_batch = inp.shape[0]
            mask_batch = attn_mask.shape[0]

            if input_batch > mask_batch:
                # Calculate how many full repeats and how many extra patches are needed
                repeat_times = input_batch // mask_batch
                patch_times = input_batch % mask_batch

                # 1. Perform integer multiple repetition
                # Construct repeat dimensions: [repeat_times, 1, 1, ...]
                repeat_dims = [repeat_times] + [1] * (attn_mask.dim() - 1)
                new_mask = attn_mask.repeat(*repeat_dims)

                # 2. Handle the remainder: Patch using the first sample (attn_mask[0])
                if patch_times != 0:
                    # Slice the first sample [0:1] to preserve the batch dimension as 1
                    padding_base = attn_mask[0:1]
                    # Repeat the first sample to match the remaining count
                    padding_dims = [patch_times] + [1] * (padding_base.dim() - 1)
                    padding_patch = padding_base.repeat(*padding_dims)
                    # Concatenate the repeated part and the padding patch
                    new_mask = torch.cat([new_mask, padding_patch], dim=0)

                filtered_kwargs["attention_mask"] = new_mask

            elif input_batch < mask_batch:
                # Raise error if input batch is smaller than the provided mask batch
                raise ValueError(
                    f"Invalid shape detected: input shape {inp.shape} is smaller than "
                    f"attention_mask shape {attn_mask.shape}. "
                    "This indicates an abnormal shape mismatch."
                )

    return filtered_kwargs
