#
# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch

from .grad_norm import ampscaler_get_grad_norm


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self) -> None:
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self._scaler = torch.amp.GradScaler("cuda")
        else:
            self._scaler = torch.cuda.amp.GradScaler()

    def __call__(
        self,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        clip_grad: float | None = None,
        parameters: object = None,
        create_graph: bool = False,
        update_grad: bool = True,
        retain_graph: bool = False,
    ) -> torch.Tensor | None:
        self._scaler.scale(loss).backward(create_graph=create_graph, retain_graph=retain_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = ampscaler_get_grad_norm(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self) -> dict[str, object]:
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict: dict[str, object]) -> None:
        self._scaler.load_state_dict(state_dict)
