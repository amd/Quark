#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
from typing import Optional, Tuple

import torch
from torch.distributed import device_mesh

"""
Reserved code
from torch.distributed._tensor import distribute_tensor, Replicate, DTensor
"""


def e4m3fn_to_e4m3fnuz(tensor: torch.Tensor, tensor_scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    ROCM_FP8_NAN_AS_INT = -128
    scale = 2.0

    if tensor is not None and torch.version.hip is not None:
        if tensor.dtype is not torch.float16:
            assert tensor.dtype == torch.float8_e4m3fn
            tensor = tensor.to(torch.float16)

        tensor = (tensor / scale).clamp(-224.0, 224.0).to(torch.float8_e4m3fnuz)

        tensor_scale = tensor_scale * 2.0

    return tensor, tensor_scale


class TPDeviceManager:
    _tp_mesh = None
    _device = None
    _rank = None

    @property
    def tp_mesh(self) -> device_mesh.DeviceMesh | None:
        """Getter for the name attribute."""
        return TPDeviceManager._tp_mesh

    @tp_mesh.setter
    def tp_mesh(self, mesh: device_mesh.DeviceMesh) -> None:
        """Setter for the name attribute with validation."""
        if not isinstance(mesh, device_mesh.DeviceMesh):
            raise TypeError("Name must be a device_mesh.")
        TPDeviceManager._tp_mesh = mesh

    @property
    def device(self) -> torch.device | None:
        """Getter for the name attribute."""
        return TPDeviceManager._device

    @device.setter
    def device(self, device: torch.device) -> None:
        """Setter for the name attribute with validation."""
        if not isinstance(device, torch.device):
            raise TypeError("Name must be a device_mesh.")
        TPDeviceManager._device = device

    @staticmethod
    def tp_mesh_init() -> None:
        if TPDeviceManager._tp_mesh is None:
            if os.environ["RANK"] is not None and os.environ["WORLD_SIZE"] is not None:
                rank = int(os.environ["RANK"])
                device = torch.device(f"cuda:{rank}")
                num_gpus = int(os.environ["WORLD_SIZE"])
                torch.cuda.set_device(device)

                if not torch.distributed.is_initialized():
                    torch.distributed.init_process_group("nccl")

                TPDeviceManager._rank = rank
                TPDeviceManager._device = device
                TPDeviceManager._tp_mesh = device_mesh.init_device_mesh("cuda", (num_gpus,), mesh_dim_names=("tp",))
            else:
                print("tp envirement settings not found!")

    @staticmethod
    def tp_cleanup() -> None:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
