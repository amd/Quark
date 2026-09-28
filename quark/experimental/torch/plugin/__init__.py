#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quantization plugins for third-party modules."""

from quark.experimental.torch.plugin.fakequant_worker import QuarkFakeQuantWorker

__all__ = ["QuarkFakeQuantWorker"]
