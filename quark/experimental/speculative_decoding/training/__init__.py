#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""EAGLE-3 cold-start training (FSDP2 trainer + LR schedule)."""

from quark.experimental.speculative_decoding.training.trainer import train

__all__ = ["train"]
