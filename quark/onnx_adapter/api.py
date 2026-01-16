#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from .run_config import RunConfig


def onnx_adapter(
    run_config: RunConfig,
) -> None:
    # TODO: Execute all selected passes
    raise NotImplementedError("Quark CLI doesn't support programmatic API yet!")
