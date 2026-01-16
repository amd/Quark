# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import contextlib

__all__ = []

with contextlib.suppress(ImportError):
    from .dd import build_dd_node

    __all__.append("build_dd_node")
