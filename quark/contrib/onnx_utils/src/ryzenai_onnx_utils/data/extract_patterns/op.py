# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnx


def op(name: str, node: onnx.NodeProto, _extractor: onnx.utils.Extractor) -> bool:
    return node.op_type == name
