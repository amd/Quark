# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import cast

import onnx
import ryzenai_dynamic_dispatch as dd

import ryzenai_onnx_utils.matcher


def user_pattern(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> bool:
    types = []
    for input_tensor in node.input:
        types.append(ryzenai_onnx_utils.matcher.get_dtype_str(input_tensor, extractor))
    for output_tensor in node.output:
        types.append(ryzenai_onnx_utils.matcher.get_dtype_str(output_tensor, extractor))
    op_info = dd.Metadata.OpInfo(node.name, node.op_type, [])
    return cast(bool, dd.OpBuilder.is_supported(op_info, types))
