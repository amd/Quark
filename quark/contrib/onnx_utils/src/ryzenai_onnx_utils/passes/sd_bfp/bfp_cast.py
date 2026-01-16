# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from typing import Any

import numpy as np
import onnx

from ryzenai_onnx_utils.matcher import (
    add_attribute,
    get_attribute,
)
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    WhiteboxBasePass,
    register_whitebox_pass,
)
from ryzenai_onnx_utils.typing import PassOutputArgs, ShapeType
from ryzenai_onnx_utils.utils import float_numpy_to_bfloat_tensor


@register_whitebox_pass("SDCastBf2Bfp")
class SDCastBf2BfpPass(WhiteboxBasePass):
    whitebox_flow_op_type = "Cast"
    force_whitelist = False

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                (2, 8, 8, 1280),
                (2, 8, 8, 2560),
                (2, 8, 4096, 40),
                (2, 8, 1024, 80),
                (2, 16, 16, 1280),
                (2, 16, 16, 2560),
                (2, 32, 32, 640),
                (2, 8, 256, 160),
                (2, 64, 64, 320),
                (2, 4096, 320),
                (2, 1, 1, 320),
                (2, 1024, 640),
                (2, 256, 1280),
                (2, 1, 1, 1280),
                (2, 64, 1280),
                (2, 4096, 1280),
                (2, 32, 32, 320),
                (2, 1024, 2560),
                (2, 16, 16, 640),
                (2, 256, 5120),
                (2, 64, 5120),
                (2, 16, 16, 1920),
                (2, 32, 32, 1280),
                (2, 32, 32, 960),
                (2, 32, 32, 1920),
                (2, 64, 64, 640),
                (2, 64, 64, 960),
            },
            "sd3": {
                # bs1
                (1, 160, 4096),
                (1, 160, 1536),
                (1, 64, 64, 16),
                (1, 128, 128, 16),
                (1, 1184, 1536),
                (1, 4256, 1536),
                (1, 160, 6144),
                (1, 1024, 1536),
                (1, 4096, 1536),
                (1, 1024, 6144),
                (1, 4096, 6144),
                (1, 160, 1536),
                # bs2
                (2, 160, 4096),
                (2, 160, 1536),
                (2, 64, 64, 16),
                (2, 128, 128, 16),
                (2, 1184, 1536),
                (2, 4256, 1536),
                (2, 160, 6144),
                (2, 1024, 1536),
                (2, 4096, 1536),
                (2, 1024, 6144),
                (2, 4096, 6144),
            },
        }
        input_shape = tuple(check_shapes["input_shape"][0])

        return input_shape in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                get_attribute(node, "input_shape"),
            ],
            "output_shape": [
                get_attribute(node, "output_shape"),
            ],
        }
        return shape_lists


@register_whitebox_pass("SDCastBfp2Bf")
class SDCastBfp2BfPass(WhiteboxBasePass):
    whitebox_flow_op_type = "Cast"
    force_whitelist = False

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                (2, 4096, 320),
                (2, 64, 64, 320),
                (2, 1024, 640),
                (2, 32, 32, 640),
                (2, 256, 1280),
                (2, 16, 16, 1280),
                (2, 64, 1280),
                (2, 8, 8, 1280),
                (2, 8, 4096, 40),
                (2, 8, 1024, 80),
                (2, 8, 256, 160),
                (2, 8, 64, 160),
                (2, 4096, 1280),
                (2, 32, 32, 320),
                (2, 1024, 2560),
                (2, 16, 16, 640),
                (2, 256, 5120),
                (2, 64, 5120),
                (2, 8, 8, 2560),
                (2, 16, 16, 2560),
                (2, 16, 16, 1920),
                (2, 32, 32, 1280),
                (2, 32, 32, 1920),
                (2, 32, 32, 960),
                (2, 64, 64, 640),
                (2, 64, 64, 960),
            },
            "sd3": {
                # bs1
                (1, 160, 1536),
                (1, 1024, 1536),
                (1, 4096, 1536),
                (1, 1024, 6144),
                (1, 4096, 6144),
                (1, 160, 6144),
                (1, 24, 1184, 64),
                (1, 24, 4256, 64),
                (1, 24, 1024, 64),
                (1, 24, 4096, 64),
                # bs2
                (2, 160, 1536),
                (2, 1024, 1536),
                (2, 4096, 1536),
                (2, 160, 6144),
                (2, 1024, 6144),
                (2, 4096, 6144),
                (2, 24, 1184, 64),
                (2, 24, 4256, 64),
                (2, 24, 1024, 64),
                (2, 24, 4096, 64),
                (2, 32, 32, 2, 2, 16),
            },
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        return input_shape in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                get_attribute(node, "input_shape"),
            ],
            "output_shape": [
                get_attribute(node, "output_shape"),
            ],
        }
        return shape_lists


def add_sd_cast_bf16_to_bfp(
    input_name: str,
    output_name: str,
    shape: ShapeType,
    domain: str,
) -> PassOutputArgs:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    tvis: list[onnx.ValueInfoProto] = []

    cast_bfp_wts = f"{input_name}_bfp.wts"
    cast_wts_init = float_numpy_to_bfloat_tensor(np.zeros(64), cast_bfp_wts, True)
    inits.append(cast_wts_init)

    cast_bfp_wts_tvi = onnx.helper.make_tensor_value_info(cast_bfp_wts, onnx.TensorProto.BFLOAT16, [64])
    tvis.append(cast_bfp_wts_tvi)

    bfp_output_tvi = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.UINT8, shape)
    tvis.append(bfp_output_tvi)

    node = onnx.helper.make_node(
        "SDCastBf2Bfp",
        inputs=[input_name, cast_bfp_wts],
        outputs=[output_name],
        domain=domain,
        name=f"{input_name}_SDCastBf2Bfp_{output_name}",
    )
    add_attribute(node, "input_shape", shape)
    add_attribute(node, "output_shape", shape)
    add_attribute(node, "in_dtypes", ["bfloat16"])
    add_attribute(node, "out_dtypes", ["bfp16ebs8"])
    nodes.append(node)
    return nodes, inits, tvis


def add_sd_cast_bfp_to_bf16(
    input_name: str,
    output_name: str,
    shape: ShapeType,
    domain: str,
) -> PassOutputArgs:
    nodes: list[onnx.NodeProto] = []
    inits: list[onnx.TensorProto] = []
    tvis: list[onnx.ValueInfoProto] = []

    cast_bfp_wts = f"{input_name}_bfp.wts"
    cast_wts_init = float_numpy_to_bfloat_tensor(np.zeros(64), cast_bfp_wts, True)
    inits.append(cast_wts_init)
    cast_bfp_wts_tvi = onnx.helper.make_tensor_value_info(cast_bfp_wts, onnx.TensorProto.BFLOAT16, [64])
    tvis.append(cast_bfp_wts_tvi)

    bf16_output_tvi = onnx.helper.make_tensor_value_info(output_name, onnx.TensorProto.BFLOAT16, shape)
    tvis.append(bf16_output_tvi)
    node = onnx.helper.make_node(
        "SDCastBfp2Bf",
        inputs=[input_name, cast_bfp_wts],
        outputs=[output_name],
        domain=domain,
        name=f"{input_name}_SDCastBfp2Bf_{output_name}",
    )
    add_attribute(node, "input_shape", shape)
    add_attribute(node, "output_shape", shape)
    add_attribute(node, "in_dtypes", ["bfp16ebs8"])
    add_attribute(node, "out_dtypes", ["bfloat16"])
    nodes.append(node)
    return nodes, inits, tvis
