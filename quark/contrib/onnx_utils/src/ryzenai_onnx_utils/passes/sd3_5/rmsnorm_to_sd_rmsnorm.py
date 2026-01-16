# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    copy_attributes,
    get_initializer_as_numpy,
)
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype,
    add_cast_dtype_to_bfloat16,
)
from ryzenai_onnx_utils.typing import PassOutputArgs
from ryzenai_onnx_utils.utils import float_numpy_to_bfloat_tensor


def is_rmsnorm_supported_shape(op_namespace, check_shapes) -> bool:
    supported_shapes = {
        "sd3": {
            # transformer
            # ((2, 160, 24, 64), (64,)),
            # ((2, 1024, 24, 64), (64,)),
        },
    }
    input_shape = tuple(check_shapes["input_shape"][0])
    scale_shape = tuple(check_shapes["input_shape"][1])
    return (input_shape, scale_shape) in supported_shapes[op_namespace]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDRMSNorm")
    rmsnorm_node = subgraph[0]

    input_shape = ryzenai_onnx_utils.matcher.get_shape(rmsnorm_node.input[0], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(subgraph[-1].output[0], extractor)
    scale_shape = ryzenai_onnx_utils.matcher.get_shape(rmsnorm_node.input[1], extractor)
    scale_f = get_initializer_as_numpy(rmsnorm_node.input[1], extractor)
    scale = float_numpy_to_bfloat_tensor(scale_f, rmsnorm_node.input[1], True)

    new_nodes = []
    new_inputs = []
    initializers = [scale]
    tvis = []

    pre_cast_output = rmsnorm_node.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
        rmsnorm_node.input[0],
        pre_cast_output,
        input_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(rmsnorm_node.input[0], extractor),
    )
    new_nodes.extend(pre_cast)
    tvis.extend(pre_cast_tvi)
    new_inputs.extend([pre_cast_output, scale.name])
    rmsnorm_output = subgraph[-1].output[0] + f".out{pass_id}"
    sd_rmsnorm_node = onnx.helper.make_node(
        "SDRMSNorm",
        inputs=new_inputs,
        outputs=[rmsnorm_output],
        domain=domain,
        name=rmsnorm_node.name,
    )
    copy_attributes(rmsnorm_node, sd_rmsnorm_node)
    add_attribute(sd_rmsnorm_node, "input_shape", input_shape)
    add_attribute(sd_rmsnorm_node, "in_dtypes", ["bfloat16", "bfloat16", "bfloat16"])
    add_attribute(sd_rmsnorm_node, "out_dtypes", ["bfloat16"])
    add_attribute(sd_rmsnorm_node, "output_shape", output_shape)
    add_attribute(sd_rmsnorm_node, "beta_shape", scale_shape)

    new_nodes.append(sd_rmsnorm_node)

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        rmsnorm_output,
        subgraph[-1].output[0],
        output_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(rmsnorm_node.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)
    new_nodes.extend(post_cast)

    return new_nodes, initializers, tvis


PATTERN = [
    ["SimplifiedLayerNormalization([?,?], b0)", "Reshape([b0, ?], ?)"],
    ["SimplifiedLayerNormalization([?, ?], ?)"],
]
REPLACEMENT = len(PATTERN) * [replacement]
