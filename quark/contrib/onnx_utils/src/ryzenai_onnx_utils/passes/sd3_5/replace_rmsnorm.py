# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

import operator
from functools import reduce

import numpy as np
import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_pattern(extractor, pow, reducemean, add, div, mul):
    if not ryzenai_onnx_utils.matcher.is_initializer(pow.input[1], extractor):
        return False
    pow_exp = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(pow.input[1], extractor)
    if pow_exp != np.array([2]):
        return False
    reducemean_dims = len(ryzenai_onnx_utils.matcher.get_shape(reducemean.input[0], extractor))
    reducemean_axis = onnx.helper.get_node_attr_value(reducemean, "axes")
    reducemean_keep_dims = onnx.helper.get_node_attr_value(reducemean, "keepdims")
    if reducemean_axis != [-1] and reducemean_axis != [reducemean_dims - 1]:
        return False
    if not bool(reducemean_keep_dims):
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(add.input[1], extractor):
        return False
    if pow.input[0] != div.input[0]:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(mul.input[1], extractor):
        return False
    mul_input_shape = ryzenai_onnx_utils.matcher.get_shape(mul.input[0], extractor)
    mul_init_shape = ryzenai_onnx_utils.matcher.get_shape(mul.input[1], extractor)
    if mul_input_shape[-1] != mul_init_shape[-1]:
        return False
    return reduce(operator.mul, mul_init_shape) == mul_input_shape[-1]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    if len(subgraph) == 6:
        pow, reducemean, add, sqrt, div, mul = subgraph
    elif len(subgraph) == 8:
        cast, pow, reducemean, add, sqrt, div, _, mul = subgraph
    else:
        raise ValueError(f"Unsupported subgraph length: {len(subgraph)}")
    if not is_supported_pattern(extractor, pow, reducemean, add, div, mul):
        return subgraph, [], None
    rms_norm_eps = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add.input[1], extractor)
    mul_init_shape = ryzenai_onnx_utils.matcher.get_shape(mul.input[1], extractor)
    mul_init = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul.input[1], extractor, False).reshape(
        mul_init_shape[-1]
    )
    scale_init = onnx.helper.make_tensor(
        name=f"rmsnormalization_scale_{pass_id}",
        data_type=ryzenai_onnx_utils.matcher.get_dtype(mul.input[1], extractor),
        dims=[mul_init.shape[0]],
        vals=mul_init.tolist(),
    )
    rms_norm_inputs = [subgraph[0].input[0], scale_init.name]
    rms_norm_outputs = mul.output
    rms_norm_node = onnx.helper.make_node(
        "SimplifiedLayerNormalization",
        inputs=rms_norm_inputs,
        outputs=rms_norm_outputs,
        name=f"SimplifiedLayerNormalization_{pass_id}",
        # domain=params.get_domain("RMSNormalization"),
        axis=-1,
    )
    ryzenai_onnx_utils.matcher.set_attribute(rms_norm_node, "epsilon", rms_norm_eps.tolist())
    return [rms_norm_node], [scale_init], None


PATTERN = [
    [
        "Pow([?, ?], a1)",
        "ReduceMean([a1], a2)",
        "Add([a2, ?], a3)",
        "Sqrt([a3], a4)",
        "Div([?,a4], a5)",
        "Mul([a5, ?], a6)",
    ],
    [
        "Cast([?], a0)",
        "Pow([a0, ?], a1)",
        "ReduceMean([a1], a2)",
        "Add([a2, ?], a3)",
        "Sqrt([a3], a4)",
        "Div([a0,a4], a5)",
        "Cast([a5], a6)",
        "Mul([a6, ?], a7)",
    ],
]


REPLACEMENT = [replacement] * len(PATTERN)
