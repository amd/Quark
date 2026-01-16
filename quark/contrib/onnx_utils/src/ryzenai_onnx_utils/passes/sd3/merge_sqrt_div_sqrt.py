# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import numpy as np
import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_pattern_supported(sqrt0, cast0, div0, cast1, sqrt1, extractor) -> bool:
    if not ryzenai_onnx_utils.matcher.is_initializer(sqrt0.input[0], extractor):
        return False
    for op in [sqrt0, cast0, div0, cast1]:
        if op is None:
            continue
        if ryzenai_onnx_utils.matcher.has_multiple_successors(op, extractor.graph):
            return False
    return True


def is_static_dim(c_shape, c_gather, c_slice, sqrt0, extractor):
    shape = ryzenai_onnx_utils.matcher.get_shape(c_shape.input[0], extractor)
    if c_gather is not None:
        if not ryzenai_onnx_utils.matcher.is_initializer(c_gather.input[1], extractor):
            return None
        gather_value = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(c_gather.input[1], extractor)
        gather_shape = []
        for v in gather_value:
            gather_shape.append(shape[v])
    else:
        gather_shape = shape
    if not ryzenai_onnx_utils.matcher.is_initializer(
        c_slice.input[1], extractor
    ) or not ryzenai_onnx_utils.matcher.is_initializer(c_slice.input[2], extractor):
        return None

    slice_start = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(c_slice.input[1], extractor)
    slice_end = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(c_slice.input[2], extractor)
    if len(slice_start) != 1 or len(slice_end) != 1:
        return None
    slice_shape = gather_shape[slice_start[0] : slice_end[0]]
    if not all(isinstance(dim, int) for dim in slice_shape):
        return None
    to_dtype = ryzenai_onnx_utils.matcher.get_dtype(sqrt0.output[0], extractor)
    if to_dtype == onnx.TensorProto.FLOAT16:
        return np.array(slice_shape).astype(np.float16)
    elif to_dtype == onnx.TensorProto.FLOAT:
        return np.array(slice_shape).astype(np.float32)
    else:
        return None


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    if len(subgraph) == 5:
        sqrt0, cast0, div, cast1, sqrt1 = subgraph
        if not is_pattern_supported(sqrt0, cast0, div, cast1, sqrt1, extractor):
            return subgraph, [], None
        input_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(sqrt0.input[0], extractor)
    elif len(subgraph) == 3:
        sqrt0, div, sqrt1 = subgraph
        cast0, cast1 = None, None
        if not is_pattern_supported(sqrt0, cast0, div, cast1, sqrt1, extractor):
            return subgraph, [], None
        input_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(sqrt0.input[0], extractor)
    elif len(subgraph) == 10:
        c_shape, c_gather, c_slice, _, _, sqrt0, cast0, div, cast1, sqrt1 = subgraph
        dim = is_static_dim(c_shape, c_gather, c_slice, sqrt0, extractor)
        if not dim:
            return subgraph, [], None
        input_data = dim
    elif len(subgraph) == 9:
        c_shape, c_slice, _, _, sqrt0, cast0, div, cast1, sqrt1 = subgraph
        dim = is_static_dim(c_shape, None, c_slice, sqrt0, extractor)
        if not dim:
            return subgraph, [], None
        input_data = dim
    elif len(subgraph) == 8:
        if subgraph[4].op_type == "Sqrt":
            c_shape, c_gather, c_slice, _, sqrt0, div, sqrt1, _ = subgraph
            dim = is_static_dim(c_shape, c_gather, c_slice, sqrt0, extractor)
            cast0, cast1 = None, None
            if not dim:
                return subgraph, [], None
            input_data = dim
        elif subgraph[4].op_type == "MemcpyFromHost":
            c_shape, c_gather, c_slice, _, _, sqrt0, div, sqrt1 = subgraph
            dim = is_static_dim(c_shape, c_gather, c_slice, sqrt0, extractor)
            cast0, cast1 = None, None
            if not dim:
                return subgraph, [], None
            input_data = dim
    elif len(subgraph) == 7:
        c_shape, c_slice, cast0, c_memcpy, sqrt0, div, sqrt1 = subgraph
        dim = is_static_dim(c_shape, None, c_slice, sqrt0, extractor)
        cast1 = None
        if not dim:
            return subgraph, [], None
        input_data = dim

    sqrt0_output = np.sqrt(input_data)
    if cast0:
        cast0_output_dtype = ryzenai_onnx_utils.matcher.get_dtype(cast0.output[0], extractor)
        cast0_numpy_dtype = onnx.helper.tensor_dtype_to_np_dtype(cast0_output_dtype)
        cast0_output = sqrt0_output.astype(cast0_numpy_dtype)
    else:
        cast0_output = sqrt0_output
    div_input0 = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(div.input[0], extractor)
    div_output = np.divide(div_input0, cast0_output)
    if cast1:
        cast1_output_dtype = ryzenai_onnx_utils.matcher.get_dtype(cast1.output[0], extractor)
        cast1_numpy_dtype = onnx.helper.tensor_dtype_to_np_dtype(cast1_output_dtype)
        cast1_output = div_output.astype(cast1_numpy_dtype)
    else:
        cast1_output_dtype = ryzenai_onnx_utils.matcher.get_dtype(div.output[0], extractor)
        cast1_output = div_output
    sqrt1_output = np.sqrt(cast1_output)
    input_name = sqrt0.input[0]
    input_tvi = onnx.helper.make_tensor_value_info(input_name, cast1_output_dtype, sqrt1_output.shape)
    input_init = onnx.helper.make_tensor(
        input_name, cast1_output_dtype, sqrt1_output.shape, sqrt1_output.tobytes(), True
    )
    sqrt1_fanouts = ryzenai_onnx_utils.matcher.find_nodes_by_input(subgraph[-1].output[0], extractor.graph)
    for fanout in sqrt1_fanouts:
        fanout.input[1] = input_name
    return [], [input_init], [input_tvi]


PATTERN = [
    [
        "Sqrt([?], b0)",
        "Cast([b0], b1)",
        "Div([?,b1], b2)",
        "Cast([b2], b3)",
        "Sqrt([b3], ?)",
    ],
    ["Sqrt([?], b0)", "Div([?,b0], b1)", "Sqrt([b1], ?)"],
    [
        "Shape([?], b0)",
        "Gather([b0,?],b1)",
        "Slice([b1,?,?],b2)",
        "Cast([b2],b3)",
        "MemcpyFromHost([b3],b4)",
        "Sqrt([b4], b5)",
        "Cast([b5], b6)",
        "Div([?,b6], b7)",
        "Cast([b7], b8)",
        "Sqrt([b8], ?)",
    ],
    [
        "Shape([?], b0)",
        "Gather([b0,?],b1)",
        "Slice([b1,?,?],b2)",
        "Cast([b2],b3)",
        "MemcpyFromHost([b3],b4)",
        "Sqrt([b4], b5)",
        "Div([?,b5], b6)",
        "Sqrt([b6], ?)",
    ],
    [
        "Shape([?], b0)",
        "Gather([b0,?],b1)",
        "Slice([b1,?,?],b2)",
        "Cast([b2],b3)",
        "Sqrt([b3], b4)",
        "Div([?,b4], b5)",
        "Sqrt([b5], b6)",
        "MemcpyFromHost([b6],b7)",
    ],
    [
        "Shape([?], b0)",
        "Slice([b0,?,?],b1)",
        "Cast([b1],b2)",
        "MemcpyFromHost([b2],b3)",
        "Sqrt([b3], b4)",
        "Div([?,b4], b5)",
        "Sqrt([b5], ?)",
    ],
    [
        "Shape([?], b0)",
        "Slice([b0,?,?],b1)",
        "Cast([b1],b2)",
        "MemcpyFromHost([b2],b3)",
        "Sqrt([b3], b4)",
        "Cast([b4],b5)",
        "Div([?,b5], b6)",
        "Cast([b6],b7)",
        "Sqrt([b7], ?)",
    ],
]


REPLACEMENT = [replacement] * len(PATTERN)
