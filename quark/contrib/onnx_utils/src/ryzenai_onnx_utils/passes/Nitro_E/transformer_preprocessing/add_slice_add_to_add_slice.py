# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import copy

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def is_supported_pattern(extractor, add0, add1, slice_node) -> bool:
    if not (
        len(ryzenai_onnx_utils.matcher.find_initializers_by_nodes(extractor, add0))
        and len(ryzenai_onnx_utils.matcher.find_initializers_by_nodes(extractor, add1))
    ):
        return False

    add0_init = add0.input[1] if ryzenai_onnx_utils.matcher.is_initializer(add0.input[1], extractor) else add0.input[0]
    add1_init = add1.input[1] if ryzenai_onnx_utils.matcher.is_initializer(add1.input[1], extractor) else add1.input[0]
    add0_init_shape = list(ryzenai_onnx_utils.matcher.get_shape(add0_init, extractor))
    add1_init_shape = list(ryzenai_onnx_utils.matcher.get_shape(add1_init, extractor))
    if not (any(x != 1 for x in add0_init_shape) and all(x == 1 for x in add1_init_shape)):
        return False
    add0_output_shape = list(ryzenai_onnx_utils.matcher.get_shape(add0.output[0], extractor))
    # Filter cases where Slice inputs start, end, step, or axis are not initializers.
    if not (
        all(ryzenai_onnx_utils.matcher.is_initializer(attr_input, extractor) for attr_input in slice_node.input[1:])
    ):
        return False
    slice_start = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice_node.input[1], extractor)[0]
    slice_end = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice_node.input[2], extractor)[0]
    slice_axe = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice_node.input[3], extractor)[0]
    return (
        slice_axe < len(add0_output_shape)
        and slice_start < add0_output_shape[slice_axe]
        and slice_end < add0_output_shape[slice_axe]
        and slice_end - slice_start < add0_output_shape[slice_axe]
    )


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    add0, slice1, add1, slice2, add2 = subgraph

    if not (
        is_supported_pattern(extractor, add0, add1, slice1) and is_supported_pattern(extractor, add0, add2, slice2)
    ):
        return subgraph, [], None

    add0_input, add0_init = (
        (add0.input[0], add0.input[1])
        if ryzenai_onnx_utils.matcher.is_initializer(add0.input[1], extractor)
        else (add0.input[1], add0.input[0])
    )

    add1_init = add1.input[1] if ryzenai_onnx_utils.matcher.is_initializer(add1.input[1], extractor) else add1.input[0]
    add2_init = add2.input[1] if ryzenai_onnx_utils.matcher.is_initializer(add2.input[1], extractor) else add2.input[0]

    add0_init_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add0_init, extractor)
    add1_init_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add1_init, extractor)
    add2_init_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add2_init, extractor)

    add_init_data = copy.deepcopy(add0_init_data)

    slice1_start = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice1.input[1], extractor)[0]
    slice1_end = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice1.input[2], extractor)[0]
    slice1_axe = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice1.input[3], extractor)[0]
    slices = [slice(None)] * add_init_data.ndim
    slices[slice1_axe] = slice(slice1_start, slice1_end)
    add_init_data[tuple(slices)] += add1_init_data

    slice2_start = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice2.input[1], extractor)[0]
    slice2_end = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice2.input[2], extractor)[0]
    slice2_axe = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(slice2.input[3], extractor)[0]
    slices = [slice(None)] * add_init_data.ndim
    slices[slice2_axe] = slice(slice2_start, slice2_end)
    add_init_data[tuple(slices)] += add2_init_data

    add_init_name = add0_init + f"_{pass_id}"
    add_init_shape = ryzenai_onnx_utils.matcher.get_shape(add0_init, extractor)
    add_init_dtype = ryzenai_onnx_utils.matcher.get_dtype(add0_init, extractor)
    add_init_tvi = onnx.helper.make_tensor_value_info(add_init_name, add_init_dtype, add_init_shape)
    add_init = onnx.helper.make_tensor(add_init_name, add_init_dtype, add_init_shape, add_init_data.tobytes(), True)
    new_add = onnx.helper.make_node(
        op_type="Add",
        inputs=[add0_input, add_init_name],
        outputs=add0.output,
        name=add0.name + f"_{pass_id}",
    )
    new_slice1 = onnx.helper.make_node(
        op_type="Slice",
        inputs=slice1.input,
        outputs=add1.output,
        name=slice1.name,
    )
    new_slice2 = onnx.helper.make_node(
        op_type="Slice",
        inputs=slice2.input,
        outputs=add2.output,
        name=slice2.name,
    )

    return [new_add, new_slice1, new_slice2], [add_init], [add_init_tvi]


PATTERN = [
    "Add([?,?], b0)",
    "Slice([b0,?,?,?], b1)",
    "Add([b1,?], ?)",
    "Slice([b0,?,?,?], b2)",
    "Add([b2,?], ?)",
]
REPLACEMENT = replacement
