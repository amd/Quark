# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    # Loop has the following inputs:
    # - Input 0: M (optional max trip count, can be empty string)
    # - Input 1: cond (initial loop condition, scalar bool)
    # - Input 2+: v_initial (loop-carried dependencies)
    #
    # Loop has the following outputs:
    # - Output 0+: v_final (final values of loop-carried dependencies)
    #
    # The body subgraph has:
    # - Input 0: iteration_num (scalar int64)
    # - Input 1: condition_in (scalar bool)
    # - Input 2+: v_in (loop-carried dependencies)
    # - Output 0: condition_out (scalar bool)
    # - Output 1+: v_out (updated loop-carried dependencies)

    # Get the body subgraph
    body = ryzenai_onnx_utils.matcher.get_attribute(node, "body")

    # For Loop, the output shapes are the same as the shapes of the
    # loop-carried dependency inputs (inputs 2+) or the corresponding
    # body graph outputs (outputs 1+)

    # The number of loop outputs equals the number of loop-carried dependencies
    num_loop_carried = len(node.input) - 2  # Subtract M and cond

    for i in range(num_loop_carried):
        input_idx = i + 2  # Skip M and cond
        output_idx = i

        # Try to get shape from input
        if input_idx < len(node.input) and node.input[input_idx]:
            input_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[input_idx], extractor)
            dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[input_idx], extractor)

            # The output shape is the same as the input shape for loop-carried dependencies
            output_shape = input_shape

            tvi = onnx.helper.make_tensor_value_info(node.output[output_idx], dtype, output_shape)
            extractor.vimap[node.output[output_idx]] = tvi
        elif len(body.output) > i + 1:  # body output 0 is condition_out
            # Try to infer from body graph output
            body_output = body.output[i + 1]
            if body_output.type.HasField("tensor_type"):
                dtype = body_output.type.tensor_type.elem_type
                shape = [
                    dim.dim_value if dim.HasField("dim_value") else dim.dim_param
                    for dim in body_output.type.tensor_type.shape.dim
                ]

                tvi = onnx.helper.make_tensor_value_info(node.output[output_idx], dtype, shape)
                extractor.vimap[node.output[output_idx]] = tvi
