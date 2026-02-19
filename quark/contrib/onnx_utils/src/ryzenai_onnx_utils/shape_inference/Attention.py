# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher


def infer_outputs(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> None:
    # Attention operator (com.microsoft domain) has the following inputs:
    # - Input 0: input (query) - shape [batch_size, sequence_length, hidden_size] or [batch_size, sequence_length, num_heads, head_size]
    # - Input 1: weights (optional) - shape [hidden_size, 3*hidden_size] for packed QKV
    # - Input 2: bias (optional)
    # - Input 3: mask_index (optional)
    # - Input 4: past (optional) - for caching
    # - Input 5: relative_position_bias (optional)
    # - Input 6: past_sequence_length (optional)
    #
    # Output 0: output - shape [batch_size, sequence_length, hidden_size]
    # Output 1: present (optional) - for caching

    # Get the input shape (query)
    input_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[0], extractor)
    dtype = ryzenai_onnx_utils.matcher.get_dtype(node.input[0], extractor)

    # For standard Attention, output shape is same as input shape
    # [batch_size, sequence_length, hidden_size]
    if len(input_shape) == 3:
        output_shape = input_shape
    elif len(input_shape) == 4:
        # If input is [batch_size, sequence_length, num_heads, head_size]
        # Output is [batch_size, sequence_length, num_heads * head_size]
        output_shape = [input_shape[0], input_shape[1], input_shape[2] * input_shape[3]]
    else:
        # Fallback: assume output shape matches input shape
        output_shape = input_shape

    # Create TVI for the main output
    tvi = onnx.helper.make_tensor_value_info(node.output[0], dtype, output_shape)
    extractor.vimap[node.output[0]] = tvi

    # If there's a second output (present for KV caching), handle it
    if len(node.output) > 1 and node.output[1] and len(node.input) > 4 and node.input[4]:
        # Present output typically has shape [2, batch_size, num_heads, past_seq_len + seq_len, head_size]
        # For simplicity, we'll try to infer from past input if available
        past_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[4], extractor)
        present_tvi = onnx.helper.make_tensor_value_info(node.output[1], dtype, past_shape)
        extractor.vimap[node.output[1]] = present_tvi
