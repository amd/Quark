# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass reduces computation and memory of the final MatMul by pruning
the sequence dimension from [batch, seq_len, hidden] → [batch, 1, hidden].

We do this by inserting a Gather(op) that extracts ONLY the last token
( index = -1 ) along axis=1 before the final MatMul.

Example:
    norm_out: [B, S, 2048]
         ↓ Gather(axis=1, index=-1)
    gathered: [B, 1, 2048]
         ↓ MatMul
    logits:   [B, 1, vocab]
"""

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs, SubPass


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # The subgraph matched by the pattern is a single MatMul node.
    matmul = subgraph[0]

    matmul_input_name = matmul.input[0]  # usually: normalized hidden states

    if not ryzenai_onnx_utils.matcher.is_output_edge_or_adjacent(matmul.output[0], extractor.graph):
        return subgraph, [], None

    # only apply this to ops in the original domain
    if matmul.domain == params.get_domain("MatMulNBits"):
        return subgraph, [], None

    new_tvis = []
    new_nodes = []

    # Get or create a constant initializer with value -1 (int64)
    # This will be used as indices for Gather.
    const_minus_1, const_minus_1_tvi = ryzenai_onnx_utils.matcher.get_integer_const_by_value(-1, extractor)
    if const_minus_1 is not None and const_minus_1_tvi is not None:
        new_nodes.append(const_minus_1)
        new_tvis.append(const_minus_1_tvi)

    # ---------------------------------------------------------------------
    # Insert Gather to take the LAST token along axis=1
    # ---------------------------------------------------------------------
    gather_out_name = matmul_input_name + "_last_token"
    gather_node = onnx.helper.make_node(
        "Gather",
        inputs=[matmul_input_name, "const_-1"],
        outputs=[gather_out_name],
        axis=1,
        name="Gather_LastToken",
    )

    new_nodes.append(gather_node)

    matmul_output_shape = list(ryzenai_onnx_utils.matcher.get_shape(matmul_input_name, extractor))
    matmul_output_shape[1] = 1
    gather_out_tvi = ryzenai_onnx_utils.matcher.build_tvi(
        matmul_input_name, extractor, gather_out_name, shape=matmul_output_shape
    )
    new_tvis.append(gather_out_tvi)

    # Rewire MatMul to consume Gather output instead of full sequence
    matmul.input[0] = gather_out_name

    # Add patched MatMul to output node list
    new_nodes.append(matmul)

    return new_nodes, [], new_tvis


PATTERN = [
    SubPass("MatMul", ["MatMul([?,?,?,?,?], [?])"]),
    SubPass("MatMulNBits", ["MatMulNBits([?,?,?,?,?], [?])"]),
]
REPLACEMENT = replacement
