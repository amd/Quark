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
from ryzenai_onnx_utils.typing import PassOutputArgs

from .hybrid_llm_prune_logits import is_logits_node, pruned_tvi


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # The subgraph matched by the pattern is a single MatMul node.
    matmul = subgraph[0]

    matmul_input_name = matmul.input[0]  # usually: normalized hidden states

    # If this MatMul is NOT actually producing a model output (not end of graph),
    # we do NOT prune it — only prune FINAL MatMul.
    if not ryzenai_onnx_utils.matcher.is_output_edge(matmul.output[0], extractor.graph):
        return subgraph, [], None

    new_tvis = []  # new ValueInfoProtos we want inserted into the graph
    new_nodes = []  # new nodes to inject (Gather, constants, patched MatMul)

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

    for index, output_tvi in enumerate(extractor.graph.output):
        if not is_logits_node(output_tvi.name):
            continue
        output_tvi = pruned_tvi(output_tvi.name, extractor)

        extractor.graph.output.remove(extractor.graph.output[index])
        extractor.graph.output.insert(index, output_tvi)

        # output tvis also need to be added to the extractor vimap
        new_tvis.append(output_tvi)

        if matmul.output[0] != output_tvi.name:
            new_tvi = pruned_tvi(matmul.output[0], extractor)
            new_tvis.append(new_tvi)

        break

    # We added:
    #   - Gather
    #   - Constant(-1) if needed
    #   - Modified MatMul
    return new_nodes, [], new_tvis


PATTERN = ["MatMul([?,?],?)"]

REPLACEMENT = replacement
