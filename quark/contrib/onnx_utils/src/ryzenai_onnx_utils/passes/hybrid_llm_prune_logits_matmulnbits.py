# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass aims to reduce the computation and tensor memory allocation of the
last MatMulNBits node (lm_head) in LLMs by pruning the last output from seq_len
-> 1.
"""

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs

from .hybrid_llm_prune_logits import is_logits_node


def pruned_tvi(name: str, extractor: onnx.utils.Extractor) -> onnx.ValueInfoProto:
    dtype = ryzenai_onnx_utils.matcher.get_dtype(name, extractor)
    shape = ryzenai_onnx_utils.matcher.get_shape(name, extractor)
    new_shape = [shape[0], 1, shape[2]]

    new_tvi = onnx.helper.make_tensor_value_info(name, dtype, new_shape)

    return new_tvi


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    matmul = subgraph[0]

    prune_last_layer = params.get_bool_attr("prune_logits", False)
    prune_lm_head = "prune_logits" in params.attributes and params.attributes["prune_logits"] == "lmhead"
    prune_logits = prune_last_layer or prune_lm_head

    if not prune_logits or matmul.domain != params.get_domain(matmul.op_type):
        return [matmul], [], None

    # make sure lm_head is the last node
    if not is_logits_node(matmul.output[0]):
        return [matmul], [], None

    new_tvis = []
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

    ryzenai_onnx_utils.matcher.add_attribute(matmul, "prune", 1)

    return [matmul], [], new_tvis


PATTERN = [
    ["MatMulNBits(?, ?)"],
    ["MatMulNBitsBf(?, ?)"],
    ["MatMul(?, ?)"],
]
REPLACEMENT = [replacement] * len(PATTERN)
