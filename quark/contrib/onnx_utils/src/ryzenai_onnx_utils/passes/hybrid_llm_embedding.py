# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces Gather nodes used for embeddings with a custom LUT op that
can be memory-mapped for reduced memory usage.
"""

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.hybrid_llm
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    old_embedding = subgraph[0]
    assert old_embedding.op_type == "Gather"

    embedding_intializer = ryzenai_onnx_utils.matcher.get_initializers(old_embedding.input, extractor, False)

    if not embedding_intializer:
        # if there are not any initializers, skip it
        return subgraph, [], None

    if len(embedding_intializer[0].dims) != 2:
        # Expect LUT to be for embedding to be rank 2 with dims [vocab_size, embedding_dim]
        return subgraph, [], None

    [vocab_size, embedding_dim] = embedding_intializer[0].dims

    domain = params.get_domain("LUT")

    nodes = []
    tvis: list[onnx.ValueInfoProto] = []

    new_gather = onnx.helper.make_node(
        "LUT",
        inputs=old_embedding.input,
        outputs=old_embedding.output,
        name=old_embedding.name,
        domain=domain,
    )

    vocab_size_attr = onnx.helper.make_attribute("vocab_size", vocab_size)
    embedding_dim_attr = onnx.helper.make_attribute("embedding_dim", embedding_dim)
    new_gather.attribute.extend([vocab_size_attr, embedding_dim_attr])

    nodes.append(new_gather)

    return nodes, [], tvis


REPLACEMENT = replacement
PATTERN = ["Gather([?,?], ?)"]
