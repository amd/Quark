# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass aims to reduce the computation and tensor memory allocation of the
last layer or the last MatMulNBits node by pruning the last output from seq_len
-> 1.
"""

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs, SubPass


def is_logits_node(output_name: str) -> bool:
    return "logits" in output_name


def pruned_tvi(name: str, extractor: onnx.utils.Extractor) -> onnx.ValueInfoProto:
    dtype = ryzenai_onnx_utils.matcher.get_dtype(name, extractor)
    shape = ryzenai_onnx_utils.matcher.get_shape(name, extractor)
    new_shape = [shape[0], 1, shape[2]]

    new_tvi = onnx.helper.make_tensor_value_info(name, dtype, new_shape)

    return new_tvi


def prune_config(node: onnx.NodeProto, new_node: onnx.NodeProto, params: ryzenai_onnx_utils.ReplaceParams) -> None:
    if not ryzenai_onnx_utils.matcher.has_attribute(node, "prune_enable"):
        return
    ryzenai_onnx_utils.matcher.add_attribute(new_node, "prune", 1)
    if "dynamic_shape_list" not in params.attributes:
        return
    padded_dimensions = []
    for val in params.attributes["dynamic_shape_list"]:
        if "sequence_length_padded" in val:
            padded_dimensions.append(val["sequence_length_padded"])
    if padded_dimensions:
        ryzenai_onnx_utils.matcher.add_attribute(new_node, "dynamic_count", len(padded_dimensions))
        ryzenai_onnx_utils.matcher.add_attribute(new_node, "dynamic_shapes", padded_dimensions)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    prune_last_layer = params.get_bool_attr("prune_logits", False)
    prune_lm_head = "prune_logits" in params.attributes and params.attributes["prune_logits"] == "lmhead"
    prune_logits = prune_last_layer or prune_lm_head
    if not prune_logits:
        return subgraph, [], None

    if ryzenai_onnx_utils.matcher.has_attribute(subgraph[-1], "prune_enable"):
        return subgraph, [], None
    # make sure lm_head is the last node
    if not is_logits_node(subgraph[-1].output[0]):
        return subgraph, [], None
    # we're assuming that none of these nodes will remain in their original domain
    # this gets validated in hybrid_llm_prune_validate.py later
    subgraph_to_prune = subgraph if prune_last_layer else [subgraph[-1]]

    for node in subgraph_to_prune:
        ryzenai_onnx_utils.matcher.add_attribute(node, "prune_enable", 1)

    new_tvis = []
    for index, output_tvi in enumerate(extractor.graph.output):
        if not is_logits_node(output_tvi.name):
            continue
        output_tvi = pruned_tvi(output_tvi.name, extractor)

        extractor.graph.output.remove(extractor.graph.output[index])
        extractor.graph.output.insert(index, output_tvi)

        # output tvis also need to be added to the extractor vimap
        new_tvis.append(output_tvi)

        # in our current set of models, lmhead may be either directly connected
        # to the output or to one other node. If it's directly connected, then
        # updating the output tvi is sufficient. If not, we may need to create a
        # new tvi for the downstream node as well.
        if subgraph[-1].output[0] != output_tvi.name:
            downstream_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(subgraph[-1].output[0], extractor.graph)
            assert len(downstream_nodes) == 1, "Expected only one child node from lmhead"

            # If the downstream node is a Slice, we don't need to create a new
            # tvi because the slice is presumably handling the size conversion
            # as a result of pruning.
            if downstream_nodes[0].op_type != "Slice":
                new_tvi = pruned_tvi(subgraph[-1].output[0], extractor)
                new_tvis.append(new_tvi)

        break

    return subgraph, [], new_tvis


PATTERN = [
    SubPass(
        "SSMLP",
        [
            "MatMulNBits([?,?,?,?],a1621)",
            "SkipSimplifiedLayerNormalization([?,a1621,?], [a1624,?,?,a1622])",
            "MatMulNBits([a1624,?,?,?],a1629)",
            "MatMulNBits([a1624,?,?,?],a1634)",
            "Sigmoid(a1629,a1635)",
            "Mul([a1629,a1635],a1636)",
            "Mul([a1636,a1634],a1637)",
            "MatMulNBits([a1637,?,?,?,?],a1641)",
            "SkipSimplifiedLayerNormalization([a1622,a1641,?], [a1644])",
            "MatMulNBits([a1644,?,?,?],?)",
        ],
    ),
    SubPass(
        "SSGMLP",
        [
            "MatMulNBits([?,?,?,?],a1)",
            "SimplifiedLayerNormalization([a1,?],a2)",
            "SkipSimplifiedLayerNormalization([?,a2,?],[a5,?,?,a7])",
            "MatMulNBits([a5,?,?,?],a11)",
            "MatMulNBits([a5,?,?,?],a15)",
            "Gelu(a11,a16)",
            "Mul([a16,a15],a17)",
            "MatMulNBits([a17,?,?,?],a21)",
            "SimplifiedLayerNormalization([a21,?],a23)",
            "SkipSimplifiedLayerNormalization([a7,a23,?],[a24])",
            "MatMulNBits([a24,?,?,?],?)",
        ],
    ),
    SubPass("MatMulNBits", ["MatMulNBits(?, ?)"]),
    SubPass("MatMulNBitsBf", ["MatMulNBitsBf(?, ?)"]),
    SubPass("MatMul", ["MatMul(?, ?)"]),
]
REPLACEMENT = replacement
