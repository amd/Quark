# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass ensures that the final logits output is in float32 format by
either modifying an existing CastAvx node or inserting a new one after the
last MatMulNBits node (lmhead). OGA always expects float32 logits and will cast
itself if the logits are not in float32.
"""

import logging

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs

from .hybrid_llm_prune_logits import is_logits_node

_logger = logging.getLogger(__name__)


def update_output_tvi(output_name: str, extractor: onnx.utils.Extractor) -> onnx.ValueInfoProto:
    new_tvi = None
    found_index = None
    for index, output_tvi in enumerate(extractor.graph.output):
        if output_tvi.name != output_name:
            continue
        new_tvi = ryzenai_onnx_utils.matcher.build_tvi(output_name, extractor, dtype=onnx.TensorProto.FLOAT)
        found_index = index
        break
    assert new_tvi is not None, f"Output {output_name} not found in graph outputs"

    old_tvi = extractor.graph.output[found_index]
    extractor.graph.output.remove(extractor.graph.output[found_index])
    extractor.graph.output.insert(found_index, new_tvi)

    return old_tvi


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # only apply to logits node output
    last_output = subgraph[-1].output[0]
    if not (is_logits_node(last_output) and ryzenai_onnx_utils.matcher.is_output_edge(last_output, extractor.graph)):
        _logger.debug(f"Skipping casting logits: {subgraph[-1].name} is not a logits output node")
        return subgraph, [], None

    if subgraph[-1].op_type == "CastAvx":
        cast_node = subgraph[-1]
        # if we already have a CastAvx, then change it to float32
        new_cast = onnx.helper.make_node(
            "CastAvx",
            inputs=cast_node.input,
            outputs=cast_node.output,
            to=onnx.TensorProto.FLOAT,
            name=cast_node.name,
            domain=cast_node.domain,
        )

        update_output_tvi(cast_node.output[0], extractor)

        new_nodes = [subgraph[0]] if len(subgraph) == 2 else []
        new_nodes.append(new_cast)

        _logger.debug(
            f"Modified existing CastAvx {cast_node.name} from {cast_node.attribute[0].i} to FP32 for logits output"
        )

        return new_nodes, [], None

    # otherwise, we need to add a cast
    matmul = subgraph[0]

    matmul_out_dtype = ryzenai_onnx_utils.matcher.get_dtype(matmul.output[0], extractor)

    # skip adding cast if matmul output is already float32
    # can happen if lm head is on cpu/model is float32
    if matmul_out_dtype == onnx.TensorProto.FLOAT:
        return subgraph, [], None

    new_net = matmul.output[0] + ".original"
    new_matmul = onnx.helper.make_node(
        matmul.op_type,
        inputs=matmul.input,
        outputs=[new_net],
        name=matmul.name,
        domain=matmul.domain,
    )

    domain = params.get_domain("CastAvx")
    new_cast = onnx.helper.make_node(
        "CastAvx",
        inputs=[new_net],
        outputs=matmul.output,
        to=onnx.TensorProto.FLOAT,
        name=f"{new_net}_{matmul.output[0]}",
        domain=domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(matmul, new_matmul)
    old_tvi = update_output_tvi(matmul.output[0], extractor)
    old_dtype = old_tvi.type.tensor_type.elem_type
    old_shape = ryzenai_onnx_utils.matcher.get_shape(old_tvi, extractor)
    new_tvi = onnx.helper.make_tensor_value_info(new_net, old_dtype, old_shape)

    _logger.debug(f"Added new CastAvx after {matmul.name} for FP32 logits output")

    return [new_matmul, new_cast], [], [new_tvi]


PATTERN = [
    ["MatMulNBits(?, a0)", "CastAvx(a0, ?)"],
    ["MatMulNBitsBf(?, a0)", "CastAvx(a0, ?)"],
    ["MatMulNBits(?, ?)"],
    ["MatMulNBitsBf(?, ?)"],
    ["CastAvx(?, ?)"],
]
REPLACEMENT = [replacement] * len(PATTERN)
