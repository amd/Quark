# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass is used for prefill fusion where the sequence length is padded to
discrete values for execution. At the end, we need to depad the logits to get
the right output for token phase. This pass is assuming logits are being pruned
so it only passes on the last valid part of the logits. It uses a Slice to do
the depadding based on the sequence length.
"""

from typing import Any

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.transform import hybrid_llm
from ryzenai_onnx_utils.typing import PassOutputArgs


def add_pad(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    lmhead = subgraph[0]
    # lm_head output is not logics in gemma2
    if (
        not ryzenai_onnx_utils.matcher.is_output_edge(lmhead.output[0], extractor.graph)
        and "lm_head" not in lmhead.output[0]
    ):
        return subgraph, [], None

    new_nodes = []
    new_tvis = []
    new_lmhead = onnx.helper.make_node(
        lmhead.op_type,
        lmhead.input,
        ["logits_padded"],
        lmhead.name,
        domain=lmhead.domain,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(lmhead, new_lmhead)
    new_nodes.append(new_lmhead)
    logits_shape: list[Any] = list(ryzenai_onnx_utils.matcher.get_shape("logits", extractor))
    logits_shape[1] = "sequence_length_padded"
    new_tvis.append(ryzenai_onnx_utils.matcher.build_tvi("logits", extractor, name="logits_padded", shape=logits_shape))

    const_0, const_0_tvi = ryzenai_onnx_utils.matcher.get_integer_const_by_value(0, extractor)
    if const_0 is not None and const_0_tvi is not None:
        new_nodes.append(const_0)
        new_tvis.append(const_0_tvi)

    ends_name = "sequence_length"
    io_to_pad = hybrid_llm.get_input_ids_name(extractor.graph, params.attributes)
    shape_node = onnx.helper.make_node("Shape", inputs=[io_to_pad], outputs=[ends_name], start=1)
    new_nodes.append(shape_node)

    new_tvis.append(
        onnx.helper.make_tensor_value_info(
            ends_name,
            onnx.TensorProto.INT64,
            (1,),
        )
    )

    const_1, const_1_tvi = ryzenai_onnx_utils.matcher.get_integer_const_by_value(1, extractor)
    if const_1 is not None and const_1_tvi is not None:
        new_nodes.append(const_1)
        new_tvis.append(const_1_tvi)

    sequence_length_1 = onnx.helper.make_node(
        "Sub",
        inputs=["sequence_length", "const_1"],
        outputs=["sequence_length_1"],
    )
    new_nodes.append(sequence_length_1)
    new_tvis.append(
        onnx.helper.make_tensor_value_info(
            "sequence_length_1",
            onnx.TensorProto.INT64,
            (1,),
        )
    )

    slice = onnx.helper.make_node(
        "Slice",
        inputs=["logits_padded", "sequence_length_1", ends_name, "const_1", "const_1"],
        outputs=lmhead.output,
        name="logits_slice",
    )
    new_nodes.append(slice)

    return new_nodes, [], new_tvis


REPLACEMENT = add_pad
PATTERN = ["MatMulNBits([?,?,?,?,?,?], ?)"]
