# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass finds casts named with `hybrid_llm_0` or `hybrid_llm_1` suffixes and
folds the casts into the preceding or succeeding node respectively if it's a
custom op. It also adds attributes to the custom ops to indicate which inputs/
outputs need to be casted. The original dtype of the graph, as if the cast never
existed, is preserved.
"""

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.cast
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    cast = subgraph[0]

    if not (cast.name.endswith(".hybrid_llm_0") or cast.name.endswith(".hybrid_llm_1")):
        return subgraph, [], None

    if cast.name.endswith(".hybrid_llm_0"):
        dtype = ryzenai_onnx_utils.matcher.get_dtype(cast.input[0], extractor)
        assert dtype == onnx.TensorProto.FLOAT16
        nodes_to_modify = ryzenai_onnx_utils.matcher.find_nodes_by_input(cast.output[0], extractor.graph)
        for node_to_modify in nodes_to_modify:
            assert node_to_modify.domain == params.get_domain(node_to_modify.op_type)
            index = list(node_to_modify.input).index(cast.output[0])

            ryzenai_onnx_utils.matcher.append_value_in_attribute(node_to_modify, "hybrid_llm_cast_input", index)

        # rewrite the inputs and outputs of the adjacent nodes so they're directly
        # connected as if the cast never existed. This is necessary to preserve
        # the float16 type in the intermediate nodes of the model. Otherwise,
        # they get replaced by bfloat16 types
        next_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_input(cast.output[0], extractor.graph)

        for next_node in next_nodes:
            for index, input_name in enumerate(next_node.input):
                if input_name == cast.output[0]:
                    next_node.input[index] = cast.input[0]
    else:
        dtype = ryzenai_onnx_utils.matcher.get_dtype(cast.output[0], extractor)
        assert dtype == onnx.TensorProto.FLOAT16
        nodes_to_modify = ryzenai_onnx_utils.matcher.find_nodes_by_output(cast.input[0], extractor.graph)

        for node_to_modify in nodes_to_modify:
            node_to_modify = nodes_to_modify[0]
            assert node_to_modify.domain == params.get_domain(node_to_modify.op_type)

            index = list(node_to_modify.output).index(cast.input[0])

            ryzenai_onnx_utils.matcher.append_value_in_attribute(node_to_modify, "hybrid_llm_cast_output", index)

        # rewrite the inputs and outputs of the adjacent nodes so they're directly
        # connected as if the cast never existed. This is necessary to preserve
        # the float16 type in the intermediate nodes of the model. Otherwise,
        # they get replaced by bfloat16 types
        prev_nodes = ryzenai_onnx_utils.matcher.find_nodes_by_output(cast.input[0], extractor.graph)

        for prev_node in prev_nodes:
            for index, output_name in enumerate(prev_node.output):
                if output_name == cast.input[0]:
                    prev_node.output[index] = cast.output[0]

    # return None to not do input/output rewriting since we do it manually
    return None, [], None


PATTERN = [["Cast(?, ?)"], ["CastAvx(?, ?)"]]
REPLACEMENT = [replacement] * 2
