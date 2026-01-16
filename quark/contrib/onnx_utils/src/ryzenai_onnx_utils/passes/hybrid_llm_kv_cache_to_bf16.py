# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass converts the KV cache inputs of GQA and GQO to bfloat16.
"""

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PatternType

from . import global_pass


@global_pass
def convert_kv_cache(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> None:
    enable = params.get_bool_attr("convert_kv_cache", True)
    if not enable:
        return

    supported_nodes = {
        "GQO": {"inputs": [3, 4], "outputs": [0, 1]},
        "GroupQueryAttention": {"inputs": [3, 4], "outputs": [1, 2]},
    }
    for node in extractor.graph.node:
        if node.op_type in supported_nodes:
            indices = supported_nodes[node.op_type]
            k_shape = ryzenai_onnx_utils.matcher.get_shape(node.input[indices["inputs"][0]], extractor)
            input_names = [node.input[x] for x in indices["inputs"]]
            input_tvis = []
            for kv_name in input_names:
                input_tvis.append(onnx.helper.make_tensor_value_info(kv_name, onnx.TensorProto.BFLOAT16, k_shape))
                if kv_name in extractor.vimap:
                    del extractor.vimap[kv_name]
            output_names = [node.output[x] for x in indices["outputs"]]
            output_tvis = []
            for kv_name in output_names:
                output_tvis.append(onnx.helper.make_tensor_value_info(kv_name, onnx.TensorProto.BFLOAT16, k_shape))
                if kv_name in extractor.vimap:
                    del extractor.vimap[kv_name]

            ryzenai_onnx_utils.matcher.remove_graph_inputs(input_names, extractor.graph)
            ryzenai_onnx_utils.matcher.remove_graph_outputs(output_names, extractor.graph)

            extractor.graph.input.extend(input_tvis)
            extractor.graph.output.extend(output_tvis)


REPLACEMENT = convert_kv_cache
PATTERN: PatternType = []
