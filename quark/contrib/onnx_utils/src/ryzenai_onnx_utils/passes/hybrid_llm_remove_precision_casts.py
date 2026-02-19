# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass replaces casts auto-inserted by ORT around operators where all non-
empty inputs and outputs have a Cast around it.
"""

import numpy as np
import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.passes import global_pass
from ryzenai_onnx_utils.typing import PassOutputArgs, PatternType

AUTO_INSERTED_CAST_PREFIX = "InsertedPrecisionFreeCast_"


@global_pass
def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    new_nodes = []

    new_tvis = []
    for node_idx, node in enumerate(extractor.graph.node):
        if node.op_type == "Cast":
            continue
        new_prefix = f"{node_idx}_"
        for idx, input_name in enumerate(node.input):
            if input_name.startswith(AUTO_INSERTED_CAST_PREFIX):
                new_name = input_name.replace(AUTO_INSERTED_CAST_PREFIX, new_prefix)
                cast_node = onnx.helper.make_node(
                    "Cast",
                    inputs=[input_name],
                    outputs=[new_name],
                    to=onnx.TensorProto.FLOAT16,
                )
                new_nodes.append(cast_node)
                new_tvis.append(
                    ryzenai_onnx_utils.matcher.build_tvi(input_name, extractor, new_name, onnx.TensorProto.FLOAT16)
                )
                node.input[idx] = new_name
        for idx, output_name in enumerate(node.output):
            if output_name.startswith(AUTO_INSERTED_CAST_PREFIX):
                new_name = output_name.replace(AUTO_INSERTED_CAST_PREFIX, new_prefix)

                initializer_name = ryzenai_onnx_utils.matcher.is_initializer_or_const(output_name, extractor)
                if initializer_name:
                    initializer = ryzenai_onnx_utils.matcher.get_initializer_or_const(output_name, extractor)
                    new_initializer = onnx.numpy_helper.from_array(
                        initializer.astype(np.float16), name=initializer_name
                    )
                    extractor.wmap[initializer_name] = new_initializer

                cast_node = onnx.helper.make_node(
                    "Cast",
                    inputs=[new_name],
                    outputs=[output_name],
                    to=onnx.TensorProto.FLOAT16,
                )
                new_nodes.append(cast_node)
                new_tvis.append(
                    ryzenai_onnx_utils.matcher.build_tvi(output_name, extractor, new_name, onnx.TensorProto.FLOAT16)
                )
                node.output[idx] = new_name
    extractor.graph.node.extend(new_nodes)
    ryzenai_onnx_utils.matcher.replace_tvis(new_tvis, extractor)


PATTERN: PatternType = []
REPLACEMENT = replacement
