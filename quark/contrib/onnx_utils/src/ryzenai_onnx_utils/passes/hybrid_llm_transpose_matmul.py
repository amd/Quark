# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
This pass checks if there is matmul fed by a transpose that is an initializer
Specifically targetting lm head that has not been quantized
"""

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    transpose = subgraph[0]
    matmul_old = subgraph[1]

    embedding_initializer = ryzenai_onnx_utils.matcher.get_initializers(transpose.input, extractor, False)

    if not embedding_initializer:
        # if there are not any initializers, skip it
        return subgraph, [], None

    new_initializers = []

    embedding = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(embedding_initializer[0].name, extractor)

    embedding_transposed = embedding.transpose().astype(np.float32)
    embedding_transposed = np.ascontiguousarray(embedding_transposed)

    input1_tensor_name = embedding_initializer[0].name + ".transpose.fp32.in1"

    new_initializers.append(
        onnx.helper.make_tensor(
            input1_tensor_name, onnx.TensorProto.FLOAT, embedding_transposed.shape, embedding_transposed.tobytes(), True
        )
    )

    domain = params.get_domain(matmul_old.op_type)

    matmul_new = onnx.helper.make_node(
        "MatMul",
        inputs=[matmul_old.input[0], input1_tensor_name],
        outputs=matmul_old.output,
        domain=domain,
        name=matmul_old.name,
    )

    new_nodes = [matmul_new]

    return new_nodes, new_initializers, None


PATTERN = ["Transpose(?, a1)", "MatMul([?, a1], ?)"]
REPLACEMENT = replacement
