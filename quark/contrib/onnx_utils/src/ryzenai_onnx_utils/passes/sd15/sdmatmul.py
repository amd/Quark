# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import logging

import numpy as np
import onnx
from ryzenai_dynamic_dispatch import sd

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.passes.sd15.matmul_to_sd_matmul import get_matmul_params
from ryzenai_onnx_utils.typing import PassOutputArgs

_logger = logging.getLogger(__name__)


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDMatMul")
    matmul = subgraph[0]
    matmul_input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[0], extractor)
    matmul_weight_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[1], extractor)
    matmul_output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.output[0], extractor)
    BI, MI, KI, NI = get_matmul_params(matmul_input_shape, matmul_weight_shape, matmul_output_shape)
    weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul.input[1], extractor)
    try:
        weight = weight.astype(np.float32)
        weight_bf16 = sd.matmul_to_bf16(weight, BI, MI)
    except RuntimeError as e:
        _logger.error("Weights shuffle failed: %s", e)
        return subgraph, [], []

    weight_bf16_name = matmul.input[1]
    weight_bf16_tensor = onnx.helper.make_tensor(
        weight_bf16_name,
        onnx.TensorProto.UINT16,
        weight_bf16.shape,
        weight_bf16.tobytes(),
        True,
    )
    new_initializers = [weight_bf16_tensor]

    new_node = onnx.helper.make_node(
        "SDMatMul",
        inputs=[matmul.input[0], weight_bf16_name],
        outputs=matmul.output,
        domain=domain,
        name=matmul.name,
    )
    ryzenai_onnx_utils.matcher.copy_attributes(matmul, new_node)
    return [new_node], new_initializers, []


PATTERN = ["SDMatMul([?,?], ?)"]
REPLACEMENT = replacement
