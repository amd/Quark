# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import add_attribute, copy_attributes
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs, is_static_shape


def _get_gemm_params(
    input_shape: tuple[int, ...], weight_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> tuple[int, int, int]:
    XI = input_shape[0]
    YI = input_shape[1]
    KY = weight_shape[0]
    KX = weight_shape[1]
    # XO = output_shape[0]
    YO = output_shape[1]
    assert YI == KX
    assert KY == YO
    return (XI, YI, YO)


def get_gemm_params(gemm: onnx.NodeProto, extractor: onnx.utils.Extractor) -> tuple[int, int, int]:
    input_shape = ryzenai_onnx_utils.matcher.get_shape(gemm.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(gemm.input[1], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(gemm.output[0], extractor)
    assert is_static_shape(input_shape) and is_static_shape(weight_shape) and is_static_shape(output_shape)
    return _get_gemm_params(input_shape, weight_shape, output_shape)


def is_gemm_padded_supported(gemm: onnx.NodeProto, op_namespace: str, extractor: onnx.utils.Extractor) -> bool:
    #   CI, YI, XI, CO, YO, XO, KY, KX
    supported_shapes = {
        "sdxlt": {
            (1, 320, 1280),
            (1, 1280, 1280),
            (1, 2816, 1280),
            (1, 1280, 320),
            (1, 1280, 640),
        }
    }

    XI, YI, YO = get_gemm_params(gemm, extractor)
    if (XI, YI, YO) not in supported_shapes[op_namespace]:
        return False

    # the bias should be in the initializer
    initializers = ryzenai_onnx_utils.matcher.get_initializers(gemm.input[1], extractor, False)
    return len(initializers) == 1


def is_gemm_supported(gemm: onnx.NodeProto, op_namespace: str, extractor: onnx.utils.Extractor) -> bool:
    #   CI, YI, XI, CO, YO, XO, KY, KX
    supported_shapes = {
        "sdxlt": {
            (64, 320, 1280),
            (64, 1280, 1280),
            (64, 2816, 1280),
            (64, 1280, 320),
            (64, 1280, 640),
        }
    }

    XI, YI, YO = get_gemm_params(gemm, extractor)
    if (XI, YI, YO) not in supported_shapes[op_namespace]:
        return False

    # the bias should be in the initializer
    initializers = ryzenai_onnx_utils.matcher.get_initializers(gemm.input[1], extractor, False)
    return len(initializers) == 1


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    gemm = subgraph[0]
    last_us = subgraph[2]
    # is_padded = False
    domain = params.get_domain("MatMul_noqdq")
    op_namespace = params.get_subgraph_op_namespace(subgraph)

    if not is_gemm_supported(gemm, op_namespace, extractor) and not is_gemm_padded_supported(
        gemm, op_namespace, extractor
    ):
        return subgraph, [], None

    assert len(gemm.input) == 3  # there's the redundant qkv_bias in the input as well
    assert len(gemm.output) == 1

    input_shape = ryzenai_onnx_utils.matcher.get_shape(gemm.input[0], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(last_us.output[0], extractor)
    np_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(gemm.input[1], extractor)
    np_f = np.transpose(np_f)
    np_bf = np_f.astype("float")
    weights = onnx.helper.make_tensor(gemm.input[1], onnx.TensorProto.FLOAT, np_bf.shape, np_bf)

    initializers = [weights]

    tvis = []

    pre_cast_output_q = gemm.input[0] + f":.out{pass_id}"
    pre_cast_q, pre_cast_tvi_q = add_cast_to_bf16(gemm.input[0], pre_cast_output_q, input_shape, domain)
    tvis.extend(pre_cast_tvi_q)

    new_inputs = [pre_cast_output_q, gemm.input[1], gemm.input[2]]
    gemm_output = last_us.output[0] + f".out{pass_id}"
    nm = gemm.name
    # if is_padded:
    #    nm = gemm.name + "_padding"
    gemm_node = onnx.helper.make_node(
        "MatMul_noqdq",
        inputs=new_inputs,
        outputs=[gemm_output],
        domain=domain,
        name=nm,
    )
    copy_attributes(gemm, gemm_node)
    add_attribute(gemm_node, "input_shape", [input_shape[0], input_shape[1]])

    post_cast, post_cast_tvi = add_cast_to_float(
        last_us.output[0] + f".out{pass_id}",
        last_us.output[0],
        output_shape,
        domain,
    )
    tvis.extend(post_cast_tvi)

    return (
        [*pre_cast_q, gemm_node, *post_cast],
        initializers,
        tvis,
    )


PATTERN = ["Gemm([?,?,?], a0)", "Unsqueeze(a0, b0)", "Unsqueeze(b0, ?)"]
REPLACEMENT = replacement
