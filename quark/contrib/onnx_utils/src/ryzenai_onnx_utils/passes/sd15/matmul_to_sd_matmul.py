# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.utils
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    get_attribute,
)
from ryzenai_onnx_utils.passes.sd3.whitebox_checker import (
    WhiteboxBasePass,
    register_whitebox_pass,
)
from ryzenai_onnx_utils.transform.cast import (
    add_cast_bfloat16_to_dtype,
    add_cast_dtype_to_bfloat16,
)
from ryzenai_onnx_utils.typing import PassOutputArgs, TupleInts4


@register_whitebox_pass("SDMatMul")
class SDMatMulPass(WhiteboxBasePass):
    whitebox_flow_op_type = "MatMul"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        #   M, K, N
        supported_shapes = {
            "sd15": {
                # from unet gemm
                (2, 1, 320, 1280),
                (2, 1, 1280, 1280),
                (2, 1, 1280, 320),
                (2, 1, 1280, 640),
                # from unet matmul
                (2, 77, 768, 320),
                (2, 77, 768, 640),
                (2, 77, 768, 1280),
                (2, 4096, 320, 320),
                (2, 4096, 1280, 320),
                (2, 1024, 640, 640),
                (2, 1024, 2560, 640),
                (2, 256, 1280, 1280),
                (2, 256, 5120, 1280),
                (2, 64, 1280, 1280),
                (2, 64, 5120, 1280),
                # without matmul_add_to_matmul pass
                (2, 4096, 320, 2560),
                (2, 1024, 640, 5120),
                (2, 256, 1280, 10240),
                (2, 64, 1280, 10240),
                # from matmul_add_to_matmul pass
                (2, 4096, 320, 1280),
                (2, 1024, 640, 2560),
                (2, 256, 1280, 5120),
                (2, 64, 1280, 5120),
                # vae decoder
                (1, 4096, 512, 512),
            },
            "sd3": {
                # mmdit
                (2, 1024, 1536, 1536),
                (2, 1024, 1536, 6144),
                (2, 1024, 1536, 64),
                (2, 1024, 6144, 1536),
                (2, 1, 1536, 1536),
                (2, 154, 1536, 1536),
                (2, 154, 1536, 6144),
                (2, 154, 4096, 1536),
                (2, 154, 6144, 1536),
                (2, 1, 2048, 1536),
                (2, 1, 256, 1536),
                # mmdit 1024
                (2, 4096, 1536, 1536),
                (2, 4096, 1536, 6144),
                (2, 4096, 1536, 64),
                (2, 4096, 6144, 1536),
                # vae decoder 512
                (1, 4096, 512, 512),
                # vae decoder 1024
                (1, 16384, 512, 512),
            },
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        weight_shape = tuple(check_shapes["input_shape"][1])
        output_shape = tuple(check_shapes["output_shape"][0])
        BI, MI, KI, NI = get_matmul_params(input_shape, weight_shape, output_shape)
        return (BI, MI, KI, NI) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                get_attribute(node, "input_shape"),
                get_attribute(node, "weight_shape"),
            ],
            "output_shape": [
                get_attribute(node, "output_shape"),
            ],
        }
        return shape_lists


# add the KY and KX to distinguish the 1x1 and 3x3 conv, not only by input and output shape
def get_matmul_params(
    input_shape: tuple[int, ...], weight_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> TupleInts4:
    if len(input_shape) == 2:
        BI, KI = input_shape
        MI = 1
    elif len(input_shape) == 3:
        BI, MI, KI = input_shape
    assert len(weight_shape) == 2
    KW, NI = weight_shape
    assert KI == KW
    if len(output_shape) == 2:
        BO, NO = output_shape
        MO = 1
    elif len(output_shape) == 3:
        BO, MO, NO = output_shape
    assert len(input_shape) == len(output_shape)
    assert BI == BO
    assert MI == MO
    assert NI == NO
    return (BI, MI, KI, NI)


def is_matmul_supported_pattern(extractor: onnx.utils.Extractor, matmul_node: onnx.NodeProto) -> bool:
    if not ryzenai_onnx_utils.matcher.is_initializer(matmul_node.input[1], extractor):
        return False
    wts_shape = ryzenai_onnx_utils.matcher.get_shape(matmul_node.input[1], extractor)
    return len(wts_shape) == 2


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDMatMul")
    (matmul,) = subgraph

    assert len(matmul.input) == 2
    assert len(matmul.output) == 1

    if not is_matmul_supported_pattern(extractor, matmul):
        return subgraph, [], None

    input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[1], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.output[0], extractor)

    tvis = []

    pre_cast_output = matmul.input[0] + f".out{pass_id}"
    matmul_input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[0], extractor)
    matmul_output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.output[0], extractor)
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
        matmul.input[0],
        pre_cast_output,
        matmul_input_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(matmul.input[0], extractor),
    )
    tvis.extend(pre_cast_tvi)

    new_inputs = [pre_cast_output, matmul.input[1]]
    matmul_output = matmul.output[0] + f".out{pass_id}"
    op_type = "SDMatMul"
    matmul_node = onnx.helper.make_node(
        op_type,
        inputs=new_inputs,
        outputs=[matmul_output],
        domain=domain,
        name=matmul.name,
    )
    # need a leading one because of how the kernel is implemented
    add_attribute(matmul_node, "input_shape", input_shape)
    add_attribute(matmul_node, "output_shape", output_shape)
    add_attribute(matmul_node, "weight_shape", weight_shape)
    add_attribute(matmul_node, "in_dtypes", ["bfloat16", "bfloat16"])
    add_attribute(matmul_node, "out_dtypes", ["bfloat16"])

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        matmul.output[0] + f".out{pass_id}",
        matmul.output[0],
        matmul_output_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(matmul.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast, matmul_node, *post_cast], [], tvis


PATTERN = ["MatMul([?,?], ?)"]
REPLACEMENT = replacement
