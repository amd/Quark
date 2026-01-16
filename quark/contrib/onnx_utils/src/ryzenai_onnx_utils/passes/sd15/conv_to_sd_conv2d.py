# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from collections.abc import Sequence
from typing import Any

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    copy_attributes,
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
from ryzenai_onnx_utils.typing import PassOutputArgs


@register_whitebox_pass("SDConv")
class SDConvPass(WhiteboxBasePass):
    whitebox_flow_op_type = "Conv"
    force_whitelist = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        #   CI, YI, XI, CO, YO, XO, KY, KX
        supported_shapes = {
            "sd15": {
                # Unet
                (1280, 16, 16, 1280, 16, 16, 1, 1),  # id 1
                (1280, 16, 16, 1280, 16, 16, 3, 3),  # id 2
                (1280, 16, 16, 1280, 8, 8, 3, 3),  # id 3
                (1280, 32, 32, 1280, 32, 32, 3, 3),  # id 4
                (1280, 32, 32, 640, 32, 32, 1, 1),  # id 5
                (1280, 32, 32, 640, 32, 32, 3, 3),  # id 6
                (1280, 8, 8, 1280, 8, 8, 1, 1),  # id 7
                (1280, 8, 8, 1280, 8, 8, 3, 3),  # id 8
                (1920, 16, 16, 1280, 16, 16, 1, 1),  # id 9
                (1920, 16, 16, 1280, 16, 16, 3, 3),  # id 10
                (1920, 32, 32, 640, 32, 32, 1, 1),  # id 11
                (1920, 32, 32, 640, 32, 32, 3, 3),  # id 12
                (2560, 16, 16, 1280, 16, 16, 1, 1),  # id 13
                (2560, 16, 16, 1280, 16, 16, 3, 3),  # id 14
                (2560, 8, 8, 1280, 8, 8, 1, 1),  # id 15
                (2560, 8, 8, 1280, 8, 8, 3, 3),  # id 16
                (320, 32, 32, 640, 32, 32, 1, 1),  # id 17
                (320, 32, 32, 640, 32, 32, 3, 3),  # id 18
                (320, 64, 64, 320, 64, 64, 1, 1),  # id 19
                (320, 64, 64, 320, 64, 64, 3, 3),  # id 20
                (320, 64, 64, 320, 32, 32, 3, 3),  # id 21
                (320, 64, 64, 4, 64, 64, 3, 3),  # id 22
                (4, 64, 64, 320, 64, 64, 3, 3),  # id 23
                (640, 16, 16, 1280, 16, 16, 1, 1),  # id 24
                (640, 16, 16, 1280, 16, 16, 3, 3),  # id 25
                (640, 32, 32, 640, 32, 32, 1, 1),  # id 26
                (640, 32, 32, 640, 32, 32, 3, 3),  # id 27
                (640, 32, 32, 640, 16, 16, 3, 3),  # id 28
                (640, 64, 64, 320, 64, 64, 1, 1),  # id 29
                (640, 64, 64, 320, 64, 64, 3, 3),  # id 30
                (640, 64, 64, 640, 64, 64, 3, 3),  # id 31
                (960, 32, 32, 640, 32, 32, 1, 1),  # id 32
                (960, 32, 32, 640, 32, 32, 3, 3),  # id 33
                (960, 64, 64, 320, 64, 64, 1, 1),  # id 34
                (960, 64, 64, 320, 64, 64, 3, 3),  # id 35
                # sd2.1-v unet
                (1280, 12, 12, 1280, 12, 12, 3, 3),
                (1280, 24, 24, 1280, 12, 12, 3, 3),
                (1280, 24, 24, 1280, 24, 24, 3, 3),
                (1280, 48, 48, 1280, 48, 48, 3, 3),
                (1920, 24, 24, 1280, 24, 24, 1, 1),
                (1920, 24, 24, 1280, 24, 24, 3, 3),
                (2560, 12, 12, 1280, 12, 12, 1, 1),
                (2560, 12, 12, 1280, 12, 12, 3, 3),
                (2560, 24, 24, 1280, 24, 24, 1, 1),
                (2560, 24, 24, 1280, 24, 24, 3, 3),
                (640, 24, 24, 1280, 24, 24, 1, 1),
                (640, 24, 24, 1280, 24, 24, 3, 3),
                (320, 96, 96, 320, 48, 48, 3, 3),
                (320, 96, 96, 320, 96, 96, 3, 3),
                (4, 96, 96, 320, 96, 96, 3, 3),
                (640, 96, 96, 320, 96, 96, 1, 1),
                (640, 96, 96, 320, 96, 96, 3, 3),
                (960, 96, 96, 320, 96, 96, 1, 1),
                (960, 96, 96, 320, 96, 96, 3, 3),
                (320, 96, 96, 4, 96, 96, 3, 3),
                (1280, 48, 48, 640, 48, 48, 1, 1),
                (1280, 48, 48, 640, 48, 48, 3, 3),
                (1920, 48, 48, 640, 48, 48, 1, 1),
                (1920, 48, 48, 640, 48, 48, 3, 3),
                (320, 48, 48, 640, 48, 48, 1, 1),
                (320, 48, 48, 640, 48, 48, 3, 3),
                (640, 48, 48, 640, 24, 24, 3, 3),
                (640, 48, 48, 640, 48, 48, 3, 3),
                (640, 96, 96, 640, 96, 96, 3, 3),
                (960, 48, 48, 640, 48, 48, 1, 1),
                (960, 48, 48, 640, 48, 48, 3, 3),
                # Vae decoder
                (128, 512, 512, 128, 512, 512, 3, 3),
                (128, 512, 512, 3, 512, 512, 3, 3),
                (256, 256, 256, 256, 256, 256, 3, 3),
                (256, 512, 512, 128, 512, 512, 1, 1),
                (256, 512, 512, 128, 512, 512, 3, 3),
                (256, 512, 512, 256, 512, 512, 3, 3),
                (4, 64, 64, 4, 64, 64, 1, 1),
                (4, 64, 64, 512, 64, 64, 3, 3),
                (512, 128, 128, 512, 128, 128, 3, 3),
                (512, 256, 256, 256, 256, 256, 1, 1),
                (512, 256, 256, 256, 256, 256, 3, 3),
                (512, 256, 256, 512, 256, 256, 3, 3),
                (512, 64, 64, 512, 64, 64, 3, 3),
                # sd2.1 vae_decoder 768
                (128, 768, 768, 128, 768, 768, 3, 3),
                (256, 768, 768, 128, 768, 768, 1, 1),
                (256, 768, 768, 128, 768, 768, 3, 3),
                (256, 384, 384, 256, 384, 384, 3, 3),
                (256, 768, 768, 256, 768, 768, 3, 3),
                (512, 384, 384, 256, 384, 384, 1, 1),
                (512, 384, 384, 256, 384, 384, 3, 3),
                (128, 768, 768, 3, 768, 768, 3, 3),
                (4, 96, 96, 512, 96, 96, 3, 3),
                (4, 96, 96, 4, 96, 96, 1, 1),
                (512, 192, 192, 512, 192, 192, 3, 3),
                (512, 384, 384, 512, 384, 384, 3, 3),
                (512, 96, 96, 512, 96, 96, 3, 3),
                # controlnet
                # (4, 512, 512, 16, 512, 512, 3, 3),
                # (16, 512, 512, 16, 512, 512, 3, 3),
                # (16, 512, 512, 32, 256, 256, 3, 3), #5
                # (32, 256, 256, 32, 256, 256, 3, 3), #4
                # (32, 256, 256, 96, 128, 128, 3, 3), # 3
                (96, 128, 128, 96, 128, 128, 3, 3),  # 2
                (96, 128, 128, 256, 64, 64, 3, 3),
                (256, 64, 64, 320, 64, 64, 3, 3),
                (320, 32, 32, 320, 32, 32, 1, 1),
                (640, 16, 16, 640, 16, 16, 1, 1),  # 1
                # sdxl-base vae_decoder
                (4, 128, 128, 4, 128, 128, 1, 1),
                (4, 128, 128, 512, 128, 128, 3, 3),
                (512, 512, 512, 512, 512, 512, 3, 3),
                (512, 512, 512, 256, 512, 512, 1, 1),
                (512, 512, 512, 256, 512, 512, 3, 3),
                (256, 1024, 1024, 256, 1024, 1024, 3, 3),
                (256, 1024, 1024, 128, 1024, 1024, 1, 1),
                (256, 1024, 1024, 128, 1024, 1024, 3, 3),
                (128, 1024, 1024, 128, 1024, 1024, 3, 3),
                (128, 1024, 1024, 3, 1024, 1024, 3, 3),
                # sdxl-base unet
                (4, 128, 128, 320, 128, 128, 3, 3),
                (320, 128, 128, 320, 128, 128, 3, 3),
                (320, 128, 128, 320, 64, 64, 3, 3),
                (320, 64, 64, 640, 64, 64, 1, 1),
                (320, 64, 64, 640, 64, 64, 3, 3),
                (640, 64, 64, 640, 32, 32, 3, 3),
                (640, 32, 32, 1280, 32, 32, 1, 1),
                (640, 32, 32, 1280, 32, 32, 3, 3),
                (2560, 32, 32, 1280, 32, 32, 1, 1),
                (2560, 32, 32, 1280, 32, 32, 1, 1),
                (2560, 32, 32, 1280, 32, 32, 3, 3),
                (1920, 32, 32, 1280, 32, 32, 1, 1),
                (1920, 32, 32, 1280, 32, 32, 3, 3),
                (1280, 64, 64, 1280, 64, 64, 3, 3),
                (1920, 64, 64, 640, 64, 64, 1, 1),
                (1920, 64, 64, 640, 64, 64, 3, 3),
                (1280, 64, 64, 640, 64, 64, 1, 1),
                (1280, 64, 64, 640, 64, 64, 3, 3),
                (960, 64, 64, 640, 64, 64, 1, 1),
                (960, 64, 64, 640, 64, 64, 3, 3),
                (640, 128, 128, 640, 128, 128, 3, 3),
                (960, 128, 128, 320, 128, 128, 1, 1),
                (960, 128, 128, 320, 128, 128, 3, 3),
                (640, 128, 128, 320, 128, 128, 1, 1),
                (640, 128, 128, 320, 128, 128, 3, 3),
                (320, 128, 128, 4, 128, 128, 3, 3),
            },
            "sd3": {
                # mmdit
                (16, 64, 64, 1536, 32, 32, 2, 2),
                # mmdit 1024
                (16, 128, 128, 1536, 64, 64, 2, 2),
                (24, 128, 128, 1536, 64, 64, 2, 2),
                # vae_decoder 512
                (16, 64, 64, 512, 64, 64, 3, 3),
                (128, 512, 512, 128, 512, 512, 3, 3),
                (128, 512, 512, 3, 512, 512, 3, 3),
                (256, 256, 256, 256, 256, 256, 3, 3),
                (256, 512, 512, 128, 512, 512, 1, 1),
                (256, 512, 512, 128, 512, 512, 3, 3),
                (256, 512, 512, 256, 512, 512, 3, 3),
                (512, 128, 128, 512, 128, 128, 3, 3),
                (512, 256, 256, 256, 256, 256, 1, 1),
                (512, 256, 256, 256, 256, 256, 3, 3),
                (512, 256, 256, 512, 256, 256, 3, 3),
                (512, 64, 64, 512, 64, 64, 3, 3),
                # vae_decoder 1024
                (16, 128, 128, 512, 128, 128, 3, 3),
                (512, 512, 512, 512, 512, 512, 3, 3),
                (512, 512, 512, 256, 512, 512, 1, 1),
                (512, 512, 512, 256, 512, 512, 3, 3),
                (256, 1024, 1024, 256, 1024, 1024, 3, 3),
                (256, 1024, 1024, 128, 1024, 1024, 1, 1),
                (256, 1024, 1024, 128, 1024, 1024, 3, 3),
                (128, 1024, 1024, 128, 1024, 1024, 3, 3),
                (128, 1024, 1024, 3, 1024, 1024, 3, 3),
                # vae_encoder 512
                (4, 512, 512, 128, 512, 512, 3, 3),
                (128, 512, 512, 128, 512, 512, 3, 3),
                (128, 512, 512, 128, 256, 256, 3, 3),
                (128, 256, 256, 256, 256, 256, 1, 1),
                (128, 256, 256, 256, 256, 256, 3, 3),
                (256, 256, 256, 256, 256, 256, 3, 3),
                (256, 256, 256, 256, 128, 128, 3, 3),
                (256, 128, 128, 512, 128, 128, 1, 1),
                (256, 128, 128, 512, 128, 128, 3, 3),
                (512, 128, 128, 512, 128, 128, 3, 3),
                (512, 128, 128, 512, 64, 64, 3, 3),
                (512, 64, 64, 512, 64, 64, 3, 3),
                (512, 64, 64, 32, 64, 64, 3, 3),
                # vae encoder 1024
                (4, 1024, 1024, 128, 1024, 1024, 3, 3),
                (128, 1024, 1024, 128, 1024, 1024, 3, 3),
                (128, 1024, 1024, 128, 512, 512, 3, 3),
                (128, 512, 512, 256, 512, 512, 1, 1),
                (128, 512, 512, 256, 512, 512, 3, 3),
                (256, 256, 256, 256, 256, 256, 3, 3),
                (256, 512, 512, 256, 256, 256, 3, 3),
                (256, 256, 256, 512, 256, 256, 1, 1),
                (256, 256, 256, 512, 256, 256, 3, 3),
                (512, 256, 256, 512, 256, 256, 3, 3),
                (512, 256, 256, 512, 128, 128, 3, 3),
                (512, 128, 128, 512, 128, 128, 3, 3),
                (512, 128, 128, 32, 128, 128, 3, 3),
            },
            "phi3.5": {
                (4, 336, 336, 1024, 24, 24, 14, 14),
            },
        }

        input_shape, weight_shape, _ = check_shapes["input_shape"]
        output_shape = check_shapes["output_shape"][0]
        BI, BO, YI, XI, CI, CO, YO, XO, KY, KX = get_conv_params(input_shape, weight_shape, output_shape)
        return (CI, YI, XI, CO, YO, XO, KY, KX) in supported_shapes[op_namespace]

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
        if len(node.input) == 3:
            bias_shape = [get_attribute(node, "output_shape")[-1]]
            shape_lists["input_shape"].append(bias_shape)
        return shape_lists


# add the KY and KX to distinguish the 1x1 and 3x3 conv, not only by input and output shape
def get_conv_params(
    input_shape: Sequence[int], weight_shape: Sequence[int], output_shape: Sequence[int]
) -> tuple[int, int, int, int, int, int, int, int, int, int]:
    BI = input_shape[0]
    YI = input_shape[1]
    XI = input_shape[2]
    CI = input_shape[3]
    KY = weight_shape[1]
    KX = weight_shape[2]
    BO = output_shape[0]
    YO = output_shape[1]
    XO = output_shape[2]
    CO = output_shape[3]
    assert BI == BO
    return (BI, BO, YI, XI, CI, CO, YO, XO, KY, KX)


# This `blacklist` filters out operators with specific shapes
# that negatively impact the accuracy of SD15, and offloads them to the CPU.
def in_conv_black_list(
    op_namespace: str, input_shape: Sequence[int], weight_shape: Sequence[int], output_shape: Sequence[int]
) -> bool:
    if op_namespace != "sd15":
        return False
    black_list = {
        "sd15": {
            # controlnet
            # (4, 512, 512, 16, 512, 512, 3, 3),
            (3, 512, 512, 16, 512, 512, 3, 3),
            (16, 512, 512, 16, 512, 512, 3, 3),
            (16, 512, 512, 32, 256, 256, 3, 3),  # 5
            (32, 256, 256, 32, 256, 256, 3, 3),  # 4
            (32, 256, 256, 96, 128, 128, 3, 3),  # 3
        }
    }
    (BI, BO, YI, XI, CI, CO, YO, XO, KY, KX) = get_conv_params(input_shape, weight_shape, output_shape)
    return (CI, YI, XI, CO, YO, XO, KY, KX) in black_list[op_namespace]


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,  # params.attributes
) -> PassOutputArgs:
    domain = params.get_domain("SDConv")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    conv = subgraph[0]

    assert len(conv.output) == 1

    input_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[1], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(conv.output[0], extractor)

    if in_conv_black_list(op_namespace, input_shape, weight_shape, output_shape):
        return subgraph, [], None

    tvis = []

    pre_cast_output = conv.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
        conv.input[0],
        pre_cast_output,
        input_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(conv.input[0], extractor),
    )
    tvis.extend(pre_cast_tvi)

    new_inputs = [pre_cast_output, conv.input[1]]
    if len(conv.input) == 3:
        new_inputs.append(conv.input[2])
    conv_output = subgraph[-1].output[0] + f".out{pass_id}"
    op_type = "SDConv"
    conv_node = onnx.helper.make_node(
        op_type,
        inputs=new_inputs,
        outputs=[conv_output],
        domain=domain,
        name=conv.name,
    )
    # need a leading one because of how the kernel is implemented
    copy_attributes(conv, conv_node)
    add_attribute(conv_node, "input_shape", input_shape)
    add_attribute(conv_node, "output_shape", output_shape)
    add_attribute(conv_node, "weight_shape", weight_shape)
    add_attribute(conv_node, "in_dtypes", ["bfloat16", "bfp16ebs8", "float"])
    # Weight dtype may change to bfp16 in a following pass that does the type conversion;
    # Here we set the dtype according to current status.
    add_attribute(conv_node, "out_dtypes", ["bfloat16"])

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        subgraph[-1].output[0] + f".out{pass_id}",
        subgraph[-1].output[0],
        ryzenai_onnx_utils.matcher.get_shape(subgraph[-1].output[0], extractor),
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(conv.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast, conv_node, *post_cast], [], tvis


PATTERN = [["NhwcConv([?,?], b0)", "Reshape([b0,?],?)"], ["NhwcConv([?,?,?], ?)"]]
REPLACEMENT = [replacement] * len(PATTERN)
