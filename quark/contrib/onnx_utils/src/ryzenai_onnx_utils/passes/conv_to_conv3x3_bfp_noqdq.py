# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import add_attribute
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs, is_static_shape


# add the KY and KX to distinguish the 1x1 and 3x3 conv, not only by input and output shape
def _get_conv3x3_params(
    input_shape: tuple[int, ...], weight_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> tuple[int, ...]:
    XI = input_shape[1]
    YI = input_shape[2]
    CI = input_shape[3]
    KY = weight_shape[1]
    KX = weight_shape[2]
    XO = output_shape[1]
    YO = output_shape[2]
    CO = output_shape[3]
    CI_pad = CI
    if CI == 4:
        CI_pad = 16
    return (YI, XI, CI, CO, YO, XO, KY, KX, CI_pad)


def get_conv3x3_params(conv: onnx.NodeProto, extractor: onnx.utils.Extractor) -> tuple[int, ...]:
    input_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[1], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(conv.output[0], extractor)
    assert is_static_shape(input_shape) and is_static_shape(weight_shape) and is_static_shape(output_shape)
    return _get_conv3x3_params(input_shape, weight_shape, output_shape)


def is_conv3x3_padded_supported(conv: onnx.NodeProto, op_namespace: str, extractor: onnx.utils.Extractor) -> bool:
    #   CI, YI, XI, CO, YO, XO, KY, KX
    supported_shapes: dict[str, set[tuple[int, ...]]] = {
        "sdxlt": set(
            {
                # (4, 64, 64, 320, 64, 64, 3, 3),
            }
        )
    }

    YI, XI, CI, CO, YO, XO, KY, KX, CI_pad = get_conv3x3_params(conv, extractor)
    if (CI, YI, XI, CO, YO, XO, KY, KX) not in supported_shapes[op_namespace]:
        return False

    # the bias should be in the initializer
    initializers = ryzenai_onnx_utils.matcher.get_initializers(conv.input[1], extractor, False)
    return len(initializers) == 1


def is_conv3x3_supported(conv: onnx.NodeProto, op_namespace: str, extractor: onnx.utils.Extractor) -> bool:
    #   CI, YI, XI, CO, YO, XO, KY, KX
    supported_shapes = {
        "sdxlt": {
            # Unet
            (320, 64, 64, 320, 64, 64, 3, 3),
            (320, 32, 32, 640, 32, 32, 3, 3),
            (320, 64, 64, 320, 32, 32, 3, 3),
            (640, 16, 16, 1280, 16, 16, 3, 3),
            (16, 64, 64, 320, 64, 64, 3, 3),
            # (320, 64, 64, 4, 64, 64, 3, 3),
            # (4, 64, 64, 320, 64, 64, 3,3),
            (1280, 16, 16, 1280, 16, 16, 3, 3),
            (1280, 32, 32, 1280, 32, 32, 3, 3),
            (1280, 32, 32, 640, 32, 32, 3, 3),
            (1920, 16, 16, 1280, 16, 16, 3, 3),
            (1920, 32, 32, 640, 32, 32, 3, 3),
            (2560, 16, 16, 1280, 16, 16, 3, 3),
            (640, 32, 32, 640, 16, 16, 3, 3),
            (640, 32, 32, 640, 32, 32, 3, 3),
            (640, 64, 64, 640, 64, 64, 3, 3),
            (640, 64, 64, 320, 64, 64, 3, 3),
            (960, 64, 64, 320, 64, 64, 3, 3),
            (960, 32, 32, 640, 32, 32, 3, 3),
            # Vae decoder
            (4, 64, 64, 512, 64, 64, 3, 3),
            (512, 64, 64, 512, 64, 64, 3, 3),
            (512, 128, 128, 512, 128, 128, 3, 3),
            (512, 256, 256, 512, 256, 256, 3, 3),
            (512, 256, 256, 256, 256, 256, 3, 3),
            (256, 256, 256, 256, 256, 256, 3, 3),
            (256, 512, 512, 256, 512, 512, 3, 3),
            (256, 512, 512, 128, 512, 512, 3, 3),
            (128, 512, 512, 128, 512, 512, 3, 3),
            (128, 512, 512, 3, 512, 512, 3, 3),
        }
    }

    YI, XI, CI, CO, YO, XO, KY, KX, CI_pad = get_conv3x3_params(conv, extractor)
    if (CI, YI, XI, CO, YO, XO, KY, KX) not in supported_shapes[op_namespace]:
        return False

    # the bias should be in the initializer
    initializers = ryzenai_onnx_utils.matcher.get_initializers(conv.input[1], extractor, False)
    return len(initializers) == 1


# add the KY and KX to distinguish the 1x1 and 3x3 conv, not only by input and output shape
def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("IConv_noqdq")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    conv = subgraph[0]

    assert len(conv.input) == 3
    assert len(conv.output) == 1

    padded_conv = False
    if not is_conv3x3_supported(conv, op_namespace, extractor):
        if not is_conv3x3_padded_supported(conv, op_namespace, extractor):
            return subgraph, [], None
        else:
            padded_conv = True
    YI, XI, CI, CO, YO, XO, KY, KX, CI_pad = get_conv3x3_params(conv, extractor)

    tvis = []

    pre_cast_output = conv.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_to_bf16(conv.input[0], pre_cast_output, [1, YI, XI, CI], domain)
    tvis.extend(pre_cast_tvi)
    bf_to_bfp_output = pre_cast_output + f".bfp{pass_id}"
    bf_to_bfp_output_tvi = onnx.helper.make_tensor_value_info(
        bf_to_bfp_output, onnx.TensorProto.UINT8, [1, int(YI * XI * CI_pad * 9 / 8)]
    )
    tvis.extend([bf_to_bfp_output_tvi])
    bf_to_bfp_node = onnx.helper.make_node(
        "BF16_to_BFP16",
        inputs=[pre_cast_output],
        outputs=[bf_to_bfp_output],
        domain=domain,
        name=f"BF16_to_BFP16_{pass_id}",
        bfp16_tensors=[bf_to_bfp_output],
        bfp16_shape_0=[1, YI, XI, CI_pad],
    )

    new_inputs = [bf_to_bfp_output, conv.input[1], conv.input[2]]
    conv_output = conv.output[0] + f".out{pass_id}"
    if padded_conv:
        node_name = "IConv_padding_noqdq"
        conv.name += "_padding_3x3"
    else:
        node_name = "IConv_noqdq"
    conv_node = onnx.helper.make_node(
        node_name,
        inputs=new_inputs,
        outputs=[conv_output],
        domain=domain,
        name=conv.name,
        bfp16_tensors=[bf_to_bfp_output],
        bfp16_shape_0=[1, YI, XI, CI_pad],
    )
    # need a leading one because of how the kernel is implemented
    add_attribute(conv_node, "input_shape", [1, YI, XI, CI_pad])

    post_cast, post_cast_tvi = add_cast_to_float(
        conv.output[0] + f".out{pass_id}",
        conv.output[0],
        [1, YO, XO, CO],
        domain,
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast, bf_to_bfp_node, conv_node, *post_cast], [], tvis


PATTERN = ["NhwcConv([?,?,?], ?)"]
REPLACEMENT = replacement
