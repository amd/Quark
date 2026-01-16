# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.matcher import add_attribute
from ryzenai_onnx_utils.transform.cast import add_cast_to_bf16, add_cast_to_float
from ryzenai_onnx_utils.typing import PassOutputArgs, is_static_shape


# add the KY and KX to distinguish the 1x1 and 3x3 conv, not only by input and output shape
def _get_conv1x1_params(
    input_shape: tuple[int, ...], weight_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> tuple[int, ...]:
    XI = input_shape[1]
    YI = input_shape[2]
    CI = input_shape[3]
    KY = weight_shape[1]
    KX = weight_shape[2]
    CO = output_shape[3]
    CI_pad = CI
    if CI == 4:
        CI_pad = 256
    return (YI, XI, CI, CO, KY, KX, CI_pad)


def get_conv1x1_params(conv: onnx.NodeProto, extractor: onnx.utils.Extractor) -> tuple[int, ...]:
    input_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(conv.input[1], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(conv.output[0], extractor)
    assert is_static_shape(input_shape) and is_static_shape(weight_shape) and is_static_shape(output_shape)
    return _get_conv1x1_params(input_shape, weight_shape, output_shape)


def is_conv1x1_padded_supported(conv: onnx.NodeProto, op_namespace: str, extractor: onnx.utils.Extractor) -> bool:
    supported_shapes: dict[str, set[tuple[int, ...]]] = {
        "sdxlt": set(
            {
                # (4096, 4, 4, 1, 1),
            }
        )
    }

    YI, XI, CI, CO, KY, KX, CI_pad = get_conv1x1_params(conv, extractor)
    M = YI * XI
    if (M, CI, CO, KY, KX) not in supported_shapes[op_namespace]:
        return False
    # the bias should be in the initializer
    initializers = ryzenai_onnx_utils.matcher.get_initializers(conv.input[1], extractor, False)
    return len(initializers) == 1


# add the KY and KX to distinguish the 1x1 and 3x3 conv, not only by input and output shape
def is_conv1x1_supported(conv: onnx.NodeProto, op_namespace: str, extractor: onnx.utils.Extractor) -> bool:
    supported_shapes = {
        "sdxlt": {
            # Unet
            (256, 640, 1280, 1, 1),
            (256, 1920, 1280, 1, 1),
            (256, 2560, 1280, 1, 1),
            (4096, 960, 320, 1, 1),
            (4096, 640, 320, 1, 1),
            (1024, 1280, 640, 1, 1),
            (1024, 1920, 640, 1, 1),
            (1024, 320, 640, 1, 1),
            (1024, 960, 640, 1, 1),
            # Vae decoder
            (65536, 512, 256, 1, 1),
            (262144, 256, 128, 1, 1),
        }
    }

    YI, XI, CI, CO, KY, KX, CI_pad = get_conv1x1_params(conv, extractor)
    M = YI * XI
    if (M, CI, CO, KY, KX) not in supported_shapes[op_namespace]:
        return False
    # the bias should be in the initializer
    initializers = ryzenai_onnx_utils.matcher.get_initializers(conv.input[1], extractor, False)
    return len(initializers) == 1


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("Conv1x1_noqdq")
    op_namespace = params.get_subgraph_op_namespace(subgraph)
    conv = subgraph[0]

    assert len(conv.input) == 3
    assert len(conv.output) == 1

    padded_conv = False
    if not is_conv1x1_supported(conv, op_namespace, extractor):
        if not is_conv1x1_padded_supported(conv, op_namespace, extractor):
            return subgraph, [], None
        else:
            padded_conv = True

    YI, XI, CI, CO, KY, KX, CI_pad = get_conv1x1_params(conv, extractor)

    tvis = []

    pre_cast_output = conv.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_to_bf16(conv.input[0], pre_cast_output, [1, YI, XI, CI], domain)
    tvis.extend(pre_cast_tvi)

    bfp_inputs = [pre_cast_output]
    bfp_output = conv.output[0] + f".out{pass_id}.bfp16"
    input_tvi = onnx.helper.make_tensor_value_info(pre_cast_output, onnx.TensorProto.BFLOAT16, [1, YI, XI, CI])
    output_tvi = onnx.helper.make_tensor_value_info(
        bfp_output, onnx.TensorProto.UINT8, [1, int(YI * XI * CI_pad / 8 * 9)]
    )
    bfp_tvi = [input_tvi, output_tvi]
    tvis.extend(bfp_tvi)
    bfp16_node = onnx.helper.make_node(
        "BF16_to_BFP16",
        inputs=bfp_inputs,
        outputs=[bfp_output],
        domain=domain,
        name=f"bf16_to_bfp16_{pass_id}",
        bfp16_tensors=[bfp_output],
        bfp16_shape_0=[1, YI, XI, CI_pad],
    )

    new_inputs = [bfp_output, conv.input[1], conv.input[2]]
    conv_output = conv.output[0] + f".out{pass_id}"
    if padded_conv:
        conv.name += "_padding_1x1"
        node_name = "Conv1x1_padding_noqdq"
    else:
        node_name = "Conv1x1_noqdq"
    conv_node = onnx.helper.make_node(
        node_name,
        inputs=new_inputs,
        outputs=[conv_output],
        domain=domain,
        name=conv.name,
        bfp16_tensors=[bfp_output],
        bfp16_shape_0=[1, YI, XI, CI_pad],
    )
    # need a leading one because of how the kernel is implemented
    add_attribute(conv_node, "input_shape", [1, YI, XI, CI_pad])

    post_cast, post_cast_tvi = add_cast_to_float(
        conv.output[0] + f".out{pass_id}",
        conv.output[0],
        [1, YI, XI, CO],
        domain,
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast, bfp16_node, conv_node, *post_cast], [], tvis


PATTERN = ["NhwcConv([?,?,?], ?)"]
REPLACEMENT = replacement
