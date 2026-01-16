# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import ml_dtypes
import numpy as np
import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    get_attribute,
    get_dtype,
    get_initializer_as_numpy,
    get_shape,
    get_shapes,
    has_multiple_successors,
    is_initializer,
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


@register_whitebox_pass("SDAdd")
class SDAddPass(WhiteboxBasePass):
    whitebox_flow_op_type = "Add"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                # Unet
                ((2, 64, 64, 320), (2, 1, 1, 320)),  # layer1
                ((2, 32, 32, 640), (2, 1, 1, 640)),  # layer2
                ((2, 16, 16, 1280), (2, 1, 1, 1280)),  # layer 3
                ((2, 8, 8, 1280), (2, 1, 1, 1280)),  # layer 4
                ((2, 64, 64, 320), (2, 64, 64, 320)),  # layer 5
                ((2, 32, 32, 640), (2, 32, 32, 640)),  # layer 6
                ((2, 16, 16, 1280), (2, 16, 16, 1280)),  # layer 7
                ((2, 8, 8, 1280), (2, 8, 8, 1280)),  # layer 8
                ((320,), (2, 4096, 320)),  # layer 9
                ((640,), (2, 1024, 640)),  # layer 10
                ((1280,), (2, 256, 1280)),  # layer 11
                ((1280,), (2, 64, 1280)),  # layer 12
                ((2560,), (2, 4096, 2560)),  # layer 13
                ((5120,), (2, 1024, 5120)),  # layer 14
                ((10240,), (2, 256, 10240)),  # layer 15
                ((10240,), (2, 64, 10240)),  # layer 16
                ((2, 4096, 320), (2, 4096, 320)),  # layer 17
                ((2, 1024, 640), (2, 1024, 640)),  # layer 18
                ((2, 256, 1280), (2, 256, 1280)),  # layer 19
                ((2, 64, 1280), (2, 64, 1280)),  # layer 20
                ((1280,), (2, 1280)),  # layer 21
                ((640,), (2, 640)),  # layer 22
                ((320,), (2, 320)),  # layer 23
                ((1280,), (2, 4096, 1280)),  # layer 24
                ((2560,), (2, 1024, 2560)),  # layer 25
                ((5120,), (2, 256, 5120)),  # layer 26
                ((5120,), (2, 64, 5120)),  # layer 27
                ((2, 32, 32, 320), (2, 32, 32, 320)),
                ((2, 16, 16, 640), (2, 16, 16, 640)),
                # sd2.1-v 768 unet
                ((2, 12, 12, 1280), (2, 12, 12, 1280)),
                ((2, 12, 12, 1280), (2, 1, 1, 1280)),
                ((2, 144, 1280), (2, 144, 1280)),
                ((2, 2304, 640), (2, 2304, 640)),
                ((2, 24, 24, 1280), (2, 1, 1, 1280)),
                ((2, 24, 24, 1280), (2, 24, 24, 1280)),
                ((2, 48, 48, 640), (2, 1, 1, 640)),
                ((2, 48, 48, 640), (2, 48, 48, 640)),
                ((2, 576, 1280), (2, 576, 1280)),
                ((2, 9216, 320), (2, 9216, 320)),
                ((2, 96, 96, 320), (2, 1, 1, 320)),
                ((2, 96, 96, 320), (2, 96, 96, 320)),
                # VAE decoder
                ((512,), (1, 4096, 512)),  # count: 1, bias_add
                ((1, 64, 64, 512), (1, 64, 64, 512)),  # count: 6
                ((1, 128, 128, 512), (1, 128, 128, 512)),  # count: 3
                ((1, 256, 256, 256), (1, 256, 256, 256)),  # count: 3
                ((1, 512, 512, 128), (1, 512, 512, 128)),  # count: 3
                # sd2.1 768 vae_decoder
                ((1, 192, 192, 512), (1, 192, 192, 512)),
                ((1, 384, 384, 256), (1, 384, 384, 256)),
                ((1, 768, 768, 128), (1, 768, 768, 128)),
                ((1, 96, 96, 512), (1, 96, 96, 512)),
                # sdxl_turbo
                ((2, 1280), (2, 1280)),
                # sd(xl)-turbo bs1
                ((1, 8, 8, 1280), (1, 1, 1, 1280)),
                ((1, 8, 8, 1280), (1, 8, 8, 1280)),
                ((1, 16, 16, 1280), (1, 1, 1, 1280)),
                ((1, 16, 16, 1280), (1, 16, 16, 1280)),
                ((1, 32, 32, 640), (1, 1, 1, 640)),
                ((1, 32, 32, 640), (1, 32, 32, 640)),
                ((1, 64, 64, 320), (1, 1, 1, 320)),
                ((1, 64, 64, 320), (1, 64, 64, 320)),
                ((1, 64, 1280), (1, 64, 1280)),
                ((1, 256, 1280), (1, 256, 1280)),
                ((1, 1024, 640), (1, 1024, 640)),
                ((1, 4096, 320), (1, 4096, 320)),
                ((1, 1280), (1, 1280)),
                # sdxl-base vae_decoder
                ((1, 512, 512, 256), (1, 512, 512, 256)),
                ((1, 256, 256, 512), (1, 256, 256, 512)),
                ((1, 1024, 1024, 128), (1, 1024, 1024, 128)),
                # sdxl-base unet
                ((2, 32, 32, 1280), (2, 32, 32, 1280)),
                ((2, 32, 32, 1280), (2, 1, 1, 1280)),
                ((2, 64, 64, 640), (2, 64, 64, 640)),
                ((2, 64, 64, 640), (2, 1, 1, 640)),
                ((2, 128, 128, 320), (2, 128, 128, 320)),
                ((2, 128, 128, 320), (2, 1, 1, 320)),
                ((2, 1024, 1280), (2, 1024, 1280)),
                ((2, 4096, 640), (2, 4096, 640)),
            },
            "sd3": {
                # mmdit 512
                ((1536,), (2, 1024, 1536)),
                ((2, 1024, 1536), (1, 1024, 1536)),
                ((1, 1024, 1536), (1, 1024, 1536)),
                ((2, 1024, 1536), (2, 1, 1536)),
                ((2, 1024, 1536), (2, 1024, 1536)),
                ((6144,), (2, 1024, 6144)),
                ((64,), (2, 1024, 64)),
                # mmdit 1024
                ((1536,), (2, 4096, 1536)),
                ((2, 4096, 1536), (1, 4096, 1536)),
                ((1, 4096, 1536), (1, 4096, 1536)),
                ((2, 4096, 1536), (2, 1, 1536)),
                ((2, 4096, 1536), (2, 4096, 1536)),
                ((6144,), (2, 4096, 6144)),
                ((64,), (2, 4096, 64)),
                # mmdit 512 + mmdit 1024
                ((1536,), (2, 1536)),
                ((1536,), (2, 154, 1536)),
                ((2, 1, 1536), (1536)),
                ((2, 1536), (1536,)),
                ((2, 1536), (2, 1536)),
                ((1, 1536), (1, 1536)),
                ((2, 154, 1536), (2, 1, 1536)),
                ((2, 154, 1536), (2, 154, 1536)),
                ((6144,), (2, 154, 6144)),
                # vae 512
                ((1, 64, 64, 512), (1, 64, 64, 512)),
                ((512,), (1, 4096, 512)),
                ((1, 256, 256, 256), (1, 256, 256, 256)),
                ((1, 512, 512, 128), (1, 512, 512, 128)),
                # vae 1024
                ((512,), (1, 16384, 512)),
                ((1, 256, 256, 512), (1, 256, 256, 512)),
                ((1, 512, 512, 256), (1, 512, 512, 256)),
                ((1, 1024, 1024, 128), (1, 1024, 1024, 128)),
                # vae 512 + vae 1024
                ((1, 128, 128, 512), (1, 128, 128, 512)),
                # mmdit seq-160
                ((2, 160, 1536), (2, 1, 1536)),
                ((2, 160, 1536), (2, 160, 1536)),
            },
            "phi3.5": {
                ((1, 577, 1024), (1, 577, 1024)),  # count 47
                ((1, 577, 1024), (1024,)),  # count 69
            },
        }
        a_shape = tuple(check_shapes["input_shape"][0])
        b_shape = tuple(check_shapes["input_shape"][1])
        return (a_shape, b_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                get_attribute(node, "a_shape"),
                get_attribute(node, "b_shape"),
            ],
            "output_shape": [
                get_attribute(node, "c_shape"),
            ],
        }
        return shape_lists


def is_add_supported_pattern(extractor: onnx.utils.Extractor, subgraph: list[onnx.NodeProto]):
    add_node = subgraph[0]
    if len(subgraph) == 2 and has_multiple_successors(add_node.output[0], extractor.graph):
        return False
    a_shape, b_shape = get_shapes(subgraph[0].input, extractor)
    if np.all(np.array(a_shape) == 1) or np.all(np.array(b_shape) == 1):
        return False
    return len(add_node.input) == 2 and len(add_node.output) == 1


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDAdd")
    add_node = subgraph[0]

    if not is_add_supported_pattern(extractor, subgraph):
        return subgraph, [], None

    (output_shape,) = get_shapes(subgraph[-1].output, extractor)
    a_shape = get_shape(add_node.input[0], extractor)
    b_shape = get_shape(add_node.input[1], extractor)
    c_shape = get_shape(add_node.output[0], extractor)

    tvis = []
    new_nodes = []
    new_inputs = []
    initializers: list[onnx.TensorProto] = []

    for i, inp in enumerate(add_node.input):
        if not is_initializer(inp, extractor):
            unique_index = f"_{pass_id}"
            pre_cast_output = inp + f".out{unique_index}"
            input_shape = get_shapes(add_node.input, extractor)[i]
            pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
                inp,
                pre_cast_output,
                input_shape,
                domain,
                get_dtype(inp, extractor),
            )
            tvis.extend(pre_cast_tvi)
            new_nodes.extend(pre_cast)
            new_inputs.append(pre_cast_output)
        else:
            wts_name = inp + "_bf16"
            wts_data = get_initializer_as_numpy(inp, extractor)
            wts_tvi = onnx.helper.make_tensor_value_info(wts_name, onnx.TensorProto.UINT16, wts_data.shape)
            wts_data_bf16 = wts_data.astype(ml_dtypes.bfloat16)
            wts_tensor = onnx.helper.make_tensor(
                wts_name,
                onnx.TensorProto.UINT16,
                wts_data.shape,
                wts_data_bf16.tobytes(),
                True,
            )
            tvis.append(wts_tvi)
            new_inputs.append(wts_name)
            initializers.append(wts_tensor)

    add_node_output = subgraph[-1].output[0] + f".out{unique_index}"
    if is_initializer(add_node.input[0], extractor):
        new_inputs = [new_inputs[1], new_inputs[0]]
        (a_shape, b_shape) = (b_shape, a_shape)
    elwadd_node = onnx.helper.make_node(
        "SDAdd",
        inputs=new_inputs,
        outputs=[add_node_output],
        domain=domain,
        name=add_node.name,
    )
    new_nodes.append(elwadd_node)
    add_attribute(elwadd_node, "a_shape", a_shape)
    add_attribute(elwadd_node, "b_shape", b_shape)
    add_attribute(elwadd_node, "c_shape", c_shape)
    add_attribute(elwadd_node, "in_dtypes", ["bfloat16", "bfloat16"])
    add_attribute(elwadd_node, "out_dtypes", ["bfloat16"])
    is_bias_add = bool(is_initializer(add_node.input[1], extractor) or is_initializer(add_node.input[0], extractor))
    add_attribute(elwadd_node, "is_bias_add", is_bias_add)

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        add_node_output,
        subgraph[-1].output[0],
        output_shape,
        domain,
        get_dtype(add_node.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)
    new_nodes.extend(post_cast)

    return new_nodes, initializers, tvis


REPLACEMENT = [replacement, replacement]
PATTERN = [["Add([?,?],b0)", "Reshape([b0, ?], ?)"], ["Add([?,?],?)"]]
