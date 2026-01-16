# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import numpy as np
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
from ryzenai_onnx_utils.utils import float_numpy_to_bfloat_tensor


@register_whitebox_pass("SDGroupNorm")
class SDGroupNormPass(WhiteboxBasePass):
    whitebox_flow_op_type = "GroupNormalization"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                # unet
                ((2, 64, 64, 320), (320,), (320,)),  # count 13
                ((2, 32, 32, 320), (320,), (320,)),  # count 1
                ((2, 32, 32, 640), (640,), (640,)),  # count 11
                ((2, 16, 16, 640), (640,), (640,)),  # count 1
                ((2, 16, 16, 1280), (1280,), (1280,)),  # count 11
                ((2, 8, 8, 1280), (1280,), (1280,)),  # count 12
                ((2, 8, 8, 2560), (2560,), (2560,)),  # count 3
                ((2, 16, 16, 2560), (2560,), (2560,)),  # count 2
                ((2, 16, 16, 1920), (1920,), (1920,)),  # count 1
                ((2, 32, 32, 1920), (1920,), (1920,)),  # count 1
                ((2, 32, 32, 1280), (1280,), (1280,)),  # count 1
                ((2, 32, 32, 960), (960,), (960,)),  # count 1
                ((2, 64, 64, 960), (960,), (960,)),  # count 1
                ((2, 64, 64, 640), (640,), (640,)),  # count 2
                # sd2.1-v unet
                ((2, 12, 12, 1280), (1280,), (1280,)),
                ((2, 12, 12, 2560), (2560,), (2560,)),
                ((2, 24, 24, 1280), (1280,), (1280,)),
                ((2, 24, 24, 1920), (1920,), (1920,)),
                ((2, 24, 24, 2560), (2560,), (2560,)),
                ((2, 24, 24, 640), (640,), (640,)),
                ((2, 48, 48, 1280), (1280,), (1280,)),
                ((2, 48, 48, 1920), (1920,), (1920,)),
                ((2, 48, 48, 320), (320,), (320,)),
                ((2, 48, 48, 640), (640,), (640,)),
                ((2, 48, 48, 960), (960,), (960,)),
                ((2, 96, 96, 320), (320,), (320,)),
                ((2, 96, 96, 640), (640,), (640,)),
                ((2, 96, 96, 960), (960,), (960,)),
                # vae_decoder
                ((1, 64, 64, 512), (512,), (512,)),  # count 11
                ((1, 128, 128, 512), (512,), (512,)),  # count 6
                ((1, 256, 256, 512), (512,), (512,)),  # count 1
                ((1, 256, 256, 256), (256,), (256,)),  # count 5
                ((1, 512, 512, 256), (256,), (256,)),  # count 1
                ((1, 512, 512, 128), (128,), (128,)),  # count 6
                # sd2.1 vae_decoder
                ((1, 192, 192, 512), (512,), (512,)),
                ((1, 384, 384, 256), (256,), (256,)),
                ((1, 384, 384, 512), (512,), (512,)),
                ((1, 768, 768, 128), (128,), (128,)),
                ((1, 768, 768, 256), (256,), (256,)),
                ((1, 96, 96, 512), (512,), (512,)),
                # sd(xl)-turbo bs1
                ((1, 64, 64, 320), (320,), (320,)),
                ((1, 32, 32, 320), (320,), (320,)),
                ((1, 32, 32, 640), (640,), (640,)),
                ((1, 16, 16, 640), (640,), (640,)),
                ((1, 16, 16, 1280), (1280,), (1280,)),
                ((1, 8, 8, 1280), (1280,), (1280,)),
                ((1, 8, 8, 2560), (2560,), (2560,)),
                ((1, 16, 16, 2560), (2560,), (2560,)),
                ((1, 16, 16, 1920), (1920,), (1920,)),
                ((1, 32, 32, 1920), (1920,), (1920,)),
                ((1, 32, 32, 1280), (1280,), (1280,)),
                ((1, 32, 32, 960), (960,), (960,)),
                ((1, 64, 64, 960), (960,), (960,)),
                ((1, 64, 64, 640), (640,), (640,)),
                # sdxl-base vae_decoder
                ((1, 512, 512, 512), (512,), (512,)),
                ((1, 1024, 1024, 256), (256,), (256,)),
                ((1, 1024, 1024, 128), (128,), (128,)),
                # sdxl-base unet
                ((2, 32, 32, 2560), (2560,), (2560,)),
                ((2, 64, 64, 1280), (1280,), (1280,)),
                ((2, 64, 64, 1920), (1920,), (1920,)),
                ((2, 128, 128, 320), (320,), (320,)),
                ((2, 128, 128, 640), (640,), (640,)),
                ((2, 128, 128, 960), (960,), (960,)),
            },
            "sd3": {
                # vae 512
                ((1, 64, 64, 512), (512,), (512,)),  # count 11
                ((1, 256, 256, 256), (256,), (256,)),  # count 5
                ((1, 512, 512, 128), (128,), (128,)),  # count 6
                # vae 1024
                ((1, 512, 512, 512), (512,), (512,)),  # count 1
                ((1, 1024, 1024, 256), (256,), (256,)),  # count 1
                ((1, 1024, 1024, 128), (128,), (128,)),  # count 6
                # vae 512 + vae 1024
                ((1, 128, 128, 512), (512,), (512,)),  # count 6/11
                ((1, 256, 256, 512), (512,), (512,)),  # count 1/6
                ((1, 512, 512, 256), (256,), (256,)),  # count 1/5
                # vae encoder 512
                ((1, 256, 256, 128), (128,), (128,)),  # count 1
                ((1, 128, 128, 256), (256,), (256,)),  # count 1
            },
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        gamma_shape = tuple(check_shapes["input_shape"][1])
        beta_shape = tuple(check_shapes["input_shape"][2])
        return (input_shape, gamma_shape, beta_shape) in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                get_attribute(node, "input_shape"),
                get_attribute(node, "gamma_shape"),
                get_attribute(node, "beta_shape"),
            ],
            "output_shape": [
                get_attribute(node, "output_shape"),
            ],
        }
        return shape_lists


def is_groupnorm_supported_pattern(extractor: onnx.utils.Extractor, groupnorm_node: onnx.NodeProto) -> bool:
    activation: str | int = get_attribute(groupnorm_node, "activation", 0)
    channel_last: int = get_attribute(groupnorm_node, "channels_last", 1)

    return not (activation != 0 or (channel_last != 1 and channel_last != -1))


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDGroupNorm")
    groupnorm = subgraph[0]
    assert len(groupnorm.input) == 3
    assert len(groupnorm.output) == 1

    if not is_groupnorm_supported_pattern(extractor, groupnorm):
        return subgraph, [], None

    nonlinear = "Silu" if len(subgraph) == 3 and subgraph[1].op_type == "Sigmoid" else ""
    input_shape = ryzenai_onnx_utils.matcher.get_shape(groupnorm.input[0], extractor)

    # if nonlinear == "Silu" and input_shape[0] == 1:
    #     return subgraph, [], None

    output_shape = ryzenai_onnx_utils.matcher.get_shape(groupnorm.output[0], extractor)
    gamma_shape = ryzenai_onnx_utils.matcher.get_shape(groupnorm.input[1], extractor)
    beta_shape = ryzenai_onnx_utils.matcher.get_shape(groupnorm.input[2], extractor)
    gamma_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(groupnorm.input[1], extractor)
    beta_f = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(groupnorm.input[2], extractor)
    wts_f = np.concatenate((gamma_f, beta_f))
    wts_name = groupnorm.name + f"_wts_{pass_id}"
    wts = float_numpy_to_bfloat_tensor(wts_f, wts_name, True)

    new_inputs = []
    new_nodes = []
    initializers = [wts]
    tvis = []

    pre_cast_output = groupnorm.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
        groupnorm.input[0],
        pre_cast_output,
        input_shape,
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(groupnorm.input[0], extractor),
    )
    new_inputs.extend([pre_cast_output, wts_name])
    tvis.extend(pre_cast_tvi)
    new_nodes.extend(pre_cast)

    groupnorm_output = groupnorm.output[0] + f".out{pass_id}"
    groupnorm_node = onnx.helper.make_node(
        "SDGroupNorm",
        inputs=new_inputs,
        outputs=[groupnorm_output],
        domain=domain,
        name=groupnorm.name,
    )
    new_nodes.append(groupnorm_node)
    copy_attributes(groupnorm, groupnorm_node)
    add_attribute(groupnorm_node, "input_shape", input_shape)
    add_attribute(groupnorm_node, "output_shape", output_shape)
    add_attribute(groupnorm_node, "wts_shape", wts_f.shape)
    add_attribute(groupnorm_node, "in_dtypes", ["bfloat16", "bfloat16"])
    add_attribute(groupnorm_node, "out_dtypes", ["bfloat16"])
    add_attribute(groupnorm_node, "gamma_shape", gamma_shape)
    add_attribute(groupnorm_node, "beta_shape", beta_shape)
    if nonlinear:
        add_attribute(groupnorm_node, "nonlinear", nonlinear)
    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        groupnorm_output,
        subgraph[-1].output[0],
        ryzenai_onnx_utils.matcher.get_shape(subgraph[-1].output[0], extractor),
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(groupnorm.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)
    new_nodes.extend(post_cast)

    return new_nodes, initializers, tvis


PATTERN = [
    ["GroupNorm([?,?,?], b0)", "Sigmoid([b0], b1)", "Mul([b0,b1], b2)"],
    ["GroupNorm([?,?,?], b0)", "Reshape([b0,?], b1)"],
    ["GroupNorm([?,?,?], ?)"],
]
REPLACEMENT = [replacement] * len(PATTERN)
