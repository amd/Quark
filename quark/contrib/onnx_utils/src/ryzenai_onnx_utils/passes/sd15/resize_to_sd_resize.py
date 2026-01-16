# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import numpy as np
import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.matcher import (
    add_attribute,
    get_attribute,
    get_dtype,
    get_initializer_as_numpy,
    get_shape,
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
from ryzenai_onnx_utils.utils import float_numpy_to_bfloat_tensor


@register_whitebox_pass("SDResize")
class SDResizePass(WhiteboxBasePass):
    whitebox_flow_op_type = "Resize"

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shapes = {
            "sd15": {
                # Unet
                (2, 8, 8, 1280),
                (2, 16, 16, 1280),
                (2, 32, 32, 640),
                # Vae
                (1, 64, 64, 512),
                (1, 128, 128, 512),
                (1, 256, 256, 256),
                # sd2.1
                (1, 192, 192, 512),
                (1, 384, 384, 256),
                (1, 96, 96, 512),
                # sd2.1-v
                (2, 12, 12, 1280),
                (2, 24, 24, 1280),
                (2, 48, 48, 640),
                # sd(xl)-turbo bs1
                (1, 8, 8, 1280),
                (1, 16, 16, 1280),
                (1, 32, 32, 640),
                # sdxl-base vae_decoder
                (1, 512, 512, 256),
                (1, 256, 256, 512),
                # sdxl-base unet
                (2, 32, 32, 1280),
                (2, 64, 64, 640),
            },
            "sd3": {
                # vae 512
                (1, 64, 64, 512),
                (1, 256, 256, 256),
                # vae 1024
                (1, 256, 256, 512),
                (1, 512, 512, 256),
                # vae 512 + vae 1024
                (1, 128, 128, 512),
            },
        }
        input_shape = tuple(check_shapes["input_shape"][0])
        return input_shape in supported_shapes[op_namespace]

    @staticmethod
    def get_input_output_shapes(node: onnx.NodeProto, extractor: onnx.utils.Extractor) -> dict[str, Any]:
        shape_lists = {
            "input_shape": [
                get_attribute(node, "a_shape"),
            ],
            "output_shape": [
                get_attribute(node, "c_shape"),
            ],
        }
        return shape_lists


def is_resize_supported_pattern(extractor, resize: int):
    if not is_initializer(resize.input[2], extractor):
        return False
    scales = get_initializer_as_numpy(resize.input[2], extractor)
    if not all(float(x).is_integer() for x in scales):
        return False
    if not np.all(scales.astype(np.int64) == np.array([1, 2, 2, 1], dtype=np.int64)):
        return False
    mode = onnx.helper.get_node_attr_value(resize, "mode").decode()
    nearest_mode = onnx.helper.get_node_attr_value(resize, "nearest_mode").decode()
    return mode == "nearest" and nearest_mode == "floor"


# +start:
def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDResize")
    resize_node = subgraph[0]

    input_shape = get_shape(resize_node.input[0], extractor)
    output_shape = get_shape(resize_node.output[0], extractor)
    resize_output_shape = output_shape

    if not is_resize_supported_pattern(extractor, resize_node):
        return subgraph, [], None

    tvis = []

    unique_index = f"_{pass_id}"
    pre_cast_output_0 = resize_node.input[0] + f".out{unique_index}"
    pre_cast_0, pre_cast_tvi_0 = add_cast_dtype_to_bfloat16(
        resize_node.input[0],
        pre_cast_output_0,
        input_shape,
        domain,
        get_dtype(resize_node.input[0], extractor),
    )
    tvis.extend(pre_cast_tvi_0)

    new_inputs = [
        pre_cast_output_0,
    ]
    # add addition wts tensor for xrt
    wts_name = resize_node.name + f".weights{pass_id}"
    wts_type = onnx.TensorProto.BFLOAT16
    wts = float_numpy_to_bfloat_tensor(np.zeros(128), wts_name, True)
    wts_shape = [128]
    wts_tvi = onnx.helper.make_tensor_value_info(wts_name, wts_type, wts_shape)
    tvis.append(wts_tvi)
    new_inputs.append(wts_name)
    resize_node_output = resize_node.output[0] + f".out{unique_index}"
    sd_resize_node = onnx.helper.make_node(
        "SDResize",
        inputs=new_inputs,
        outputs=[resize_node_output],
        domain=domain,
        name=resize_node.name,
    )
    add_attribute(sd_resize_node, "a_shape", input_shape)
    add_attribute(sd_resize_node, "c_shape", output_shape)
    add_attribute(sd_resize_node, "in_dtypes", ["bfloat16"])
    add_attribute(sd_resize_node, "out_dtypes", ["bfloat16"])
    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        resize_node_output,
        resize_node.output[0],
        resize_output_shape,
        domain,
        get_dtype(resize_node.output[0], extractor),
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast_0, sd_resize_node, *post_cast], [wts], tvis


REPLACEMENT = replacement
PATTERN = ["Resize([?,?,?],?)"]
# -end:
