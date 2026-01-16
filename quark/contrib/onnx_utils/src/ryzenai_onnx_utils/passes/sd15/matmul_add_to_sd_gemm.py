# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


from typing import Any

import numpy as np
import onnx

import ryzenai_onnx_utils.matcher
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
from ryzenai_onnx_utils.typing import PassOutputArgs

from .matmul_to_sd_matmul import get_matmul_params


@register_whitebox_pass("SDGemm")
class SDGemmPass(WhiteboxBasePass):
    whitebox_flow_op_type: str = "Gemm"
    force_whitelist: bool = True

    @staticmethod
    def is_supported_shape(op_namespace: str, check_shapes: dict[str, Any]) -> bool:
        supported_shape = {
            "sd15": {
                # unet
                ((2, 1, 320), (320, 1280)),
                ((2, 1, 1280), (1280, 1280)),
                ((2, 1, 1280), (1280, 640)),
                ((2, 1, 1280), (1280, 320)),
                ((2, 77, 768), (768, 320)),
                ((2, 77, 768), (768, 640)),
                ((2, 77, 768), (768, 1280)),
                ((2, 4096, 320), (320, 320)),
                ((2, 4096, 320), (320, 1280)),
                ((2, 4096, 1280), (1280, 320)),
                ((2, 1024, 640), (640, 640)),
                ((2, 1024, 640), (640, 2560)),
                ((2, 1024, 2560), (2560, 640)),
                ((2, 256, 1280), (1280, 1280)),
                ((2, 256, 1280), (1280, 5120)),
                ((2, 256, 5120), (5120, 1280)),
                ((2, 64, 1280), (1280, 1280)),
                ((2, 64, 1280), (1280, 5120)),
                ((2, 64, 5120), (5120, 1280)),
                # sd2.1-v/sd_turbo 768 unet
                ((2, 144, 1280), (1280, 1280)),
                ((2, 144, 1280), (1280, 5120)),
                ((2, 144, 5120), (5120, 1280)),
                ((2, 2304, 2560), (2560, 640)),
                ((2, 2304, 640), (640, 2560)),
                ((2, 2304, 640), (640, 640)),
                ((2, 576, 1280), (1280, 1280)),
                ((2, 576, 1280), (1280, 5120)),
                ((2, 576, 5120), (5120, 1280)),
                ((2, 9216, 1280), (1280, 320)),
                ((2, 9216, 320), (320, 1280)),
                ((2, 9216, 320), (320, 320)),
                ((2, 144, 1280), (1280, 5120)),
                ((2, 2304, 640), (640, 2560)),
                ((2, 576, 1280), (1280, 5120)),
                # ((2, 9216, 320), (320, 1280)),
                # 512 vae
                ((1, 4096, 512), (512, 512)),
                # sd2.1 768 vae
                ((1, 9216, 512), (512, 512)),
                # sdxl_turbo
                ((2, 77, 2048), (2048, 640)),
                ((2, 77, 2048), (2048, 1280)),
                ((2, 2816), (2816, 1280)),
                # 512 turbo
                ((2, 77, 1024), (1024, 320)),
                ((2, 77, 1024), (1024, 640)),
                ((2, 77, 1024), (1024, 1280)),
                # sd(xl)-turbo bs1
                ((1, 1, 320), (320, 1280)),
                ((1, 1, 1280), (1280, 1280)),
                ((1, 1, 1280), (1280, 640)),
                ((1, 1, 1280), (1280, 320)),
                ((1, 1, 2816), (2816, 1280)),
                ((1, 4096, 320), (320, 320)),
                ((1, 4096, 320), (320, 1280)),
                ((1, 4096, 1280), (1280, 320)),
                ((1, 1024, 640), (640, 640)),
                ((1, 1024, 640), (640, 2560)),
                ((1, 1024, 2560), (2560, 640)),
                ((1, 256, 1280), (1280, 1280)),
                ((1, 256, 1280), (1280, 5120)),
                ((1, 256, 5120), (5120, 1280)),
                ((1, 64, 1280), (1280, 1280)),
                ((1, 64, 1280), (1280, 5120)),
                ((1, 64, 5120), (5120, 1280)),
                # sdxl-base vae_decoder
                ((1, 16384, 512), (512, 512)),
                # sdxl-base unet
                ((2, 1, 2816), (2816, 1280)),
                ((2, 1024, 1280), (1280, 1280)),
                ((2, 1024, 1280), (1280, 5120)),
                ((2, 1024, 5120), (5120, 5120)),
                ((2, 4096, 640), (640, 640)),
                ((2, 4096, 640), (640, 2560)),
                ((2, 4096, 2560), (2560, 640)),
                ((2, 1024, 5120), (5120, 1280)),
            },
            "sd3": {
                # 512 mmdit
                ((2, 154, 4096), (4096, 1536)),
                ((2, 154, 1536), (1536, 1536)),
                ((2, 154, 1536), (1536, 6144)),
                ((2, 154, 6144), (6144, 1536)),
                ((2, 1, 2048), (2048, 1536)),
                ((1, 1, 2048), (2048, 1536)),
                ((2, 1, 1536), (1536, 1536)),
                ((1, 1, 1536), (1536, 1536)),
                ((2, 1, 256), (256, 1536)),
                ((1, 1, 256), (256, 1536)),
                ((2, 1024, 1536), (1536, 1536)),
                ((1, 1024, 1536), (1536, 1536)),
                ((2, 1024, 1536), (1536, 6144)),
                ((2, 1024, 6144), (6144, 1536)),
                ((1, 1024, 6144), (6144, 1536)),
                ((2, 1024, 1536), (1536, 64)),
                ((1, 1024, 1536), (1536, 64)),
                # # 1024 mmdit
                ((2, 4096, 1536), (1536, 1536)),
                ((1, 4096, 1536), (1536, 1536)),
                ((2, 4096, 1536), (1536, 6144)),
                ((2, 4096, 6144), (6144, 1536)),
                ((1, 4096, 6144), (6144, 1536)),
                ((2, 4096, 1536), (1536, 64)),
                ((1, 4096, 1536), (1536, 64)),
                # # 512 vae
                ((1, 4096, 512), (512, 512)),
                # # 1024 vae
                ((1, 16384, 512), (512, 512)),
                # mmdit seq-160
                ((2, 160, 1536), (1536, 1536)),
                ((2, 160, 1536), (1536, 6144)),
                ((2, 160, 4096), (4096, 1536)),
                ((1, 160, 4096), (4096, 1536)),
                ((2, 160, 6144), (6144, 1536)),
                ((1, 160, 6144), (6144, 1536)),
            },
            "phi3.5": {
                ((1, 577, 1024), (1024, 1024)),  # count 69
            },
        }
        matmul_input1_shape = tuple(check_shapes["input_shape"][0])
        matmul_input2_shape = tuple(check_shapes["input_shape"][1])
        return (matmul_input1_shape, matmul_input2_shape) in supported_shape[op_namespace]

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


def is_gemm_supported_pattern(extractor: onnx.utils.Extractor, matmul: onnx.NodeProto, add: onnx.NodeProto) -> bool:
    if len(matmul.input) != 2:
        return False
    if not ryzenai_onnx_utils.matcher.is_initializer(matmul.input[1], extractor):
        return False
    if len(ryzenai_onnx_utils.matcher.get_shape(matmul.input[1], extractor)) != 2:
        return False
    if add:
        if ryzenai_onnx_utils.matcher.has_multiple_successors(matmul.output[0], extractor.graph):
            return False
        if len(add.input) != 2:
            return False
        add_init = ryzenai_onnx_utils.matcher.find_initializers_by_nodes(extractor, add, False)
        if len(add_init) is None:
            return False
        add_shape = ryzenai_onnx_utils.matcher.get_shape(add_init[0], extractor)
        if len(add_shape) != 1:
            return False
    return True


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    domain = params.get_domain("SDGemm")
    if len(subgraph) >= 2 and subgraph[1].op_type == "Add":
        (matmul, add) = subgraph[:2]
    else:
        matmul = subgraph[0]
        add = None

    if not is_gemm_supported_pattern(extractor, matmul, add):
        return subgraph, [], None

    tvis = []
    initializers: list[onnx.TensorProto] = []
    pre_cast_output = matmul.input[0] + f".out{pass_id}"
    pre_cast, pre_cast_tvi = add_cast_dtype_to_bfloat16(
        matmul.input[0],
        pre_cast_output,
        ryzenai_onnx_utils.matcher.get_shape(matmul.input[0], extractor),
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(matmul.input[0], extractor),
    )
    tvis.extend(pre_cast_tvi)

    matmul_weight = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(matmul.input[1], extractor)
    assert len(matmul_weight.shape) == 2

    if add:
        (add_init,) = ryzenai_onnx_utils.matcher.find_initializers_by_nodes(extractor, add, False)
        bias_shape = ryzenai_onnx_utils.matcher.get_shape(add_init)
        bias_name = add_init.name
        n = (ryzenai_onnx_utils.matcher.get_shape(add.output[0], extractor))[-1]
        assert bias_shape == (n,)
    else:
        bias_name = matmul.output[0] + "_fake_bias"
        bias_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.output[0], extractor)[-1:]
        dtype = ryzenai_onnx_utils.matcher.get_dtype(matmul.output[0], extractor)
        bias_tvi = onnx.helper.make_tensor_value_info(
            bias_name,
            dtype,
            bias_shape,
        )
        bias_tensor = onnx.helper.make_tensor(
            bias_name,
            dtype,
            bias_shape,
            np.zeros(bias_shape, dtype=matmul_weight.dtype),
        )
        initializers.append(bias_tensor)
        tvis.append(bias_tvi)

    gemm_node = onnx.helper.make_node(
        "SDGemm",
        inputs=[pre_cast_output, matmul.input[1], bias_name],
        outputs=[subgraph[-1].output[0] + f".out{pass_id}"],
        name=matmul.name,
        domain=domain,
    )

    input_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[0], extractor)
    weight_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.input[1], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(matmul.output[0], extractor)
    BI, MI, KI, NI = get_matmul_params(input_shape, weight_shape, output_shape)

    add_attribute(gemm_node, "input_shape", [BI, MI, KI])
    add_attribute(gemm_node, "output_shape", [BI, MI, NI])
    add_attribute(gemm_node, "weight_shape", [KI, NI])
    add_attribute(gemm_node, "in_dtypes", ["bfloat16", "bfp16ebs8", "bfloat16"])
    add_attribute(gemm_node, "out_dtypes", ["bfloat16"])
    add_attribute(gemm_node, "bias_enable", True)
    if "Gelu" in [op.op_type for op in subgraph]:
        add_attribute(gemm_node, "nonlinear", "Gelu")

    post_cast, post_cast_tvi = add_cast_bfloat16_to_dtype(
        subgraph[-1].output[0] + f".out{pass_id}",
        subgraph[-1].output[0],
        ryzenai_onnx_utils.matcher.get_shape(subgraph[-1].output[0], extractor),
        domain,
        ryzenai_onnx_utils.matcher.get_dtype(subgraph[-1].output[0], extractor),
    )
    tvis.extend(post_cast_tvi)

    return [*pre_cast, gemm_node, *post_cast], initializers, tvis


PATTERN = [
    ["MatMul([?,?], b0)", "Add([?,b0], b1)", "Reshape([b1, ?], ?)"],
    ["MatMul([?,?], b0)", "Add([b0, ?], b1)", "Reshape([b1, ?], ?)"],
    ["MatMul([?,?], b0)", "Add([?, b0], b1)", "Gelu([b1], ?)"],
    ["MatMul([?,?], b0)", "Add([?,b0], ?)"],
    ["MatMul([?,?], b0)", "Add([b0,?], ?)"],
    ["MatMul([?,?], b0)", "Reshape([b0, ?], ?)"],
    ["MatMul([?,?], b0)"],
]
REPLACEMENT = [replacement] * len(PATTERN)
