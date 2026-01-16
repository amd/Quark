# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.


import numpy as np
import onnx

import ryzenai_onnx_utils
from ryzenai_onnx_utils.transform import build_dd_node
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    first_node = subgraph[0]
    conv_in = subgraph[1]
    input_shape = ryzenai_onnx_utils.matcher.get_shape(first_node.input[0], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(conv_in.output[0], extractor)
    conv_inp_padding = np.concatenate((input_shape, output_shape), axis=0)
    first_node.name += "Conv3x3_inpadding"
    conv_out = subgraph[-1]
    input_shape_out = onnx.helper.get_node_attr_value(conv_out, "input_shape")
    output_shape_out = ryzenai_onnx_utils.matcher.get_shape(conv_out.output[0], extractor)
    conv_out_padding = np.concatenate((input_shape_out, output_shape_out), axis=0)
    # added padding pattern to the first node name as fused node will have the same name as firs node
    first_node.name += "Conv3x3_outpadding"

    dd_node = build_dd_node(
        extractor,
        subgraph,
        params,
        extra_attributes={
            "inp_padding": conv_inp_padding,
            "out_padding": conv_out_padding,
        },
    )

    return [dd_node], [], None


REPLACEMENT = replacement
# sdxl-turbo vae_decoder:
# NhwcConv_0_out-/post_quant_conv/Conv_output_0
# NhwcConv_35_out-sample
PATTERN = [
    "BF16_to_BFP16(?,b1)",
    "IConv_noqdq([b1,?,?],b0)",
    "GroupNorm_noqdq([b0,?,?],a7)",
    "SILU_noqdq(a7,a8)",
    "IConv_noqdq([a8,?,?],a11)",
    "GroupNorm_noqdq([a11,?,?],a14)",
    "SILU_noqdq(a14,a15)",
    "IConv_noqdq([a15,?,?],a18)",
    "ElwAdd_noqdq([b0,a18],a19)",
    "GroupNorm_noqdq([a19,?,?],a22)",
    "BF16_to_BFP16(a22,a23)",
    "MatMul_Transpose_noqdq([a23,?,?],a26)",
    "MatMul_noqdq([a23,?,?],a29)",
    "MatMul_noqdq([a23,?,?],a32)",
    "ActMatmul_noqdq([a32,a29],a33)",
    "Softmax_noqdq(a33,a34)",
    "BF16_to_BFP16(a34,a35)",
    "ActMatmul_noqdq([a35,a26],a36)",
    "MatMul_noqdq([a36,?,?],a39)",
    "ElwAdd_noqdq([a39,a19],a40)",
    "GroupNorm_noqdq([a40,?,?],a43)",
    "SILU_noqdq(a43,a44)",
    "IConv_noqdq([a44,?,?],a47)",
    "GroupNorm_noqdq([a47,?,?],a50)",
    "SILU_noqdq(a50,a51)",
    "IConv_noqdq([a51,?,?],a54)",
    "ElwAdd_noqdq([a54,a40],a55)",
    "GroupNorm_noqdq([a55,?,?],a58)",
    "SILU_noqdq(a58,a59)",
    "IConv_noqdq([a59,?,?],a62)",
    "GroupNorm_noqdq([a62,?,?],a65)",
    "SILU_noqdq(a65,a66)",
    "IConv_noqdq([a66,?,?],a69)",
    "ElwAdd_noqdq([a69,a55],a70)",
    "GroupNorm_noqdq([a70,?,?],a73)",
    "SILU_noqdq(a73,a74)",
    "IConv_noqdq([a74,?,?],a77)",
    "GroupNorm_noqdq([a77,?,?],a80)",
    "SILU_noqdq(a80,a81)",
    "IConv_noqdq([a81,?,?],a84)",
    "ElwAdd_noqdq([a84,a70],a85)",
    "GroupNorm_noqdq([a85,?,?],a88)",
    "SILU_noqdq(a88,a89)",
    "IConv_noqdq([a89,?,?],a92)",
    "GroupNorm_noqdq([a92,?,?],a95)",
    "SILU_noqdq(a95,a96)",
    "IConv_noqdq([a96,?,?],a99)",
    "ElwAdd_noqdq([a99,a85],a100)",
    "NNI_noqdq(a100,a101)",
    "BF16_to_BFP16(a101,a102)",
    "IConv_noqdq([a102,?,?],a105)",
    "GroupNorm_noqdq([a105,?,?],a108)",
    "SILU_noqdq(a108,a109)",
    "IConv_noqdq([a109,?,?],a112)",
    "GroupNorm_noqdq([a112,?,?],a115)",
    "SILU_noqdq(a115,a116)",
    "IConv_noqdq([a116,?,?],a119)",
    "ElwAdd_noqdq([a105,a119],a120)",
    "GroupNorm_noqdq([a120,?,?],a123)",
    "SILU_noqdq(a123,a124)",
    "IConv_noqdq([a124,?,?],a127)",
    "GroupNorm_noqdq([a127,?,?],a130)",
    "SILU_noqdq(a130,a131)",
    "IConv_noqdq([a131,?,?],a134)",
    "ElwAdd_noqdq([a120,a134],a135)",
    "GroupNorm_noqdq([a135,?,?],a138)",
    "SILU_noqdq(a138,a139)",
    "IConv_noqdq([a139,?,?],a142)",
    "GroupNorm_noqdq([a142,?,?],a145)",
    "SILU_noqdq(a145,a146)",
    "IConv_noqdq([a146,?,?],a149)",
    "ElwAdd_noqdq([a135,a149],a150)",
    "NNI_noqdq(a150,a151)",
    "BF16_to_BFP16(a151,a152)",
    "IConv_noqdq([a152,?,?],a155)",
    "GroupNorm_noqdq([a155,?,?],a158)",
    "SILU_noqdq(a158,a159)",
    "IConv_noqdq([a159,?,?],a162)",
    "GroupNorm_noqdq([a162,?,?],a165)",
    "SILU_noqdq(a165,a166)",
    "IConv_noqdq([a166,?,?],a169)",
    "BF16_to_BFP16(a155,a170)",
    "Conv1x1_noqdq([a170,?,?],a173)",
    "ElwAdd_noqdq([a173,a169],a174)",
    "GroupNorm_noqdq([a174,?,?],a177)",
    "SILU_noqdq(a177,a178)",
    "IConv_noqdq([a178,?,?],a181)",
    "GroupNorm_noqdq([a181,?,?],a184)",
    "SILU_noqdq(a184,a185)",
    "IConv_noqdq([a185,?,?],a188)",
    "ElwAdd_noqdq([a174,a188],a189)",
    "GroupNorm_noqdq([a189,?,?],a192)",
    "SILU_noqdq(a192,a193)",
    "IConv_noqdq([a193,?,?],a196)",
    "GroupNorm_noqdq([a196,?,?],a199)",
    "SILU_noqdq(a199,a200)",
    "IConv_noqdq([a200,?,?],a203)",
    "ElwAdd_noqdq([a189,a203],a204)",
    "NNI_noqdq(a204,a205)",
    "BF16_to_BFP16(a205,a206)",
    "IConv_noqdq([a206,?,?],a209)",
    "GroupNorm_noqdq([a209,?,?],a212)",
    "SILU_noqdq(a212,a213)",
    "IConv_noqdq([a213,?,?],a216)",
    "GroupNorm_noqdq([a216,?,?],a219)",
    "SILU_noqdq(a219,a220)",
    "IConv_noqdq([a220,?,?],a223)",
    "BF16_to_BFP16(a209,a224)",
    "Conv1x1_noqdq([a224,?,?],a227)",
    "ElwAdd_noqdq([a227,a223],a228)",
    "GroupNorm_noqdq([a228,?,?],a231)",
    "SILU_noqdq(a231,a232)",
    "IConv_noqdq([a232,?,?],a235)",
    "GroupNorm_noqdq([a235,?,?],a238)",
    "SILU_noqdq(a238,a239)",
    "IConv_noqdq([a239,?,?],a242)",
    "ElwAdd_noqdq([a228,a242],a243)",
    "GroupNorm_noqdq([a243,?,?],a246)",
    "SILU_noqdq(a246,a247)",
    "IConv_noqdq([a247,?,?],a250)",
    "GroupNorm_noqdq([a250,?,?],a253)",
    "SILU_noqdq(a253,a254)",
    "IConv_noqdq([a254,?,?],a257)",
    "ElwAdd_noqdq([a243,a257],a258)",
    "GroupNorm_noqdq([a258,?,?],a261)",
    "SILU_noqdq(a261,a262)",
    "IConv_noqdq([a262,?,?],?)",
]
