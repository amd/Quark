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
    conv_inp = subgraph[0]
    input_shape = ryzenai_onnx_utils.matcher.get_shape(conv_inp.input[0], extractor)
    output_shape = ryzenai_onnx_utils.matcher.get_shape(conv_inp.output[0], extractor)
    conv_inp_padding = np.concatenate((input_shape, output_shape), axis=0)
    conv_inp.name += "Conv3x3_inpadding"

    conv_out = subgraph[-1]
    input_shape_out = ryzenai_onnx_utils.matcher.get_shape(conv_out.input[0], extractor)
    output_shape_out = ryzenai_onnx_utils.matcher.get_shape(conv_out.output[0], extractor)
    conv_out_padding = np.concatenate((input_shape_out, output_shape_out), axis=0)
    conv_inp.name += "Conv3x3_outpadding"

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
    "IConv_noqdq([?,?,?],a0)",
    "GroupNorm_noqdq([a0,?,?],a3)",
    "SILU_noqdq(a3,a4)",
    "IConv_noqdq([a4,?,?],a7)",
    "GroupNorm_noqdq([a7,?,?],a10)",
    "SILU_noqdq(a10,a11)",
    "IConv_noqdq([a11,?,?],a14)",
    "ElwAdd_noqdq([?,a14],a15)",
    "GroupNorm_noqdq([a15,?,?],a18)",
    "MatMul_noqdq([a18,?,?],a21)",
    "MatMul_noqdq([a18,?,?],a24)",
    "MatMul_noqdq([a18,?,?],a27)",
    "ActMatmul_noqdq([a27,a24],a28)",
    "Softmax_noqdq(a28,a29)",
    "ActMatmul_noqdq([a29,a21],a30)",
    "MatMul_noqdq([a30,?,?],a33)",
    "ElwAdd_noqdq([a33,a15],a34)",
    "GroupNorm_noqdq([a34,?,?],a37)",
    "SILU_noqdq(a37,a38)",
    "IConv_noqdq([a38,?,?],a41)",
    "GroupNorm_noqdq([a41,?,?],a44)",
    "SILU_noqdq(a44,a45)",
    "IConv_noqdq([a45,?,?],a48)",
    "ElwAdd_noqdq([a48,a34],a49)",
    "GroupNorm_noqdq([a49,?,?],a52)",
    "SILU_noqdq(a52,a53)",
    "IConv_noqdq([a53,?,?],a56)",
    "GroupNorm_noqdq([a56,?,?],a59)",
    "SILU_noqdq(a59,a60)",
    "IConv_noqdq([a60,?,?],a63)",
    "ElwAdd_noqdq([a63,a49],a64)",
    "GroupNorm_noqdq([a64,?,?],a67)",
    "SILU_noqdq(a67,a68)",
    "IConv_noqdq([a68,?,?],a71)",
    "GroupNorm_noqdq([a71,?,?],a74)",
    "SILU_noqdq(a74,a75)",
    "IConv_noqdq([a75,?,?],a78)",
    "ElwAdd_noqdq([a78,a64],a79)",
    "GroupNorm_noqdq([a79,?,?],a82)",
    "SILU_noqdq(a82,a83)",
    "IConv_noqdq([a83,?,?],a86)",
    "GroupNorm_noqdq([a86,?,?],a89)",
    "SILU_noqdq(a89,a90)",
    "IConv_noqdq([a90,?,?],a93)",
    "ElwAdd_noqdq([a93,a79],a94)",
    "NNI_noqdq(a94,a95)",
    "IConv_noqdq([a95,?,?],a98)",
    "GroupNorm_noqdq([a98,?,?],a101)",
    "SILU_noqdq(a101,a102)",
    "IConv_noqdq([a102,?,?],a105)",
    "GroupNorm_noqdq([a105,?,?],a108)",
    "SILU_noqdq(a108,a109)",
    "IConv_noqdq([a109,?,?],a112)",
    "ElwAdd_noqdq([a98,a112],a113)",
    "GroupNorm_noqdq([a113,?,?],a116)",
    "SILU_noqdq(a116,a117)",
    "IConv_noqdq([a117,?,?],a120)",
    "GroupNorm_noqdq([a120,?,?],a123)",
    "SILU_noqdq(a123,a124)",
    "IConv_noqdq([a124,?,?],a127)",
    "ElwAdd_noqdq([a113,a127],a128)",
    "GroupNorm_noqdq([a128,?,?],a131)",
    "SILU_noqdq(a131,a132)",
    "IConv_noqdq([a132,?,?],a135)",
    "GroupNorm_noqdq([a135,?,?],a138)",
    "SILU_noqdq(a138,a139)",
    "IConv_noqdq([a139,?,?],a142)",
    "ElwAdd_noqdq([a128,a142],a143)",
    "NNI_noqdq(a143,a144)",
    "IConv_noqdq([a144,?,?],a147)",
    "GroupNorm_noqdq([a147,?,?],a150)",
    "SILU_noqdq(a150,a151)",
    "IConv_noqdq([a151,?,?],a154)",
    "GroupNorm_noqdq([a154,?,?],a157)",
    "SILU_noqdq(a157,a158)",
    "IConv_noqdq([a158,?,?],a161)",
    "Conv1x1_noqdq([a147,?,?],a164)",
    "ElwAdd_noqdq([a164,a161],a165)",
    "GroupNorm_noqdq([a165,?,?],a168)",
    "SILU_noqdq(a168,a169)",
    "IConv_noqdq([a169,?,?],a172)",
    "GroupNorm_noqdq([a172,?,?],a175)",
    "SILU_noqdq(a175,a176)",
    "IConv_noqdq([a176,?,?],a179)",
    "ElwAdd_noqdq([a165,a179],a180)",
    "GroupNorm_noqdq([a180,?,?],a183)",
    "SILU_noqdq(a183,a184)",
    "IConv_noqdq([a184,?,?],a187)",
    "GroupNorm_noqdq([a187,?,?],a190)",
    "SILU_noqdq(a190,a191)",
    "IConv_noqdq([a191,?,?],a194)",
    "ElwAdd_noqdq([a180,a194],a195)",
    "NNI_noqdq(a195,a196)",
    "IConv_noqdq([a196,?,?],a199)",
    "GroupNorm_noqdq([a199,?,?],a202)",
    "SILU_noqdq(a202,a203)",
    "IConv_noqdq([a203,?,?],a206)",
    "GroupNorm_noqdq([a206,?,?],a209)",
    "SILU_noqdq(a209,a210)",
    "IConv_noqdq([a210,?,?],a213)",
    "Conv1x1_noqdq([a199,?,?],a216)",
    "ElwAdd_noqdq([a216,a213],a217)",
    "GroupNorm_noqdq([a217,?,?],a220)",
    "SILU_noqdq(a220,a221)",
    "IConv_noqdq([a221,?,?],a224)",
    "GroupNorm_noqdq([a224,?,?],a227)",
    "SILU_noqdq(a227,a228)",
    "IConv_noqdq([a228,?,?],a231)",
    "ElwAdd_noqdq([a217,a231],a232)",
    "GroupNorm_noqdq([a232,?,?],a235)",
    "SILU_noqdq(a235,a236)",
    "IConv_noqdq([a236,?,?],a239)",
    "GroupNorm_noqdq([a239,?,?],a242)",
    "SILU_noqdq(a242,a243)",
    "IConv_noqdq([a243,?,?],a246)",
    "ElwAdd_noqdq([a232,a246],a247)",
    "GroupNorm_noqdq([a247,?,?],a250)",
    "SILU_noqdq(a250,a251)",
    "IConv_noqdq([a251,?,?],a254)",
]
