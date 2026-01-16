# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

"""
The only difference between SubPass1 and SubPass2
is whether conv is converted to NhwcConv. Since NhwcConv is not supported by
CPU and Loop, Scan is not supported by Dml, we use SubPass2
not to convert Nhwcconv and run on CPU, to check validity of preprocessing flow.
"""

import numpy as np
import onnx
from onnx import GraphProto, NodeProto, TensorProto, helper

import ryzenai_onnx_utils
from ryzenai_onnx_utils.typing import PassOutputArgs, SubPass


def build_graph_from_nodes(
    node_list: list[NodeProto],
    graph_name: str,
    extractor: onnx.utils.Extractor,
    reverse: bool = False,
) -> GraphProto:
    used_inputs: set[str] = set()
    produced_outputs: set[str] = set()
    const_outputs: set[str] = set()
    initializer_names: set[str] = set()

    for node in node_list:
        initializer_names.update(i for i in node.input if ryzenai_onnx_utils.matcher.is_initializer(i, extractor))
    initializers = [ryzenai_onnx_utils.matcher.get_initializer(i, extractor) for i in initializer_names]

    for node in node_list:
        used_inputs.update(i for i in node.input if i and i not in initializer_names)
        produced_outputs.update(o for o in node.output if o)

        if node.op_type == "Constant":
            const_outputs.update(o for o in node.output if o)

    candidate_inputs = used_inputs - produced_outputs

    true_initializers = candidate_inputs & (initializer_names | const_outputs)
    real_inputs = candidate_inputs - true_initializers

    real_outputs = produced_outputs - used_inputs

    def make_vi(name: str) -> onnx.ValueInfoProto:
        dtype = ryzenai_onnx_utils.matcher.get_dtype(name, extractor)
        shape = list(ryzenai_onnx_utils.matcher.get_shape(name, extractor))
        shape[0] = 1
        return helper.make_tensor_value_info(name, dtype, shape)

    dummy_state_in = helper.make_tensor_value_info("dummy_in", TensorProto.FLOAT, [])
    dummy_state_out = helper.make_tensor_value_info("dummy_out", TensorProto.FLOAT, [])
    identity_node = helper.make_node("Identity", ["dummy_in"], ["dummy_out"])
    input_vinfo = [dummy_state_in] + [make_vi(name) for name in sorted(real_inputs)]
    output_vinfo = [dummy_state_out] + [make_vi(name) for name in sorted(real_outputs)]

    value_info = []
    for node in node_list:
        value_info.extend([make_vi(name) for name in node.input if name])
        value_info.extend([make_vi(name) for name in node.output if name])

    return helper.make_graph(
        nodes=node_list + [identity_node],
        name=graph_name,
        inputs=input_vinfo,
        outputs=output_vinfo,
        initializer=initializers,
        value_info=value_info,
    )


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    # This pass is only enabled in the main graph
    if extractor.model.graph.name == "DPU_Subgraph":
        return subgraph, [], None

    add = subgraph[-1]

    if subgraph[0].op_type == "NhwcConv":
        nhwcconv, _, concat = subgraph[:3]
        graph_input = [nhwcconv.input[0], concat.input[0]]
    else:
        concat = subgraph[0]
        graph_input = [concat.input[0], concat.input[1]]

    tvis = []
    nodes = []
    initializers: list[onnx.TensorProto] = []
    scan_input = []
    for i in graph_input:
        dtype = ryzenai_onnx_utils.matcher.get_dtype(i, extractor)
        i_shape = ryzenai_onnx_utils.matcher.get_shape(i, extractor)
        list(i_shape).insert(1, 1)
        i_unsqueeze_tvi = onnx.helper.make_tensor_value_info(
            i + "_unsqueeze",
            dtype,
            i_shape,
        )
        tvis.append(i_unsqueeze_tvi)
        i_unsqueeze_tensor = onnx.numpy_helper.from_array(np.array([1], dtype=np.int64), name=i + "_unsqueeze_tensor")
        initializers.append(i_unsqueeze_tensor)
        scan_input.append(i_unsqueeze_tvi)
        i_unsqueeze_node = helper.make_node(
            "Unsqueeze",
            inputs=[i, i_unsqueeze_tensor.name],
            outputs=[i_unsqueeze_tvi.name],
        )
        nodes.append(i_unsqueeze_node)
    add_o_shape_tuple = ryzenai_onnx_utils.matcher.get_shape(add.output[0], extractor)
    add_o_shape = [add_o_shape_tuple[0], 1, add_o_shape_tuple[1], add_o_shape_tuple[2]]
    add_o_tvi = helper.make_tensor_value_info(add.output[0] + "_unsqueeze", dtype, add_o_shape)
    tvis.append(add_o_tvi)
    scan_g = build_graph_from_nodes(subgraph, "DPU_Subgraph", extractor, reverse=True)
    dpu_state_in = onnx.numpy_helper.from_array(np.array(0, dtype=np.float32), name="_dpu_state_in")
    initializers.append(dpu_state_in)
    scan_node = helper.make_node(
        "Scan",
        inputs=[dpu_state_in.name] + [i.name for i in scan_input],
        outputs=["_dpu_state_out", add_o_tvi.name],
        num_scan_inputs=len(graph_input),
        body=scan_g,
    )
    nodes.append(scan_node)
    # model = helper.make_model(scan_g)
    # onnx.save(model, "scan_g.onnx")
    add_squeeze_tensor = onnx.numpy_helper.from_array(
        np.array([1], dtype=np.int64), name=add_o_tvi.name + "_squeeze_tensor"
    )
    initializers.append(add_squeeze_tensor)
    add_squeeze_node = helper.make_node(
        "Squeeze",
        inputs=[add_o_tvi.name, add_squeeze_tensor.name],
        outputs=[add.output[0]],
    )
    nodes.append(add_squeeze_node)
    [extractor.wmap.pop(i) for i in [init.name for init in scan_g.initializer]]

    return nodes, initializers, tvis


common_part = [
    "Add([a2,?], a3)",
    "LayerNormalization([a3,?,?], a4)",
    "LayerNormalization([a4,?,?], a5)",
    "MatMul([a5,?,?], a6)",
    "MatMul([a5,?,?], a7)",
    "MatMul([a5,?,?], a8)",
    "Add([a6,?], a9)",
    "Add([a7,?], a10)",
    "Add([a8,?], a11)",
    "MultiHeadAttention([a10,a9,a11,?], a12)",
    "MatMulNBits([a12,?,?,?,?,?], a13)",
    "Add([a13,a4], a14)",
    "LayerNormalization([a14,?,?], a15)",
    "MatMulNBits([a15,?,?,?,?,?], a16)",
    "Gelu([a16,?], a17)",
    "MatMulNBits([a17,?,?,?,?,?], a18)",
    "Add([a18,a14], a19)",
    "LayerNormalization([a19,?,?], a20)",
    "MatMul([a20,?,?], a21)",
    "MatMul([a20,?,?], a22)",
    "MatMul([a20,?,?], a23)",
    "Add([a21,?], a24)",
    "Add([a22,?], a25)",
    "Add([a23,?], a26)",
    "MultiHeadAttention([a25,a24,a26,?], a27)",
    "MatMulNBits([a27,?,?,?,?,?], a28)",
    "Add([a28,a19], a29)",
    "LayerNormalization([a29,?,?], a30)",
    "MatMulNBits([a30,?,?,?,?,?], a31)",
    "Gelu([a31,?], a32)",
    "MatMulNBits([a32,?,?,?,?,?], a33)",
    "Add([a33,a29], a34)",
    "LayerNormalization([a34,?,?], a35)",
    "MatMul([a35,?,?], a36)",
    "MatMul([a35,?,?], a37)",
    "MatMul([a35,?,?], a38)",
    "Add([a36,?], a39)",
    "Add([a37,?], a40)",
    "Add([a38,?], a41)",
    "MultiHeadAttention([a40,a39,a41,?], a42)",
    "MatMulNBits([a42,?,?,?,?,?], a43)",
    "Add([a43,a34], a44)",
    "LayerNormalization([a44,?,?], a45)",
    "MatMulNBits([a45,?,?,?,?,?], a46)",
    "Gelu([a46,?], a47)",
    "MatMulNBits([a47,?,?,?,?,?], a48)",
    "Add([a48,a44], a49)",
    "LayerNormalization([a49,?,?], a50)",
    "MatMul([a50,?,?], a51)",
    "MatMul([a50,?,?], a52)",
    "MatMul([a50,?,?], a53)",
    "Add([a51,?], a54)",
    "Add([a52,?], a55)",
    "Add([a53,?], a56)",
    "MultiHeadAttention([a55,a54,a56,?], a57)",
    "MatMulNBits([a57,?,?,?,?,?], a58)",
    "Add([a58,a49], a59)",
    "LayerNormalization([a59,?,?], a60)",
    "MatMulNBits([a60,?,?,?,?,?], a61)",
    "Gelu([a61,?], a62)",
    "MatMulNBits([a62,?,?,?,?,?], a63)",
    "Add([a63,a59], a64)",
    "LayerNormalization([a64,?,?], a65)",
    "MatMul([a65,?,?], a66)",
    "MatMul([a65,?,?], a67)",
    "MatMul([a65,?,?], a68)",
    "Add([a66,?], a69)",
    "Add([a67,?], a70)",
    "Add([a68,?], a71)",
    "MultiHeadAttention([a70,a69,a71,?], a72)",
    "MatMulNBits([a72,?,?,?,?,?], a73)",
    "Add([a73,a64], a74)",
    "LayerNormalization([a74,?,?], a75)",
    "MatMulNBits([a75,?,?,?,?,?], a76)",
    "Gelu([a76,?], a77)",
    "MatMulNBits([a77,?,?,?,?,?], a78)",
    "Add([a78,a74], a79)",
    "LayerNormalization([a79,?,?], a80)",
    "MatMul([a80,?,?], a81)",
    "MatMul([a80,?,?], a82)",
    "MatMul([a80,?,?], a83)",
    "Add([a81,?], a84)",
    "Add([a82,?], a85)",
    "Add([a83,?], a86)",
    "MultiHeadAttention([a85,a84,a86,?], a87)",
    "MatMulNBits([a87,?,?,?,?,?], a88)",
    "Add([a88,a79], a89)",
    "LayerNormalization([a89,?,?], a90)",
    "MatMulNBits([a90,?,?,?,?,?], a91)",
    "Gelu([a91,?], a92)",
    "MatMulNBits([a92,?,?,?,?,?], a93)",
    "Add([a93,a89], a94)",
    "LayerNormalization([a94,?,?], a95)",
    "MatMul([a95,?,?], a96)",
    "MatMul([a95,?,?], a97)",
    "MatMul([a95,?,?], a98)",
    "Add([a96,?], a99)",
    "Add([a97,?], a100)",
    "Add([a98,?], a101)",
    "MultiHeadAttention([a100,a99,a101,?], a102)",
    "MatMulNBits([a102,?,?,?,?,?], a103)",
    "Add([a103,a94], a104)",
    "LayerNormalization([a104,?,?], a105)",
    "MatMulNBits([a105,?,?,?,?,?], a106)",
    "Gelu([a106,?], a107)",
    "MatMulNBits([a107,?,?,?,?,?], a108)",
    "Add([a108,a104], a109)",
    "LayerNormalization([a109,?,?], a110)",
    "MatMul([a110,?,?], a111)",
    "MatMul([a110,?,?], a112)",
    "MatMul([a110,?,?], a113)",
    "Add([a111,?], a114)",
    "Add([a112,?], a115)",
    "Add([a113,?], a116)",
    "MultiHeadAttention([a115,a114,a116,?], a117)",
    "MatMulNBits([a117,?,?,?,?,?], a118)",
    "Add([a118,a109], a119)",
    "LayerNormalization([a119,?,?], a120)",
    "MatMulNBits([a120,?,?,?,?,?], a121)",
    "Gelu([a121,?], a122)",
    "MatMulNBits([a122,?,?,?,?,?], a123)",
    "Add([a123,a119], a124)",
    "LayerNormalization([a124,?,?], a125)",
    "MatMul([a125,?,?], a126)",
    "MatMul([a125,?,?], a127)",
    "MatMul([a125,?,?], a128)",
    "Add([a126,?], a129)",
    "Add([a127,?], a130)",
    "Add([a128,?], a131)",
    "MultiHeadAttention([a130,a129,a131,?], a132)",
    "MatMulNBits([a132,?,?,?,?,?], a133)",
    "Add([a133,a124], a134)",
    "LayerNormalization([a134,?,?], a135)",
    "MatMulNBits([a135,?,?,?,?,?], a136)",
    "Gelu([a136,?], a137)",
    "MatMulNBits([a137,?,?,?,?,?], a138)",
    "Add([a138,a134], a139)",
    "LayerNormalization([a139,?,?], a140)",
    "MatMul([a140,?,?], a141)",
    "MatMul([a140,?,?], a142)",
    "MatMul([a140,?,?], a143)",
    "Add([a141,?], a144)",
    "Add([a142,?], a145)",
    "Add([a143,?], a146)",
    "MultiHeadAttention([a145,a144,a146,?], a147)",
    "MatMulNBits([a147,?,?,?,?,?], a148)",
    "Add([a148,a139], a149)",
    "LayerNormalization([a149,?,?], a150)",
    "MatMulNBits([a150,?,?,?,?,?], a151)",
    "Gelu([a151,?], a152)",
    "MatMulNBits([a152,?,?,?,?,?], a153)",
    "Add([a153,a149], a154)",
    "LayerNormalization([a154,?,?], a155)",
    "MatMul([a155,?,?], a156)",
    "MatMul([a155,?,?], a157)",
    "MatMul([a155,?,?], a158)",
    "Add([a156,?], a159)",
    "Add([a157,?], a160)",
    "Add([a158,?], a161)",
    "MultiHeadAttention([a160,a159,a161,?], a162)",
    "MatMulNBits([a162,?,?,?,?,?], a163)",
    "Add([a163,a154], a164)",
    "LayerNormalization([a164,?,?], a165)",
    "MatMulNBits([a165,?,?,?,?,?], a166)",
    "Gelu([a166,?], a167)",
    "MatMulNBits([a167,?,?,?,?,?], a168)",
    "Add([a168,a164], a169)",
    "LayerNormalization([a169,?,?], a170)",
    "MatMul([a170,?,?], a171)",
    "MatMul([a170,?,?], a172)",
    "MatMul([a170,?,?], a173)",
    "Add([a171,?], a174)",
    "Add([a172,?], a175)",
    "Add([a173,?], a176)",
    "MultiHeadAttention([a175,a174,a176,?], a177)",
    "MatMulNBits([a177,?,?,?,?,?], a178)",
    "Add([a178,a169], a179)",
    "LayerNormalization([a179,?,?], a180)",
    "MatMulNBits([a180,?,?,?,?,?], a181)",
    "Gelu([a181,?], a182)",
    "MatMulNBits([a182,?,?,?,?,?], a183)",
    "Add([a183,a179], a184)",
    "LayerNormalization([a184,?,?], a185)",
    "MatMul([a185,?,?], a186)",
    "MatMul([a185,?,?], a187)",
    "MatMul([a185,?,?], a188)",
    "Add([a186,?], a189)",
    "Add([a187,?], a190)",
    "Add([a188,?], a191)",
    "MultiHeadAttention([a190,a189,a191,?], a192)",
    "MatMulNBits([a192,?,?,?,?,?], a193)",
    "Add([a193,a184], a194)",
    "LayerNormalization([a194,?,?], a195)",
    "MatMulNBits([a195,?,?,?,?,?], a196)",
    "Gelu([a196,?], a197)",
    "MatMulNBits([a197,?,?,?,?,?], a198)",
    "Add([a198,a194], a199)",
    "LayerNormalization([a199,?,?], a200)",
    "MatMul([a200,?,?], a201)",
    "MatMul([a200,?,?], a202)",
    "MatMul([a200,?,?], a203)",
    "Add([a201,?], a204)",
    "Add([a202,?], a205)",
    "Add([a203,?], a206)",
    "MultiHeadAttention([a205,a204,a206,?], a207)",
    "MatMulNBits([a207,?,?,?,?,?], a208)",
    "Add([a208,a199], a209)",
    "LayerNormalization([a209,?,?], a210)",
    "MatMulNBits([a210,?,?,?,?,?], a211)",
    "Gelu([a211,?], a212)",
    "MatMulNBits([a212,?,?,?,?,?], a213)",
    "Add([a213,a209], a214)",
    "LayerNormalization([a214,?,?], a215)",
    "MatMul([a215,?,?], a216)",
    "MatMul([a215,?,?], a217)",
    "MatMul([a215,?,?], a218)",
    "Add([a216,?], a219)",
    "Add([a217,?], a220)",
    "Add([a218,?], a221)",
    "MultiHeadAttention([a220,a219,a221,?], a222)",
    "MatMulNBits([a222,?,?,?,?,?], a223)",
    "Add([a223,a214], a224)",
    "LayerNormalization([a224,?,?], a225)",
    "MatMulNBits([a225,?,?,?,?,?], a226)",
    "Gelu([a226,?], a227)",
    "MatMulNBits([a227,?,?,?,?,?], a228)",
    "Add([a228,a224], a229)",
    "LayerNormalization([a229,?,?], a230)",
    "MatMul([a230,?,?], a231)",
    "MatMul([a230,?,?], a232)",
    "MatMul([a230,?,?], a233)",
    "Add([a231,?], a234)",
    "Add([a232,?], a235)",
    "Add([a233,?], a236)",
    "MultiHeadAttention([a235,a234,a236,?], a237)",
    "MatMulNBits([a237,?,?,?,?,?], a238)",
    "Add([a238,a229], a239)",
    "LayerNormalization([a239,?,?], a240)",
    "MatMulNBits([a240,?,?,?,?,?], a241)",
    "Gelu([a241,?], a242)",
    "MatMulNBits([a242,?,?,?,?,?], a243)",
    "Add([a243,a239], a244)",
    "LayerNormalization([a244,?,?], a245)",
    "MatMul([a245,?,?], a246)",
    "MatMul([a245,?,?], a247)",
    "MatMul([a245,?,?], a248)",
    "Add([a246,?], a249)",
    "Add([a247,?], a250)",
    "Add([a248,?], a251)",
    "MultiHeadAttention([a250,a249,a251,?], a252)",
    "MatMulNBits([a252,?,?,?,?,?], a253)",
    "Add([a253,a244], a254)",
    "LayerNormalization([a254,?,?], a255)",
    "MatMulNBits([a255,?,?,?,?,?], a256)",
    "Gelu([a256,?], a257)",
    "MatMulNBits([a257,?,?,?,?,?], a258)",
    "Add([a258,a254], a259)",
    "LayerNormalization([a259,?,?], a260)",
    "MatMul([a260,?,?], a261)",
    "MatMul([a260,?,?], a262)",
    "MatMul([a260,?,?], a263)",
    "Add([a261,?], a264)",
    "Add([a262,?], a265)",
    "Add([a263,?], a266)",
    "MultiHeadAttention([a265,a264,a266,?], a267)",
    "MatMulNBits([a267,?,?,?,?,?], a268)",
    "Add([a268,a259], a269)",
    "LayerNormalization([a269,?,?], a270)",
    "MatMulNBits([a270,?,?,?,?,?], a271)",
    "Gelu([a271,?], a272)",
    "MatMulNBits([a272,?,?,?,?,?], a273)",
    "Add([a273,a269], a274)",
    "LayerNormalization([a274,?,?], a275)",
    "MatMul([a275,?,?], a276)",
    "MatMul([a275,?,?], a277)",
    "MatMul([a275,?,?], a278)",
    "Add([a276,?], a279)",
    "Add([a277,?], a280)",
    "Add([a278,?], a281)",
    "MultiHeadAttention([a280,a279,a281,?], a282)",
    "MatMulNBits([a282,?,?,?,?,?], a283)",
    "Add([a283,a274], a284)",
    "LayerNormalization([a284,?,?], a285)",
    "MatMulNBits([a285,?,?,?,?,?], a286)",
    "Gelu([a286,?], a287)",
    "MatMulNBits([a287,?,?,?,?,?], a288)",
    "Add([a288,a284], a289)",
    "LayerNormalization([a289,?,?], a290)",
    "MatMul([a290,?,?], a291)",
    "MatMul([a290,?,?], a292)",
    "MatMul([a290,?,?], a293)",
    "Add([a291,?], a294)",
    "Add([a292,?], a295)",
    "Add([a293,?], a296)",
    "MultiHeadAttention([a295,a294,a296,?], a297)",
    "MatMulNBits([a297,?,?,?,?,?], a298)",
    "Add([a298,a289], a299)",
    "LayerNormalization([a299,?,?], a300)",
    "MatMulNBits([a300,?,?,?,?,?], a301)",
    "Gelu([a301,?], a302)",
    "MatMulNBits([a302,?,?,?,?,?], a303)",
    "Add([a303,a299], a304)",
    "LayerNormalization([a304,?,?], a305)",
    "MatMul([a305,?,?], a306)",
    "MatMul([a305,?,?], a307)",
    "MatMul([a305,?,?], a308)",
    "Add([a306,?], a309)",
    "Add([a307,?], a310)",
    "Add([a308,?], a311)",
    "MultiHeadAttention([a310,a309,a311,?], a312)",
    "MatMulNBits([a312,?,?,?,?,?], a313)",
    "Add([a313,a304], a314)",
    "LayerNormalization([a314,?,?], a315)",
    "MatMulNBits([a315,?,?,?,?,?], a316)",
    "Gelu([a316,?], a317)",
    "MatMulNBits([a317,?,?,?,?,?], a318)",
    "Add([a318,a314], a319)",
    "LayerNormalization([a319,?,?], a320)",
    "MatMul([a320,?,?], a321)",
    "MatMul([a320,?,?], a322)",
    "MatMul([a320,?,?], a323)",
    "Add([a321,?], a324)",
    "Add([a322,?], a325)",
    "Add([a323,?], a326)",
    "MultiHeadAttention([a325,a324,a326,?], a327)",
    "MatMulNBits([a327,?,?,?,?,?], a328)",
    "Add([a328,a319], a329)",
    "LayerNormalization([a329,?,?], a330)",
    "MatMulNBits([a330,?,?,?,?,?], a331)",
    "Gelu([a331,?], a332)",
    "MatMulNBits([a332,?,?,?,?,?], a333)",
    "Add([a333,a329], a334)",
    "LayerNormalization([a334,?,?], a335)",
    "MatMul([a335,?,?], a336)",
    "MatMul([a335,?,?], a337)",
    "MatMul([a335,?,?], a338)",
    "Add([a336,?], a339)",
    "Add([a337,?], a340)",
    "Add([a338,?], a341)",
    "MultiHeadAttention([a340,a339,a341,?], a342)",
    "MatMulNBits([a342,?,?,?,?,?], a343)",
    "Add([a343,a334], a344)",
    "LayerNormalization([a344,?,?], a345)",
    "MatMulNBits([a345,?,?,?,?,?], a346)",
    "Gelu([a346,?], a347)",
    "MatMulNBits([a347,?,?,?,?,?], a348)",
    "Add([a344,a348], a349)",
]

PATTERN = [
    SubPass(
        "pattern1",
        [
            "NhwcConv([?,?], a0)",
            "Reshape([a0,?], a1)",
            "Concat([?,a1], a2)",
        ]
        + common_part,
    ),
    SubPass(
        "pattern2",
        [
            "Concat([?,?], a2)",
        ]
        + common_part,
    ),
]
REPLACEMENT = replacement
