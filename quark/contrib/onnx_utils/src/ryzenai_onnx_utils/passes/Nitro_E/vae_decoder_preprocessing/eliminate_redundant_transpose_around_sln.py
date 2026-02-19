# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

import onnx

import ryzenai_onnx_utils.matcher
from ryzenai_onnx_utils.typing import PassOutputArgs


def replacement(
    extractor: onnx.utils.Extractor,
    pass_id: str,
    subgraph: list[onnx.NodeProto],
    params: ryzenai_onnx_utils.ReplaceParams,
) -> PassOutputArgs:
    transpose0, transpose1, add0, pow, reducemean, add, sqrt, div, mul, add1, relu, transpose2 = subgraph

    # 1. add
    add0_output_name = add0.output[0] + f"_{pass_id}"
    add0_output_shape = ryzenai_onnx_utils.matcher.get_shape(add0.output[0], extractor)
    add0_output_shape = (add0_output_shape[0], add0_output_shape[2], add0_output_shape[3], add0_output_shape[1])
    add0_output_dtype = ryzenai_onnx_utils.matcher.get_dtype(add0.output[0], extractor)
    add0_output_tvi = onnx.helper.make_tensor_value_info(add0_output_name, add0_output_dtype, add0_output_shape)
    add0_new = onnx.helper.make_node(
        "Add",
        inputs=[transpose0.input[0], transpose1.input[0]],
        outputs=[add0_output_name],
        name=f"{add0.name}_{pass_id}",
    )

    # 2. rmsnorm
    rms_norm_eps = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add.input[1], extractor)
    mul_init_shape = ryzenai_onnx_utils.matcher.get_shape(mul.input[1], extractor)[1]
    mul_init = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(mul.input[1], extractor, False).reshape(
        mul_init_shape
    )
    scale_init_name = f"rmsnormalization_scale_{pass_id}"
    scale_init_dtype = ryzenai_onnx_utils.matcher.get_dtype(mul.input[1], extractor)
    scale_init_shape = [mul_init_shape]
    scale_init_tvi = onnx.helper.make_tensor_value_info(scale_init_name, scale_init_dtype, scale_init_shape)
    scale_init = onnx.helper.make_tensor(
        name=scale_init_name,
        data_type=scale_init_dtype,
        dims=scale_init_shape,
        vals=mul_init.tolist(),
    )

    rms_norm_output_name = mul.output[0] + f"_{pass_id}"
    rms_norm_output_shape = ryzenai_onnx_utils.matcher.get_shape(mul.output[0], extractor)
    rms_norm_output_shape = (
        rms_norm_output_shape[0],
        rms_norm_output_shape[2],
        rms_norm_output_shape[3],
        rms_norm_output_shape[1],
    )
    rms_norm_output_dtype = ryzenai_onnx_utils.matcher.get_dtype(mul.output[0], extractor)
    rms_norm_output_tvi = onnx.helper.make_tensor_value_info(
        rms_norm_output_name, rms_norm_output_dtype, rms_norm_output_shape
    )
    rms_norm = onnx.helper.make_node(
        "SimplifiedLayerNormalization",
        inputs=[add0_output_name, scale_init_name],
        outputs=[rms_norm_output_name],
        name=f"SimplifiedLayerNormalization_{pass_id}",
        # domain=params.get_domain("RMSNormalization"),
        axis=-1,
    )
    ryzenai_onnx_utils.matcher.set_attribute(rms_norm, "epsilon", rms_norm_eps.tolist())

    # 3. Add
    add1_init_name = add1.input[1] + f"_{pass_id}"
    add1_init_shape = ryzenai_onnx_utils.matcher.get_shape(add1.input[1], extractor)[1]
    add1_init_dtype = ryzenai_onnx_utils.matcher.get_dtype(add1.input[1], extractor)
    add1_init_data = ryzenai_onnx_utils.matcher.get_initializer_as_numpy(add1.input[1], extractor, False).reshape(
        add1_init_shape
    )
    add1_init_tvi = onnx.helper.make_tensor_value_info(add1_init_name, add1_init_dtype, [add1_init_shape])
    add1_init = onnx.helper.make_tensor(
        name=add1_init_name,
        data_type=add1_init_dtype,
        dims=[add1_init_shape],
        vals=add1_init_data.tolist(),
    )

    add1_output_name = add1.output[0] + f"_{pass_id}"
    add1_output_shape = ryzenai_onnx_utils.matcher.get_shape(add1.output[0], extractor)
    add1_output_shape = (add1_output_shape[0], add1_output_shape[2], add1_output_shape[3], add1_output_shape[1])
    add1_output_dtype = ryzenai_onnx_utils.matcher.get_dtype(mul.output[0], extractor)
    add1_output_tvi = onnx.helper.make_tensor_value_info(add1_output_name, add1_output_dtype, add1_output_shape)

    add1_new = onnx.helper.make_node(
        "Add",
        inputs=[rms_norm_output_name, add1_init_name],
        outputs=[add1_output_name],
        name=f"{add1.name}_{pass_id}",
    )

    # 4. Relu
    relu_new = onnx.helper.make_node(
        "Relu",
        inputs=[add1_output_name],
        outputs=transpose2.output,
        name=f"{relu.name}_{pass_id}",
    )

    return (
        [add0_new, rms_norm, add1_new, relu_new],
        [scale_init, add1_init],
        [add0_output_tvi, scale_init_tvi, rms_norm_output_tvi, add1_init_tvi, add1_output_tvi],
    )


PATTERN = [
    "Transpose([?], b0)",
    "Transpose([?], b1)",
    "Add([b0, b1], b2)",
    "Pow([b2, ?], b3)",
    "ReduceMean([b3], b4)",
    "Add([b4,?], b5)",
    "Sqrt([b5], b6)",
    "Div([b2,b6], b7)",
    "Mul([b7,?], b8)",
    "Add([b8,?], b9)",
    "Relu([b9], b10)",
    "Transpose([b10], b11)",
]
REPLACEMENT = replacement
