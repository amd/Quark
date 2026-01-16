# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse
from pathlib import Path
from typing import Any

import onnx

import ryzenai_onnx_utils.matcher


def postprocess_three(
    input_model_path: list[Path], output_model_path: Path, external_data_extension: str, size_threshold: int
) -> None:
    """
    Post-process three models. This assumes the first model will be the
    reference (used for input/outputs), second will be used for prefill
    and the third will be used for token.

    Args:
        input_model_path (list[Path]): Path to the input models
        output_model_path (Path): Path to the output model
    """
    assert len(input_model_path) == 3

    reference_model_path = input_model_path[0]
    prefill_model_path = input_model_path[1]
    token_model_path = input_model_path[2]
    new_external_file_name = f"{output_model_path.stem}.{external_data_extension}"

    new_nodes = []
    new_tvis = []

    input_node = onnx.helper.make_node("Shape", inputs=["input_ids"], outputs=["shape"], start=1, end=2)
    new_nodes.append(input_node)

    constant_node = onnx.helper.make_node("Constant", inputs=[], outputs=["comparison"], value_int=1)
    new_tvis.append(onnx.helper.make_tensor_value_info("comparison", onnx.TensorProto.INT64, [1]))

    new_nodes.append(constant_node)

    equal_node = onnx.helper.make_node("Equal", inputs=["shape", "comparison"], outputs=["cond"])
    new_nodes.append(equal_node)

    token_model = ryzenai_onnx_utils.matcher.load_model(token_model_path, load_external_data=False, infer_shapes=False)
    prefill_model = ryzenai_onnx_utils.matcher.load_model(prefill_model_path, load_external_data=False)
    reference_model = ryzenai_onnx_utils.matcher.load_model(
        reference_model_path, load_external_data=False, infer_shapes=False
    )

    new_input_tvis = list(reference_model.graph.input)
    new_output_tvis = list(reference_model.graph.output)
    output_names = [x.name for x in reference_model.graph.output]

    del token_model.graph.input[:]
    del prefill_model.graph.input[:]
    if_node = onnx.helper.make_node(
        "If",
        inputs=["cond"],
        outputs=output_names,
        then_branch=token_model.graph,
        else_branch=prefill_model.graph,
    )
    new_nodes.append(if_node)

    graph = onnx.helper.make_graph(new_nodes, "name", new_input_tvis, new_output_tvis, value_info=new_tvis)
    opset = onnx.OperatorSetIdProto()
    opset.version = 21
    model = onnx.helper.make_model(graph, opset_imports=[opset])
    onnx.shape_inference.infer_shapes(model)

    extractor = ryzenai_onnx_utils.matcher.get_extractor(model)
    (output_model_path.parent / new_external_file_name).unlink(True)
    ryzenai_onnx_utils.matcher.save_external_data_with_extractor(
        extractor.model, extractor, new_external_file_name, output_model_path.parent, True, True, size_threshold
    )
    ryzenai_onnx_utils.matcher.save_model_without_external_data(extractor.model, output_model_path)


def postprocess(
    args: argparse.Namespace, input_model_path: list[Path], output_model_path: Path, options: dict[str, Any]
) -> None:
    if len(input_model_path) == 2:
        # assume TPS fusion (so eager prefill) and use that as the reference model
        postprocess_three(
            [input_model_path[0], *input_model_path],
            output_model_path,
            options["external_data_extension"],
            args.external_data_threshold,
        )
    elif len(input_model_path) == 3:
        postprocess_three(
            input_model_path, output_model_path, options["external_data_extension"], args.external_data_threshold
        )
    else:
        raise ValueError(f"Unexpected number of input graphs passed: {len(input_model_path)}")
