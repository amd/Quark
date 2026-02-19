# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

import argparse
from pathlib import Path
from typing import Any

import onnx

import ryzenai_onnx_utils.matcher
import ryzenai_onnx_utils.transform.hybrid_llm as hybrid_llm


def postprocess_three(
    input_model_path: list[Path],
    output_model_path: Path,
    external_data_extension: str,
    size_threshold: int,
    options: dict[str, Any],
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

    eager_switch_threshold = int(options.get("eager_switch_threshold", 0))
    suffix = options.get("suffix", "")

    reference_model_path = input_model_path[0]
    prefill_model_path = input_model_path[1]
    token_model_path = input_model_path[2]
    new_external_file_name = f"{output_model_path.stem}.{external_data_extension}"

    reference_model = ryzenai_onnx_utils.matcher.load_model(
        reference_model_path, load_external_data=False, infer_shapes=False
    )
    input_name = hybrid_llm.get_input_ids_name(reference_model.graph, options)

    new_nodes = []
    new_tvis = []

    input_node = onnx.helper.make_node("Shape", inputs=[input_name], outputs=[f"shape{suffix}"], start=1, end=2)
    new_nodes.append(input_node)

    constant_node = onnx.helper.make_node("Constant", inputs=[], outputs=[f"comparison{suffix}"], value_int=1)
    new_tvis.append(onnx.helper.make_tensor_value_info(f"comparison{suffix}", onnx.TensorProto.INT64, [1]))
    new_nodes.append(constant_node)

    if eager_switch_threshold > 0:
        constant_node_4096 = onnx.helper.make_node(
            "Constant", inputs=[], outputs=[f"comparison_threshold{suffix}"], value_int=eager_switch_threshold
        )
        new_tvis.append(
            onnx.helper.make_tensor_value_info(f"comparison_threshold{suffix}", onnx.TensorProto.INT64, [1])
        )
        new_nodes.append(constant_node_4096)
        equal_output = f"cond_1{suffix}"

        equal_node_2 = onnx.helper.make_node(
            "Greater", inputs=[f"shape{suffix}", f"comparison_threshold{suffix}"], outputs=[f"cond_threshold{suffix}"]
        )
        new_nodes.append(equal_node_2)
        logical_or = onnx.helper.make_node(
            "Or", inputs=[f"cond_1{suffix}", f"cond_threshold{suffix}"], outputs=[f"cond{suffix}"]
        )
        new_nodes.append(logical_or)
    else:
        equal_output = f"cond{suffix}"

    equal_node = onnx.helper.make_node(
        "Equal", inputs=[f"shape{suffix}", f"comparison{suffix}"], outputs=[equal_output]
    )
    new_nodes.append(equal_node)

    token_model = ryzenai_onnx_utils.matcher.load_model(token_model_path, load_external_data=False, infer_shapes=False)
    prefill_model = ryzenai_onnx_utils.matcher.load_model(prefill_model_path, load_external_data=False)

    ryzenai_onnx_utils.matcher.delete_custom_metadata_props(token_model)
    ryzenai_onnx_utils.matcher.delete_custom_metadata_props(prefill_model)
    ryzenai_onnx_utils.matcher.delete_custom_metadata_props(reference_model)

    new_input_tvis = list(reference_model.graph.input)
    new_output_tvis = list(reference_model.graph.output)
    output_names = [x.name for x in reference_model.graph.output]

    del token_model.graph.input[:]
    del prefill_model.graph.input[:]
    if_node = onnx.helper.make_node(
        "If",
        inputs=[f"cond{suffix}"],
        outputs=output_names,
        then_branch=token_model.graph,
        else_branch=prefill_model.graph,
    )
    new_nodes.append(if_node)

    graph = onnx.helper.make_graph(new_nodes, f"subgraph{suffix}", new_input_tvis, new_output_tvis, value_info=new_tvis)
    opset = onnx.OperatorSetIdProto()
    opset.version = 21
    model = onnx.helper.make_model(graph, opset_imports=[opset])
    onnx.shape_inference.infer_shapes(model)

    extractor = ryzenai_onnx_utils.matcher.get_extractor(model)
    (output_model_path.parent / new_external_file_name).unlink(True)
    ryzenai_onnx_utils.matcher.save_initializers_with_extractor(
        extractor, output_model_path.parent, new_external_file_name, True, size_threshold
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
            options,
        )
    elif len(input_model_path) == 3:
        postprocess_three(
            input_model_path,
            output_model_path,
            options["external_data_extension"],
            args.external_data_threshold,
            options,
        )
    else:
        raise ValueError(f"Unexpected number of input graphs passed: {len(input_model_path)}")
