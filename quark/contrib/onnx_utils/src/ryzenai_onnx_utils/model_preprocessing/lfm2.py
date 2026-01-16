# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import onnx
import onnxruntime as rt

import ryzenai_onnx_utils.preprocess


def finalize() -> list[str]:
    return []


def optimize(
    input_model_path: Path,
    output_model_path: Path,
    save_as_external: bool,
    size_threshold: int,
    external_data_extension: str,
) -> None:
    execution_providers = ["CPUExecutionProvider"]
    session_config = {}

    model = onnx.load_model(input_model_path, load_external_data=True)

    tmp_model = output_model_path.parent / "tmp_optimize.onnx"

    symbolic_model = ryzenai_onnx_utils.preprocess.infer_symbolic_shapes(model)
    onnx.save_model(
        symbolic_model,
        tmp_model,
        save_as_external_data=save_as_external,
        location=f"{output_model_path.stem}.{external_data_extension}",
        size_threshold=size_threshold,
    )

    sess_options = rt.SessionOptions()
    if session_config is not None:
        for key, value in session_config.items():
            sess_options.add_session_config_entry(key, value)

    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = str(output_model_path)

    rt.InferenceSession(tmp_model, providers=execution_providers, sess_options=sess_options)

    onnx.shape_inference.infer_shapes_path(output_model_path)

    onnx.save_model(
        onnx.load_model(output_model_path),
        output_model_path,
        save_as_external_data=save_as_external,
        location=f"{output_model_path.stem}.{external_data_extension}",
        size_threshold=size_threshold,
    )

    tmp_model.unlink(missing_ok=True)
    tmp_model.with_suffix(f".{external_data_extension}").unlink(missing_ok=True)
