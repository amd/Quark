# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import onnx
import onnxruntime as rt

import ryzenai_onnx_utils.preprocess


def finalize() -> list[str]:
    return [
        # remove FP16 casts added unconditionally by ORT
        "hybrid_llm_remove_precision_casts",
        "remove_parallel_casts",
        "simplify_casts",
        "remove_dangling_nodes",
        "normalize_constants",
        "normalize_binary_ops",
        # ORT will break apart Gelu which we want to keep fused
        "sd3.transfer_pow_mul_add_tanh_to_gelu",
        "sd3.transfer_mul_add_tanh_to_gelu",
    ]


def optimize(
    input_model_path: Path,
    output_model_path: Path,
    save_as_external: bool,
    size_threshold: int,
    external_data_extension: str,
) -> None:
    model = onnx.load_model(input_model_path, load_external_data=True)

    tmp_model = output_model_path.parent / "tmp_optimize.onnx"

    symbolic_model = ryzenai_onnx_utils.preprocess.infer_symbolic_shapes(model)
    onnx.save_model(
        symbolic_model,
        tmp_model,
        save_as_external_data=save_as_external,
        location=f"{tmp_model.stem}.{external_data_extension}",
        size_threshold=size_threshold,
    )

    execution_providers = ["CPUExecutionProvider"]
    session_config = {}
    if save_as_external:
        session_config.update(
            {
                "session.optimized_model_external_initializers_file_name": f"{output_model_path.stem}.{external_data_extension}",
                "session.optimized_model_external_initializers_min_size_in_bytes": str(size_threshold),
            }
        )

    sess_options = rt.SessionOptions()
    if session_config:
        for key, value in session_config.items():
            sess_options.add_session_config_entry(key, value)

    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    sess_options.optimized_model_filepath = str(output_model_path)

    # passing the model as a serialized string as opposed to a path results in
    # a RuntimeError: narrowing_error for some reason with certain models like
    # Llama3.1-8B
    rt.InferenceSession(tmp_model, providers=execution_providers, sess_options=sess_options)

    onnx.shape_inference.infer_shapes_path(output_model_path)

    tmp_model.unlink(missing_ok=True)
    tmp_model.with_suffix(f".{external_data_extension}").unlink(missing_ok=True)
