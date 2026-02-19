# Copyright (C) 2024 - 2026 Advanced Micro Devices, Inc. All rights reserved.

import os
from pathlib import Path

import onnx
import onnxruntime as rt
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions

import ryzenai_onnx_utils
import ryzenai_onnx_utils.preprocess
from ryzenai_onnx_utils.preprocess import has_dynamic_inputs


def pre_optimize_passes() -> list[str]:
    return ["sd3.replace_3d_instance_norm_to_4d"]


def finalize() -> list[str]:
    return [
        "normalize_binary_ops",
        "phi3_5.conv_to_nhwc_conv",
        "sd3.merge_sqrt_div_sqrt",
        "sd15.unet_preprocessing.gemm_to_matmul",
        "sd3.replace_mha",
        "sd15.vae_decoder_preprocessing.eliminate_reshapes",
        "sd15.vae_decoder_preprocessing.transpose_nchw_resize_to_nhwc",
        "sd15.vae_decoder_preprocessing.transpose_nchw_add_to_nhwc",
        "sd15.vae_decoder_preprocessing.transpose_reshape_transpose_to_reshape",
        "sd15.vae_decoder_preprocessing.remove_transposes",
        "sd15.vae_decoder_preprocessing.merge_mul_coeff_to_matmul_add",
        "sd15.vae_decoder_preprocessing.transfer_reshapelike_transpose",
        "sd15.vae_decoder_preprocessing.merge_reshapes",
        "sd15.unet_preprocessing.groupnorm_to_groupnorm_silu",
        "sd15.vae_decoder_preprocessing.conv_transpose_add_div_clip_to_conv_transpose",
        "remove_dangling_nodes",
        "remove_unused_io",
    ]


def optimize(
    input_model_path: Path,
    output_model_path: Path,
    save_as_external: bool,
    size_threshold: int,
    external_data_extension: str,
) -> None:
    execution_providers = ["DmlExecutionProvider"]
    session_config = {
        # use dummy variable because we don't actually need this for optimizing
        "dd_root": "dd_root",
        "model_name": "DECODER",
    }

    tmp_model = output_model_path.parent / "tmp_optimize.onnx"

    # ONNX default optimizer
    opt_level = 0
    verbose = False

    # ONNX Transformer optimizer
    optimization_options = FusionOptions(model_type="vae")
    # layernorm
    optimization_options.enable_skip_layer_norm = False
    optimization_options.enable_bias_skip_layer_norm = False
    # matmul
    optimization_options.enable_qordered_matmul = False
    optimization_options.enable_bias_add = False
    # gelu
    optimization_options.enable_bias_gelu = False
    optimization_options.enable_bias_splitgelu = False
    # groupnorm+silu
    # optimization_options.enable_group_norm = False
    # optimization_options.group_norm_channels_last = False
    optimization_options.enable_skip_group_norm = False
    # conv
    optimization_options.enable_nhwc_conv = True
    # MHA
    optimization_options.enable_attention = True
    optimization_options.use_multi_head_attention = True
    optimization_options.enable_packed_qkv = False
    optimization_options.enable_packed_kv = False
    optimization_options.disable_multi_head_attention_bias = True
    optimization_options.disable_attention_mask()

    model = onnx.load_model(input_model_path, load_external_data=True)

    symbolic_model = ryzenai_onnx_utils.preprocess.infer_symbolic_shapes(model)

    optimized_model = optimizer.optimize_model(
        symbolic_model,
        model_type="vae",
        optimization_options=optimization_options,
        opt_level=opt_level,
        verbose=verbose,
    )
    # optimized_model.convert_float_to_float16()
    optimized_model.save_model_to_file(str(tmp_model), use_external_data_format=save_as_external)
    optimized_model.get_fused_operator_statistics()

    # ONNX default optimizer
    sess_options = rt.SessionOptions()
    if session_config is not None:
        for key, value in session_config.items():
            sess_options.add_session_config_entry(key, value)

    # Set graph optimization level
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_EXTENDED

    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath = str(output_model_path)

    rt.InferenceSession(tmp_model, providers=execution_providers, sess_options=sess_options)

    onnx.shape_inference.infer_shapes_path(output_model_path)

    if save_as_external:
        ryzenai_onnx_utils.matcher.convert_to_external(
            output_model_path,
            output_model_path,
            external_data_extension,
            size_threshold,
        )

    stem = output_model_path.stem

    model = onnx.load_model(output_model_path)
    if has_dynamic_inputs(model.graph):
        symbolic_model = ryzenai_onnx_utils.preprocess.infer_symbolic_shapes(model)
    else:
        symbolic_model = model

    if save_as_external:
        location_path = output_model_path.parent / f"{stem}.{external_data_extension}"
        if os.path.exists(location_path):
            Path(location_path).unlink(True)

    onnx.save_model(
        symbolic_model,
        output_model_path,
        save_as_external_data=save_as_external,
        location=f"{stem}.{external_data_extension}",
        size_threshold=size_threshold,
    )

    tmp_model.unlink()
    tmp_model.with_suffix(f".{external_data_extension}").unlink(True)
