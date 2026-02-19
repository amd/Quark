# Copyright (C) 2024 - 2025 Advanced Micro Devices, Inc. All rights reserved.

from pathlib import Path

import onnx
import onnxruntime as rt
from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions

import ryzenai_onnx_utils.preprocess


def pre_optimize_passes() -> list[str]:
    return [
        "sd3.replace_einsum",
    ]


def finalize() -> list[str]:
    return [
        "normalize_binary_ops",
        "Nitro_E.transformer_preprocessing.normalize_mul_and_add_constants",
        "phi3_5.conv_to_nhwc_conv",
        "sd15.unet_preprocessing.gemm_to_matmul",
        "sd3.merge_mmdit_reshape_concat",
        "sd3.detach_packed_qkv_mha",
        "sd3_5.merge_shape_gather_add_div",
        "sd3_5.merge_mul",
        "sd3.merge_sqrt_div_sqrt",
        "sd3_5.merge_shape_slice",
        "sd3_5.merge_shape_gather_unsqueeze",
        "sd3_5.merge_shape_gather_div",
        "sd3_5.concat_constant_folding",
        "sd3_5.replace_rmsnorm",
        "sd3_5.normalize_mha_with_concat",
        "sd3_5.normalize_mha",
        "sd3.replace_mha",
        "sd3.transfer_pow_mul_add_tanh_to_gelu",
        "sd3.transfer_mul_add_tanh_to_gelu",
        "fastgelu_to_gelu",
        "sd15.unet_preprocessing.unsqueeze_to_reshape",
        "sd15.unet_preprocessing.remove_expand",
        "sd15.unet_preprocessing.remove_redundant_slice",
        "sd15.unet_preprocessing.matmul_add_slice_to_matmul",
        "sd15.vae_decoder_preprocessing.transpose_reshape_transpose_to_reshape",
        "sd15.vae_decoder_preprocessing.merge_reshapes",
        "sd3.add_reshape_add_to_reshape_add",
        "sd3.merge_adds",
        "sd3.matmul_reshape_add_to_matmul_add_reshape",
        #  "sd3.nhwcconv_to_conv  # NCHW for run MMDIT on CPUEP."
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
        "model_name": "UNET",
    }

    tmp_model = output_model_path.parent / "tmp_optimize.onnx"

    # ONNX default optimizer
    opt_level = 0
    verbose = True

    # ONNX Transformer optimizer
    optimization_options = FusionOptions(model_type="unet")
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
    optimization_options.enable_nhwc_conv = False
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
        model_type="unet",
        optimization_options=optimization_options,
        opt_level=opt_level,
        verbose=verbose,
    )
    # optimized_model.convert_float_to_float16()

    optimized_model.save_model_to_file(str(tmp_model), use_external_data_format=save_as_external)

    # ONNX default optimizer
    sess_options = rt.SessionOptions()
    if session_config is not None:
        for key, value in session_config.items():
            sess_options.add_session_config_entry(key, value)

    # Set graph optimization level
    sess_options.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_BASIC

    # To enable model serialization after graph optimization set this
    sess_options.optimized_model_filepath = str(output_model_path)

    rt.InferenceSession(tmp_model, providers=execution_providers, sess_options=sess_options)

    onnx.shape_inference.infer_shapes_path(output_model_path)

    # if save_as_external:
    #     ryzenai_onnx_utils.matcher.convert_to_external(
    #         output_model_path, output_model_path, external_data_extension, size_threshold
    #     )

    stem = output_model_path.stem
    onnx.save_model(
        onnx.load_model(output_model_path),
        output_model_path,
        save_as_external_data=save_as_external,
        location=f"{stem}.{external_data_extension}",
        size_threshold=size_threshold,
    )

    tmp_model.unlink()
    tmp_model.with_suffix(f".{external_data_extension}").unlink(True)
