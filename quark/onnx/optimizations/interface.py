#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import Any

import onnx
from onnxruntime import GraphOptimizationLevel, SessionOptions
from onnxruntime.transformers.fusion_gelu import FusionGelu
from onnxruntime.transformers.fusion_layernorm import FusionLayerNormalization
from onnxruntime.transformers.onnx_model import OnnxModel

from quark.onnx.quantization.quant_utils import get_opset_version, load_model_with_shape_infer
from quark.onnx.utils.model_utils import create_infer_session_for_onnx_model
from quark.shares.utils.import_utils import _is_package_available
from quark.shares.utils.log import ScreenLogger, log_errors

from .optimize import Optimizer

logger = ScreenLogger(__name__)


@log_errors
def optimize_model(
    model: onnx.ModelProto,
    op_types_to_quantize: list[str],
    nodes_to_quantize: list[str] | None,
    nodes_to_exclude: list[str] | None,
    convert_bn_to_conv: bool = True,
    convert_reduce_mean_to_global_avg_pool: bool = True,
    split_large_kernel_pool: bool = True,
    convert_split_to_slice: bool = True,
    fuse_instance_norm: bool = True,
    fuse_l2_norm: bool = True,
    fuse_gelu: bool = True,
    fuse_layer_norm: bool = True,
    fold_batch_norm: bool = True,
    convert_clip_to_relu: bool = True,
    fold_batch_norm_after_concat: bool = True,
    dedicate_dq_node: bool = False,
) -> onnx.ModelProto:
    """
    Optimize an ONNX model to improve accuracy, meet specific constraints and requirements for deployment on an CPU/NPU.

    This function applies various optimization techniques to the provided ONNX model based on the specified parameters.
    The optimizations include fusing operations, converting specific layers, and folding batch normalization layers, among others.

    :param onnx.ModelProto model: The ONNX model to be optimized.
    :param List[str] op_types_to_quantize: List of operation types to be quantized.
    :param Optional[List[str]] nodes_to_quantize: List of node names to explicitly quantize. If ``None``, quantization is applied based on the operation types.
    :param Optional[List[str]] nodes_to_exclude: List of node names to exclude from quantization.
    :param bool convert_bn_to_conv: Flag indicating whether to convert BatchNorm layers to Conv layers.
    :param bool convert_reduce_mean_to_global_avg_pool: Flag indicating whether to convert ReduceMean layers to GlobalAveragePool layers.
    :param bool split_large_kernel_pool: Flag indicating whether to split large kernel pooling operations.
    :param bool convert_split_to_slice: Flag indicating whether to convert Split layers to Slice layers.
    :param bool fuse_instance_norm: Flag indicating whether to fuse InstanceNorm layers.
    :param bool fuse_l2_norm: Flag indicating whether to fuse L2Norm layers.
    :param bool fuse_gelu: Flag indicating whether to fuse Gelu layers.
    :param bool fuse_layer_norm: Flag indicating whether to fuse LayerNorm layers.
    :param bool fold_batch_norm: Flag indicating whether to fold BatchNorm layers into preceding Conv layers.
    :param bool convert_clip_to_relu: Flag indicating whether to convert Clip layers to ReLU layers.
    :param bool fold_batch_norm_after_concat: Flag indicating whether to fold BatchNorm layers after concatenation operations.

    :return: The optimized ONNX model.
    """

    onnx_model = OnnxModel(model)
    opset_version = get_opset_version(onnx_model.model)

    optimizer = Optimizer(
        model,
        op_types_to_quantize,
        nodes_to_quantize,
        nodes_to_exclude,
    )

    if fuse_instance_norm:
        optimizer.fuse_instance_norm()

    if convert_reduce_mean_to_global_avg_pool:
        optimizer.convert_reduce_mean_to_global_avg_pool()

    if split_large_kernel_pool:
        optimizer.split_large_kernel_pool()

    if convert_split_to_slice:
        optimizer.convert_split_to_slice()

    if fuse_l2_norm:
        optimizer.fuse_l2_norm()

    if fuse_layer_norm:
        if opset_version < 17:
            logger.warning(f"The opset version is {opset_version} < 17. Skipping fusing layer normalization.")
        else:
            fusion_layernorm = FusionLayerNormalization(onnx_model)
            fusion_layernorm.apply()

    if fuse_gelu:
        if opset_version < 20:
            logger.warning(f"The opset version is {opset_version} < 20. Skipping fusing Gelu.")
        else:
            fusion_gelu = FusionGelu(onnx_model)
            fusion_gelu.apply()

    if fold_batch_norm:
        optimizer.fold_batch_norm()

    if convert_clip_to_relu:
        optimizer.convert_clip_to_relu()

    if fold_batch_norm_after_concat:
        optimizer.fold_batch_norm_after_concat()

    if convert_bn_to_conv:
        optimizer.convert_bn_to_conv()

    # Only for quantization post-processing
    if dedicate_dq_node:
        optimizer.dedicate_dq_node()

    return optimizer.model


@log_errors
def optimize_model_using_onnxrt(inp_model: Path | onnx.ModelProto, opt_model_path: Path) -> onnx.ModelProto:
    """
    Generate model that applies graph optimization (constant folding, etc.).

    This is a derivative version of optimize_model in onnxruntime.quantization.quant_utils,
    it supports accepting a ModelProto that is <2GB.

    :param Union[Path, onnx.ModelProto] inp_model: the original onnx model to optimize.
    :param Path opt_model_path: path to the optimized onnx model.

    :return: The optimized ONNX model.
    """

    model = inp_model.SerializeToString() if isinstance(inp_model, onnx.ModelProto) else inp_model

    sess_option = SessionOptions()
    sess_option.optimized_model_filepath = opt_model_path.as_posix()
    sess_option.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_BASIC
    kwargs = {}
    # This will rename constant initializer names, disable it to make test pass.
    kwargs["disabled_optimizers"] = ["ConstantSharing"]
    _ = create_infer_session_for_onnx_model(model, sess_option, providers=["CPUExecutionProvider"], **kwargs)

    return load_model_with_shape_infer(opt_model_path)


@log_errors
def optimize_model_using_onnxslim(inp_model: onnx.ModelProto, config: dict[str, Any] = {}) -> onnx.ModelProto:
    """
    Simplify the model using the third-party library ``onnxslim``.

    :param onnx.ModelProto inp_model: the original onnx model to optimize.
    :param dict[str, Any] config: Optional configuration dictionary for onnxslim optimization.
                                  All parameters are passed directly to ``onnxslim.slim()`` as keyword arguments.
                                  Example configurations (e.g., {'skip_fusion_patterns': ['EliminationSlice']} to skip
                                  slice fusion patterns, or {'verbose': True, 'model_check': True} for debugging).
                                  For a complete list of supported parameters, refer to the `onnxslim source code
                                  <https://github.com/inisis/OnnxSlim>`_ or the ``onnxslim.slim()`` function signature.
    :return: The optimized ONNX model.
    """

    if not _is_package_available("onnxslim")[0]:
        raise ImportError("The 'onnxslim' is required but not installed. Please install it via 'pip install onnxslim'.")

    from onnxslim import slim

    try:
        opt_model = slim(inp_model, **config)
    except Exception as e:
        logger.warning(f"Fail to Simplify ONNX model because of {e}.")
        opt_model = inp_model

    return opt_model
