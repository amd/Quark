#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any

import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantType

from quark.onnx.algorithm import apply_post_quant_algorithms
from quark.onnx.calibration import CachedDataReader
from quark.onnx.optimizations import optimize_model
from quark.onnx.quantization.quant_utils import convert_fp16_scale_to_fp32
from quark.onnx.tools.convert_bias_int32_to_int16 import convert_bias_int32_to_int16
from quark.onnx.tools.insert_clip_bfloat16_qdq import insert_clip_bfloat16_qdq
from quark.onnx.tools.remove_bf16_cast import remove_bf16_cast
from quark.onnx.tools.remove_qdq_between_ops import remove_qdq_between_ops
from quark.onnx.tools.remove_qdq_mul_add import remove_qdq_mul_add
from quark.onnx.tools.replace_bfloat16_qdq_cast import replace_bfloat16_qdq_cast
from quark.onnx.utils.system_utils import Profiler
from quark.shares.utils.log import ScreenLogger, log_errors

logger = ScreenLogger(__name__)


@log_errors
def apply_post_optimization_before_algo(
    quant_model: onnx.ModelProto,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    op_types_to_quantize: list[str] | None = None,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This is a function to apply post-optimization that is used to meet compilers' requirements.
    Note that these processes may affect accuracy and CAN be optimized through PTQ algorithms.

    :param onnx.ModelProto quant_model: The quantized model to be optimized.
    :param list[str] nodes_to_quantize: List of nodes names to quantize. When this list is not None only the nodes in this list.
    :param list[str] nodes_to_exclude: List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.
    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.
    :param Dict[str, Any] extra_options: Options for the post-process.

    :return: The optimized quantized model.
    """

    if extra_options.get("UseMatMulNBits", False):
        # No need post-process for the quantization using MatMulNBits
        return quant_model

    if extra_options.get("RemoveQDQMulAdd", False):
        remove_qdq_mul_add(quant_model)

    if "RemoveQDQBetweenOps" in extra_options:
        between_ops = extra_options.get("RemoveQDQBetweenOps")
        if not (
            isinstance(between_ops, list)
            and all(
                isinstance(item, tuple) and len(item) == 2 and all(isinstance(elem, str) for elem in item)
                for item in between_ops
            )
        ):
            logger.warning(
                f"The option of 'RemoveQDQBetweenOps' should be a list of (str, str) tuples but actually {between_ops}"
            )
        remove_qdq_between_ops(quant_model, between_ops)

    # Convert float16 scale to float32
    quantize_fp16 = extra_options.get("QuantizeFP16", False)
    if extra_options.get("UseFP32Scale", quantize_fp16):
        if quantize_fp16:
            quant_model = convert_fp16_scale_to_fp32(quant_model)
        else:
            logger.warning("The option of 'UseFP32Scale' is available only if 'QuantizeFP16' is enabled.")

    # Convert Int32 bias to Int16
    if extra_options.get("Int16Bias", False):
        try:
            quant_model, _ = convert_bias_int32_to_int16(quant_model)
        except Exception as e:
            logger.warning(
                f"Failed to convert bias from int32 to int16 beacuse {e}skip converting bias from int32 to int16."
            )

    # This optimization should after calibration
    if extra_options.get("ConvertClipToRelu", False):
        quant_model = optimize_model(
            quant_model,
            op_types_to_quantize,
            nodes_to_quantize,
            nodes_to_exclude,
            convert_bn_to_conv=False,
            convert_reduce_mean_to_global_avg_pool=False,
            split_large_kernel_pool=False,
            convert_split_to_slice=False,
            fuse_instance_norm=False,
            fuse_l2_norm=False,
            fuse_gelu=False,
            fuse_layer_norm=False,
            fold_batch_norm=False,
            convert_clip_to_relu=True,
            fold_batch_norm_after_concat=False,
            dedicate_dq_node=False,
        )

    return quant_model


@log_errors
def apply_post_quantization_algorithms(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    data_reader: CachedDataReader,
    calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
    use_external_data_format: bool = False,
    include_auto_mp: bool = False,
    include_fast_ft: bool = False,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This is an interface function used for applying post-quant algorithms on the quantized model.

    :param onnx.ModelProto float_model: The float model for reference.
    :param onnx.ModelProto quant_model: The quantized model to be optimized.
    :param CachedDataReader data_reader: Data reader for the algorithm.
    :param CalibrationMethod calibrate_method: Calibration method, the default is CalibrationMethod.MinMax.
    :param QuantType activation_type: The quantization type to mix in activation tensors.
    :param QuantType weight_type: The quantization type to mix in weight tensors.
    :param bool use_external_data_format: Option used for large size (>2GB) model.
    :param bool include_auto_mp: Option for enabling auto mixed precision.
    :param bool include_fast_ft: Option for enabling fast fine-tuningg.
    :param Dict[str, Any] extra_options: Options for the algorithm.

    :return: The tuned quantized model.
    """

    if extra_options.get("UseGPTQ", False):
        quant_model = apply_post_quant_algorithms(
            float_model,
            quant_model,
            data_reader,
            use_external_data_format=use_external_data_format,
            extra_options=extra_options,
            algorithm="GPTQ",
        )
        data_reader.reset_iter()

    if extra_options.get("BiasCorrection", False):
        quant_model = apply_post_quant_algorithms(
            float_model,
            quant_model,
            data_reader,
            calibrate_method=calibrate_method,
            activation_type=activation_type,
            use_external_data_format=use_external_data_format,
            extra_options=extra_options,
            algorithm="BiasCorrection",
        )
        data_reader.reset_iter()

    if include_auto_mp:
        quant_model = apply_post_quant_algorithms(
            float_model,
            quant_model,
            data_reader,
            activation_type=activation_type,
            weight_type=weight_type,
            use_external_data_format=use_external_data_format,
            extra_options=extra_options,
            algorithm="AutoMixprecision",
        )
        data_reader.reset_iter()

    if include_fast_ft:
        quant_model = apply_post_quant_algorithms(
            float_model,
            quant_model,
            data_reader,
            use_external_data_format=use_external_data_format,
            extra_options=extra_options,
            algorithm="FastFinetune",
        )
        data_reader.reset_iter()

    return quant_model


@log_errors
def apply_post_optimization_after_algo(
    quant_model: onnx.ModelProto,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    op_types_to_quantize: list[str] | None = None,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This is a function to apply post-optimization that is used to meet compilers' requirements.
    Note that these processes may affect accuracy and CANNOT be optimized through PTQ algorithms.

    :param onnx.ModelProto quant_model: The quantized model to be optimized.
    :param list[str] nodes_to_quantize: List of nodes names to quantize. When this list is not None only the nodes in this list.
    :param list[str] nodes_to_exclude: List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.
    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.
    :param Dict[str, Any] extra_options: Options for the post-process.

    :return: The optimized quantized model.
    """

    if extra_options.get("UseMatMulNBits", False):
        # No need post-process for the quantization using MatMulNBits
        return quant_model

    # Convert BFloat16 Q/DQ nodes to Cast
    if extra_options.get("BF16QDQToCast", extra_options.get("EnableVaimlBF16", False)):
        quant_model = replace_bfloat16_qdq_cast(quant_model)

    # Remove BFloat16 Cast and convert bfloat16 weights to float32
    if extra_options.get("EnableVaimlBF16", False):
        quant_model = remove_bf16_cast(quant_model)

    # Insert Clip nodes before BFloat16 QuantizeLinear nodes
    if extra_options.get("BF16WithClip", False):
        quant_model = insert_clip_bfloat16_qdq(quant_model)

    # This is a post processing of quantization
    if extra_options.get("DedicateDQNode", False):
        quant_model = optimize_model(
            quant_model,
            op_types_to_quantize,
            nodes_to_quantize,
            nodes_to_exclude,
            convert_bn_to_conv=False,
            convert_reduce_mean_to_global_avg_pool=False,
            split_large_kernel_pool=False,
            convert_split_to_slice=False,
            fuse_instance_norm=False,
            fuse_l2_norm=False,
            fuse_gelu=False,
            fuse_layer_norm=False,
            fold_batch_norm=False,
            convert_clip_to_relu=False,
            fold_batch_norm_after_concat=False,
            dedicate_dq_node=True,
        )

    return quant_model


@log_errors
@Profiler(msg=[["post process(including finetuning)"]])
def apply_post_process(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    data_reader: CalibrationDataReader,
    calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    op_types_to_quantize: list[str] | None = None,
    use_external_data_format: bool = False,
    include_auto_mp: bool = False,
    include_fast_ft: bool = False,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This is an interface function used for applying post-process on the quantized model.

    :param onnx.ModelProto float_model: The float model for reference.
    :param onnx.ModelProto quant_model: The quantized model to be optimized.
    :param CalibrationDataReader data_reader: Data reader for the algorithm.
    :param CalibrationMethod calibrate_method: Calibration method, the default is CalibrationMethod.MinMax.
    :param QuantType activation_type: The quantization type to mix in activation tensors.
    :param QuantType weight_type: The quantization type to mix in weight tensors.
    :param list[str] nodes_to_quantize: List of nodes names to quantize. When this list is not None only the nodes in this list.
    :param list[str] nodes_to_exclude: List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.
    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.
    :param bool use_external_data_format: Option used for large size (>2GB) model.
    :param bool include_auto_mp: Option for enabling auto mixed precision.
    :param bool include_fast_ft: Option for enabling fast fine-tuningg.
    :param Dict[str, Any] extra_options: Options for the algorithm.

    :return: The quantized model that has been processed.
    """

    quant_model = apply_post_optimization_before_algo(
        quant_model,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options,
    )

    quant_model = apply_post_quantization_algorithms(
        float_model,
        quant_model,
        data_reader,
        calibrate_method,
        activation_type,
        weight_type,
        use_external_data_format,
        include_auto_mp,
        include_fast_ft,
        extra_options,
    )

    quant_model = apply_post_optimization_after_algo(
        quant_model,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options,
    )

    return quant_model
