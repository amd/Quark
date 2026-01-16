#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
from pathlib import Path
from typing import Any

import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantType

from quark.onnx.algorithm import apply_pre_quant_algorithms
from quark.onnx.calibration import CachedDataReader
from quark.onnx.optimizations import optimize_model, optimize_model_using_onnxrt, optimize_model_using_onnxslim
from quark.onnx.quantization.quant_utils import (
    ExtendedQuantType,
    load_model_with_shape_infer,
    remove_initializer_from_input,
)
from quark.onnx.tools import convert_shared_initializer_to_unique
from quark.onnx.tools.convert_opset_version import convert_opset_version
from quark.onnx.tools.fix_shapes import fix_input_and_output_shapes, infer_all_tensors_shape, save_all_tensors_shape
from quark.onnx.tools.float16 import convert_float16_to_float
from quark.onnx.utils.model_utils import convert_nchw_to_nhwc as convert_func
from quark.onnx.utils.model_utils import save_onnx_model_with_external_data
from quark.onnx.utils.system_utils import Profiler
from quark.shares.utils.log import ScreenLogger, log_errors

logger = ScreenLogger(__name__)


@log_errors
def apply_pre_optimization_before_algo(
    float_model: onnx.ModelProto,
    model_path: Path,
    calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    op_types_to_quantize: list[str] | None = None,
    use_external_data_format: bool = False,
    convert_fp16_to_fp32: bool = False,
    convert_nchw_to_nhwc: bool = False,
    optimize_model_flag: bool = False,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This is a function to apply pre-optimization before appling algorithms.
    Incorporating with the following algorithms, it can generally make the model easier to be quantized.

    :param onnx.ModelProto float_model: The float model to be optimized.
    :param Path model_path: The path of storing the float model.
    :param CalibrationMethod calibrate_method: Calibration method, the default is CalibrationMethod.MinMax.
    :param QuantType activation_type: The quantization type for activation tensors.
    :param QuantType weight_type: The quantization type for weight tensors.
    :param list[str] nodes_to_quantize: List of nodes names to quantize. When this list is not None only the nodes in this list.
    :param list[str] nodes_to_exclude: List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.
    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.
    :param bool convert_fp16_to_fp32: Convert the fp16 model to a fp32 one. Default is False.
    :param bool convert_nchw_to_nhwc: Convert the dimension order of inputs from 'nchw' to 'nhwc', which has better performance on NPU. Default is False.
    :param bool optimize_model_flag: Optimize the model using onnxruntime's graph optimization. Default is False.
    :param Dict[str, Any] extra_options: Options for the pre-process.

    :return: The optimized model.
    """

    # Step1. Common conversion
    if isinstance(extra_options.get("ConvertOpsetVersion"), int):
        try:
            float_model = convert_opset_version(float_model, extra_options["ConvertOpsetVersion"])
        except Exception as e:
            logger.warning(f"Fail to convert opset version beacuse of {e}, skip the conversion.")

    if convert_fp16_to_fp32:
        try:
            float_model = convert_float16_to_float(float_model)
        except Exception as e:
            logger.warning(f"Fail to convert fp16 to fp32 beacuse of {e}, skip the conversion.")

    if convert_nchw_to_nhwc:
        try:
            float_model = convert_func(float_model)
        except Exception as e:
            logger.warning(f"Failed to convert nchw to nhwc beacuse of {e}, skip the conversion.")

    if extra_options.get("SimplifyModel", True):
        try:
            float_model = optimize_model_using_onnxslim(float_model)
        except Exception as e:
            logger.warning(f"Fail to simplify the float model because of {e}.")

    # Step2. Dealing initializers
    if extra_options.get("RemoveInputInit", True):
        try:
            float_model = remove_initializer_from_input(float_model)
        except Exception as e:
            logger.warning(f"Fail to remove initializers from inputs because of {e}.")

    shared_init_optypes = extra_options.get("CopySharedInit")
    if shared_init_optypes is not None:
        try:
            float_model = convert_shared_initializer_to_unique.convert(float_model, shared_init_optypes)
        except Exception as e:
            logger.warning(f"Fail to duplicate the shared initializers because of {e}.")

    shared_bias_init_optypes = extra_options.get("CopyBiasInit", ["Conv", "ConvTranspose", "Gemm"])
    if shared_bias_init_optypes is not None:
        supported_quant_types = [
            QuantType.QUInt8,
            QuantType.QInt8,
            QuantType.QUInt16,
            QuantType.QInt16,
            ExtendedQuantType.QInt8,
            ExtendedQuantType.QUInt8,
            ExtendedQuantType.QInt16,
            ExtendedQuantType.QUInt16,
        ]
        if (
            (weight_type in supported_quant_types)
            and (activation_type in supported_quant_types)
            and (calibrate_method in CalibrationMethod)
        ):
            try:
                float_model = convert_shared_initializer_to_unique.convert(
                    float_model, shared_bias_init_optypes, prefix="duplicated", only_bias=True
                )
            except Exception as e:
                logger.warning(f"Fail to duplicate the shared bias initializers because of {e}.")

    if optimize_model_flag:
        save_onnx_model_with_external_data(float_model, model_path, use_external_data_format)
        try:
            opt_model_path = Path(os.path.join(os.path.dirname(model_path), "optimized_model.onnx"))
            float_model = optimize_model_using_onnxrt(Path(model_path), opt_model_path)
        except Exception as e:
            logger.warning(f"Failed to optimize the model using graph optimization of ort because of {e}.")
            float_model = load_model_with_shape_infer(model_path)

    # Fusing operators
    fold_batch_norm = extra_options.get("FoldBatchNorm", optimize_model_flag)
    fuse_instance_norm = extra_options.get("FuseInstanceNorm", True)
    fuse_l2_norm = extra_options.get("FuseL2Norm", True)
    fuse_gelu = extra_options.get("FuseGelu", True)
    fuse_layer_norm = extra_options.get("FuseLayerNorm", True)

    if fold_batch_norm or fuse_instance_norm or fuse_l2_norm or fuse_gelu or fuse_layer_norm:
        logger.info("Folding or fusing operators for better accuracy and performance.")

        float_model = optimize_model(
            float_model,
            op_types_to_quantize,
            nodes_to_quantize,
            nodes_to_exclude,
            convert_bn_to_conv=False,
            convert_reduce_mean_to_global_avg_pool=False,
            split_large_kernel_pool=False,
            convert_split_to_slice=False,
            fuse_instance_norm=fuse_instance_norm,
            fuse_l2_norm=fuse_l2_norm,
            fuse_gelu=fuse_gelu,
            fuse_layer_norm=fuse_layer_norm,
            fold_batch_norm=fold_batch_norm,
            convert_clip_to_relu=False,
            fold_batch_norm_after_concat=fold_batch_norm,
            dedicate_dq_node=False,
        )

    # Step4. Fixing all the tensors' shape
    if "FixShapes" in extra_options:
        fix_name_shape = extra_options["FixShapes"]
        try:
            model_temp = fix_input_and_output_shapes(float_model, fix_name_shape)
            tensor_name_shape_dict = infer_all_tensors_shape(model_temp, use_external_data_format)
            float_model = save_all_tensors_shape(model_temp, tensor_name_shape_dict)
        except Exception as e:
            logger.warning(f"Fail to fix shapes of the model beacuse of {e}.")

    return float_model


@log_errors
def apply_pre_quantization_algorithms(
    float_model: onnx.ModelProto,
    data_reader: CachedDataReader,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    op_types_to_quantize: list[str] | None = None,
    use_external_data_format: bool = False,
    include_cle: bool = False,
    include_sq: bool = False,
    include_rotation: bool = False,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This is an interface function used for applying pre-quant algorithms on the float model.

    :param onnx.ModelProto float_model: The float model.
    :param CachedDataReader data_reader: Data reader for the algorithm.
    :param list[str] nodes_to_quantize: List of nodes names to quantize. When this list is not None only the nodes in this list.
    :param list[str] nodes_to_exclude: List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.
    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.
    :param bool use_external_data_format: Option used for large size (>2GB) model.
    :param bool include_cle: Option for enabling algorithm CLE.
    :param bool include_sq: Option for enabling algorithm SmoothQuant.
    :param bool include_rotation: Option for enabling algorithm QuaRot.
    :param Dict[str, Any] extra_options: Options for the algorithm.

    :return: The transformed quantized model.
    """

    if include_cle:
        float_model = apply_pre_quant_algorithms(
            float_model,
            op_types_to_quantize=op_types_to_quantize,
            nodes_to_quantize=nodes_to_quantize,
            nodes_to_exclude=nodes_to_exclude,
            extra_options=extra_options,
            algorithm="CLE",
        )

    if include_sq:
        float_model = apply_pre_quant_algorithms(
            float_model,
            data_reader=data_reader,
            use_external_data_format=use_external_data_format,
            extra_options=extra_options,
            algorithm="SmoothQuant",
        )
        data_reader.reset_iter()

    if include_rotation:
        float_model = apply_pre_quant_algorithms(
            float_model,
            data_reader=data_reader,
            use_external_data_format=use_external_data_format,
            extra_options=extra_options,
            algorithm="QuaRot",
        )
        data_reader.reset_iter()

    return float_model


@log_errors
def apply_pre_optimization_after_algo(
    float_model: onnx.ModelProto,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    op_types_to_quantize: list[str] | None = None,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This is a function to apply pre-optimization after appling algorithms.
    It's mainly designed to meet the requirements of the compilers.

    :param onnx.ModelProto float_model: The float model to be optimized.
    :param list[str] nodes_to_quantize: List of nodes names to quantize. When this list is not None only the nodes in this list.
    :param list[str] nodes_to_exclude: List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.
    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.
    :param Dict[str, Any] extra_options: Options for the pre-process.

    :return: The quantized model.
    """

    convert_split_to_slice = extra_options.get("ConvertSplitToSlice", False)
    convert_bn_to_conv = extra_options.get("ConvertBNToConv", False)
    convert_reduce_mean_to_global_avg_pool = extra_options.get("ConvertReduceMeanToGlobalAvgPool", False)
    split_large_kernel_pool = extra_options.get("SplitLargeKernelPool", False)

    if (
        convert_bn_to_conv
        or convert_reduce_mean_to_global_avg_pool
        or split_large_kernel_pool
        or convert_split_to_slice
    ):
        logger.info("Optimizing the model for a better hardware compatibility.")

        float_model = optimize_model(
            float_model,
            op_types_to_quantize,
            nodes_to_quantize,
            nodes_to_exclude,
            convert_bn_to_conv=convert_bn_to_conv,
            convert_reduce_mean_to_global_avg_pool=convert_reduce_mean_to_global_avg_pool,
            split_large_kernel_pool=split_large_kernel_pool,
            convert_split_to_slice=convert_split_to_slice,
            fuse_instance_norm=False,
            fuse_l2_norm=False,
            fuse_gelu=False,
            fuse_layer_norm=False,
            fold_batch_norm=False,
            convert_clip_to_relu=False,
            fold_batch_norm_after_concat=False,
            dedicate_dq_node=False,
        )

    return float_model


@log_errors
@Profiler(msg=[["pre process"]])
def apply_pre_process(
    float_model: onnx.ModelProto,
    model_path: Path,
    data_reader: CalibrationDataReader,
    calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    op_types_to_quantize: list[str] | None = None,
    use_external_data_format: bool = False,
    convert_fp16_to_fp32: bool = False,
    convert_nchw_to_nhwc: bool = False,
    optimize_model_flag: bool = False,
    include_cle: bool = False,
    include_sq: bool = False,
    include_rotation: bool = False,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This is an interface function used for applying pre-process on the float model.

    :param onnx.ModelProto float_model: The float model to be optimized.
    :param Path model_path: The path of storing the float model.
    :param CalibrationDataReader data_reader: Data reader for the algorithm.
    :param CalibrationMethod calibrate_method: Calibration method, the default is CalibrationMethod.MinMax.
    :param QuantType activation_type: The quantization type for activation tensors.
    :param QuantType weight_type: The quantization type for weight tensors.
    :param list[str] nodes_to_quantize: List of nodes names to quantize. When this list is not None only the nodes in this list.
    :param list[str] nodes_to_exclude: List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.
    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.
    :param bool use_external_data_format: Option used for large size (>2GB) model.
    :param bool convert_fp16_to_fp32: Convert the fp16 model to a fp32 one. Default is False.
    :param bool convert_nchw_to_nhwc: Convert the dimension order of inputs from 'nchw' to 'nhwc', which has better performance on NPU. Default is False.
    :param bool optimize_model_flag: Optimize the model using onnxruntime's graph optimization. Default is False.
    :param bool include_cle: Option for enabling algorithm CLE.
    :param bool include_sq: Option for enabling algorithm SmoothQuant.
    :param bool include_rotation: Option for enabling algorithm QuaRot.
    :param Dict[str, Any] extra_options: Options for the pre-process.

    :return: The optimized model.
    """

    float_model = apply_pre_optimization_before_algo(
        float_model,
        model_path,
        calibrate_method,
        activation_type,
        weight_type,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        use_external_data_format,
        convert_fp16_to_fp32,
        convert_nchw_to_nhwc,
        optimize_model_flag,
        extra_options,
    )

    float_model = apply_pre_quantization_algorithms(
        float_model,
        data_reader,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        use_external_data_format,
        include_cle,
        include_sq,
        include_rotation,
        extra_options,
    )

    float_model = apply_pre_optimization_after_algo(
        float_model,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options,
    )

    return float_model
