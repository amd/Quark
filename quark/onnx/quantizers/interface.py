#
# Modifications copyright(c) 2025 Advanced Micro Devices,Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
from typing import Any

import onnx
from onnxruntime.quantization.calibrate import (
    CalibrationDataReader,
    CalibrationMethod,
    TensorsData,
)
from onnxruntime.quantization.quant_utils import (
    QuantFormat,
    QuantizationMode,
    QuantType,
)
from onnxruntime.quantization.registry import IntegerOpsRegistry

from quark.onnx.calibration import Int16Method, LayerWiseMethod, PowerOfTwoMethod
from quark.onnx.quantization.quant_utils import ExtendedQuantFormat
from quark.onnx.utils.system_utils import Profiler
from quark.shares.utils.log import ScreenLogger, log_errors

from .extended_quantizer import ExtendedQDQQuantizer
from .matmul_nbits_quantizer import (
    DefaultWeightOnlyQuantConfig,
    GPTQWeightOnlyQuantConfig,
    HQQWeightOnlyQuantConfig,
    MatMulNBitsQuantizer,
)
from .npu_cnn_quantizer import XINT8QDQQuantizer
from .npu_transformer_quantizer import TransformerQDQQuantizer
from .onnx_quantizer import ExtendedONNXQuantizer, ONNXQuantizer
from .qdq_quantizer import BaseExtendedQDQQuantizer
from .registry import NPUCnnRegistry, NPUTransformerRegistry, QDQRegistry, QLinearOpsRegistry

logger = ScreenLogger(__name__)


def set_parameters_and_domain(
    float_model: onnx.ModelProto,
    extra_options: dict[str, Any] = {},
) -> None:
    """Preparation before the quantization"""
    from onnxruntime.quantization.quant_utils import ms_domain

    from quark.onnx.quantization.quant_utils import (
        COP_DOMAIN,
        add_or_update_opset_import,
        annotate_op_type,
        remove_qdq_op_type,
    )

    # There is a process of removing QDQ within the quantizers,
    # so we have to set the op types for the removal before-hand.
    if extra_options.get("RemoveQDQConvClip", True):
        remove_qdq_op_type.append("Clip")
    if extra_options.get("RemoveQDQConvRelu", True):
        remove_qdq_op_type.append("Relu")
    if extra_options.get("RemoveQDQConvLeakyRelu", True):
        remove_qdq_op_type.append("LeakyRelu")
    if extra_options.get("RemoveQDQConvPRelu", True):
        remove_qdq_op_type.append("PRelu")
    if extra_options.get("RemoveQDQConvGelu", False):
        remove_qdq_op_type.append("Gelu")
    if extra_options.get("RemoveQDQInstanceNorm", False):
        annotate_op_type.append("InstanceNormalization")

    add_or_update_opset_import(float_model, ms_domain, 1)
    add_or_update_opset_import(float_model, COP_DOMAIN, 1)


@log_errors
def create_matmul_nbits_quantizer(
    float_model: onnx.ModelProto,
    data_reader: CalibrationDataReader,
    extra_options: dict[str, Any] = {},
) -> Any:
    """
    This is an interface function used to create a quantizer for static quantization using matmul nbits scheme.

    :param onnx.ModelProto float_model: ONNX model to quantize.
    :param CalibrationDataReader data_reader: Data reader for model quantization.
    :param Dict[str, Any] extra_options: Extra options for quantization, it also contains additional options for quantizers.

    :return: The quantizer instance.
    """
    matmul_nbits_quantize_dict = extra_options.get("MatMulNBitsParams", {})
    assert isinstance(matmul_nbits_quantize_dict, dict), (
        "The parameter 'MatMulNBitsParams' in extra_options must be a dict."
    )

    if "GroupSize" in matmul_nbits_quantize_dict:
        matmul_nbits_group_size = matmul_nbits_quantize_dict["GroupSize"]
    else:
        matmul_nbits_group_size = 128

    if "Symmetric" in matmul_nbits_quantize_dict:
        matmul_nbits_symmetric = matmul_nbits_quantize_dict["Symmetric"]
    else:
        matmul_nbits_symmetric = True

    if "Bits" in matmul_nbits_quantize_dict:
        matmul_nbits_bits = matmul_nbits_quantize_dict["Bits"]
    else:
        matmul_nbits_bits = 4

    if "AccuracyLevel" in matmul_nbits_quantize_dict:
        matmul_nbits_accuracy_level = matmul_nbits_quantize_dict["AccuracyLevel"]
    else:
        matmul_nbits_accuracy_level = 0

    algo_config: DefaultWeightOnlyQuantConfig | GPTQWeightOnlyQuantConfig | HQQWeightOnlyQuantConfig | None = None

    if extra_options.get("MatMulNBitsParams", {}).get("Algorithm", "DEFAULT") == "GPTQ":
        algo_config = GPTQWeightOnlyQuantConfig(
            calibration_data_reader=data_reader,
            percdamp=extra_options.get("GPTQParams", {}).get("PercDamp", 0.01),
            block_size=extra_options.get("GPTQParams", {}).get("BlockSize", 128),
            actorder=extra_options.get("GPTQParams", {}).get("ActOrder", False),
            mse=extra_options.get("GPTQParams", {}).get("MSE", False),
            perchannel=extra_options.get("GPTQParams", {}).get("PerChannel", False),
        )
    elif extra_options.get("MatMulNBitsParams", {}).get("Algorithm", "DEFAULT") == "HQQ":
        algo_config = HQQWeightOnlyQuantConfig(
            block_size=matmul_nbits_group_size,
            bits=matmul_nbits_bits,
        )
    else:
        algo_config = DefaultWeightOnlyQuantConfig(
            block_size=matmul_nbits_group_size,
            is_symmetric=matmul_nbits_symmetric,
            bits=matmul_nbits_bits,
            accuracy_level=matmul_nbits_accuracy_level,
        )

    return MatMulNBitsQuantizer(
        float_model,
        matmul_nbits_group_size,
        matmul_nbits_symmetric,
        matmul_nbits_bits,
        accuracy_level=matmul_nbits_accuracy_level,
        algo_config=algo_config,
        extra_options=extra_options,
    )


@log_errors
def run_matmul_nbits_quantization(
    float_model: onnx.ModelProto,
    data_reader: CalibrationDataReader,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This is an interface function used for the matmul nbits quantization.

    :param onnx.ModelProto float_model: ONNX model to quantize.
    :param CalibrationDataReader data_reader: Data reader for model quantization.
    :param Dict[str, Any] extra_options: Extra options for quantization, it also contains additional options for quantizers.

    :return: The quantized model.
    """

    quantizer = create_matmul_nbits_quantizer(float_model, data_reader, extra_options)

    quantizer.quantize_model()

    return quantizer.model.model


@log_errors
def get_static_op_types(
    float_model: onnx.ModelProto,
    op_types_to_quantize: list[str] | None = None,
    extra_op_types_to_quantize: list[str] = [],
    enable_npu_cnn: bool = False,
    enable_npu_transformer: bool = False,
    quant_format: QuantFormat | ExtendedQuantFormat = QuantFormat.QDQ,
    extra_options: dict[str, Any] = {},
) -> list[str]:
    """
    This is an interface function used to determine the types of operators for the static quantization.

    :param onnx.ModelProto float_model: ONNX model to quantize.
    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.
    :param List[str] extra_op_types_to_quantize: Besides the supported operators, the extra types of operators to quantize. The default is an empty list.
    :param bool enable_npu_cnn: Flag to enable the quantization scheme for deploying CNN-based models on NPU. The default is ``False``.
    :param bool enable_npu_transformer: Flag to enable the quantization scheme for deploying Transformer-based models on NPU. The default is ``False``.
    :param Union[QuantFormat, ExtendedQuantType] quant_format: Format of quantization. Default is ``QuantFormat.QDQ``.

    :return: The updated list of types of operators to quantize.
    """

    op_types_all = []
    for node in float_model.graph.node:
        if node.op_type not in op_types_all:
            op_types_all.append(node.op_type)

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        if enable_npu_transformer:
            op_types_to_quantize = list(NPUTransformerRegistry.keys())
        else:
            q_linear_ops = list(QLinearOpsRegistry.keys())
            qdq_ops = list(QDQRegistry.keys())
            if enable_npu_cnn or quant_format is ExtendedQuantFormat.QDQ:
                dpu_ops = list(NPUCnnRegistry.keys())
                qdq_ops = list(set(dpu_ops + qdq_ops))
            op_types_to_quantize = list(set(q_linear_ops + qdq_ops))

    if extra_options.get("QuantizeAllOpTypes", False):
        extra_op_types_to_quantize.extend(op_types_all)

    op_types_absent = [op_type for op_type in extra_op_types_to_quantize if op_type not in op_types_all]
    if op_types_absent:
        logger.warning(f"The model does not contain the following op types: {', '.join(op_types_absent)}")

    op_types_to_quantize += extra_op_types_to_quantize
    op_types_to_quantize = list(set(op_types_to_quantize))  # To eliminate duplicates

    return op_types_to_quantize


@log_errors
def create_static_quantizer(
    float_model: onnx.ModelProto,
    tensors_range: TensorsData,
    per_channel: bool = False,
    reduce_range: bool = False,
    weight_type: QuantType = QuantType.QInt8,
    activation_type: QuantType = QuantType.QInt8,
    enable_npu_cnn: bool = False,
    enable_npu_transformer: bool = False,
    quant_format: QuantFormat | ExtendedQuantFormat = QuantFormat.QDQ,
    calibrate_method: CalibrationMethod | LayerWiseMethod | PowerOfTwoMethod = CalibrationMethod.MinMax,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    op_types_to_quantize: list[str] | None = None,
    extra_options: dict[str, Any] = {},
) -> Any:
    """
    This is an interface function used to create a quantizer for static quantization.

    :param onnx.ModelProto float_model: ONNX model to quantize.
    :param TensorsData tensors_range: Data range for all quantizing tensors.
    :param bool per_channel: Quantize weights per-channel.
    :param bool reduce_range: Quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode.
    :param QuantType weight_type: The quantization type of weight. Default is ``QuantType.QInt8``.
    :param QuantType activation_type: The quantization type of activation. Default is ``QuantType.QInt8``.
    :param bool enable_npu_cnn: Flag to enable the quantization scheme for deploying CNN-based models on NPU. The default is ``False``.
    :param bool enable_npu_transformer: Flag to enable the quantization scheme for deploying Transformer-based models on NPU. The default is ``False``.
    :param Union[QuantFormat, ExtendedQuantType] quant_format: Format of quantization. Default is ``QuantFormat.QDQ``.
    :param Union[CalibrationMethod, LayerWiseMethod, PowerOfTwoMethod] calibrate_method: Calibration method to use.
    :param list[str] nodes_to_quantize: List of nodes names to quantize. When this list is not None only the nodes in this list.
    :param list[str] nodes_to_exclude: List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.
    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.
    :param Dict[str, Any] extra_options: Extra options for quantization, it also contains additional options for quantizers.

    :return: The quantizer instance.
    """

    mode = QuantizationMode.QLinearOps
    static = True

    if (calibrate_method in CalibrationMethod) or (calibrate_method in LayerWiseMethod):
        if quant_format is QuantFormat.QOperator:
            return ONNXQuantizer(
                float_model,
                per_channel,
                reduce_range,
                mode,
                static,
                weight_type,
                activation_type,
                tensors_range,
                nodes_to_quantize,
                nodes_to_exclude,
                op_types_to_quantize,
                extra_options,
            )
        elif quant_format is QuantFormat.QDQ:
            if enable_npu_transformer:
                return TransformerQDQQuantizer(
                    float_model,
                    per_channel,
                    reduce_range,
                    mode,
                    static,
                    weight_type,
                    activation_type,
                    tensors_range,
                    nodes_to_quantize,
                    nodes_to_exclude,
                    op_types_to_quantize,
                    extra_options,
                )
            else:
                return BaseExtendedQDQQuantizer(
                    float_model,
                    per_channel,
                    reduce_range,
                    mode,
                    static,
                    weight_type,
                    activation_type,
                    tensors_range,
                    nodes_to_quantize,
                    nodes_to_exclude,
                    op_types_to_quantize,
                    calibrate_method,
                    extra_options,
                )
        elif quant_format is ExtendedQuantFormat.QDQ:
            return ExtendedQDQQuantizer(
                float_model,
                per_channel,
                reduce_range,
                mode,
                static,
                weight_type,
                activation_type,
                tensors_range,
                nodes_to_quantize,
                nodes_to_exclude,
                op_types_to_quantize,
                calibrate_method,
                extra_options,
            )
        else:
            raise ValueError("No corresponding quantizer for this set of arguments.")
    elif calibrate_method in PowerOfTwoMethod or calibrate_method in Int16Method:
        if quant_format is QuantFormat.QOperator:
            return ExtendedONNXQuantizer(
                float_model,
                per_channel,
                reduce_range,
                mode,
                static,
                weight_type,
                activation_type,
                tensors_range,
                nodes_to_quantize,
                nodes_to_exclude,
                op_types_to_quantize,
                calibrate_method,
                extra_options,
            )
        elif quant_format is QuantFormat.QDQ:
            if enable_npu_cnn:
                return XINT8QDQQuantizer(
                    float_model,
                    per_channel,
                    reduce_range,
                    mode,
                    static,
                    weight_type,
                    activation_type,
                    tensors_range,
                    nodes_to_quantize,
                    nodes_to_exclude,
                    op_types_to_quantize,
                    calibrate_method,
                    extra_options,
                )
            else:
                return BaseExtendedQDQQuantizer(
                    float_model,
                    per_channel,
                    reduce_range,
                    mode,
                    static,
                    weight_type,
                    activation_type,
                    tensors_range,
                    nodes_to_quantize,
                    nodes_to_exclude,
                    op_types_to_quantize,
                    calibrate_method,
                    extra_options,
                )
        elif quant_format is ExtendedQuantFormat.QDQ:
            return ExtendedQDQQuantizer(
                float_model,
                per_channel,
                reduce_range,
                mode,
                static,
                weight_type,
                activation_type,
                tensors_range,
                nodes_to_quantize,
                nodes_to_exclude,
                op_types_to_quantize,
                calibrate_method,
                extra_options,
            )
        else:
            raise ValueError("No corresponding quantizer for this set of arguments.")


@log_errors
@Profiler(msg=[["static quantization"]])
def run_static_quantization(
    float_model: onnx.ModelProto,
    tensors_range: TensorsData,
    per_channel: bool = False,
    reduce_range: bool = False,
    weight_type: QuantType = QuantType.QInt8,
    activation_type: QuantType = QuantType.QInt8,
    enable_npu_cnn: bool = False,
    enable_npu_transformer: bool = False,
    quant_format: QuantFormat | ExtendedQuantFormat = QuantFormat.QDQ,
    calibrate_method: CalibrationMethod | LayerWiseMethod | PowerOfTwoMethod = CalibrationMethod.MinMax,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    op_types_to_quantize: list[str] | None = None,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This is an interface function used for static quantization.

    :param onnx.ModelProto float_model: ONNX model to quantize.
    :param TensorsData tensors_range: Data range for all quantizing tensors.
    :param bool per_channel: Quantize weights per-channel.
    :param bool reduce_range: Quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode.
    :param QuantType weight_type: The quantization type of weight. Default is ``QuantType.QInt8``.
    :param QuantType activation_type: The quantization type of activation. Default is ``QuantType.QInt8``.
    :param bool enable_npu_cnn: Flag to enable the quantization scheme for deploying CNN-based models on NPU. The default is ``False``.
    :param bool enable_npu_transformer: Flag to enable the quantization scheme for deploying Transformer-based models on NPU. The default is ``False``.
    :param Union[QuantFormat, ExtendedQuantType] quant_format: Format of quantization. Default is ``QuantFormat.QDQ``.
    :param Union[CalibrationMethod, LayerWiseMethod, PowerOfTwoMethod] calibrate_method: Calibration method to use.
    :param list[str] nodes_to_quantize: List of nodes names to quantize. When this list is not None only the nodes in this list.
    :param list[str] nodes_to_exclude: List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.
    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.
    :param Dict[str, Any] extra_options: Extra options for quantization, it also contains additional options for quantizers.

    :return: The quantized model.
    """

    # TODO: need to refactor this part
    set_parameters_and_domain(float_model, extra_options)

    quantizer = create_static_quantizer(
        float_model,
        tensors_range,
        per_channel,
        reduce_range,
        weight_type,
        activation_type,
        enable_npu_cnn,
        enable_npu_transformer,
        quant_format,
        calibrate_method,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options,
    )

    quantizer.quantize_model()

    return quantizer.model.model


@log_errors
def get_dynamic_op_types(op_types_to_quantize: list[str] | None = None) -> list[str]:
    """
    This is an interface function used to determine the types of operators for the dynamic quantization.

    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.

    :return: The updated list of types of operators to quantize.
    """

    if not op_types_to_quantize or len(op_types_to_quantize) == 0:
        op_types_to_quantize = list(IntegerOpsRegistry.keys())

    return op_types_to_quantize


@log_errors
def create_dynamic_quantizer(
    float_model: onnx.ModelProto,
    per_channel: bool = False,
    reduce_range: bool = False,
    weight_type: QuantType = QuantType.QInt8,
    activation_type: QuantType = QuantType.QUInt8,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    op_types_to_quantize: list[str] | None = None,
    extra_options: dict[str, Any] = {},
) -> Any:
    """
    This is an interface function used to create a quantizer for dynamic quantization.

    :param onnx.ModelProto float_model: ONNX model to quantize.
    :param bool per_channel: Quantize weights per-channel.
    :param bool reduce_range: Quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode.
    :param QuantType weight_type: The quantization type of weight. Default is ``QuantType.QInt8``.
    :param QuantType activation_type: The quantization type of activation. Default is ``QuantType.QUInt8``.
    :param Union[CalibrationMethod, LayerWiseMethod, PowerOfTwoMethod] calibrate_method: Calibration method to use.
    :param list[str] nodes_to_quantize: List of nodes names to quantize. When this list is not None only the nodes in this list.
    :param list[str] nodes_to_exclude: List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.
    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.
    :param Dict[str, Any] extra_options: Extra options for quantization, it also contains additional options for quantizers.

    :return: The quantizer instance.
    """

    mode = QuantizationMode.IntegerOps
    static = False

    if activation_type is not QuantType.QUInt8:
        logger.warning("The activation type of dynamic quantization only supports QUInt8.")

    return ONNXQuantizer(
        float_model,
        per_channel,
        reduce_range,
        mode,
        static,
        weight_type,
        QuantType.QUInt8,
        None,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options,
    )


@log_errors
def run_dynamic_quantization(
    float_model: onnx.ModelProto,
    per_channel: bool = False,
    reduce_range: bool = False,
    weight_type: QuantType = QuantType.QInt8,
    activation_type: QuantType = QuantType.QUInt8,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    op_types_to_quantize: list[str] | None = None,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This is an interface function used for dynamic quantization.

    :param onnx.ModelProto float_model: ONNX model to quantize.
    :param bool per_channel: Quantize weights per-channel.
    :param bool reduce_range: Quantize weights with 7-bits. It may improve the accuracy for some models running on non-VNNI machine, especially for per-channel mode.
    :param QuantType weight_type: The quantization type of weight. Default is ``QuantType.QInt8``.
    :param QuantType activation_type: The quantization type of activation. Default is ``QuantType.QUInt8``.
    :param Union[CalibrationMethod, LayerWiseMethod, PowerOfTwoMethod] calibrate_method: Calibration method to use.
    :param list[str] nodes_to_quantize: List of nodes names to quantize. When this list is not None only the nodes in this list.
    :param list[str] nodes_to_exclude: List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not None.
    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.
    :param Dict[str, Any] extra_options: Extra options for quantization, it also contains additional options for quantizers.

    :return: The quantized model.
    """

    quantizer = create_dynamic_quantizer(
        float_model,
        per_channel,
        reduce_range,
        weight_type,
        activation_type,
        nodes_to_quantize,
        nodes_to_exclude,
        op_types_to_quantize,
        extra_options,
    )

    quantizer.quantize_model()

    return quantizer.model.model
