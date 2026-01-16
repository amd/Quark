#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import time
from typing import Any

import onnx
from onnxruntime.quantization.calibrate import CalibrationDataReader, CalibrationMethod
from onnxruntime.quantization.onnx_model import ONNXModel
from onnxruntime.quantization.quant_utils import QuantType

from quark.onnx.utils.system_utils import Profiler
from quark.shares.utils.log import ScreenLogger, log_errors

from .bc.bias_correction import bias_correction
from .cle.equalization import cle_transforms
from .finetuning.fast_finetune import fast_finetune
from .gptq.gptq import GptqProcessor
from .mprecision.auto_mixprecision import auto_mixprecision
from .quarot.quarot import rotation_transforms
from .sq.smooth_quant import smooth_transforms

logger = ScreenLogger(__name__)


def apply_CLE(
    float_model: onnx.ModelProto,
    op_types_to_quantize: list[str] | None = None,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This algorithm is referenced from the cross-layer equalization proposed in the following paper:
    "Markus Nagel et al., Data-Free Quantization Through Weight Equalization and Bias Correction,
    arXiv:1906.04721, 2019."

    :param onnx.ModelProto float_model: ONNX model for the transform.
    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.
    :param List[str] nodes_to_quantize: List of nodes names to quantize. When this list is not ``None`` and empty only the nodes in this list.
    :param List[str] nodes_to_exclude: List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not ``None`` and empty.
    :param Dict[str, Any] extra_options: Options for the algorithm.

    :return: The transformed model.
    """

    if extra_options.get("ReplaceClip6Relu", False):
        from .cle.equalization import replace_all_clip6_to_relu

        model_replaced = replace_all_clip6_to_relu(
            float_model, op_types_to_quantize, nodes_to_quantize, nodes_to_exclude
        )

        topo_model = ONNXModel(model_replaced)
        topo_model.topological_sort()

        float_model = topo_model.model

    cle_steps = extra_options.get("CLESteps", 1)
    cle_balance_method = extra_options.get("CLEBalanceMethod", "max")
    cle_weight_threshold = extra_options.get("CLEWeightThreshold", 0.5)
    cle_scale_append_bias = extra_options.get("CLEScaleAppendBias", True)
    cle_scale_use_threshold = extra_options.get("CLEScaleUseThreshold", True)
    cle_total_layer_diff_threshold = extra_options.get("CLETotalLayerDiffThreshold", 2e-7)

    return cle_transforms(
        float_model,
        op_types_to_quantize if op_types_to_quantize is not None else [],
        nodes_to_quantize,
        nodes_to_exclude,
        cle_steps,
        cle_balance_method,
        cle_weight_threshold,
        cle_scale_append_bias,
        cle_scale_use_threshold,
        cle_total_layer_diff_threshold,
    )


@Profiler(msg=[["", "", "pre process: smooth quant"]])
def apply_SmoothQuant(
    float_model: onnx.ModelProto,
    data_reader: CalibrationDataReader,
    use_external_data_format: bool = False,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This algorithm is referenced from the Smooth Quant proposed in the following paper:
    "Guangxuan Xiao et al., SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models,
    arXiv:2211.10438, 2022."

    :param onnx.ModelProto float_model: ONNX model for the transform.
    :param CalibrationDataReader data_reader: Data reader for the algorithm.
    :param bool use_external_data_format: Option used for large size (>2GB) model.
    :param Dict[str, Any] extra_options: Options for the algorithm.

    :return: The transformed model.
    """

    smooth_alpha = extra_options.get("SmoothAlpha", 0.5)
    assert data_reader is not None, "No data reader for the SmoothQuant."

    return smooth_transforms(
        float_model,
        data_reader,
        alpha=smooth_alpha,
        use_external_data_format=use_external_data_format,
    )


def apply_QuaRot(
    float_model: onnx.ModelProto, use_external_data_format: bool = False, extra_options: dict[str, Any] = {}
) -> onnx.ModelProto:
    """
    This algorithm is referenced from the QuaRot proposed in the following paper:
    "Saleh Ashkboos et al., QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs,
    arXiv:2404.00456, 2024."

    :param onnx.ModelProto float_model: ONNX model for the transform.
    :param bool use_external_data_format: Option used for large size (>2GB) model.
    :param Dict[str, Any] extra_options: Options for the algorithm.

    :return: The transformed model.
    """

    hidden_size = extra_options.get("RMatrixDim", 4096)
    random_had = extra_options.get("UseRandomHad", False)
    rotation_config_file = extra_options.get("RConfigPath")
    assert rotation_config_file is not None, "Error! Please specify the rotation config via option 'RConfigPath'"

    try:
        from quark.torch.algorithm.rotation.rotation_utils import get_rotation_matrix

        r1_matrix = get_rotation_matrix(num_channels=hidden_size, random=random_had, device="cpu")
    except Exception as e:
        raise AssertionError(f"Error! The dim of the target R1 matrix is not support due to {e}.")
    r_mat = {"R1": r1_matrix.numpy()}

    return rotation_transforms(
        float_model,
        r_mat,
        rotation_config_file,
        use_external_data_format=use_external_data_format,
    )


@log_errors
def apply_pre_quant_algorithms(
    float_model: onnx.ModelProto,
    data_reader: CalibrationDataReader | None = None,
    op_types_to_quantize: list[str] | None = None,
    nodes_to_quantize: list[str] = [],
    nodes_to_exclude: list[str] = [],
    use_external_data_format: bool = False,
    extra_options: dict[str, Any] | None = {},
    algorithm: str = "CLE",
) -> onnx.ModelProto:
    """
    This is an interface function used for applying pre-quant algorithms on the float model.

    :param onnx.ModelProto float_model: ONNX model for the transform.
    :param CalibrationDataReader data_reader: Data reader for some algorithms.
    :param Optional[List[str]] op_types_to_quantize: Specify the types of operators to quantize. It quantizes all supported operators by default.
    :param List[str] nodes_to_quantize: List of nodes names to quantize. When this list is not ``None`` and empty only the nodes in this list.
    :param List[str] nodes_to_exclude: List of nodes names to exclude. The nodes in this list will be excluded from quantization when it is not ``None`` and empty.
    :param bool use_external_data_format: Option used for large size (>2GB) model.
    :param Dict[str, Any] extra_options: Extra options for quantization, which contains additional options for the algorithms.
    :param str algorithm: The algorithm to apply, the valid options are 'CLE', 'SmoothQuant' and 'QuaRot'.

    :return: The transformed model.
    """

    logger.info(f"Start applying {algorithm} algorithm on the float model...")
    start_time = time.perf_counter()

    if algorithm == "CLE":
        transformed_model = apply_CLE(
            float_model, op_types_to_quantize, nodes_to_quantize, nodes_to_exclude, extra_options
        )
    elif algorithm == "SmoothQuant":
        transformed_model = apply_SmoothQuant(float_model, data_reader, use_external_data_format, extra_options)
    elif algorithm == "QuaRot":
        transformed_model = apply_QuaRot(float_model, use_external_data_format, extra_options)
    else:
        raise ValueError(f"Unsupported {algorithm} algorithm.")

    end_time = time.perf_counter()
    total_time = end_time - start_time
    logger.info(f"The {algorithm} algorithm has been applied. It took {total_time:.1f}s to complete.")

    return transformed_model


def apply_BiasCorrection(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    data_reader: CalibrationDataReader,
    calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
    activation_type: QuantType = QuantType.QInt8,
    use_external_data_format: bool = False,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This algorithm is referenced from the bias correction proposed in the following paper:
    "Markus Nagel et al., Data-Free Quantization Through Weight Equalization and Bias Correction,
    arXiv:1906.04721, 2019."

    Based on the original algorithm, we have made improvements to deliver better accuracy and applicability.

    :param onnx.ModelProto float_model: The float model for reference.
    :param onnx.ModelProto quant_model: The quantized model to be optimized.
    :param CalibrationDataReader data_reader: Data reader for the algorithm.
    :param CalibrationMethod calibrate_method: Calibration method, the default is CalibrationMethod.MinMax.
    :param QuantType activation_type: The quantization type to mix in activation tensors.
    :param bool use_external_data_format: Option used for large size (>2GB) model.
    :param Dict[str, Any] extra_options: Options for the algorithm.

    :return: The quantized model that has been optimized.
    """

    return bias_correction(
        float_model,
        quant_model,
        use_external_data_format,
        data_reader,
        activation_type,
        calibrate_method,
        extra_options,
    )


def apply_AutoMixedPrecision(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    data_reader: CalibrationDataReader,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
    use_external_data_format: bool = False,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This function applies auto mixed precision to balance efficiency and performance.

    :param onnx.ModelProto float_model: The float model for reference.
    :param onnx.ModelProto quant_model: The quantized model to be optimized.
    :param CalibrationDataReader data_reader: Data reader for the algorithm.
    :param QuantType activation_type: The quantization type to mix in activation tensors.
    :param QuantType weight_type: The quantization type to mix in weight tensors.
    :param bool use_external_data_format: Option used for large size (>2GB) model.
    :param Dict[str, Any] extra_options: Options for the algorithm.

    :return: The quantized model that has been optimized.
    """

    return auto_mixprecision(
        float_model,
        quant_model,
        use_external_data_format,
        data_reader,
        activation_type,
        weight_type,
        extra_options,
    )


def apply_FastFinetune(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    data_reader: CalibrationDataReader,
    use_external_data_format: bool = False,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This function applies two widely used PTQ algorithms, AdaRound and AdaQuant.

    The AdaRound is referenced from the following paper:
    "Markus Nagel et al., Up or Down? Adaptive Rounding for Post-Training Quantization,
    arXiv:2004.10568, 2020."

    The AdaQuant is referenced from the following paper:
    "Itay Hubara et al., Improving Post Training Neural Quantization: Layer-wise Calibration and Integer Programming,
    arXiv:2006.10518, 2020."

    Based on the original algorithms, we have made improvements to deliver better accuracy and applicability.

    :param onnx.ModelProto float_model: The float model for reference.
    :param onnx.ModelProto quant_model: The quantized model to be optimized.
    :param CalibrationDataReader data_reader: Data reader for the algorithm.
    :param bool use_external_data_format: Option used for large size (>2GB) model.
    :param Dict[str, Any] extra_options: Options for the algorithm.

    :return: The quantized model that has been optimized.
    """

    return fast_finetune(float_model, quant_model, use_external_data_format, data_reader, extra_options)


def apply_GPTQ(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    data_reader: CalibrationDataReader,
    use_external_data_format: bool = False,
    extra_options: dict[str, Any] = {},
) -> onnx.ModelProto:
    """
    This algorithm is referenced from the GPTQ proposed in the following paper:
    "Elias Frantar et al., GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers,
    arXiv:2210.17323, 2022."

    :param onnx.ModelProto float_model: The float model for reference.
    :param onnx.ModelProto quant_model: The quantized model to be optimized.
    :param CalibrationDataReader data_reader: Data reader for the algorithm.
    :param bool use_external_data_format: Option used for large size (>2GB) model.
    :param Dict[str, Any] extra_options: Options for the algorithm.

    :return: The quantized model that has been optimized.
    """

    gptq_processor = GptqProcessor(
        float_model,
        quant_model,
        data_reader,
        extra_options,
        use_external_data_format=use_external_data_format,
    )
    return gptq_processor.apply()


def apply_post_quant_algorithms(
    float_model: onnx.ModelProto,
    quant_model: onnx.ModelProto,
    data_reader: CalibrationDataReader,
    calibrate_method: CalibrationMethod = CalibrationMethod.MinMax,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
    use_external_data_format: bool = False,
    extra_options: dict[str, Any] = {},
    algorithm: str = "FastFinetune",
) -> onnx.ModelProto:
    """
    This is an interface function used for applying post-quant algorithms on the quantized model.

    :param onnx.ModelProto float_model: The float model for reference.
    :param onnx.ModelProto quant_model: The quantized model to be optimized.
    :param CalibrationDataReader data_reader: Data reader for the algorithm.
    :param CalibrationMethod calibrate_method: Calibration method, the default is CalibrationMethod.MinMax.
    :param QuantType activation_type: The quantization type to mix in activation tensors.
    :param QuantType weight_type: The quantization type to mix in weight tensors.
    :param bool use_external_data_format: Option used for large size (>2GB) model.
    :param Dict[str, Any] extra_options: Options for the algorithm.
    :param str algorithm: The algorithm to apply, the valid options are 'BiasCorrection', 'AutoMixprecision', 'FastFinetune' and 'GPTQ'.

    :return: The quantized model that has been optimized.
    """

    logger.info(f"Start applying {algorithm} algorithm on the quant model...")
    start_time = time.perf_counter()

    if algorithm == "BiasCorrection":
        optimized_model = apply_BiasCorrection(
            float_model,
            quant_model,
            data_reader,
            calibrate_method,
            activation_type,
            use_external_data_format,
            extra_options,
        )
    elif algorithm == "AutoMixprecision":
        optimized_model = apply_AutoMixedPrecision(
            float_model, quant_model, data_reader, activation_type, weight_type, use_external_data_format, extra_options
        )
    elif algorithm == "FastFinetune":
        optimized_model = apply_FastFinetune(
            float_model, quant_model, data_reader, use_external_data_format, extra_options
        )
    elif algorithm == "GPTQ":
        optimized_model = apply_GPTQ(float_model, quant_model, data_reader, use_external_data_format, extra_options)
    else:
        raise ValueError(f"Unsupported {algorithm} algorithm.")

    end_time = time.perf_counter()
    total_time = end_time - start_time
    logger.info(f"The {algorithm} algorithm has been applied. It took {total_time:.1f}s to complete.")

    return optimized_model
