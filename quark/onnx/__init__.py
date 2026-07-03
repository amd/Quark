#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

try:
    from onnxruntime.quantization.calibrate import (
        CalibraterBase,
        CalibrationDataReader,
        CalibrationMethod,
        MinMaxCalibrater,
    )
    from onnxruntime.quantization.quant_utils import QuantFormat, QuantizationMode, QuantType, write_calibration_table

except ModuleNotFoundError:
    raise ImportError(
        "Quark depends on ONNXRuntime. Please install ONNXRuntime by following the instructions at: https://onnxruntime.ai/docs/install/"
    ) from None

from quark.onnx.calibration import (
    CachedDataReader,
    ExtendedCalibrationMethod,
    Int16Method,
    LayerWiseMethod,
    PathDataReader,
    PowerOfTwoMethod,
    RandomDataReader,
    create_calibrator_float_scale,
    create_calibrator_power_of_two,
)
from quark.onnx.operators.custom_ops import _COP_DOMAIN, _COP_VERSION, get_library_path
from quark.onnx.quantization.api import ModelQuantizer
from quark.onnx.quantization.auto_search import (
    AutoSearch,
    AutoSearchPro,
    SearchSpace,
    generate_all_configs,
    get_auto_search_config,
)
from quark.onnx.quantization.config.algorithm import (
    AdaQuantConfig,
    AdaRoundConfig,
    AlgoConfig,
    AutoMixprecisionConfig,
    BiasCorrectionConfig,
    CLEConfig,
    GPTQConfig,
    QuarotConfig,
    SmoothQuantConfig,
)
from quark.onnx.quantization.config.config import Config, QConfig
from quark.onnx.quantization.config.custom_config import (
    A8W8_ADAQUANT_QCONFIG,
    A8W8_ADAROUND_QCONFIG,
    A8W8_QCONFIG,
    A16W8_ADAQUANT_QCONFIG,
    A16W8_ADAROUND_QCONFIG,
    A16W8_QCONFIG,
    BF16_QCONFIG,
    BFP16_QCONFIG,
    VINT8_QCONFIG,
    XINT8_ADAQUANT_QCONFIG,
    XINT8_ADAROUND_QCONFIG,
    XINT8_QCONFIG,
)
from quark.onnx.quantization.config.data_type import (
    BFP16,
    BFloat16,
    DataType,
    Float16,
    Int8,
    Int16,
    Int32,
    UInt8,
    UInt16,
    UInt32,
)
from quark.onnx.quantization.config.legacy import QuantizationConfig
from quark.onnx.quantization.config.spec import (
    BFloat16Spec,
    BFP16Spec,
    CalibMethod,
    Int8Spec,
    Int16Spec,
    Int32Spec,
    QLayerConfig,
    QTensorConfig,
    QuantGranularity,
    ScaleType,
    UInt8Spec,
    UInt16Spec,
    UInt32Spec,
    XInt8Spec,
)
from quark.onnx.quantization.quant_utils import ExtendedQuantFormat, ExtendedQuantType, VitisQuantFormat, VitisQuantType
from quark.onnx.quantization.quantize import quantize_dynamic, quantize_static
from quark.onnx.utils.deploy_utils import dump_model

__all__ = [
    "CalibraterBase",
    "CalibrationDataReader",
    "CalibrationMethod",
    "MinMaxCalibrater",
    "QuantizationMode",
    "QuantFormat",
    "QuantType",
    "write_calibration_table",
    "Int16Method",
    "PowerOfTwoMethod",
    "LayerWiseMethod",
    "ExtendedCalibrationMethod",
    "CachedDataReader",
    "RandomDataReader",
    "PathDataReader",
    "create_calibrator_power_of_two",
    "create_calibrator_float_scale",
    "VitisExtendedQuantizer",
    "ExtendedQuantType",
    "ExtendedQuantFormat",
    "VitisQuantType",
    "VitisQuantFormat",
    "quantize_static",
    "quantize_dynamic",
    "AutoSearch",
    "AutoSearchPro",
    "generate_all_configs",
    "get_auto_search_config",
    "SearchSpace",
    "ModelQuantizer",
    "QConfig",
    "Config",
    "QuantizationConfig",
    "CalibMethod",
    "ScaleType",
    "QuantGranularity",
    "QTensorConfig",
    "Int8Spec",
    "UInt8Spec",
    "XInt8Spec",
    "Int16Spec",
    "UInt16Spec",
    "Int32Spec",
    "UInt32Spec",
    "BFloat16Spec",
    "BFP16Spec",
    "QLayerConfig",
    "DataType",
    "Int8",
    "UInt8",
    "Int16",
    "UInt16",
    "Int32",
    "UInt32",
    "Float16",
    "BFloat16",
    "BFP16",
    "AlgoConfig",
    "SmoothQuantConfig",
    "CLEConfig",
    "BiasCorrectionConfig",
    "GPTQConfig",
    "AutoMixprecisionConfig",
    "AdaRoundConfig",
    "AdaQuantConfig",
    "QuarotConfig",
    "XINT8_QCONFIG",
    "XINT8_ADAROUND_QCONFIG",
    "XINT8_ADAQUANT_QCONFIG",
    "VINT8_QCONFIG",
    "A8W8_QCONFIG",
    "A8W8_ADAROUND_QCONFIG",
    "A8W8_ADAQUANT_QCONFIG",
    "A16W8_QCONFIG",
    "A16W8_ADAROUND_QCONFIG",
    "A16W8_ADAQUANT_QCONFIG",
    "BF16_QCONFIG",
    "BFP16_QCONFIG",
    "dump_model",
    "_COP_DOMAIN",
    "_COP_VERSION",
    "get_library_path",
]
