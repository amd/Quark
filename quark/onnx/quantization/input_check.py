#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path
from typing import Any

import onnx
from onnxruntime.quantization.calibrate import CalibrationMethod
from onnxruntime.quantization.quant_utils import QuantFormat, QuantType

from quark.common.utils.import_utils import _is_package_available
from quark.common.utils.log import ScreenLogger, log_errors
from quark.onnx.calibration import Int16Method, LayerWiseMethod, PowerOfTwoMethod
from quark.onnx.quantization.quant_utils import (
    DEQUANT_OP_TYPES,
    FN_OP_TYPES,
    QUANT_OP_TYPES,
    ExtendedQuantFormat,
    ExtendedQuantType,
)

logger = ScreenLogger(__name__)


def check_ir_version(input_model: str | Path | onnx.ModelProto) -> bool:
    model = input_model if isinstance(input_model, onnx.ModelProto) else onnx.load(input_model)
    ir_version = model.ir_version
    return ir_version >= 4


def check_opset_version(input_model: str | Path | onnx.ModelProto) -> bool:
    model = input_model if isinstance(input_model, onnx.ModelProto) else onnx.load(input_model)
    opset_version: int = model.opset_import[0].version
    return opset_version >= 10


def check_qdq_model(input_model: str | Path | onnx.ModelProto) -> bool:
    model = input_model if isinstance(input_model, onnx.ModelProto) else onnx.load(input_model)
    nodes = [node.op_type for node in model.graph.node]
    qdq_ops = QUANT_OP_TYPES + DEQUANT_OP_TYPES + FN_OP_TYPES
    is_qdq_model = any(op in qdq_ops for op in nodes)
    return is_qdq_model


@log_errors
def check_static_quant_arguments(
    model_input: str | Path | onnx.ModelProto,
    quant_format: QuantFormat | ExtendedQuantFormat,
    activation_type: QuantType | ExtendedQuantType,
    weight_type: QuantType | ExtendedQuantType,
    calibrate_method: CalibrationMethod | PowerOfTwoMethod | Int16Method | LayerWiseMethod,
    extra_options: dict[str, Any],
) -> None:
    quark_qwb_types = [
        ExtendedQuantType.QInt32,
        ExtendedQuantType.QUInt32,
        ExtendedQuantType.QFloat16,
        ExtendedQuantType.QBFloat16,
    ]
    if (
        activation_type in quark_qwb_types or weight_type in quark_qwb_types
    ) and quant_format != ExtendedQuantFormat.QDQ:
        raise ValueError("Only ExtendedQuantFormat.QDQ supports wide bits quantization types.")

    quark_fp_types = [
        ExtendedQuantType.QFloat16,
        ExtendedQuantType.QBFloat16,
        ExtendedQuantType.QBFP,
        ExtendedQuantType.QMX,
    ]
    if len(extra_options.get("TensorQuantOverrides", {})):
        for tensor_name, quant_overrides in extra_options["TensorQuantOverrides"].items():
            if not isinstance(quant_overrides, list):
                raise ValueError(f"Invalid quant overrides {quant_overrides} for tensor {tensor_name}.")

            for override in quant_overrides:
                if not isinstance(override, dict):
                    raise ValueError(f"The quant override {override} should be a dict.")

                quant_type = override.get("quant_type", weight_type)
                if "axis" in override and quant_type in quark_fp_types:
                    raise ValueError(f"The per-channel quant override {override} can not be applied on {quant_type}.")

    if not check_ir_version(model_input):
        logger.warning(
            "The ir version of input model is below 4. It is recommended to upgrade ir version to 7 or higher."
        )
    if not check_opset_version(model_input):
        logger.warning(
            "The opset version of input model is below 10. It is recommended to upgrade opset version to 17 or higher."
        )
    if check_qdq_model(model_input):
        logger.error(
            "The input model is already a quantized model. Please make sure that input model is a float model."
        )


@log_errors
def check_fast_fintune_arguments(
    activation_type: QuantType | ExtendedQuantType,
    weight_type: QuantType | ExtendedQuantType,
    extra_options: dict[str, Any],
) -> None:
    if weight_type in [ExtendedQuantType.QFloat16, ExtendedQuantType.QBFloat16]:
        if "AddQDQPairToWeight" in extra_options and not extra_options["AddQDQPairToWeight"]:
            logger.warning(
                "Fast finetune requires not to fold QuantizeLinear for weights. "
                "AddQDQPairToWeight has been changed to True."
            )
        extra_options["AddQDQPairToWeight"] = True
    else:
        if "AddQDQPairToWeight" in extra_options and extra_options["AddQDQPairToWeight"]:
            logger.warning(
                "Fast finetune requires folding QuantizeLinear for weights. "
                "AddQDQPairToWeight has been changed to False."
            )
        extra_options["AddQDQPairToWeight"] = False


@log_errors
def check_crypto_mode_arguments(
    model_input: str | Path | onnx.ModelProto,
    use_external_data_format: bool,
    extra_options: dict[str, Any],
) -> None:
    if not isinstance(model_input, onnx.ModelProto):
        raise ValueError("For the crypto mode, the input model should be in onnx.ModelProto format.")

    if use_external_data_format:
        raise ValueError(
            "Can not use external data for onnx model since we can't save exposed data to disk in crypto mode."
        )

    if extra_options.get("EncryptionAlgorithm", "") == "AES-256":
        if not _is_package_available("cryptography")[0]:
            raise ImportError(
                "The 'cryptography' package is required for crypto mode but not installed. "
                "Please install it via 'pip install \"amd-quark[crypto]\"' (pins the patched "
                "cryptography>=48.0.1 floor)."
            )

    if extra_options.get("FastFinetune", {}).get("MemOptLevel", 1) == 2:
        logger.warning("The 'MemOptLevel' cannot be set to 2 in crypto mode, change it back to the default value of 1.")
        extra_options["FastFinetune"]["MemOptLevel"] = 1

    if extra_options.get("CalibOptimizeMem", True):
        logger.warning("The optimization of memory consumption for calibration will be disabled in crypto mode.")
        extra_options["CalibOptimizeMem"] = False
