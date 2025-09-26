#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from typing import Any, Dict, List, Optional

from onnx import ModelProto
from onnxruntime.quantization.quant_utils import QuantizationMode

from .qdq_quantizer import VitisQDQQuantizer


class VitisQDQCPUQuantizer(VitisQDQQuantizer):
    """
    VitisQDQCPUQuantizer is specific for CPU quantization config.
    Class VitisQDQCPUQuantizer inherits from Class VitisQDQQuantizer and
    can handle float onnx models with inf/-inf initialization.
    """

    def __init__(
        self,
        model: ModelProto,
        per_channel: bool,
        reduce_range: bool,
        mode: QuantizationMode.QLinearOps,
        static: bool,
        weight_qType: Any,
        activation_qType: Any,
        tensors_range: Any,
        nodes_to_quantize: list[str],
        nodes_to_exclude: list[str],
        op_types_to_quantize: list[str],
        calibrate_method: Any,
        quantized_tensor_type: dict[Any, Any] = {},
        extra_options: dict[str, Any] | None = None,
    ):
        self.calibrate_method = calibrate_method
        VitisQDQQuantizer.__init__(
            self,
            model,
            per_channel,
            reduce_range,
            mode,
            static,
            weight_qType,
            activation_qType,
            tensors_range,
            nodes_to_quantize,
            nodes_to_exclude,
            op_types_to_quantize,
            calibrate_method,
            quantized_tensor_type,
            extra_options,
        )
