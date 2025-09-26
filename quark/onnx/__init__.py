#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from onnxruntime.quantization.calibrate import (CalibraterBase, CalibrationDataReader, CalibrationMethod,
                                                MinMaxCalibrater)
from onnxruntime.quantization.quant_utils import (QuantizationMode, QuantFormat, QuantType, write_calibration_table)

from quark.onnx.calibration import (Int16Method, PowerOfTwoMethod, LayerWiseMethod, ExtendedCalibrationMethod,
                                    CachedDataReader, RandomDataReader, PathDataReader, create_calibrator_power_of_two,
                                    create_calibrator_float_scale)

from .qdq_quantizer import VitisExtendedQuantizer
from .quant_utils import ExtendedQuantType, ExtendedQuantFormat, VitisQuantType, VitisQuantFormat
from .quantize import quantize_static
from .auto_search import AutoSearch, SearchSpace

from quark.onnx.quantization.api import ModelQuantizer
from quark.onnx.utils.deploy_utils import dump_model

from quark.onnx.operators.custom_ops import (_COP_DOMAIN, _COP_VERSION, get_library_path)
