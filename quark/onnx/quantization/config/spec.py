#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from enum import Enum
from typing import Type

from quark.shares.data_type import BaseDataType

from .data_type import BFP16, BFloat16, Int8, Int16, Int32, UInt8, UInt16, UInt32


# TODO: Write a separate class for each calibration method.
class CalibMethod(Enum):
    """
    Enumeration of calibration methods used for determining quantization parameters.
    """

    MinMax = 0
    MinMSE = 1
    Percentile = 2
    Entropy = 3
    LayerwisePercentile = 4
    Distribution = 5


class ScaleType(Enum):
    """
    Enumeration of scale types used in quantization.
    """

    Float32 = 0
    PowerOf2 = 1
    Int16 = 2


class QuantGranularity(Enum):
    """
    Enumeration of quantization granularity.
    """

    Tensor = 0
    Channel = 1
    Group = 2


# TODO: Move QTensorConfig into the quark/shares
class QTensorConfig:
    """
    Configuration for a quantized tensor.

    Args:
        symmetric (bool): Whether to use symmetric quantization.
        scale_type (ScaleType): Type of scaling to apply.
        calibration_method (CalibMethod): Method for calibration.
        quant_granularity (QuantGranularity): Level of quantization granularity.
        data_type (BaseDataType): Data type of quantization.
    """

    def __init__(
        self,
        symmetric: bool,
        scale_type: ScaleType,
        calibration_method: CalibMethod,
        quant_granularity: QuantGranularity,
        data_type: BaseDataType,
    ) -> None:
        self.symmetric = symmetric
        self.scale_type = scale_type
        self.calibration_method = calibration_method
        self.quant_granularity = quant_granularity
        self.data_type = data_type

    def set_symmetric(self, symmetric: bool) -> None:
        """Set whether symmetric quantization is used."""
        self.symmetric = symmetric

    def set_scale_type(self, scale_type: ScaleType) -> None:
        """Set the scale type."""
        self.scale_type = scale_type

    def set_calibration_method(self, calibration_method: CalibMethod) -> None:
        """Set the calibration method."""
        self.calibration_method = calibration_method

    def set_quant_granularity(self, quant_granularity: QuantGranularity) -> None:
        """Set the quantization granularity."""
        self.quant_granularity = quant_granularity

    def set_data_type(self, data_type: BaseDataType) -> None:
        """Set the data type."""
        self.data_type = data_type


class Int8Spec(QTensorConfig):
    """
    Quantization specification for int8 tensors (default Float32 scaling and Percentile calibration).
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.Percentile,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = Int8,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class UInt8Spec(QTensorConfig):
    """
    Quantization specification for uint8 tensors.
    """

    def __init__(
        self,
        symmetric: bool = False,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.Percentile,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = UInt8,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class XInt8Spec(Int8Spec):
    """
    Quantization specification for int8 tensors with power-of-2 scaling.
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.PowerOf2,
        calibration_method: CalibMethod = CalibMethod.MinMSE,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = Int8,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class Int16Spec(QTensorConfig):
    """
    Quantization specification for int16 tensors.
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.Percentile,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = Int16,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class UInt16Spec(QTensorConfig):
    """
    Quantization specification for uint16 tensors.
    """

    def __init__(
        self,
        symmetric: bool = False,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.Percentile,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = UInt16,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class Int32Spec(QTensorConfig):
    """
    Quantization specification for int32 tensors.
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.Percentile,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = Int32,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class UInt32Spec(QTensorConfig):
    """
    Quantization specification for uint32 tensors.
    """

    def __init__(
        self,
        symmetric: bool = False,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.Percentile,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = UInt32,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class BFloat16Spec(QTensorConfig):
    """
    Specification for bfloat16 tensors.
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.MinMax,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = BFloat16,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class BFP16Spec(QTensorConfig):
    """
    Specification for Block Floating Point (BFP16) tensors.
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.MinMax,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = BFP16,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


# TODO: Add MX Specs.


# TODO: Move QLayerConfig into the quark/shares
class QLayerConfig:
    """
    Layer-level quantization configuration.

    Args:
        activation (QTensorConfig): Quantization spec for activations.
        weight (QTensorConfig): Quantization spec for weights.
    """

    def __init__(self, activation: QTensorConfig, weight: QTensorConfig):
        self.activation = activation
        self.weight = weight
