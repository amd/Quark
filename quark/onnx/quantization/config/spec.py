#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from enum import Enum
from typing import Any

from quark.common.data_type import BaseDataType
from quark.common.utils.log import ScreenLogger

from .data_type import (
    BFP16,
    MX4,
    MX6,
    MX9,
    MXFP4E2M1,
    MXFP6E2M3,
    MXFP6E3M2,
    MXFP8E4M3,
    MXFP8E5M2,
    BFloat16,
    Int8,
    Int16,
    Int32,
    MXInt8,
    UInt8,
    UInt16,
    UInt32,
    parse_data_type,
)
from .utils import config_to_dict

logger = ScreenLogger(__name__)


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


ENUM_CLASS_MAP = {
    "CalibMethod": CalibMethod,
    "ScaleType": ScaleType,
    "QuantGranularity": QuantGranularity,
}


def parse_enum(v: str) -> Any:
    """
    Parse a string representation of an enum into the corresponding
    Enum member.

    The expected format is "<EnumClass>.<MemberName>". If the string
    does not match a known enum type, the original value is returned.

    :param str v: String to be parsed.
    :return: Enum member if parsing succeeds, otherwise the original value.
    """
    if not isinstance(v, str) or "." not in v:
        return v

    cls_name, member = v.split(".", 1)
    enum_cls = ENUM_CLASS_MAP.get(cls_name)
    if enum_cls is None:
        return v  # not enum, return raw string

    return getattr(enum_cls, member)


# TODO: Move QTensorConfig into the quark/common
class QTensorConfig:
    """
    Configuration for a quantized tensor.

    :param bool symmetric: Whether to use symmetric quantization.
    :param ScaleType scale_type: Type of scaling to apply.
    :param CalibMethod calibration_method: Method for calibration.
    :param QuantGranularity quant_granularity: Level of quantization granularity.
    :param BaseDataType data_type: Data type of quantization.
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

    def to_dict(self) -> dict[str, Any]:
        return config_to_dict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Any:
        """
        Convert a dictionary into a QTensorConfig object.

        Enum values and data types are parsed and normalized during
        the conversion process.

        :param dict[str, Any] d: Dictionary representation of a tensor config.
        :return: Constructed QTensorConfig instance.
        """

        kwargs = {}
        for k, v in d.items():
            if k == "data_type":
                kwargs[k] = parse_data_type(v)
            elif isinstance(v, str) and "." in v:
                kwargs[k] = parse_enum(v)
            else:
                kwargs[k] = v
        return QTensorConfig(**kwargs)


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


class MX4Spec(QTensorConfig):
    """
    Specification for MX4 tensors.
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.MinMax,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = MX4,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class MX6Spec(QTensorConfig):
    """
    Specification for MX6 tensors.
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.MinMax,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = MX6,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class MX9Spec(QTensorConfig):
    """
    Specification for MX9 tensors.
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.MinMax,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = MX9,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class MXFP4E2M1Spec(QTensorConfig):
    """
    Specification for MXFP4E2M1 tensors.
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.MinMax,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = MXFP4E2M1,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class MXFP6E3M2Spec(QTensorConfig):
    """
    Specification for MXFP6E3M2 tensors.
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.MinMax,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = MXFP6E3M2,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class MXFP6E2M3Spec(QTensorConfig):
    """
    Specification for MXFP6E2M3 tensors.
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.MinMax,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = MXFP6E2M3,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class MXFP8E5M2Spec(QTensorConfig):
    """
    Specification for MXFP8E5M2 tensors.
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.MinMax,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = MXFP8E5M2,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class MXFP8E4M3Spec(QTensorConfig):
    """
    Specification for MXFP8E4M3 tensors.
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.MinMax,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = MXFP8E4M3,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


class MXInt8Spec(QTensorConfig):
    """
    Specification for MXInt8 tensors.
    """

    def __init__(
        self,
        symmetric: bool = True,
        scale_type: ScaleType = ScaleType.Float32,
        calibration_method: CalibMethod = CalibMethod.MinMax,
        quant_granularity: QuantGranularity = QuantGranularity.Tensor,
        data_type: type[BaseDataType] = MXInt8,
    ):
        super().__init__(symmetric, scale_type, calibration_method, quant_granularity, data_type)


# TODO: Move QLayerConfig into the quark/common
class QLayerConfig:
    """
    Layer-level quantization configuration.
    :param QTensorConfig input_tensors: Quantization spec for input_tensors.
    :param QTensorConfig activation: Quantization spec for activations.
    :param QTensorConfig weight: Quantization spec for weights.
    :param QTensorConfig bias: Quantization spec for bias.
    :param QTensorConfig output_tensors: Quantization spec for output_tensors.
    """

    def __init__(
        self,
        input_tensors: QTensorConfig | None = None,
        activation: QTensorConfig | None = None,
        weight: QTensorConfig | None = None,
        bias: QTensorConfig | None = None,
        output_tensors: QTensorConfig | None = None,
    ):
        self.input_tensors = input_tensors
        self.activation = activation
        self.weight = weight
        self.bias = bias
        self.output_tensors = output_tensors

    def to_dict(self) -> dict[str, Any]:
        return config_to_dict(self)

    @staticmethod
    def from_dict(d: dict[str, Any]) -> Any:
        """
        Convert a dictionary into a QLayerConfig object.
        Tensor configurations for inputs, weights, bias, and outputs
        are parsed if present.
        :param dict[str, Any] d: Dictionary representation of a layer config.
        :return: Constructed QLayerConfig instance.
        """
        return QLayerConfig(
            input_tensors=QTensorConfig.from_dict(d["input_tensors"]) if "input_tensors" in d else None,
            weight=QTensorConfig.from_dict(d["weight"]) if "weight" in d else None,
            bias=QTensorConfig.from_dict(d["bias"]) if "bias" in d else None,
            output_tensors=QTensorConfig.from_dict(d["output_tensors"]) if "output_tensors" in d else None,
        )
