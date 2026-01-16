#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark ONNX Quantization Data Type Classes"""

from onnx import TensorProto
from onnxruntime.quantization.quant_utils import QuantType

from quark.onnx.quantization.quant_utils import ExtendedQuantType
from quark.shares.data_type import (
    BaseBFloat16,
    BaseBFP16,
    BaseDataType,
    BaseFloat16,
    BaseInt4,
    BaseInt8,
    BaseInt16,
    BaseInt32,
    BaseMX4,
    BaseMX6,
    BaseMX9,
    BaseMXFP4_E2M1,
    BaseMXFP6_E2M3,
    BaseMXFP6_E3M2,
    BaseMXFP8_E4M3,
    BaseMXFP8_E5M2,
    BaseMXInt8,
    BaseUInt4,
    BaseUInt8,
    BaseUInt16,
    BaseUInt32,
)


class DataType(BaseDataType):
    """
    Base class for representing a quantization data type.
    """

    # Corresponding ONNX TensorProto data type.
    onnx_proto_dtype: TensorProto

    # Mapping to ONNX Runtime quantization type.
    map_onnx_format: ExtendedQuantType | QuantType


class Int4(BaseInt4):
    """Signed 4-bit integer quark onnx quantization data type."""

    onnx_proto_dtype: TensorProto.INT4  # type: ignore
    map_onnx_format = ExtendedQuantType.QInt4


class UInt4(BaseUInt4):
    """Unsigned 4-bit integer quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.UINT4
    map_onnx_format = ExtendedQuantType.QUInt4


class Int8(BaseInt8):
    """Signed 8-bit integer quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.INT8
    map_onnx_format = QuantType.QInt8


class UInt8(BaseUInt8):
    """Unsigned 8-bit integer quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.UINT8
    map_onnx_format = QuantType.QUInt8


class Int16(BaseInt16):
    """Signed 16-bit integer quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.INT16
    map_onnx_format = ExtendedQuantType.QInt16


class UInt16(BaseUInt16):
    """Unsigned 16-bit integer quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.UINT16
    map_onnx_format = ExtendedQuantType.QUInt16


class Int32(BaseInt32):
    """Signed 32-bit integer quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.INT32
    map_onnx_format = ExtendedQuantType.QInt32


class UInt32(BaseUInt32):
    """Unsigned 32-bit integer quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.UINT32
    map_onnx_format = ExtendedQuantType.QUInt32


class Float16(BaseFloat16):
    """16-bit floating point quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.FLOAT16
    map_onnx_format = ExtendedQuantType.QFloat16


class BFloat16(BaseBFloat16):
    """16-bit Brain Floating Point quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.BFLOAT16
    map_onnx_format = ExtendedQuantType.QBFloat16


class BFP16(BaseBFP16):
    """Block Floating Point quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.UNDEFINED
    map_onnx_format = ExtendedQuantType.QBFP


class MX4(BaseMX4):
    """MX4 quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.UNDEFINED
    map_onnx_format = ExtendedQuantType.QBFP


class MX6(BaseMX6):
    """MX6 quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.UNDEFINED
    map_onnx_format = ExtendedQuantType.QBFP


class MX9(BaseMX9):
    """MX9 quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.UNDEFINED
    map_onnx_format = ExtendedQuantType.QBFP


class MXFP4E2M1(BaseMXFP4_E2M1):
    """MXFP4E2M1 quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.UNDEFINED
    map_onnx_format = ExtendedQuantType.QMX


class MXFP6E3M2(BaseMXFP6_E3M2):
    """MXFP6E3M2 quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.UNDEFINED
    map_onnx_format = ExtendedQuantType.QMX


class MXFP6E2M3(BaseMXFP6_E2M3):
    """MXFP6E2M3 quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.UNDEFINED
    map_onnx_format = ExtendedQuantType.QMX


class MXFP8E5M2(BaseMXFP8_E5M2):
    """MXFP8E5M2 quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.UNDEFINED
    map_onnx_format = ExtendedQuantType.QMX


class MXFP8E4M3(BaseMXFP8_E4M3):
    """MXFP8E4M3 quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.UNDEFINED
    map_onnx_format = ExtendedQuantType.QMX


class MXInt8(BaseMXInt8):
    """MXInt8 quark onnx quantization data type."""

    onnx_proto_dtype = TensorProto.UNDEFINED
    map_onnx_format = ExtendedQuantType.QMX
