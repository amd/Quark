#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark ONNX Quantization Data Type Classes"""

from onnx import TensorProto
from onnxruntime.quantization.quant_utils import QuantType

from quark.common.data_type import (
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
from quark.onnx.quantization.quant_utils import ExtendedQuantType


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


dt_name_map = {
    Int4: "Int4",
    UInt4: "UInt4",
    Int8: "Int8",
    UInt8: "UInt8",
    Int16: "Int16",
    UInt16: "UInt16",
    Int32: "Int32",
    UInt32: "UInt32",
    Float16: "Float16",
    BFloat16: "BFloat16",
    BFP16: "BFP16",
    MX4: "MX4",
    MX6: "MX6",
    MX9: "MX9",
    MXFP4E2M1: "MXFP4E2M1",
    MXFP6E3M2: "MXFP6E3M2",
    MXFP6E2M3: "MXFP6E2M3",
    MXFP8E5M2: "MXFP8E5M2",
    MXFP8E4M3: "MXFP8E4M3",
    MXInt8: "MXInt8",
}

name_dt_map = {}
for k, v in dt_name_map.items():
    name_dt_map[v] = k


def parse_data_type(dtype_name: str) -> BaseDataType:
    """
    Parse a data type name into a BaseDataType instance.

    :param str dtype_name: Name of the data type.
    :return: Instantiated data type object.
    """
    if dtype_name is None:
        return None
    cls = name_dt_map[dtype_name]
    return cls()
