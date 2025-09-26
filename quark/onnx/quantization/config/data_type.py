#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark ONNX Quantization Data Type Classes"""

from typing import Union

from onnx import TensorProto
from onnxruntime.quantization.quant_utils import QuantType

from quark.onnx.quant_utils import ExtendedQuantType
from quark.shares.data_type import (
    BaseBFloat16,
    BaseBFP16,
    BaseDataType,
    BaseFloat16,
    BaseInt4,
    BaseInt8,
    BaseInt16,
    BaseInt32,
    BaseUInt4,
    BaseUInt8,
    BaseUInt16,
    BaseUInt32,
)


class DataType(BaseDataType):
    """
    Base class for representing a quantization data type.
    Attributes:
        onnx_proto_dtype (TensorProto): Corresponding ONNX TensorProto data type.
        map_onnx_format (Union[ExtendedQuantType, QuantType]): Mapping to ONNX Runtime quantization type.
    """

    onnx_proto_dtype: TensorProto
    map_onnx_format: Union[ExtendedQuantType, QuantType]


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


# TODO: Add Quark ONNX MX data type classes.
