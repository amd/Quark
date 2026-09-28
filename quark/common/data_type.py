#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quark Quantization Base Data Type Classes"""

from enum import Enum


class BaseDtype(Enum):
    pass


class BaseQSchemeType(Enum):
    pass


class BaseRoundType(Enum):
    pass


class BaseScaleType(Enum):
    pass


class BaseZeroPointType(Enum):
    pass


class BaseObserverBase:
    pass


class BaseDataType:
    """
    Base class for representing a quantization data type.

    Attributes:
        bitwidth (int): Number of bits used to represent the value.
        min_value (Union[float, int]): Minimum representable value.
        max_value (Union[float, int]): Maximum representable value.
    """

    bitwidth: int
    min_value: float | int
    max_value: float | int


class BaseMX(BaseDataType):
    """Microscaling data type."""

    pass


class BaseMicroexponent(BaseDataType):
    """
    Base class for the MicroeXponent (MX4/MX6/MX9) two-level block formats.

    Attributes:
        bitwidth (int): The ONNX ``bit_width`` attribute value for this format, as consumed
            by the ``BFPQuantizeDequantize`` custom op. This is a *format selector*, not a
            per-element width -- see ``avg_bits_per_element`` and ``element_bits``.
        k1 (int): Number of elements sharing the first-level (block) scale.
        k2 (int): Number of elements sharing the second-level (sub-block) scale.
        d1 (int): Bit-width of the first-level block scale.
        d2 (int): Bit-width of the second-level sub-block scale.
        mantissa_bits (int): Mantissa bits per element (``m`` in the paper).
        element_bits (int): Bits per stored element code, ``mantissa_bits + 1`` for the
            sign bit. This is the ``quant_bit`` value passed to the emulation kernel.
        avg_bits_per_element (int): Amortized storage cost per element,
            ``(m + 1) + d1 / k1 + d2 / k2``. This is the number the format is named after.
    """

    k1: int
    k2: int
    d1: int
    d2: int
    mantissa_bits: int
    element_bits: int
    avg_bits_per_element: int


class BaseInt2(BaseDataType):
    """Signed 2-bit integer quantization data type."""

    bitwidth = 2
    min_value = -2
    max_value = 1


class BaseUInt2(BaseDataType):
    """Unsigned 2-bit integer quantization data type."""

    bitwidth = 2
    min_value = 0
    max_value = 3


class BaseInt3(BaseDataType):
    """Signed 3-bit integer quantization data type."""

    bitwidth = 3
    min_value = -4
    max_value = 3


class BaseInt4(BaseDataType):
    """Signed 4-bit integer quantization data type."""

    bitwidth = 4
    min_value = -8
    max_value = 7


class BaseUInt4(BaseDataType):
    """Unsigned 4-bit integer quantization data type."""

    bitwidth = 4
    min_value = 0
    max_value = 15


class BaseInt8(BaseDataType):
    """Signed 8-bit integer quantization data type."""

    bitwidth = 8
    min_value = -128
    max_value = 127


class BaseUInt8(BaseDataType):
    """Unsigned 8-bit integer quantization data type."""

    bitwidth = 8
    min_value = 0
    max_value = 255


class BaseInt16(BaseDataType):
    """Signed 16-bit integer quantization data type."""

    bitwidth = 16
    min_value = -32768
    max_value = 32767


class BaseUInt16(BaseDataType):
    """Unsigned 16-bit integer quantization data type."""

    bitwidth = 16
    min_value = 0
    max_value = 65535


class BaseInt32(BaseDataType):
    """Signed 32-bit integer quantization data type."""

    bitwidth = 32
    min_value = -(2**31)
    max_value = 2**31 - 1


class BaseUInt32(BaseDataType):
    """Unsigned 32-bit integer quantization data type."""

    bitwidth = 32
    min_value = 0
    max_value = 2**32 - 1


class BaseFloat16(BaseDataType):
    """16-bit floating point quantization data type."""

    bitwidth = 16


class BaseBFloat16(BaseDataType):
    """16-bit Brain Floating Point quantization data type."""

    bitwidth = 16


class BaseBFP16(BaseDataType):
    """Block Floating Point data type."""

    bitwidth = 16


class BaseFP8_E5M2(BaseDataType):
    """8-bit floating point with E5M2 format."""

    bitwidth = 8
    min_value = -57344.0
    max_value = 57344.0


class BaseMXFP8_E5M2(BaseDataType):
    """8-bit floating point with E5M2 format using microscaling data type."""

    bitwidth = 8
    min_value = -57344.0
    max_value = 57344.0


class BaseFP8_E5M3(BaseDataType):
    """8-bit floating point with E5M3 format (unsigned)."""

    bitwidth = 8
    min_value = 0.0
    max_value = 114688.0


class BaseFP8_E4M3(BaseDataType):
    """8-bit floating point with E4M3 format."""

    bitwidth = 8
    min_value = -448.0
    max_value = 448.0


class BaseMXFP8_E4M3(BaseDataType):
    """8-bit floating point with E4M3 format using microscaling data type."""

    bitwidth = 8
    min_value = -448.0
    max_value = 448.0


class BaseFP6_E3M2(BaseDataType):
    """6-bit floating point with E3M2 format."""

    bitwidth = 6
    min_value = -28.0
    max_value = 28.0


class BaseMXFP6_E3M2(BaseDataType):
    """6-bit floating point with E3M2 format using microscaling data type."""

    bitwidth = 6
    min_value = -28.0
    max_value = 28.0


class BaseFP6_E2M3(BaseDataType):
    """6-bit floating point with E2M3 format."""

    bitwidth = 6
    min_value = -7.5
    max_value = 7.5


class BaseMXFP6_E2M3(BaseDataType):
    """6-bit floating point with E2M3 format using microscaling data type."""

    bitwidth = 6
    min_value = -7.5
    max_value = 7.5


class BaseFP4(BaseDataType):
    """4-bit floating point quantization data type."""

    bitwidth = 4
    min_value = -6.0
    max_value = 6.0


class BaseMXFP4_E2M1(BaseDataType):
    """4-bit floating point quantization data type using microscaling data type."""

    bitwidth = 4
    min_value = -6.0
    max_value = 6.0


class BaseMX4(BaseMicroexponent):
    """MX4 MicroeXponent data type: 4 amortized bits per element."""

    bitwidth = 11
    k1 = 16
    k2 = 2
    d1 = 8
    d2 = 1
    mantissa_bits = 2
    element_bits = 3
    avg_bits_per_element = 4


class BaseMX6(BaseMicroexponent):
    """MX6 MicroeXponent data type: 6 amortized bits per element."""

    bitwidth = 13
    k1 = 16
    k2 = 2
    d1 = 8
    d2 = 1
    mantissa_bits = 4
    element_bits = 5
    avg_bits_per_element = 6


class BaseMX9(BaseMicroexponent):
    """MX9 MicroeXponent data type: 9 amortized bits per element."""

    bitwidth = 16
    k1 = 16
    k2 = 2
    d1 = 8
    d2 = 1
    mantissa_bits = 7
    element_bits = 8
    avg_bits_per_element = 9


class BaseMXInt8(BaseDataType):
    """8-bit int microscaling data type."""

    bitwidth = 8
    min_value = -128
    max_value = 127
