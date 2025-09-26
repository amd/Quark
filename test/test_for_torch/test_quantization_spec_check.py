#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from quark.shares.utils.log import ScreenLogger
from quark.torch.quantization.config.config import QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.observer import (
    PerChannelMinMaxObserver,
    PerGroupMinMaxObserver,
    PerTensorMinMaxObserver,
)

logger = ScreenLogger(__name__)


def test_normal_int_config():
    try:
        INT8_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_tensor,
            observer_cls=PerTensorMinMaxObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            is_dynamic=False,
        )
    except Exception as e:
        raise ValueError("This unittest should not throw a error")
    else:
        logger.info("Finish Test normal int quant config")


def test_int_lack_observer():
    try:
        INT8_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_tensor,
            observer_cls=None,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            is_dynamic=False,
        )
    except Exception as e:
        logger.info(f"Finish test no observer, with error: {type(e).__name__}")
    else:
        raise ValueError("This unittest must throw a error")


def test_int_lack_is_dynamic():
    try:
        INT8_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_tensor,
            observer_cls=PerTensorMinMaxObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            is_dynamic=None,
        )
    except Exception as e:
        logger.info(f"Finish test no is_dynamic, with error: {type(e).__name__}")
    else:
        raise ValueError("This unittest must throw a error")


def test_int_lack_qscheme():
    try:
        INT8_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=None,
            observer_cls=PerTensorMinMaxObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            is_dynamic=False,
        )
    except Exception as e:
        logger.info(f"Finish test no qscheme, with error: {type(e).__name__}")
    else:
        raise ValueError("This unittest must throw a error")


def test_int_lack_symmetric():
    try:
        INT8_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_tensor,
            observer_cls=PerTensorMinMaxObserver,
            symmetric=None,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            is_dynamic=False,
        )
    except Exception as e:
        logger.info(f"Finish test no symmetric, with error: {type(e).__name__}")
    else:
        raise ValueError("This unittest must throw a error")


def test_int_lack_round_method():
    try:
        INT8_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_tensor,
            observer_cls=PerTensorMinMaxObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=None,
            is_dynamic=False,
        )
    except Exception as e:
        logger.info(f"Finish test no round_method, with error: {type(e).__name__}")
    else:
        raise ValueError("This unittest must throw a error")


def test_int_lack_scale_type():
    try:
        INT8_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_tensor,
            observer_cls=PerTensorMinMaxObserver,
            symmetric=True,
            scale_type=None,
            round_method=RoundType.half_even,
            is_dynamic=False,
        )
    except Exception as e:
        logger.info(f"Finish test no round_method, with error: {type(e).__name__}")
    else:
        raise ValueError("This unittest must throw a error")


def test_int_per_tensor_mis_match_observer():
    try:
        INT8_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_tensor,
            observer_cls=PerChannelMinMaxObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            is_dynamic=False,
        )
    except Exception as e:
        logger.info(f"Finish test no mis_match_observer, with error: {type(e).__name__}")
    else:
        raise ValueError("This unittest must throw a error")


def test_normal_int_per_channel_config():
    try:
        INT8_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_channel,
            observer_cls=PerChannelMinMaxObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            ch_axis=0,
            is_dynamic=False,
        )
    except Exception as e:
        raise ValueError("This unittest should not throw a error")
    else:
        logger.info("Finish Test normal int quant config")


def test_int_per_channel_mis_match_observer():
    try:
        INT8_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_channel,
            observer_cls=PerTensorMinMaxObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            ch_axis=0,
            is_dynamic=False,
        )
    except Exception as e:
        logger.info(f"Finish test no mis_match_observer, with error: {type(e).__name__}")
    else:
        raise ValueError("This unittest must throw a error")


def test_int_per_channel_lack_axis():
    try:
        INT8_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_channel,
            observer_cls=PerChannelMinMaxObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            ch_axis=None,
            is_dynamic=False,
        )
    except Exception as e:
        logger.info(f"Finish test no lack_axis, with error: {type(e).__name__}")
    else:
        raise ValueError("This unittest must throw a error")


def test_normal_int_per_group_config():
    try:
        INT8_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_group,
            observer_cls=PerGroupMinMaxObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            ch_axis=0,
            group_size=64,
            is_dynamic=False,
        )
    except Exception as e:
        raise ValueError("This unittest should not throw a error")
    else:
        logger.info("Finish Test normal int quant config")


def test_int_per_group_lack_ch_axis():
    try:
        INT8_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_group,
            observer_cls=PerGroupMinMaxObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            ch_axis=None,
            group_size=64,
            is_dynamic=False,
        )
    except Exception as e:
        logger.info(f"Finish test no lack_axis, with error: {type(e).__name__}")
    else:
        raise ValueError("This unittest must throw a error")


def test_int_per_group_lack_group_size():
    try:
        INT8_SPEC = QuantizationSpec(
            dtype=Dtype.int8,
            qscheme=QSchemeType.per_group,
            observer_cls=PerGroupMinMaxObserver,
            symmetric=True,
            scale_type=ScaleType.float,
            round_method=RoundType.half_even,
            ch_axis=0,
            group_size=None,
            is_dynamic=False,
        )
    except Exception as e:
        logger.info(f"Finish test no group_size, with error: {type(e).__name__}")
    else:
        raise ValueError("This unittest must throw a error")


def test_mx_quant_type():
    try:
        MX6_SPEC = QuantizationSpec(dtype=Dtype.mx6, ch_axis=-1, group_size=64)
    except Exception as e:
        raise ValueError("This unittest should not throw a error")
    else:
        logger.info("This is a norm mx quant config.")


def test_mx_quant_lack_ch_axis():
    try:
        MX6_SPEC = QuantizationSpec(dtype=Dtype.mx6, ch_axis=None, group_size=64)
    except Exception as e:
        logger.info(f"Finish test no ch_axis, with error: {type(e).__name__}")
    else:
        raise ValueError("This unittest must throw a error")


def test_mx_quant_lack_group_size():
    try:
        MX6_SPEC = QuantizationSpec(dtype=Dtype.mx6, ch_axis=-1, group_size=None)
    except Exception as e:
        logger.info(f"Finish test no ch_axis, with error: {type(e).__name__}")
    else:
        raise ValueError("This unittest must throw a error")


def test_dtype_change_quant():
    try:
        BFLOAT16_SPEC = QuantizationSpec(dtype=Dtype.bfloat16, ch_axis=-1)
    except Exception as e:
        raise ValueError("This unittest must throw a error")

    else:
        logger.info("Finish test but with warring: redundant ch_axis")


if __name__ == "__main__":
    # test_int_per_channel_mis_match_observer()
    # test_normal_int_per_group_config()
    # test_int_per_group_lack_ch_axis()
    # test_int_per_group_lack_group_size()
    test_dtype_change_quant()
