#
# Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch

from quark.shares.utils.log import ScreenLogger

from .constants import QUARK_DEBUG_NAN

logger = ScreenLogger(__name__)


def assert_no_nan(tensor: torch.Tensor, message: str) -> None:
    """
    Asserts that the tensor does not contain any NaN value. If it does, it will raise a `AssertionError` with the given message.

    Only does the assertion if the environment variable ``QUARK_DEBUG_NAN`` is set to ``1``. This is useful to avoid the overhead of checking for NaNs in production code.

    :param torch.Tensor tensor: The tensor to check for NaNs.
    :param str message : The message to display in the ``AssertionError`` if the tensor contains NaNs.
    """
    if QUARK_DEBUG_NAN:
        torch._assert_async(~torch.isnan(tensor).any(), message)
