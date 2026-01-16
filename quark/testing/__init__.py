#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from quark.testing.common_utils import (  # type: ignore[unused-ignore, import-not-found]
    skip_if_no_gpu,
    slow_test,
    slow_test_if,
)

__all__ = ["skip_if_no_gpu", "slow_test", "slow_test_if"]
