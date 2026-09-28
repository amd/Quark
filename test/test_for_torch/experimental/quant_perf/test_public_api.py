#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from quark.experimental.torch.quant_perf import perfopt


def test_perfopt_declared_public_symbols_resolve() -> None:
    for name in perfopt.__all__:
        assert getattr(perfopt, name) is not None
