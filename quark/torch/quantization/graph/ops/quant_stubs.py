#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
from torch.library import Library, impl

__all__ = ["QuantStub", "DeQuantStub"]

quark_quant = Library("quark_quant", "DEF")  # type: ignore[no-untyped-call]
quark_quant.define("QuantStub(Tensor input) -> Tensor")  # type: ignore[no-untyped-call]


@impl(quark_quant, "QuantStub", "CompositeExplicitAutograd")  # type: ignore[misc]
def quant_stub(input: torch.Tensor) -> torch.Tensor:
    return input


quark_quant.define("DeQuantStub(Tensor input) -> Tensor")  # type: ignore[no-untyped-call]


@impl(quark_quant, "DeQuantStub", "CompositeExplicitAutograd")  # type: ignore[misc]
def de_quantStub(input: torch.Tensor) -> torch.Tensor:
    return input


QuantStub = torch.ops.quark_quant.QuantStub  # type: ignore[attr-defined]
DeQuantStub = torch.ops.quark_quant.DeQuantStub  # type: ignore[attr-defined]
