#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch

from quark.shares.utils.testing_utils import torch_device
from quark.torch.quantization.config.config import QuantizationSpec, TQTSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType, TQTThresholdInitMeth
from quark.torch.quantization.observer.tqt_observer import TQTObserver
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize

DEFAULT_QAT_INT8_PER_TENSOR_SPEC_TQT_WEIGHT = QuantizationSpec(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=TQTObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
    qat_spec=TQTSpec(threshold_init_meth=TQTThresholdInitMeth._3SD),
)

DEFAULT_QAT_INT4_PER_TENSOR_SPEC_TQT_INPUT = QuantizationSpec(
    dtype=Dtype.int4,
    qscheme=QSchemeType.per_tensor,
    observer_cls=TQTObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
    qat_spec=TQTSpec(threshold_init_meth=TQTThresholdInitMeth._KL_J),
)

seed = 11
torch.manual_seed(seed=seed)


def test_tqt_FakeQuantize():
    data = torch.randn((1, 3, 16, 16)).to(torch_device)
    tqt_quantizer_weight = ScaledFakeQuantize(DEFAULT_QAT_INT8_PER_TENSOR_SPEC_TQT_WEIGHT)
    tqt_quantizer_input = ScaledFakeQuantize(DEFAULT_QAT_INT4_PER_TENSOR_SPEC_TQT_INPUT)

    output1 = tqt_quantizer_weight(data)
    output2 = tqt_quantizer_weight(data)
    loss = torch.nn.CrossEntropyLoss()(output1, output1)
    loss.backward()
    output3 = tqt_quantizer_input(data)


if __name__ == "__main__":
    test_tqt_FakeQuantize()
