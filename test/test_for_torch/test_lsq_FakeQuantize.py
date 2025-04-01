#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize
from quark.torch.quantization.config.config import QuantizationSpec
from quark.torch.quantization.config.type import Dtype, QSchemeType, ScaleType, RoundType
from quark.torch.quantization.observer.lsq_observer import LSQObserver
from quark.shares.utils.testing_utils import require_torch_gpu, torch_device

DEFAULT_QAT_INT8_PER_CHANNEL_SPEC_LSQ_WEIGHT = QuantizationSpec(
    dtype=Dtype.int8,
    ch_axis=1,
    observer_cls=LSQObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False)

DEFAULT_QAT_INT8_PER_TENSOR_SPEC_LSQ_INPUT = QuantizationSpec(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=LSQObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False)

seed = 11
torch.manual_seed(seed=seed)

@require_torch_gpu
def test_lsq_FakeQuantize():
    data = torch.randn((1, 3, 16, 16)).to(torch_device)
    lsq_quantizer_weight = ScaledFakeQuantize(
        DEFAULT_QAT_INT8_PER_CHANNEL_SPEC_LSQ_WEIGHT, device=torch_device)
    lsq_quantizer_input = ScaledFakeQuantize(
        DEFAULT_QAT_INT8_PER_TENSOR_SPEC_LSQ_INPUT, device=torch_device)

    output1 = lsq_quantizer_weight(data)
    output2 = lsq_quantizer_weight(data)
    loss = torch.nn.CrossEntropyLoss()(output1, output1)
    loss.backward()
    output3 = lsq_quantizer_input(data)
