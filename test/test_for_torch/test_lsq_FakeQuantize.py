#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch

from quark.common.utils.testing_utils import require_torch_cuda, torch_device
from quark.torch.quantization.config.config import QTensorConfig
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType
from quark.torch.quantization.observer.lsq_observer import LSQObserver
from quark.torch.quantization.tensor_quantize import FakeQuantizeBase

DEFAULT_QAT_INT8_PER_CHANNEL_SPEC_LSQ_WEIGHT = QTensorConfig(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_channel,
    ch_axis=1,
    observer_cls=LSQObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)

DEFAULT_QAT_INT8_PER_TENSOR_SPEC_LSQ_INPUT = QTensorConfig(
    dtype=Dtype.int8,
    qscheme=QSchemeType.per_tensor,
    observer_cls=LSQObserver,
    symmetric=True,
    scale_type=ScaleType.float,
    round_method=RoundType.half_even,
    is_dynamic=False,
)

seed = 11
torch.manual_seed(seed=seed)


@require_torch_cuda
def test_lsq_FakeQuantize():
    data = torch.randn((1, 3, 16, 16)).to(torch_device)
    lsq_quantizer_weight = FakeQuantizeBase.get_fake_quantize(
        DEFAULT_QAT_INT8_PER_CHANNEL_SPEC_LSQ_WEIGHT, device=torch_device
    )
    lsq_quantizer_input = FakeQuantizeBase.get_fake_quantize(
        DEFAULT_QAT_INT8_PER_TENSOR_SPEC_LSQ_INPUT, device=torch_device
    )

    output1 = lsq_quantizer_weight(data)
    _ = lsq_quantizer_weight(data)
    loss = torch.nn.CrossEntropyLoss()(output1, output1)
    loss.backward()
    _ = lsq_quantizer_input(data)


if __name__ == "__main__":
    test_lsq_FakeQuantize()
