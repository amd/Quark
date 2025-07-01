#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import torch
from quark.torch.export.nn.modules import realquantizer
from quark.torch.quantization import FP4PerGroupSpec
from quark.testing import skip_if_no_gpu


@skip_if_no_gpu
def test_realquantizer():
    torch.manual_seed(42)

    qspec = FP4PerGroupSpec(ch_axis=-1,
                            group_size=32,
                            scale_format="e8m0",
                            scale_calculation_mode="even",
                            is_dynamic=True).to_quantization_spec()

    input_quantizer = realquantizer.get_real_quantizer(
        qspec=qspec,
        quantizer=None,
        real_quantized=False,
        float_dtype=torch.bfloat16,
        device="cuda"
    )

    x = torch.randn(256, 11008, device="cuda", dtype=torch.bfloat16)

    qdqx = input_quantizer(x)
    assert not torch.isinf(qdqx).any()

    qdqx_2 = input_quantizer(x)
    assert not torch.isinf(qdqx_2).any()

    torch.testing.assert_close(qdqx, qdqx_2)

    g = torch.cuda.CUDAGraph()

    with torch.cuda.graph(g):
        qdqx_graph = input_quantizer(x)
    g.replay()

    assert not torch.isinf(qdqx_graph).any()
    torch.testing.assert_close(qdqx, qdqx_graph)
