#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.shares.utils.testing_utils import torch_device
from quark.torch.quantization.api import ModelQuantizer
from quark.torch.quantization.config.config import AlgoConfig, Config, QuantizationConfig


def assert_non_destructive_transform(algo_config: AlgoConfig):
    model_id = "HuggingFaceTB/SmolLM-135M"

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model = model.eval()
    model = model.to(torch_device)

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    inp = tokenizer("Today I am in Paris and I will eat croissant.", return_tensors="pt").to(torch_device)

    with torch.no_grad():
        res_ref = model(**inp).logits

    # No quantization.
    quant_config = Config(global_quant_config=QuantizationConfig(), algo_config=[algo_config])

    # 4-2. In-place replacement of model modules with quantized versions.
    quantizer = ModelQuantizer(quant_config)
    model = quantizer.quantize_model(model)

    with torch.no_grad():
        res_no_quant = model(**inp).logits

    assert torch.allclose(res_ref, res_no_quant, atol=1e-2, rtol=1e-2)
