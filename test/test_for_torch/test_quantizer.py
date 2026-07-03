#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import torch
from transformers import AutoModelForCausalLM

from quark.common.utils.testing_utils import (
    PatchEverywhere,
    torch_device,
)
from quark.torch import LLMTemplate, ModelQuantizer
from quark.torch.utils.llm import preprocess_for_quantization


def test_memory_logging():
    model_id = "amd-quark/tiny-llama-fast-tokenizer"

    model = AutoModelForCausalLM.from_pretrained(model_id)
    model.eval()
    model = model.to(torch_device)
    # Get dataloader, if multi_gpu, give the first layer's device

    preprocess_for_quantization(model)

    template = LLMTemplate(model_type="llama")
    quant_config = template.get_config("mxfp4")

    quantizer = ModelQuantizer(quant_config)

    with PatchEverywhere("LOG_EVERY_SECONDS", 0.001, module_name_prefix="quark"), torch.no_grad():
        quant_model = quantizer.quantize_model(model)
        quant_model = quantizer.freeze(quant_model)

    return quant_model
