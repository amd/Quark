#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from __future__ import annotations

import copy

import pytest
import torch
from torch.utils.data import DataLoader

from quark.common.utils.testing_utils import require_torch_cuda, torch_device
from quark.experimental.torch.algorithm.blockwise_joint_tuning.quantize.learnable_linear import (
    ExperimentalLearnableQuantizedLinear,
)
from quark.torch.algorithm.api import blockwise_tuning_algo
from quark.torch.algorithm.config import BlockwiseJointTuningConfig
from quark.torch.quantization import Int2PerGroupSpec, QLayerConfig
from quark.torch.quantization.nn.modules.quantize_linear import QuantLinear


@require_torch_cuda
def test_blockwise_joint_tuning_tiny_llama_end_to_end():
    """Run full blockwise joint tuning flow on a tiny real model."""
    transformers = pytest.importorskip("transformers", reason="transformers is required for this integration test")
    AutoModelForCausalLM = transformers.AutoModelForCausalLM
    LlamaConfig = transformers.LlamaConfig

    torch.manual_seed(42)

    model_cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=32,
    )
    model = AutoModelForCausalLM.from_config(model_cfg).to(torch_device).eval()
    fp_model = copy.deepcopy(model).eval()

    # Prepare one quantized projection so blockwise replacement can infer bit-width/group-size.
    weight_spec = Int2PerGroupSpec(ch_axis=1, is_dynamic=False, group_size=32).to_quantization_spec()
    qlayer_cfg = QLayerConfig(weight=weight_spec)
    model.model.layers[0].self_attn.q_proj = QuantLinear.from_float(model.model.layers[0].self_attn.q_proj, qlayer_cfg)

    # Keep calibration tiny to make this integration test fast.
    calib_input_ids = torch.randint(0, model_cfg.vocab_size, (1, 4), dtype=torch.long)
    calib_loader = DataLoader(calib_input_ids, batch_size=1)

    algo_cfg = BlockwiseJointTuningConfig(
        epochs=1,
        weight_lr=1e-5,
        qparam_lr=1e-4,
        weight_decay=0.0,
        qparam_weight_decay=0.0,
        min_lr_factor=20.0,
        max_grad_norm=0.3,
        model_decoder_layers="model.layers",
        trainable_modules=[],
        quant_trainable_modules=[],
    )

    tuned_model = blockwise_tuning_algo(
        fp_model=fp_model,
        model=model,
        blockwise_tuning_config=algo_cfg,
        is_accelerate=False,
        dataloader=calib_loader,
    )

    tuned_q_proj = tuned_model.model.layers[0].self_attn.q_proj
    assert isinstance(tuned_q_proj, ExperimentalLearnableQuantizedLinear)
    assert tuned_q_proj.weight_quantizer.num_bits == 2
    assert tuned_q_proj.weight_quantizer.group_size == 32

    eval_input_ids = torch.randint(0, model_cfg.vocab_size, (1, 4), dtype=torch.long, device=torch_device)
    with torch.no_grad():
        logits = tuned_model(input_ids=eval_input_ids).logits
    assert logits.shape == (1, 4, model_cfg.vocab_size)
