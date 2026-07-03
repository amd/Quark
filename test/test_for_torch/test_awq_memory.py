#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import gc
import os

import pytest
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM

from quark.common.utils.testing_utils import require_torch_cuda, slow_test
from quark.torch import ModelQuantizer
from quark.torch.quantization import AWQConfig, QConfig, QLayerConfig, Uint4PerGroupSpec

# CI-specific model path
MODEL_PATH = "/group/amdneuralopt/hf_download/Meta-Llama-3.1-8B-Instruct"


class RandomIntDataset(Dataset):
    def __init__(self, total_samples=10, seq_len=512):
        self.total_samples = total_samples
        self.seq_len = seq_len

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        return torch.randint(1, 32768, (self.seq_len,), dtype=torch.int64)


def get_dataloader(batch_size, model_device, total_samples=10):
    dataset = RandomIntDataset(batch_size, seq_len=512)

    def collate_fn(batch):
        return torch.stack(batch).to(model_device)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)


def get_model(MODEL_ID):
    max_memory = {0: "8GB", "cpu": "1737.5GB"}
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", max_memory=max_memory, torch_dtype="auto", attn_implementation="eager"
    )
    model.eval()
    for name, module in model.named_modules():
        module.module_name = name
    return model


def get_quark_model(model, batch_size=59):
    calib_dataloader = get_dataloader(batch_size=batch_size, model_device=torch.device("cuda"))

    UINT4_PER_GROUP_ASYM_SPEC = Uint4PerGroupSpec(
        scale_type="float", ch_axis=1, is_dynamic=False, group_size=128
    ).to_quantization_spec()

    global_quant_config = QLayerConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)

    algo_config = AWQConfig(
        name="awq",
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.layers",
    )

    quant_config = QConfig(global_quant_config=global_quant_config, algo_config=[algo_config], exclude=["lm_head"])

    quantizer = ModelQuantizer(quant_config, multi_device=True)
    quant_model = quantizer.quantize_model(model, calib_dataloader)
    return quant_model


@slow_test
@require_torch_cuda
@pytest.mark.skipif(not os.path.exists(MODEL_PATH), reason="CI-specific model not available")
def test_gpu_memory(monkeypatch: pytest.MonkeyPatch):
    """Test AWQ memory optimization with QUARK_AWQ_MEMORY_OPTIMIZATION=1."""
    # Patch imported constants directly because they are cached when the target modules are imported.
    monkeypatch.setattr("quark.torch.algorithm.awq.awq.QUARK_AWQ_MEMORY_OPTIMIZATION", True)
    monkeypatch.setattr("quark.torch.algorithm.utils.utils.QUARK_AWQ_MEMORY_OPTIMIZATION", True)
    monkeypatch.setattr("quark.torch.kernel.hw_emulation.hw_emulation_interface.QUARK_AWQ_MEMORY_OPTIMIZATION", True)
    monkeypatch.setattr("quark.torch.utils.QUARK_AWQ_MEMORY_OPTIMIZATION", True)

    model = get_model(MODEL_PATH)
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_memory = torch.cuda.memory_reserved() / 1024**3
    with torch.no_grad():
        quant_model = get_quark_model(model)  # noqa
    peak_memory = torch.cuda.max_memory_reserved() / 1024**3 - baseline_memory
    std_memory = 10.583984375
    assert peak_memory < (std_memory * 1.1), (
        f"Quantization memory usage {peak_memory}GB exceeds limit {std_memory * 1.1}GB"
    )
