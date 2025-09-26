#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import json
import os
import shutil
from typing import Tuple
from unittest.mock import Mock

import pytest
import torch
from sentencepiece import SentencePieceProcessor

from quark.torch.export.gguf_export.api import convert_exported_model_to_gguf
from quark.torch.export.gguf_export.gguf_model_writer import LlamaModelWriter


def generate_mock_tensors() -> tuple[str, torch.Tensor]:
    return [
        ("model.layers.0.self_attn.q_proj.weight", torch.rand((32, 32), dtype=torch.float32)),
        ("model.layers.0.self_attn.q_proj.weight_scale", torch.ones((32, 1), dtype=torch.float32)),
        ("model.layers.0.self_attn.q_proj.weight_zero_point", torch.zeros((32, 1), dtype=torch.float32)),
    ]


SentencePieceProcessor.__init__ = Mock((), return_value=None)
SentencePieceProcessor.vocab_size = Mock((), return_value=10)
SentencePieceProcessor.id_to_piece = Mock((), return_value="a")
SentencePieceProcessor.get_score = Mock((), return_value=0.0)
SentencePieceProcessor.is_unknown = Mock((), return_value=False)
SentencePieceProcessor.is_control = Mock((), return_value=False)
SentencePieceProcessor.is_unused = Mock((), return_value=False)
SentencePieceProcessor.is_byte = Mock((), return_value=False)
LlamaModelWriter.get_tensors = Mock((), return_value=generate_mock_tensors())


def generate_llama_json():
    llama_json = {
        "config": {
            "vocab_size": 32000,
            "max_position_embeddings": 4096,
            "hidden_size": 4096,
            "intermediate_size": 11008,
            "num_hidden_layers": 32,
            "num_attention_heads": 4,
            "num_key_value_heads": 32,
            "hidden_act": "silu",
            "initializer_range": 0.02,
            "rms_norm_eps": 1e-05,
            "pretraining_tp": 1,
            "use_cache": True,
            "rope_theta": 10000.0,
            "rope_scaling": None,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "torch_dtype": "float16",
            "tie_word_embeddings": False,
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "_name_or_path": "/group/ossmodelzoo/quark_torch/huggingface_pretrained_models/meta-llama/Llama-2-7b-hf",
            "transformers_version": "4.37.2",
            "model_type": "llama",
        },
        "structure": {
            "model.layers.0.self_attn.q_proj": {
                "name": "model.layers.0.self_attn.q_proj",
                "type": "QuantLinear",
                "weight": "model.layers.0.self_attn.q_proj.weight",
                "weight_quant": {
                    "scale": "model.layers.0.self_attn.q_proj.weight_scale",
                    "zero_point": "model.layers.0.self_attn.q_proj.weight_zero_point",
                    "dtype": "uint4",
                    "qscheme": "per_group",
                    "ch_axis": 1,
                    "group_size": 32,
                    "round_method": "half_even",
                    "scale_type": "float",
                },
            }
        },
    }
    with open(".tmp/llama.json", "w") as f:
        json.dump(llama_json, f)


@pytest.fixture
def prepare_files():
    if not os.path.exists(".tmp"):
        os.makedirs(".tmp")
    generate_llama_json()
    with open(".tmp/tokenizer.model", "w") as f:
        f.write("")

    yield
    shutil.rmtree(".tmp")


def test_gguf_api(prepare_files):
    convert_exported_model_to_gguf(
        model_name="llama2",
        json_path=".tmp/llama.json",
        safetensor_path="",
        tokenizer_dir=".tmp",
        output_file_path="./.tmp/llama.gguf",
    )


if __name__ == "__main__":
    test_gguf_api()
