#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from typing import Union

from quark.torch.quantization.config.config import (
    AutoSmoothQuantConfig,
    AWQConfig,
    GPTQConfig,
    RotationConfig,
    SmoothQuantConfig,
)

# AWQ configs, these are the default configs for each model, can be updated by user
AWQ_MAP = {
    "llama": AWQConfig(
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
    ),
    "qwen": AWQConfig(
        scaling_layers=[
            {"prev_op": "ln_1", "layers": ["attn.c_attn"], "inp": "attn.c_attn", "module2inspect": "attn"},
            {"prev_op": "ln_2", "layers": ["mlp.w2", "mlp.w1"], "inp": "mlp.w2", "module2inspect": "mlp"},
            {"prev_op": "mlp.w1", "layers": ["mlp.c_proj"], "inp": "mlp.c_proj"},
        ],
        model_decoder_layers="transformer.h",
    ),
    "opt": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "self_attn_layer_norm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.out_proj"], "inp": "self_attn.out_proj"},
            {"prev_op": "final_layer_norm", "layers": ["fc1"], "inp": "fc1"},
            {"prev_op": "fc1", "layers": ["fc2"], "inp": "fc2"},
        ],
        model_decoder_layers="model.decoder.layers",
    ),
    "phi": AWQConfig(
        scaling_layers=[{"prev_op": "self_attn.v_proj", "layers": ["self_attn.dense"], "inp": "self_attn.dense"}],
        model_decoder_layers="model.layers",
    ),
    "mistral": AWQConfig(
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
    ),
    "instella": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "pre_attention_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {
                "prev_op": "pre_feedforward_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.layers",
    ),
    "qwen2_moe": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
        ]
        + [
            {
                "prev_op": f"mlp.experts.{i}.up_proj",
                "layers": [f"mlp.experts.{i}.down_proj"],
                "inp": f"mlp.experts.{i}.down_proj",
            }
            for i in range(60)
        ],
        model_decoder_layers="model.layers",
    ),
    "qwen2": AWQConfig(
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
    ),
    "phi3": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.qkv_proj"],
                "inp": "self_attn.qkv_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.qkv_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_up_proj"],
                "inp": "mlp.gate_up_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.gate_up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.layers",
    ),
    "olmo": AWQConfig(
        scaling_layers=[
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.layers",
    ),
    "mixtral": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
        ],
        model_decoder_layers="model.layers",
    ),
    "grok-1": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "pre_attn_norm",
                "layers": ["attn.q_proj", "attn.k_proj", "attn.v_proj"],
                "inp": "attn.q_proj",
                "module2inspect": "attn",
                "has_kwargs": True,
            },
            {"prev_op": "attn.v_proj", "layers": ["attn.o_proj"], "inp": "attn.o_proj", "has_kwargs": False},
            {
                "prev_op": "pre_moe_norm",
                "layers": ["moe_block.experts.0.linear_v", "moe_block.experts.0.linear"],
                "inp": "moe_block",
                "module2inspect": "moe_block",
                "has_kwargs": False,
            },
        ]
        + [
            {
                "prev_op": f"moe_block.experts.{i}.linear",
                "layers": [f"moe_block.experts.{i}.linear_1"],
                "inp": f"moe_block.experts.{i}.linear_1",
                "has_kwargs": False,
            }
            for i in range(8)
        ],
        model_decoder_layers="model.layers",
    ),
    "gptj": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "ln_1",
                "layers": ["attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.fc_in"],
                "inp": "attn.q_proj",
                "module2inspect": "",
            },
            {"prev_op": "attn.v_proj", "layers": ["attn.out_proj"], "inp": "attn.out_proj"},
        ],
        model_decoder_layers="transformer.h",
    ),
    "chatglm": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attention.query_key_value"],
                "inp": "self_attention.query_key_value",
            },
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.dense_h_to_4h"],
                "inp": "mlp.dense_h_to_4h",
                "module2inspect": "mlp",
            },
        ],
        model_decoder_layers="transformer.encoder.layers",
    ),
    "gemma2": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {
                "prev_op": "pre_feedforward_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.layers",
    ),
    "deepseek_v2": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.layers",
    ),
    "deepseek_v3": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.layers",
    ),
    "gemma3": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {
                "prev_op": "pre_feedforward_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    "gemma3_text": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {
                "prev_op": "pre_feedforward_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.layers",
    ),
    "llama4": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {"prev_op": "up_proj", "layers": ["down_proj"], "inp": "down_proj"},
        ],
        model_decoder_layers="language_model.model.layers",
    ),
    "gpt_oss": AWQConfig(
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
        ],
        model_decoder_layers="model.layers",
    ),
}

# GPTQ configs, these are the default configs for each model, can be updated by user
GPTQ_MAP = {
    "llama": GPTQConfig(
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
        ],
        model_decoder_layers="model.layers",
        block_size=128,
        damp_percent=0.01,
    ),
    "qwen": GPTQConfig(
        inside_layer_modules=["attn.c_attn", "mlp.w2", "mlp.w1", "mlp.c_proj"], model_decoder_layers="transformer.h"
    ),
    "opt": GPTQConfig(
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.out_proj",
            "fc1",
            "fc2",
        ],
        model_decoder_layers="model.decoder.layers",
    ),
    "phi": GPTQConfig(
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.dense",
            "mlp.fc1",
            "mlp.fc2",
        ],
        model_decoder_layers="model.layers",
    ),
    "mistral": GPTQConfig(
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
        ],
        model_decoder_layers="model.layers",
    ),
    "deepseek": GPTQConfig(
        inside_layer_modules=[
            "self_attn.q_a_proj",
            "self_attn.q_b_proj",
            "self_attn.kv_a_proj_with_mqa",
            "self_attn.kv_b_proj",
            "self_attn.o_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
            "mlp.experts.*.up_proj",
            "mlp.experts.*.gate_proj",
            "mlp.experts.*.down_proj",
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.down_proj",
        ],
        model_decoder_layers="model.layers",
        desc_act=True,
    ),
    "qwen2": GPTQConfig(
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
        ],
        model_decoder_layers="model.layers",
    ),
    "phi3": GPTQConfig(
        inside_layer_modules=["self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"],
        model_decoder_layers="model.layers",
    ),
    "mixtral": GPTQConfig(
        inside_layer_modules=["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj"],
        model_decoder_layers="model.layers",
    ),
    "gptj": GPTQConfig(
        inside_layer_modules=["attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.fc_in", "attn.out_proj", "mlp.fc_out"],
        model_decoder_layers="transformer.h",
    ),
    "chatglm": GPTQConfig(
        inside_layer_modules=[
            "self_attention.query_key_value",
            "self_attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        ],
        model_decoder_layers="transformer.encoder.layers",
    ),
    "llama4": GPTQConfig(
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
        model_decoder_layers="language_model.model.layers",
        block_size=128,
        damp_percent=0.01,
    ),
    "deepseek_v2": GPTQConfig(
        inside_layer_modules=[
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
            "mlp.experts.*.up_proj",
            "mlp.experts.*.gate_proj",
            "mlp.experts.*.down_proj",
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.down_proj",
        ],
        model_decoder_layers="model.layers",
        desc_act=True,
    ),
    "deepseek_v3": GPTQConfig(
        inside_layer_modules=[
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
            "mlp.experts.*.up_proj",
            "mlp.experts.*.gate_proj",
            "mlp.experts.*.down_proj",
            "mlp.shared_experts.gate_proj",
            "mlp.shared_experts.up_proj",
            "mlp.shared_experts.down_proj",
        ],
        model_decoder_layers="model.layers",
        desc_act=True,
    ),
}

# SmoothQuant configs, these are the default configs for each model, can be updated by user
SQ_MAP = {
    "llama": SmoothQuantConfig(
        alpha=1,
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
    ),
    "qwen": SmoothQuantConfig(
        alpha=1,
        scaling_layers=[
            {"prev_op": "ln_1", "layers": ["attn.c_attn"], "inp": "attn.c_attn", "module2inspect": "attn"},
            {"prev_op": "ln_2", "layers": ["mlp.w2", "mlp.w1"], "inp": "mlp.w2", "module2inspect": "mlp"},
            {"prev_op": "mlp.w1", "layers": ["mlp.c_proj"], "inp": "mlp.c_proj"},
        ],
        model_decoder_layers="transformer.h",
    ),
    "opt": SmoothQuantConfig(
        alpha=1,
        scaling_layers=[
            {
                "prev_op": "self_attn_layer_norm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.out_proj"], "inp": "self_attn.out_proj"},
            {"prev_op": "final_layer_norm", "layers": ["fc1"], "inp": "fc1"},
            {"prev_op": "fc1", "layers": ["fc2"], "inp": "fc2"},
        ],
        model_decoder_layers="model.decoder.layers",
    ),
    "phi": SmoothQuantConfig(
        alpha=1,
        scaling_layers=[{"prev_op": "self_attn.v_proj", "layers": ["self_attn.dense"], "inp": "self_attn.dense"}],
        model_decoder_layers="model.layers",
    ),
    "mistral": SmoothQuantConfig(
        alpha=1,
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
    ),
    "qwen2": SmoothQuantConfig(
        alpha=1,
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
    ),
    "phi3": SmoothQuantConfig(
        alpha=1,
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.qkv_proj"],
                "inp": "self_attn.qkv_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.qkv_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_up_proj"],
                "inp": "mlp.gate_up_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.gate_up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.layers",
    ),
    "gptj": SmoothQuantConfig(
        alpha=1,
        scaling_layers=[
            {
                "prev_op": "ln_1",
                "layers": ["attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.fc_in"],
                "inp": "attn.q_proj",
                "module2inspect": "",
            },
            {"prev_op": "attn.v_proj", "layers": ["attn.out_proj"], "inp": "attn.out_proj"},
        ],
        model_decoder_layers="transformer.h",
    ),
    "cohere": SmoothQuantConfig(
        alpha=0.50,
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.layers",
    ),
    "chatglm": SmoothQuantConfig(
        alpha=1,
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attention.query_key_value"],
                "inp": "self_attention.query_key_value",
            },
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.dense_h_to_4h"],
                "inp": "mlp.dense_h_to_4h",
                "module2inspect": "mlp",
            },
        ],
        model_decoder_layers="transformer.encoder.layers",
    ),
    "deepseek_v2": SmoothQuantConfig(
        alpha=0.8,
        scale_clamp_min=1e-3,
        scaling_layers=[
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
            {"prev_op": "up_proj", "layers": ["down_proj"], "inp": "down_proj"},
        ],
        model_decoder_layers="model.layers",
    ),
    "deepseek_v3": SmoothQuantConfig(
        alpha=0.8,
        scale_clamp_min=1e-3,
        scaling_layers=[
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
            {"prev_op": "up_proj", "layers": ["down_proj"], "inp": "down_proj"},
        ],
        model_decoder_layers="model.layers",
    ),
    "llama4": SmoothQuantConfig(
        alpha=0.8,
        scale_clamp_min=1e-3,
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {"prev_op": "up_proj", "layers": ["down_proj"], "inp": "down_proj"},
        ],
        model_decoder_layers="language_model.model.layers",
    ),
    "mixtral": SmoothQuantConfig(
        alpha=0.8,
        scale_clamp_min=1e-3,
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
        ],
        model_decoder_layers="model.layers",
    ),
}

# AutoSmoothQuant configs, these are the default configs for each model
AUTOSMOOTHQUANT_MAP = {
    "llama": AutoSmoothQuantConfig(
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
        compute_scale_loss="MAE",
    ),
    "llama4": AutoSmoothQuantConfig(
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {"prev_op": "up_proj", "layers": ["down_proj"], "inp": "down_proj"},
        ],
        model_decoder_layers="language_model.model.layers",
        compute_scale_loss="MAE",
    ),
    "mixtral": AutoSmoothQuantConfig(
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
        ],
        model_decoder_layers="model.layers",
        compute_scale_loss="MAE",
    ),
    "deepseek_v2": AutoSmoothQuantConfig(
        scaling_layers=[
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
            {"prev_op": "up_proj", "layers": ["down_proj"], "inp": "down_proj"},
        ],
        model_decoder_layers="model.layers",
        compute_scale_loss="MAE",
    ),
    "deepseek_v3": AutoSmoothQuantConfig(
        scaling_layers=[
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
            {"prev_op": "up_proj", "layers": ["down_proj"], "inp": "down_proj"},
        ],
        model_decoder_layers="model.layers",
        compute_scale_loss="MAE",
    ),
}

# Rotation configs, these are the default configs for each model
ROTATION_MAP = {
    "llama": RotationConfig(
        model_decoder_layers="model.layers",
        scaling_layers={
            "first_layer": [
                {
                    "prev_modules": ["model.embed_tokens"],
                    "norm_module": "model.layers.layer_id.input_layernorm",
                    "next_modules": [
                        "model.layers.layer_id.self_attn.q_proj",
                        "model.layers.layer_id.self_attn.k_proj",
                        "model.layers.layer_id.self_attn.v_proj",
                    ],
                },
                {
                    "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                    "norm_module": "model.layers.layer_id.post_attention_layernorm",
                    "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"],
                },
            ],
            "middle_layers": [
                {
                    "prev_modules": ["model.layers.pre_layer_id.mlp.down_proj"],
                    "norm_module": "model.layers.layer_id.input_layernorm",
                    "next_modules": [
                        "model.layers.layer_id.self_attn.q_proj",
                        "model.layers.layer_id.self_attn.k_proj",
                        "model.layers.layer_id.self_attn.v_proj",
                    ],
                },
                {
                    "prev_modules": ["model.layers.layer_id.self_attn.o_proj"],
                    "norm_module": "model.layers.layer_id.post_attention_layernorm",
                    "next_modules": ["model.layers.layer_id.mlp.up_proj", "model.layers.layer_id.mlp.gate_proj"],
                },
            ],
            "last_layer": [
                {
                    "prev_modules": ["model.layers.layer_id.mlp.down_proj"],
                    "norm_module": "model.norm",
                    "next_modules": ["lm_head"],
                }
            ],
        },
    ),
}


def get_algo_config(
    algo_type: str, model_type: str
) -> Union[AWQConfig, GPTQConfig, SmoothQuantConfig, AutoSmoothQuantConfig, RotationConfig, None]:
    algo_type = algo_type.lower()

    if algo_type == "awq":
        if model_type not in AWQ_MAP:
            return None
        return AWQ_MAP[model_type]

    elif algo_type == "gptq":
        if model_type not in GPTQ_MAP:
            return None
        return GPTQ_MAP[model_type]

    elif algo_type == "smoothquant":
        if model_type not in SQ_MAP:
            return None
        return SQ_MAP[model_type]

    elif algo_type == "autosmoothquant":
        if model_type not in AUTOSMOOTHQUANT_MAP:
            return None
        return AUTOSMOOTHQUANT_MAP[model_type]
    elif algo_type == "rotation":
        if model_type not in ROTATION_MAP:
            return None
        return ROTATION_MAP[model_type]
    else:
        raise ValueError(
            f"Unsupported algorithm type: {algo_type}. Supported types: awq, gptq, smoothquant, autosmoothquant, rotation"
        )
