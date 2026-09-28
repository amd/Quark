#
# Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from collections.abc import Mapping
from typing import TYPE_CHECKING

from quark.torch.quantization.config.config import (
    AlgoConfig,
    AutoRoundConfig,
    AutoSmoothQuantConfig,
    AWQConfig,
    GPTAQConfig,
    GPTQConfig,
    QronosConfig,
    RotationConfig,
    SmoothQuantConfig,
)

if TYPE_CHECKING:
    from quark.experimental.torch.twobitscalar.config import TwoBitScalarConfig

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
    "hunyuan_v1_dense": AWQConfig(
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
    # TODO qwen3_moe: AWQConfig
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
    # AWQ smooths ONLY the MLP, not attention. The reparameterization is exact in full precision for any
    # preceding norm, so the input RMSNorm is not the issue; the differentiator is QK-norm. gemma4 applies
    # q_norm/k_norm to the q/k projection *outputs*, which re-normalizes them over head_dim and distorts the
    # per-channel activation-magnitude signal the scale search relies on. The shared q/k/v scale is then
    # miscalibrated and hurts the attention quant grid (worse than plain RTN). The MLP has no such downstream
    # renorm, so smoothing there still helps. Same structure as qwen3_5 (also QK-norm) and gemma4_unified;
    # Llama/Qwen2 lack QK-norm and scale attention fine.
    #
    # The three entries target the two distinct FFN paths by their HF module names:
    #   - ``mlp.*`` (Gemma4TextMLP) is the always-on dense branch -- the routed MoE's shared expert on the
    #     26B, and the sole MLP on the dense E-series E2B/E4B/31B. Entries 1-2 scale it in both cases.
    #   - ``experts.*`` are the 128 routed experts (26B only). The wildcard third entry scales only these,
    #     matching every expert regardless of count and matching nothing on the dense E-series.
    # Do NOT collapse entry 3 to ``["down_proj"]``: the MoE routing globs ``"*" + layer_name``, so it would
    # also re-match ``mlp.down_proj`` (already scaled by entry 2) and double-scale the shared branch.
    # One config thus serves both variants under the shared model_type=gemma4 -- no hardcoded expert count.
    "gemma4": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "pre_feedforward_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
            {"prev_op": "up_proj", "layers": ["experts.*.down_proj"], "inp": "down_proj"},
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    # AWQ smooths ONLY the MLP, not attention. The reparameterization is exact in full precision for any
    # preceding norm, so the input RMSNorm is not the issue; the differentiator is QK-norm. gemma4 applies
    # q_norm/k_norm to the q/k projection *outputs*, re-normalizing them over head_dim and distorting the
    # activation-magnitude signal the scale search uses, so the shared q/k/v scale hurts the attention
    # quant grid. Empirically confirmed (12B GSM8K: attn+MLP 0.759 vs MLP-only 0.895). Same reasoning as
    # qwen3_5_text (also QK-norm); Llama/Qwen2 lack QK-norm and scale attention fine.
    "gemma4_unified": AWQConfig(
        scaling_layers=[
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
        ],
        model_decoder_layers="model.layers",
    ),
    "granitemoehybrid": AWQConfig(
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
    # Qwen3.5 dense model
    # AWQ smooths ONLY the MLP. Smoothing attention / linear_attn empirically *hurts* on this model
    "qwen3_5": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    "muse_glimmer": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "input_layernorm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.gate_proj"],
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
    # Hybrid conv+attention decoder: the attention groups hang off operator_norm and out_proj,
    # not input_layernorm/o_proj. Scaling attention despite LFM2's QK-norm (q_layernorm/
    # k_layernorm) is a deliberate trade-off measured on uint4_wo_128 PPL: it helps LFM2-2.6B
    # (-6.4%), LFM2-2.6B-Exp (-2.9%) and LFM2.5-1.2B-Instruct (-4.6%), but regresses
    # LFM2.5-1.2B-Thinking (+10.2%) and LFM2.5-350M (+67.6%); override via
    # --quant_algo_config_file if the MLP-only variant suits a given checkpoint better.
    # conv-only layers have no self_attn and get_layers_for_scaling skips the missing modules;
    # conv.in_proj/conv.out_proj are excluded from quantization entirely (see template.py).
    "lfm2": AWQConfig(
        scaling_layers=[
            {
                "prev_op": "operator_norm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.out_proj"], "inp": "self_attn.out_proj"},
            {
                "prev_op": "ffn_norm",
                "layers": ["feed_forward.w1", "feed_forward.w3"],
                "inp": "feed_forward.w1",
                "module2inspect": "feed_forward",
            },
            {"prev_op": "feed_forward.w3", "layers": ["feed_forward.w2"], "inp": "feed_forward.w2"},
        ],
        model_decoder_layers="model.layers",
    ),
    "qwen3_5_moe": AWQConfig(
        scaling_layers=[
            {"prev_op": "up_proj", "layers": ["down_proj"], "inp": "down_proj"},
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    # Bare leaf names: get_layers_for_scaling takes the MoE branch and re-anchors prev_op and inp
    # against each matched module's own parent, so one entry pairs every routed expert and the
    # shared expert with its own up_proj. No attention entry: there is no layernorm to fold into,
    # and expert gate/up cannot be smoothed (their predecessor also feeds the router).
    "qwen4_exp": AWQConfig(
        scaling_layers=[{"prev_op": "up_proj", "layers": ["down_proj"], "inp": "down_proj"}],
        model_decoder_layers="model.language_model.layers",
    ),
    # Alias: live loading resolves model_type to "qwen4_exp_text" (see template.py).
    "qwen4_exp_text": AWQConfig(
        scaling_layers=[{"prev_op": "up_proj", "layers": ["down_proj"], "inp": "down_proj"}],
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
    "hunyuan_v1_dense": GPTQConfig(
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
    "grok-1": GPTQConfig(
        inside_layer_modules=[
            "attn.k_proj",
            "attn.v_proj",
            "attn.q_proj",
            "attn.o_proj",
            "*.linear",
            "*.linear_1",
            "*.linear_v",
        ],
        model_decoder_layers="model.layers",
        block_size=128,
        damp_percent=0.01,
    ),
    "gemma2": GPTQConfig(
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
        desc_act=True,
    ),
    "granitemoehybrid": GPTQConfig(
        inside_layer_modules=["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj"],
        model_decoder_layers="model.layers",
    ),
    "gemma3_text": GPTQConfig(
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
        model_decoder_layers="model.layers",
    ),
    "gemma3": GPTQConfig(
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    "gemma4": GPTQConfig(
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
            "experts.*.up_proj",
            "experts.*.gate_proj",
            "experts.*.down_proj",
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    "gemma4_unified": GPTQConfig(
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    "gpt_oss": GPTQConfig(
        inside_layer_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "gate_up_proj",
            "down_proj",
        ],
        model_decoder_layers="model.layers",
    ),
    "muse_glimmer": GPTQConfig(
        inside_layer_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "self_attn.gate_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    "olmo": GPTQConfig(
        inside_layer_modules=["self_attn.v_proj", "self_attn.o_proj", "mlp.up_proj", "mlp.down_proj"],
        model_decoder_layers="model.layers",
    ),
    "qwen3_5": GPTQConfig(
        # Hybrid architecture: blocks alternate between full self_attn layers and gated
        # linear_attn layers (separate in_proj_qkv/in_proj_z/in_proj_a/in_proj_b + out_proj
        # Linears, no fused qkv). Listing both attention flavors here is safe -- the shared
        # blockwise processor (algorithm/common.py) fnmatch-filters against each block's actual
        # modules, so entries absent from a given block are simply skipped.
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_a",
            "linear_attn.in_proj_b",
            "linear_attn.out_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
        ],
        model_decoder_layers="model.language_model.layers",
        block_size=128,
        damp_percent=0.01,
    ),
    # conv.in_proj/conv.out_proj intentionally absent -- excluded from quantization entirely
    # (see template.py); the shared blockwise processor fnmatch-filters against each block's
    # actual modules, so conv-only layers just skip the self_attn.* entries below.
    "lfm2": GPTQConfig(
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.out_proj",
            "feed_forward.w1",
            "feed_forward.w3",
            "feed_forward.w2",
        ],
        model_decoder_layers="model.layers",
    ),
    # qwen4_exp: only routed MoE experts quantized (see template.py); rest excluded, no
    # precedent for this architecture's novel components.
    "qwen4_exp": GPTQConfig(
        inside_layer_modules=[
            # Leaf suffixes: selection fnmatches "*" + pattern against the preprocessed module
            # names, so one entry covers every routed expert and the shared expert alike. Note a
            # path-like "mlp.experts.down_proj" would match nothing -- after preprocessing the
            # routed names are indexed (mlp.experts.<i>.down_proj).
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    # Alias: live loading resolves model_type to "qwen4_exp_text" (see template.py).
    "qwen4_exp_text": GPTQConfig(
        inside_layer_modules=[
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        model_decoder_layers="model.layers",
    ),
    "qwen3_5_moe": GPTQConfig(
        # Same hybrid attention layout as the dense qwen3_5 config, with the dense MLP replaced by
        # the sparse MoE block: a shared expert plus routed experts. `mlp.gate` (the router) and
        # `mlp.shared_expert_gate` stay unlisted -- they are tiny projections whose outputs pick
        # experts rather than carry activations, so quantizing them perturbs routing decisions for
        # no meaningful compression. model_decoder_layers points at the multimodal wrapper path.
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_a",
            "linear_attn.in_proj_b",
            "linear_attn.out_proj",
            "mlp.experts.*.up_proj",
            "mlp.experts.*.gate_proj",
            "mlp.experts.*.down_proj",
            "mlp.shared_expert.gate_proj",
            "mlp.shared_expert.up_proj",
            "mlp.shared_expert.down_proj",
        ],
        model_decoder_layers="model.language_model.layers",
        block_size=128,
        damp_percent=0.01,
    ),
}

GPTAQ_MAP = {
    "llama": GPTAQConfig(
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
    ),
    "qwen": GPTAQConfig(
        inside_layer_modules=["attn.c_attn", "mlp.w2", "mlp.w1", "mlp.c_proj"], model_decoder_layers="transformer.h"
    ),
    "opt": GPTAQConfig(
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
    "phi": GPTAQConfig(
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
    "mistral": GPTAQConfig(
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
    "deepseek": GPTAQConfig(
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
    "qwen2": GPTAQConfig(
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
    "phi3": GPTAQConfig(
        inside_layer_modules=["self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"],
        model_decoder_layers="model.layers",
    ),
    "mixtral": GPTAQConfig(
        inside_layer_modules=["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj"],
        model_decoder_layers="model.layers",
    ),
    "gptj": GPTAQConfig(
        inside_layer_modules=["attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.fc_in", "attn.out_proj", "mlp.fc_out"],
        model_decoder_layers="transformer.h",
    ),
    "chatglm": GPTAQConfig(
        inside_layer_modules=[
            "self_attention.query_key_value",
            "self_attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        ],
        model_decoder_layers="transformer.encoder.layers",
    ),
    "llama4": GPTAQConfig(
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
    ),
    "deepseek_v2": GPTAQConfig(
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
    "deepseek_v3": GPTAQConfig(
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
    "granitemoehybrid": GPTAQConfig(
        inside_layer_modules=["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj"],
        model_decoder_layers="model.layers",
    ),
    # See qwen3_5 note in GPTQ_MAP above — both self_attn.* and linear_attn.* patterns
    # are listed since layers are heterogeneous; unmatched patterns are skipped per-layer.
    "qwen3_5": GPTAQConfig(
        inside_layer_modules=[
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_a",
            "linear_attn.in_proj_b",
            "linear_attn.in_proj_z",
            "linear_attn.out_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    "qwen4_exp": GPTAQConfig(
        inside_layer_modules=[
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    # Alias: live loading resolves model_type to "qwen4_exp_text" (see template.py).
    "qwen4_exp_text": GPTAQConfig(
        inside_layer_modules=[
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        model_decoder_layers="model.layers",
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
    "hunyuan_v1_dense": SmoothQuantConfig(
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
    "granitemoehybrid": SmoothQuantConfig(
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
    "gpt_oss": SmoothQuantConfig(
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
    "gemma2": SmoothQuantConfig(
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
    "gemma3": SmoothQuantConfig(
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
    # Smooth ONLY the MLP, not attention. Smoothing is exact in full precision for any preceding norm; the
    # differentiator is QK-norm. gemma4's q_norm/k_norm re-normalize the q/k projection *outputs* over
    # head_dim, distorting the activation-magnitude signal, so the shared q/k/v smoothing scale hurts the
    # attention quant grid. Same reasoning as the gemma4 AWQ config.
    #
    # The three entries target the two distinct FFN paths by their HF module names:
    #   - ``mlp.*`` (Gemma4TextMLP) is the always-on dense branch -- the routed MoE's shared expert on the
    #     26B, and the sole MLP on the dense E-series E2B/E4B/31B. Entries 1-2 smooth it in both cases.
    #   - ``experts.*`` are the 128 routed experts (26B only). The wildcard third entry smooths only these,
    #     matching every expert regardless of count and matching nothing on the dense E-series.
    # Do NOT collapse entry 3 to ``["down_proj"]``: the MoE routing globs ``"*" + layer_name``, so it would
    # also re-match ``mlp.down_proj`` (already smoothed by entry 2) and double-scale the shared branch.
    # One config thus serves both variants under the shared model_type=gemma4 -- no hardcoded expert count.
    "gemma4": SmoothQuantConfig(
        scaling_layers=[
            {
                "prev_op": "pre_feedforward_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
            {"prev_op": "up_proj", "layers": ["experts.*.down_proj"], "inp": "down_proj"},
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    # Smooth ONLY the MLP, not attention. Smoothing is exact in full precision for any preceding norm; the
    # differentiator is QK-norm. gemma4's q_norm/k_norm re-normalize the q/k projection *outputs* over
    # head_dim, distorting the activation-magnitude signal, so the shared q/k/v smoothing scale hurts the
    # attention quant grid. Same reasoning as the gemma4_unified AWQ config.
    "gemma4_unified": SmoothQuantConfig(
        scaling_layers=[
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
    "gemma3_text": SmoothQuantConfig(
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
    "olmo": SmoothQuantConfig(
        scaling_layers=[
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.o_proj"], "inp": "self_attn.o_proj"},
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.layers",
    ),
    # Qwen3.5 dense model. Following the AWQ finding for this architecture, smoothing is
    # restricted to the MLP; the MLP structure is identical in both the standard-attention
    # and gated-delta linear-attention decoder layers, so a single config covers all layers.
    # alpha=0.5 is required here: the default alpha=1 pushes all quantization difficulty onto
    # the weights and badly hurts W8A8 accuracy on this model (perplexity 122 vs. 48.5 for the
    # un-smoothed W8A8 baseline); alpha=0.5 brings it down to ~33.
    "qwen3_5": SmoothQuantConfig(
        alpha=0.5,
        scaling_layers=[
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    "lfm2": SmoothQuantConfig(
        scaling_layers=[
            {
                "prev_op": "operator_norm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.out_proj"], "inp": "self_attn.out_proj"},
            {
                "prev_op": "ffn_norm",
                "layers": ["feed_forward.w1", "feed_forward.w3"],
                "inp": "feed_forward.w1",
                "module2inspect": "feed_forward",
            },
            {"prev_op": "feed_forward.w3", "layers": ["feed_forward.w2"], "inp": "feed_forward.w2"},
        ],
        model_decoder_layers="model.layers",
    ),
    "qwen3_5_moe": SmoothQuantConfig(
        alpha=0.5,
        scaling_layers=[
            {"prev_op": "up_proj", "layers": ["down_proj"], "inp": "down_proj"},
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    "grok-1": SmoothQuantConfig(
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
}


# Qronos configs, these are the default configs for each model, can be updated by user
QRONOS_MAP = {
    "llama": QronosConfig(
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
    ),
    "qwen": QronosConfig(
        inside_layer_modules=["attn.c_attn", "mlp.w2", "mlp.w1", "mlp.c_proj"], model_decoder_layers="transformer.h"
    ),
    "opt": QronosConfig(
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
    "phi": QronosConfig(
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
    "mistral": QronosConfig(
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
    "deepseek": QronosConfig(
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
    "qwen2": QronosConfig(
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
    "phi3": QronosConfig(
        inside_layer_modules=["self_attn.qkv_proj", "self_attn.o_proj", "mlp.gate_up_proj", "mlp.down_proj"],
        model_decoder_layers="model.layers",
    ),
    "mixtral": QronosConfig(
        inside_layer_modules=["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj"],
        model_decoder_layers="model.layers",
    ),
    "gptj": QronosConfig(
        inside_layer_modules=["attn.q_proj", "attn.k_proj", "attn.v_proj", "mlp.fc_in", "attn.out_proj", "mlp.fc_out"],
        model_decoder_layers="transformer.h",
    ),
    "chatglm": QronosConfig(
        inside_layer_modules=[
            "self_attention.query_key_value",
            "self_attention.dense",
            "mlp.dense_h_to_4h",
            "mlp.dense_4h_to_h",
        ],
        model_decoder_layers="transformer.encoder.layers",
    ),
    "llama4": QronosConfig(
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
    ),
    "deepseek_v2": QronosConfig(
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
    "deepseek_v3": QronosConfig(
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
    "granitemoehybrid": QronosConfig(
        inside_layer_modules=["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj", "self_attn.o_proj"],
        model_decoder_layers="model.layers",
    ),
    # Qwen3.5 dense model has a hybrid decoder: some layers use standard attention
    # (self_attn.*), others use gated-delta linear attention (linear_attn.*). Both
    # sets of module names are listed here; fnmatch skips whichever set is absent in a
    # given layer, so a single config covers all decoder layers.
    "qwen3_5": QronosConfig(
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_b",
            "linear_attn.in_proj_a",
            "linear_attn.out_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
        ],
        model_decoder_layers="model.language_model.layers",
        block_size=128,
    ),
    "qwen4_exp": QronosConfig(
        inside_layer_modules=[
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    # Alias: live loading resolves model_type to "qwen4_exp_text" (see template.py).
    "qwen4_exp_text": QronosConfig(
        inside_layer_modules=[
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        model_decoder_layers="model.layers",
    ),
    "qwen3_5_moe": QronosConfig(
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_b",
            "linear_attn.in_proj_a",
            "linear_attn.out_proj",
            "mlp.experts.*.up_proj",
            "mlp.experts.*.gate_proj",
            "mlp.experts.*.down_proj",
            "mlp.shared_expert.gate_proj",
            "mlp.shared_expert.up_proj",
            "mlp.shared_expert.down_proj",
        ],
        model_decoder_layers="model.language_model.layers",
        block_size=128,
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
    "hunyuan_v1_dense": AutoSmoothQuantConfig(
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
    "qwen3_moe": AutoSmoothQuantConfig(
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
            for i in range(128)
        ],
        model_decoder_layers="model.layers",
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
    "granitemoehybrid": AutoSmoothQuantConfig(
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
    "lfm2": AutoSmoothQuantConfig(
        scaling_layers=[
            {
                "prev_op": "operator_norm",
                "layers": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
                "inp": "self_attn.q_proj",
                "module2inspect": "self_attn",
            },
            {"prev_op": "self_attn.v_proj", "layers": ["self_attn.out_proj"], "inp": "self_attn.out_proj"},
            {
                "prev_op": "ffn_norm",
                "layers": ["feed_forward.w1", "feed_forward.w3"],
                "inp": "feed_forward.w1",
                "module2inspect": "feed_forward",
            },
            {"prev_op": "feed_forward.w3", "layers": ["feed_forward.w2"], "inp": "feed_forward.w2"},
        ],
        model_decoder_layers="model.layers",
    ),
    # MLP-only, matching SQ_MAP/AWQ_MAP["qwen3_5"]: smoothing attention hurts on this model.
    "qwen3_5": AutoSmoothQuantConfig(
        scaling_layers=[
            {
                "prev_op": "post_attention_layernorm",
                "layers": ["mlp.gate_proj", "mlp.up_proj"],
                "inp": "mlp.gate_proj",
                "module2inspect": "mlp",
            },
            {"prev_op": "mlp.up_proj", "layers": ["mlp.down_proj"], "inp": "mlp.down_proj"},
        ],
        model_decoder_layers="model.language_model.layers",
    ),
    # Same routed-expert entry as AWQ_MAP["qwen4_exp"] -- see the rationale there.
    "qwen4_exp": AutoSmoothQuantConfig(
        scaling_layers=[{"prev_op": "up_proj", "layers": ["down_proj"], "inp": "down_proj"}],
        model_decoder_layers="model.language_model.layers",
    ),
    # Alias: live loading resolves model_type to "qwen4_exp_text" (see template.py).
    "qwen4_exp_text": AutoSmoothQuantConfig(
        scaling_layers=[{"prev_op": "up_proj", "layers": ["down_proj"], "inp": "down_proj"}],
        model_decoder_layers="model.layers",
    ),
    "qwen3_5_moe": AutoSmoothQuantConfig(
        scaling_layers=[
            {"prev_op": "up_proj", "layers": ["down_proj"], "inp": "down_proj"},
        ],
        model_decoder_layers="model.language_model.layers",
        compute_scale_loss="MAE",
    ),
}

# Rotation configs, these are the default configs for each model
ROTATION_MAP = {
    "llama": RotationConfig(
        backbone="model",
        model_decoder_layers="model.layers",
        v_proj="self_attn.v_proj",
        o_proj="self_attn.o_proj",
        self_attn="self_attn",
        mlp="mlp",
        r1=True,
        r2=False,
        r3=False,
        r4=False,
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


# AutoRound (SignRound) configs — default per-model configs, can be updated by the user.
# Hyperparameters (iters, lr, batch_size, ...) keep their AutoRoundConfig defaults here; the
# per-model entries fix the model-structure fields (inside_layer_modules / model_decoder_layers).
AUTOROUND_MAP = {
    "llama": AutoRoundConfig(
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
    "mistral": AutoRoundConfig(
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
    "qwen3_5": AutoRoundConfig(
        # Hybrid architecture: blocks alternate between full self_attn layers and gated
        # linear_attn layers (separate in_proj_qkv/in_proj_z/in_proj_a/in_proj_b + out_proj
        # Linears, no fused qkv). Listing both attention flavors here is safe -- AutoRoundProcessor
        # skips any entry absent from a given block.
        inside_layer_modules=[
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.q_proj",
            "self_attn.o_proj",
            "linear_attn.in_proj_qkv",
            "linear_attn.in_proj_z",
            "linear_attn.in_proj_a",
            "linear_attn.in_proj_b",
            "linear_attn.out_proj",
            "mlp.up_proj",
            "mlp.gate_proj",
            "mlp.down_proj",
        ],
        model_decoder_layers="model.language_model.layers",
    ),
}


def _twobitscalar_default() -> "TwoBitScalarConfig":
    """Model-agnostic TwoBitScalar default (2-bit, g=64, AWQ+SRHT, Lloyd-Max).

    Only ``embed_tokens``/``lm_head`` are excluded, which is standard across these
    decoder-LLM architectures, so every supported model type maps to the same
    settings. A fresh instance is returned per call to avoid shared mutable state.
    """
    from quark.experimental.torch.twobitscalar.config import TwoBitScalarConfig

    return TwoBitScalarConfig(
        bits=2,
        group_size=64,
        act_scale_alpha=0.5,
        enable_incoherence=True,
        use_lloyd_max_levels=True,
        exclude_layers=["*embed_tokens*", "*lm_head*"],
    )


# Built-in TwoBitScalar defaults per model type (users can still override via a
# config file / `algo_configs`). The settings are model-agnostic; the same default
# is used for every supported decoder-LLM architecture.
TWOBITSCALAR_MAP: Mapping[str, AlgoConfig] = {
    model_type: _twobitscalar_default()  # type: ignore[misc]
    for model_type in ("llama", "qwen2", "qwen3", "phi3", "mistral")
}


ALGORITHM_CONFIG_MAPS: dict[str, Mapping[str, AlgoConfig]] = {
    "awq": AWQ_MAP,
    "gptq": GPTQ_MAP,
    "gptaq": GPTAQ_MAP,
    "qronos": QRONOS_MAP,
    "smoothquant": SQ_MAP,
    "autosmoothquant": AUTOSMOOTHQUANT_MAP,
    "rotation": ROTATION_MAP,
    "autoround": AUTOROUND_MAP,
    "twobitscalar": TWOBITSCALAR_MAP,
}


def get_supported_algorithm_types() -> list[str]:
    """Return the supported algorithm type names, core's own followed by the registry's.

    A ``QuarkAlgorithm`` claiming a core algorithm's name is listed once, in core's position: it
    overrides that algorithm rather than adding a second one.
    """
    # Imported lazily: an algorithm module imports the config classes defined alongside this one,
    # so a module-level import here would be a cycle.
    from quark.torch.algorithm.registry import ALGORITHM_REGISTRY

    supported_algorithm_types = list(ALGORITHM_CONFIG_MAPS)
    supported_algorithm_types.extend(
        algorithm.name
        for algorithm in ALGORITHM_REGISTRY.get_algorithms()
        if algorithm.name not in ALGORITHM_CONFIG_MAPS
    )

    return supported_algorithm_types


def get_algo_config(algo_type: str, model_type: str) -> AlgoConfig | None:
    from quark.torch.algorithm.registry import ALGORITHM_REGISTRY

    normalized_algorithm_type = algo_type.lower()

    # The registry is consulted first, so a `QuarkAlgorithm` claiming a core algorithm's name ships
    # the defaults for it — the same precedence the config loader and the processor dispatch use.
    algorithm = ALGORITHM_REGISTRY.get(normalized_algorithm_type)
    if algorithm is not None:
        # `algo_config_map` is typed on `BaseAlgoConfig`, `AlgoConfig`'s base, so an algorithm may
        # hold a config that is not statically an `AlgoConfig`. Not narrowed at runtime: core's own
        # `TWOBITSCALAR_MAP` above puts exactly such a value in an `AlgoConfig`-typed map under the
        # same suppression, so rejecting one here would make a registered algorithm stricter than
        # the identical built-in. `AlgoConfig` is an empty marker subclass; nothing reads past it.
        return algorithm.algo_config_map.get(model_type)  # type: ignore[return-value]

    if normalized_algorithm_type not in ALGORITHM_CONFIG_MAPS:
        supported_algorithm_types = ", ".join(get_supported_algorithm_types())
        raise ValueError(
            f"Unsupported algorithm type: {normalized_algorithm_type}. Supported types: {supported_algorithm_types}"
        )

    algorithm_config_map = ALGORITHM_CONFIG_MAPS[normalized_algorithm_type]
    return algorithm_config_map.get(model_type)
