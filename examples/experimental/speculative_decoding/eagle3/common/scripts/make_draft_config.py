#!/usr/bin/env python3
# Derive a single-layer EAGLE-3 draft config from a target's public config.json.
import argparse
import json
import os


def _text_config(config):
    current = config
    for _ in range(3):
        nested = next(
            (
                current[key]
                for key in ("text_config", "language_config", "llm_config")
                if isinstance(current.get(key), dict)
            ),
            None,
        )
        if nested is None:
            break
        current = nested
    return current


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True, help="target model dir or its config.json")
    parser.add_argument("--out", required=True, help="where to write the draft config JSON")
    parser.add_argument(
        "--draft-vocab-size",
        type=int,
        default=0,
        help="0 (default) -> full vocab (vLLM-loadable, no d2t/t2d mapping)",
    )
    args = parser.parse_args()

    config_path = args.target
    if os.path.isdir(config_path):
        config_path = os.path.join(config_path, "config.json")
    with open(config_path) as f:
        config = _text_config(json.load(f))

    hidden_size = config["hidden_size"]
    attention_heads = config["num_attention_heads"]
    head_dim = config.get("head_dim") or (hidden_size // attention_heads)
    vocab_size = config["vocab_size"]
    target_depth = config["num_hidden_layers"]
    # Published EAGLE-3 low/middle/high triple, the default in vLLM, SGLang and
    # NeMo. The high state has to sit near the output; an evenly spaced rule
    # stops short of it and measurably lowers acceptance length. Shallow targets
    # fall back to spacing, where the triple would collide or leave the range.
    aux_layers = [2, target_depth // 2, target_depth - 3]
    if aux_layers != sorted(set(aux_layers)) or aux_layers[-1] >= target_depth:
        aux_layers = [(index * target_depth) // 4 for index in (1, 2, 3)]
    target_intermediate = config.get("intermediate_size")
    intermediate_size = (
        target_intermediate
        if isinstance(target_intermediate, int) and target_intermediate >= 2 * hidden_size
        else 3 * hidden_size
    )

    draft = {
        "architectures": ["LlamaForCausalLMEagle3"],
        "model_type": "llama",
        "num_hidden_layers": 1,
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "num_attention_heads": attention_heads,
        "num_key_value_heads": config.get("num_key_value_heads", attention_heads),
        "head_dim": head_dim,
        # The Llama EAGLE-3 draft uses a dense SwiGLU MLP even when the target
        # exposes a model-specific MoE expert activation.
        "hidden_act": "silu",
        "attention_bias": config.get("attention_bias", False),
        "attention_dropout": 0.0,
        "rms_norm_eps": config.get("rms_norm_eps", 1e-6),
        "rope_theta": config.get("rope_theta", 10000.0),
        "rope_scaling": config.get("rope_scaling"),
        "max_position_embeddings": config.get("max_position_embeddings", 4096),
        "initializer_range": config.get("initializer_range", 0.02),
        "bos_token_id": config.get("bos_token_id"),
        "eos_token_id": config.get("eos_token_id"),
        "tie_word_embeddings": False,
        "torch_dtype": config.get("torch_dtype", "bfloat16"),
        "use_cache": True,
        "vocab_size": vocab_size,
        "draft_vocab_size": args.draft_vocab_size or vocab_size,
        "target_hidden_size": hidden_size,
        "target_num_hidden_layers": target_depth,
        "eagle_aux_hidden_state_layer_ids": aux_layers,
        "fc_norm": True,
        "norm_output": True,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(draft, f, indent=2)
    print(
        f"wrote draft config (1L, hidden={hidden_size}, heads={attention_heads}, "
        f"kv={draft['num_key_value_heads']}, vocab={vocab_size}, aux={aux_layers}) -> {args.out}"
    )


if __name__ == "__main__":
    main()
