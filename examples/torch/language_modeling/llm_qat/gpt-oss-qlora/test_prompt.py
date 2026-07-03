#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import transformers
from transformers import AutoModelForCausalLM, Mxfp4Config
from trl import ModelConfig, ScriptArguments, SFTConfig, TrlParser
from utils import QuarkQuantArguments

from quark.torch import import_model_from_safetensors
from quark.torch.utils.llm import preprocess_for_quantization


def main(quark_args, script_args, training_args, model_args):
    """
    Load and test a quantized language model with a sample prompt.
    """
    # 1. Prepare the original GPT-OSS model
    print("\n[INFO]: Loading model ...")
    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        device_map=quark_args.device_map,
        torch_dtype=model_args.dtype,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)
    model.eval()
    preprocess_for_quantization(model)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=model_args.trust_remote_code
    )

    if quark_args.import_model_dir is not None:
        print("\nRestore quantized model from hf_format safetensors file ...")
        model = import_model_from_safetensors(
            model, model_dir=quark_args.import_model_dir, multi_device=quark_args.multi_device
        )

    user_prompt = "Tell me 5 ways to make fire."
    messages = [
        {"role": "user", "content": user_prompt},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
    gen_kwargs = {"max_new_tokens": 512, "do_sample": True, "temperature": 0.6, "top_p": None, "top_k": None}

    output_ids = model.generate(input_ids, **gen_kwargs)
    response = tokenizer.batch_decode(output_ids)[0]
    print(response)


if __name__ == "__main__":
    parser = TrlParser((QuarkQuantArguments, ScriptArguments, SFTConfig, ModelConfig))
    quark_args, script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(quark_args, script_args, training_args, model_args)
