#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
import sys

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
from transformers import AutoModelForCausalLM, Mxfp4Config
from trl import SFTTrainer

from quark.common.utils.import_utils import is_accelerate_available
from quark.torch.quantization.config.config import QLayerConfig
from quark.torch.utils import setattr_recursive

if is_accelerate_available():
    from accelerate.hooks import add_hook_to_module
from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors
from quark.torch.quantization.nn.modules.quantize_linear import QLoRaQuantLinear, QuantLinear

# TODO: Using sys.path.append is bad practice.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from datasets import load_dataset  # load_from_disk
from trl import ModelConfig, ScriptArguments, SFTConfig, TrlParser
from utils import QuarkQuantArguments

from quark.contrib.llm_eval import eval_model
from quark.torch.utils.llm import (
    get_calib_dataloader,
    preprocess_for_quantization,
)


# NOTE these 4 funcs are used to modify the quantized model to QLoRA trainable model
def mark_only_lora_layer_as_trainable(model: nn.Module) -> None:
    """
    make the lora_a & lora_b trainable during training, that can save memory.
    """
    for n, p in model.named_parameters():
        p.requires_grad = False

    for _, module in model.named_modules():
        if isinstance(module, QLoRaQuantLinear):
            if isinstance(module.lora_A, nn.Linear):
                module.lora_A.weight.requires_grad = True
            if isinstance(module.lora_B, nn.Linear):
                module.lora_B.weight.requires_grad = True


def disable_adapters(model: nn.Module, disable: bool = True) -> None:
    """
    disable the adapter, that adapter will not take effect.
        output = quant(in) * quant(w)
    """
    for _, module in model.named_modules():
        if isinstance(module, QLoRaQuantLinear):
            if hasattr(module, "active_adapters"):
                module.active_adapters = not disable
    return


def merge_weight(model: nn.Module) -> None:
    """
    orginal:
        output = quant(in) * quant(w) + lora_b * (lora_a * in)
    new:
        new_weight = w + lora_b * lora_a
        output = qiant(in) * quant(new_weight)
    """
    for _, module in model.named_modules():
        if isinstance(module, QLoRaQuantLinear):
            module.merge()
    return


def trans_quant_linear_2_qlora_quantLinear(model: nn.Module) -> None:
    """
    replace every QuantLinear -> QLoRaQuantLinear
    """
    named_modules = dict(model.named_modules(remove_duplicate=False))
    replace_num = 0
    for name, module in tqdm(named_modules.items()):
        module_name = module.__class__.__name__
        if not (isinstance(module, QuantLinear) and module_name == "QuantLinear"):
            continue
        # replace QuantLinear -> QLoRaQuantLinear
        bias: bool = module.bias is None
        # load quantizer & config
        empty_config = QLayerConfig()
        qlora_quanr_linear = QLoRaQuantLinear(
            module.in_features, module.out_features, module.weight.device, bias, empty_config
        )
        qlora_quanr_linear._input_qspec = module._input_qspec
        qlora_quanr_linear._output_qspec = module._output_qspec
        qlora_quanr_linear._weight_qspec = module._weight_qspec
        qlora_quanr_linear._bias_qspec = module._bias_qspec
        qlora_quanr_linear._input_quantizer = module._input_quantizer
        qlora_quanr_linear._output_quantizer = module._output_quantizer
        qlora_quanr_linear._weight_quantizer = module._weight_quantizer
        qlora_quanr_linear._bias_quantizer = module._bias_quantizer

        # reload weight
        qlora_quanr_linear.weight = module.weight
        qlora_quanr_linear.bias = module.bias
        quark_hook = module._hf_hook if hasattr(module, "_hf_hook") else None
        if quark_hook is not None:
            add_hook_to_module(qlora_quanr_linear, quark_hook)
        setattr_recursive(model, name, qlora_quanr_linear)
        replace_num += 1
    print(f"\n[INFO]: Totally replace {replace_num} QuantLinear -> QLoRaQuantLinear.")
    return


def main(quark_args, script_args, training_args, model_args):
    """
    Main function to perform model quantization, QLoRA fine-tuning, evaluation, and export.
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

    # 2. Perform Quark PTQ quant process.
    if not quark_args.skip_quantization:
        print("\n[INFO]: Loading dataset ...")
        calib_dataloader = get_calib_dataloader(
            dataset_name=quark_args.calib_dataset,
            tokenizer=tokenizer,
            batch_size=quark_args.batch_size,
            num_calib_data=quark_args.num_calib_data,
            seqlen=quark_args.seq_len,
            device=model.device,
        )
        # Load algorithm configs from files if provided
        quant_config = LLMTemplate.get("gpt_oss").get_config(scheme=quark_args.quant_scheme)
        quantizer = ModelQuantizer(quant_config, quark_args.multi_device)
        model = quantizer.quantize_model(model, calib_dataloader)

    # 3. Perform Quark Qlora fine-tune process.
    if (not quark_args.skip_qlora_train) and (not quark_args.skip_quantization):
        trans_quant_linear_2_qlora_quantLinear(model)
        mark_only_lora_layer_as_trainable(model)
        disable_adapters(model, False)

        # to save GPU memory during training.
        model.config.use_cache = False
        if model.supports_gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            print(f"Gradient Checkpointing: {model.is_gradient_checkpointing}")

        dataset = load_dataset(
            script_args.dataset_name, cache_dir=quark_args.cache_dir
        )  # or use load_from_disk("./PATH")
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset[script_args.dataset_train_split],
            eval_dataset=None,
            processing_class=tokenizer,
            peft_config=None,
        )
        trainer.train()
        merge_weight(trainer.model)
        model.config.use_cache = True  # restore use_cache -> true

    if not quark_args.skip_quantization:
        model = quantizer.freeze(model)

    torch.cuda.empty_cache()
    if not quark_args.skip_evaluation:
        print("\n[INFO]: Evaluating ...")
        quark_args.use_ppl_eval_model = True
        quark_args.model_dir = model_args.model_name_or_path
        eval_model(
            quark_args,
            model,
            model.device,
            save_metrics_to_csv=quark_args.save_metrics_to_csv,
            output_dir=quark_args.metrics_output_dir,
        )

    if quark_args.model_export == "hf_format" and (not quark_args.skip_quantization):
        print("\n[INFO]: Exporting hugging face format safetensors...")
        with torch.no_grad():
            export_safetensors(
                model=model,
                output_dir=quark_args.export_path,
                custom_mode=quark_args.custom_mode,
                weight_format=quark_args.export_weight_format,
                pack_method=quark_args.pack_method,
            )
            tokenizer.save_pretrained(quark_args.export_path)
    return 0


if __name__ == "__main__":
    parser = TrlParser((QuarkQuantArguments, ScriptArguments, SFTConfig, ModelConfig))
    quark_args, script_args, training_args, model_args, _ = parser.parse_args_and_config(return_remaining_strings=True)
    main(quark_args, script_args, training_args, model_args)
