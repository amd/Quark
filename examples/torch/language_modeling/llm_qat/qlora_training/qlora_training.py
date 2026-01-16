#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
from transformers import AutoProcessor, Trainer

from quark.shares.utils.import_utils import is_accelerate_available
from quark.torch.quantization.config.config import QLayerConfig
from quark.torch.quantization.nn.modules.quantize_linear import QLoRaQuantLinear, QuantLinear
from quark.torch.utils import setattr_recursive

if is_accelerate_available():
    from accelerate.hooks import add_hook_to_module
from utils import (  # type: ignore
    DataArguments,
    PTQQuantArguments,
    TrainingArguments,
    make_supervised_data_module,
)

from quark.torch import LLMTemplate, ModelQuantizer

# TODO: Using sys.path.append is bad practice.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from quark.contrib.llm_eval import eval_model
from quark.torch.utils.llm import get_calib_dataloader, get_model, get_tokenizer, prepare_for_moe_quant


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


def qlora_training() -> None:
    # Step 1. prepare args (training, PTQ, training dataset)
    # NOTE:
    #   1.1 PTQQuantArguments:
    #      args about quark's PTQ, more info refer /examples/torch/language_modeling/llm_ptq/quantize_quark.py
    #   1.2 DataArguments: args about QLoRA training data
    #   1.3 TrainingArguments: inherit from transformers.TrainingArguments, args about training.
    #       User can refer transformers.TrainingArguments and adjust for better training results.
    parser = transformers.HfArgumentParser((DataArguments, TrainingArguments, PTQQuantArguments))
    data_args, training_args, quant_args = parser.parse_args_into_dataclasses()

    # Step 2. Load LLM model and perfrom modification (if necessary).
    print("\n[INFO]: Loading model ...")
    device = quant_args.device
    model, model_dtype = get_model(
        quant_args.model_dir,
        quant_args.data_type,
        device,
        quant_args.multi_gpu,
        quant_args.multi_device,
        quant_args.model_attn_implementation,
        trust_remote_code=quant_args.trust_remote_code,
    )

    prepare_for_moe_quant(model)

    # model_type = get_model_type(model)
    model_type = model.config.model_type if hasattr(model.config, "model_type") else model.config.architectures[0]
    tokenizer = get_tokenizer(
        quant_args.model_dir,
        max_seq_len=quant_args.seq_len,
        model_type=model_type,
        trust_remote_code=quant_args.trust_remote_code,
    )

    multimodal = True if model_type in ["mllama", "llama4", "gemma3_mllm"] else False
    if multimodal:
        processor = AutoProcessor.from_pretrained(quant_args.model_dir)  # type: ignore
        if quant_args.model_export is not None:
            export_dir = Path(quant_args.quant_out_dir)
            export_dir.mkdir(parents=True, exist_ok=True)
            processor.save_pretrained(quant_args.quant_out_dir)

    # Step 3. Define calibration dataloader.
    print("\n[INFO]: Loading dataset ...")
    # When the model is small, accelerate will place it on the last device
    main_device = model.device if quant_args.multi_gpu or quant_args.multi_device else quant_args.device
    calib_dataloader = get_calib_dataloader(
        dataset_name=quant_args.calib_dataset,
        processor=processor if multimodal else None,
        tokenizer=tokenizer,
        batch_size=quant_args.batch_size,
        num_calib_data=quant_args.num_calib_data,
        seqlen=quant_args.seq_len,
        device=main_device,
    )

    # Step 4. Perfrom Quark PTQ.
    # NOTE this part of code from quantize_quark.py

    model_config_type = (
        model.config.model_type if hasattr(model.config, "model_type") else model.config.architectures[0]
    )
    template = LLMTemplate.get(model_config_type)
    # add layer_quant_config
    layer_config, algo_configs = {}, {}  # NOTE for this example, skip layer_config & algo_configs

    quant_config = template.get_config(
        scheme=quant_args.quant_scheme,
        algorithm=quant_args.quant_algo,
        kv_cache_scheme=quant_args.kv_cache_dtype,
        min_kv_scale=quant_args.min_kv_scale,
        layer_config=layer_config,
        attention_scheme=quant_args.attention_dtype,
        exclude_layers=quant_args.exclude_layers,
        algo_configs=algo_configs if algo_configs else None,
    )

    quantizer = ModelQuantizer(quant_config, quant_args.multi_device)
    model = quantizer.quantize_model(model, calib_dataloader)  # type: ignore
    quant_args.exclude_layers = quantizer.config.exclude

    # Step 5: Perform QLoRA training
    # 5.1: prepare the QLoRA model and make only lora's adapter trainable
    # replace QuantLinear -> QLoRaQuantLinear
    trans_quant_linear_2_qlora_quantLinear(model)
    mark_only_lora_layer_as_trainable(model)
    disable_adapters(model, False)
    if not training_args.skip_qlora_train:
        # 5.2: prepare trainer and training dataset
        train_tokenizer = transformers.AutoTokenizer.from_pretrained(
            quant_args.model_dir, model_max_length=training_args.model_max_length
        )  # type: ignore
        tokenizer.pad_token_id = tokenizer.eos_token_id  # type: ignore

        data_module = make_supervised_data_module(
            dataset=data_args.train_dataset,
            tokenizer=train_tokenizer,
            cache_dir=data_args.cache_dir,
            train_size=data_args.train_size,
            eval_size=data_args.eval_size,
        )

        # to save GPU memory during training.
        model.enable_input_require_grads()
        if model.supports_gradient_checkpointing:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            print(f"Gradient Checkpointing: {model.is_gradient_checkpointing}")

        trainer = Trainer(
            model=model,
            processing_class=tokenizer,
            args=training_args,
            **data_module,
        )
        trainer.train()
        # let lora_A * lora_B merge to linear's weight
        merge_weight(trainer.model)

    # After quantization, models are frozen - moving from soft weights that are quantized on the fly to e.g. `QuantLinear.weight` actually holding the fake quantized weights.
    model = quantizer.freeze(model)
    torch.cuda.empty_cache()
    if not quant_args.skip_evaluation:
        print("\n[INFO]: Evaluating ...")
        quant_args.use_ppl_eval_model = True
        eval_model(
            quant_args,
            model,
            main_device,
            save_metrics_to_csv=quant_args.save_metrics_to_csv,
            output_dir=quant_args.metrics_output_dir,
            multimodal=multimodal,
        )
    print("finished")
    return


if __name__ == "__main__":
    qlora_training()
