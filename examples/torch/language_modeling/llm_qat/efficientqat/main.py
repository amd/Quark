#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import os
from dataclasses import dataclass, field
from pprint import pformat

import torch
import torch.nn as nn
from accelerate import Accelerator
from data_preparation import get_blockwise_tuning_dataloader
from datasets import load_dataset
from quantization_schemes import SUPPORTED_QUANT_SCHEMES, weight_only_quantize
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from transformers import TrainingArguments as HFTrainingArguments

from quark.contrib.llm_eval import ppl_eval
from quark.torch import export_safetensors
from quark.torch.utils.llm import preprocess_for_quantization


@dataclass
class QuarkTrainingArguments(HFTrainingArguments):
    model: str | None = field(default="THUDM/chatglm3-6b", metadata={"help": "Specify where the HuggingFace model is."})
    model_trust_remote_code: bool = field(default=False)
    do_ptq: bool = field(
        default=False, metadata={"help": "Whether to run post-training quantization before further stages."}
    )
    quant_scheme: str = field(
        default="w_uint4_asym",
        metadata={
            "help": (
                "Supported quant_scheme in the script."
                "If there is no suitable quantization strategy among the options,"
                "users can customize the quantization configuration according to their own needs."
                "If None, the model will be quantized by float16"
            ),
            "choices": SUPPORTED_QUANT_SCHEMES,
        },
    )
    group_size: int = field(default=128, metadata={"help": "Group size for per_group quantization."})
    attn_implementation: str = field(default="eager")


@dataclass
class BlockwiseJointTuningArguments:
    do_blockwise_joint_tuning: bool = field(
        default=False,
        metadata={"help": "Whether to apply blockwise_joint_tuning (post-optimization) after QAT."},
    )
    blockwise_tuning_dataset: str = field(
        default="c4", metadata={"help": "Dataset for blockwise tuning calibration.", "choices": ["c4", "redpajama"]}
    )
    blockwise_tuning_train_size: int = field(
        default=4096, metadata={"help": "Number of calibration samples used for blockwise tuning train dataloader."}
    )
    blockwise_tuning_epochs: int = field(default=2, metadata={"help": "Number of epochs for blockwise tuning."})
    blockwise_weight_lr: float = field(default=1e-5, metadata={"help": "Weight learning rate for blockwise tuning."})
    blockwise_quant_lr: float = field(
        default=1e-4, metadata={"help": "Quantization parameter learning rate for blockwise tuning."}
    )


@dataclass
class EvalArguments:
    do_evaluation: bool = field(default=False)
    max_eval_samples: int | None = field(
        default=None, metadata={"help": "It will sample max_eval_samples from eval dataset"}
    )


@dataclass
class ExportArguments:
    model_export: str | None = field(
        default=None, metadata={"help": "Model export format.", "choices": ["onnx", "hf_format", "gguf", None]}
    )
    model_export_dir: str = field(default="exported_model")
    export_weight_format: str = field(
        default="real_quantized",
        metadata={
            "help": "Whether to export weights compressed or uncompressed",
            "choices": ["fake_quantized", "real_quantized"],
        },
    )


def print_training_args(args: QuarkTrainingArguments, exclude_defaults: bool = True) -> None:
    args_dict = vars(args).copy()
    if exclude_defaults:
        default_args = QuarkTrainingArguments("tmp_output_dir")
        default_dict = vars(default_args)
        args_dict = {k: v for k, v in args_dict.items() if str(v) != str(default_dict.get(k, None))}

    banner = "=" * 40 + " Arguments " + "=" * 40
    formatted = pformat(args_dict, indent=2, width=120, sort_dicts=False)
    accelerator.print(banner)
    accelerator.print(formatted)


def print_runtime_args_snapshot(
    training_args: QuarkTrainingArguments,
    blockwise_args: BlockwiseJointTuningArguments,
    eval_args: EvalArguments,
    export_args: ExportArguments,
) -> None:
    sections = {
        "training_args": vars(training_args),
        "blockwise_args": vars(blockwise_args),
        "eval_args": vars(eval_args),
        "export_args": vars(export_args),
    }
    accelerator.print("\n[QUARK-INFO]: Runtime arguments snapshot...")
    for name, values in sections.items():
        msg = "\n".join([f"{k:<26}: {v}" for k, v in values.items()])
        accelerator.print(f"\n[{name}]\n{msg}")


def load_original_model(training_args: QuarkTrainingArguments) -> tuple[nn.Module, AutoTokenizer]:
    accelerator.print("\n[QUARK-INFO]: Loading Model and Tokenizer... ")
    model = AutoModelForCausalLM.from_pretrained(
        training_args.model,
        torch_dtype="auto",
        trust_remote_code=training_args.model_trust_remote_code,
        attn_implementation=training_args.attn_implementation,
        device_map="auto",
    )

    tokenizer_kwargs = {"trust_remote_code": training_args.model_trust_remote_code, "use_fast": False}
    try:
        tokenizer = AutoTokenizer.from_pretrained(training_args.model, legacy=False, **tokenizer_kwargs)
    except TypeError:
        tokenizer = AutoTokenizer.from_pretrained(training_args.model, **tokenizer_kwargs)

    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id
    if "sop" in tokenizer.get_added_vocab() and not tokenizer.bos_token:
        tokenizer.add_special_tokens({"bos_token": "sop"})
        embeddings = model.get_input_embeddings()
        if len(tokenizer) > embeddings.weight.shape[0]:
            model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def run(
    training_args: QuarkTrainingArguments,
    blockwise_args: BlockwiseJointTuningArguments,
    eval_args: EvalArguments,
    export_args: ExportArguments,
) -> None:
    print_runtime_args_snapshot(training_args, blockwise_args, eval_args, export_args)

    model, tokenizer = load_original_model(training_args)

    if training_args.do_ptq:
        accelerator.print("\n[QUARK-INFO]: Running post-training quantization... ")
        preprocess_for_quantization(model)
        model = weight_only_quantize(model, training_args.quant_scheme, training_args.group_size)

    if blockwise_args.do_blockwise_joint_tuning:
        accelerator.print("\n[QUARK-INFO]: Blockwise joint tuning... ")

        ref_model, _ = load_original_model(training_args)
        ref_model.eval()

        blockwise_tuning_dataloader = get_blockwise_tuning_dataloader(
            name=blockwise_args.blockwise_tuning_dataset,
            tokenizer=tokenizer,
            train_size=blockwise_args.blockwise_tuning_train_size,
        )

        blockwise_cfg_dict = {
            "name": "blockwise_joint_tuning",
            "epochs": blockwise_args.blockwise_tuning_epochs,
            "weight_lr": blockwise_args.blockwise_weight_lr,
            "qparam_lr": blockwise_args.blockwise_quant_lr,
            "weight_decay": 0.0,
            "qparam_weight_decay": 0.0,
            "min_lr_factor": 20.0,
            "max_grad_norm": 0.3,
            "model_decoder_layers": "model.layers",
            "trainable_modules": [],
            "quant_trainable_modules": [],
        }

        from quark.torch.algorithm.api import blockwise_tuning_algo
        from quark.torch.algorithm.config import BlockwiseJointTuningConfig

        blockwise_cfg = BlockwiseJointTuningConfig.from_dict(blockwise_cfg_dict)
        accelerator.print(f"Blockwise joint tuning config: {blockwise_cfg}")

        is_accelerate = hasattr(model, "hf_device_map")
        model = blockwise_tuning_algo(ref_model, model, blockwise_cfg, is_accelerate, blockwise_tuning_dataloader)
        accelerator.wait_for_everyone()

    if export_args.model_export is not None:
        if accelerator.is_main_process:
            os.makedirs(export_args.model_export_dir, exist_ok=True)
            with torch.no_grad():
                export_safetensors(
                    model=model,
                    output_dir=export_args.model_export_dir,
                    custom_mode="quark",
                    weight_format=export_args.export_weight_format,
                    pack_method="reorder",
                )
                tokenizer.save_pretrained(export_args.model_export_dir)
        accelerator.wait_for_everyone()

    if eval_args.do_evaluation:
        model.eval()
        model.to("cuda")
        dtype = training_args.quant_scheme if training_args.do_ptq else str(next(model.parameters()).dtype)
        accelerator.print(f"\n[QUARK-INFO]: Evaluating ({dtype})... ")
        if accelerator.is_main_process:
            accelerator.print("\n[QUARK-INFO]: Evaluating PPL (wikitext-2)... ")
            testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
            test_text = (
                testdata["text"][: eval_args.max_eval_samples]
                if eval_args.max_eval_samples is not None
                else testdata["text"]
            )
            testenc = tokenizer("\n\n".join(test_text), return_tensors="pt")
            ppl = ppl_eval(model, testenc, next(model.parameters()).device)
            accelerator.print(f"\n[QUARK-INFO]: PPL (wikitext-2): {ppl}")
        accelerator.wait_for_everyone()


if __name__ == "__main__":
    trainer_parser = HfArgumentParser(
        (QuarkTrainingArguments, BlockwiseJointTuningArguments, EvalArguments, ExportArguments)
    )
    accelerator = Accelerator()
    training_args, blockwise_args, eval_args, export_args = trainer_parser.parse_args_into_dataclasses()
    accelerator.print("\n" + "\n".join([f"{k:<26}: {v}" for k, v in vars(eval_args).items()]))
    print_training_args(training_args)
    run(training_args, blockwise_args, eval_args, export_args)
