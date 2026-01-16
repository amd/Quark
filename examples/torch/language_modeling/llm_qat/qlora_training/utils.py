#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import datasets
import torch
import transformers
from transformers import default_data_collator

from quark.torch import LLMTemplate

IGNORE_INDEX = -100


@dataclass
class DataArguments:
    """
    DataArguments: finetune dataset param.
    """

    train_dataset: str = field(
        default="Daring-Anteater",
        metadata={"help": "Specify the dataset.", "choices": ["Daring-Anteater"]},
    )
    train_size: int = field(
        default=100,
        metadata={"help": "Number of training samples to use."},
    )
    eval_size: int = field(
        default=10,
        metadata={"help": "Number of evaluation samples to use."},
    )
    cache_dir: str | None = field(
        default=None,
        metadata={"help": "Specify the dataset cache path."},
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """
    TrainingArguments: args for training the model.
    """

    # inherent from org class
    skip_qlora_train: bool = field(default=False, metadata={"help": "Whether to skip training."})

    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})
    output_dir: str | None = field(
        default="./out_test1",
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written. Defaults to 'trainer_output' if not provided."
        },
    )
    num_train_epochs: float = field(default=1.0, metadata={"help": "Total number of training epochs to perform."})
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per device accelerator core/CPU for training."}
    )
    per_device_eval_batch_size: int = field(
        default=1, metadata={"help": "Batch size per device accelerator core/CPU for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )
    eval_accumulation_steps: int | None = field(
        default=1,
        metadata={"help": "Number of predictions steps to accumulate before moving the tensors to the CPU."},
    )
    save_steps: float = field(
        default=100,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "The evaluation strategy to use."},
    )
    eval_steps: float | None = field(
        default=50,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    load_best_model_at_end: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether or not to load the best model found during training at the end of training. When this option"
                " is enabled, the best checkpoint will always be saved. See `save_total_limit` for more."
            )
        },
    )
    save_total_limit: int | None = field(
        default=2,
    )
    learning_rate: float = field(default=1e-4, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    warmup_ratio: float = field(
        default=0.1, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    logging_steps: float = field(
        default=1,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    model_max_length: int = field(
        default=4096,
        metadata={"help": ("Maximum s")},
    )
    dataloader_drop_last: bool = field(default=True)
    bf16: bool = field(default=True)


@dataclass
class PTQQuantArguments:
    """
    PTQQuantArguments: args to perfrom Quark PTQ.
    """

    model_dir: str = field(default="{PATH}/meta-llama/Llama-3.2-1B-Instruct")
    device: str = field(
        default="cuda", metadata={"help": "Device for running the quantizer", "choices": ["cuda", "cpu"]}
    )
    multi_gpu: bool = field(default=True)
    multi_device: bool = field(
        default=False,
        metadata={
            "help": (
                "we allow you to use this mode to run a model quantization that exceeds the size of your gpu memory if you use args.multi_gpu and still run into OOM "
                "now it only supports thr common quantization without algorithms, please note that this can lead to very slow quantization."
            )
        },
    )
    model_attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "The attention implementation to use in the model",
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    calib_dataset: str = field(
        default="pileval",
        metadata={
            "help": "Dataset for calibration",
            "choices": [
                "pileval",
                "wikitext",
                "cnn_dailymail",
                "pileval_for_awq_benchmark",
                "wikitext_for_gptq_benchmark",
                "HuggingFaceH4/ultrachat_200k",
                "ScienceQA",
            ],
        },
    )
    data_type: str = field(
        default="auto",
        metadata={"help": "Datatype of the model", "choices": ["auto", "float16", "bfloat16", "float32"]},
    )
    seq_len: int = field(default=512, metadata={"help": "Sequence length of data"})
    batch_size: int = field(default=1, metadata={"help": "Batch size for calibration."})
    num_calib_data: int = field(default=128, metadata={"help": "Number of samples for calibration."})
    group_size: int = field(default=128, metadata={"help": "Group size for per_group quantization."})
    quant_scheme: str = field(
        default="mxfp4",
        metadata={
            "help": "Supported quant_scheme in the script. If there is no suitable quantization strategy among the options, users can customize the quantization configuration according to their own needs.",
            "choices": LLMTemplate.get_supported_schemes(),
        },
    )
    kv_cache_dtype: str | None = field(default="fp8", metadata={"help": "KV Cache dtype.", "choices": ["fp8", None]})
    min_kv_scale: float = field(default=0.0, metadata={"help": "Minimum value of KV Cache scale."})
    attention_dtype: str | None = field(
        default=None, metadata={"help": "The dtype of attention quantization.", "choices": ["fp8", None]}
    )
    quant_algo: str | None = field(
        default=None,
        metadata={
            "help": "Algorithms used for quantization.",
            "choices": ["awq", "gptq", "smoothquant", "rotation", None],
        },
    )
    exclude_layers: str | None = field(
        default=None, metadata={"help": "List of layers to exclude from quantization. Default depends on model type."}
    )
    model_export: str | None = field(
        default=None,
        metadata={
            "help": "Model export format",
            "choices": [
                None,
                "hf_format",
            ],
        },
    )
    custom_mode: str = field(
        default="quark", metadata={"help": "Model export format", "choices": ["quark", "awq", "fp8"]}
    )
    pack_method: str = field(
        default="reorder", metadata={"help": "Pack method for awq_export.", "choices": ["order", "reorder"]}
    )
    quant_out_dir: str = field(default="exported_model", metadata={"help": "Path for quantized model."})
    skip_evaluation: bool = field(default=False, metadata={"help": "Whether skip evaluation after quantization."})
    use_ppl_eval_model: bool = field(default=False)
    save_metrics_to_csv: bool = field(default=False)
    metrics_output_dir: str = field(default="metrics_output_dir")
    use_ppl_eval_for_kv_cache: bool = field(default=False)
    ppl_eval_for_kv_cache_context_size: int = field(
        default=1024, metadata={"help": "Context size used in PPL evaluation for KV cache."}
    )
    ppl_eval_for_kv_cache_sample_size: int = field(
        default=512, metadata={"help": "Sample size used in PPL evaluation for KV cache."}
    )
    ppl_eval_for_kv_cache_patch_size: int | None = field(
        default=None, metadata={"help": "Patch size used in PPL evaluation for KV cache."}
    )
    eval_batch_size: int = field(default=1, metadata={"help": "Batch size used for evaluation."})
    max_eval_batch_size: int = field(
        default=64, metadata={"help": "Maximal batch size to try with `--batch_size auto`."}
    )
    num_eval_data: int = field(
        default=-1,
        metadata={
            "help": "Number of samples for evaluation. The default value is -1, which means the entire dataset is used for evaluation."
        },
    )
    num_fewshot: int | None = field(default=None, metadata={"help": "Number of examples in few-shot context."})  # NOTE
    apply_chat_template: bool = field(
        default=False,
        metadata={
            "help": "Providing `--apply_chat_template` without an argument will apply the default chat template to the prompt."
        },
    )
    use_mlperf_rouge: bool = field(default=False)
    eval_data_dir: str | None = field(default=None, metadata={"help": "Dataset for evaluation."})  # NOTE
    use_tp: bool = field(
        default=False, metadata={"help": "Enable tensor parallelism exclusively for model evaluation."}
    )
    trust_remote_code: bool = field(default=True)
    tasks: int | None = field(default=None)
    group_size_per_layer: int | None = field(default=None)


@contextmanager
def main_process_first():  # type: ignore
    """Context manager to run code on the main process first."""
    if not torch.distributed.is_initialized():
        yield
        return
    rank = torch.distributed.get_rank()
    if rank == 0:
        yield
        torch.distributed.barrier()
    else:
        torch.distributed.barrier()
        yield
    torch.distributed.barrier()


def get_daring_anteater(
    tokenizer: transformers.AutoTokenizer,
    cache_dir: str | None = None,
    split: str = "train",
    max_length: int = 4096,
    train_size: int = 0,
    eval_size: int = 0,
) -> Any:
    """prepare the training data."""

    def process_and_tokenize(sample) -> dict:  # type: ignore
        conversations = sample["conversations"]
        all_input_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []  # type: ignore
        all_labels = [IGNORE_INDEX] if tokenizer.bos_token_id else []  # type: ignore

        for conversation in conversations:
            role = conversation["from"]
            input_ids = tokenizer.encode(conversation["value"] + "\n", add_special_tokens=False)  # type: ignore
            labels = input_ids if role == "Assistant" else [IGNORE_INDEX] * len(input_ids)

            all_input_ids.extend(input_ids)
            all_labels.extend(labels)

            if len(all_input_ids) > max_length:
                break

        all_input_ids.append(tokenizer.eos_token_id)  # type: ignore
        all_labels.append(IGNORE_INDEX)
        all_attention_mask = [1] * len(all_input_ids)

        cur_seq_length = len(all_input_ids)
        if cur_seq_length < max_length:
            pad_token = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id  # type: ignore
            all_input_ids += [pad_token] * (max_length - cur_seq_length)
            all_attention_mask += [0] * (max_length - cur_seq_length)
            all_labels += [IGNORE_INDEX] * (max_length - cur_seq_length)

        return {
            "input_ids": all_input_ids[:max_length],
            "attention_mask": all_attention_mask[:max_length],
            "labels": all_labels[:max_length],
        }

    if hasattr(get_daring_anteater, "cached_dataset"):
        dataset = get_daring_anteater.cached_dataset
    else:
        with main_process_first():
            dataset = datasets.load_dataset(
                "nvidia/Daring-Anteater",
                split="train",
                cache_dir=cache_dir,
            )
            # Shuffle and subsample the dataset
            eval_size = 2000 if eval_size == 0 else eval_size
            train_size = len(dataset) - eval_size if train_size == 0 else train_size
            assert train_size + eval_size <= len(dataset) and train_size > 0 and eval_size > 0, (
                "not enough data for train-eval split"
            )
            dataset = dataset.shuffle(seed=42).select(range(train_size + eval_size))
            dataset = dataset.map(process_and_tokenize, remove_columns=list(dataset.features))
            dataset = dataset.train_test_split(test_size=eval_size, shuffle=True, seed=42)
        get_daring_anteater.cached_dataset = dataset  # type: ignore
    return dataset[split]


def make_supervised_data_module(
    dataset: str = "Daring-Anteater",
    tokenizer: transformers.PreTrainedTokenizer = None,  # type: ignore
    cache_dir: str | None = None,
    train_size: int = 0,
    eval_size: int = 0,
) -> dict:  # type: ignore
    """Make dataset and collmtor for supervised fine-tuning."""
    if dataset == "Daring-Anteater":
        train_dataset = get_daring_anteater(
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            split="train",
            max_length=tokenizer.model_max_length,
            train_size=train_size,
            eval_size=eval_size,
        )
        val_dataset = get_daring_anteater(
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            split="test",
            max_length=tokenizer.model_max_length,
            train_size=train_size,
            eval_size=eval_size,
        )
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    return {
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": default_data_collator,
    }
