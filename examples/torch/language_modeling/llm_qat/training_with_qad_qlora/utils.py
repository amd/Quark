#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Shared helpers and HuggingFace-style argument dataclasses for QAD / QAT + QLoRA fine-tuning examples."""

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import transformers
from tqdm import tqdm
from transformers import default_data_collator

from quark.common.utils.import_utils import is_accelerate_available, is_datasets_available
from quark.common.utils.log import ScreenLogger
from quark.torch import LLMTemplate
from quark.torch.quantization.config.config import QLayerConfig
from quark.torch.quantization.nn.modules.quantize_linear import QLoRaQuantLinear, QuantLinear
from quark.torch.utils import setattr_recursive

logger = ScreenLogger(__name__)

if is_accelerate_available():
    from accelerate.hooks import add_hook_to_module

if is_datasets_available():
    import datasets

IGNORE_INDEX = -100


def freeze_all_parameters(model: nn.Module) -> None:
    """Set ``requires_grad=False`` on every parameter in ``model``."""
    for _parameter_name, parameter in model.named_parameters():
        parameter.requires_grad = False


def mark_only_quant_linear_as_trainable(model: nn.Module) -> None:
    """Freeze every parameter except ``QuantLinear.weight`` (QAT on quantized linears)."""
    freeze_all_parameters(model)
    for _module_name, module in model.named_modules():
        if isinstance(module, QuantLinear):
            # Intentionally leave ``bias`` frozen: QAT updates only the quantized weight tensor.
            module.weight.requires_grad = True


def mark_only_qlora_adapter_as_trainable(model: nn.Module) -> None:
    """Freeze all parameters, then unfreeze only QLoRA ``lora_A`` / ``lora_B`` weights."""
    freeze_all_parameters(model)
    for _module_name, module in model.named_modules():
        if isinstance(module, QLoRaQuantLinear):
            if isinstance(module.lora_A, nn.Linear):
                module.lora_A.weight.requires_grad = True
            if isinstance(module.lora_B, nn.Linear):
                module.lora_B.weight.requires_grad = True


def disable_adapters(model: nn.Module, adapter_disabled: bool = True) -> None:
    """
    Control ``QLoRaQuantLinear.active_adapters``. If ``adapter_disabled`` is True, LoRA is turned off
    and the block matches fake-quant linear with no adapter residual.
    """
    for _module_name, module in model.named_modules():
        if isinstance(module, QLoRaQuantLinear) and hasattr(module, "active_adapters"):
            module.active_adapters = not adapter_disabled


def merge_qlora_adapters_into_weights(model: nn.Module) -> None:
    """
    Fold trained LoRA deltas into the quantized weight for each :class:`~QLoRaQuantLinear`
    (forward path becomes ``quant(x) @ quant(w_eff)`` without a separate LoRA side path).
    """
    for _module_name, module in model.named_modules():
        if isinstance(module, QLoRaQuantLinear):
            module.merge()


def replace_quant_linears_with_qlora(model: nn.Module) -> None:
    """
    In-place: swap each :class:`QuantLinear` for :class:`QLoRaQuantLinear` with the same
    quantizer state, weights, and accelerate hooks.
    """
    named_modules = dict(model.named_modules(remove_duplicate=False))
    num_replaced = 0
    for module_name, module in tqdm(named_modules.items(), desc="QuantLinear to QLoRaQuantLinear"):
        if type(module) is not QuantLinear:
            continue
        # Keeps the historical ``QuantLinear`` → QLoRA conversion signature from upstream examples.
        bias: bool = module.bias is not None
        empty_config = QLayerConfig()
        qlora_linear = QLoRaQuantLinear(
            module.in_features, module.out_features, module.weight.device, bias, empty_config
        )
        qlora_linear._input_qspec = module._input_qspec
        qlora_linear._output_qspec = module._output_qspec
        qlora_linear._weight_qspec = module._weight_qspec
        qlora_linear._bias_qspec = module._bias_qspec
        qlora_linear._input_quantizer = module._input_quantizer
        qlora_linear._output_quantizer = module._output_quantizer
        qlora_linear._weight_quantizer = module._weight_quantizer
        qlora_linear._bias_quantizer = module._bias_quantizer
        qlora_linear.weight = module.weight
        qlora_linear.bias = module.bias
        quark_hook = module._hf_hook if hasattr(module, "_hf_hook") else None
        if quark_hook is not None:
            add_hook_to_module(qlora_linear, quark_hook)
        setattr_recursive(model, module_name, qlora_linear)
        num_replaced += 1
    logger.info("Replaced %s QuantLinear layer(s) with QLoRaQuantLinear.", num_replaced)


# --- CLI dataclasses (HfArgumentParser) ---


@dataclass
class FineTuneArguments:
    """
    After shared Quark PTQ (calibration dataloader → ``ModelQuantizer``), select one of four
    post-quantization fine-tuning recipes.
    """

    training_mode: str = field(
        default="qad_qlora",
        metadata={
            "help": (
                "All modes first run PTQ; this only selects distillation vs CE loss and full linear vs QLoRA. "
                "Choices: qad — QADTrainer (KL) on QuantLinear; "
                "qad_qlora — QLoRA adapters, QADTrainer (KL), merge adapters at end; "
                "qat — HuggingFace Trainer (CE) on QuantLinear, no teacher; "
                "qat_qlora — QLoRA + Trainer (CE), merge at end. "
                "Default: qad_qlora."
            ),
            "choices": ["qad", "qad_qlora", "qat", "qat_qlora"],
        },
    )


@dataclass
class DataArguments:
    """Dataset selection and sizing for supervised fine-tuning."""

    train_dataset: str = field(
        default="Daring-Anteater",
        metadata={
            "help": "Registered dataset name for supervised fine-tuning. Default: Daring-Anteater.",
        },
    )
    train_size: int = field(
        default=0,
        metadata={
            "help": (
                "Number of training samples after split. "
                "0 means use the default split logic in the data loader (reserve eval_size for validation)."
            ),
        },
    )
    eval_size: int = field(
        default=0,
        metadata={
            "help": ("Number of held-out evaluation samples. 0 means use the default split logic in the data loader."),
        },
    )
    cache_dir: str | None = field(
        default=None,
        metadata={
            "help": "Optional Hugging Face Datasets cache directory. Default: None (library default).",
        },
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    """HuggingFace ``TrainingArguments`` plus a few fields used by this example script."""

    skip_qlora_train: bool = field(
        default=False,
        metadata={
            "help": (
                "If True, skip the entire supervised fine-tuning step (QAD, QAT, and QLoRA all respect this flag). "
                "Default: False."
            ),
        },
    )

    do_train: bool = field(default=True, metadata={"help": "Whether to run training. Default: True."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run evaluation on the dev split. Default: True."})
    output_dir: str | None = field(
        default="./qad_qlora_output",
        metadata={
            "help": "Output directory for checkpoints, logs, and trainer artifacts. Default: ./qad_qlora_output.",
        },
    )
    num_train_epochs: float = field(default=1.0, metadata={"help": "Number of training epochs. Default: 1.0."})
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Per-device batch size for training. Default: 1."}
    )
    per_device_eval_batch_size: int = field(
        default=1, metadata={"help": "Per-device batch size for evaluation. Default: 1."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Gradient accumulation steps before each optimizer update. Default: 1."},
    )
    eval_accumulation_steps: int | None = field(
        default=1,
        metadata={
            "help": "Forward passes to accumulate before reducing eval tensors (Hugging Face Trainer). Default: 1.",
        },
    )
    save_steps: float = field(
        default=100,
        metadata={
            "help": (
                "Save checkpoint every this many optimizer steps, or a fraction in [0, 1) of total training steps. "
                "Default: 100."
            ),
        },
    )
    eval_strategy: str = field(
        default="steps",
        metadata={
            "help": "Evaluation strategy: 'steps', 'epoch', or 'no'. Default: steps.",
        },
    )
    eval_steps: float | None = field(
        default=100,
        metadata={
            "help": "Run evaluation every this many steps, or a fraction in [0, 1) of total steps. Default: 100.",
        },
    )
    load_best_model_at_end: bool = field(
        default=False,
        metadata={
            "help": "If True, reload the best checkpoint by evaluation metric when training ends. Default: False.",
        },
    )
    save_total_limit: int | None = field(
        default=2,
        metadata={
            "help": "Maximum number of checkpoints to keep (older checkpoints deleted). Default: 2.",
        },
    )
    learning_rate: float = field(default=1e-4, metadata={"help": "Initial learning rate for AdamW. Default: 1e-4."})
    weight_decay: float = field(default=0.0, metadata={"help": "AdamW weight decay. Default: 0.0."})
    warmup_ratio: float = field(
        default=0.1,
        metadata={"help": "Linear learning-rate warmup as a fraction of total training steps. Default: 0.1."},
    )
    logging_steps: float = field(
        default=1,
        metadata={
            "help": "Log every this many optimizer steps, or a fraction in [0, 1) of total steps. Default: 1.",
        },
    )
    model_max_length: int = field(
        default=4096,
        metadata={"help": "Maximum token length passed to the tokenizer for supervised data. Default: 4096."},
    )
    dataloader_drop_last: bool = field(default=True)
    bf16: bool = field(
        default=True,
        metadata={"help": "Use bfloat16 autocast / weights when supported. Default: True."},
    )
    kd_temperature: float = field(
        default=1.0,
        metadata={
            "help": (
                "KL distillation temperature for QAD (passed to QADTrainer as ``temperature``). "
                "Higher values produce softer distributions. Default: 1.0."
            ),
        },
    )


@dataclass
class PTQQuantArguments:
    """Arguments for Quark PTQ (calibration, scheme, and downstream eval / export)."""

    model_dir: str | None = field(
        default=None,
        metadata={
            "help": (
                "Path to the pretrained model or Hugging Face model id (same as ``from_pretrained``). "
                "Required: pass explicitly (no default). Example: a local path or "
                "'meta-llama/Llama-3.2-1B-Instruct'."
            ),
        },
    )
    device: str | None = field(
        default="cuda",
        metadata={
            "help": (
                "Device string for PTQ tensor placement and evaluation (for example 'cuda' or 'cpu'). "
                "Required: pass explicitly (no default inside this script). Example: --device cuda."
            ),
        },
    )
    multi_gpu: bool = field(
        default=False,
        metadata={"help": "If True, use Hugging Face multi-GPU loading where applicable. Default: False."},
    )
    multi_device: bool = field(
        default=False,
        metadata={
            "help": (
                "Model-parallel or multi-device path for very large models (can be much slower; "
                "some algorithms may be unavailable). Default: False."
            ),
        },
    )
    model_attn_implementation: str = field(
        default="eager",
        metadata={
            "help": "Attention implementation passed to the model loader.",
            "choices": ["eager", "sdpa", "flash_attention_2"],
        },
    )
    calibration_dataset: str = field(
        default="pileval",
        metadata={
            "help": "Calibration dataset name for ``get_calib_dataloader``.",
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
        metadata={
            "help": "Torch dtype policy for the base model.",
            "choices": ["auto", "float16", "bfloat16", "float32"],
        },
    )
    seq_len: int = field(
        default=512,
        metadata={"help": "Sequence length for calibration batches. Default: 512."},
    )
    batch_size: int = field(
        default=1,
        metadata={"help": "Calibration batch size. Default: 1."},
    )
    num_calibration_examples: int = field(
        default=128,
        metadata={"help": "Number of calibration samples for PTQ data loading. Default: 128."},
    )
    group_size: int = field(
        default=128,
        metadata={"help": "Group size when using per-group quantization schemes. Default: 128."},
    )
    quant_scheme: str = field(
        default="mxfp4",
        metadata={
            "help": "Quantization scheme name from ``LLMTemplate``; extend templates for custom schemes.",
            "choices": LLMTemplate.get_supported_schemes(),
        },
    )
    kv_cache_dtype: str | None = field(
        default=None,
        metadata={"help": "Optional KV cache dtype (for example 'fp8').", "choices": ["fp8", None]},
    )
    min_kv_scale: float = field(
        default=0.0,
        metadata={"help": "Minimum KV cache scale when applicable. Default: 0.0."},
    )
    attention_dtype: str | None = field(
        default=None,
        metadata={"help": "Optional attention quantization dtype.", "choices": ["fp8", None]},
    )
    quant_algo: str | None = field(
        default=None,
        metadata={
            "help": "Optional post-training algorithm when the template supports it.",
            "choices": ["awq", "gptq", "smoothquant", "rotation", None],
        },
    )
    exclude_layers: str | None = field(
        default=None,
        metadata={
            "help": "Comma-separated or list-style layer name patterns to exclude from quantization (template-specific).",
        },
    )
    model_export: str | None = field(
        default=None,
        metadata={
            "help": "Optional export mode after PTQ.",
            "choices": [None, "hf_format"],
        },
    )
    custom_mode: str = field(
        default="quark",
        metadata={"help": "Export / packing backend identifier.", "choices": ["quark", "awq", "fp8"]},
    )
    pack_method: str = field(
        default="reorder",
        metadata={"help": "Packing order for some AWQ export paths.", "choices": ["order", "reorder"]},
    )
    quant_out_dir: str = field(
        default="exported_model",
        metadata={
            "help": "Directory for sidecar artifacts (for example processor) during PTQ. Default: exported_model."
        },
    )
    skip_evaluation: bool = field(
        default=False,
        metadata={
            "help": "If True, skip ``eval_model`` after training and freeze. Default: False.",
        },
    )
    use_ppl_eval_model: bool = field(
        default=False,
        metadata={
            "help": "Set by the script when running perplexity evaluation on the quantized model. Default: False.",
        },
    )
    save_metrics_to_csv: bool = field(
        default=False,
        metadata={"help": "If True, save evaluation metrics to CSV under ``metrics_output_dir``. Default: False."},
    )
    metrics_output_dir: str = field(
        default="metrics_output_dir",
        metadata={"help": "Directory for metrics CSV and related eval outputs. Default: metrics_output_dir."},
    )
    use_ppl_eval_for_kv_cache: bool = field(
        default=False,
        metadata={"help": "If True, use perplexity-style eval path for KV cache. Default: False."},
    )
    ppl_eval_for_kv_cache_context_size: int = field(
        default=1024,
        metadata={"help": "Context length for PPL / KV evaluation. Default: 1024."},
    )
    ppl_eval_for_kv_cache_sample_size: int = field(
        default=512,
        metadata={"help": "Number of positions or samples for PPL / KV evaluation. Default: 512."},
    )
    ppl_eval_for_kv_cache_patch_size: int | None = field(
        default=None,
        metadata={"help": "Optional patch size for vision-style PPL / KV evaluation. Default: None."},
    )
    eval_batch_size: int = field(
        default=1,
        metadata={"help": "Batch size for the general evaluation harness. Default: 1."},
    )
    max_eval_batch_size: int = field(
        default=64,
        metadata={"help": "Upper cap when the eval harness auto-tunes batch size. Default: 64."},
    )
    num_eval_data: int = field(
        default=-1,
        metadata={"help": "Number of evaluation samples; -1 means the full task split. Default: -1."},
    )
    num_fewshot: int | None = field(
        default=None,
        metadata={"help": "Few-shot count for task-based evaluation; None uses the task default. Default: None."},
    )
    apply_chat_template: bool = field(
        default=False,
        metadata={"help": "If True, render prompts through the model chat template before eval. Default: False."},
    )
    use_mlperf_rouge: bool = field(
        default=False,
        metadata={"help": "If True, use MLPerf ROUGE settings where applicable. Default: False."},
    )
    eval_data_dir: str | None = field(
        default=None,
        metadata={"help": "Optional override path for evaluation corpora. Default: None."},
    )
    use_tp: bool = field(
        default=False,
        metadata={"help": "If True, use tensor-parallel evaluation path where supported. Default: False."},
    )
    trust_remote_code: bool = field(
        default=True,
        metadata={"help": "Whether to trust remote code when loading configs and weights. Default: True."},
    )
    tasks: str | None = field(
        default=None,
        metadata={"help": "Optional comma-separated lm-eval task list. Default: None."},
    )
    group_size_per_layer: int | None = field(
        default=None,
        metadata={"help": "Optional per-layer group size override for some schemes. Default: None."},
    )
    evaluation_dataset: str | None = field(
        default="wikitext",
        metadata={"help": "Default evaluation dataset name for perplexity. Default: wikitext."},
    )
    export_model_output_dir: str | None = field(
        default=None,
        metadata={
            "help": "If set, run ``export_safetensors`` to this directory after training. Default: None.",
        },
    )


@contextmanager
def main_process_first() -> Any:
    """Run the wrapped code on distributed rank 0 first, then synchronize (dataset prep)."""
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
    """Build a cached Daring-Anteater train/test tokenized split; labels are assistant tokens only."""
    if not is_datasets_available():
        raise RuntimeError(
            "The datasets package is required for Daring-Anteater loading. Install datasets or use an environment "
            "that includes it."
        )

    def _process_conversation_row(conversation_row: Any) -> dict[str, Any]:
        conversations = conversation_row["conversations"]
        all_input_ids = [tokenizer.bos_token_id] if tokenizer.bos_token_id else []  # type: ignore[union-attr]
        all_labels = [IGNORE_INDEX] if tokenizer.bos_token_id else []  # type: ignore[union-attr]
        for conversation in conversations:
            role = conversation["from"]
            input_ids = tokenizer.encode(conversation["value"] + "\n", add_special_tokens=False)  # type: ignore[union-attr]
            labels = input_ids if role == "Assistant" else [IGNORE_INDEX] * len(input_ids)
            all_input_ids.extend(input_ids)
            all_labels.extend(labels)
            if len(all_input_ids) > max_length:
                break
        all_input_ids.append(tokenizer.eos_token_id)  # type: ignore[union-attr]
        all_labels.append(IGNORE_INDEX)
        all_attention_mask = [1] * len(all_input_ids)
        cur_seq_length = len(all_input_ids)
        if cur_seq_length < max_length:
            pad_tok = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id  # type: ignore[union-attr]
            all_input_ids += [pad_tok] * (max_length - cur_seq_length)
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
            eval_size = 2000 if eval_size == 0 else eval_size
            train_size = len(dataset) - eval_size if train_size == 0 else train_size
            assert train_size + eval_size <= len(dataset) and train_size > 0 and eval_size > 0, (
                "not enough data for train-eval split"
            )
            dataset = dataset.shuffle(seed=42).select(range(train_size + eval_size))
            dataset = dataset.map(_process_conversation_row, remove_columns=list(dataset.features))
            dataset = dataset.train_test_split(test_size=eval_size, shuffle=True, seed=42)
        get_daring_anteater.cached_dataset = dataset  # type: ignore[attr-defined]
    return dataset[split]


def make_supervised_data_module(
    dataset: str = "Daring-Anteater",
    tokenizer: transformers.PreTrainedTokenizer = None,  # type: ignore[assignment]
    cache_dir: str | None = None,
    train_size: int = 0,
    eval_size: int = 0,
) -> dict[str, Any]:
    """Return a dict with ``train_dataset``, ``eval_dataset``, and ``data_collator`` for ``Trainer``."""
    if dataset == "Daring-Anteater":
        train_dataset = get_daring_anteater(
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            split="train",
            max_length=tokenizer.model_max_length,
            train_size=train_size,
            eval_size=eval_size,
        )
        eval_dataset = get_daring_anteater(
            tokenizer=tokenizer,
            cache_dir=cache_dir,
            split="test",
            max_length=tokenizer.model_max_length,
            train_size=train_size,
            eval_size=eval_size,
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")
    return {
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "data_collator": default_data_collator,
    }
