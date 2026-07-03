#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""
Unified entry: run Quark PTQ, then select a fine-tuning recipe via ``--training_mode``.

Pipeline (all modes):
    1. Load the base Hugging Face model.
    2. Build tokenizer and ``AutoProcessor`` when needed (multimodal checkpoints).
    3. Build calibration dataloader and call ``preprocess_for_quantization`` (see ``llm_ptq/quantize_quark.py`` order).
    4. Build ``ModelQuantizer`` and run ``quantize_model`` to obtain a PTQ student.
    5. Enable QAD (``QADTrainer`` + teacher) or QAT (``Trainer``), with or without QLoRA + merge.

``--training_mode`` (after PTQ):
    * ``qad`` — train ``QuantLinear`` with ``QADTrainer`` and a full-precision teacher.
    * ``qad_qlora`` — QLoRA layers, ``QADTrainer``, merge adapters after training.
    * ``qat`` — train ``QuantLinear`` with ``Trainer`` and label loss (no teacher).
    * ``qat_qlora`` — QLoRA + ``Trainer``, merge adapters after training.
"""

from pathlib import Path

import torch
import transformers
from transformers import AutoProcessor, Trainer
from utils import (  # type: ignore
    DataArguments,
    FineTuneArguments,
    PTQQuantArguments,
    TrainingArguments,
    disable_adapters,
    freeze_all_parameters,
    make_supervised_data_module,
    mark_only_qlora_adapter_as_trainable,
    mark_only_quant_linear_as_trainable,
    merge_qlora_adapters_into_weights,
    replace_quant_linears_with_qlora,
)

from quark.common.utils.log import ScreenLogger
from quark.contrib.llm_eval import eval_model
from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors
from quark.torch.algorithm.qad_trainer import QADTrainer
from quark.torch.utils.llm import get_calib_dataloader, get_model, get_tokenizer, preprocess_for_quantization

logger = ScreenLogger(__name__)

# Architectures in this example that use ``AutoProcessor`` in addition to ``tokenizer`` for PTQ.
_MULTIMODAL_TYPES = frozenset({"mllama", "llama4", "gemma3", "qwen3_vl_moe", "deepseek_vl_v2"})


def fine_tune_process() -> None:
    """Parse CLI args, run PTQ, optional SFT, evaluation, and optional export."""
    parser = transformers.HfArgumentParser((FineTuneArguments, DataArguments, TrainingArguments, PTQQuantArguments))
    finetune_args, data_args, training_args, quant_args = parser.parse_args_into_dataclasses()
    if not quant_args.model_dir:
        raise ValueError("--model_dir is required: pass a local path or Hugging Face model id.")
    ft_mode = finetune_args.training_mode

    logger.info(
        "training_mode=%s (PTQ runs first; this flag only selects the post-quantization recipe).",
        ft_mode,
    )

    device_string = quant_args.device
    logger.info("Loading model from %s", quant_args.model_dir)
    model, _ = get_model(
        quant_args.model_dir,
        quant_args.data_type,
        device_string,
        quant_args.multi_gpu,
        quant_args.multi_device,
        quant_args.model_attn_implementation,
        trust_remote_code=quant_args.trust_remote_code,
    )

    model_type = getattr(model.config, "model_type", None)
    if model_type is None:
        model_type = model.config.architectures[0]
    tokenizer = get_tokenizer(
        quant_args.model_dir,
        max_seq_len=quant_args.seq_len,
        model_type=model_type,
        trust_remote_code=quant_args.trust_remote_code,
    )
    is_multimodal = model_type in _MULTIMODAL_TYPES
    processor = None
    if is_multimodal:
        processor = AutoProcessor.from_pretrained(quant_args.model_dir)  # type: ignore
        if quant_args.model_export is not None:
            out_dir = Path(quant_args.quant_out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            processor.save_pretrained(quant_args.quant_out_dir)

    logger.info("Loading calibration dataset for PTQ")
    main_device = model.device if quant_args.multi_gpu or quant_args.multi_device else device_string
    calibration_dataloader = get_calib_dataloader(
        dataset_name=quant_args.calibration_dataset,
        processor=processor if is_multimodal else None,
        tokenizer=tokenizer,
        batch_size=quant_args.batch_size,
        num_calib_data=quant_args.num_calibration_examples,
        seqlen=quant_args.seq_len,
        device=main_device,
    )

    preprocess_for_quantization(model)
    template = LLMTemplate.get(model_type)
    layer_config: dict = {}
    algorithm_configs: dict = {}
    quant_config = template.get_config(
        scheme=quant_args.quant_scheme,
        algorithm=quant_args.quant_algo,
        kv_cache_scheme=quant_args.kv_cache_dtype,
        min_kv_scale=quant_args.min_kv_scale,
        layer_config=layer_config,
        attention_scheme=quant_args.attention_dtype,
        exclude_layers=quant_args.exclude_layers,
        algo_configs=algorithm_configs or None,
    )
    quantizer = ModelQuantizer(quant_config, quant_args.multi_device)
    quant_args.exclude_layers = quantizer.config.exclude

    logger.info("Running ModelQuantizer.quantize_model (PTQ calibration)")
    model = quantizer.quantize_model(model, calibration_dataloader)  # type: ignore[assignment]
    logger.info("PTQ finished; configuring fine-tuning objective and optional adapters")

    use_qlora_finetune = ft_mode in ("qad_qlora", "qat_qlora")
    use_qad_trainer = ft_mode in ("qad", "qad_qlora")

    if use_qlora_finetune:
        replace_quant_linears_with_qlora(model)
        mark_only_qlora_adapter_as_trainable(model)
        disable_adapters(model, adapter_disabled=False)
    else:
        mark_only_quant_linear_as_trainable(model)

    if not training_args.skip_qlora_train:
        supervised_tokenizer = transformers.AutoTokenizer.from_pretrained(
            quant_args.model_dir, model_max_length=training_args.model_max_length
        )  # type: ignore[assignment]
        supervised_tokenizer.pad_token_id = supervised_tokenizer.eos_token_id  # type: ignore[assignment]
        data_module = make_supervised_data_module(
            dataset=data_args.train_dataset,
            tokenizer=supervised_tokenizer,
            cache_dir=data_args.cache_dir,
            train_size=data_args.train_size,
            eval_size=data_args.eval_size,
        )
        model.enable_input_require_grads()
        if model.supports_gradient_checkpointing:
            model.config.use_cache = False
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            logger.info("Gradient checkpointing enabled: %s", model.is_gradient_checkpointing)
        if use_qad_trainer:
            teacher_model = transformers.AutoModelForCausalLM.from_pretrained(
                quant_args.model_dir,
                device_map="auto",
                torch_dtype="auto",
                trust_remote_code=quant_args.trust_remote_code,
            )
            teacher_model.eval()
            teacher_model.config.use_cache = False
            freeze_all_parameters(teacher_model)
            model.teacher = teacher_model
            trainer = QADTrainer(
                model=model,
                processing_class=tokenizer,
                temperature=training_args.kd_temperature,
                args=training_args,
                **data_module,
            )
        else:
            trainer = Trainer(
                model=model,
                processing_class=tokenizer,
                args=training_args,
                **data_module,
            )
        trainer.train()
        if use_qlora_finetune:
            merge_qlora_adapters_into_weights(trainer.model)
        if getattr(model, "teacher", None) is not None:
            model.teacher = None

    model = quantizer.freeze(model)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not quant_args.skip_evaluation:
        logger.info("Running evaluation")
        quant_args.use_ppl_eval_model = True
        eval_model(
            quant_args,
            model,
            main_device,
            save_metrics_to_csv=quant_args.save_metrics_to_csv,
            output_dir=quant_args.metrics_output_dir,
            multimodal=is_multimodal,
        )
    if quant_args.export_model_output_dir:
        logger.info("export_safetensors (Hugging Face layout)")
        model.config.use_cache = True
        with torch.no_grad():
            export_safetensors(
                model=model,
                output_dir=quant_args.export_model_output_dir,
                custom_mode="quark",
                weight_format="real_quantized",
                pack_method="reorder",
            )
            export_tokenizer = transformers.AutoTokenizer.from_pretrained(quant_args.model_dir)
            export_tokenizer.save_pretrained(quant_args.export_model_output_dir)
        logger.info("Export finished")
    logger.info("Finished")


if __name__ == "__main__":
    fine_tune_process()
