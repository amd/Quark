#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import copy
import gc
import os
from pathlib import Path
from typing import Any

import datasets
import safetensors
import torch
import transformers
from datasets import Dataset
from transformers import TrainerCallback, TrainingArguments, default_data_collator
from transformers.integrations.integration_utils import TensorBoardCallback

from quark.common.utils.log import ScreenLogger
from quark.contrib.llm_eval import eval_model, ppl_eval
from quark.torch import LLMTemplate, ModelQuantizer, export_safetensors
from quark.torch.algorithm.rotation.cayley import SGDG
from quark.torch.algorithm.rotation.rotation import RotationLinear, RotationProcessor
from quark.torch.algorithm.rotation.training import AdamAndSGDGOptimizer, OrthogonalTrainingCallback, RotationTrainer
from quark.torch.quantization.config.config import load_quant_algo_config_from_file
from quark.torch.utils.llm import (
    get_calib_dataloader,
    get_model,
    get_tokenizer,
    get_wikitext2,
    preprocess_for_quantization,
)

logger = ScreenLogger(__name__)

transformers.logging.set_verbosity_info()


class EvalCallback(TrainerCallback):
    def __init__(self, tensorboard_callback: TensorBoardCallback, model, wikitext_data):
        self.model = model
        self.tensorboard_callback = tensorboard_callback
        self.wikitext_data = wikitext_data
        self.device = model.device

    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step % 30 == 0:
            with torch.no_grad():
                self.model.eval()

                ppl = ppl_eval(self.model, self.wikitext_data, self.device)

                print(f"Wikitext perplexity: {ppl}", flush=True)

                tb_writer = self.tensorboard_callback.tb_writer
                tb_writer.add_scalar("eval_wikitext_perplexity", ppl, state.global_step)

                self.model.train()


class CustomJsonDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset, tokenizer, add_special_tokens: bool, block_size: int = 1024) -> None:
        raw_data = dataset
        self.tokenizer = tokenizer
        self.block_size = block_size

        self.add_special_tokens = add_special_tokens
        tokenized_datasets = []

        for d in raw_data:
            tokenized_datasets.append(self.tokenize_function(d))

        grouped_dataset, total_length = self.group_texts(tokenized_datasets)
        self.input_ids = grouped_dataset["input_ids"]
        self.labels = grouped_dataset["labels"]

        print(f"Training orthogonal matrices over num_tokens={total_length} tokens.")
        print("self.input_ids", len(self.input_ids), len(self.input_ids[0]))

        self.data = [dict(input_ids=self.input_ids[i], labels=self.labels[i]) for i in range(len(self.input_ids))]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i) -> dict[str, Any]:
        return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __iter__(self):
        return iter(self.data)

    def tokenize_function(self, examples):
        return self.tokenizer(examples, add_special_tokens=self.add_special_tokens)

    def group_texts(self, examples):
        # Concatenate all texts.
        # Initialize an empty dictionary
        concatenated_examples = {}

        # Loop through the list of dictionaries
        for d in examples:
            # Loop through the keys in each dictionary
            for key in d:
                # If the key is not already a key in the dict_of_lists, create a new list
                if key not in concatenated_examples:
                    concatenated_examples[key] = []
                # Append the value to the list associated with the key in dict_of_lists
                concatenated_examples[key].extend(d[key])
        total_length = len(concatenated_examples["input_ids"])

        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= self.block_size:
            total_length = (total_length // self.block_size) * self.block_size

        # Split by chunks of max_len.
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }

        result["labels"] = result["input_ids"].copy()
        return result, total_length


def main(args: argparse.Namespace) -> None:
    # 1. Define original model
    print("\n[INFO]: Loading model ...")

    device = args.device

    model, _ = get_model(
        args.model_dir,
        args.data_type,
        device,
        args.multi_gpu,
        args.multi_device,
        args.model_attn_implementation,
        trust_remote_code=False,
    )
    model = model.eval()

    if args.loss_type != "origin":
        original_model, _ = get_model(
            args.model_dir,
            args.data_type,
            device,
            args.multi_gpu,
            args.multi_device,
            args.model_attn_implementation,
            trust_remote_code=False,
        )
    else:
        original_model = None

    preprocess_for_quantization(model)

    model_type = model.config.model_type

    if model_type not in ["llama", "qwen3"] and not args.force_custom_architecture:
        raise ValueError(
            f"The requested model uses the architecture model_type='{model_type}', but this script has only been validated for llama. You can force-enable arbitrary architectures with `--force_custom_architecture`. Use at your own risk."
        )

    if args.multi_gpu or args.multi_device:
        raise NotImplementedError(
            "The arguments `--multi_gpu` and `--multi_device` are currently not supported in this script."
        )

    if args.quant_algo_config_file is None:
        args.quant_algo_config_file = [["rotation", args.rotation_algo_config_file]]
    else:
        if len(args.quant_algo_config_file) != 1:
            raise NotImplementedError(
                f"This script has not been tested with more than one additional algorithm on top of rotation, got args.quant_algo_config_file={args.quant_algo_config_file}."
            )

        if args.quant_algo_config_file[0][0] != "gptq":
            raise NotImplementedError(
                f"This script has not been tested with other additional algorithms than gptq on top of rotation, got {args.quant_algo_config_file[0][0]}."
            )

        args.quant_algo_config_file = [["rotation", args.rotation_algo_config_file]] + args.quant_algo_config_file

    tokenizer = get_tokenizer(args.model_dir, model_type=model_type, trust_remote_code=False)

    # Define the training and validation datasets.
    print("\n[INFO]: Loading dataset ...")

    DATASET_TO_PARAMS = {
        "pileval": {"path": "mit-han-lab/pile-val-backup", "split": "validation"},
        "wikitext": {"path": "Salesforce/wikitext", "name": "wikitext-2-raw-v1", "split": "train"},
    }

    dataset = datasets.load_dataset(**DATASET_TO_PARAMS[args.training_dataset])
    dataset = dataset.shuffle(seed=42)
    dataset = dataset["text"]
    dataset = dataset[: args.num_samples]

    train_data = CustomJsonDataset(dataset, tokenizer, block_size=2048, add_special_tokens=args.add_special_tokens)

    validation_data = get_wikitext2(tokenizer, nsamples=256, seqlen=1024, device="cuda")
    eval_dataset = Dataset.from_dict(dict(input_ids=validation_data))

    def f(examples):
        examples["labels"] = examples["input_ids"]
        return examples

    eval_dataset = eval_dataset.map(f)

    wikitext_data = datasets.load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wikitext_data = tokenizer("\n\n".join(wikitext_data["text"]), return_tensors="pt")

    template = LLMTemplate.get(model_type)

    # Load algorithm configs from files if provided
    algo_configs = {}
    for algo_name, algo_config_file in args.quant_algo_config_file:
        algo_configs[algo_name] = load_quant_algo_config_from_file(algo_config_file)
        print(f"[INFO]: Loaded algorithm configuration for {algo_name} from {algo_config_file}.")

    quant_algo = [algo_and_config_path[0] for algo_and_config_path in args.quant_algo_config_file]

    quant_config = template.get_config(
        scheme=args.quant_scheme,
        algorithm=quant_algo,
        layer_config={},
        attention_scheme=args.attention_dtype,
        exclude_layers=args.exclude_layers,
        algo_configs=algo_configs if algo_configs else None,
    )

    algo_config = copy.deepcopy(quant_config.algo_config)

    # Prepare the quantization config used for learning rotations only.
    quant_config_rotation = copy.deepcopy(quant_config)
    quant_config_rotation.algo_config = quant_config_rotation.algo_config[:1]

    if args.train_activation_only:
        quant_config_rotation.global_quant_config.weight = None

    input_tensors_dynamic = False
    if quant_config_rotation.global_quant_config.input_tensors is not None:
        input_tensors_dynamic = True
        quant_config_rotation.global_quant_config.input_tensors.is_dynamic = True

    if quant_config_rotation.global_quant_config.weight is not None:
        quant_config_rotation.global_quant_config.weight.is_dynamic = True

    if quant_config.global_quant_config.bias is not None:
        raise NotImplementedError(
            f"This script has not been tested with bias quantization, got quant_config.global_quant_config.bias={quant_config.global_quant_config.bias}."
        )

    if quant_config.global_quant_config.output_tensors is not None:
        raise NotImplementedError(
            f"This script has not been tested with layer output (activation output) quantization, got quant_config.global_quant_config.output_tensors={quant_config.global_quant_config.output_tensors}."
        )

    # Prepare the quantization config used for actual quantization.
    quant_config_no_rotation = quant_config
    quant_config_no_rotation.algo_config = quant_config_no_rotation.algo_config[1:]

    requires_calibration_data = input_tensors_dynamic or len(quant_config.algo_config) > 1

    quantizer = ModelQuantizer(quant_config_rotation, args.multi_device)

    with torch.no_grad():
        model = quantizer.quantize_model(model)

    args.exclude_layers = quantizer.config.exclude
    model.use_cache = False

    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    rotation_config = quant_config_rotation.algo_config[0]

    trainable_parameters_ortho, trainable_parameters_adam = RotationProcessor.get_trainable_parameters(
        model, rotation_config=rotation_config
    )

    for param in trainable_parameters_ortho:
        print("Trainable orthogonal parameter:", param.shape)

    for param in trainable_parameters_adam:
        print("Trainable standard parameter (using Adam optimizer):", param.shape)

    # TODO: stiefel=False is not stable.
    # Other options are geoopt.optim.RiemannianSGD and geoopt.optim.RiemannianAdam
    # as used in OSTQuant, but they seem to be not very stable on the orthogonal group.
    if len(trainable_parameters_adam) == 0:
        optimizer = SGDG(trainable_parameters_ortho, lr=args.learning_rate, stiefel=True)
    elif len(trainable_parameters_ortho) == 0:
        optimizer = torch.optim.Adam(trainable_parameters_adam, lr=args.smooth_learning_rate)
    else:
        optimizer = AdamAndSGDGOptimizer(
            sgdg_params=trainable_parameters_ortho,
            adam_params=trainable_parameters_adam,
            learning_rate=args.learning_rate,
            smooth_learning_rate=args.smooth_learning_rate,
        )

    # TODO: why is meta-llama/Llama-2-7b-hf using dtype="fp16" in its config.json?
    fp16 = False
    bf16 = False
    if args.train_dtype == "bf16":
        bf16 = True
    elif args.train_dtype == "fp16":
        fp16 = True

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=fp16,
        bf16=bf16,
        log_on_each_node=False,
        logging_steps=1.0,
        learning_rate=args.learning_rate,
        logging_dir=args.output_dir,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=30,
        eval_on_start=True,
        do_train=True,
        overwrite_output_dir=True,
        # weight_decay=0.,  # TODO: does it really have in influence as we use our own optimizer?
        gradient_checkpointing=True,
        # gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=args.max_steps,
        lr_scheduler_type="cosine",  # scheduler used in spinquant.
        save_strategy="no",  # do not save intermediary checkpoints.
    )
    training_args.model_max_length = (2048,)  # passed in bash in spinquant?

    print("training_args:", training_args)

    print("Trainable parameters summary:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.dtype, param.shape, param.device, "ptr:", param.data_ptr())

    print("Model:", model)

    trainer = RotationTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
        optimizers=(optimizer, None),
        original_model=original_model,
        loss_type=args.loss_type,
    )

    for callback in trainer.callback_handler.callbacks:
        if isinstance(callback, TensorBoardCallback):
            tensorboard_callback = callback
            break
    else:
        raise RuntimeError(
            f"Could not find a TensorBoardCallback among {trainer.callback_handler.callbacks}. Please install tensorboard with `pip install tensorboard`."
        )

    # Logs statistics for each training parameter in tensorboard.
    stats_callback = OrthogonalTrainingCallback(tensorboard_callback, trainer.model, rotation_config)

    # Runs additional evaluation than `Trainer` does.
    eval_callback = EvalCallback(tensorboard_callback, trainer.model, wikitext_data=wikitext_data)

    trainer.add_callback(stats_callback)
    trainer.add_callback(eval_callback)

    if not args.skip_training:
        trainer.train()

    torch.cuda.synchronize()
    if original_model is not None:
        del original_model
        del trainer

    torch.cuda.empty_cache()
    gc.collect()

    if args.export_rotation:
        rotations = {}

        if rotation_config.r1:
            if not rotation_config.online_r1_rotation:
                rotations["r1"] = model.shared_r1_rotation.data
            else:
                for name, submodule in model.named_modules():
                    if isinstance(submodule, RotationLinear) and submodule.hint_in == "r1":
                        rotations[name + "_r1"] = submodule.rotation_in.data.clone()

        if rotation_config.r2:
            for name, submodule in model.named_modules():
                if isinstance(submodule, RotationLinear) and submodule.hint_out == "r2":
                    rotations[name + "_r2"] = submodule.rotation_out.data.clone()

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        safetensors.torch.save_file(rotations, Path(args.output_dir, "rotations.safetensors"))

    model = model.eval()

    # Sanity check that perplexity does not change after `RotationProcessor.post_process_trained_rotation` call.
    # NOTE: This call is significantly slower than the previous one when using MXFP4 quantization for weights,
    # as dynamic MXFP4 quantization (used during training) has a fast code path, but not static (used here).
    with torch.no_grad():
        ppl = ppl_eval(model, wikitext_data, device)

        print(f"Wikitext perplexity before `RotationProcessor.post_process_trained_rotation` call: {ppl}", flush=True)

    model = RotationProcessor.post_process_trained_rotation(model, quantization_config=quant_config_no_rotation)

    # Quantize the pre-processed (rotated) model.
    calibration_dataloader = None
    if requires_calibration_data:
        calibration_dataloader = get_calib_dataloader(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
            num_calib_data=args.num_calib_data,
            seqlen=args.seq_len,
            device=device,
        )

    quantizer = ModelQuantizer(quant_config_no_rotation, args.multi_device)

    with torch.no_grad():
        model = quantizer.quantize_model(model, calibration_dataloader)

        # `quant_config_rotation.algo_config` might be updated when running rotation pre-processing (e.g. filling `online_rotation_layers`).
        algo_config[0] = quant_config_rotation.algo_config[0]

        model.quant_config.algo_config = algo_config

    model = quantizer.freeze(model)

    # Export the model.
    if args.model_export is not None:
        print("\n[INFO]: Exporting hugging face format safetensors...")
        with torch.no_grad():
            export_safetensors(
                model=model,
                output_dir=args.output_dir,
                weight_format=args.export_weight_format,
                pack_method=args.pack_method,
            )
            tokenizer.save_pretrained(args.output_dir)

    # Evaluate the model.
    if not args.skip_evaluation:
        print("\n[INFO]: Evaluating ...")
        args.use_ppl_eval_model = True
        eval_model(
            args=args,
            model=model,
            main_device=device,
            save_metrics_to_csv=args.save_metrics_to_csv,
            output_dir=args.metrics_output_dir,
            multimodal=False,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    # Argument for model
    parser.add_argument(
        "--model_dir",
        help="Specify where the HuggingFace model is. This example support Llama, OPT models",
        required=True,
    )
    parser.add_argument(
        "--force_custom_architecture",
        action="store_true",
        help="Allow training other architectures than llama. Only llama has been validated. Use at your own risk.",
    )

    parser.add_argument("--device", help="Device for running the quantizer", default="cuda:0")
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument(
        "--model_attn_implementation",
        help="The attention implementation to use in the model",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
    )
    parser.add_argument(
        "--multi_device",
        action="store_true",
        help="we allow you to use this mode to run a model quantization that exceeds the size of your gpu memory if you use args.multi_gpu and still run into OOM "
        "now it only supports thr common quantization without algorithms, please note that this can lead to very slow quantization.",
    )

    # Argument for training & calibration datasets.
    parser.add_argument(
        "--training_dataset",
        help="Dataset for training.",
        default="pileval",
        choices=[
            "pileval",
            "wikitext",
        ],
    )
    parser.add_argument(
        "--dataset",
        help="Dataset for calibration (optional).",
        default="pileval_for_awq_benchmark",
        choices=[
            "pileval",
            "wikitext",
            "pileval_for_awq_benchmark",
            "wikitext_for_gptq_benchmark",
        ],
    )
    parser.add_argument(
        "--data_type", help="Datatype of the model", default="auto", choices=["auto", "float16", "bfloat16", "float32"]
    )
    parser.add_argument("--seq_len", type=int, help="Sequence length of training  and calibration data.", default=512)
    parser.add_argument("--train_batch_size", help="Batch size for training.", type=int, default=1)
    parser.add_argument("--batch_size", help="Batch size for calibration.", type=int, default=1)
    parser.add_argument("--num_samples", help="Number of samples for training.", type=int, default=1000)
    parser.add_argument("--num_calib_data", help="Number of samples for calibration.", type=int, default=512)

    parser.add_argument(
        "--add_special_tokens",
        help="Whether to add special tokens (BOS, etc.) in the training data.",
        action="store_true",
    )
    parser.add_argument(
        "--train_activation_only",
        action="store_true",
        help="If set, QDQ will be applied only on activations during rotation training. This is e.g. useful to reproduce SpinQuant results which later uses GPTQ to quantize weights.",
    )
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1.5,
        help="Learning rate for parameters with orthogonality constraint (rotations) using Cayley SGD optimizer.",
    )
    parser.add_argument(
        "--smooth_learning_rate",
        type=float,
        default=1e-2,
        help="Learning rate for other parameters (e.g. SmoothQuant scales) using standard Adam optimizer.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="origin",
        help="Which loss to optimize on. Can be any of `original` (cross-entropy loss), `kl_top_k` (KL-Divergence with k classes only) where `k` is the number of highest likelihood classes considered for loss evaluation, using the original non-quantized model (e.g. `kl_top_10000`).",
    )
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)

    parser.add_argument("--max_steps", type=int, default=100)

    parser.add_argument(
        "--train_dtype",
        type=str,
        default="bf16",
        help="Whether to set HF's TrainingArguments.bf16 or TrainingArguments.fp16 to True.",
        choices=["fp32", "bf16", "fp16"],
    )

    # Argument for quantization
    parser.add_argument("--group_size", help="Group size for per_group quantization.", type=int, default=128)
    parser.add_argument(
        "--quant_scheme",
        help="Quantization scheme to use. Supported schemes: all built-in schemes and custom schemes registered."
        "For the built-in schemes and their detailed configuration, see https://quark.docs.amd.com/latest/pytorch/user_guide_config_for_llm.html.",
        choices=LLMTemplate.get_supported_schemes(),
        default=None,
        type=str,
    )
    parser.add_argument(
        "--rotation_algo_config_file", type=str, help="The JSON file path for the rotation algo.", required=True
    )
    parser.add_argument(
        "--quant_algo_config_file",
        action="append",
        nargs=2,
        metavar=("ALGO_NAME", "CONFIG_FILE"),
        help="Specify a configuration file for a specific quantization algorithm. "
        "Can be repeated for multiple algorithms. "
        "Example: --quant_algo_config_file awq ./awq_config.json --quant_algo_config_file gptq ./gptq_config.json "
        "(provides custom config files for AWQ and GPTQ algorithms).",
    )
    parser.add_argument(
        "--exclude_layers",
        type=str,
        nargs="*",  # Allows to pass a list of strings
        default=None,  # Default is None to allow model-specific layer exclusion
        help='List of layers to exclude from quantization. Default depends on model type. Usage: `--exclude_layers "*down_proj*" "*31.fc*" "*k_proj"`. To avoid excluding layers at all, simply use `--exclude_layers` without any argument.',
    )
    parser.add_argument("--scale_format", help="Scale format", default="e4m3", choices=["e4m3", "float32"])
    parser.add_argument(
        "--scale_calculation_mode", help="Scale calculation mode", default="even", choices=["even", "floor", "ceil"]
    )

    # Argument for custom quantization
    parser.add_argument(
        "--attention_dtype", help="The dtype of attention quantization.", type=str, default=None, choices=["fp8"]
    )
    # Argument for export
    parser.add_argument(
        "--model_export",
        help="Model export format",
        default=None,
        type=str,
        choices=[None, "hf_format"],
    )
    parser.add_argument("--export_rotation", action="store_true")

    parser.add_argument("--output_dir", default="exported_model")

    parser.add_argument(
        "--export_weight_format",
        type=str,
        help="Whether to export weights compressed or uncompressed",
        default="real_quantized",
        choices=["fake_quantized", "real_quantized"],
    )

    parser.add_argument(
        "--pack_method", type=str, help="Pack method for awq_export", default="reorder", choices=["order", "reorder"]
    )

    # Argument for evaluation
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--use_ppl_eval_model", action="store_true")
    parser.add_argument(
        "--evaluation_dataset",
        help="Dataset for evaluation",
        default="wikitext",
        choices=["wikitext", "wikitext_gpt_oss_120b", "wikitext_gpt_oss_20b"],
    )
    parser.add_argument("--save_metrics_to_csv", action="store_true")
    parser.add_argument("--metrics_output_dir", default="metrics_output_dir", help="Output path of csv with metrics.")
    parser.add_argument(
        "--tasks",
        default=None,
        type=str,
        metavar="task1,task2",
        help="Comma-separated list of task names or task groupings to evaluate on.",
    )
    parser.add_argument("--use_ppl_eval_for_kv_cache", action="store_true")
    parser.add_argument(
        "--ppl_eval_for_kv_cache_context_size",
        type=int,
        help="Context size used in PPL evaluation for KV cache.",
        default=1024,
    )
    parser.add_argument(
        "--ppl_eval_for_kv_cache_sample_size",
        type=int,
        help="Sample size used in PPL evaluation for KV cache.",
        default=512,
    )
    parser.add_argument(
        "--ppl_eval_for_kv_cache_patch_size",
        type=int,
        help="Patch size used in PPL evaluation for KV cache.",
        default=None,
    )
    parser.add_argument(
        "--eval_batch_size",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Batch size for evaluation. Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default is `1`.",
    )
    parser.add_argument(
        "--max_eval_batch_size",
        type=int,
        default=64,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--num_eval_data",
        help="Number of samples for evaluation. The default value is -1, which means the entire dataset is used for evaluation.",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--num_fewshot", type=int, default=None, metavar="N", help="Number of examples in few-shot context"
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        help="Providing `--apply_chat_template` without an argument will apply the default chat template to the prompt.",
    )
    parser.add_argument("--use_mlperf_rouge", action="store_true")
    parser.add_argument("--eval_data_dir", help="Dataset for evaluation", type=str, default=None)
    args = parser.parse_args()

    if args.quant_algo_config_file is not None:
        for algo_config in args.quant_algo_config_file:
            if len(algo_config) != 2:
                raise ValueError(
                    f"Invalid --quant_algo_config_file argument: {algo_config}. "
                    f"Expected exactly 2 values (ALGO_NAME, CONFIG_FILE), but got {len(algo_config)}."
                )
            algo_name, config_file = algo_config
            if not os.path.isfile(config_file):
                raise ValueError(
                    f"Configuration file '{config_file}' for algorithm '{algo_name}' does not exist. "
                    f"Please provide a valid config file path."
                )

    main(args)
