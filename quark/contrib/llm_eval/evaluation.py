#
# Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import csv
import datetime
import json
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

import evaluate  # type: ignore
import nltk  # type: ignore
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset, load_dataset
from lm_eval.evaluator import simple_evaluate
from lm_eval.loggers import EvaluationTracker
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import (
    handle_non_serializable,
    load_yaml_config,
    make_table,
    simple_parse_args_string,
)
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, PreTrainedTokenizer  # type: ignore[attr-defined]

from quark.shares.utils.log import ScreenLogger

if TYPE_CHECKING:
    from lm_eval.api.model import T

logger = ScreenLogger(__name__)

SUPPORTED_MODEL_ARGS = {"enable_thinking", "think_end_token", "chat_template_args"}


def eval_model(
    args: argparse.Namespace,
    model: nn.Module,
    main_device: str,
    save_metrics_to_csv: bool = False,
    output_dir: Path | str = "metrics_output_dir",
    multimodal: bool = False,
) -> None:
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(  # type: ignore
        args.model_dir,
        trust_remote_code=True,
    )

    if args.num_eval_data != -1 and not args.use_mlperf_rouge:
        testenc = tokenizer("\n\n".join(testdata["text"][: args.num_eval_data]), return_tensors="pt")
    else:
        testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    metrics = []
    main_device = model.device

    # eval kv_cache ppl
    if args.use_ppl_eval_for_kv_cache:
        ppl_eval_for_kv_cache(
            model,
            testenc,
            args.ppl_eval_for_kv_cache_context_size,
            args.ppl_eval_for_kv_cache_sample_size,
            args.ppl_eval_for_kv_cache_patch_size,
            main_device,
        )
    # eval model ppl
    elif args.use_ppl_eval_model:
        ppl = ppl_eval(model, testenc, main_device)
        print(f"\n[INFO] Perplexity: {ppl.item()}")
        metrics.append(["Perplexity", ppl.cpu().numpy()])

    # eval model mlperf_rouge
    if args.use_mlperf_rouge:
        mlperf_rouge_eval(
            eval_data_dir=args.eval_data_dir,
            num_eval_data=args.num_eval_data,
            model=model,
            model_dir=args.model_dir,
            main_device=main_device,
            batch_size=args.eval_batch_size,
        )

    # eval tasks
    if args.tasks is not None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        task_eval(
            model,
            tokenizer,
            batch_size=args.eval_batch_size,
            max_batch_size=args.max_eval_batch_size,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            apply_chat_template=args.apply_chat_template,
            multimodal=multimodal,
            gen_kwargs=getattr(args, "gen_kwargs", None),
            model_args=getattr(args, "model_args", ""),
            fewshot_as_multiturn=getattr(args, "fewshot_as_multiturn", False),
            output_path=getattr(args, "output_path", None),
            log_samples=getattr(args, "log_samples", False),
        )

    # save result into csv
    if save_metrics_to_csv and len(metrics) > 0:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        evaluation_metrics_path = Path(output_dir, "evaluation_metrics.csv").as_posix()
        with open(evaluation_metrics_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            for metric in metrics:
                writer.writerow(metric)
        print(f"[INFO] Saved evaluation_metrics to {evaluation_metrics_path}.")


@torch.no_grad()
def ppl_eval(model: nn.Module, testenc: Any, dev: str, file_format: str = "hf_format") -> torch.Tensor:
    if file_format != "onnx_format":
        model.eval()
    # Set sequence length as 2048 for wikitext dataset evaluation
    seqlen_for_eval = 2048
    testenc = testenc.input_ids
    nsamples = testenc.numel() // seqlen_for_eval

    testenc = testenc.to(dev)
    nlls = []

    if file_format == "onnx_format":
        try:
            import onnxruntime_genai as og  # type: ignore
        except ModuleNotFoundError:
            raise ImportError(
                "Quark depends on ONNX Runtime GenAI. Please install ONNX Runtime GenAI by following the instructions at: https://onnxruntime.ai/docs/genai/howto/install"
            )

        params = og.GeneratorParams(model)
        params.try_graph_capture_with_max_batch_size(1)
        search_options = {}
        search_options["max_length"] = seqlen_for_eval + 1
        params.set_search_options(**search_options)

    for i in tqdm(range(nsamples)):
        batch = testenc[:, (i * seqlen_for_eval) : ((i + 1) * seqlen_for_eval)].to(dev)
        if file_format == "onnx_format":
            # onnx model logits using oga
            generator = og.Generator(model, params)
            generator.append_tokens(batch.cpu().numpy())
            shift_logits = torch.tensor(generator.get_output("logits")[0][:-1]).to(dev)
        else:
            lm_logits = model(batch)["logits"]
            shift_logits = lm_logits[:, :-1, :].contiguous()

        # for some models like deepseek-r1 with mtp layer, the second dimension of shift_logits may longer than
        # seqlen_for_eval-1, so we need to change the batch_step_size of shift_labels to shift_logits.size(1)+1
        batch_step_size = shift_logits.size(1) + 1
        shift_labels = testenc[:, (i * batch_step_size) : ((i + 1) * batch_step_size)][:, 1:]
        loss_fct = torch.nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        neg_log_likelihood = loss.float() * seqlen_for_eval
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen_for_eval))

    return ppl


def lm_eval_entrypoint(
    model: str,
    model_args: dict[str, Any] | str,
    tasks: str | None,
    limit: int | None = None,
    samples: str | None = None,
    metadata: None | dict[str, Any] = None,
    include_path: str | None = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    log_samples: bool = False,
    output_path: str | None = None,
    verbosity: str | None = None,
    trust_remote_code: bool = False,
    gen_kwargs: dict[str, Any] | None = None,
    num_fewshot: int | None = None,
    batch_size: str | int = 1,
    max_batch_size: int | None = None,
    seed: list[int] | None = None,
    system_instruction: str | None = None,
    use_cache: str | None = None,
    check_integrity: str | None = None,
    write_out: bool = False,
    device: str | None = None,
    confirm_run_unsafe_code: bool = False,
    show_config: bool = False,
    **kwargs: Any,
) -> None:
    """
    This entrypoint supports `model_args` containing `pretrained` as an already instantiated nn.Module.

    `cli_evaluate` from lm-evaluation-harness currently does not properly support it.
    """
    hf_hub_log_args = ""

    if len(kwargs) > 0:
        logger.warning(f"Ignored arguments in the evaluation in AMD Quark: {kwargs.keys()}")

    if seed is None:
        seed = [0, 1234, 1234, 1234]

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # update the evaluation tracker args with the output path and the HF token
    if output_path:
        hf_hub_log_args += f",output_path={output_path}"
    if os.environ.get("HF_TOKEN", None):
        hf_hub_log_args += f",token={os.environ.get('HF_TOKEN')}"
    evaluation_tracker_args = simple_parse_args_string(hf_hub_log_args)
    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    if log_samples and not output_path:
        raise ValueError("Specify --output_path if providing --log_samples")

    if fewshot_as_multiturn and apply_chat_template is False:
        raise ValueError(
            "When `fewshot_as_multiturn` is selected, `apply_chat_template` must be set (either to `True` or to the chosen template name)."
        )

    if include_path is not None:
        logger.info(f"Including path: {include_path}")
    metadata = (
        simple_parse_args_string(model_args)
        if isinstance(model_args, str)
        else model_args
        if isinstance(model_args, dict)
        else {}
    ) | (metadata if isinstance(metadata, dict) else simple_parse_args_string(metadata))

    # `metata` is later cloned, and we can not afford cloning nn.Module in terms of memory.
    metadata = {key: value for key, value in metadata.items() if key != "pretrained"}

    task_manager = TaskManager(include_path=include_path, metadata=metadata)

    if limit:
        logger.warning(" --limit SHOULD ONLY BE USED FOR TESTING.REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")

    samples_loaded: Any = None
    if samples:
        assert limit is None, "If --samples is not None, then --limit must be None."
        samples_path = Path(samples)
        if samples_path.is_file():
            samples_loaded = json.loads(samples_path.read_text())
        else:
            samples_loaded = json.loads(samples)

    if tasks is None:
        raise ValueError("Please provide `tasks` argument, got tasks=None.")
    elif tasks in ["list", "list_groups", "list_tags", "list_subtasks"]:
        raise ValueError(f"tasks={tasks} is not supported in AMD Quark. Please run `lm_eval --tasks {tasks}` instead.")
    elif isinstance(tasks, str):
        if os.path.isdir(tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = load_yaml_config(task)
                    task_names.append(config)
            task_missing = [
                task for task in task_list if task not in task_names and "*" not in task
            ]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                logger.error(
                    f"Tasks were not found: {missing}\n Try `lm-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lm-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )
    else:
        task_names = tasks

    if trust_remote_code:
        logger.info("Passed `--trust_remote_code`, setting environment variable `HF_DATASETS_TRUST_REMOTE_CODE=true`")
        # HACK: import datasets and override its HF_DATASETS_TRUST_REMOTE_CODE value internally,
        # because it's already been determined based on the prior env var before launching our
        # script--`datasets` gets imported by lm_eval internally before these lines can update the env.
        import datasets
        from packaging.version import parse as vparse

        if vparse(datasets.__version__) < vparse("4.0.0"):
            datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
        else:
            logger.warning(
                "trust_remote_code and datasets scripts are no longer supported on datasets>=4.0.0. Skipping. If your task still requires this, please downgrade to datasets==3.6.0 or earlier."
            )

        if isinstance(model_args, dict):
            model_args["trust_remote_code"] = True
        else:
            model_args = model_args + ",trust_remote_code=True"

    logger.info(f"Selected Tasks: {task_names}")

    results = simple_evaluate(
        model=model,
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
        device=device,
        use_cache=use_cache,
        limit=limit,
        samples=samples_loaded,
        check_integrity=check_integrity,
        write_out=write_out,
        log_samples=log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=system_instruction,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        gen_kwargs=gen_kwargs,
        task_manager=task_manager,
        random_seed=seed[0],
        numpy_random_seed=seed[1],
        torch_random_seed=seed[2],
        fewshot_random_seed=seed[3],
        confirm_run_unsafe_code=confirm_run_unsafe_code,
        metadata=metadata,
    )

    if results is not None:
        if log_samples:
            samples = results.pop("samples")
        dumped = json.dumps(results, indent=2, default=handle_non_serializable, ensure_ascii=False)
        if show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        evaluation_tracker.save_results_aggregated(results=results, samples=samples if log_samples else None)

        if log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

        print(
            f"{model} ({model_args}), gen_kwargs: ({gen_kwargs}), limit: {limit}, num_fewshot: {num_fewshot}, "
            f"batch_size: {batch_size}{f' ({batch_sizes})' if batch_sizes else ''}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))


@torch.no_grad()
def task_eval(
    model: nn.Module,
    tokenizer: "PreTrainedTokenizer",
    max_batch_size: int | None,
    batch_size: int | str = 1,
    tasks: list[str] | None = None,
    num_fewshot: int | None = None,
    apply_chat_template: bool = False,
    multimodal: bool = False,
    output_path: str | None = None,
    gen_kwargs: dict[str, Any] | None = None,
    model_args: dict[str, Any] | str = "",
    fewshot_as_multiturn: bool = False,
    log_samples: bool = False,
) -> None:
    def create_from_arg_obj(
        cls: type["T"], arg_dict: dict[str, Any], additional_config: dict[str, Any] | None = None
    ) -> "T":
        arg_dict_keys = set(arg_dict.keys())
        dropped_keys = arg_dict_keys - {
            "pretrained",
            "tokenizer",
            "enable_thinking",
            "think_end_token",
            "chat_template_args",
        }

        # `additional_config` comes from https://github.com/EleutherAI/lm-evaluation-harness/blob/v0.4.9.1/lm_eval/evaluator.py#L230-L237,
        # and contains `batch_size` and `max_batch_size` information.
        # As the `pretrained` argument contains the model already initialized,
        # we drop the `device` argument here.
        if additional_config is None:
            additional_config = {}
        additional_config.pop("device")
        dropped_keys.add("device")

        assert additional_config["max_batch_size"] == max_batch_size
        assert additional_config["batch_size"] == batch_size

        pretrained = arg_dict.pop("pretrained")
        enable_thinking = arg_dict.pop("enable_thinking", None)
        think_end_token = arg_dict.pop("think_end_token", None)
        chat_template_args = arg_dict.pop("chat_template_args", None)

        if len(dropped_keys) > 0:
            print(
                f"The following parameters are not passed by AMD Quark to lm-evaluation-harness when instantiating {cls}: {dropped_keys}"
            )

        if multimodal:
            model_obj = cls(
                pretrained=pretrained,
                enable_thinking=enable_thinking,
                think_end_token=think_end_token,
                chat_template_args=chat_template_args,
                **additional_config,
            )
        else:
            tokenizer = arg_dict.pop("tokenizer")
            model_obj = cls(
                pretrained=pretrained,
                tokenizer=tokenizer,
                enable_thinking=enable_thinking,
                think_end_token=think_end_token,
                chat_template_args=chat_template_args,
                **additional_config,
            )
        return model_obj

    # TODO: check whether we really need to override `create_from_arg_obj`.
    HFLM.create_from_arg_obj = classmethod(create_from_arg_obj)

    if isinstance(model_args, str):
        model_args_dict = simple_parse_args_string(model_args)
    elif isinstance(model_args, dict):
        model_args_dict = model_args
    else:
        raise ValueError(f"model_args of type {type(model_args)}, but only str or dict is supported.")

    if any(key not in SUPPORTED_MODEL_ARGS for key in model_args_dict):
        raise ValueError(
            f"The provided model_args='{model_args}' is not supported in AMD Quark. The only supported model_args keys are: {','.join(list(SUPPORTED_MODEL_ARGS))}."
        )

    if multimodal:
        lm_eval_model = "hf-multimodal"
        model_args = {"pretrained": model, **model_args_dict}
    else:
        lm_eval_model = "hf"
        model_args = {"pretrained": model, "tokenizer": tokenizer, **model_args_dict}

    lm_eval_entrypoint(
        model=lm_eval_model,
        model_args=model_args,
        tasks=tasks,
        apply_chat_template=apply_chat_template,
        fewshot_as_multiturn=fewshot_as_multiturn,
        log_samples=log_samples,
        output_path=output_path,
        gen_kwargs=gen_kwargs,
        num_fewshot=num_fewshot,
        batch_size=batch_size,
        max_batch_size=max_batch_size,
    )


# use_ppl_eval_for_kv_cache
def update_model_kwargs_for_generation(
    outputs: Any,
    model_kwargs: dict[str, Any],
    is_encoder_decoder: bool = False,
) -> dict[str, Any]:
    """
    Update model arguments for next generation.
    """
    # update past_key_values
    for key in ["past_key_values", "mems", "past_buckets_states"]:
        if key in outputs:
            model_kwargs["past_key_values"] = outputs[key]

    if "state" in outputs:
        model_kwargs["state"] = outputs.state

    if not is_encoder_decoder:
        # update attention mask
        if "attention_mask" in model_kwargs:
            attention_mask = model_kwargs["attention_mask"]
            model_kwargs["attention_mask"] = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )

    return model_kwargs


@torch.no_grad()
def get_cumulative_logprob(model: nn.Module, input_tokens: torch.Tensor, future_context: list[int], dev: str) -> float:
    """
    Calculate the cumulative logprob of the given future context.
    Parameters:
        model : AutoModelForCausalLM
        input_tokens : torch.Tensor
        future_context : list
            A list of token IDs, typically adjacent to the input token. Used for calculating cumulative log probabilities.
        dev : str
            The compute device.
    Returns:
        float
            The cumulative log probabilities of input_tokens and future_context.
    """
    # init
    input_tokens = input_tokens.to(dev)
    model_kwargs = {
        "use_cache": True,
        "attention_mask": torch.ones(input_tokens.shape, dtype=int).to(dev),
    }

    cumulative_logpro = 0.0
    for next_token in future_context:
        model_inputs = model.prepare_inputs_for_generation(input_tokens, **model_kwargs)  # type: ignore
        outputs = model(
            **model_inputs,
            return_dict=True,
            output_hidden_states=True,
        )

        # get the output of the last decoder layer
        hidden_states = outputs.hidden_states[-1].clone().detach()  # type: ignore

        # get logits
        selected_token_indices = torch.Tensor([(model_inputs["input_ids"].numel() - 1)]).to(int).to(dev)
        logits = model.lm_head(hidden_states[0, :, :].index_select(0, selected_token_indices))  # type: ignore
        logprobs = torch.log_softmax(logits, dim=-1, dtype=torch.float)

        # get the specific token's logprob and accumulate
        cumulative_logpro += logprobs[0, next_token].item()

        # update for next step
        input_tokens = torch.cat(
            (
                input_tokens,
                torch.tensor(
                    [
                        [
                            next_token,
                        ],
                    ]
                ).to(dev),
            ),
            dim=-1,
        )
        model_kwargs = update_model_kwargs_for_generation(
            outputs,
            model_kwargs,
            is_encoder_decoder=model.config.is_encoder_decoder,  # type: ignore
        )
    return cumulative_logpro


@torch.no_grad()
def ppl_eval_for_kv_cache(
    model: nn.Module, testenc: torch.Tensor, context_size: int, sample_size: int, patch_size: int, dev: str
) -> None:
    """
    A perplexity-computing test for the KV cache system.
    Parameters:
        model : nn.Module
        testenc : torch.Tensor
            The input token IDs.
        context_size : int
            The size of the context used for generation.
        sample_size : int
            The number of the output generated by the given context size. This his variable also \
            defines the size of the individual patch.
        patch_size : int
            The size of patches, if specified, will determine the number of patches. If not, \
            the number of patches will be determined by sample_size and the number of input tokens.
        dev : str
    Returns:
        None
    """
    print(f"Initializing @ {datetime.datetime.now()}")

    # Prepare parameters
    my_enc = testenc.input_ids
    n_samples = sample_size
    n_patches = math.ceil((my_enc.numel() - context_size - 1) / n_samples)
    if patch_size is not None:
        n_patches = patch_size

    ppl = 0.0
    num_tokens_generated = 0
    starting_time = datetime.datetime.now()
    print(
        f"Starting generation @ {starting_time} "
        f"will try to process {n_patches} patch(es), "
        f"generating {n_samples} tokens in each patch "
        f"from the initial context of {context_size} tokens."
    )
    for idx in range(n_patches):
        context = my_enc[:, idx * n_samples : idx * n_samples + context_size]
        upper_boundary = min((idx + 1) * n_samples + context_size, my_enc.numel())
        future_context = my_enc[0, idx * n_samples + context_size : upper_boundary].tolist()

        logprobs = get_cumulative_logprob(model=model, input_tokens=context, future_context=future_context, dev=dev)

        ppl -= logprobs
        num_tokens_generated += len(future_context)

        print(
            f"Iteration {idx + 1} of {n_patches} Intermediate "
            "Estimates:\n"
            f"\tCross-entropy_intermediate={ppl / num_tokens_generated}\n"
            f"\tPerplexity_intermediate={math.exp(ppl / num_tokens_generated)}"
        )

    ending_time = datetime.datetime.now()
    print(
        f"Done @ {ending_time} after processing for "
        f"{ending_time - starting_time} generated {num_tokens_generated} tokens."
    )
    print(
        f"Integral Cross-Entropy={ppl} Average Cross-Entropy="
        f"{ppl / num_tokens_generated} PPL={math.exp(ppl / num_tokens_generated)}"
    )


def rouge_meteor_generations(
    dataset: str,
    model: nn.Module,
    tokenizer: "PreTrainedTokenizer",
    num_eval_data: int,
    import_file_format: str,
    import_model_dir: str,
    model_args: dict[str, Any],
    batch_size: int,
    device: str,
    max_new_toks: int,
    seq_len: int,
) -> Dataset:
    def get_generation(sample: dict[str, Any], input_field: str) -> dict[str, Any]:
        inputs = tokenizer(  # type: ignore
            sample[input_field],
            return_tensors="pt",
            max_length=seq_len,
            truncation=True,
        )

        if import_file_format == "onnx_format" and "chatglm3-6b" in model_args["pretrained"].lower():
            with open(import_model_dir + "/genai_config.json") as f:
                genai_config = json.load(f)
            config = AutoConfig.from_pretrained(model_args["pretrained"], trust_remote_code=True)
            config.num_key_value_heads = genai_config["model"]["decoder"]["num_key_value_heads"]
            past_key_values = [
                (
                    torch.zeros(
                        (
                            batch_size,
                            config.num_key_value_heads,
                            inputs["input_ids"].shape[1],
                            config.hidden_size // config.num_attention_heads,
                        )
                    ),
                    torch.zeros(
                        (
                            batch_size,
                            config.num_key_value_heads,
                            inputs["input_ids"].shape[1],
                            config.hidden_size // config.num_attention_heads,
                        )
                    ),
                )
                for i in range(config.num_layers)
            ]
            model.past_key_values = past_key_values

            output_ids = model.generate(  # type: ignore
                inputs["input_ids"].to(device),
                max_new_tokens=max_new_toks,
                pad_token_id=tokenizer.eos_token_id,  # type: ignore
                past_key_values=past_key_values,
            )
        else:
            output_ids = model.generate(  # type: ignore
                inputs["input_ids"].to(device),
                max_new_tokens=max_new_toks,
                pad_token_id=tokenizer.eos_token_id,  # type: ignore
            )

        sample["predicted"] = tokenizer.decode(output_ids[0], skip_special_tokens=True)  # type: ignore
        return sample

    def get_dataset_sample(dataset: Dataset, num_eval_data: int) -> Dataset:
        if num_eval_data != -1:
            return Dataset.from_dict(dataset[:num_eval_data])  # type: ignore
        else:
            return dataset

    # For datasets >= 4.0, the dataset no longer supports the legacy .py builder. We use the preprocessed Parquet/Arrow data.
    # 1. load dataset, 2. retrieve a subsample of the dataset, if provided (for quicker evals) 3. generate preds
    dataset_generations: Dataset
    if dataset == "xsum":
        dataset_loaded = load_dataset("knkarthick/xsum", split="test")
        dataset_loaded = get_dataset_sample(dataset_loaded, num_eval_data)  # type: ignore
        print("Generating predictions for XSUM Dataset")
        dataset_generations = dataset_loaded.map(lambda x: get_generation(x, input_field="dialogue"))  # type: ignore
    elif dataset == "cnn_dm":
        dataset_loaded = load_dataset(
            "abisee/cnn_dailymail",
            "3.0.0",
            split="test",
        )
        dataset_loaded = get_dataset_sample(dataset_loaded, num_eval_data)  # type: ignore
        print("Generating predictions for CNN_DM Dataset")
        dataset_generations = dataset_loaded.map(lambda x: get_generation(x, input_field="article"))  # type: ignore
    elif dataset == "samsum":
        dataset_loaded = load_dataset("knkarthick/samsum", split="test")
        dataset_loaded = get_dataset_sample(dataset_loaded, num_eval_data)  # type: ignore
        print("Generating predictions for SAMSUM Dataset")
        dataset_generations = dataset_loaded.map(lambda x: get_generation(x, input_field="dialogue"))  # type: ignore
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return dataset_generations


def rouge_eval(dataset: str, generations: Dataset) -> dict[str, float]:
    rouge = evaluate.load("rouge")
    if dataset == "xsum":
        print("Calculating Rouge on XSUM Dataset")
        rouge_preds = rouge.compute(predictions=generations["predicted"], references=generations["summary"])
    if dataset == "cnn_dm":
        print("Calculating Rouge on CNN DM Dataset")
        rouge_preds = rouge.compute(predictions=generations["predicted"], references=generations["highlights"])
    if dataset == "samsum":
        print("Calculating Rouge on SAMSUM Dataset")
        rouge_preds = rouge.compute(predictions=generations["predicted"], references=generations["summary"])

    for k, val in rouge_preds.items():
        rouge_preds[k] = float(val)
    return rouge_preds


def meteor_eval(dataset: str, generations: Dataset) -> dict[str, float]:
    meteor = evaluate.load("meteor")
    if dataset == "xsum":
        print("Calculating Rouge on XSUM Dataset")
        meteor_preds = meteor.compute(predictions=generations["predicted"], references=generations["summary"])
    if dataset == "cnn_dm":
        print("Calculating Rouge on CNN DM Dataset")
        meteor_preds = meteor.compute(predictions=generations["predicted"], references=generations["highlights"])
    if dataset == "samsum":
        print("Calculating Rouge on SAMSUM Dataset")
        meteor_preds = meteor.compute(predictions=generations["predicted"], references=generations["summary"])

    for k, val in meteor_preds.items():
        meteor_preds[k] = float(val)

    return meteor_preds


def calculate_rouge_score(model_outputs: list[str], ref_outputs: list[str]) -> dict[str, float]:
    metric = evaluate.load("rouge")

    m_preds = [pred.strip() for pred in model_outputs]
    m_targets = [target.strip() for target in ref_outputs]

    # rougeLSum expects newline after each sentence
    m_preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in m_preds]
    m_targets = ["\n".join(nltk.sent_tokenize(target)) for target in m_targets]
    m_result = metric.compute(predictions=m_preds, references=m_targets, use_stemmer=True, use_aggregator=False)
    m_rouge_result = {k: round(np.mean(v) * 100, 4) for k, v in m_result.items()}

    return m_rouge_result


def evaluate_openorca(df: pd.DataFrame, result_keys: dict[str, str]) -> dict[str, float]:
    print("Evaluating OpenOrca score...")
    gen_output = df[f"{result_keys['result']}"].tolist()
    gt_output = df.output.tolist()
    score = calculate_rouge_score(gen_output, gt_output)
    gen_token_len = df[result_keys["length"]].tolist()
    gen_token_per_sample = sum(gen_token_len) / len(gen_token_len)
    print(f"OpenOrca score: {score}, gen_token_per_sample: {gen_token_per_sample}")
    return score


@torch.no_grad()
def mlperf_rouge_infer(
    eval_data_dir: str | Path, num_eval_data: int, model: nn.Module, model_dir: str, main_device: str, batch_size: str
) -> pd.DataFrame:
    G_MAX_OUTPUT_SEQLEN = 1024

    df = pd.read_pickle(eval_data_dir)

    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(  # type: ignore
        model_dir,
        padding_side="left",
        trust_remote_code=True,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # gen parameter. We stop at 1024
    gen_kwargs = {
        "max_new_tokens": G_MAX_OUTPUT_SEQLEN,
        "do_sample": False,
        "temperature": None,
        "top_p": None,
    }

    # Start inference
    BS = int(batch_size)
    bidx = 0
    model.eval()

    input_tokens = []
    input_tokens_lens = []
    output_tokens = []
    output_tokens_lens = []
    output_texts = []

    tic = time.time()
    n_samples = min(len(df), num_eval_data)
    for idx in range(0, n_samples, BS):
        tac = time.time()
        print(f"Processing {idx}/{n_samples}, time: {tac - tic}s")
        sidx = idx
        eidx = min(sidx + BS, n_samples)

        # We use batch_encode_plus for batch inference.
        batch_texts = df["input"][sidx:eidx].tolist()
        batch_ids = tokenizer.batch_encode_plus(batch_texts, return_tensors="pt", padding=True)
        tok_input_length = batch_ids["attention_mask"].sum(axis=1).to(torch.int32).tolist()
        input_tokens_lens += tok_input_length
        tok_input_id = batch_ids["input_ids"].to(torch.int32).tolist()
        # Remove eos from the input id
        tok_input_id = [
            [element for element in sublist if element != tokenizer.eos_token_id] for sublist in tok_input_id
        ]
        input_tokens += tok_input_id

        batch_ids = batch_ids.to(main_device)
        _, length = batch_ids.input_ids.shape
        outputs = model.generate(**batch_ids, num_return_sequences=1, **gen_kwargs)

        output_ids = outputs[:, length:].cpu().tolist()
        output_tokens += output_ids

        # Filter out EOS
        id_filtered = [[num for num in sublist if num != tokenizer.eos_token_id] for sublist in output_ids]
        output_id_len = [len(out) for out in id_filtered]
        output_tokens_lens += output_id_len

        # Detokenizer
        output_msgs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        output_texts += output_msgs
        bidx += 1

    # Assemble the output
    output_df = df[: len(output_tokens)].copy()
    output_df["ref_output"] = output_texts
    output_df["tok_ref_output"] = output_tokens
    output_df["tok_ref_output_length"] = output_tokens_lens

    return output_df


def mlperf_rouge_eval(
    eval_data_dir: str | Path, num_eval_data: int, model: nn.Module, model_dir: str, main_device: str, batch_size: str
) -> None:
    nltk.download("punkt_tab")
    result_keys = {"result": "ref_output", "length": "tok_ref_output_length"}
    df = mlperf_rouge_infer(
        eval_data_dir=eval_data_dir,
        num_eval_data=num_eval_data,
        model=model,
        model_dir=model_dir,
        main_device=main_device,
        batch_size=batch_size,
    )
    evaluate_openorca(df, result_keys)
