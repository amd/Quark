#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

# Wrapper for the "onnx-validate" subcommand.
import argparse
import time
from typing import Any, List

import numpy as np
import onnxruntime
import torch
from datasets import load_dataset
from optimum.onnxruntime import ORTModelForCausalLM
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from tqdm import tqdm
from transformers import AutoTokenizer

from quark.experimental.cli import base_cli
from quark.onnx.operators.custom_ops import get_library_path

from .helper_utils import AverageMeter, load_dataloader


class TextDataset(Dataset):
    def __init__(self, tokenizer, block_size=512):
        testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = ""
        for i in testdata:
            text += i["text"]
        self.examples = []
        tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
        for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
            self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return torch.tensor(self.examples[item])


def load_and_cache_examples(args, tokenizer):
    dataset = TextDataset(
        tokenizer,
        block_size=args.block_size,
    )
    return dataset


def evaluate_wikitext_onnx(args, model, tokenizer):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    test_data = ""
    for i in testdata:
        test_data += i["text"]

    eval_dataset = load_and_cache_examples(args, tokenizer)

    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.per_gpu_eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        inputs, labels = (batch, batch)
        with torch.no_grad():
            # generate position_ids. Required by optimum >1.13.2
            position_ids = torch.arange(inputs.size(1), dtype=torch.long).unsqueeze(0)

            outputs = model(
                input_ids=inputs,
                attention_mask=inputs.new_ones(inputs.shape),
                position_ids=position_ids,  # required by newest optimum version
            )

            # Shift so that tokens < n predict n
            lm_logits = outputs[0]
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            lm_loss = loss_fct(shift_logits.float().view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    result = {"perplexity": perplexity}

    return result


def accuracy_np(output, target):
    max_indices = np.argsort(output, axis=1)[:, ::-1]
    top5 = 100 * np.equal(max_indices[:, :5], target[:, np.newaxis]).sum(axis=1).mean()
    top1 = 100 * np.equal(max_indices[:, 0], target).mean()
    return top1, top5


def evaluate_image_onnx(onnx_model_path, sess_options, providers, data_loader, print_freq):
    session = onnxruntime.InferenceSession(onnx_model_path, sess_options, providers=providers)
    input_name = session.get_inputs()[0].name

    batch_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    for i, (input, target) in enumerate(data_loader):
        # run the net and return prediction
        output = session.run([], {input_name: input.data.numpy()})
        output = output[0]

        # measure accuracy and record loss
        prec1, prec5 = accuracy_np(output, target.numpy())
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % print_freq == 0:
            print(
                f"Test: [{i}/{len(data_loader)}]\t"
                f"Time {batch_time.val:.3f} ({batch_time.avg:.3f}, {input.size(0) / batch_time.avg:.3f}/s, "
                f"{100 * batch_time.avg / input.size(0):.3f} ms/sample) \t"
                f"Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t"
                f"Prec@5 {top5.val:.3f} ({top5.avg:.3f})"
            )

    return top1, top5


def llm_evaluation(args: Any, device: str, providers: list[str]):
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.register_custom_ops_library(get_library_path(device))

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        do_lower_case=False,
        cache_dir=None,
    )

    tokenizer.add_bos_token = False
    if tokenizer.pad_token != "<unk>":
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.block_size <= 0:
        args.block_size = (
            tokenizer.max_len_single_sentence
        )  # Our input block size will be the max possible for the model

    results = {}
    global_step = ""

    model = ORTModelForCausalLM.from_pretrained(
        args.model_path, providers=providers, use_cache=False, use_io_binding=False, session_options=sess_options
    )
    result = evaluate_wikitext_onnx(args, model, tokenizer)
    result = dict((k + f"_{global_step}", v) for k, v in result.items())
    results.update(result)

    print(results)


def image_evaluation(args: Any, device: str, providers: list[str]):
    sess_options = onnxruntime.SessionOptions()
    sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.register_custom_ops_library(get_library_path(device))

    dataloader = load_dataloader(
        image_folder=args.val_data, model_name=args.model_name, batch_size=args.per_gpu_eval_batch_size
    )

    f_top1, f_top5 = evaluate_image_onnx(args.model_path, sess_options, providers, dataloader, 1)
    print(f" * Prec@1 {f_top1.avg:.3f} ({100 - f_top1.avg:.3f}) Prec@5 {f_top5.avg:.3f} ({100.0 - f_top5.avg:.3f})")


class ONNXValidate_CLI(base_cli.BaseQuarkCLICommand):
    """
    This class is the Quark CLI onnx-validate subcommand.

    This command helps evaluate the model against the given validation data.

    Design
    ------
    Like the onnx-ptq subcommand this tries to reconcile several different duplicate onnx_validate.py
    scripts in the examples/onnx folder. It does this by assuming the model is a language model first, trying to initialize the tokenizer, if that fails it falls back to the assumption that its a vision model.
    """

    @staticmethod
    def register_subcommand(parser: argparse.ArgumentParser):
        parser.add_argument("--val_data", help="Data to validate against")
        parser.add_argument(
            "--model_path",
            type=str,
            required=True,
            help="The model checkpoint or name for weights initialization.",
        )
        parser.add_argument(
            "--model_name",
            type=str,
            help="The name of the model.",
        )
        parser.add_argument(
            "--block_size",
            default=1024,
            type=int,
            help="Optional input sequence length after tokenization."
            "The training dataset will be truncated in block of this size for training."
            "Default to the model max input length for single sentence inputs (take into account special tokens).",
        )
        parser.add_argument(
            "--per_gpu_eval_batch_size", default=4, type=int, help="Batch size per GPU/CPU for evaluation."
        )
        parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    def run(self):
        args = self.args

        if args.no_cuda:
            device = "CPU"
            providers = ["CPUExecutionProvider"]
        else:
            if "ROCMExecutionProvider" in onnxruntime.get_available_providers():
                device = "ROCM"
                providers = ["ROCMExecutionProvider"]
            elif "CUDAExecutionProvider" in onnxruntime.get_available_providers():
                device = "CUDA"
                providers = ["CUDAExecutionProvider"]

        # try to initialize tokenizer, it that fails it's likely a vision model
        try:
            llm_evaluation(args, device, providers)
        except ValueError:
            image_evaluation(args, device, providers)
