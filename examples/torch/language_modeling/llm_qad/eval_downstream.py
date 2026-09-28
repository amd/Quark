#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Evaluate a model on downstream tasks using lm-evaluation-harness."""

import argparse
import os

import torch
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--tasks", required=True, help="Comma-separated task names")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max_gen_toks", type=int, default=None, help="Override max generation tokens for generate_until tasks"
    )
    args = parser.parse_args()

    print(f"[INFO] Loading model from {args.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    model.eval()

    task_manager = TaskManager()
    task_list = args.tasks.split(",")
    task_names = task_manager.match_tasks(task_list)
    print(f"[INFO] Tasks: {task_names}")

    lm_kwargs = dict(pretrained=model, tokenizer=tokenizer, batch_size=args.batch_size)
    if args.max_gen_toks is not None:
        lm_kwargs["max_gen_toks"] = args.max_gen_toks
    lm = HFLM(**lm_kwargs)

    eval_kwargs = dict(
        model=lm,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        task_manager=task_manager,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    if args.max_gen_toks is not None:
        eval_kwargs["gen_kwargs"] = f"max_gen_toks={args.max_gen_toks}"
    results = simple_evaluate(**eval_kwargs)

    if results:
        print(make_table(results))


if __name__ == "__main__":
    main()
