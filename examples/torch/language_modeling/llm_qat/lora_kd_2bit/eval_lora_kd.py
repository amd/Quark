#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Evaluate a LoRA+KD model by loading the PTQ base + LoRA adapters separately.

Usage:
  python eval_lora_kd.py \
    --base_model_dir /tmp/qwq_32b_lloyd_max_g64 \
    --adapter_dir /tmp/qwq_32b_lora_kd_v13/lora_adapters \
    --multi_gpu \
    --eval_ppl \
    --eval_tasks arc_challenge,hellaswag,winogrande,boolq,openbookqa

This avoids storing the full merged model (~62GB) and instead keeps only
the small LoRA adapter (~1-3GB) plus the shared PTQ base.
"""

import argparse
import math
import os

import torch
from peft import PeftModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def eval_ppl(model, tokenizer, seq_len=1024, max_samples=256):
    """WikiText-2 validation perplexity."""
    from datasets import load_dataset

    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(raw["text"])
    enc = tokenizer(text, return_attention_mask=False, return_tensors="pt")
    input_ids = enc["input_ids"].squeeze(0)
    total_len = (input_ids.shape[0] // seq_len) * seq_len
    chunks = input_ids[:total_len].view(-1, seq_len)
    if max_samples and len(chunks) > max_samples:
        chunks = chunks[:max_samples]

    model.eval()
    losses = []
    with torch.no_grad():
        for i in range(0, len(chunks), 4):
            batch = chunks[i : i + 4]
            if not hasattr(model, "hf_device_map"):
                batch = batch.to(model.device)
            loss = model(input_ids=batch, labels=batch).loss
            losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))


def main():
    parser = argparse.ArgumentParser(description="Evaluate PTQ base + LoRA adapters")
    parser.add_argument(
        "--base_model_dir", type=str, required=True, help="Path to PTQ base model (e.g. /tmp/qwq_32b_lloyd_max_g64)"
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="Path to LoRA adapter dir (contains adapter_model.safetensors + adapter_config.json)",
    )
    parser.add_argument(
        "--output_dir", type=str, default="", help="Directory to save lm_eval results (default: adapter_dir/eval_*)"
    )
    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--eval_ppl", action="store_true", help="Evaluate WikiText-2 PPL")
    parser.add_argument(
        "--eval_tasks",
        type=str,
        default="",
        help="Comma-separated lm_eval tasks (e.g. arc_challenge,hellaswag,mmlu,gsm8k)",
    )
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--batch_size", type=str, default="auto:4")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--max_ppl_samples", type=int, default=256)
    parser.add_argument(
        "--merge", action="store_true", help="Merge LoRA into base weights before eval (slightly faster inference)"
    )
    parser.add_argument("--save_merged", type=str, default="", help="If set, save the merged model to this path")
    args = parser.parse_args()

    if not args.output_dir:
        args.output_dir = os.path.dirname(args.adapter_dir)

    device_map = "auto" if args.multi_gpu else "cuda"

    print("=" * 70)
    print("Loading PTQ base model + LoRA adapters")
    print(f"  Base:     {args.base_model_dir}")
    print(f"  Adapters: {args.adapter_dir}")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    # The base is a fake-quantized PTQ export: the 2-bit error is already baked into
    # plain bf16 weights, and its config.json carries an (informational) quark
    # quantization_config. Loading through the HF quark quantizer would try to rebuild
    # the quantizer and fail for some schemes (e.g. bfp16). Drop the quant config and
    # load the bf16 weights directly.
    base_cfg = AutoConfig.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if hasattr(base_cfg, "quantization_config"):
        del base_cfg.quantization_config
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        config=base_cfg,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )

    print("Applying LoRA adapters...")
    model = PeftModel.from_pretrained(base_model, args.adapter_dir, autocast_adapter_dtype=False)

    if args.merge:
        print("Merging LoRA into base weights...")
        model = model.merge_and_unload()

    if args.save_merged:
        print(f"Saving merged model to {args.save_merged}...")
        merged = model if args.merge else model.merge_and_unload()
        merged.save_pretrained(args.save_merged)
        tokenizer.save_pretrained(args.save_merged)
        print(f"Merged model saved to {args.save_merged}")
        if not args.merge:
            model = PeftModel.from_pretrained(base_model, args.adapter_dir)

    model.eval()
    print(f"Model loaded. merge={args.merge}")

    if args.eval_ppl:
        print("\nEvaluating WikiText-2 PPL...")
        ppl = eval_ppl(model, tokenizer, args.seq_len, args.max_ppl_samples)
        print(f"  >>> WikiText-2 PPL: {ppl:.4f}")

    if args.eval_tasks:
        task_names = [t.strip() for t in args.eval_tasks.split(",")]
        print(f"\nRunning lm_eval benchmarks: {task_names}")

        from lm_eval.evaluator import simple_evaluate
        from lm_eval.models.huggingface import HFLM
        from lm_eval.utils import make_table

        eval_model = model.merge_and_unload() if not args.merge else model
        eval_model.eval()

        lm = HFLM(
            pretrained=eval_model,
            tokenizer=tokenizer,
            batch_size=args.batch_size,
        )

        gen_kwargs = {}
        if "gsm8k" in args.eval_tasks:
            gen_kwargs["max_gen_toks"] = 1024

        results = simple_evaluate(
            model=lm,
            tasks=task_names,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            gen_kwargs=gen_kwargs if gen_kwargs else None,
        )

        print("\nBenchmark Results:")
        print(make_table(results))

        import json

        results_path = os.path.join(args.output_dir, "benchmark_results.json")
        with open(results_path, "w") as f:
            f.write(json.dumps(results["results"], indent=2, default=str))
        print(f"Results saved to {results_path}")


if __name__ == "__main__":
    main()
