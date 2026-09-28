#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""PPL + downstream for the Phi-4 factored LoRA+KD export WITH INT16 activations.

The factored model computes, per quantized projection:
    x_rot = (x / awq_s_vec) @ shared_R     # rotation
    y     = Quantized2BitLinear(x_rot) + lora_B(lora_A(x)) * scaling

For NPU deployment the activation feeding the 2-bit weight matmul is INT16.
So we per-token (per last-dim row) symmetric INT16 fake-quant the INPUT to every
Quantized2BitLinear (i.e. x_rot). With --act_bits 0 this is the bf16 baseline.

Optionally (--quant_lora) also fake-quant the raw input feeding lora_A.

Usage:
  CUDA_VISIBLE_DEVICES=0 python3 eval_factored_int16_acts.py \
      --model_dir /amd_models/phi-4-2bit-lora-kd-linear-v16-factored \
      --act_bits 16 --tasks arc_challenge,hellaswag,winogrande,boolq,openbookqa
"""

from __future__ import annotations

import argparse
import math
import os

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def fake_quant_act_per_token(x: torch.Tensor, n_bits: int) -> torch.Tensor:
    if n_bits <= 0:
        return x
    qmax = (1 << (n_bits - 1)) - 1
    od = x.dtype
    xf = x.float()
    amax = xf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    return (torch.round(xf / scale).clamp(-qmax, qmax) * scale).to(od)


def patch_int16_acts(model, n_bits: int, quant_lora: bool):
    """Monkeypatch Quantized2BitLinear.forward (and optionally lora_A input) to
    apply per-token INT n_bits fake-quant. Returns number of patched classes."""
    # Find the Quantized2BitLinear class from a live instance.
    qcls = None
    fcls = None
    for _, m in model.named_modules():
        cn = type(m).__name__
        if cn == "Quantized2BitLinear" and qcls is None:
            qcls = type(m)
        if cn == "FactoredAWQRotateLinearWithLoRA" and fcls is None:
            fcls = type(m)
    if qcls is None:
        raise RuntimeError("Quantized2BitLinear not found — is this the factored export?")

    if not hasattr(qcls, "_orig_forward"):
        qcls._orig_forward = qcls.forward
    nb = n_bits

    def q_forward(self, x):
        x = fake_quant_act_per_token(x, nb)  # x == x_rot here (input to 2-bit matmul)
        return qcls._orig_forward(self, x)

    qcls.forward = q_forward

    if quant_lora and fcls is not None:
        if not hasattr(fcls, "_orig_forward"):
            fcls._orig_forward = fcls.forward

        def f_forward(self, x):
            x = fake_quant_act_per_token(x, nb)  # raw input feeding both branches
            return fcls._orig_forward(self, x)

        fcls.forward = f_forward
    return qcls.__name__


@torch.no_grad()
def eval_ppl(model, tok, seq_len=1024, max_samples=256):
    raw = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    text = "\n\n".join(raw["text"])
    ids = tok(text, return_attention_mask=False, return_tensors="pt")["input_ids"].squeeze(0)
    total = (ids.shape[0] // seq_len) * seq_len
    chunks = ids[:total].view(-1, seq_len)[:max_samples]
    dev = next(model.parameters()).device
    losses = []
    for i in range(0, len(chunks), 4):
        b = chunks[i : i + 4].to(dev)
        losses.append(model(input_ids=b, labels=b).loss.item())
    return math.exp(sum(losses) / len(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="/amd_models/phi-4-2bit-lora-kd-linear-v16-factored")
    ap.add_argument("--act_bits", type=int, default=16, help="0 = bf16 baseline; 16 = INT16 acts")
    ap.add_argument("--quant_lora", action="store_true", help="also int-quant the raw LoRA-path input")
    ap.add_argument("--tasks", default="arc_challenge,hellaswag,winogrande,boolq,openbookqa")
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--ppl_only", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    print(f"[load] {args.model_dir}")
    tok = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(args.device)
        .eval()
    )

    if args.act_bits > 0:
        cls = patch_int16_acts(model, args.act_bits, args.quant_lora)
        print(
            f"[int16] per-token INT{args.act_bits} fake-quant on input to {cls} "
            f"(2-bit matmul activation){' + LoRA-path input' if args.quant_lora else ''}"
        )
    else:
        print("[baseline] no activation quant (bf16 acts)")

    ppl = eval_ppl(model, tok)
    print(f"\nWikiText-2 PPL (act_bits={args.act_bits}) = {ppl:.6f}")

    if args.ppl_only:
        return

    from lm_eval.evaluator import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from lm_eval.utils import make_table

    tm = TaskManager()
    names = tm.match_tasks(args.tasks.split(","))
    lm = HFLM(pretrained=model, tokenizer=tok, batch_size=args.batch_size)
    res = simple_evaluate(
        model=lm,
        tasks=names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        task_manager=tm,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    print(make_table(res))


if __name__ == "__main__":
    main()
