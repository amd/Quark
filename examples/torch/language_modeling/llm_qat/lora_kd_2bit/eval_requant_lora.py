#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Evaluate the effect of re-quantizing LoRA+KD v14 base weights to {-1,-1/3,1/3,1} * scale.

Plan:
  1. Load PTQ base model  (W_ptq)   — /tmp/phi4_lloyd_max_g64/
  2. Load v14 merged model (W_v14)  — /tmp/phi4_lora_kd_v14/
  3. For each proj linear:
       lora_delta = W_v14 - W_ptq
       W_requant  = quantize_to_4level(W_ptq, group_size=64, method=...)
       W_final    = W_requant + lora_delta
  4. Measure WikiText-2 PPL for:
       (a) Original v14 merged model
       (b) Re-quantized base (RTN scales) + LoRA delta
       (c) Re-quantized base (optimal MSE scales) + LoRA delta
       (d) Re-quantized base (optimal scales + adaptive rounding) + LoRA delta
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer

_DIR = Path(__file__).resolve().parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))


# ── 4-level codebook: {-1, -1/3, 1/3, 1} ──────────────────────────────────

LEVELS = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])


def _snap_rtn(u: torch.Tensor) -> torch.Tensor:
    """Round-to-nearest snap to {-1, -1/3, 1/3, 1}."""
    u_abs = u.abs()
    u_q_abs = torch.where(u_abs < 2.0 / 3.0, 1.0 / 3.0, 1.0)
    return u.sign() * u_q_abs


def quantize_group_rtn(w_group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """RTN quantization: max-abs scale, round-to-nearest.
    Returns (w_quantized, scale)."""
    scale = w_group.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    u = (w_group / scale).clamp(-1.5, 1.5)
    u_q = _snap_rtn(u)
    return u_q * scale, scale


def quantize_group_optimal_scale(w_group: torch.Tensor, n_steps: int = 200) -> tuple[torch.Tensor, torch.Tensor]:
    """Find per-group scale that minimizes MSE, still using RTN rounding.

    Grid search around max-abs scale: try scales from 0.5*max_abs to 1.5*max_abs
    and pick the one with lowest MSE per group.
    """
    max_abs = w_group.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    best_mse = torch.full(max_abs.shape, float("inf"), device=w_group.device)
    best_scale = max_abs.clone()
    best_wq = torch.zeros_like(w_group)

    for i in range(n_steps):
        ratio = 0.5 + 1.5 * i / max(n_steps - 1, 1)
        trial_scale = max_abs * ratio
        u = (w_group / trial_scale).clamp(-1.5, 1.5)
        u_q = _snap_rtn(u)
        wq = u_q * trial_scale
        mse = ((wq - w_group) ** 2).mean(dim=-1, keepdim=True)
        improved = mse < best_mse
        best_mse = torch.where(improved, mse, best_mse)
        best_scale = torch.where(improved, trial_scale, best_scale)
        best_wq = torch.where(improved.expand_as(wq), wq, best_wq)

    return best_wq, best_scale


def quantize_group_adaptive_rounding(
    w_group: torch.Tensor,
    n_scale_steps: int = 200,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimal scale + adaptive rounding (per-element best direction).

    For each element, after choosing the optimal scale, try both floor and ceil
    from the 4-level codebook and pick whichever is closer to the original weight.
    This is better than RTN because the scale shift can make RTN suboptimal.
    """
    max_abs = w_group.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    best_mse = torch.full(max_abs.shape, float("inf"), device=w_group.device)
    best_scale = max_abs.clone()
    best_wq = torch.zeros_like(w_group)

    _B1, _B2, _B3 = -2.0 / 3.0, 0.0, 2.0 / 3.0

    for i in range(n_scale_steps):
        ratio = 0.5 + 1.5 * i / max(n_scale_steps - 1, 1)
        trial_scale = max_abs * ratio
        u = (w_group / trial_scale).clamp(-1.5, 1.5)

        # floor and ceil levels
        fl = torch.where(u < _B1, -1.0, torch.where(u < _B2, -1.0 / 3.0, torch.where(u < _B3, 1.0 / 3.0, 1.0)))
        cl = torch.where(u < _B1, -1.0 / 3.0, torch.where(u < _B2, 1.0 / 3.0, torch.where(u < _B3, 1.0, 1.0)))

        # pick floor or ceil per-element based on which is closer
        wq_fl = fl * trial_scale
        wq_cl = cl * trial_scale
        err_fl = (wq_fl - w_group).abs()
        err_cl = (wq_cl - w_group).abs()
        wq = torch.where(err_fl <= err_cl, wq_fl, wq_cl)

        mse = ((wq - w_group) ** 2).mean(dim=-1, keepdim=True)
        improved = mse < best_mse
        best_mse = torch.where(improved, mse, best_mse)
        best_scale = torch.where(improved, trial_scale, best_scale)
        best_wq = torch.where(improved.expand_as(wq), wq, best_wq)

    return best_wq, best_scale


def requantize_weight(w: torch.Tensor, group_size: int, method: str = "rtn") -> torch.Tensor:
    """Re-quantize a 2D weight to {-1,-1/3,1/3,1}*scale per group.

    Methods: 'rtn', 'optimal_scale', 'adaptive_rounding'
    """
    o, i = w.shape
    g = group_size
    assert i % g == 0, f"in_features {i} not divisible by group_size {g}"
    n_groups = i // g
    w_grouped = w.float().view(o, n_groups, g)

    if method == "rtn":
        wq, _ = quantize_group_rtn(w_grouped)
    elif method == "optimal_scale":
        wq, _ = quantize_group_optimal_scale(w_grouped)
    elif method == "adaptive_rounding":
        wq, _ = quantize_group_adaptive_rounding(w_grouped)
    else:
        raise ValueError(f"Unknown method: {method}")

    return wq.view(o, i).to(w.dtype)


def verify_on_grid(w: torch.Tensor, group_size: int) -> dict:
    """Check what fraction of weights lie exactly on {-1,-1/3,1/3,1}*scale."""
    o, i = w.shape
    g = group_size
    w_grouped = w.float().view(o, i // g, g)
    scale = w_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    u = w_grouped / scale
    # Check if each normalized value is close to one of the 4 levels
    levels = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0], device=w.device)
    dists = (u.unsqueeze(-1) - levels.view(1, 1, 1, -1)).abs()
    min_dist = dists.min(dim=-1).values
    on_grid = (min_dist < 0.02).float().mean().item()
    avg_dist = min_dist.mean().item()
    return {"on_grid_fraction": on_grid, "avg_dist_to_grid": avg_dist}


# ── PPL evaluation ──────────────────────────────────────────────────────────


@torch.no_grad()
def eval_ppl(model, tokenizer, seq_len: int = 1024, max_samples: int = 256) -> float:
    """Evaluate WikiText-2 validation perplexity."""
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
    for i in range(0, len(chunks), 4):
        batch = chunks[i : i + 4].to(model.device)
        loss = model(input_ids=batch, labels=batch).loss
        losses.append(loss.item())
    return math.exp(sum(losses) / len(losses))


# ── Main ────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ptq_dir", type=str, default="/tmp/phi4_lloyd_max_g64")
    parser.add_argument("--v14_dir", type=str, default="/tmp/phi4_lora_kd_v14")
    parser.add_argument("--group_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--max_eval", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--multi_gpu", action="store_true", help="Use device_map=auto for model loading")
    parser.add_argument(
        "--methods",
        type=str,
        default="rtn,optimal_scale,adaptive_rounding",
        help="Comma-separated re-quantization methods to try",
    )
    args = parser.parse_args()

    import gc

    methods = [m.strip() for m in args.methods.split(",")]
    device_map = "auto" if args.multi_gpu else args.device

    print("=" * 70)
    print("LoRA+KD v14: Re-quantize base weights to {-1,-1/3,1/3,1} * scale")
    print("=" * 70)

    # ── Load safetensors directly for weight surgery ──
    print("\n[1] Loading weight tensors...")
    t0 = time.time()
    ptq_st = load_file(f"{args.ptq_dir}/model.safetensors")
    v14_st = load_file(f"{args.v14_dir}/model.safetensors")
    print(f"    Loaded in {time.time() - t0:.1f}s")

    proj_keys = sorted(k for k in v14_st if "proj" in k and "weight" in k)
    print(f"    {len(proj_keys)} proj weight tensors to process")

    # ── Analyze the PTQ base weights: are they already on the grid? ──
    print("\n[2] Checking how close PTQ base weights are to the {-1,-1/3,1/3,1} grid...")
    for k in proj_keys[:4]:
        stats = verify_on_grid(ptq_st[k], args.group_size)
        print(f"    {k}: {stats['on_grid_fraction'] * 100:.1f}% on grid, avg dist={stats['avg_dist_to_grid']:.5f}")

    # ── Compute LoRA deltas ──
    print("\n[3] Computing LoRA deltas (W_v14 - W_ptq)...")
    lora_deltas = {}
    total_delta_norm = 0.0
    total_base_norm = 0.0
    for k in proj_keys:
        delta = v14_st[k].float() - ptq_st[k].float()
        lora_deltas[k] = delta
        total_delta_norm += delta.norm().item() ** 2
        total_base_norm += ptq_st[k].float().norm().item() ** 2
    ratio = (total_delta_norm**0.5) / (total_base_norm**0.5)
    print(f"    ||LoRA delta|| / ||W_ptq|| = {ratio:.6f} ({ratio * 100:.3f}%)")

    # ── Eval original v14 PPL first, then free the model ──
    print("\n[4] Evaluating original v14 model PPL...")
    tokenizer = AutoTokenizer.from_pretrained(args.v14_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.v14_dir,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    ppl_orig = eval_ppl(model, tokenizer, args.seq_len, args.max_eval)
    print(f"    >>> Original v14 PPL: {ppl_orig:.4f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Re-quantize and evaluate each method ──
    results = {}
    for method in methods:
        print(f"\n[5-{method}] Re-quantizing PTQ base weights with method='{method}'...")
        t0 = time.time()

        total_mse = 0.0
        total_elems = 0
        modified_weights = {}

        for idx, k in enumerate(proj_keys):
            w_ptq = ptq_st[k].to(args.device)
            delta = lora_deltas[k].to(args.device)

            w_requant = requantize_weight(w_ptq, args.group_size, method=method)
            mse = ((w_ptq.float() - w_requant.float()) ** 2).mean().item()
            total_mse += mse * w_ptq.numel()
            total_elems += w_ptq.numel()

            modified_weights[k] = (w_requant.float() + delta.float()).to(ptq_st[k].dtype).cpu()
            del w_ptq, delta, w_requant
            torch.cuda.empty_cache()

            if idx < 4 or idx % 40 == 0:
                print(f"    [{idx + 1}/{len(proj_keys)}] {k}: MSE={mse:.2e}")

        avg_mse = total_mse / total_elems
        print(f"    Done in {time.time() - t0:.1f}s, Avg MSE(PTQ vs {method}): {avg_mse:.2e}")

        # Load model, inject weights, evaluate
        print("    Loading model for PPL eval...")
        model = AutoModelForCausalLM.from_pretrained(
            args.v14_dir,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
        sd = model.state_dict()
        for k in proj_keys:
            if k in sd:
                sd[k] = modified_weights[k].to(sd[k].dtype)
        model.load_state_dict(sd)
        del sd, modified_weights
        gc.collect()

        ppl = eval_ppl(model, tokenizer, args.seq_len, args.max_eval)
        print(f"    >>> {method} PPL: {ppl:.4f}  (delta vs v14: {ppl - ppl_orig:+.4f})")
        results[method] = ppl
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # ── Bonus: re-quantized base WITHOUT LoRA delta ──
    print("\n[6] Bonus: Re-quantized base WITHOUT LoRA (adaptive_rounding)...")
    modified_weights = {}
    for k in proj_keys:
        w_ptq = ptq_st[k].to(args.device)
        w_requant = requantize_weight(w_ptq, args.group_size, method="adaptive_rounding")
        modified_weights[k] = w_requant.to(ptq_st[k].dtype).cpu()
        del w_ptq, w_requant
        torch.cuda.empty_cache()

    model = AutoModelForCausalLM.from_pretrained(
        args.v14_dir,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    sd = model.state_dict()
    for k in proj_keys:
        if k in sd:
            sd[k] = modified_weights[k].to(sd[k].dtype)
    model.load_state_dict(sd)
    del sd, modified_weights
    gc.collect()

    ppl_no_lora = eval_ppl(model, tokenizer, args.seq_len, args.max_eval)
    print(f"    >>> Requant base (no LoRA) PPL: {ppl_no_lora:.4f}")
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # ── Summary ──
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Original v14 (merged):                     PPL = {ppl_orig:.4f}")
    for method, ppl in results.items():
        label = f"  Requant+LoRA ({method}):"
        print(f"{label:<50s} PPL = {ppl:.4f}  ({ppl - ppl_orig:+.4f})")
    print(f"  Requant base (no LoRA, adaptive rounding):  PPL = {ppl_no_lora:.4f}")
    print("  (Baseline QAD v2 RTN from plan:             PPL ~ 14.37)")
    print("=" * 70)


if __name__ == "__main__":
    main()
