#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Build a self-loading STRUCTURED "newexport" checkpoint directly from:

  1. the 2-bit TwoBitScalar student  (non-quantized weights: embed / norms / lm_head)
  2. the TwoBitScalar sidecar        (--twobitscalar_dump_sidecar): the rotated-domain
     4-level indices, per-group scales, and the SRHT/AWQ params
  3. a trained LoRA adapter          (peft adapter_model.safetensors)

Output layout matches `modeling_phi4_structured_lora.Phi3StructuredLoraForCausalLM`
(the same intermediate the shipped `-v16-newexport` uses), so it can be fed straight
into `convert_v16_to_factored.py` to produce the compact factored export.

Per quantized projection (160 for Phi-4 / QwQ):
  pre_linear.awq_s_vec   fp16  [in]          (from sidecar awq_scale)
  pre_linear.srht_perm   int16 [in]          (from sidecar srht_perm)
  pre_linear.srht_sign   int8  [in]          (from sidecar srht_signs, ±1)
  linear.packed_levels   uint8 [out, in/4]   (packed from sidecar W_q_levels)
  linear.group_scale     fp16  [out, in/64]  (from sidecar group_scale)
  lora_A.weight          fp16  [r, in]        (from adapter)
  lora_B.weight          fp16  [out, r]       (from adapter)

Everything is derived from PTQ outputs + the adapter — no dependence on any
pre-built newexport artifact.

Usage:
  python3 export_structured_newexport.py \
      --student_dir /tmp/phi4_lloyd_max_g64 \
      --sidecar_dir /tmp/phi4_linear_g64_sidecar \
      --adapter     /tmp/phi4_lorakd_qadtr_fused/checkpoint-8000/adapter_model.safetensors \
      --output_dir  /tmp/phi4_structured_newexport \
      --lora_rank 64 --lora_alpha 128 --group_size 64
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

TARGET_SUFFIXES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
DEFAULT_LEVELS = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]
# Affine encoding of the uniform Lloyd-Max codebook (step 2/3).
LEVEL_ZERO_POINT = 1.5
LEVEL_SCALE = 2.0 / 3.0


def pack_int2(idx: torch.Tensor) -> torch.Tensor:
    """LSB-first pack of 4 uint2 indices per byte (matches modeling unpack_int2)."""
    idx = idx.to(torch.uint8) & 0x3
    a = idx[..., 0::4]
    b = idx[..., 1::4] << 2
    c = idx[..., 2::4] << 4
    d = idx[..., 3::4] << 6
    return (a | b | c | d).contiguous()


def levels_to_idx(w_q: torch.Tensor) -> torch.Tensor:
    """Map the 4 discrete level values {-1,-1/3,1/3,1} to indices {0,1,2,3}."""
    return (w_q > -0.5).to(torch.uint8) + (w_q > 0.0).to(torch.uint8) + (w_q > 0.5).to(torch.uint8)


def load_sidecar(sidecar_dir: str):
    fields = {}
    for name in ("W_q_levels", "group_scale", "awq_scale", "srht_perm", "srht_signs"):
        path = os.path.join(sidecar_dir, f"{name}.safetensors")
        with safe_open(path, framework="pt") as f:
            fields[name] = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118 (safe_open is not iterable)
    return fields


def load_student_nonquant(student_dir: str):
    """All student tensors that are NOT a target-projection .weight."""
    out = {}
    for shard in sorted(glob.glob(os.path.join(student_dir, "model*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():  # noqa: SIM118 (safe_open is not iterable)
                if any(k.endswith(s + ".weight") for s in TARGET_SUFFIXES):
                    continue
                out[k] = f.get_tensor(k)
    return out


def load_adapter_lora(adapter_path: str):
    """Strip peft's `base_model.model.` prefix → canonical module names."""
    out = {}
    with safe_open(adapter_path, framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 (safe_open is not iterable)
            if ".lora_A." not in k and ".lora_B." not in k:
                continue
            nk = k.replace("base_model.model.", "")
            # peft may insert an adapter name (".default.") — normalize it out.
            nk = nk.replace(".lora_A.default.", ".lora_A.").replace(".lora_B.default.", ".lora_B.")
            out[nk] = f.get_tensor(k)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--student_dir", required=True)
    ap.add_argument("--sidecar_dir", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--group_size", type=int, default=64)
    ap.add_argument("--arch_class", default="Phi3StructuredLoraForCausalLM")
    ap.add_argument("--modeling_file", default="modeling_phi4_structured_lora.py")
    ap.add_argument(
        "--quark_pack",
        action="store_true",
        help="Store the 2-bit weight via Quark's Pack_uint2 + weight_scale/weight_zero_point "
        "(float zp=1.5) so the weight half dequantizes through Quark's own kernels. "
        "Sets the Quark-backed modeling file/arch class by default.",
    )
    args = ap.parse_args()

    if args.quark_pack:
        from quark.torch.utils.pack import Pack_uint2

        _packer = Pack_uint2(qscheme=None, dtype="uint2")
        if args.arch_class == "Phi3StructuredLoraForCausalLM":
            args.arch_class = "Phi3StructuredQuarkForCausalLM"
        if args.modeling_file == "modeling_phi4_structured_lora.py":
            args.modeling_file = "modeling_phi4_structured_quark.py"

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[load] sidecar: {args.sidecar_dir}")
    side = load_sidecar(args.sidecar_dir)
    stems = sorted(side["W_q_levels"].keys())
    print(f"  {len(stems)} quantized projections")

    print(f"[load] student non-quant: {args.student_dir}")
    state = load_student_nonquant(args.student_dir)
    print(f"  {len(state)} non-quant tensors")

    print(f"[load] adapter: {args.adapter}")
    lora = load_adapter_lora(args.adapter)
    print(f"  {len(lora)} lora tensors")

    print("[build] structured per-projection tensors")
    for stem in tqdm(stems, unit="proj"):
        w_q = side["W_q_levels"][stem]  # [out, in] fp32, values in the 4 levels
        idx = levels_to_idx(w_q)
        gscale = side["group_scale"][stem]
        if args.quark_pack:
            # Quark-native weight: Pack_uint2 + affine (zp=1.5, scale=2/3*group_scale).
            n_groups = gscale.shape[1]
            state[f"{stem}.linear.packed_levels"] = _packer.pack(idx.to(torch.uint8))
            state[f"{stem}.linear.weight_scale"] = (LEVEL_SCALE * gscale).to(torch.float16)
            state[f"{stem}.linear.weight_zero_point"] = torch.full(
                (gscale.shape[0], n_groups), LEVEL_ZERO_POINT, dtype=torch.float16
            )
        else:
            state[f"{stem}.linear.packed_levels"] = pack_int2(idx)
            state[f"{stem}.linear.group_scale"] = gscale.to(torch.float16)
        state[f"{stem}.pre_linear.awq_s_vec"] = side["awq_scale"][stem].to(torch.float16)
        state[f"{stem}.pre_linear.srht_perm"] = side["srht_perm"][stem].to(torch.int16)
        state[f"{stem}.pre_linear.srht_sign"] = side["srht_signs"][stem].sign().to(torch.int8)
        for br in ("lora_A", "lora_B"):
            lk = f"{stem}.{br}.weight"
            if lk not in lora:
                raise KeyError(f"missing {lk} in adapter (targets must include the fused projections)")
            state[lk] = lora[lk].to(torch.float16)

    out_path = os.path.join(args.output_dir, "model.safetensors")
    print(f"[save] {out_path}  ({len(state)} tensors)")
    save_file(state, out_path, metadata={"format": "pt"})

    # config.json: reuse student config, add self-load + srht_awq_lora meta
    with open(os.path.join(args.student_dir, "config.json")) as f:
        cfg = json.load(f)
    cfg["architectures"] = [args.arch_class]
    cfg["auto_map"] = {
        "AutoModelForCausalLM": f"{args.modeling_file[:-3]}.{args.arch_class}",
    }
    # Self-loads via the custom modeling file (auto_map), not the HF quark quantizer.
    # Drop the student's stale PTQ quantization_config so from_pretrained doesn't try
    # to rebuild a quantizer (e.g. bfp16, which isn't loadable) and fail.
    cfg.pop("quantization_config", None)
    cfg["srht_awq_lora"] = {
        "rank": args.lora_rank,
        "alpha": args.lora_alpha,
        "group_size": args.group_size,
        "target_suffixes": list(TARGET_SUFFIXES),
        "levels": DEFAULT_LEVELS,
    }
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    shutil.copy(os.path.join(REPO_ROOT, args.modeling_file), os.path.join(args.output_dir, args.modeling_file))
    for fn in (
        "generation_config.json",
        "chat_template.jinja",
        "tokenizer.json",
        "tokenizer_config.json",
        "tokenizer.model",
        "special_tokens_map.json",
    ):
        sp = os.path.join(args.student_dir, fn)
        if os.path.exists(sp):
            shutil.copy(sp, os.path.join(args.output_dir, fn))

    meta = {
        "scheme": "structured newexport built from student + sidecar + LoRA adapter",
        "student_dir": args.student_dir,
        "sidecar_dir": args.sidecar_dir,
        "adapter": args.adapter,
        "n_projections": len(stems),
    }
    with open(os.path.join(args.output_dir, "srht_awq_structured_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[done] -> {args.output_dir}")


if __name__ == "__main__":
    main()
