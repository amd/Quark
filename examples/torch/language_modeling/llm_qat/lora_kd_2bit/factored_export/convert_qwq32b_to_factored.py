#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Convert the QwQ-32B dense LoRA+KD export -> FACTORED form (shared rotations).

Mirror of convert_v16_to_factored.py for QwQ-32B:
  - Drops the 448 per-projection dense <proj>.pre_linear.weight ([in,in] fp16).
  - Adds 2 shared rotation matrices at top level (in=5120 and in=27648).
  - Adds per-projection <proj>.awq_s_vec (fp16 [in]) from the PTQ sidecar.
  - Copies everything else (linear.*, lora_*, bias_param, embeddings, norms)
    byte-identical from the dense export.

Mathematically identical to the source. ~16 GB output (vs 122 GB dense).

Usage:
  python3 convert_qwq32b_to_factored.py \
      --src     /tmp/qwq-32b-2bit-lora-kd-linear-v21 \
      --sidecar /tmp/qwq_32b_linear_g64_sidecar \
      --out     /tmp/qwq_32b_lora_kd_v21_factored
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from srht_utils import rotate_last_dim, srht_block  # noqa: E402


def build_dense_R(perm, signs, in_dim, device, dtype=torch.float16):
    eye = torch.eye(in_dim, device=device, dtype=torch.float32)
    R = rotate_last_dim(eye, perm, signs, srht_block(in_dim))
    return R.to(dtype).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/tmp/qwq-32b-2bit-lora-kd-linear-v21")
    ap.add_argument("--sidecar", default="/tmp/qwq_32b_linear_g64_sidecar")
    ap.add_argument("--out", default="/tmp/qwq_32b_lora_kd_v21_factored")
    ap.add_argument(
        "--target-suffixes",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    args = ap.parse_args()

    src, sidecar, out = Path(args.src), Path(args.sidecar), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffixes = tuple(args.target_suffixes)
    t0 = time.time()

    # 1. sidecar perm/signs/awq (one perm/sign per unique in_features; awq per projection)
    print(f"[load] sidecar {sidecar}")
    perm_by_dim, sign_by_dim = {}, {}
    with safe_open(sidecar / "srht_perm.safetensors", framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 (safe_open is not iterable)
            d = f.get_tensor(k).shape[0]
            if d not in perm_by_dim:
                perm_by_dim[d] = f.get_tensor(k)
    with safe_open(sidecar / "srht_signs.safetensors", framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 (safe_open is not iterable)
            d = f.get_tensor(k).shape[0]
            if d not in sign_by_dim:
                sign_by_dim[d] = f.get_tensor(k)
    awq_by_stem = {}
    with safe_open(sidecar / "awq_scale.safetensors", framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 (safe_open is not iterable)
            awq_by_stem[k] = f.get_tensor(k).to(torch.float16).contiguous()
    print(f"  unique in_features: {sorted(perm_by_dim.keys())}  | awq projections: {len(awq_by_stem)}")

    # 2. build shared R per dim (fp16)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    R_by_dim = {}
    for d in sorted(perm_by_dim.keys()):
        print(f"[build] shared_R_{d}  [{d},{d}] fp16  ({d * d * 2 / 1e9:.2f} GB)")
        R_by_dim[d] = build_dense_R(perm_by_dim[d], sign_by_dim[d], d, device).cpu()

    # 3. walk dense export shards
    shards = sorted(src.glob("model*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"no model*.safetensors in {src}")

    def matched_stem(key):
        if not key.endswith(".pre_linear.weight"):
            return None
        stem = key[: -len(".pre_linear.weight")]
        return stem if any(stem.endswith("." + s) for s in suffixes) else None

    new_state = {}
    n_drop, n_copy = 0, 0
    seen_stems = set()
    for shard in shards:
        print(f"[load] {shard.name}")
        with safe_open(shard, framework="pt") as f:
            for k in tqdm(list(f.keys()), desc=f"  {shard.name}", unit="t"):  # noqa: SIM118 (safe_open is not iterable)
                stem = matched_stem(k)
                if stem is not None:
                    seen_stems.add(stem)  # drop the dense pre_linear.weight
                    n_drop += 1
                else:
                    new_state[k] = f.get_tensor(k).clone()
                    n_copy += 1

    # 4. add per-projection awq_s_vec (from sidecar, matched by stem)
    n_awq = 0
    for stem in sorted(seen_stems):
        if stem not in awq_by_stem:
            raise KeyError(f"sidecar awq_scale missing stem: {stem}")
        new_state[f"{stem}.awq_s_vec"] = awq_by_stem[stem]
        n_awq += 1

    # 5. add shared R
    for d, R in R_by_dim.items():
        new_state[f"shared_R_{d}"] = R

    print(f"\n[summary] dropped pre_linear={n_drop}  copied={n_copy}  awq_added={n_awq}  shared_R={len(R_by_dim)}")

    # 6. save (sharded if >40GB; here ~16GB so single file)
    out_st = out / "model.safetensors"
    print(f"[save] {out_st}  ({len(new_state)} tensors)")
    save_file(new_state, str(out_st), metadata={"format": "pt"})

    # 7. config
    cfg = json.loads((src / "config.json").read_text())
    cfg["architectures"] = ["Qwen2FactoredLoraForCausalLM"]
    # Self-loads via the custom modeling file (auto_map), not the HF quark quantizer;
    # drop any stale PTQ quantization_config so from_pretrained doesn't try to rebuild
    # a quantizer (e.g. bfp16, which isn't loadable) and fail.
    cfg.pop("quantization_config", None)
    cfg["auto_map"] = {"AutoModelForCausalLM": "modeling_qwq32b_factored_lora.Qwen2FactoredLoraForCausalLM"}
    cfg["factored_rotation_dims"] = sorted(R_by_dim.keys())
    if "srht_awq_lora" not in cfg:
        cfg["srht_awq_lora"] = {"rank": 64, "alpha": 128, "group_size": 64, "target_suffixes": list(suffixes)}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))

    # 8. modeling + tokenizer
    mf = REPO_ROOT / "modeling_qwq32b_factored_lora.py"
    shutil.copy(mf, out / mf.name)
    for fn in (
        "chat_template.jinja",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "vocab.json",
        "merges.txt",
    ):
        sp = src / fn
        if sp.exists():
            shutil.copy(sp, out / fn)

    (out / "factored_meta.json").write_text(
        json.dumps(
            {
                "scheme": "QwQ-32B v21 LoRA+KD factored: 2-bit + per-proj awq_s_vec + 2 shared rotations",
                "src": str(src),
                "sidecar": str(sidecar),
                "rotation_dims": sorted(R_by_dim.keys()),
                "n_pre_linear_replaced": n_drop,
                "elapsed_s": round(time.time() - t0, 2),
            },
            indent=2,
        )
    )

    # 9. report
    total = sum(p.stat().st_size for p in out.iterdir())
    print(f"\n[output] {out}  total = {total / 1e9:.2f} GB")
    print(f"[done] elapsed = {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
