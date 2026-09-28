#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Convert v16 newexport (or v16-linear-dense) -> FACTORED form.

Replaces 160 per-projection (awq_s_vec, srht_perm, srht_sign) triplets with:
  - 1 dense rotation R per unique in_features (top-level shared_R_<in>)
  - 160 awq_s_vec per projection (unchanged)

Mathematically identical to the source. ~7 GB output (vs 6.34 GB for newexport
or 38.3 GB for v16-linear-dense).

Usage:
  python3 convert_v16_to_factored.py \
      --src /amd_models/phi-4-2bit-lora-kd-linear-v16-newexport \
      --sidecar /tmp/phi4_linear_g64_sidecar \
      --out /tmp/phi4_lora_kd_v16_factored
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

# Reuse helpers (rotate_last_dim, srht_block) from the existing dense exporter.
sys.path.insert(0, str(REPO_ROOT))
from srht_utils import (  # noqa: E402
    rotate_last_dim,
    srht_block,
)


def build_dense_R(perm: torch.Tensor, signs: torch.Tensor, in_dim: int, device, dtype=torch.float16) -> torch.Tensor:
    """Build the dense [in, in] rotation matrix R such that
        rotate_last_dim(eye, perm, signs, blk(in)) = R
    so that for any matrix M, rotate_last_dim(M) = M @ R.
    Returned in `dtype` (fp16 by default).
    """
    eye = torch.eye(in_dim, device=device, dtype=torch.float32)
    R = rotate_last_dim(eye, perm, signs, srht_block(in_dim))
    return R.to(dtype).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="/amd_models/phi-4-2bit-lora-kd-linear-v16-newexport")
    ap.add_argument("--sidecar", default="/tmp/phi4_linear_g64_sidecar")
    ap.add_argument("--out", default="/tmp/phi4_lora_kd_v16_factored")
    ap.add_argument("--target-suffixes", nargs="+", default=["qkv_proj", "o_proj", "gate_up_proj", "down_proj"])
    ap.add_argument(
        "--quark_pack",
        action="store_true",
        help="Source is a Quark-native structured export (uint2 packed_levels + "
        "weight_scale + float weight_zero_point). Emits the Quark-backed factored "
        "modeling (Phi3FactoredQuarkForCausalLM) which dequantizes via Quark kernels.",
    )
    args = ap.parse_args()

    src = Path(args.src)
    sidecar = Path(args.sidecar)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffixes = tuple(args.target_suffixes)

    t_start = time.time()

    # ── 1. Load sidecar perm + signs (just need one per unique in_features) ──
    print(f"[load] sidecar from {sidecar}")
    perm_by_dim = {}
    sign_by_dim = {}
    with safe_open(sidecar / "srht_perm.safetensors", framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 (safe_open is not iterable)
            t = f.get_tensor(k)
            d = t.shape[0]
            if d not in perm_by_dim:
                perm_by_dim[d] = t
    with safe_open(sidecar / "srht_signs.safetensors", framework="pt") as f:
        for k in f.keys():  # noqa: SIM118 (safe_open is not iterable)
            t = f.get_tensor(k)
            d = t.shape[0]
            if d not in sign_by_dim:
                sign_by_dim[d] = t

    print(f"  unique in_features: {sorted(perm_by_dim.keys())}")

    # ── 2. Build dense R per unique in_features (fp16) ──────────────────
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    R_by_dim = {}
    for d in sorted(perm_by_dim.keys()):
        print(f"[build] R_{d}  shape=[{d}, {d}]  dtype=fp16  ({d * d * 2 / 1e6:.1f} MB)")
        R = build_dense_R(perm_by_dim[d], sign_by_dim[d], d, device, dtype=torch.float16)
        R_by_dim[d] = R.cpu()

    # ── 3. Walk the source export ───────────────────────────────────────
    src_shards = sorted(src.glob("model*.safetensors"))
    if not src_shards:
        raise FileNotFoundError(f"no model*.safetensors in {src}")

    new_state = {}
    n_pre_linear_replaced = 0
    n_copied_other = 0

    def is_pre_linear_key(key: str):
        """Returns the projection stem if key is a pre_linear sub-tensor of a target."""
        for kind in (".pre_linear.awq_s_vec", ".pre_linear.srht_perm", ".pre_linear.srht_sign", ".pre_linear.weight"):
            if key.endswith(kind):
                stem = key[: -len(kind)]
                for s in suffixes:
                    if stem.endswith("." + s):
                        return stem, kind
        return None, None

    seen_stems = set()
    awq_by_stem = {}
    in_dim_by_stem = {}

    for shard in src_shards:
        print(f"[load] {shard}")
        with safe_open(shard, framework="pt") as f:
            keys = list(f.keys())  # noqa: SIM118 (safe_open is not iterable)
            for k in tqdm(keys, desc=f"  {shard.name}", unit="t"):
                stem, kind = is_pre_linear_key(k)
                if stem is not None:
                    if kind == ".pre_linear.awq_s_vec":
                        awq_by_stem[stem] = f.get_tensor(k).to(torch.float16).contiguous()
                        in_dim_by_stem[stem] = awq_by_stem[stem].shape[0]
                        seen_stems.add(stem)
                        n_pre_linear_replaced += 1
                    # Drop srht_perm, srht_sign, and the dense pre_linear.weight
                    # (all encoded in the shared R now).
                else:
                    new_state[k] = f.get_tensor(k).clone()
                    n_copied_other += 1

    # ── 4. Recover awq_s_vec from sidecar for any missing stems (if source was v16-linear-dense) ──
    if not awq_by_stem:
        print("[fallback] source had no .pre_linear.awq_s_vec keys; loading from sidecar awq_scale.safetensors")
        with safe_open(sidecar / "awq_scale.safetensors", framework="pt") as f:
            for k in f.keys():  # noqa: SIM118 (safe_open is not iterable)
                stem = k  # sidecar uses bare stems like "model.layers.0.self_attn.qkv_proj"
                t = f.get_tensor(k).to(torch.float16).contiguous()
                awq_by_stem[stem] = t
                in_dim_by_stem[stem] = t.shape[0]

    # ── 5. Write per-projection awq_s_vec ───────────────────────────────
    for stem, awq in awq_by_stem.items():
        new_state[f"{stem}.awq_s_vec"] = awq

    # ── 6. Add the shared R buffers at top level ────────────────────────
    for d, R in R_by_dim.items():
        new_state[f"shared_R_{d}"] = R

    print(f"\n[summary] pre_linear stems handled: {n_pre_linear_replaced}  other copied: {n_copied_other}")
    print(f"  unique in dims: {sorted(R_by_dim.keys())}")

    # ── 7. Save consolidated single-file safetensors ────────────────────
    out_safetensors = out / "model.safetensors"
    print(f"[save] {out_safetensors}  ({len(new_state)} tensors)")
    save_file(new_state, str(out_safetensors), metadata={"format": "pt"})

    # ── 8. Patch config.json ────────────────────────────────────────────
    src_cfg = json.loads((src / "config.json").read_text())
    arch_class = "Phi3FactoredQuarkForCausalLM" if args.quark_pack else "Phi3FactoredLoraForCausalLM"
    modeling_stem = "modeling_phi4_factored_quark" if args.quark_pack else "modeling_phi4_factored_lora"
    src_cfg["architectures"] = [arch_class]
    src_cfg["auto_map"] = {
        "AutoModelForCausalLM": f"{modeling_stem}.{arch_class}",
    }
    # This checkpoint self-loads via the custom modeling file above (auto_map), not the
    # HF quark quantizer. Drop any stale PTQ quantization_config so from_pretrained
    # doesn't try to rebuild a quantizer (e.g. bfp16, which isn't loadable) and fail.
    src_cfg.pop("quantization_config", None)
    src_cfg["factored_rotation_dims"] = sorted(R_by_dim.keys())
    if "srht_awq_lora" not in src_cfg:
        src_cfg["srht_awq_lora"] = {
            "rank": 64,
            "alpha": 128,
            "group_size": 64,
            "target_suffixes": list(suffixes),
        }
    (out / "config.json").write_text(json.dumps(src_cfg, indent=2))

    # ── 9. Copy modeling file + tokenizer ───────────────────────────────
    src_modeling = REPO_ROOT / f"{modeling_stem}.py"
    if not src_modeling.exists():
        raise FileNotFoundError(f"missing modeling file: {src_modeling}")
    shutil.copy(src_modeling, out / src_modeling.name)
    for fn in ("chat_template.jinja", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
        sp = src / fn
        if sp.exists():
            shutil.copy(sp, out / fn)

    # ── 10. Meta JSON ───────────────────────────────────────────────────
    meta = {
        "scheme": "Phi-4 v16 LoRA+KD factored: 2-bit linear + per-projection awq_s_vec + 2 shared rotation matrices",
        "src": str(src),
        "sidecar": str(sidecar),
        "rotation_dims": sorted(R_by_dim.keys()),
        "n_pre_linear_replaced": n_pre_linear_replaced,
        "elapsed_s": round(time.time() - t_start, 2),
    }
    (out / "factored_meta.json").write_text(json.dumps(meta, indent=2))

    # ── 11. Report ──────────────────────────────────────────────────────
    print(f"\n[output sizes] -> {out}")
    total = 0
    rows = []
    for p in out.iterdir():
        sz = p.stat().st_size
        total += sz
        rows.append((sz, p.name))
    for sz, name in sorted(rows, reverse=True):
        print(f"  {sz / 1e6:10.2f} MB   {name}")
    print(f"  {'-' * 40}")
    print(f"  {total / 1e9:10.3f} GB   TOTAL")
    print(f"\n[done] elapsed = {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
