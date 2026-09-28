#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""
Generic **Quark-native** export of a 2-bit LoRA+KD student (any model).

Unlike the custom `trust_remote_code` factored export, this writes a checkpoint that
loads through Quark's own `import_model_from_safetensors`:

  * weight  : uint2, packed by Quark's `Pack_uint2` (per-group)         -> `{layer}.weight`
  * scale   : fp16 [n_groups, out] (transposed, Quark convention)        -> `{layer}.weight_scale`
  * zp      : fp16 [n_groups, out] = 1.5 (float zero-point, unpacked)    -> `{layer}.weight_zero_point`
  * awq     : fp16 [in] = 1/awq_s_vec, applied before the rotation       -> `{layer}.input_prescale`
  * rotation: fp16 [in, in] SRHT matrix, ONE per unique in_features      -> top-level `shared_input_rotation_<in>`

The uniform Lloyd-Max levels {-1,-1/3,1/3,1} are encoded as an affine uint2 quant
(zp=1.5, scale=(2/3)*group_scale). The rotation + AWQ are carried by Quark's
`QParamsLinearWithRotation` (online, trainable float path) via a `RotationConfig`
whose `online_config.online_rotation_layers` lists the target layers explicitly
(model-agnostic; no per-architecture `scaling_layers` template needed).

LoRA is NOT part of the native base: it is applied on top as a standard peft adapter
(pass the trained adapter dir to `PeftModel.from_pretrained` after import).

Requires the Quark-core "shared/fp16/prescale rotation import" changes.

Usage:
  python3 export_quark_native.py \
      --student_dir /tmp/phi4_lloyd_max_g64 \
      --sidecar_dir /tmp/phi4_linear_g64_sidecar \
      --output_dir  /tmp/phi4_quark_native
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM

from quark.torch.quantization.config.config import (
    OnlineRotationConfig,
    QConfig,
    QLayerConfig,
    QTensorConfig,
    RotationConfig,
)
from quark.torch.quantization.config.type import Dtype, QSchemeType, RoundType, ScaleType, ZeroPointType
from quark.torch.quantization.observer.observer import PerGroupMinMaxObserver
from quark.torch.utils.pack import Pack_uint2

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO_ROOT)
from srht_utils import rotate_last_dim, srht_block  # noqa: E402

TARGET_SUFFIXES = (
    "qkv_proj",
    "o_proj",
    "gate_up_proj",
    "down_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "gate_proj",
    "up_proj",
)
LEVEL_ZERO_POINT = 1.5
LEVEL_SCALE = 2.0 / 3.0
GROUP_SIZE = 64


def levels_to_idx(w_q: torch.Tensor) -> torch.Tensor:
    return (w_q > -0.5).to(torch.uint8) + (w_q > 0.0).to(torch.uint8) + (w_q > 0.5).to(torch.uint8)


def build_R(perm, signs, in_dim, dtype=torch.float16):
    eye = torch.eye(in_dim, dtype=torch.float32)
    R = rotate_last_dim(eye, perm, signs, srht_block(in_dim))
    return R.to(dtype).contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student_dir", required=True)
    ap.add_argument("--sidecar_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--group_size", type=int, default=GROUP_SIZE)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    side = {}
    for fld in ("W_q_levels", "group_scale", "awq_scale", "srht_perm", "srht_signs"):
        with safe_open(os.path.join(args.sidecar_dir, f"{fld}.safetensors"), framework="pt") as f:
            side[fld] = {k: f.get_tensor(k) for k in f.keys()}  # noqa: SIM118 (safe_open is not iterable)
    stems = sorted(side["W_q_levels"].keys())
    print(f"[native export] {len(stems)} quantized projections")

    # true in_features per target from the model skeleton (generic)
    cfg = AutoConfig.from_pretrained(args.student_dir)
    skel = AutoModelForCausalLM.from_config(cfg)
    in_by_name = {n: m.in_features for n, m in skel.named_modules() if isinstance(m, nn.Linear)}

    packer = Pack_uint2(qscheme="per_group", dtype="uint2")
    state = {}
    # non-quantized tensors (embed / norms / lm_head): copy from student
    for shard in sorted(__import__("glob").glob(os.path.join(args.student_dir, "model*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():  # noqa: SIM118 (safe_open is not iterable)
                stem = k[: -len(".weight")] if k.endswith(".weight") else None
                if stem is not None and stem in side["W_q_levels"]:
                    continue  # quantized projection weight -> replaced below
                state[k] = f.get_tensor(k)

    uniq_in = set()
    for stem in tqdm(stems, unit="proj"):
        in_dim = in_by_name[stem]
        uniq_in.add(in_dim)
        wq = side["W_q_levels"][stem]  # [out, in]
        idx = levels_to_idx(wq)
        state[f"{stem}.weight"] = packer.pack(idx.to(torch.uint8))  # [in, out/4]
        gs = side["group_scale"][stem].float()  # [out, n_groups]
        state[f"{stem}.weight_scale"] = (LEVEL_SCALE * gs).t().contiguous().to(torch.float16)  # [ng, out]
        state[f"{stem}.weight_zero_point"] = torch.full(
            (gs.shape[1], gs.shape[0]), LEVEL_ZERO_POINT, dtype=torch.float16
        )  # [ng, out]
        state[f"{stem}.input_prescale"] = (1.0 / side["awq_scale"][stem].float()).to(torch.float16)  # [in]

    for d in sorted(uniq_in):
        # one shared SRHT rotation per in_features (use any layer with that in_dim)
        stem = next(s for s in stems if in_by_name[s] == d)
        state[f"shared_input_rotation_{d}"] = build_R(side["srht_perm"][stem], side["srht_signs"][stem], d)
        print(f"  shared_input_rotation_{d}: [{d},{d}] fp16 ({d * d * 2 / 1e6:.1f} MB)")

    out_path = os.path.join(args.output_dir, "model.safetensors")
    print(f"[save] {out_path}  ({len(state)} tensors)")
    save_file(state, out_path, metadata={"format": "pt"})

    # ---- config.json with Quark quantization_config (schema via QConfig.to_dict) ----
    weight_spec = QTensorConfig(
        dtype=Dtype.uint2,
        observer_cls=PerGroupMinMaxObserver,
        symmetric=False,
        qscheme=QSchemeType.per_group,
        ch_axis=1,
        group_size=args.group_size,
        is_dynamic=False,
        scale_type=ScaleType.float,
        zero_point_type=ZeroPointType.float32,
        round_method=RoundType.half_even,
    )
    online = OnlineRotationConfig(
        shared_parallel=False,
        online_rotation_layers=list(stems),
        use_input_prescale=True,
    )
    rot = RotationConfig(
        scaling_layers=None,
        r1=True,
        r2=False,
        r3=False,
        r4=False,
        online_r1_rotation=True,
        trainable=True,
        online_config=online,
    )
    qconf = QConfig(global_quant_config=QLayerConfig(weight=weight_spec), algo_config=[rot], exclude=["lm_head"])
    qc_dict = qconf.to_dict()
    qc_dict["export"] = {
        "kv_cache_group": [],
        "min_kv_scale": 0.0,
        "pack_method": "reorder",
        "weight_format": "real_quantized",
        "weight_merge_groups": None,
    }

    with open(os.path.join(args.student_dir, "config.json")) as f:
        hf_cfg = json.load(f)
    hf_cfg["quantization_config"] = qc_dict
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(hf_cfg, f, indent=2)

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

    print(f"[done] Quark-native base -> {args.output_dir}")
    print("  Load: import_model_from_safetensors(model_skeleton, dir); then PeftModel.from_pretrained(model, lora_dir)")


if __name__ == "__main__":
    main()
