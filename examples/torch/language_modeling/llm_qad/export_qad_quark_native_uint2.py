#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""QAD checkpoint -> fully Quark-NATIVE uint2 export (importable by import_model_from_safetensors).

Quark layout per proj layer:
  weight              uint8, Pack_uint2 (per-group transpose + 4x uint2/byte), [in, out/4]
  weight_scale        fp16,  [in/group, out]   (= (2/3)*max_abs, transposed for storage)
  weight_zero_point   fp16,  [in/group, out]   (= 1.5, float zero-point)
config.json embeds quantization_config (quant_method="quark", dtype="uint2", per_group,
zero_point_type="float32"). No modeling file / trust_remote_code.
"""

import argparse
import json
import os

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from quark.torch.utils import create_pack_method

LEVELS = torch.tensor([-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0])


def is_quant_linear(name, m):
    return isinstance(m, torch.nn.Linear) and "proj" in name and "embed" not in name and "lm_head" not in name


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--group_size", type=int, default=64)
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    g = a.group_size
    pm = create_pack_method("per_group", "uint2")

    print(f"[1/4] load {a.model_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir, torch_dtype=torch.bfloat16, device_map="cpu", trust_remote_code=True
    )
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    qlayers = [n for n, m in model.named_modules() if is_quant_linear(n, m)]
    print(f"[2/4] encode {len(qlayers)} proj layers (Quark native layout)")
    orig = model.state_dict()
    sd, qkeys, max_err = {}, set(), 0.0
    for n in qlayers:
        m = model.get_submodule(n)
        w = m.weight.data.float()
        o, i = w.shape
        wg = w.view(o, i // g, g)
        maxabs = wg.abs().amax(-1).clamp(min=1e-8)
        idx = (
            (wg.unsqueeze(-1) / maxabs.unsqueeze(-1).unsqueeze(-1) - LEVELS.view(1, 1, 1, 4))
            .abs()
            .argmin(-1)
            .view(o, i)
        )
        scale_q = (2.0 / 3.0) * maxabs
        zp = torch.full((o, i // g), 1.5)
        sd[f"{n}.weight"] = pm.pack(idx.to(torch.int32), True)  # [in, out/4]
        sd[f"{n}.weight_scale"] = scale_q.t().contiguous().to(torch.float16)  # [in/g, out]
        sd[f"{n}.weight_zero_point"] = zp.t().contiguous().to(torch.float16)  # [in/g, out]
        qkeys.add(f"{n}.weight")
        qkeys.add(f"{n}.bias")
        if m.bias is not None:
            sd[f"{n}.bias"] = m.bias.data.to(torch.float16)
        rec = (idx.float() - 1.5) * scale_q.repeat_interleave(g, dim=1)
        max_err = max(max_err, (w - rec).abs().max().item())
    print(f"    worst-case |QAD - uint2 reconstruct| = {max_err:.3e}")
    for k, v in orig.items():
        if k in qkeys:
            continue
        sd[k] = v.clone()

    print(f"[3/4] save model.safetensors ({len(sd)} tensors)")
    save_file(sd, os.path.join(a.output_dir, "model.safetensors"), metadata={"format": "pt"})

    with open(os.path.join(a.model_dir, "config.json")) as f:
        cfg = json.load(f)
    cfg.pop("auto_map", None)
    cfg.pop("quantization_config", None)
    wspec = {
        "dtype": "uint2",
        "qscheme": "per_group",
        "ch_axis": 1,
        "group_size": g,
        "symmetric": False,
        "zero_point_type": "float32",
        "scale_type": "float",
        "is_dynamic": False,
        "round_method": "half_even",
        "observer_cls": "PerGroupMinMaxObserver",
    }
    cfg["quantization_config"] = {
        "quant_method": "quark",
        "global_quant_config": {"weight": wspec, "bias": None, "input_tensors": None, "output_tensors": None},
        "exclude": ["lm_head"],
        "layer_quant_config": {},
        "layer_type_quant_config": {},
        "kv_cache_quant_config": {},
        "quant_mode": "eager_mode",
        "export": {
            "weight_format": "real_quantized",
            "pack_method": "reorder",
            "kv_cache_group": [],
            "min_kv_scale": 0.0,
        },
    }
    with open(os.path.join(a.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    tok.save_pretrained(a.output_dir)
    for extra in ("generation_config.json",):
        src = os.path.join(a.model_dir, extra)
        if os.path.exists(src):
            import shutil

            shutil.copy(src, os.path.join(a.output_dir, extra))
    print(f"[4/4] done -> {a.output_dir}")


if __name__ == "__main__":
    main()
