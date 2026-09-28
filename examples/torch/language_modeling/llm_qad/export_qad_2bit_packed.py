#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
"""Export a QAD-trained 2-bit model to a packed int2 + per-group-scale format.

A QAD-trained checkpoint is a vanilla HF model whose proj-layer weights take
only 4 levels per group (`{-1, -1/3, 1/3, 1} x per-group scale`). This utility
packs those into a compact deployable bundle:

    <output_dir>/
      qweights_packed.safetensors   # uint8, 4 x int2 indices per byte (LSB-first)
      scales.safetensors            # per-group fp16 scales [out, in/group]
      quantization_config.json      # metadata (levels, packing, layers, ...)
      hf_model/                     # bf16 fake-quant copy for HF eval (unless --skip_hf_model)

Model-agnostic: detects quantized linears by name (``*proj*`` excluding
``embed_tokens`` / ``lm_head``), so it works for Phi-4 (Phi3) and QwQ-32B (Qwen2).

Usage:
    python export_qad_2bit_packed.py --model_dir /path/to/qad_ckpt --output_dir /path/to/export
    # skip the large hf_model/ copy:
    python export_qad_2bit_packed.py --model_dir ... --output_dir ... --skip_hf_model
"""

import argparse
import json
import os

import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_LEVELS = [-1.0, -1.0 / 3.0, 1.0 / 3.0, 1.0]


def _is_quantized_linear(name: str, module: torch.nn.Module) -> bool:
    return isinstance(module, torch.nn.Linear) and "proj" in name and "embed" not in name and "lm_head" not in name


def pack_qad_2bit(weight: torch.Tensor, group_size: int, levels: torch.Tensor):
    """Pack a 4-level weight [out, in] into (packed_uint8 [out, in/4], scale [out, in/group]).

    Per group, scale = max|w|; each element is snapped to the nearest level and
    its index (0..3) packed 4-per-byte LSB-first.
    """
    w = weight.float()
    o, i = w.shape
    if i % group_size != 0 or i % 4 != 0:
        raise ValueError(f"in_features {i} not divisible by group_size {group_size} / 4")
    n_groups = i // group_size
    w_grouped = w.view(o, n_groups, group_size)
    scale = w_grouped.abs().amax(dim=-1).clamp(min=1e-8)
    u = (w_grouped / scale.unsqueeze(-1)).clamp(-1.5, 1.5)
    indices = (u.unsqueeze(-1) - levels.view(1, 1, 1, 4)).abs().argmin(dim=-1)
    idx_flat = indices.view(o, -1).to(torch.uint8)
    packed = torch.zeros(o, i // 4, dtype=torch.uint8)
    for k in range(4):
        packed |= idx_flat[:, k::4] << (2 * k)
    return packed, scale.to(torch.float16)


def unpack_qad_2bit(
    packed: torch.Tensor, scale: torch.Tensor, in_features: int, group_size: int, levels: torch.Tensor
) -> torch.Tensor:
    """Inverse of :func:`pack_qad_2bit` -> dense fp32 weight [out, in]."""
    o = packed.shape[0]
    idx = torch.empty(o, in_features, dtype=torch.long)
    for k in range(4):
        idx[:, k::4] = ((packed >> (2 * k)) & 0x3).long()
    w_levels = levels[idx]  # [o, in]
    per_col_scale = scale.float().repeat_interleave(group_size, dim=1)  # [o, in]
    return w_levels * per_col_scale


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_dir", required=True, help="QAD-trained HF checkpoint (4-level weights).")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--group_size", type=int, default=64)
    ap.add_argument("--skip_hf_model", action="store_true", help="Do not copy the bf16 hf_model/.")
    ap.add_argument("--device", default="cpu", help="Device for packing math (cpu is fine).")
    ap.add_argument(
        "--model_name", default=None, help="Base model id for metadata (default: model.config._name_or_path)."
    )
    ap.add_argument(
        "--description",
        default="QAD 2-bit weight quantization (4 levels per group) with INT16 activations.",
        help="Human-readable description recorded in quantization_config.json.",
    )
    ap.add_argument(
        "--training_meta",
        default=None,
        help="Optional JSON file with training provenance (version, steps, kd params, ...).",
    )
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    levels = torch.tensor(DEFAULT_LEVELS)
    g = args.group_size

    print(f"[1/4] Loading model from {args.model_dir} ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.bfloat16, device_map=args.device, trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    quantized_layers = [n for n, m in model.named_modules() if _is_quantized_linear(n, m)]
    print(f"[2/4] Packing 2-bit weights from {len(quantized_layers)} layers (group_size={g}) ...")

    scales_dict, qweights_dict, layer_info = {}, {}, {}
    max_err_global = 0.0

    for name in quantized_layers:
        module = model.get_submodule(name)
        w = module.weight.data.float()
        o, i = w.shape
        packed, scale = pack_qad_2bit(w, g, levels)
        scales_dict[f"{name}.scale"] = scale
        qweights_dict[f"{name}.qweight"] = packed

        w_rec = unpack_qad_2bit(packed, scale, i, g, levels)
        err = (w - w_rec).abs().max().item()
        max_err_global = max(max_err_global, err)
        layer_info[name] = {
            "shape": list(w.shape),
            "n_groups": i // g,
            "group_size": g,
            "scale_shape": list(scale.shape),
            "qweight_packed_shape": list(packed.shape),
            "max_reconstruction_error": f"{err:.2e}",
        }

    print(f"  worst-case reconstruction error: {max_err_global:.2e}")

    print(f"[3/4] Saving export to {args.output_dir} ...")
    save_file(scales_dict, os.path.join(args.output_dir, "scales.safetensors"))
    save_file(qweights_dict, os.path.join(args.output_dir, "qweights_packed.safetensors"))

    model_name = args.model_name or getattr(model.config, "_name_or_path", "") or args.model_dir
    training_meta = {}
    if args.training_meta:
        with open(args.training_meta) as f:
            training_meta = json.load(f)

    quant_config = {
        "quant_method": "qad_2bit_linear",
        "description": args.description,
        "weight_quantization": {
            "bits": 2,
            "group_size": g,
            "symmetric": True,
            "levels_normalized": DEFAULT_LEVELS,
            "level_indices": {"neg1": 0, "neg1_3": 1, "pos1_3": 2, "pos1": 3},
            "packing": "4 x 2-bit values per uint8 byte, LSB first",
            "scale_dtype": "float16",
            "scale_shape": "[out_features, in_features // group_size]",
            "reconstruction": "weight = levels_normalized[index] * scale",
        },
        "activation_quantization": {
            "bits": 16,
            "scheme": "dynamic_per_token_symmetric",
            "dtype": "int16",
            "formula": "scale = max(|x_row|) / 32767; x_q = round(x / scale).clamp(-32768, 32767)",
            "applied_to": "inputs of all proj linear layers (excluding embed_tokens, lm_head)",
        },
        "training": training_meta,
        "quantized_layers": quantized_layers,
        "non_quantized_layers": ["model.embed_tokens", "lm_head"],
        "model": model_name,
        "base_dtype": "bfloat16",
        "source_checkpoint": args.model_dir,
        "layer_details": layer_info,
    }
    with open(os.path.join(args.output_dir, "quantization_config.json"), "w") as f:
        json.dump(quant_config, f, indent=2)

    if not args.skip_hf_model:
        hf_dir = os.path.join(args.output_dir, "hf_model")
        print(f"[4/4] Saving bf16 hf_model/ to {hf_dir} ...")
        model.save_pretrained(hf_dir, safe_serialization=True)
        tokenizer.save_pretrained(hf_dir)
    else:
        print("[4/4] Skipping hf_model/ copy (--skip_hf_model).")

    total_scale = sum(v.nelement() * 2 for v in scales_dict.values())
    total_q = sum(v.nelement() for v in qweights_dict.values())
    print(f"\nExport complete: {args.output_dir}")
    print(f"  quantized layers : {len(quantized_layers)}")
    print(f"  packed qweights  : {total_q / 1e6:.1f} MB")
    print(f"  scales           : {total_scale / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
