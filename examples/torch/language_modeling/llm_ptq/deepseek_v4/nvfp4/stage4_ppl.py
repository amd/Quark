#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Loads and runs the DeepSeek-V4-Pro checkpoint's model code (inference/model.py).
# Copyright (c) 2023 DeepSeek. Licensed under the MIT License.
#

"""
Stage 4: NVFP4 PPL evaluation for a DeepSeek-V4-Pro checkpoint produced by this
pipeline (Stage 1 ``stage1_quantize_weight.py`` + Stage 3 ``stage3_merge.py``).

Loads an *already* NVFP4-quantized checkpoint and measures its PPL. Both weights
and activations are quantized to NVFP4 (per-group-16 FP4 + per-tensor
``input_scale``), matching real NVFP4 inference; this requires the merged
``input_scale`` from Stages 2-3. Pass ``--no-quant-act`` to keep activations in
BF16 instead.

On-disk layout per quantized expert weight ``<w>``:

    <w>             U8       packed FP4 nibbles (2 codes / byte, inner dim / 2)
    <w>_scale       F8_E4M3  per-group(16) block scale (the scale, in FP8)
    <w>_scale_2     F32      per-tensor global scale (scale-of-scale)
    <w>.input_scale F32      per-tensor activation scale

and the kept (non-expert) linears stay in the source FP8 layout:

    <w>             F8_E4M3  fp8 weight
    <w>_scale       F8_E8M0  per-block(128x128) scale

Evaluation strategy (BF16-dequant path):
  Each quantized weight is dequantized to BF16 on the GPU transiently and the
  GEMM runs as a plain ``F.linear``. The activation is fake-quantized to NVFP4
  just before the GEMM (unless ``--no-quant-act``). The non-GEMM ops (attention,
  indexer) use the triton/PyTorch kernels in ``kernels`` (which replace the
  checkpoint's tilelang kernels on import).

Memory: the compact quantized weights live in CPU RAM; one decoder block is
moved to GPU per forward, where ``DequantLinear`` expands its weights to BF16
just for that block.

Usage::

    CUDA_VISIBLE_DEVICES=0 python stage4_ppl.py \\
        --model-dir /path/to/output-nvfp4 \\
        --batch-size 4

    # Smoke test (first 2 blocks only, nonsense PPL expected):
    CUDA_VISIBLE_DEVICES=0 python stage4_ppl.py \\
        --model-dir /path/to/output-nvfp4 \\
        --n-blocks 2 --batch-size 2

The ``--model-dir`` checkpoint must contain the model code under ``inference/``
(``model.py`` + ``config.json``) in addition to the quantized shards and index.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

_SRC_DIR = str(Path(__file__).parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# dsv4_common imports `dsv4_kernels`, which registers the synthetic ``kernel`` /
# ``fast_hadamard_transform`` modules so the checkpoint model.py picks up the
# triton/PyTorch implementations instead of tilelang.
from dsv4_common import (  # noqa: E402
    install_block_offload,
    load_and_wrap,
    load_model_module,
    load_tokenizer,
    ppl_eval,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Stage 4: NVFP4 PPL evaluation on wikitext-2 for a checkpoint produced by this "
            "pipeline (weights and activations quantized to NVFP4; pass --no-quant-act for BF16 acts)."
        )
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="NVFP4 checkpoint dir to evaluate; must contain the inference/ model code (model.py + config.json).",
    )
    p.add_argument("--device", default="cuda:0", help="Compute device (default: %(default)s).")
    p.add_argument(
        "--batch-size", type=int, default=4, help="Number of seqlen-token chunks per forward (default: %(default)s)."
    )
    p.add_argument("--seqlen", type=int, default=2048, help="Token length of each eval chunk (default: %(default)s).")
    p.add_argument(
        "--n-blocks",
        type=int,
        default=None,
        help="Limit to the first N decoder blocks; PPL is meaningless with few blocks (default: all layers).",
    )
    p.add_argument(
        "--n-gpu-blocks",
        type=int,
        default=0,
        help="Number of leading decoder blocks pinned on the GPU; the rest are streamed one at a "
        "time (default: %(default)s).",
    )
    p.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Evaluate only the first N chunks for a quick (less accurate) result (default: all chunks).",
    )
    p.add_argument(
        "--quant-act",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Quantize activations to NVFP4 (per-group-16 FP4 + per-tensor input_scale) for the full "
        "NVFP4 PPL; requires the merged input_scale (Stages 2-3). Pass --no-quant-act to keep "
        "activations in BF16 (default: %(default)s).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model_dir = Path(args.model_dir)

    eval_mode = "NVFP4 (weights + activations)" if args.quant_act else "NVFP4 weights, BF16 activations"
    header = (
        "NVFP4 PPL — DeepSeek-V4-Pro (Quark f2f checkpoint, BF16-dequant path)\n"
        f"  model_dir    : {model_dir}\n"
        f"  device       : {device}\n"
        f"  batch_size   : {args.batch_size}\n"
        f"  seqlen       : {args.seqlen}\n"
        f"  n_gpu_blocks : {args.n_gpu_blocks}\n"
        f"  eval mode    : {eval_mode}"
    )
    if args.n_blocks:
        header += f"\n  n_blocks     : {args.n_blocks} (smoke-test)"
    logger.info(header)

    logger.info("[1] Loading model module …")
    _mod = load_model_module(model_dir)
    Transformer = _mod.Transformer
    ModelArgs = _mod.ModelArgs

    with open(model_dir / "inference" / "config.json") as f:
        cfg = json.load(f)

    # Build the graph in BF16 (we dequant all weights ourselves). expert_dtype
    # left as bf16 so MoE Expert builds plain Linear (we replace them anyway).
    cfg["dtype"] = "bf16"
    cfg["scale_dtype"] = "bf16"
    cfg["expert_dtype"] = "bf16"
    cfg["n_mtp_layers"] = 0
    cfg.pop("scale_fmt", None)
    cfg["max_seq_len"] = args.seqlen
    cfg["max_batch_size"] = args.batch_size
    if args.n_blocks:
        cfg["n_layers"] = args.n_blocks
        ratios = cfg.get("compress_ratios", [])
        cfg["compress_ratios"] = tuple(ratios[: args.n_blocks]) + (0,)
    else:
        cfg["compress_ratios"] = tuple(cfg["compress_ratios"])

    valid = set(ModelArgs.__dataclass_fields__)
    model_args = ModelArgs(**{k: v for k, v in cfg.items() if k in valid})

    logger.info("[2] Building model on CPU …")
    with torch.device("cpu"):
        model = Transformer(model_args)
    model.eval()

    logger.info("[3] Loading + wrapping weights (dequant on forward) …")
    load_and_wrap(model, str(model_dir), "nvfp4", quant_act=args.quant_act)

    logger.info("[4] Installing per-block GPU offload …")
    # Move only NON-block params/buffers to GPU (embed / head / norm / freqs_cis).
    # Decoder blocks stay on CPU and are streamed to GPU per-forward.
    for mod_name, mod in model.named_modules():
        in_block = mod_name == "layers" or mod_name.startswith("layers.")
        if in_block:
            continue
        for pname, param in list(mod._parameters.items()):
            if param is not None and not param.is_meta:
                mod._parameters[pname] = nn.Parameter(param.to(device), requires_grad=param.requires_grad)
        for bname, buf in list(mod._buffers.items()):
            if buf is not None and not buf.is_meta:
                mod._buffers[bname] = buf.to(device)

    install_block_offload(model, device, n_gpu_blocks=args.n_gpu_blocks)

    # Indexer kv_cache buffers live inside blocks; keep them bf16. They get moved
    # with the block, but the cache is large — cast dtype now (cheap, on CPU).
    for mod in model.modules():
        if type(mod).__name__ == "Indexer" and hasattr(mod, "kv_cache") and mod.kv_cache is not None:
            mod.kv_cache = mod.kv_cache.to(torch.bfloat16)

    logger.info("[5] Loading dataset and tokenizer …")

    tokenizer = load_tokenizer(model_dir)
    testdata = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")
    testenc_ids = testenc.input_ids.to(device)
    total_tokens = testenc_ids.numel()
    if args.max_chunks is not None:
        testenc_ids = testenc_ids[:, : args.max_chunks * args.seqlen]
    nsamples = testenc_ids.numel() // args.seqlen
    logger.info(f"Total tokens: {total_tokens:,}  →  evaluating {nsamples} chunks of {args.seqlen}")

    # All-position logits, one sequence at a time (avoid the big vocab matrix).
    def _get_logits_all(x):
        out = []
        for b in range(x.shape[0]):
            out.append(F.linear(x[b].float(), model.head.weight))
        return torch.stack(out, dim=0)

    model.head.get_logits = _get_logits_all

    logger.info("[6] Running PPL evaluation …")
    ppl = ppl_eval(model, testenc_ids, device, batch_size=args.batch_size, seqlen=args.seqlen)

    peak_gpu = torch.cuda.max_memory_allocated(device) / 1e9
    logger.info(
        f"PPL (wikitext-2-raw-v1): {ppl:.4f}\n"
        f"Peak GPU memory        : {peak_gpu:.2f} GB\n"
        "  [for a comparable BF16 baseline, run ppl_baseline.py on the original checkpoint]"
    )


if __name__ == "__main__":
    main()
