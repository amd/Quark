#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Loads and runs the DeepSeek-V4-Pro checkpoint's model code (inference/model.py).
# Copyright (c) 2023 DeepSeek. Licensed under the MIT License.
#

"""
Shared helpers for the DeepSeek-V4-Pro NVFP4 example scripts.

Collects the model-assembly and evaluation utilities that the Stage 2 calibration
script and the Stage 4 / baseline PPL scripts all need:

* :func:`load_model_module` — load the checkpoint's ``inference/model.py`` and
  monkeypatch its tilelang-based ``Attention`` / ``Indexer`` forwards with the
  triton/PyTorch kernels in ``kernels``.
* :func:`load_tokenizer` — transformers-version-robust tokenizer loader.
* :func:`reset_kv_cache` — zero/clear the model's KV caches between forwards.
* :func:`ppl_eval` — wikitext-2 perplexity via teacher-forcing.
* :func:`install_block_offload` — per-block GPU streaming for single-GPU runs.

Importing this module requires the sibling ``kernels`` module to be importable
(put this example folder on ``PYTHONPATH``).
"""

from __future__ import annotations

import importlib.util as _ilu
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from dsv4_kernels import (
    dequant_fp8_block_weight,
    dequant_mxfp4_weight,
    dequant_nvfp4_weight,
    quant_dequant_nvfp4_act,
    sparse_attn,
)
from safetensors.torch import safe_open
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from quark.common.utils.log import ScreenLogger

logger = ScreenLogger(__name__)

_LINEAR_CLASSES = {"Linear", "ColumnParallelLinear", "RowParallelLinear"}


def load_model_module(model_dir: Path):
    """Load the checkpoint's ``inference/model.py`` and patch its attention path.

    The checkpoint's ``Attention`` / ``Indexer`` forwards call tilelang kernels;
    this replaces them with forwards that use the triton/PyTorch ``sparse_attn``
    (and the other kernels registered by importing ``kernels``).

    :param Path model_dir: Checkpoint directory containing ``inference/model.py``.

    :return: The loaded model module with patched forwards.
    """
    spec = _ilu.spec_from_file_location(
        "deepseek_v4_model", model_dir / "inference" / "model.py", submodule_search_locations=[]
    )
    mod = _ilu.module_from_spec(spec)
    sys.modules["deepseek_v4_model"] = mod
    spec.loader.exec_module(mod)

    _orig_win_topk = mod.get_window_topk_idxs
    _orig_compress_topk = mod.get_compress_topk_idxs

    def _get_window_topk_idxs(window_size, bsz, seqlen, start_pos, device=None):
        t = _orig_win_topk(window_size, bsz, seqlen, start_pos)
        return t if device is None else t.to(device)

    def _get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset, device=None):
        t = _orig_compress_topk(ratio, bsz, seqlen, start_pos, offset)
        return t if device is None else t.to(device)

    def _attention_forward_patched(self, x, start_pos):
        bsz, seqlen, _ = x.size()
        dev = x.device
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        win = self.window_size
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        if ratio and self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            if self.indexer is not None:
                self.indexer.freqs_cis = self.freqs_cis
        qr = q = self.q_norm(self.wq_a(x))
        q = self.wq_b(q).unflatten(-1, (self.n_local_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        mod.apply_rotary_emb(q[..., -rd:], freqs_cis)
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        mod.apply_rotary_emb(kv[..., -rd:], freqs_cis)
        mod.act_quant(kv[..., :-rd], 64, mod.scale_fmt, mod.scale_dtype, True)
        topk_idxs = _get_window_topk_idxs(win, bsz, seqlen, start_pos, device=dev)
        if ratio:
            offset = kv.size(1) if start_pos == 0 else win
            if self.indexer is not None:
                compress_topk_idxs = self.indexer(x, qr, start_pos, offset)
                compress_topk_idxs = compress_topk_idxs.to(dev)
            else:
                compress_topk_idxs = _get_compress_topk_idxs(ratio, bsz, seqlen, start_pos, offset, device=dev)
            topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()
        if start_pos == 0:
            if seqlen <= win:
                self.kv_cache[:bsz, :seqlen] = kv
            else:
                cutoff = seqlen % win
                self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = kv[:, -win:].split(
                    [win - cutoff, cutoff], dim=1
                )
            if ratio:
                if (kv_compress := self.compressor(x, start_pos)) is not None:
                    kv = torch.cat([kv, kv_compress], dim=1)
            o = sparse_attn(q, kv, self.attn_sink, topk_idxs, self.softmax_scale)
        else:
            self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
            if ratio:
                self.compressor(x, start_pos)
            o = sparse_attn(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        mod.apply_rotary_emb(o[..., -rd:], freqs_cis, True)
        o = o.view(bsz, seqlen, self.n_local_groups, -1)
        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, -1)
        o = torch.einsum("bsgd,grd->bsgr", o, wo_a.to(o.dtype))
        return self.wo_b(o.flatten(2))

    mod.Attention.forward = _attention_forward_patched

    def _indexer_forward_patched(self, x, qr, start_pos, offset):
        bsz, seqlen, _ = x.size()
        dev = x.device
        freqs_cis = self.freqs_cis[start_pos : start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis
        q = self.wq_b(qr).unflatten(-1, (self.n_local_heads, self.head_dim))
        mod.apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = mod.rotate_activation(q)
        mod.fp4_act_quant(q, mod.fp4_block_size, True)
        self.compressor(x, start_pos)
        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads**-0.5)
        index_score = torch.einsum("bshd,btd->bsht", q, self.kv_cache[:bsz, : end_pos // ratio].to(q.dtype))
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
        if start_pos == 0:
            rows = torch.arange(seqlen // ratio, device=dev).unsqueeze(0).repeat(seqlen, 1)
            cols = torch.arange(1, seqlen + 1, device=dev).unsqueeze(1) // ratio
            mask = rows >= cols
            index_score = index_score + torch.where(mask, float("-inf"), torch.zeros(1, device=dev))
        topk_idxs = index_score.topk(min(self.index_topk, end_pos // ratio), dim=-1)[1]
        if start_pos == 0:
            mask2 = topk_idxs >= torch.arange(1, seqlen + 1, device=dev).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask2, -1, topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset
        return topk_idxs

    mod.Indexer.forward = _indexer_forward_patched
    return mod


def load_tokenizer(model_dir: Path):
    """Load the checkpoint tokenizer in a transformers-version-robust way.

    ``AutoTokenizer.from_pretrained`` routes through ``AutoConfig``, which fails
    on DeepSeek-V4-Pro under newer transformers because the model_type is not
    registered and the rope config is not standard. Fall back to a direct
    ``PreTrainedTokenizerFast`` over ``tokenizer.json`` so it works on any
    transformers version.

    :param Path model_dir: Checkpoint directory containing ``tokenizer.json``.

    :return: A loaded tokenizer.
    """
    try:
        return AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    except Exception:
        return PreTrainedTokenizerFast(tokenizer_file=str(model_dir / "tokenizer.json"))


def reset_kv_cache(model):
    """Zero/clear the model's KV caches between forward passes.

    kv_cache buffers can become "inference tensors" (blocks moved with ``.to()``
    inside ``@torch.inference_mode()``). In-place ``zero_()`` on them is only
    allowed inside inference_mode, so wrap the reset.
    """
    with torch.inference_mode():
        for mod in model.modules():
            if hasattr(mod, "kv_cache") and mod.kv_cache is not None:
                mod.kv_cache.zero_()
            # The compressor cache is re-created on the next forward, so clearing
            # the reference is enough (no need to zero it first).
            if hasattr(mod, "compressor") and hasattr(mod.compressor, "kv_cache"):
                mod.compressor.kv_cache = None


@torch.no_grad()
def ppl_eval(model, testenc_ids, device, batch_size=1, seqlen=2048):
    """Compute wikitext-2 perplexity by teacher-forcing fixed-length chunks.

    :param nn.Module model: The wrapped (dequant-on-forward) model.
    :param torch.Tensor testenc_ids: ``[1, total_tokens]`` tokenized test text.
    :param torch.device device: Compute device.
    :param int batch_size: Chunks per forward.
    :param int seqlen: Chunk length.

    :return: Perplexity (float).
    """
    model.eval()
    total_tokens = testenc_ids.numel()
    nsamples = total_tokens // seqlen
    chunks = testenc_ids[0, : nsamples * seqlen].view(nsamples, seqlen)
    nlls = []
    loss_fct = torch.nn.CrossEntropyLoss()
    t0 = time.time()
    for i in tqdm(range(0, nsamples, batch_size), desc="PPL eval"):
        batch_chunks = chunks[i : i + batch_size]
        bs = batch_chunks.shape[0]
        reset_kv_cache(model)
        logits = model(batch_chunks, start_pos=0)
        batch_nll = 0.0
        for b in range(bs):
            seq_logits = logits[b, :-1, :].contiguous()
            seq_labels = batch_chunks[b, 1:]
            loss = loss_fct(seq_logits, seq_labels)
            batch_nll = batch_nll + loss.float() * seqlen
        nlls.append(batch_nll)
    elapsed = time.time() - t0
    total_nll = torch.stack(nlls).sum()
    ppl = torch.exp(total_nll / (nsamples * seqlen)).item()
    logger.info(f"Elapsed: {elapsed:.1f}s  ({nsamples} chunks, {elapsed / nsamples:.2f}s/chunk)")
    return ppl


def install_block_offload(model, device, n_gpu_blocks=0):
    """Per-block GPU streaming.

    DequantLinear stores its quantized weights as *buffers* (uint8 / fp8 cannot
    be nn.Parameter), so we move whole decoder blocks to the GPU around their
    forward ourselves: pre-hook -> block.to(device); post-hook -> block.to('cpu').
    Only one block is resident at a time. Leading ``n_gpu_blocks`` blocks are
    pinned on GPU (no hooks).
    """
    layers = model.layers
    n = len(layers)
    n_gpu_blocks = max(0, min(n_gpu_blocks, n))
    for i in range(n_gpu_blocks):
        layers[i].to(device)

    def _make(blk):
        def pre(mod, inp):
            mod.to(device)

        def post(mod, inp, out):
            mod.to("cpu")
            torch.cuda.empty_cache()

        return pre, post

    for i in range(n_gpu_blocks, n):
        pre, post = _make(layers[i])
        layers[i].register_forward_pre_hook(pre)
        layers[i].register_forward_hook(post)
    logger.info(f"block offload: {n_gpu_blocks} pinned on GPU, {n - n_gpu_blocks} streamed per-forward")


# ---------------------------------------------------------------------------
# DequantLinear + load_and_wrap: load a compact quantized checkpoint and expand
# each weight to BF16 on the fly. Shared by stage4_ppl (NVFP4 output) and
# ppl_baseline (the original MXFP4 checkpoint); the `scheme` arg selects the
# on-disk layout.
# ---------------------------------------------------------------------------


class DequantLinear(nn.Module):
    """Drop-in for DS-V4 Linear / Column / RowParallelLinear.

    Holds the compact quantized weight + scale(s) as registered buffers so they
    move GPU<->CPU per block. On forward (and on direct ``.weight`` access, e.g.
    attention's wo_a) it dequantizes to BF16.

    fmt:
      "nvfp4" : qweight U8 [out,in/2], wscale F8_E4M3 [out,in/16], wscale2 F32 []
      "mxfp4" : qweight I8 [out,in/2], wscale F8_E8M0 [out,in/32]   (wscale2 unused)
      "fp8"   : qweight F8_E4M3 [out,in], wscale F8_E8M0 [out/128,in/128]
      "bf16"  : qweight BF16 [out,in], no scale
    """

    def __init__(self, fmt, qweight, wscale, wscale2, out_features, in_features, bias, row_parallel, input_scale=None):
        super().__init__()
        self.fmt = fmt
        self.out_features = out_features
        self.in_features = in_features
        self.row_parallel = row_parallel
        self.register_buffer("qweight", qweight, persistent=True)
        self.register_buffer("wscale", wscale, persistent=True)
        self.register_buffer("wscale2", wscale2, persistent=True)
        # Per-tensor NVFP4 activation scale (Stage 2). Present only for nvfp4
        # experts and only when activation quantization is on; ``None`` means the
        # activation stays BF16.
        if input_scale is not None:
            self.register_buffer("input_scale", input_scale.to(torch.float32), persistent=True)
        else:
            self.input_scale = None
        if bias is not None:
            self.register_buffer("bias", bias.to(torch.bfloat16), persistent=True)
        else:
            self.bias = None

    def _dequant(self, device) -> torch.Tensor:
        if self.fmt == "nvfp4":
            return dequant_nvfp4_weight(self.qweight, self.wscale, self.wscale2, device)
        if self.fmt == "mxfp4":
            return dequant_mxfp4_weight(self.qweight, self.wscale, device)
        if self.fmt == "fp8":
            return dequant_fp8_block_weight(self.qweight, self.wscale, device)
        return self.qweight.to(device=device, dtype=torch.bfloat16)

    @property
    def weight(self) -> torch.Tensor:
        # Direct-access paths (attention wo_a) read .weight; dequant on the
        # buffer's current device.
        return self._dequant(self.qweight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        w = self._dequant(device)
        xb = x.to(torch.bfloat16)
        # NVFP4: quantize the activation too (per-group-16 FP4 + per-tensor
        # input_scale), matching real NVFP4 inference. When input_scale is None
        # the activation stays BF16.
        if self.fmt == "nvfp4" and self.input_scale is not None:
            xb = quant_dequant_nvfp4_act(xb, self.input_scale.to(device))
        y = F.linear(xb, w)
        # DeepSeek-V4-Pro builds every Linear with bias=False (inference/model.py's
        # `linear()` asserts `bias is None`), so only the row-parallel reduce path
        # ever carries a bias here; a non-row-parallel bias would be dropped, but
        # none exists in this checkpoint.
        if self.row_parallel and self.bias is not None:
            y = y + self.bias.to(y.dtype)
        del w
        # The original ``linear()`` returns BF16 from the fp8/fp4 GEMM kernels
        # (downstream code, e.g. rotate_activation, asserts bf16). For native
        # bf16 weights it would return x.dtype -- mimic both.
        if self.fmt == "bf16":
            return y.to(x.dtype)
        return y.to(torch.bfloat16)


def load_and_wrap(model: nn.Module, hf_path: str, scheme: str, quant_act: bool = False) -> tuple[int, int]:
    """Replace every Linear-family module with a DequantLinear holding the matching
    compact quantized tensors read from the checkpoint. Non-linear params/buffers
    (norms, embed, head, gate, attn_sink, freqs, etc.) are loaded directly as
    BF16/native tensors.

    :param nn.Module model: The freshly built (BF16) DeepSeek-V4-Pro model.
    :param str hf_path: Path to the checkpoint directory.
    :param str scheme: ``"nvfp4"`` (this pipeline's output: ``weight_scale`` +
        ``weight_scale_2``) or ``"mxfp4"`` (the original checkpoint: ``scale``).
    :param bool quant_act: When True (nvfp4 only), also load each expert's
        ``input_scale`` and quantize activations to NVFP4 on forward. When False,
        the activation stays BF16.

    :return: ``(n_wrapped_linears, n_assigned_non_linear)``.
    """
    if scheme not in ("nvfp4", "mxfp4"):
        raise ValueError(f"scheme must be 'nvfp4' or 'mxfp4', got {scheme!r}")
    quant_act = quant_act and scheme == "nvfp4"
    # NVFP4 weights are stored U8 with two scales (per-group weight_scale + global
    # weight_scale_2); the original MXFP4 weights are I8 (or float4_e2m1fn_x2) with
    # a single per-group scale named just "scale".
    fp4_fmt = scheme
    scale_key = "weight_scale" if scheme == "nvfp4" else "scale"
    scale2_key = "weight_scale_2"  # nvfp4 only

    hf_path = Path(hf_path)
    weight_map = json.loads((hf_path / "model.safetensors.index.json").read_text())["weight_map"]

    modules = dict(model.named_modules())
    param_keys = {n for n, _ in model.named_parameters()}
    buffer_keys = {n for n, _ in model.named_buffers()}

    linear_names = {name for name, m in modules.items() if type(m).__name__ in _LINEAR_CLASSES}

    def linear_keys(name):
        ks = [f"{name}.weight"]
        if f"{name}.{scale_key}" in weight_map:
            ks.append(f"{name}.{scale_key}")
        if scheme == "nvfp4" and f"{name}.{scale2_key}" in weight_map:
            ks.append(f"{name}.{scale2_key}")
        if quant_act and f"{name}.input_scale" in weight_map:
            ks.append(f"{name}.input_scale")
        if f"{name}.bias" in weight_map:
            ks.append(f"{name}.bias")
        return ks

    wanted: set[str] = set()
    for name in linear_names:
        for k in linear_keys(name):
            if k in weight_map:
                wanted.add(k)
    for k in param_keys | buffer_keys:
        if k in weight_map:
            wanted.add(k)

    by_shard: dict[str, list[str]] = {}
    for k in wanted:
        by_shard.setdefault(weight_map[k], []).append(k)
    logger.info(f"reading {len(wanted)} tensors from {len(by_shard)} shard(s) …")
    loaded: dict[str, torch.Tensor] = {}
    for shard in sorted(by_shard):
        with safe_open(str(hf_path / shard), framework="pt", device="cpu") as f:
            avail = set(f.keys())
            for k in by_shard[shard]:
                if k in avail:
                    loaded[k] = f.get_tensor(k)

    n_wrapped = 0
    n_act_quant = 0
    for name in sorted(linear_names):
        m = modules[name]
        wkey = f"{name}.weight"
        if wkey not in loaded:
            continue
        w = loaded[wkey]
        wscale = loaded.get(f"{name}.{scale_key}")
        wscale2 = loaded.get(f"{name}.{scale2_key}")
        input_scale = loaded.get(f"{name}.input_scale") if quant_act else None
        bias = loaded.get(f"{name}.bias")
        out_features = getattr(m, "out_features", w.shape[0])
        in_features = getattr(m, "in_features", w.shape[1] if w.dim() > 1 else 1)

        if w.dtype == torch.uint8 or w.dtype == torch.int8 or w.dtype == torch.float4_e2m1fn_x2:
            fmt = fp4_fmt
            in_features = w.shape[1] * 2
            wscale2 = wscale2 if wscale2 is not None else torch.tensor(1.0)
        elif w.dtype == torch.float8_e4m3fn:
            fmt = "fp8"
            wscale2 = torch.tensor(0.0)
            input_scale = None  # activation quant only defined for nvfp4 experts
        else:
            fmt = "bf16"
            w = w.to(torch.bfloat16)
            wscale = torch.tensor(0.0)
            wscale2 = torch.tensor(0.0)
            input_scale = None
        if wscale is None:
            wscale = torch.tensor(0.0)

        row_parallel = type(m).__name__ == "RowParallelLinear"
        wrapper = DequantLinear(
            fmt, w, wscale, wscale2, out_features, in_features, bias, row_parallel, input_scale=input_scale
        )
        if input_scale is not None:
            n_act_quant += 1

        parent_name, _, child = name.rpartition(".")
        parent = modules[parent_name] if parent_name else model
        setattr(parent, child, wrapper)
        n_wrapped += 1

    # Refresh module map (wrappers replaced children) for non-linear assignment.
    modules = dict(model.named_modules())

    n_assigned = 0
    for full_key, tensor in loaded.items():
        base = full_key.rsplit(".", 1)[0]
        if base in linear_names:
            continue
        parts = full_key.split(".")
        mod = model
        ok = True
        for p in parts[:-1]:
            if not hasattr(mod, p):
                ok = False
                break
            mod = getattr(mod, p)
        if not ok:
            continue
        attr = parts[-1]
        if hasattr(mod, "_parameters") and attr in mod._parameters and mod._parameters[attr] is not None:
            existing = mod._parameters[attr]
            t = tensor
            if not existing.is_meta and t.is_floating_point() and existing.dtype != t.dtype:
                t = t.to(existing.dtype)
            mod._parameters[attr] = nn.Parameter(t, requires_grad=False)
            n_assigned += 1
        elif hasattr(mod, "_buffers") and attr in mod._buffers:
            mod._buffers[attr] = tensor
            n_assigned += 1
    logger.info(f"wrapped {n_wrapped} linears; assigned {n_assigned} non-linear tensors")
    if quant_act:
        if n_act_quant == 0:
            raise ValueError(
                "activation quantization requested but no expert input_scale found in the checkpoint; "
                "run Stage 2 (calibrate) + Stage 3 (merge) first, or pass --no-quant-act."
            )
        logger.info(f"activation NVFP4 quant ON for {n_act_quant} expert linears (input_scale applied)")
    return n_wrapped, n_assigned
