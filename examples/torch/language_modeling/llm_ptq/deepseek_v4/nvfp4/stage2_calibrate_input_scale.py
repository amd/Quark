#
# Copyright (C) 2023 - 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Loads and runs the DeepSeek-V4-Pro checkpoint's model code (inference/model.py).
# Copyright (c) 2023 DeepSeek. Licensed under the MIT License.
#

"""
Stage 2: NVFP4 input (activation) scale calibration + safetensors export for
DeepSeek-V4-Pro.

This script:
  1. Calibrates the NVFP4 INPUT (activation) quantizers with the real NVFP4
     input config (FP4 per-group dynamic + FP8E4M3 per-tensor static); input
     only, weight=None (saves GPU memory). No PPL.
  2. Converts the collected per-tensor input_scale into the NVFP4 safetensors
     tensor layout:
        layers.<L>.ffn.experts.<E>.<wK>.input_scale          (routed, F32 scalar)
     so it can be merged (Stage 3) into the NVFP4 weights from Stage 1.

Only the routed experts are quantized to NVFP4; the shared experts keep their
original FP8 format, so no input_scale is calibrated for them.

Missing experts: with few calibration tokens some routed experts never fire.
Each missing (layer, projection) is filled with the MAX input_scale over the
calibrated experts in that same layer+projection (a larger per-tensor scale
never clips a rare expert).

Calibration data: mirrors the NVIDIA ModelOpt deepseek_v4 recipe — two sources,
cnn_dailymail (news articles) and Nemotron-Post-Training-Dataset-v2 (multi-turn
chat, message contents joined). ``--n-calib-samples`` is the number of rows taken
from EACH source (= ModelOpt's ``num_samples=[calib_size]*len(datasets)``), so the
default 64 yields 128 calibration rows total. If a source cannot be loaded the run
falls back to wikitext-2.

Architecture:
  NativeLinear wraps each native FP4/FP8 linear in the checkpoint.
  Its `self.proxy` is a plain nn.Linear that Quark replaces with QuantLinear.
  On forward(), NativeLinear dequantizes its stored weights to BF16 transiently
  so the input observer sees real activations; weights are NOT quantized.
  Only one block's BF16 weights exist on GPU at a time.

Usage:
    CUDA_VISIBLE_DEVICES=0 python stage2_calibrate_input_scale.py \\
        --model-dir /path/to/DeepSeek-V4-Pro \\
        --batch-size 48 --n-calib-samples 64 \\
        --output input_scale.safetensors
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from datasets import load_dataset
from safetensors.torch import save_file
from tqdm import tqdm

from quark.common.utils.log import ScreenLogger
from quark.torch import ModelQuantizer
from quark.torch.quantization.config.config import QConfig, QLayerConfig
from quark.torch.quantization.config.template import QuantizationSchemeCollection
from quark.torch.quantization.nn.modules.quantize_linear import QuantMixin
from quark.torch.quantization.tensor_quantize import ScaledFakeQuantize
from quark.torch.utils.per_block_runner.lazy_loader import prepare

logger = ScreenLogger(__name__)

# Put this example's src/ folder and its own directory on sys.path so the support
# modules and the sibling Stage 1 script (for the two default exclude lists) import.
_HERE = str(Path(__file__).parent)
_SRC_DIR = str(Path(__file__).parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# dsv4_common imports `dsv4_kernels`, which registers the synthetic ``kernel`` /
# ``fast_hadamard_transform`` modules so the checkpoint model.py picks up the
# triton/PyTorch implementations instead of tilelang.
from dsv4_collect import build_input_scale_tensors, collect_input_minmax  # noqa: E402
from dsv4_common import load_model_module, load_tokenizer, reset_kv_cache  # noqa: E402
from dsv4_native_linear import NativeLinear, load_all_weights_native, wrap_native_linears  # noqa: E402
from dsv4_offload import install_disk_offload_hooks  # noqa: E402
from stage1_quantize_weight import (  # noqa: E402
    EXCLUDE_LAYERS_DEFAULT,
    KEEP_ORIGINAL_FORMAT_LAYERS_DEFAULT,
)

# Input spec from Quark's built-in nvfp4 scheme (no model-specific template needed).
_NVFP4_INPUT_SPEC = QuantizationSchemeCollection().get_scheme("nvfp4").config.input_tensors


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Stage 2: calibrate the NVFP4 activation input_scale for DeepSeek-V4-Pro "
            "and export it as a safetensors file to be merged by Stage 3."
        )
    )
    p.add_argument(
        "--model-dir",
        required=True,
        help="Source (un-quantized) DeepSeek-V4-Pro checkpoint dir; must contain the inference/ model code.",
    )
    p.add_argument(
        "--output",
        default="input_scale.safetensors",
        help="Where to write the per-expert *.input_scale safetensors, NVFP4 HF layout (default: %(default)s).",
    )
    p.add_argument("--device", default="cuda:0", help="Single GPU device (default: %(default)s).")
    p.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Calibration batch size; lower it if the run runs out of GPU memory (default: %(default)s).",
    )
    p.add_argument(
        "--seqlen",
        type=int,
        default=2048,
        help="Model max sequence length (max_seq_len); must be >= --calib-seqlen (default: %(default)s).",
    )
    p.add_argument(
        "--calib-seqlen",
        type=int,
        default=512,
        help="Token length of each calibration chunk; must be <= --seqlen (default: %(default)s).",
    )
    p.add_argument(
        "--n-calib-samples",
        type=int,
        default=64,
        help="Calibration rows (seqlen-long chunks) taken from EACH calibration source "
        "(cnn_dailymail + Nemotron-Post-Training-Dataset-v2), mirroring ModelOpt's "
        "num_samples=[calib_size]*len(datasets); default %(default)s => 128 rows total.",
    )
    p.add_argument(
        "--n-experts-per-layer",
        type=int,
        default=384,
        help="Expected routed experts per layer (DeepSeek-V4-Pro = 384). Used only as a "
        "sanity cross-check on the emitted input_scale set; a mismatch logs a warning "
        "but never changes the output (default: %(default)s).",
    )
    p.add_argument(
        "--exclude_layers",
        nargs="+",
        default=EXCLUDE_LAYERS_DEFAULT,
        metavar="PATTERN",
        help="fnmatch patterns (matched against the module name) for modules that are "
        "TRULY NOT quantized (16-bit / BF16 layers: embeddings, output head, norms, router "
        "gate, MTP block). MUST mirror Stage 1's --exclude_layers so weight quantization and "
        "input_scale calibration cover the same experts (default: %(default)s).",
    )
    p.add_argument(
        "--keep_original_format_layers",
        nargs="+",
        default=KEEP_ORIGINAL_FORMAT_LAYERS_DEFAULT,
        metavar="PATTERN",
        help="fnmatch patterns (matched against the module name) for modules kept in the "
        "model's ORIGINAL quantized format (passthrough, e.g. FP8 attention and shared "
        "experts): not re-quantized to NVFP4 and not calibrated. MUST mirror Stage 1's "
        "--keep_original_format_layers. Drop '*shared_experts*' to instead quantize + "
        "calibrate the shared experts (default: %(default)s).",
    )
    p.add_argument(
        "--n-blocks",
        type=int,
        default=None,
        help="Limit to the first N decoder blocks for a smoke test; Stage 1's input and Stage 4 "
        "must use the same N (default: all layers).",
    )
    p.add_argument(
        "--n-gpu-blocks",
        type=int,
        default=0,
        help="Number of leading decoder blocks pinned on the GPU; the rest are streamed one at a "
        "time (default: %(default)s).",
    )
    p.add_argument(
        "--ram-budget",
        type=int,
        default=None,
        help="Max CPU RAM in GB for calibration block state; blocks that don't fit spill to disk "
        "(default: auto, ~80%% of MemAvailable).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    model_dir = Path(args.model_dir)

    header = (
        "NVFp4 input min/max/scale collection — DeepSeek-V4-Pro (weight not quantized)\n"
        f"  model_dir      : {model_dir}\n"
        f"  device         : {device}\n"
        f"  batch_size     : {args.batch_size}\n"
        f"  seqlen         : {args.seqlen}\n"
        f"  n_gpu_blocks   : {args.n_gpu_blocks}\n"
        f"  n_calib_samples: {args.n_calib_samples}  calib_seqlen: {args.calib_seqlen}"
    )
    if args.n_blocks:
        header += f"\n  n_blocks       : {args.n_blocks} (smoke-test)"
    logger.info(header)

    # ------------------------------------------------------------------
    # [1] Build Transformer on CPU
    # ------------------------------------------------------------------
    logger.info("[1] Loading model module …")
    _mod = load_model_module(model_dir)
    Transformer = _mod.Transformer
    ModelArgs = _mod.ModelArgs

    with open(model_dir / "inference" / "config.json") as f:
        cfg = json.load(f)

    cfg["dtype"] = "fp8"
    cfg["scale_dtype"] = "fp8"
    cfg["expert_dtype"] = "fp4"
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

    # ------------------------------------------------------------------
    # [3] Load native weights (FP4/FP8) into CPU RAM
    # ------------------------------------------------------------------
    logger.info("[3] Loading native weights (FP4/FP8) into CPU RAM …")
    load_all_weights_native(model, str(model_dir))

    # ------------------------------------------------------------------
    # [4] Wrap native FP4/FP8 linears with NativeLinear
    #     Each wrapper stores the native weight on CPU and exposes a BF16
    #     proxy nn.Linear for Quark to replace with QuantLinear.
    # ------------------------------------------------------------------
    # Both groups are skipped during calibration: truly-not-quantized (16-bit) and
    # kept-original-format (FP8 passthrough). Combine into one exclude set so the same
    # modules Stage 1 left un-NVFP4'd are also left uncalibrated here.
    combined_exclude = args.exclude_layers + args.keep_original_format_layers
    logger.info(f"[4] Wrapping native linears with NativeLinear (exclude_layers={combined_exclude}) …")
    n_wrapped = wrap_native_linears(model, exclude_layers=combined_exclude)
    logger.info(f"Wrapped {n_wrapped} linear modules.")

    # ------------------------------------------------------------------
    # [5] Quark: replace proxy nn.Linear -> QuantLinear (NVFp4)
    #     Weights are on CPU (real BF16 placeholders), so QuantLinear
    #     buffers allocate on CPU without meta-device issues.
    # ------------------------------------------------------------------
    logger.info("[5] Quark: replace proxy nn.Linear -> QuantLinear (NVFp4 input observer) …")

    # NVFp4 input quantization config:
    #   first_stage : FP4PerGroup(group_size=16, dynamic) — per-group micro-scale
    #   second_stage: FP8E4M3PerTensor(static, min_max)   — per-tensor global scale
    # We quantize the INPUT only; weight=None (no weight quant → saves GPU mem).
    # After calibration we read the input observers' min/max AND the resulting
    # scale (both stages) and dump them to disk.
    # Only the non-excluded MoE experts are wrapped with NativeLinear, so only their
    # proxies are nn.Linear → QuantLinear. DS-V4's other linears inherit nn.Module and
    # stay invisible to Quark; the default --keep_original_format_layers also skips the
    # shared experts so they keep their FP8 format.
    # Sanity check (log only, not used by the quantization): count how many
    # expert projections got wrapped as NativeLinear — i.e. how many will be
    # calibrated. Use this to confirm the wrap step matched the expected count
    # (e.g. one layer = 384*3 routed projections).
    n_proxies = sum(1 for _, m in model.named_modules() if isinstance(m, NativeLinear))
    logger.info(f"MoE expert proxies to observe (NVFp4 input): {n_proxies}")

    quant_config = QConfig(
        global_quant_config=QLayerConfig(
            input_tensors=_NVFP4_INPUT_SPEC,
            output_tensors=None,
            weight=None,
        ),
    )
    quantizer = ModelQuantizer(quant_config)

    # Wrap model so Quark's forward uses the correct signature.
    class _ModelWrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids):
            reset_kv_cache(self.inner)
            return self.inner(input_ids, start_pos=0)

    wrapped = _ModelWrapper(model)

    # NOTE: this uses two private Quark entry points (`_prepare_model` here and
    # `_calibrate_all_params` no-op'd below) instead of the public
    # `quantizer.quantize_model(model, dataloader)`. The public path assumes a
    # standard model it can run forward on directly; here the model is a custom
    # DeepSeek checkpoint on the meta device with per-block lazy offload and a
    # `forward(ids, start_pos)` signature, so calibration is driven manually
    # (see [7] below). If Quark refactors these internals this example must be
    # updated; there is currently no public "prepare graph + calibrate input only,
    # skip weight, drive forward myself" API.
    #
    # _prepare_model: replaces proxy nn.Linear -> QuantLinear (no forward pass).
    wrapped = quantizer._prepare_model(wrapped)
    model = wrapped.inner

    # ------------------------------------------------------------------
    # [6] Install lazy_loader (CPU RAM mode)
    #     After prepare(), block params are on meta device.
    #     _calibrate_all_params is patched to no-op because it tries to
    #     access meta-device weights via HF hooks (not available here).
    #     Weight calibration happens naturally during calib forward passes.
    # ------------------------------------------------------------------
    logger.info("[6] Installing lazy_loader (CPU RAM mode) …")
    prepare(model, str(model_dir), target_device=device, decoder_layers_path="layers", n_gpu_blocks=args.n_gpu_blocks)

    _gpu_block_prefixes = tuple(f"layers.{i}." for i in range(args.n_gpu_blocks))
    for mod_name, mod in model.named_modules():
        in_block = mod_name == "layers" or mod_name.startswith("layers.")
        in_gpu_block = any(mod_name.startswith(p) for p in _gpu_block_prefixes) if _gpu_block_prefixes else False
        if not in_block:
            for pname, param in list(mod._parameters.items()):
                if param is not None and not param.is_meta:
                    mod._parameters[pname] = nn.Parameter(param.to(device), requires_grad=param.requires_grad)
        # Quantizer stage buffers (scale, zero_point, min_val, max_val) inside
        # offloaded blocks stay on CPU; NativeLinear.forward() moves them
        # GPU↔CPU on demand.  GPU-resident block buffers go to device normally.
        is_quant_stage = isinstance(mod, ScaledFakeQuantize)
        if in_block and not in_gpu_block and is_quant_stage:
            continue
        for bname, buf in list(mod._buffers.items()):
            if buf is not None and not buf.is_meta:
                mod._buffers[bname] = buf.to(device)

    for mod in model.modules():
        if type(mod).__name__ == "Indexer" and hasattr(mod, "kv_cache") and mod.kv_cache is not None:
            mod.kv_cache = mod.kv_cache.to(torch.bfloat16)

    quantizer._calibrate_all_params = lambda *a, **kw: None

    # ------------------------------------------------------------------
    # [6b] Install disk offload hooks for blocks that exceed RAM budget.
    #      Blocks within budget keep state in CPU RAM (fast); excess blocks
    #      spill observer/NativeLinear state to disk between forward calls.
    # ------------------------------------------------------------------
    n_disk, offload_dir, n_ram_blocks = install_disk_offload_hooks(model, args.ram_budget)
    if n_disk > 0:
        logger.info(f"{n_disk} blocks will use disk offload for observer state")

    # ------------------------------------------------------------------
    # [6c] Tokenizer.
    # ------------------------------------------------------------------

    tokenizer = load_tokenizer(model_dir)

    # ------------------------------------------------------------------
    # [7] Calibration: forward passes with calib data to collect input
    #     activation min/max for Quark's input observers.
    #
    #     During each forward pass through a block, NativeLinear.forward()
    #     dequants FP4→BF16 so the input observer sees real activations.
    #     Weights are NOT quantized (weight=None). We enable the input
    #     observer only (no fake-quant) — we just need min/max statistics.
    # ------------------------------------------------------------------
    logger.info("[7] Calibration …")

    def _iter_fq_stages(mod):
        for fq in (mod._weight_quantizer, mod._input_quantizer, mod._output_quantizer):
            if fq is None:
                continue
            stages = [fq] if isinstance(fq, ScaledFakeQuantize) else list(fq)
            yield from stages

    # [7a] Enable observers + fake-quant on the NVFp4 input quantizers so the
    #      activations follow the real NVFp4 quantization path and the static
    #      per-tensor scale is computed exactly as in production.
    for mod in model.modules():
        if not isinstance(mod, QuantMixin):
            continue
        for stage in _iter_fq_stages(mod):
            if hasattr(stage, "enable_observer"):
                stage.enable_observer()
            if hasattr(stage, "enable_fake_quant"):
                stage.enable_fake_quant()

    # [7b] Load calibration data (tokenizer already loaded in [6c]).
    # Mirror the NVIDIA ModelOpt deepseek_v4 recipe: two sources, cnn_dailymail
    # (article) + Nemotron-Post-Training-Dataset-v2 (messages joined), each
    # contributing `n_calib_samples` rows (num_samples=[N]*len(datasets)).
    per = args.n_calib_samples
    seqlen = args.calib_seqlen

    def _chunks_from_text(text: str) -> torch.Tensor:
        ids = tokenizer(text, return_tensors="pt", truncation=False).input_ids
        avail_rows = ids.shape[1] // seqlen
        k = min(per, avail_rows)
        return ids[0, : k * seqlen].view(k, seqlen) if k > 0 else ids.new_empty((0, seqlen))

    def _load_cnn_text() -> str:
        ds = load_dataset("cnn_dailymail", name="3.0.0", split="train")
        return "\n\n".join(ds["article"][: per * 4])  # 4x oversample for chunking

    def _load_nemotron_text() -> str:
        # Stream the four post-training splits; join each conversation's message
        # contents (ModelOpt's _join_messages_content). Stop once we have enough
        # docs to fill `per` rows after chunking (4x oversample).
        texts: list[str] = []
        for split in ("stem", "chat", "math", "code"):
            try:
                ds = load_dataset("nvidia/Nemotron-Post-Training-Dataset-v2", split=split, streaming=True)
            except Exception as e:  # split missing / load error: skip it
                logger.warning(f"nemotron split '{split}' unavailable: {e}")
                continue
            for row in ds:
                msgs = row.get("messages")
                if msgs:
                    texts.append("\n".join(t["content"] for t in msgs))
                    if len(texts) >= per * 4:
                        break
            if len(texts) >= per * 4:
                break
        if not texts:
            raise RuntimeError("no nemotron rows collected")
        return "\n\n".join(texts)

    sources = (("cnn_dailymail", _load_cnn_text), ("nemotron-post-training-dataset-v2", _load_nemotron_text))
    chunk_list: list[torch.Tensor] = []
    names: list[str] = []
    for name, loader in sources:
        try:
            chunks = _chunks_from_text(loader())
        except Exception as e:
            logger.warning(f"calib source '{name}' failed, skipping: {e}")
            continue
        if chunks.shape[0] > 0:
            chunk_list.append(chunks)
            names.append(f"{name}({chunks.shape[0]})")

    if not chunk_list:  # both sources failed -> fall back to wikitext-2
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
        chunks = _chunks_from_text("\n\n".join(ds["text"]))
        chunk_list.append(chunks)
        names.append(f"wikitext-2({chunks.shape[0]})")

    calib_chunks = torch.cat(chunk_list, dim=0).to(device)
    n_calib = calib_chunks.shape[0]
    used_tokens = n_calib * seqlen
    calib_dataset_name = " + ".join(names)
    logger.info(f"Calib dataset   : {calib_dataset_name}")
    logger.info(f"Calib samples   : {n_calib} x seqlen {seqlen} = {used_tokens:,} tokens used")
    logger.info(f"Calib batches   : {-(-n_calib // args.batch_size)} (batch_size={args.batch_size})")

    # [7c] Run calibration forward passes.
    calib_bs = args.batch_size

    model.eval()
    t_calib = time.time()
    with torch.no_grad():
        for i in tqdm(range(0, n_calib, calib_bs), desc="Calibration"):
            reset_kv_cache(model)
            model(calib_chunks[i : i + calib_bs], start_pos=0)
            torch.cuda.empty_cache()

    logger.info(f"Calibration done in {time.time() - t_calib:.1f}s  (calib_bs={calib_bs})")

    # ------------------------------------------------------------------
    # [8] Collect per-layer NVFp4 input min/max/scale (no PPL).
    # ------------------------------------------------------------------
    logger.info("[8] Collecting per-layer NVFp4 input min/max/scale …")
    stats = collect_input_minmax(model, offload_dir, n_ram_blocks)
    n_total_proxies = sum(1 for _, m in model.named_modules() if isinstance(m, NativeLinear))
    n_skipped = n_total_proxies - len(stats)
    logger.info(
        f"Collected input stats for {len(stats)} layers ({n_skipped} experts never routed during calibration, skipped)."
    )

    scale_map = {name: s["scale"] for name, s in stats.items() if "scale" in s}
    # Full set of calibrated proxy names (drives which input_scale keys are emitted);
    # includes sparse experts that never fired and so are absent from scale_map.
    wrapped_names = [name for name, m in model.named_modules() if isinstance(m, QuantMixin)]

    # ------------------------------------------------------------------
    # [9] Convert collected input_scale -> NVFp4 safetensors (HF layout).
    # ------------------------------------------------------------------

    logger.info("[9] Converting input_scale -> NVFp4 safetensors …")
    tensors, report = build_input_scale_tensors(scale_map, wrapped_names, args.n_experts_per_layer)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path))
    logger.info(
        f"total  : {report['n_total']} input_scale tensors\n"
        f"scales : {report['n_calibrated']} calibrated, {report['n_filled']} filled (never-fired)\n"
        f"wrote  : {out_path}"
    )
    for tmpl, expected, actual in report.get("expert_count_mismatches", []):
        logger.warning(
            f"expert-count check: template '{tmpl}' emitted {actual} keys, "
            f"expected {expected} (= n_layers x {args.n_experts_per_layer}); "
            "output is unaffected."
        )

    peak_gpu = torch.cuda.max_memory_allocated(device) / 1e9
    summary = [
        f"Done: {report['n_total']} input_scale tensors",
        f"  safetensors : {out_path}",
    ]
    for k, s in list(stats.items())[:3]:
        sc = s.get("scale")
        sc_s = f"{sc:.6g}" if sc is not None else "n/a"
        summary.append(f"    e.g. {k}: amax={s['amax']:.4f} scale={sc_s}")
    summary.append(f"Peak GPU memory        : {peak_gpu:.2f} GB")
    logger.info("\n".join(summary))


if __name__ == "__main__":
    main()
