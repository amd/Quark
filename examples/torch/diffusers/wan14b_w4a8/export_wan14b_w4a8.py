#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Quantize Wan2.2-T2V-A14B (dual-expert) to w4a8 and export packed MXFP4 to safetensors.

w4a8 = MXFP4 weight (per-1x32) + static FP8-e4m3 per-tensor activation (needs a short
calibration forward). A14B has TWO transformers (``transformer``, ``transformer_2``);
each is quantized, (optionally SVDQuant-corrected), frozen, and exported into its own
subfolder as a **packed** real_quantized checkpoint (``.weight`` = float4_e2m1fn_x2,
~1/4 the bf16 size). Other pipeline components are copied so ``WanPipeline.from_pretrained``
loads the whole thing back.

Reload with ``reload_wan14b_w4a8.py``. NOTE: ``from_pretrained`` on a quark export needs
diffusers with the quark auto-quantizer (PR #14077 / the bundled integration).

``--model`` takes either a local diffusers checkpoint directory or a HF repo id (which is
resolved with ``snapshot_download`` first, since the pipeline components are copied from
disk).

Usage:
  PYTHONPATH=<quark-repo> python3 export_wan14b_w4a8.py \
      --model Wan-AI/Wan2.2-T2V-A14B-Diffusers --out wan14b_w4a8_export [--svd]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil

import torch
from svd_correction import mark_correction_required, save_svd_correction, unwrap_error_corrected

DEFAULT_MODEL = os.environ.get("WAN_MODEL", "Wan-AI/Wan2.2-T2V-A14B-Diffusers")
DEFAULT_OUT = os.environ.get("OUT", "wan14b_w4a8_export")
EXPERTS = ("transformer", "transformer_2")

# Structural layers kept in bf16 (embedders / norms / final proj).
WAN_EXCLUDE = ["*time_embedder*", "*patch_embedding*", "*condition_embedder*", "*norm*", "*proj_out*"]
# SVDQuant skips the same structural layers. NOTE: SVDQuantConfig.exclude_patterns is
# matched with fnmatch, so these MUST be globs -- bare substrings match nothing, and
# SVD would then wrap e.g. condition_embedder.time_embedder.linear_1 (which Wan keeps in
# float32) in an ErrorCorrectedModule whose correction is float32 while the runtime
# activation is bf16 -> "self and mat2 must have the same dtype" at the calib forward.
SVD_EXCLUDE = list(WAN_EXCLUDE)


def build_qconfig(svd: bool):
    from quark.torch.quantization.config.config import QConfig, SVDQuantConfig
    from quark.torch.quantization.config.template import QuantizationSchemeCollection

    lc = QuantizationSchemeCollection().get_scheme("mxfp4_fp8").config  # mxfp4 wt + static fp8 act
    algo = None
    exclude = list(WAN_EXCLUDE)
    if svd:
        algo = [SVDQuantConfig(svd_rank=32, search_alpha=False, exclude_patterns=list(SVD_EXCLUDE))]
        exclude = [*exclude, "*correction*"]
    return QConfig(
        global_quant_config=lc, layer_type_quant_config={}, layer_quant_config={}, exclude=exclude, algo_config=algo
    )


def resolve_model_dir(model: str) -> str:
    """Return a local directory for ``--model``, fetching a HF repo id if that is what it is.

    The component copy below is a filesystem copy, so a repo id ("Wan-AI/Wan2.2-T2V-A14B-
    Diffusers") would make every ``os.path.isdir`` miss and produce an export with the
    transformers but no vae / text_encoder / model_index.json -- unloadable, with no error.
    Resolving to the snapshot directory up front also means ``from_pretrained`` and the copy
    read the same files.
    """
    if os.path.isdir(model):
        return model
    from huggingface_hub import snapshot_download

    print(f"[export] --model is not a local directory; snapshot_download({model}) ...", flush=True)
    return snapshot_download(model)


def copy_pipeline_components(src_dir: str, out_dir: str, skip: tuple[str, ...]) -> list[str]:
    """Copy every pipeline component listed in ``model_index.json`` except ``skip``.

    Driven by the index rather than a hardcoded list, so variants with extra components
    (an I2V pipeline's ``image_encoder`` / ``image_processor``) are carried over instead of
    being dropped silently. Entries are ``[library, class_name]``; scalar entries are plain
    config values (e.g. ``boundary_ratio``) with no subfolder, and ``[null, null]`` marks a
    component the checkpoint does not ship.
    """
    index = os.path.join(src_dir, "model_index.json")
    if not os.path.isfile(index):
        raise FileNotFoundError(f"{src_dir} has no model_index.json -- is it a diffusers pipeline checkpoint?")
    with open(index) as f:
        cfg = json.load(f)

    os.makedirs(out_dir, exist_ok=True)
    shutil.copy2(index, os.path.join(out_dir, "model_index.json"))

    copied = []
    for name, value in cfg.items():
        if name.startswith("_") or name in skip or not isinstance(value, list | tuple):
            continue
        src = os.path.join(src_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, os.path.join(out_dir, name), dirs_exist_ok=True)
            copied.append(name)
        elif value[0] is not None:
            print(f"[export] WARNING: model_index.json lists '{name}' but {src} does not exist", flush=True)
    return copied


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--svd", action="store_true", help="add the SVDQuant low-rank correction (+ its own file)")
    ap.add_argument("--n_calib", type=int, default=4, help="denoise-step inputs for static FP8 min/max")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--frames", type=int, default=17)
    ap.add_argument("--guidance", type=float, default=5.0)
    A = ap.parse_args()

    from diffusers import WanPipeline

    from quark.torch import export_safetensors
    from quark.torch.quantization.api import ModelQuantizer
    from quark.torch.utils.diffusers import get_calib_dataloader

    model_dir = resolve_model_dir(A.model)
    print(f"[export] loading bf16 WanPipeline ({model_dir}) ...", flush=True)
    pipe = WanPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    present = [e for e in EXPERTS if getattr(pipe, e, None) is not None]
    print(f"[export] experts present: {present}", flush=True)

    # Collect calibration for EVERY expert FIRST, while all experts are intact. Wan's
    # scheduler switches between transformer/transformer_2 across denoise steps (at
    # boundary_ratio), so running the pipe needs both live -- we cannot null one expert
    # and then calibrate the other.
    calib_by_expert = {}
    for expert in present:
        print(f"[export] calib capture for {expert} ({A.n_calib} steps) ...", flush=True)
        calib_by_expert[expert] = get_calib_dataloader(
            pipe,
            getattr(pipe, expert),
            prompts=["a red sports car on a coastal road at sunset"],
            n_steps=A.n_calib,
            seed=0,
            device="cuda",
            height=A.height,
            width=A.width,
            num_frames=A.frames,
            guidance_scale=A.guidance,
        )

    # Now quantize + export each expert from its captured calib (no more pipe runs).
    for expert in present:
        print(f"\n[export] === {expert} ===", flush=True)
        target = getattr(pipe, expert)
        q = ModelQuantizer(build_qconfig(A.svd))
        print(f"[export] quantize_model (mxfp4_fp8{', svd' if A.svd else ''}) + calib ...", flush=True)
        tf = q.quantize_model(target, calib_by_expert[expert])

        tdir = os.path.join(A.out, expert)
        n_ecm = 0
        if A.svd:
            n_ecm = save_svd_correction(tf, tdir)
            print(f"[export] saved SVD correction: {n_ecm} ErrorCorrectedModules", flush=True)
            # Unwrap the ECMs so the checkpoint has the SAME key layout as a plain
            # (non-SVD) export -- otherwise everything serializes under `<name>.layer.*`
            # and the reload (which maps QParamsLinear onto `<name>`) leaves the quantizer
            # buffers on the meta device. The low-rank branch rides in its own file and is
            # re-attached after load by attach_svd_correction().
            n_uw = unwrap_error_corrected(tf)
            print(f"[export] unwrapped {n_uw} ErrorCorrectedModules for a flat checkpoint", flush=True)

        tf = q.freeze(tf)
        os.makedirs(tdir, exist_ok=True)
        print(f"[export] export_safetensors(real_quantized) -> {tdir}", flush=True)
        export_safetensors(tf, tdir, custom_mode="quark", weight_format="real_quantized")
        if n_ecm:
            # Stamp config.json so a reload of this (residual-weight) checkpoint without
            # the correction file fails loudly instead of producing an uncorrected model.
            mark_correction_required(tdir)

        setattr(pipe, expert, None)
        del tf, q
        calib_by_expert[expert] = None
        gc.collect()
        torch.cuda.empty_cache()

    print("\n[export] copying non-transformer pipeline components ...", flush=True)
    copied = copy_pipeline_components(model_dir, A.out, skip=EXPERTS)
    print(f"[export] copied: {', '.join(copied)} (+ model_index.json)", flush=True)

    # Report packed size + quant_method for each expert.
    for expert in EXPERTS:
        cfgp = os.path.join(A.out, expert, "config.json")
        if os.path.exists(cfgp):
            with open(cfgp) as cfg_f:
                qc = json.load(cfg_f).get("quantization_config", {})
            exp = qc.get("export", {})
            print(
                f"[export] {expert}: quant_method={qc.get('quant_method')} "
                f"weight_format={exp.get('weight_format')} pack_method={exp.get('pack_method')}",
                flush=True,
            )
    print(f"\n[export] DONE -> {A.out}", flush=True)


if __name__ == "__main__":
    main()
