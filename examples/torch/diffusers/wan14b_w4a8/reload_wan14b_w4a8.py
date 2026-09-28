#
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
"""Reload a packed-MXFP4 w4a8 Wan2.2-A14B export and run native FlyDSL inference.

Loads the checkpoint written by ``export_wan14b_w4a8.py`` via
``WanPipeline.from_pretrained`` (the quark auto-quantizer rebuilds each expert's packed
``QParamsLinear``), optionally re-attaches the SVDQuant low-rank correction from its
companion file, converts each expert to native inference (FlyDSL w4a8 / aiter MXFP4) -- which
re-shuffles the export-`reorder` packed weight into the kernel's ASM layout at load --
and generates a short video.

Usage:
  OUT=wan14b_w4a8_export
  PYTHONPATH=<quark-repo> python3 reload_wan14b_w4a8.py --export $OUT \
      --native_linear_mode flydsl_a8w4   # or mxfp4 (aiter ASM), or none (QDQ)
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from svd_correction import attach_svd_correction, has_correction_file, require_correction_file

# video_io.py sits one level up: it is shared with quantize_diffusers.py, which uses the
# same uint8/PIL conversion for its Wan smoke-check artifacts.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from video_io import save_video, to_uint8  # noqa: E402

DEFAULT_OUT = os.environ.get("OUT", "wan14b_w4a8_export")
# Wan's standard negative prompt; generating with an empty one degrades quality badly.
DEFAULT_NEGATIVE_PROMPT = "normal quality, low quality, worst quality, low res, blurry, nsfw, nude."
EXPERTS = ("transformer", "transformer_2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default=DEFAULT_OUT)
    ap.add_argument(
        "--native_linear_mode",
        default="flydsl_a8w4",
        choices=["flydsl_a8w4", "flydsl_svdquant", "mxfp4", "none"],
        help="native kernel to convert reloaded QParamsLinear into; 'none' keeps QDQ",
    )
    ap.add_argument("--prompt", default="a red sports car on a coastal road at sunset")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    # Wan needs a full denoise schedule and a real negative prompt; under-sampling
    # (e.g. 12-20 steps, 17 frames, empty negative prompt) produces heavily artefacted
    # video even in *unquantized* bf16, which is easy to mistake for a quantization bug.
    ap.add_argument("--frames", type=int, default=81)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=4.0)
    ap.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_video", default="wan14b_w4a8_reload.mp4")
    ap.add_argument(
        "--reference_npy",
        default="",
        help="frames .npy from a bf16 (or known-good) run with the SAME prompt/seed/"
        "settings; enables the real cos/PSNR quality gate",
    )
    ap.add_argument(
        "--min_cos", type=float, default=0.95, help="minimum frame cosine vs --reference_npy (w4a8 typically ~0.977)"
    )
    A = ap.parse_args()

    from diffusers import WanPipeline

    import quark.integrations.diffusers  # noqa: F401  registers the quark auto-quantizer
    from quark.torch.export.nn.modules.qparamslinear import QParamsLinear

    print(f"[reload] WanPipeline.from_pretrained({A.export}) ...", flush=True)
    pipe = WanPipeline.from_pretrained(A.export, torch_dtype=torch.bfloat16).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    for expert in EXPERTS:
        target = getattr(pipe, expert, None)
        if target is None:
            continue
        n_qp = sum(1 for m in target.modules() if isinstance(m, QParamsLinear))
        print(f"[reload] {expert}: {n_qp} packed QParamsLinear reloaded", flush=True)
        assert n_qp > 0, f"{expert}: no packed QParamsLinear -- checkpoint is not real_quantized?"

        exp_dir = os.path.join(A.export, expert)
        # Raises when the export is marked SVDQuant but its correction file is gone: the
        # weights are the bare residual, so loading them uncorrected would be silently wrong.
        require_correction_file(exp_dir)
        if has_correction_file(exp_dir):
            n_ecm = attach_svd_correction(target, exp_dir)
            print(f"[reload] {expert}: re-attached {n_ecm} SVD corrections", flush=True)

        if A.native_linear_mode != "none":
            from quark.torch.quantization.nn.modules.native_inference_linear_common import NativeInferenceLinear
            from quark.torch.quantization.utils import RuntimeOptions, enable_native_inference

            enable_native_inference(target, runtime_options=RuntimeOptions(native_linear_mode=A.native_linear_mode))
            n_nat = sum(1 for m in target.modules() if isinstance(m, NativeInferenceLinear))
            print(f"[reload] {expert}: {n_nat} native ({A.native_linear_mode}) linears", flush=True)

    print("[reload] generating video ...", flush=True)
    result = pipe(
        prompt=A.prompt,
        negative_prompt=A.negative_prompt,
        height=A.height,
        width=A.width,
        num_frames=A.frames,
        num_inference_steps=A.steps,
        guidance_scale=A.guidance,
        generator=torch.Generator(device="cpu").manual_seed(A.seed),
    )
    frames = to_uint8(result.frames[0])
    mean = float(frames.mean())

    # NB: save_video(), not export_to_video() -- see video_io.py. Passing a uint8 array
    # straight to export_to_video silently colour-inverts every frame.
    try:
        saved = save_video(frames, A.out_video, fps=16)
    except Exception as exc:  # noqa: BLE001 -- no diffusers export / no video encoder
        print(f"[reload] could not write mp4 ({exc}); raw frames only", flush=True)
        np.save(A.out_video + ".npy", frames)
        saved = A.out_video + ".npy"
    print(f"[reload] saved {saved}  pixel mean={mean:.1f}", flush=True)

    # Cheap liveness gate only. NOTE: a pixel-mean range says almost nothing about
    # quality -- badly artefacted video passes it easily. Use --reference_npy for the
    # real check.
    assert 5.0 < mean < 250.0, f"video pixel mean {mean:.1f} is degenerate (all black/white)"
    np.save(A.out_video + ".npy", frames)

    if A.reference_npy:
        ref = np.load(A.reference_npy).astype(np.float32)
        cur = frames.astype(np.float32)
        n = min(len(ref), len(cur))
        a, b = ref[:n].ravel(), cur[:n].ravel()
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        mse = float(((ref[:n] - cur[:n]) ** 2).mean())
        psnr = 10 * np.log10(255.0**2 / mse) if mse > 0 else float("inf")
        print(f"[reload] vs reference: cos={cos:.4f} PSNR={psnr:.2f} dB ({n} frames)", flush=True)
        assert cos >= A.min_cos, (
            f"video cos {cos:.4f} < --min_cos {A.min_cos}: the reloaded model diverges from the "
            f"reference far more than quantization alone should explain."
        )
        print("[reload] PASS (matches reference within tolerance)", flush=True)
    else:
        print("[reload] PASS (liveness only -- pass --reference_npy for a real quality gate)", flush=True)


if __name__ == "__main__":
    main()
