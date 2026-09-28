<!--
Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT
-->

# Wan2.2-T2V-A14B — packed MXFP4 w4a8 export + native-FlyDSL reload

End-to-end workflow for quantizing the dual-expert **Wan2.2-T2V-A14B** text-to-video
transformer to **w4a8** (MXFP4 weight × FP8-e4m3 activation), exporting **packed
FP4** weights that `diffusers.from_pretrained` can reload, and running **native FlyDSL /
aiter** inference on the reloaded model.

## What this demonstrates

| Stage | What happens |
|-------|--------------|
| **Quantize** | each expert (`transformer`, `transformer_2`) → `mxfp4_fp8` (per-1×32 MXFP4 weight + static per-tensor FP8 activation), with a short calibration forward for the activation scales |
| **Export** | `export_safetensors(weight_format="real_quantized")` writes **packed** `float4_e2m1fn_x2` weights (~7.5 GB/expert, ≈¼ the bf16 size) + a `config.json` marked `quant_method=quark` so diffusers auto-dispatches to the Quark quantizer on load |
| **Reload** | `WanPipeline.from_pretrained` rebuilds each expert's packed `QParamsLinear` directly from the fp4 bytes (no bf16 masters) |
| **Native** | `enable_native_inference(...)` re-shuffles the export layout into the aiter/FlyDSL ASM layout **at load** and swaps in the native w4a8 / MXFP4 kernels |
| **SVD (optional)** | with `--svd`, the low-rank correction (`l1`/`l2`/`smooth`) is saved to a side `svd_correction.safetensors` and re-attached after load (diffusers only reloads the residual) |

## Requirements

- gfx950 (MI350) for the native FlyDSL/aiter kernels: the `flydsl` wheel and a matching
  `aiter` (the a8w4 GEMM is vendored in Quark — a `$FLYDSL_REPO` checkout is not used; see
  `docs/source/pytorch/pytorch_troubleshooting.rst`).
- A diffusers build with the Quark auto-quantizer (the bundled `quark.integrations.diffusers`
  registration / diffusers PR #14077).
- The A14B checkpoint (`WanPipeline`, two `WanTransformer3DModel` experts).

## Usage

```bash
export PYTHONPATH="<quark-repo>" HIP_VISIBLE_DEVICES=0

# 1. Quantize both experts + export packed fp4 (plain w4a8)
python3 export_wan14b_w4a8.py \
    --model Wan-AI/Wan2.2-T2V-A14B-Diffusers \
    --out   wan14b_w4a8_export --n_calib 4
#   add --svd to also emit the SVDQuant low-rank correction file

# 2. Reload via from_pretrained, convert to native, generate a video
python3 reload_wan14b_w4a8.py \
    --export wan14b_w4a8_export \
    --native_linear_mode mxfp4        # or flydsl_a8w4 / flydsl_svdquant / none (QDQ)
```

Pass `--reference_npy <frames.npy>` to enable the cos/PSNR quality gate, where the reference
is a bf16 run at the **same** prompt/seed/resolution/frames/steps/guidance/negative prompt,
written through the same frame path (`video_io.save_video`). Without it the script only
performs a liveness (non-degenerate pixel-mean) check.

## Three configurations

Two of these share **one checkpoint**. The weight on disk is MXFP4 either way — w4a8 and
w4a4 differ only in how the *activation* is quantized, which is chosen at load. Only
SVDQuant needs its own export (it also writes the low-rank correction file).

### 1. Plain w4a8 — no SVD

```bash
python3 export_wan14b_w4a8.py --out wan14b_w4a8_export --n_calib 4
python3 reload_wan14b_w4a8.py --export wan14b_w4a8_export \
    --native_linear_mode flydsl_a8w4
```

MXFP4 weight × MXFP8 dynamic per-1×32 activation, FlyDSL GEMM. The straightforward
baseline; needs the `flydsl` wheel. Use `--native_linear_mode none` for the
kernel-independent QDQ reference.

### 2. Plain w4a4 — no SVD, no FlyDSL

```bash
# same checkpoint as (1) -- no re-export needed
python3 reload_wan14b_w4a8.py --export wan14b_w4a8_export \
    --native_linear_mode mxfp4
```

MXFP4 weight × MXFP4 dynamic per-1×32 activation via aiter's ASM `gemm_a4w4`. Despite the
name, `mxfp4` is the **w4a4** path. Lowest bandwidth (4-bit activations halve the A traffic)
and no FlyDSL dependency. The checkpoint's per-tensor FP8 input scales are ignored here.

### 3. w4a8 + SVDQuant

```bash
python3 export_wan14b_w4a8.py --out wan14b_w4a8_svd_export --n_calib 4 --svd
python3 reload_wan14b_w4a8.py --export wan14b_w4a8_svd_export \
    --native_linear_mode flydsl_svdquant
```

Quantizes the SVD *residual* `R = W - L2 L1` and keeps the low-rank branch in high
precision. Requires `--svd` at export (writes `svd_correction.safetensors` per expert,
~0.3 GB); the reload re-attaches it automatically. Most accurate of the three, and the gap
widens as activation precision drops — which is why `flydsl_svdquant_w4a4` exists.

## Files

- `export_wan14b_w4a8.py` — dual-expert quantize + packed export (captures calibration for
  both experts *before* quantizing either, since Wan's scheduler needs both live).
- `reload_wan14b_w4a8.py` — `from_pretrained` → assert packed `QParamsLinear` → (optional)
  attach SVD → `enable_native_inference` → generate + quality/liveness check.
- `../video_io.py` — `save_video()` / `to_uint8()`, shared with `quantize_diffusers.py`;
  never pass uint8 frames to `export_to_video` directly.
- `svd_correction.py` — `save_svd_correction` / `attach_svd_correction` (the SVDQuant low-rank
  reload workaround), plus the `svdquant_correction_file` config.json marker that makes a
  correction-less reload of an SVD export fail loudly instead of silently dropping it.

## Notes

- **Packed size:** each expert exports at ~7.5 GB (400 packed fp4 linears), or ~7.8 GB with
  `--svd` (including a ~293 MB `svd_correction.safetensors`). Note the stock A14B
  checkpoint is stored **F32** (54 GB/expert), so a raw 54 -> 7.5 GB comparison conflates two
  wins: fp32 -> bf16 (54 -> 27 GB) is free via `torch_dtype=bfloat16` and needs no
  quantization; the **quantization** win is bf16-equivalent 27 GB -> 7.5 GB, i.e. **3.6x**.
- **What is *not* quantized:** the UMT5 text encoder stays bf16 and becomes the largest single
  component of the export (11 GB of 26 GB, 42%). The VAE is only 485 MB.
- **Speed:** w4a8 is a size optimization, not a throughput one. End-to-end generation measured
  +1.9% (RTN) and +0.5% (SVDQuant) vs bf16 on Wan2.2-A14B at default settings, consistent with
  the a8w4 GEMM being only ~6% faster than bf16 (1.355 ms vs 1.439 ms) and linears being one
  part of the step cost.
- **Reload → native fidelity:** the `reorder` export layout and the aiter ASM kernel layout
  differ; `enable_native_inference` repacks (`_pack_weight_asm`) at load. Validated: reloaded
  native output tracks the QDQ path (tiny-model cos ≈ 0.9998) and the full A14B produces a
  coherent video (pixel mean ~100).
- **Why the correction file needs the unwrap:** SVDQuant leaves each layer as
  `ErrorCorrectedModule(correction, layer)`, which would serialize as `<name>.layer.*` /
  `<name>.correction.*`. The diffusers reload maps `QParamsLinear` onto `<name>` directly, so
  those nested keys match nothing and the quantizer buffers stay on the meta device. The
  export therefore saves the low-rank branch separately and **unwraps** those modules, giving
  a checkpoint whose key layout is identical to a plain (non-SVD) export;
  `attach_svd_correction()` re-wraps it after load.
- **`exclude_patterns` must be globs:** `SVDQuantConfig.exclude_patterns` is matched with
  `fnmatch`, so bare substrings match nothing. Getting this wrong lets SVD wrap Wan's
  float32 `time_embedder`, which then fails the bf16 calibration forward with
  "self and mat2 must have the same dtype".
- **Frames:** WanPipeline returns `[0,1]` float frames — scale ×255 before uint8 (handled).
- **Never hand uint8 frames to `diffusers.export_to_video`.** It assumes `list[np.ndarray]`
  is float `[0,1]` and does `(frame * 255).astype(np.uint8)`; on uint8 input that wraps mod
  256, i.e. `x -> 256 - x`, **colour-inverting every frame**. Motion and structure survive,
  so it does not look like a bug — it looks like catastrophic quantization damage (orange
  skies, magenta foliage, flat neon patches). Use `video_io.save_video()`, which routes
  through PIL and is version-stable. This single bug produced two successive wrong root
  causes (blamed on quantization, then on sampling settings) before being found.
- **Always generate a bf16 reference through the *identical* code path** before concluding
  anything about quantization quality — it is the cheapest possible control and it would
  have caught the above immediately. Under-sampling (few steps/frames, empty
  `negative_prompt`) does *not* cause colour artifacts; measured correctly it only raises
  diffuse chroma noise (`chroma_edge_ratio` 0.97–1.24 vs 0.47–0.85). The defaults here
  (81 frames, 30 steps, guidance 4.0, Wan's standard negative prompt) still give the best
  output, just not for the reason originally recorded here.
- **Quality gate:** the pixel-mean check is only a liveness test (degenerate all-black /
  all-white); video full of artifacts passes it trivially. Pass `--reference_npy` (frames from a
  bf16 run with identical settings) for the real cos/PSNR gate.
- **Every native number above came from Quark's *vendored* FlyDSL GEMM**
  (`quark.torch.kernel.flydsl.kernels.preshuffle_gemm`), which carries the fused SVD
  epilogue that has not landed upstream. That snapshot is the implementation that runs; it
  needs the `flydsl` wheel plus `aiter` (its MFMA epilogue / pipeline helpers come from
  `aiter.ops.flydsl`), and Quark never consults a `$FLYDSL_REPO` checkout. See
  `docs/source/pytorch/pytorch_troubleshooting.rst` for the version matrix.
- **QDQ and native do not use the same activation scheme.** The `mxfp4_fp8` QDQ path applies a
  *static per-tensor* FP8-e4m3 activation scale; the native a8w4 kernel applies *dynamic
  per-1×32 MXFP8*. They are therefore not numerically equivalent by construction, before any
  kernel bug. The activation-quant backend is chosen by `QUARK_MXFP8_QUANT`
  (`triton` default, or `flydsl` / `aiter` / `eager`); `eager`, `triton` and `flydsl` are
  verified **bit-exact** here (identical fp8 values *and* e8m0 scales), and `aiter` is absent
  in this build and silently falls back to Triton — so the backend is not the bias source.
- **`--native_linear_mode none`** keeps the QDQ (`QParamsLinear`) path for a
  kernel-independent reference.
