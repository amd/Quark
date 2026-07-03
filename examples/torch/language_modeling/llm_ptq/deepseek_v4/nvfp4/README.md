# NVFP4 Quantization for DeepSeek-V4-Pro

End-to-end recipe that turns an original DeepSeek-V4-Pro checkpoint into a final
NVFP4 checkpoint. Only the MoE expert weights (routed
`layers.N.ffn.experts.*` and shared `layers.N.ffn.shared_experts.*`) are
quantized to NVFP4 — FP4 (E2M1) weights in groups of 16, with a per-group FP8
scale and a per-tensor FP32 scale-of-scale, plus a per-tensor activation
`input_scale`. Attention, the router gate, norms, embeddings, the LM head and the
MTP block are kept in their original format.

| Stage | Script | Produces |
|-------|--------|----------|
| 1. Weight quantization | `stage1_quantize_weight.py` | `weight` / `weight_scale` / `weight_scale_2` |
| 2. Activation calibration | `stage2_calibrate_input_scale.py` | `input_scale.safetensors` (one F32 scalar per expert projection) |
| 3. Merge | `stage3_merge.py` | **final** NVFP4 checkpoint (attaches `input_scale`) |
| 4. PPL eval (optional) | `stage4_ppl.py` | wikitext-2 perplexity of the NVFP4 weights |

A single GPU is enough: every stage streams one decoder block to the GPU at a
time.

## Setup

```bash
pip install -r requirements.txt
```

One thing to get right, or Stages 2/4 fail with a file-not-found error: **the
source checkpoint must keep its `inference/` directory** (`model.py` +
`config.json` + `kernel.py`), which Stages 2 and 4 load the model definition
from. Stage 1 does **not** copy `inference/` into its output, so Stage 4 needs it
linked in (the one-shot script does this automatically).

When running from a Quark checkout (not a pip install), put it on `PYTHONPATH`.

## Quick start (one-shot)

`run_pipeline.sh` runs all four stages, handles the `inference/` link before
Stage 4, and aborts on the first failure. It auto-detects the Quark checkout from
its own location, so usually only `SRC` and `OUT` are needed:

```bash
SRC=/path/to/DeepSeek-V4-Pro \
OUT=/path/to/output-nvfp4 \
    bash run_pipeline.sh
```

Overridable env vars: `SRC`, `OUT`, `QUARK`, `GPU` (default 0), `CALIB_BATCH`,
`N_CALIB_SAMPLES`, `CALIB_SEQLEN`, `PPL_BATCH`, `PPL_MAX_CHUNKS`.

To run a stage on its own, invoke the script directly; run any of them with
`--help` for the full flag list and per-flag constraints. The one thing `--help`
cannot express: Stage 4 needs the source `inference/` linked into the quantized
checkpoint dir first (`ln -s $SRC/inference $OUT/inference`; the one-shot script
does this automatically).

Stage 2 fills any routed expert that never fired during calibration with the max
`input_scale` over the calibrated experts in the same layer+projection (a larger
per-tensor scale never clips a rare expert).

## Model Evaluation

Evaluated on wikitext-2-raw-v1 (141 chunks × 2048 tokens). Baseline and NVFP4 use
the **identical eval path** (`ppl_baseline.py` on the original checkpoint vs
`stage4_ppl.py` on this pipeline's output), so compare **within a row**:

| Backend | Baseline PPL | NVFP4 PPL | Δ |
|---------|--------------|-----------|---|
| AMD MI300X (ROCm 7.0.0, torch 2.11.0.dev, transformers 5.8.1) | 2.3250 | **2.2884** | −0.037 |
| AMD MI355 / gfx950 (ROCm 7.2.2, torch 2.10.0, transformers 5.9.0) | 2.1212 | **2.1303** | +0.009 |

The NVFP4 weight-only PPL is on par with the BF16 baseline on every backend —
the MoE-expert NVFP4 quantization is effectively lossless on this metric. Peak GPU
memory is ~33–34 GB. (MI355's lower absolute PPL is only its newer `transformers`
modeling DeepSeek-V4 slightly differently; compare within a row, not across.)

Reproduce the BF16 baseline by running `ppl_baseline.py` on the **original**
checkpoint (same eval path as Stage 4, so the two numbers are comparable):

```bash
CUDA_VISIBLE_DEVICES=0 python ppl_baseline.py --model-dir /path/to/DeepSeek-V4-Pro
```
