# NVFP4 Quantization for DeepSeek-V4-Pro

End-to-end recipe that converts an original DeepSeek-V4-Pro checkpoint into an
NVFP4 checkpoint. By default only the routed MoE expert weights
(`layers.N.ffn.experts.*`) are quantized to NVFP4: FP4 (E2M1) weights in groups
of 16, with a per-group FP8 scale, a per-tensor FP32 scale-of-scale, and a
per-tensor activation `input_scale`. The shared experts
(`layers.N.ffn.shared_experts.*`), attention, the router gate, norms, embeddings,
the LM head, and the MTP block keep their original format.

What is NOT re-quantized to NVFP4 is controlled by two flags on Stages 1 and 2 (both
stages must receive the same lists); see their `--help`:

- `--exclude_layers` — modules that are **truly not quantized**: they are kept as
  **16-bit (BF16)** in the output. These are already 16-bit in the source checkpoint
  (router gate, `*ffn_norm`, embeddings, LM head, final norm, MTP block).
- `--keep_original_format_layers` — modules kept in the model's **original quantized
  format** (passthrough, still a quantized representation): FP8 attention and shared
  experts. They are neither re-quantized to NVFP4 nor dequantized to BF16.

With the defaults only the routed MoE experts become NVFP4 and the shared experts stay
FP8. To also quantize the shared experts (full NVFP4 W4A4), drop `*shared_experts*` from
`--keep_original_format_layers`.

| Stage | Script | Produces |
|-------|--------|----------|
| 1. Weight quantization | `stage1_quantize_weight.py` | `weight` / `weight_scale` / `weight_scale_2` |
| 2. Activation calibration | `stage2_calibrate_input_scale.py` | `input_scale.safetensors` (one F32 scalar per expert projection) |
| 3. Merge | `stage3_merge.py` | final NVFP4 checkpoint |
| 4. PPL eval (optional) | `stage4_ppl.py` | wikitext-2 perplexity of the NVFP4 model |

One GPU is enough: every stage streams a single decoder block to the GPU at a
time. Stage 2 fills any routed expert that never fired during calibration with
the max `input_scale` over the calibrated experts in the same layer and
projection, so a larger per-tensor scale never clips a rare expert.

## Setup

```bash
pip install -r requirements.txt
```

The source checkpoint must keep its `inference/` directory (`model.py`,
`config.json`, `kernel.py`); Stages 2 and 4 load the model definition from there.
Stage 1 does not copy `inference/` into its output, so Stage 4 needs it linked in
(the one-shot script does this automatically). When running from a Quark checkout
rather than a pip install, put the checkout on `PYTHONPATH`.

## Quick start

`run_pipeline.sh` runs all four stages, links `inference/` before Stage 4, and
aborts on the first failure. It auto-detects the Quark checkout from its own
location, so usually only `SRC` and `OUT` are needed:

```bash
SRC=/path/to/DeepSeek-V4-Pro \
OUT=/path/to/output-nvfp4 \
    bash run_pipeline.sh
```

Overridable env vars: `SRC`, `OUT`, `QUARK`, `GPU` (default 0), `CALIB_BATCH`,
`N_CALIB_SAMPLES`, `CALIB_SEQLEN`, `PPL_BATCH`, `PPL_MAX_CHUNKS`, `EXCLUDE_LAYERS`
(kept-16-bit patterns), and `KEEP_ORIGINAL_FORMAT_LAYERS` (kept-original-format
patterns). Both are fnmatch pattern lists threaded into both stages; the defaults keep
attention and shared experts FP8 and quantize only the routed experts. To quantize the
shared experts too, set `KEEP_ORIGINAL_FORMAT_LAYERS="*attn*"` (drop `*shared_experts*`).

To run a stage on its own, invoke the script directly and pass `--help` for its
flags and constraints. The only thing `--help` cannot express: Stage 4 needs the
source `inference/` linked into the quantized checkpoint dir first
(`ln -s $SRC/inference $OUT/inference`).

## Model evaluation

Both checkpoints are evaluated with the same path (`ppl_baseline.py` on the
original checkpoint, `stage4_ppl.py` on this pipeline's output) on
wikitext-2-raw-v1 (141 chunks × 2048 tokens), so the perplexities are directly
comparable. Recovery is `baseline PPL / NVFP4 PPL`. All numbers below are measured
on the same **AMD MI300X (gfx942, ROCm 7.0.0)**; PPL varies between GPUs, so do not
compare these against numbers measured on other hardware.

Reproduce the baseline with:

```bash
CUDA_VISIBLE_DEVICES=0 python ppl_baseline.py --model-dir /path/to/DeepSeek-V4-Pro
```

| Calibration data | Baseline PPL | NVFP4 PPL | Recovery |
|------------------|--------------|-----------|----------|
| cnn_dailymail (64 samples) | 2.3250 | 2.3759 | 97.9% |
| cnn_dailymail + nemotron-v2 (128 samples) | 2.3250 | 2.3505 | 98.9% |

The second row aligns calibration with NVIDIA's DeepSeek-V4 recipe (cnn_dailymail +
Nemotron-Post-Training-Dataset-v2, 64 samples each) and gives the better perplexity
(2.3505 vs 2.3759), so it is the recommended calibration for this benchmark. For
reference, quantizing the shared experts as well (full NVFP4 W4A4) on the same
MI300X gives 2.4030, so keeping the shared experts in FP8 (this pipeline) improves
PPL by 0.053 over full quantization.
