# NVFP4 Quantization for DeepSeek-V4-Pro

End-to-end recipe that converts an original DeepSeek-V4-Pro checkpoint into an
NVFP4 checkpoint. Only the MoE expert weights (routed `layers.N.ffn.experts.*`
and shared `layers.N.ffn.shared_experts.*`) are quantized to NVFP4: FP4 (E2M1)
weights in groups of 16, with a per-group FP8 scale, a per-tensor FP32
scale-of-scale, and a per-tensor activation `input_scale`. Attention, the router
gate, norms, embeddings, the LM head, and the MTP block keep their original
format.

| Stage | Script | Produces |
|-------|--------|----------|
| 1. Weight quantization | `stage1_quantize_weight.py` | `weight` / `weight_scale` / `weight_scale_2` |
| 2. Activation calibration | `stage2_calibrate_input_scale.py` | `input_scale.safetensors` (one F32 scalar per expert projection) |
| 3. Merge | `stage3_merge.py` | final NVFP4 checkpoint (attaches `input_scale`) |
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
`N_CALIB_SAMPLES`, `CALIB_SEQLEN`, `PPL_BATCH`, `PPL_MAX_CHUNKS`.

To run a stage on its own, invoke the script directly and pass `--help` for its
flags and constraints. The only thing `--help` cannot express: Stage 4 needs the
source `inference/` linked into the quantized checkpoint dir first
(`ln -s $SRC/inference $OUT/inference`).

## Model evaluation

Both checkpoints are evaluated with the same path (`ppl_baseline.py` on the
original checkpoint, `stage4_ppl.py` on this pipeline's output) on
wikitext-2-raw-v1 (141 chunks × 2048 tokens), so the perplexities are directly
comparable. Recovery is `baseline PPL / NVFP4 PPL`.

Reproduce the baseline with:

```bash
CUDA_VISIBLE_DEVICES=0 python ppl_baseline.py --model-dir /path/to/DeepSeek-V4-Pro
```

| Baseline PPL | NVFP4 PPL | Recovery |
|--------------|-----------|----------|
| 2.1212 | 2.2042 | 96.2% |
