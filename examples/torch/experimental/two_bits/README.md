# TwoBitScalar: 2-bit weight-only PTQ

TwoBitScalar quantizes weights to 4 symmetric levels per group (AWQ scaling +
SRHT incoherence rotation + MSE-optimal scale search), at an effective ~2.25
bits/weight. It is the PTQ used to initialize the 2-bit QAD and LoRA+KD
pipelines.

It is an **experimental** algorithm
(`quark.experimental.torch.twobitscalar`) and is driven through the
standard PTQ entry point (`examples/torch/language_modeling/llm_ptq/quantize_quark.py`)
using `--quant_algo twobitscalar` plus a config file — the same config-driven
pattern used by every other algorithm, with no per-algorithm CLI flags. See
`quark/experimental/torch/twobitscalar/README.md` for the algorithm.

> **Requires Python 3.11–3.13** (matches Quark's `requires-python = ">=3.11,<3.14"`).

## Config

Hyperparameters live in [`twobitscalar_config.json`](./twobitscalar_config.json):

```json
{
    "name": "twobitscalar",
    "bits": 2,
    "group_size": 64,
    "act_scale_alpha": 0.5,
    "enable_incoherence": true,
    "use_lloyd_max_levels": true,
    "exclude_layers": ["*embed_tokens*", "*lm_head*"]
}
```

## Run

Helper script (env-driven):

```bash
MODEL_DIR=/path/to/phi-4 OUTPUT_DIR=/path/to/phi4_2bit bash run_twobitscalar_ptq.sh
```

Or invoke the PTQ entry point directly with the config file:

```bash
python ../../language_modeling/llm_ptq/quantize_quark.py \
  --model_dir /path/to/phi-4 \
  --quant_scheme bfp16 --quant_algo twobitscalar \
  --quant_algo_config_file twobitscalar ./twobitscalar_config.json \
  --exclude_layers "*embed_tokens*" "*lm_head*" \
  --num_calib_data 4096 --seq_len 1024 --batch_size 1 \
  --evaluation_dataset wikitext --export_weight_format fake_quantized --model_export hf_format \
  --output_dir /path/to/phi4_2bit
```

For **QwQ-32B**, point `--model_dir` at QwQ-32B and change `--output_dir`.

## The sidecar (for factored / shared-rotation exports)

By default the exported checkpoint is **fake-quantized in the original weight
basis**: TwoBitScalar applies the AWQ scaling and SRHT rotation, quantizes, then
*undoes* both, so `nn.Linear.weight` is plain bf16 carrying the 2-bit error — no
runtime rotation needed, but the rotation/scaling information is gone.

The **sidecar** records the per-layer transforms so downstream tooling can
reconstruct them:

- `awq_scale.safetensors` — per-channel AWQ scale vector `s_vec` per layer
- `srht_perm.safetensors` — SRHT random permutation (column reordering)
- `srht_signs.safetensors` — SRHT random sign flips (`±1` per column)

These are needed by the **shared-rotation / factored exports** (keep weights in
the rotated/quantized domain and rotate activations at runtime). The sidecar is
**not** needed for plain fake-quantized PTQ or for QAD.

To emit it, add `"dump_sidecar_dir"` to the config JSON:

```json
{
    "name": "twobitscalar",
    "bits": 2,
    "group_size": 64,
    "act_scale_alpha": 0.5,
    "enable_incoherence": true,
    "use_lloyd_max_levels": true,
    "exclude_layers": ["*embed_tokens*", "*lm_head*"],
    "dump_sidecar_dir": "/path/to/phi4_2bit_sidecar"
}
```

## Results

WikiText-2 perplexity (Phi-4, QwQ-32B, Qwen3-8B) and the comparison against Quark GPTQ at the
same 2-bit / group_size live in the algorithm README:
[`quark/experimental/torch/twobitscalar/README.md`](../../../../quark/experimental/torch/twobitscalar/README.md).
