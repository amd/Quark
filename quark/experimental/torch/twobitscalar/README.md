# TwoBitScalar: 2-Bit Scalar PTQ

> **Requires Python 3.11–3.13** (matches Quark's `requires-python = ">=3.11,<3.14"`).

## Overview

TwoBitScalar is a **2-bit weight-only Post-Training Quantization (PTQ)** algorithm for
large language models. It quantizes each weight to one of **4 symmetric levels** per group,
combining three techniques:

- **AWQ-style activation-aware scaling** — protects high-activation (salient) channels.
- **SRHT incoherence rotation** — a Structured Random Hadamard Transform that makes the
  weight entries approximately i.i.d. Gaussian, so 4-level quantization is near-optimal.
- **MSE-optimal per-row scale search** with **Lloyd-Max** reconstruction levels.

The transforms are applied only for quantization and then **exactly undone**, so the exported
`nn.Linear.weight` is plain fake-quantized bf16 (no runtime rotation needed). The effective
rate is **~2.25 bits/weight** (2 bits per element + one fp16 scale per group).

It is an experimental algorithm (`quark.experimental.torch.twobitscalar`), driven through the
standard PTQ entry point with `--quant_algo twobitscalar` plus a config file — the same
config-driven pattern as every other algorithm, with no per-algorithm CLI flags. See
`examples/torch/experimental/two_bits/` for the runnable example.

## Results (WikiText-2)

**PPL configuration:** dataset `wikitext-2-raw-v1`, **test** split, text joined with `\n\n`,
**seq_len = 2048** non-overlapping windows, mean token cross-entropy → `exp(mean)`. Config:
`bits=2, group_size=64, alpha=0.5, Lloyd-Max levels, incoherence(SRHT) on` (~2.25 effective
bits/weight). These are weight-only 2-bit PTQ numbers, *before* any QAD / LoRA+KD recovery.

| Model | BF16 | TwoBitScalar 2-bit | Δ |
|-------|-----:|-------------------:|---:|
| Phi-4 (14B) | 6.46 | 10.72 | +4.26 |
| QwQ-32B     | 6.34 | 10.29 | +3.95 |
| Qwen3-8B    | 9.73 | 18.80 | +9.07 |

**vs Quark GPTQ** at the same 2-bit / group_size 64 (GPTQ = `int2` per-group, symmetric),
same harness:

| Model | BF16 | TwoBitScalar 2-bit | Quark GPTQ 2-bit |
|-------|-----:|-------------------:|-----------------:|
| Phi-4 (14B) | 6.46 | **10.72** | 28.22 |
| QwQ-32B     | 6.34 | **10.29** | 73.63 |
| Qwen3-8B    | 9.73 | **18.80** | 67.34 |

AWQ scaling + SRHT incoherence rotation + Lloyd-Max levels give substantially lower PPL than
plain 2-bit GPTQ (which has none of these).

## Algorithm Pipeline

For each `nn.Linear` layer in the model (excluding embeddings and LM head), the algorithm
proceeds through the following stages:

### Stage 1: Calibration Data Collection

```text
For each linear layer:
    Register forward hook → collect input activations X ∈ R^{N × d_in}
    Record per-channel activation maximums: in_amax[j] = max_i |X[i,j]|
```

The model is run on calibration data (typically 1024 batches of 1024-token sequences from WikiText-2 or C4). Each layer stores up to `max_samples_per_layer` (default 2048) input rows and the running per-channel maximum activation magnitude.

### Stage 2: AWQ-style Activation-Aware Scaling

**Purpose:** Protect high-activation channels from quantization error.

```text
s[j] = clip( (in_amax[j] / mean(in_amax))^α, s_min, s_max )

W_scaled = W * diag(s)        # scale columns of weight
X_scaled = X / diag(s)        # compensate in activations (so W·x is unchanged)
```

- `α` (default 0.5): Controls how aggressively to protect salient channels
- `s_min=0.25, s_max=4.0`: Prevents extreme scales from destabilizing quantization

**Key insight:** Channels with large activations (high `in_amax`) get their weights divided by a larger scale, making those weights smaller and easier to quantize accurately. The inverse scale is applied to the input activations, preserving the `W·x` product exactly.

### Stage 3: Global Incoherence Rotation (SRHT)

**Purpose:** Transform weights so that large-magnitude entries are spread across many positions, making the distribution more uniform and Gaussian-like within each group.

The algorithm applies a **Structured Random Hadamard Transform (SRHT)** to the column dimension of each weight matrix:

```text
W_rotated = W_scaled @ R^T     where R = (D · H · P) / √n
X_rotated = X_scaled @ R^T

Components:
  P = random permutation matrix (breaks structured correlations)
  H = Walsh-Hadamard matrix (orthogonal transform, O(n log n))
  D = random diagonal sign matrix (entries ±1)
  1/√n = normalization to preserve norms
```

**Why this helps:** Chee et al. (2023) prove that multiplying by a random orthogonal matrix makes the weight entries approximately i.i.d. sub-Gaussian. Since the SRHT is a near-isometry (preserves distances), quantization error in the rotated space maps back to similar error in the original space, but the error is now spread evenly rather than concentrated on a few large weights.

The rotation is applied per-group (columns `gs:ge`) during the per-group quantization loop, and **exactly undone** after quantization.

### Stage 4: Per-Group Quantization

The input dimension `d_in` is partitioned into groups of size `g` (default 64). Each group is quantized independently.

**Quantization levels** — each weight value is mapped to one of **4 symmetric levels**:

| Mode | Levels | Description |
|---|---|---|
| Default (uniform) | {-1, -1/3, +1/3, +1} | Equally spaced 2-bit levels |
| Lloyd-Max (Gaussian) | {-1, -0.2998, +0.2998, +1} | Optimal for Gaussian distributions |

**Per-row scale search** — for each row the algorithm searches a grid of scale fractions for the scale that minimizes the quantization MSE, using nearest-level assignment.

**Sub-group quantization** — when `sub_group_size < group_size`, SRHT decorrelation is applied at the group level but quantization uses finer per-row scales, trading bits for accuracy:

```text
eff_bits = 2 + (16 / sub_group_size)      # 16 bits = one fp16 scale per sub-group per row

sub_g = 64 (no sub-group):  2.25 bits
sub_g = 32:                 2.50 bits
```

### Stage 5: Undo Transforms

All transforms are applied and undone in strict reverse order:

```text
1. Undo SRHT rotation:   W_q_unrotated = W_q_rotated @ R    (R orthogonal, R^T = R^{-1})
2. Undo AWQ scaling:     W_final = W_q_unrotated / diag(s)
```

The final quantized-dequantized weights are stored back in the original `nn.Linear.weight` tensor in the original dtype. This is "fake quantization" — the weights carry the 2-bit error but are stored at full precision for inference compatibility.

### Stage 6: Disable Framework Quantizers

After TwoBitScalar processing, the Quark framework's own fake-quantizers are disabled on processed layers to prevent double quantization noise.

## Configuration Reference

All options are in `TwoBitScalarConfig` (`quark/experimental/torch/twobitscalar/config.py`):

| Parameter | Default | Description |
|---|---|---|
| `bits` | 2 | Fixed 2-bit (4 levels); only `bits=2` is supported |
| `group_size` | 64 | Column group size for SRHT and quantization |
| `sub_group_size` | 0 | Sub-group size (0 = same as group_size) |
| `act_scale_alpha` | 0.5 | AWQ scaling exponent (0 disables) |
| `enable_incoherence` | False | Enable SRHT incoherence rotation |
| `use_lloyd_max_levels` | False | Use Gaussian-optimal levels {-1, ±0.2998, 1} |
| `exclude_layers` | None | Layer-name wildcards to skip (e.g. embeddings, LM head) |
| `dump_sidecar_dir` | None | If set, write the per-layer sidecar (see below) |

The shipped example enables `enable_incoherence` + `use_lloyd_max_levels`; all other tuning
knobs default to `False`.

## Usage

TwoBitScalar is config-driven — supply the hyperparameters through a JSON config file via the
standard `--quant_algo_config_file` mechanism:

```bash
python3 examples/torch/language_modeling/llm_ptq/quantize_quark.py \
  --model_dir <path> \
  --quant_scheme bfp16 --quant_algo twobitscalar \
  --quant_algo_config_file twobitscalar examples/torch/experimental/two_bits/twobitscalar_config.json \
  --exclude_layers "*embed_tokens*" "*lm_head*" \
  --evaluation_dataset wikitext ...
```

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

## The sidecar (for factored / shared-rotation exports)

By default the exported checkpoint is fake-quantized in the **original** weight basis:
TwoBitScalar applies the AWQ scaling and SRHT rotation, quantizes, then *undoes* both, so
`nn.Linear.weight` is plain bf16 carrying the 2-bit error — no runtime rotation needed, but the
rotation/scaling information is discarded.

The **sidecar** optionally records the per-layer transforms so downstream tooling can
reconstruct them:

- `awq_scale.safetensors` — per-channel AWQ scale vector `s_vec` per layer
- `srht_perm.safetensors` — SRHT random permutation (column reordering)
- `srht_signs.safetensors` — SRHT random sign flips (±1 per column)

Together these reconstruct the input-side transform `x_rot = SRHT(x / awq_scale)`. That lets
the **shared-rotation / factored exports** keep the weights in the rotated/quantized domain
(genuinely 2-bit-compressible) and rotate the *activations* at runtime, instead of undoing the
rotation into the weights. The sidecar is **not** needed for plain fake-quantized PTQ or for
QAD.

Enable it by adding `"dump_sidecar_dir": "/path/to/sidecar"` to the config JSON (requires
`enable_incoherence=true` and `num_hadamard_passes=1`).

## Mathematical Foundation

**Why incoherence processing works.** The core difficulty of low-bit quantization is that
weight matrices have non-uniform entry magnitudes — a few large outliers dominate the error
when forced into coarse levels. Multiplying by a random orthogonal matrix (Chee et al., 2023)
makes the entries approximately i.i.d. sub-Gaussian, so after rotation all entries have roughly
the same magnitude (uniform quantization is near-optimal), the distribution is approximately
Gaussian (Lloyd-Max levels are optimal), and the error is spread evenly instead of concentrated
on outliers.

**Why AWQ scaling helps.** For `y = Wx`, the MSE contribution of column `j` is proportional to
`||ΔW[:,j]||² · E[x_j²]`. Scaling salient (high-activation) columns up before quantization —
and compensating in the activations — gives those columns effectively more quantization
precision while leaving `W·x` unchanged.

## File Structure

```text
quark/experimental/torch/twobitscalar/
├── __init__.py         # Package init
├── config.py           # TwoBitScalarConfig dataclass (all parameters)
├── twobitscalar.py     # TwoBitScalarProcessor: calibration, quantization, transforms
└── README.md           # This file

examples/torch/experimental/two_bits/
├── twobitscalar_config.json   # Example config
├── run_twobitscalar_ptq.sh    # Env-driven runscript
└── README.md                  # Example usage

examples/torch/language_modeling/llm_ptq/
└── quantize_quark.py          # PTQ entry point (--quant_algo twobitscalar)
```

## References

- **QuIP:** Chee, Tsai, Ren, Calandra, Basu. "QuIP: 2-Bit Quantization of Large Language Models With Guarantees." NeurIPS 2023.
- **QuIP#:** Tseng, Chee, Sun, Ré, Basu. "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks." ICML 2024.
- **GPTQ:** Frantar, Ashkboos, Hoefler, Alistarh. "GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers." ICLR 2023.
- **AWQ:** Lin, Tang, Tang, Yao, Han. "AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration." MLSys 2024.
