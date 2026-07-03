# Mix Precision Auto-Search

Automatically finds the optimal mixed-precision quantization configuration for an LLM.
It searches from least to most aggressive quantization and selects the most aggressively
quantized model whose accuracy loss stays within the configured threshold.

## Supported Quantization Modes

| Hardware | Supported Modes                                         |
|----------|---------------------------------------------------------|
| MI300    | native, fp8, ptpc_fp8                                   |
| MI325    | native, fp8, ptpc_fp8                                   |
| MI355    | native, fp8, ptpc_fp8, mxfp4, mxfp4_fp8, mxfp6_e2m3   |

## Environment Setup

This workflow MUST run inside the vLLM ROCm container. The container provides the required vLLM + ROCm + PyTorch stack, which is much easier than setting the environment manually.

### Step 1: Pull and start the vLLM ROCm container

**REQUIRED:** use this exact container environment:

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  -v /path/to/models:/models \
  -v /path/to/Quark:/workspace/quark \
  vllm/vllm-openai-rocm:v0.19.0 \
  bash
```

**Path mapping guide:**

- Replace `/path/to/models` with your host model directory (e.g., `/data/Qwen`)
- Replace `/path/to/Quark` with your host Quark repo path (e.g., `/home/user/Quark_0521`)
- Inside container: models will be at `/models/`, Quark at `/workspace/quark/`

**Example:**

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  -v /data/Qwen:/models \
  -v /home/yuzho/0415_mp/Quark_0521:/workspace/quark \
  vllm/vllm-openai-rocm:v0.19.0 \
  bash
```

### Step 2: Inside the container, install Quark

**After you are inside the container**, install Quark:

```bash
# Install from source (recommended for development)
cd /workspace/quark && pip install -e .

# OR install from PyPI
pip install amd-quark
```

## Quick Start

**Prerequisites:** Make sure you are inside the vLLM ROCm container (see Environment Setup above).

```bash
# Inside the container, navigate to the mix_precision directory
cd /workspace/quark/examples/torch/experimental/mix_precision

# Run with default settings (searches all modes for MI355)
python mix_precision.py \
    --model_dir /models/Qwen3.5-397B-A17B \
    --hardware mi355 \
    -tp 8 \
    --gpu-memory-utilization 0.8 \
    --export_best_model
```

This evaluates all candidate configs using GSM8K (5-shot) and exports the best model
to `Qwen3.5-397B-A17B-bestquantconfig/`.

**Expected output (Qwen3.5-397B-A17B on MI355, 8xGPU):**

```text
Original model gsm8k: 0.8931
--------------------------------------------------------------------------------------
Rank  gsm8k   linear_attn  self_attn  mlp        kv_cache  attention  is_best_config
--------------------------------------------------------------------------------------
1     0.8863  native       native     ptpc_fp8   native    native     NO
.
.
.
12    0.9007  ptpc_fp8     mxfp4_fp8  mxfp4_fp8  native    native     YES
13    0.8158  ptpc_fp8     mxfp4_fp8  mxfp4      native    native     NO
14    0.7695  native       mxfp4      mxfp4      native    native     NO
15    0.7976  ptpc_fp8     mxfp4      mxfp4      native    native     NO
.
.
.
19    0.6687  mxfp4        mxfp4      mxfp4      native    native     NO
```

## Workflow

1. **Load model on meta device** — builds the layer graph for quantization config
   generation with no GPU memory cost
2. **Generate candidate configs** — sorted from least to most aggressive quantization
   (e.g. rank 1: `self_attn=native, mlp=ptpc_fp8`; last rank: `self_attn=mxfp4, mlp=mxfp4`)
3. **Evaluate baseline** — run GSM8K on the original (unquantized) vLLM model
4. **Search loop** — for each config in rank order:
   - Re-quantize the vLLM model in-process via `QuarkFakeQuantWorker`
   - Evaluate GSM8K accuracy
   - Check if accuracy is within `--eval_threshold`
   - Reset model back to original state
5. **Select best config** — the most aggressively quantized config that passes the threshold
6. **Export** (optional with `--export_best_model`) — apply best config and save as safetensors

## Arguments

### Required

| Argument      | Description                                        |
|---------------|----------------------------------------------------|
| `--model_dir` | HuggingFace model path or local directory          |

### Key Options

| Argument                | Default                   | Description                                                                                   |
|-------------------------|---------------------------|-----------------------------------------------------------------------------------------------|
| `--hardware`            | `mi300`                   | Target hardware: `mi300`, `mi325`, `mi355`                                                    |
| `--granularity`         | `module`                  | Search granularity (only `module` is currently supported)                                     |
| `--eval_threshold`      | `1.02`                    | Max allowed accuracy degradation ratio (1.02 = allow up to 2% relative drop in GSM8K)        |
| `--gsm8k_num_samples`   | `1319`                    | Number of GSM8K questions to evaluate (max 1319)                                              |
| `--gsm8k_max_new_tokens`| `256`                     | Max tokens to generate per GSM8K question                                                     |
| `--num_calib_samples`   | `64`                      | Number of calibration samples                                                                 |
| `--calib_seq_len`       | `2048`                    | Calibration sequence length                                                                   |
| `--max_configs`         | `None`                    | Limit number of configs to evaluate; `None` evaluates all                                     |
| `--early_stop`          | False                     | Stop as soon as a config exceeds the threshold                                                |
| `--export_best_model`   | False                     | Export the best-config model as safetensors                                                   |
| `--output_dir`          | `<model>-bestquantconfig` | Export directory                                                                              |
| `--skip-baseline-eval`  | False                     | Skip baseline GSM8K evaluation before search                                                  |
| `--kv-cache-quant` / `--no-kv-cache-quant` | False        | Include KV cache quantization in the search. Default: off (KV cache stays native — KV FP8 not yet supported). |
| `--self-attn-only`      | False                     | Search only self-attention layers; keep MLP native                                            |
| `--mlp-only`            | False                     | Search only MLP/MoE layers; keep self-attention native                                        |
| `--search_configs`      | (all modes for hardware)  | Quantization modes to search; e.g. `--search_configs native ptpc_fp8 mxfp4`                  |
| `--exclude`             | (defaults)                | Layer-name patterns to exclude from quantization (shell-quote wildcards)                      |

Any additional flags (e.g. `--tensor-parallel-size 4`) are forwarded directly to vLLM.

## Output

Results are printed as an ASCII table sorted from least to most aggressive quantization
(rank 1 = least aggressive / highest precision):

```text
--------------------------------------------------------------------------------------------
Rank  gsm8k            self_attn        mlp              is_best_config
--------------------------------------------------------------------------------------------
1     0.8121           native           ptpc_fp8         NO
2     0.8034           fp8              fp8              YES
3     0.7412           ptpc_fp8         fp8              NO
--------------------------------------------------------------------------------------------
```

A dot plot of metric vs. rank is also printed, with `@` marking the best config and
`-` marking the baseline.
