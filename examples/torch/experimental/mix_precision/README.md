# Mix Precision Auto-Search

Automatically finds the optimal mixed-precision quantization configuration for an LLM.
It uses a hardware-aware decode Roofline to walk adjacent performance candidates and
selects the fastest evaluated configuration whose accuracy stays within the threshold.

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
  vllm/vllm-openai-rocm:v0.25.0 \
  bash
```

**Path mapping guide:**

- Replace `/path/to/models` with your host model directory (e.g., `/data/Qwen`)
- Replace `/path/to/Quark` with your host Quark repo path (e.g., `/home/user/Quark`)
- Inside container: models will be at `/models/`, Quark at `/workspace/quark/`

**Example:**

```bash
docker run --rm -it \
  --device /dev/kfd --device /dev/dri \
  --group-add video \
  -v /data/Qwen:/models \
  -v /path/to/Quark:/workspace/Quark \
  vllm/vllm-openai-rocm:v0.25.0 \
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

The workflow preserves vLLM's default ``auto`` MoE backend selection. Explicit
``auto``, standard ``triton``, ``emulation``, ``aiter``, and
``aiter_mxfp4_bf16`` selections are accepted. For prequantized
``AITER_MXFP4_BF16`` sources, Quark retains the packed source weights and
injects target activation QDQ between the two AITER MoE stages. AITER one-stage
execution fails closed when target a2 QDQ is required. ``triton_unfused`` and
prequantized Quark W4A8 MoE sources that require inverse conversion remain out
of scope.

### Python API (Recommended)

```python
from quark.experimental.torch.mix_precision import MixPrecisionConfig, MixPrecisionQuantizer

config = MixPrecisionConfig(
    hardware="mi355",
    eval_metrics=["gsm8k"],
    eval_threshold=1.02,
    search_modes=["native", "ptpc_fp8", "mxfp4"],
    early_stop=True,
    file2file_quantization=True,
)

quantizer = MixPrecisionQuantizer(config)
result = quantizer.search(
    model_path="/models/Qwen3.5-397B-A17B",
    runtime_args=[
        "-tp",
        "8",
        "--gpu-memory-utilization",
        "0.8",
    ],
)

if result.best_config is not None:
    quantizer.export_best("/output/Qwen3.5-397B-A17B-bestquantconfig")
```

File-to-file mode avoids loading the full model for calibration-free configs
(`ptpc_fp8`, `mxfp4`, and `mxfp6_e2m3`). Before evaluation, Quark prints a warning
and removes every candidate containing a calibration-dependent mode (`fp8` or
`mxfp4_fp8`, also known as W4A8) from the search space. The export-time fallback
remains as a safety check for API callers that request file-to-file only after search.

Candidate generation, model partition detection, default exclusions, Roofline ordering,
early stopping, vLLM reset, and result construction are internal to `MixPrecisionQuantizer`.

### CLI

```bash
# Inside the container, navigate to the mix_precision directory
cd /workspace/quark/examples/torch/experimental/mix_precision

# Search all supported modes for MI355 (full sweep; omit --early_stop to
# evaluate every candidate)
python mix_precision.py \
    --model_dir /models/Qwen3.5-397B-A17B \
    --hardware mi355 \
    -tp 8 \
    --gpu-memory-utilization 0.8 \
    --export_best_model \
    --file2file_quantization
```

Both examples evaluate candidates using GSM8K (5-shot) and optionally export the best model.

**Expected output shape (abridged; scores vary by model/runtime):**

```text
Original model gsm8k: 0.8931
File-to-file quantization was requested, but some generated candidate configs require calibration
and cannot use file-to-file quantization. Removing them from the search space.
--------------------------------------------------------------------------------------
Rank  gsm8k   linear_attn  self_attn  dense_mlp  routed_moe  kv_cache  attention  is_best_config
--------------------------------------------------------------------------------------
1     0.8863  native       native     native     mxfp4      native    native     NO
.
.
.
N     0.8870  ptpc_fp8     ptpc_fp8   ptpc_fp8   mxfp4      native    native     YES
.
.
.
M     0.6687  mxfp4        mxfp4      mxfp4      mxfp4      native    native     NO
```

Calibration-dependent `fp8` and `mxfp4_fp8` rows do not appear when
`--file2file_quantization` is enabled.

## Workflow

1. **Load model on meta device** — builds the layer graph for quantization config
   generation with no GPU memory cost
2. **Generate candidate configs** — constrained by the target hardware and user-selected modes;
   with `--file2file_quantization`, warn and remove candidates that require calibration
3. **Compute the Roofline** — walk the meta model's Linear shapes and score every candidate with
   per-op GEMM, FusedMoE, and SDPA ceilings, with aggregate memory as a fallback
   (default `ISL=8192`, `OSL=1024`); report a separate prefill Roofline without
   using it for candidate ordering
4. **Evaluate baseline** — run GSM8K on the original (unquantized) vLLM model
5. **Search loop** — start from the hardware anchor and walk adjacent Roofline candidates:
   - Re-quantize the vLLM model in-process via `QuarkFakeQuantWorker`
   - Evaluate GSM8K accuracy
   - Check if accuracy is within `--eval_threshold`
   - Reset model back to original state
6. **Select best config** — the highest-scoring evaluated config under the Roofline model that passes the threshold
7. **Export** (optional with `--export_best_model`) — apply the best config and save as
   safetensors. `--file2file_quantization` uses shard-by-shard export; its search-space
   filtering guarantees the selected config is calibration-free.

All supported targets use the same Roofline-guided search. MI300 and MI325 start from
`routed_moe=ptpc_fp8` (or `dense_mlp=ptpc_fp8` on dense models); MI355 starts from
`routed_moe=mxfp4` (or `dense_mlp=mxfp4`). Other layer partitions start as `native`.

## Arguments

### Required

| Argument      | Description                                        |
|---------------|----------------------------------------------------|
| `--model_dir` | Hugging Face model ID or local model directory     |

### Key Options

| Argument                | Default                   | Description                                                                                   |
|-------------------------|---------------------------|-----------------------------------------------------------------------------------------------|
| `--hardware`            | `mi355`                   | Target hardware: `mi300`, `mi325`, `mi355`                                                    |
| `--granularity`         | `module`                  | Search granularity (only `module` is currently supported)                                     |
| `--eval_threshold`      | `1.02`                    | Max allowed accuracy degradation ratio (1.02 = allow up to 2% relative drop in GSM8K)        |
| `--gsm8k_num_samples`   | `1319`                    | Number of GSM8K questions to evaluate (max 1319)                                              |
| `--gsm8k_max_new_tokens`| `256`                     | Max tokens to generate per GSM8K question                                                     |
| `--num_calib_samples`   | `64`                      | Number of calibration samples                                                                 |
| `--calib_seq_len`       | `2048`                    | Calibration sequence length                                                                   |
| `--max_configs`         | `None`                    | Limit evaluations after full-space Roofline scoring; `None` evaluates all                     |
| `--early_stop`          | False                     | Stop as soon as the accuracy frontier is crossed (the first config beyond the accuracy gate), instead of sweeping every candidate. Best for accuracy-sensitive runs; leave off to evaluate all modes |
| `--export_best_model`   | False                     | Export the best-config model as safetensors                                                   |
| `--file2file_quantization` | False                  | Remove calibration-dependent candidates with a warning, resolve Hub IDs locally, require safetensors input, then use shard-by-shard export for the best remaining config |
| `--output_dir`          | `<model>-bestquantconfig` | Export directory                                                                              |
| `--skip-baseline-eval`  | False                     | Skip baseline GSM8K evaluation before search                                                  |
| `--kv-cache-quant` / `--no-kv-cache-quant` | False        | Include KV cache quantization in the search. Default: off (KV cache stays native — KV FP8 not yet supported). |
| `--search_configs`      | (all modes for hardware)  | Quantization modes to search; e.g. `--search_configs native ptpc_fp8 mxfp4`                  |
| `--exclude`             | (defaults)                | Layer-name patterns to exclude from quantization (shell-quote wildcards)                      |

Any additional flags (e.g. `--tensor-parallel-size 4`) are forwarded directly to vLLM.

## Output

Results are printed as an ASCII table sorted from least to most aggressive quantization
(rank 1 = least aggressive / highest precision):

```text
--------------------------------------------------------------------------------------------
Rank  gsm8k            self_attn        dense_mlp        is_best_config
--------------------------------------------------------------------------------------------
1     0.8121           native           ptpc_fp8         NO
2     0.8034           fp8              fp8              YES
3     0.7412           ptpc_fp8         fp8              NO
--------------------------------------------------------------------------------------------
```

A dot plot of metric vs. rank is also printed, with `@` marking the best config and
`-` marking the baseline.
