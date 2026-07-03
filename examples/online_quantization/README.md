# vLLM Online Quantization

## Background

vLLM's built-in online quantization (`quantization="online"`) supports per-tensor FP8, block-scale FP8, and MXFP8. It does not cover per-channel FP8, MXFP4, or mixed linear/MoE schemes.

A second gap: checkpoints that already carry an offline quantization config (e.g., DeepSeek-R1 ships with FP8 block-scale weights) cannot be re-quantized to a different scheme at load time — for example, to per-channel FP8 to match Quark's offline `ptpc_fp8` output and reuse the same inference kernels.

## Design

`quark.online_quantization.vllm` introduces `QuarkVllmOnlineConfig`, registered in vLLM as the `"quark_online"` quantization backend. It quantizes weights at model-load time inside vLLM's `process_weights_after_loading` hook.

Internally the config holds two optional inner configs:

- **`_online_quant_config`** (`QuarkConfig`): the target online scheme. Per-layer config lookup (fnmatch patterns in `layer_quant_config`, `layer_type_quant_config`, `global_quant_config` fallback, fused-shard consistency) is delegated entirely to the existing `QuarkConfig` machinery.
- **`_offline_quant_config`** (optional): the checkpoint's existing offline config, parsed from the HF `quantization_config` field. When present, the offline method loads the on-disk quantized weights; the online path then converts them back to bf16 and re-quantizes to the target scheme.

This covers two scenarios:

- **Scenario A** — unquantized (bf16/fp16) checkpoint: the online method quantizes directly from bf16.
- **Scenario B** — offline-quantized checkpoint (currently supports `quant_method: "fp8"`, both per-channel and block-scale variants): the offline method loads the quantized weights; `OnlineRequantMethod` / `OnlineRequantMoeMethod` dequantize to bf16, then delegate to the online method.

The `hf_overrides` entry point is a picklable callable (`_OnlineQuantHfOverride`) that merges the chosen online config into the HF `quantization_config` at load time. For Scenario B it also preserves the original offline config under an `offline_quant` sub-key, where `QuarkVllmOnlineConfig.from_config` picks it up.

## Usage

### Prerequisites

- vLLM 0.21
- AMD Quark installed

Quark online quantization can be used in two ways:

- **Python API**: pass `quantization="quark_online"` and `hf_overrides` when constructing `vllm.LLM`.
- **Native vLLM CLI**: pass `online_quant_config` through `vllm serve --additional-config`.

### Built-in presets

Three presets are available via `HF_QUANTIZATION_CONFIGS`:

| Python key | Scheme |
|------------|--------|
| `ptpc_fp8` | FP8 E4M3, per-channel weight + dynamic per-channel activation |
| `mxfp4` | MXFP4, per-group (group size 32) with E8M0 block scale |
| `linear_ptpc_fp8_moe_mxfp4` | Mixed: attention in FP8, MoE experts in MXFP4 |

> **Note:** The Python API and the native vLLM CLI both use `ptpc_fp8` for the per-channel FP8 online quantization scheme.

### Python API example

```bash
python3 vllm_online_quantization.py
```

The script runs `ptpc_fp8` on `Qwen/Qwen3-30B-A3B-Thinking-2507` by default.

To use a different model, edit the `model_name` variable inside `main()`.

To switch presets, use the same API with a different `quant_scheme`:

```python
from vllm import LLM, SamplingParams

from quark.online_quantization.vllm import HF_QUANTIZATION_CONFIGS


def main(quant_scheme: str = "mxfp4"):
    hf_overrides = HF_QUANTIZATION_CONFIGS[quant_scheme]

    llm = LLM(
        model="Qwen/Qwen3-30B-A3B-Thinking-2507",
        quantization="quark_online",
        enforce_eager=True,
        tensor_parallel_size=1,
        hf_overrides=hf_overrides,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=100)
    output = llm.generate(["The capital of France is"], sampling_params=sampling_params)
    print(f"[{quant_scheme}] output = {output}")


if __name__ == "__main__":
    main(quant_scheme="mxfp4")
```

### Native vLLM CLI example

After AMD Quark is installed, the Quark online quantization plugin can also be used through native vLLM commands. Pass `online_quant_config` with `--additional-config`; the example below corresponds to the `ptpc_fp8` preset:

```bash
export VLLM_PLUGINS="${VLLM_PLUGINS:-quark_online_quant}"

ONLINE_QUANT_CONFIG='{"online_quant_config": {"global_quant_config": "ptpc_fp8", "exclude_layer": ["lm_head"]}}'

vllm serve Qwen/Qwen3-8B \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --additional-config "$ONLINE_QUANT_CONFIG"
```

### Re-quantizing an offline checkpoint

Pass the same `hf_overrides` for a checkpoint that already carries an offline `quantization_config`. The callable detects the existing config and merges it automatically — no extra arguments needed.

```python
from vllm import LLM
from quark.online_quantization.vllm import HF_QUANTIZATION_CONFIGS

llm = LLM(
    model="deepseek-ai/DeepSeek-R1",   # carries quant_method: "fp8" in HF config
    quantization="quark_online",
    hf_overrides=HF_QUANTIZATION_CONFIGS["ptpc_fp8"],
    tensor_parallel_size=8,
)
```

### Custom online quantization configs with the Python API

For advanced use cases, `online_quant_overrides` can wrap a Quark online quantization config dict and pass it through `hf_overrides`:

```python
from vllm import LLM
from quark.online_quantization.vllm import online_quant_overrides

my_config = {
    "quant_method": "quark_online",
    "quant_mode": "eager_mode",
    "exclude": ["lm_head"],
    "global_quant_config": {
        "weight": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_channel",
            "ch_axis": 0,
            "is_dynamic": False,
        },
        "input_tensors": {
            "dtype": "fp8_e4m3",
            "qscheme": "per_channel",
            "ch_axis": 1,
            "is_dynamic": True,
        },
    },
    "layer_quant_config": {},
    "layer_type_quant_config": {},
}

llm = LLM(
    model="Qwen/Qwen3-8B",
    quantization="quark_online",
    hf_overrides=online_quant_overrides(my_config),
)
```
