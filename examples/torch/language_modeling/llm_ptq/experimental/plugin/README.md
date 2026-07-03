# vLLM Serve with Quark Fake Quantization

This example demonstrates how to serve quantized models using Quark fake quantization with vLLM. The quantization process leverages `quark.experimental.plugin.vllm_plugin` to automatically handle vLLM-specific layer types without manual replacement.

## Overview

The `vllm_serve_fakequant.py` and related files provide integration between Quark quantization and vLLM, enabling automatic quantization during model loading and serving.

## Files

- **`vllm_serve_fakequant.py`**: Main entry point that starts the vLLM server and configures it to use `QuarkFakeQuantWorker`
- **`vllm_generate_fakequant.py`**: Script for offline inference using `LLM.generate` method with Quark quantization
- **`gsm8k_eval_generate.py`**: GSM8K evaluation script using `LLM.generate` method (no API server needed)

Note: The `QuarkFakeQuantWorker` class is implemented in `quark.experimental.plugin.fakequant_worker`.

## Prerequisites

- vLLM installed (supported version: v0.16.0)
- Quark installed
- HuggingFace token configured (if accessing private models or datasets)

## Environment Variables

The quantization behavior is controlled through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `QUANT_CFG` | Quantization path to QConfig JSON file | `None` |
| `QUANT_DATASET` | Calibration dataset name | `cnn_dailymail` |
| `QUANT_CALIB_SIZE` | Number of calibration samples | `512` |

### Supported Datasets

- `cnn_dailymail` (default)
- `pileval`
- `wikitext`
- `pileval_for_awq_benchmark`
- `wikitext_for_gptq_benchmark`

### Quantization Configuration

The `QUANT_CFG` environment variable can be set like this:

 **JSON Config File**: Provide a path to a QConfig JSON file

   ```bash
   export QUANT_CFG=/path/to/quantization_config.json
   ```

When using a scheme name, the system will automatically detect the model type from vLLM's configuration and use the appropriate `LLMTemplate` to generate the quantization config. If auto-detection fails, you can explicitly set `QUANT_MODEL_TYPE`.

## Usage

### Option 1: Using LLM.generate (Offline Inference)

For offline inference using the `LLM.generate` method:

```bash
# Set path to quantization config file
export QUANT_CFG=/path/to/quantization_config.json
export QUANT_DATASET=pileval
export QUANT_CALIB_SIZE=64

# Run generation
python3 vllm_generate_fakequant.py <model_path> \
    --prompt "Hello, my name is" \
    --max-tokens 100 \
    --temperature 0.7 \
    --tensor-parallel-size 1 \
    --enforce-eager

# Or with multiple prompts
python3 vllm_generate_fakequant.py <model_path> \
    --prompts "Hello" "The capital of France is" \
    --max-tokens 50 \
    --enforce-eager
```

### Option 2: Using API Server

For serving via OpenAI-compatible API:

```bash
# Set path to quantization config file
export QUANT_CFG=/path/to/quantization_config.json
export QUANT_DATASET=pileval
export QUANT_CALIB_SIZE=64

# Start the server
python3 vllm_serve_fakequant.py <model_path> -tp 1 --host 0.0.0.0 --port 8000 --enforce-eager
```

## Quantization Config File Format

When using a JSON config file, it should follow Quark's QConfig format. Example:

```json
{
    "global_quant_config": {
        "input_tensors": {
            "dtype": "fp8_e4m3",
            "is_dynamic": false,
            "qscheme": "per_tensor",
            "symmetric": true,
            "round_method": "half_even",
            "scale_type": "float",
            "observer_cls": "PerTensorMinMaxObserver"
        },
        "weight": {
            "dtype": "fp8_e4m3",
            "is_dynamic": false,
            "qscheme": "per_tensor",
            "symmetric": true,
            "round_method": "half_even",
            "scale_type": "float",
            "observer_cls": "PerTensorMinMaxObserver"
        }
    },
    "exclude": ["lm_head"],
    "layer_quant_config": {}
}
```

## How It Works

1. **Plugin Registration**: When the worker initializes, `register_vllm_quantization_plugins()` is called to register vLLM-specific layer types with Quark's quantization system.

2. **Config Loading**: The worker loads the quantization configuration either from:
   - A JSON file (if `QUANT_CFG` points to a file)
   - A predefined scheme via `LLMTemplate` (if `QUANT_CFG` is a scheme name)

3. **Config Adaptation**: The configuration is automatically adapted for vLLM's layer naming conventions:
   - `q_proj`, `k_proj`, `v_proj` patterns are mapped to `qkv_proj`
   - `gate_proj`, `up_proj` patterns are mapped to `gate_up_proj`

4. **Model Preparation**: The model is prepared with quantization modules using `ModelQuantizer`.

5. **Calibration**: A calibration dataset is loaded and used to calibrate the quantization observers.

6. **Freezing**: After calibration, the model is frozen to apply the quantization parameters.

## Testing the Server

Once the server is running, you can test it using curl:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model_path>",
    "messages": [
      {"role": "user", "content": "Please introduce Tom and Jerry"}
    ],
    "max_tokens": 100
  }'
```

Or use the OpenAI Python client:

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

response = client.chat.completions.create(
    model="<model_path>",
    messages=[
        {"role": "user", "content": "Please introduce Tom and Jerry"}
    ]
)

print(response.choices[0].message.content)
```
