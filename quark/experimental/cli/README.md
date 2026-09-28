# AMD Quark CLI User Guide

`quark-cli` is the primary command-line interface for the AMD Quark quantizer, used to optimize machine learning models. This guide will help you understand how to install, use, and configure `quark-cli` through its various subcommands.

## 1. Introduction

`quark-cli` offers several subcommands, each targeting a specific quantization workflow. You can configure the quantization process in detail using command-line arguments.

### Syntax

```bash
quark-cli [SUBCOMMAND] [ARGUMENTS ...]
```

### Getting Help

* **List all available subcommands:**

    ```bash
    quark-cli -h
    ```

* **Get help for a specific subcommand:** Place the `-h` option after the subcommand, for example:

    ```bash
    quark-cli torch-llm-ptq -h
    ```

## 2. Preparation

Before using `quark-cli`, ensure your Python environment meets the following requirements:

### 2.1 Install AMD Quark

Quark must be installed in your Python environment. This typically requires PyTorch (supporting both CPU and GPU). Refer to the official Quark documentation for detailed installation instructions: [https://quark.docs.amd.com/latest/install.html](https://quark.docs.amd.com/latest/install.html)

### 2.2 Install `quark-cli` Dependencies

`quark-cli` has its own optional dependencies. Install them with:

```bash
pip install amd-quark[cli]
```

Or, from a local source checkout:

```bash
pip install -r requirements-cli.txt
```

## 3. Overview of Main Subcommands

`quark-cli` provides the following main subcommands:

* **`torch-llm-ptq`**: Post-training quantization for PyTorch LLM (Large Language Models).

This guide will focus on the `torch-llm-ptq` subcommand.

## 4. Detailed `torch-llm-ptq` Subcommand

`torch-llm-ptq` is used for post-training quantization of PyTorch LLMs.

### 4.1 Example Usage

Before running, replace `MODEL_DIR` and `OUTPUT_DIR` with your chosen input and output directories.

```bash
MODEL_DIR=dev/models/Llama-3.1-8b/
OUTPUT_DIR=dev/models_output/
```

### Example 1: Basic Quantization

```bash
quark-cli torch-llm-ptq --model_dir $MODEL_DIR --output_dir $OUTPUT_DIR --quant_scheme w_int8
```

### Example 2: Quantization with Calibration and Layer Exclusion

```bash
quark-cli torch-llm-ptq \
    --model_dir $MODEL_DIR \
    --output_dir $OUTPUT_DIR \
    --quant_scheme w_int8 \
    --dataset wikitext \
    --num_calib_data 128 \
    --exclude_layers "*down_proj*"
```

### 4.2 Common Parameters

Here are some common and important parameters for the `torch-llm-ptq` subcommand:

#### Model Parameters

* `--model_dir <PATH>` **(Required)**: Specify where the HuggingFace model is. Supports models like Llama, OPT, etc.
* `--device {cuda,cpu}`: Device for running the quantizer. Defaults to `cuda` if available, otherwise `cpu`.
* `--multi_device`: Allow using this mode to run a model quantization that exceeds the size of your GPU memory. Note that this can lead to very slow quantization.
* `--no_trust_remote_code`: Disable execution of custom model code from the Hub (safer, recommended if unsure).

#### Calibration Dataset Parameters

* `--dataset {pileval,wikitext,cnn_dailymail,...}`: Dataset for calibration. Defaults to `pileval`.
* `--seq_len <INT>`: Sequence length of data. Defaults to 512.
* `--batch_size <INT>`: Batch size for calibration. Defaults to 1.
* `--num_calib_data <INT>`: Number of samples for calibration. Defaults to 512.

#### Quantization Parameters

* `--quant_scheme <SCHEME>`: Supported quantization scheme name (e.g., `w_int8`, `w_mxfp8`). Must be a built-in quantization scheme supported by LLMTemplate.
* `--layer_quant_scheme <PATTERN> <QUANT_SCHEME>`: Directly specify a quantization scheme for layers matching the given pattern. Can be repeated. Example: `--layer_quant_scheme lm_head int4_wo_32`.
* `--kv_cache_dtype <TYPE>` (or `--kv_cache_quant_scheme`): KV Cache dtype (e.g., `fp8`, `int8_per_tensor_dynamic`).
* `--quant_algo <alg1,alg2,...>`: Comma-separated list of algorithms. Options include `awq`, `gptq`, `smoothquant`, `autosmoothquant`, `rotation`, `quarot`.
* `--exclude_layers <LAYER1> <LAYER2> ...`: List of layers to exclude from quantization. Usage: `--exclude_layers "*down_proj*" "*k_proj"`.

#### Output Parameters

* `--output_dir <PATH>`: Output directory for the exported model. Defaults to `exported_model`.

#### Evaluation Parameters

* `--skip_evaluation`: Skip model evaluation.
* `--evaluation_dataset {wikitext,wikitext_gpt_oss_120b}`: Dataset for evaluation. Defaults to `wikitext`.

### 4.3 Detailed Workflow (`torch-llm-ptq`)

The `torch-llm-ptq` subcommand performs the following steps:

1. **Define Original Model**:
    * Loads the HuggingFace model and tokenizer/processor from the specified `--model_dir`.
    * Example: `quark-cli torch-llm-ptq --model_dir dev/models/Llama-3.1-8b/`

2. **Define Calibration Data Loader**:
    * Prepares the calibration dataset based on parameters like `--dataset`, `--num_calib_data`, `--seq_len`, etc.
    * Example: `quark-cli torch-llm-ptq --dataset pileval --num_calib_data 128`

3. **Quantization**:
    * Sets up the quantization configuration based on parameters like `--quant_scheme`, `--layer_quant_scheme`, etc.
    * Uses `ModelQuantizer` to replace model modules with quantized versions in place.
    * Example: `quark-cli torch-llm-ptq --model_dir $MODEL_DIR --quant_scheme w_mxfp8`

4. **Model Freezing and Export**:
    * Freezes the quantized model.
    * Exports the model to the `--output_dir` in Hugging Face `safetensors` format.

5. **Evaluation**:
    * If `--skip_evaluation` is not specified, evaluates the quantized model using the specified `--evaluation_dataset`.

## 5. Custom LLM Templates (`--template_file`)

`torch-llm-ptq --template_file` lets you quantize models whose `model_type` is not covered by the built-in templates, by loading a custom LLM template definition from a JSON file.

### 5.1 Template JSON Format

Only `model_type` is required; all other keys are optional.

```json
{
    "model_type": "my_custom_model",
    "kv_layers_name": ["*k_proj", "*v_proj"],
    "q_layer_name": "*q_proj",
    "gate_up_layers_name": ["gate_proj", "up_proj"],
    "exclude_layers_name": ["lm_head"],
    "algorithm_configs": {
        "awq": {"name": "awq", "model_decoder_layers": "model.layers"},
        "gptq": {"name": "gptq", "block_size": 128, "damp_percent": 0.01}
    }
}
```

| Key | Type | Required | Description |
| --- | --- | --- | --- |
| `model_type` | string | Yes | The HuggingFace `model_type` of the model this template applies to. |
| `kv_layers_name` | list of strings | No | Name patterns of the K/V projection layers. |
| `q_layer_name` | string or list of strings | No | Name pattern(s) of the Q projection layer. |
| `gate_up_layers_name` | list of strings | No | Gate/up projection layer names for shared scale groups. Defaults to `["gate_proj", "up_proj"]`. |
| `exclude_layers_name` | list of strings | No | Layer name patterns excluded from quantization. Defaults to `[]`. |
| `algorithm_configs` | object | No | Quantization algorithm configurations registered with the template, keyed by algorithm name (e.g. `awq`, `gptq`, `smoothquant`, `rotation`). Each value is a JSON object in the same format as `quark.torch.quantization.load_quant_algo_config_from_file` (with a `name` field plus algorithm-specific fields). This allows `--quant_algo` to be used with models that have no built-in algorithm configuration. |

### 5.2 Example Usage

```bash
quark-cli torch-llm-ptq --model_dir $MODEL_DIR --output_dir $OUTPUT_DIR \
    --template_file my_template.json --quant_scheme fp8 --quant_algo awq
```

### 5.3 `torch-llm-ptq` Template Parameter

* `--template_file <PATH>`: Path to a JSON file defining a custom LLM template (see 5.1 for the format). The template is registered before quantization, which allows quantizing models whose `model_type` is not covered by the built-in templates. The `model_type` in the file must match the `model_type` of the loaded model.

Note: if the `model_type` in the file matches an existing (built-in) template, the file's template overwrites it for the duration of the CLI run (a warning is logged by Quark).
