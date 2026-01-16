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

`quark-cli` also has its own dependencies. If you have a local copy of Quark, you can install them from the `quark/experimental/cli/requirements.txt` file:

```bash
pip3 install -r quark/experimental/cli/requirements.txt
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
