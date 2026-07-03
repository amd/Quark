<!-- Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved. -->

# Release Notes

## Release 0.12

AMD Quark 0.12 is tested against PyTorch 2.10 and 2.11, and compatible with upstream `transformers==4.57.6` and `transformers==5.2`.

### AMD Quark Infrastructure

#### New Features

- Support for Python 3.11 up to 3.13
- Bumped the minimum required `numpy` to `>= 2.0` across the ONNX and PyTorch flows.
- **Pre-built wheels** are now published for PyTorch 2.10+ on the **AMD package index** (CPU, CUDA 12.8, ROCm 7.1, ROCm 7.2; Linux/Windows; Python 3.11–3.13). They ship pre-compiled C++ extensions, so no C++ compiler is needed and the first `import quark` no longer triggers a one-time kernel/custom-op build. To fetch a pre-built wheel, point `pip` at the matching index:

  ```bash
  pip install amd-quark --extra-index-url https://pypi.amd.com/quark/cpu/simple    # CPU
  pip install amd-quark --extra-index-url https://pypi.amd.com/quark/cu128/simple  # CUDA 12.8
  pip install amd-quark --extra-index-url https://pypi.amd.com/quark/rocm71/simple # ROCm 7.1 (Linux only)
  pip install amd-quark --extra-index-url https://pypi.amd.com/quark/rocm72/simple # ROCm 7.2 (Linux only)
  ```

#### Deprecations and breaking changes

- The `quark.testing` module has been deprecated and removed. All testing utilities have been consolidated into `quark.common.utils.testing_utils`. Update your imports as follows:
  - `from quark.testing import skip_if_no_gpu, slow_test, slow_test_if` → `from quark.common.utils.testing_utils import skip_if_no_gpu, slow_test, slow_test_if`
  - `from quark.testing.common_utils import TestCase` → `from quark.common.utils.testing_utils import TestCase`

### Quark Shapeshifter (formerly Quark ONNX Adapter)

#### New Features

- Added support for Quark ONNX post-processing workflows, including Q/DQ cleanup, scale alignment, bfloat16 adaptation, and XINT8/NPU simulation.
- Added support for Quark Torch workflows through PyTorch model transformation passes, including dropout removal and model tracing.
- Added automatic pass discovery and registration for official passes in `quark/shapeshifter/passes/` and optional community passes in `quark/contrib/shapeshifter_community_passes/`. Shapeshifter validates pass types so that each workflow contains either ONNX passes or PyTorch passes, but not both.

#### Deprecations and breaking changes

- The `quark-cli onnx-adapter` command is deprecated and will be removed in a future release. Please use `quark-cli shapeshifter` instead. Both commands are functionally identical during this release to provide backward compatibility.

### AMD Quark for PyTorch

#### New Features

- Support for Python 3.11 up to 3.13
- Support NVFP4 quantization (scheme: `nvfp4`).
- Support FP4 quantization with E5M3 per-block scales, called AMDFP4 quantization (`amdfp4`, `amdfp4_g32`).
- Support FP4 quantization with E5M3 per-block scales and a global FP32 scale (schemes: `amdfp4_global16`, `amdfp4_global32`).
- Support native inference for xDiT and Diffusers workflows
- Pre-quantized layers excluded from quantization (`FP8Linear`, compressed-tensors, HF-dequantized MXFP4) are now preserved in their original format on export instead of being dequantized to bf16/fp16.
- Support for `compressed-tensors==0.15` in PyTorch export/import and file-to-file quantization flows.

#### Model Support

Supported out-of-box model architectures:

- DeepSeek-V4-Pro, DeepSeek-V4-Flash
- GLM-5, GLM-5.1, GLM-5.2
- Kimi-K2.5, Kimi-K2.6 ([Reference](https://quark.docs.amd.com/latest/pytorch/quantizing_large_models.html))
- MiniMax-M2.5, MiniMax-M2.7, MiniMax-M3
- Qwen3.5-397B-A17B, Qwen3.5-35B-A3B

#### Bug fixes and minor improvements

- Fixed MXFP4 dequantization kernel failures for large tensor shapes.
- Fixed E5M3 Triton kernel dispatch on correct device in multi-device setting.
- Fixed `LLMTemplate` validation to raise a clear error when a required algorithm configuration is missing.
- Fixed a bug where MOE calibration diagnostics was not warning when static activation quantizers were not receiving calibration tokens.
- Fixed AWQ scaling for Qwen3.5-style RMSNorm.

#### Diffusion model quantization and Hugging Face Diffusers integration

- AMD Quark now plugs directly into Hugging Face `diffusers`. Importing `quark.integrations.diffusers` self-registers Quark into the diffusers `AUTO_QUANTIZER_MAPPING` / `AUTO_QUANTIZATION_CONFIG_MAPPING`, so quantized diffusion models can be saved and reloaded through the standard `save_pretrained` / `from_pretrained` APIs:

  ```python
  from quark.integrations import diffusers  # registers the "quark" quantizer
  from diffusers import DiffusionPipeline

  pipe = DiffusionPipeline.from_pretrained("<org>/<sdxl-or-flux-quark-checkpoint>")
  ```

- **Export**: `DiffusersSafetensorsExporter` writes a quantized pipeline submodule via `save_pretrained`, embedding the serialized Quark `QConfig` under `quantization_config` in `config.json`. **Reload** reconstructs the quantized layers automatically (meta-device / `low_cpu_mem_usage` loading supported) and freezes them for inference.
- **On-the-fly quantization**: a pipeline submodule (`pipe.unet` / `pipe.transformer`) can be quantized in-process, without a separate export/reload round-trip.
- **Calibration utilities promoted into the library**: `quark.torch.utils.diffusers.get_calib_dataloader(pipe, target_module, prompts, n_steps=...)` runs the pipeline, captures the submodule's intermediate inputs, and returns a dataloader ready for `ModelQuantizer.quantize_model` — no more copying calibration code out of the examples.
- Works with round-to-nearest, SmoothQuant, and SVDQuant. A PR to add Quark to `diffusers` upstream is planned; the self-registration path works today.

#### SVDQuant (SVD-based low-bit error correction)

- Added **SVDQuant** (`quark.torch.algorithm.svdquant`, configured via `SVDQuantConfig`), from [SVDQuant: Absorbing Outliers by Low-Rank Components for 4-Bit Diffusion Models](https://arxiv.org/abs/2411.05007). It pairs SmoothQuant-style smoothing with a high-precision low-rank correction branch, making INT4 / MXFP4 / NVFP4 weight quantization (optionally with 4-bit activations) viable. The algorithm applies to **both diffusion models and LLMs**.
- Ready-made schemes via `build_quant_layer_config`: `w4a16`, `w4a4`, `mxfp4`, and `nvfp4`. Optional **GPTQ** residual quantization (`use_gptq=True`) and **per-layer alpha search** (`search_alpha=True`).
- A **calibration / grid-search helper**, `examples/torch/diffusers/svdquant_calibrate.py`, sweeps the smoothing alpha, GPTQ on/off, and the number of calibration samples, scoring each configuration by reference-image quality (PSNR / MSE, plus `lpips` when installed) or quantized-submodule MSE, and reports the best.
- On FLUX.1-dev, SVDQuant in W4A4 / MXFP4 / NVFP4 nearly matches the FP16 CLIP score.
- **Native inference**: SVDQuant-MXFP4 models can run with real low-bit `aiter` GEMM kernels (ROCm) via `quark.torch.enable_native_inference` — the MXFP4 residual GEMM plus the low-rank correction branch — replacing the emulation/QDQ path for faster, lower-memory inference.

#### Agent Skills

Added a Claude Code skill suite for the PyTorch flow, auto-discovered from `.claude/skills/` and routed by file type (HuggingFace / safetensors / PyTorch checkpoints → Torch skills, never silently mixed with the ONNX flow):

- `quark-torch-ptq` — end-to-end PTQ pipeline for HF / safetensors models (FP8, INT4, MXFP4, etc.), stopping at the quantized output.
- `quark-torch-llm-ptq-eval` — PTQ plus validation and perplexity evaluation in one flow.
- `quark-torch-file2file-quantization` — file-to-file quantization for ultra-large models.
- `quark-torch-model-intake` — inspects a model and assesses quantization support.
- `quark-torch-export` — exports quantized models (e.g. GGUF, ONNX).
- `quark-torch-install` / `quark-torch-debug` — set up the matching PyTorch build and diagnose failed PTQ runs.

The backend-neutral `quark-env-preflight` and `quark-install` skills apply to both the Torch and ONNX flows.

#### vLLM Online Quantization

Added `quark.online_quantization.vllm` (tested against vLLM 0.21), bringing Quark's powerful online quantization flow into vLLM at load time. It extends vLLM's built-in online quantization and is designed to map Quark online quantization configs rather than being limited to a fixed list of schemes. The current release supports three schemas: per-channel FP8 (`ptpc_fp8`), MXFP4 (`mxfp4`), and a mixed linear-FP8/MoE-MXFP4 scheme (`linear_ptpc_fp8_moe_mxfp4`). It also supports re-quantizing offline-quantized checkpoints (e.g., DeepSeek-R1 FP8 block-scale) to a different online scheme at load time. Online versions of more Quark quantization algorithms are planned for future updates. See [the runnable example](../../examples/online_quantization/vllm_online_quantization.py).

### AMD Quark for ONNX

#### New Features

- Added additional quantize/dequantize node pairs at the mixed‑precision tensors to simulate node-wise quantization under QDQ mode.
- Added support for excluding specific nodes' outputs from quantization via setting a new extra option `NodesToExcludeOutputQuantization`.
- Refined the block axis of BFP and MX to make sure it's always on the reduction dimension in matrix multiplication operations.

#### Enhancements

Enhancements for calibration:

- Added Selective Calibration Propagation (SCP) via the `CalibPassthroughOpTypes` extra option. Distribution-preserving operators (e.g., `Reshape`, `Transpose`, `Gather`) are skipped during calibration, reducing calibration time and memory.
- Added `CalibOptimizeDisk` option for `LayerwisePercentile` calibration. When `True` (default), activation tensors are processed on-the-fly and never written to disk or held in memory, eliminating disk usage at the cost of slightly increased runtime.
- Reduced the peak memory of `MinMax` and `NonOverflow` calibration methods by disabling CPU memory arena (only available for the CPU Executive Provider).
- Reduced the peak memory of `Percentile`, `Entropy` and `Distribution` calibration methods by removing references of outputs in session run.
- Reduced the peak memory of `LayerwisePercentile` calibration method by processing activation tensors in chunks rather than accumulating them in full.
- Reduced the disk usage of `MinMSE` calibration method (`All` mode) by replacing raw data accumulation with a fixed-size per-tensor histogram built online as each inference batch arrives.

Enhancements for fast fine-tuning:

- Skipped layers with dynamic weights from fine-tuning because there is no object for tuning.
- Added support for inferring kernel size from weights if no attribute `kernel_shape` exists in the `Conv` nodes for fine-tuning.

Other important enhancements:

- Added support for ONNX Runtime 1.23.2, 1.24.2, and 1.25.1.
- Added support for converting float16 subgraphs to float32 with Cast nodes at the subgraph boundaries to make sure the converted model is runnable.
- Enabled the `ForceQuantizeNoInputCheck` option by default in built-in configurations to ensure op types like `Resize`, `Transpose` can be quantized even though their inputs are not quantized.
- Extended `SaveAndRestore` to cover FastFinetune checkpointing. Calibration ranges, the intermediate quantized model, and selected layer indices are persisted to a `.json` file and automatically reloaded on resume, avoiding redundant calibration passes.
- Added `FillAllValueInfo` option to post-processing. When set to `True`, missing `graph.value_info` entries are populated for all intermediate tensors (activations, weights, biases) using one ORT inference pass, which is required by many downstream compilers to infer tensor shapes. The default is `False`.
- Improved the parameter summary printed by `ModelQuantizer` on the `QConfig` path: it now shows the effective options actually used for quantization (grouped by category) and flags any user-set options that do not apply to the selected quantizer, making it easier to spot incorrect settings.
- Extended Cross-Layer Equalization (CLE) to support 5D `Conv` (`Conv3d`) weights.

#### Bug fixes

- Fixed a `TensorQuantOverrides` validation error that could prevent quantization (for example on the YOLO12 head) when pre-processing folded or fused away tensors referenced by the overrides.
- Fixed several Cross-Layer Equalization (CLE) edge cases on graphs with missing weight initializers or non-standard `Clip`/`Relu` patterns.

#### Agent Skills

Introduced a Claude Code skill suite for the ONNX-to-ONNX flow, auto-discovered from `.claude/skills/` and routed by file type (`.onnx` inputs → ONNX skills, never silently mixed with the Torch flow):

- `quark-onnx-install` — installs and verifies the correct `onnxruntime` / `onnxruntime-gpu` / ROCm build and matching `onnx` package for the user's accelerator before any quantization step.
- `quark-onnx-model-intake` — inspects a `.onnx` model (opset / IR version, I/O shapes and dtypes, op-type histogram, quantizable-op count, >2 GB external-data check) and assesses compatibility with CPU / CUDA / ROCm / AMD NPU CNN / AMD NPU Transformer targets.
- `quark-onnx-ptq` — end-to-end ONNX PTQ workflow for `.onnx` inputs (with optional sibling `.onnx_data`): intake, quantization planning, calibration-script generation, manifest, and confirmed execution for schemes such as XINT8, A8W8, BFP16, MXFP*, and weights-only INT4 for ONNX LLMs.
- `quark-onnx-autosearch-pro` — drives `quark.onnx.AutoSearchPro` (Optuna-based hyperparameter search) to find the best activation / weight spec, calibration method, CLE, AdaRound / AdaQuant, and FastFinetune params, including built-in `ADVANCED_SEARCH`, `XINT8_SEARCH`, `A8W8_SEARCH`, and `A16W8_SEARCH` presets.
- `quark-onnx-debug` — diagnoses failed installation, calibration, quantization, custom-op compilation, or export attempts, including ORT execution-provider mismatches, silent CPU fallback, calibration OOM, `BFPQuantizeDequantize` / `MXQuantizeDequantize` / `Extended*` custom-op load failures, >2 GB external-data issues, and AdaRound / GPTQ ONNX / QuaRot divergence.
- `quark-onnx-result-validator` — post-quantization inspection of `model.onnx` (and `model.onnx_data`) via four lightweight checks: auxiliary file copy alignment, expected non-quantized initializer MD5 byte-identity (inline `raw_data` and external-data byte ranges), model metadata equality after stripping quantization-only opset entries and Quark domains, and fuzzy node-pattern / op-type / dtype summaries with QDQ and `com.amd.quark` custom-op presence checks.

#### Deprecations and breaking changes

- Support for Python 3.10 is deprecated and was removed.
- Support for ONNX Runtime 1.20.1, 1.21.1 and 1.22.2 are deprecated and no longer tested.
- The `Percentile` mode of `MinMSE` calibration method is deprecated.
- The `LogSeverityLevel` extra option is removed. Use the `QUARK_LOG_LEVEL` environment variable instead.

## Release 0.11.1

### AMD Quark for PyTorch

#### Model Support

Supported out-of-box model architectures:

- Kimi-K2-Thinking, Kimi-K2-Instruct, Kimi-K2.5
- Qwen3 MoE, Qwen3 Coder, Qwen3 Coder-Next
- DeepSeek-V3.2, DeepSeek-OCR
- GLM-4.7
- Minimax-M2.1

#### New Features

- Added File-to-File quantization for ultra-large models. This mode supports **weight-only quantization** and **dynamic activation quantization + weight quantization**, exports **hf_format** only, and can also accept pre-quantized inputs (deepseek-style FP8, compressed-tensors) and re-quantize them to a different format.

  For example, the command below runs file-to-file quantization to MXFP4:

  ```bash
  python3 quantize_quark.py --model_dir [model checkpoint folder] \
                            --output_dir [output folder] \
                            --quant_scheme mxfp4 \
                            --file2file_quantization \
                            --skip_evaluation
  ```

- Added a pre-quantization compatibility check for `transformers` in LLM PTQ workflows, and enabled dry-run compatibility checking by default with clearer error messages when model loading fails.

#### Bug fixes and minor improvements

- Fixed weight calibration coverage to ensure complete calibration even for weights outside the forward path, and added token distribution coverage warnings during calibration.

### AMD Quark for ONNX

#### New Features

- Support using a YAML file as input to perform custom preprocessing for float models before quantization.

#### Enhancements

- Memory optimization has been extended to all calibration methods, particularly further reducing memory usage during activation data collection.

#### Bug fixes and minor improvements

- Infer kernel size from weights if the attribute `kernel_shape` of Conv nodes are not presented explicitly during fast finetuning.
- Fix scale extraction to handle int32_data in ONNX initializers.
- Add optional config parameter to onnxslim optimization.
- Update requirements.txt for onnxslim from 0.1.77 to 0.1.84.
- Handle NaN and Inf values in model inference output.

## Release 0.11

### AMD Quark for PyTorch

AMD Quark 0.11 is tested against PyTorch 2.9, and compatible with upstream `transformers==4.57`.

#### Fused "rotation" and "quarot" algorithms in a single interface

The pre-quantization algorithms "rotation" and "quarot" are fused together into a single rotation algorithm. It can be configured using `RotationConfig`. By default, only `R1` rotation is applied, corresponding to the previous `quant_algo="rotation"` behavior.

#### Quark Torch Quantization Config Refactor

- The quantization configuration classes have been renamed for better clarity and consistency:

  - `QuantizationSpec` is deprecated in favor of `QTensorConfig`.
  - `QuantizationConfig` is deprecated in favor of `QLayerConfig`.
  - `Config` is deprecated in favor of `QConfig`.

- The deprecated class names (`QuantizationSpec`, `QuantizationConfig`, `Config`) are still available as aliases for backward compatibility, but will be removed in a future release.

- Before Refactor:

  ```python
  from quark.torch.quantization.config.config import Config, QuantizationConfig, QuantizationSpec

  quant_spec = QuantizationSpec(dtype=Dtype.int8, ...)
  quant_config = QuantizationConfig(weight=quant_spec, ...)
  config = Config(global_quant_config=quant_config, ...)
  ```

- After Refactor:

  ```python
  from quark.torch.quantization.config.config import QConfig, QLayerConfig, QTensorConfig

  quant_spec = QTensorConfig(dtype=Dtype.int8, ...)
  quant_config = QLayerConfig(weight=quant_spec, ...)
  config = QConfig(global_quant_config=quant_config, ...)
  ```

#### `quark torch-llm-ptq` CLI Refactor and Simplification

The CLI has been significantly refactored to use the new `LLMTemplate` interface and remove redundant features:

- Removed model-specific algorithm configuration files (e.g., `awq_config.json`, `gptq_config.json`, `smooth_config.json`). Algorithm configurations are now automatically handled by `LLMTemplate`.
- Removed unnecessary CLI arguments, retaining only a dozen or so essential arguments.
- Simplified export: The CLI now only exports to Hugging Face safetensors format.
- Simplified evaluation: Evaluation now uses perplexity (PPL) on wikitext-2 dataset instead of the previous multi-task evaluation framework.

#### Code Organization and Examples Refactor

Moved common utilities to `quark.torch.utils`:

- `model_preparation.py` and `data_preparation.py` are now available in `quark.torch.utils` for easier reuse across examples and applications.
- `module_replacement` utilities are now located in `quark.torch.utils.module_replacement`.

Moved LLM evaluation code to `quark.contrib`:

- The `llm_eval` module has been moved to `quark.contrib.llm_eval` and `examples/contrib/llm_eval`.
- Perplexity evaluation (`ppl_eval`) is now shared between CLI and examples via `quark.contrib.llm_eval`.

Reorganized example scripts:

- Removed model-specific algorithm configuration files (e.g., `awq_config.json`, `gptq_config.json`, `smooth_config.json`). Algorithm configurations are now automatically handled by `LLMTemplate`.

Extended `quantize_quark.py` example script and `quark torch-llm-ptq` CLI with new features:

- Support for custom model templates and quantization schemes registration (example script only).
- Support for per-layer quantization scheme configuration via `--layer_quant_scheme` argument.
- Support for custom algorithm configurations via `--quant_algo_config_file` argument (example script only).
- Simplified quantization scheme naming, directly use the built-in scheme names (see breaking changes below).

#### Setting log level with `QUARK_LOG_LEVEL`

Logging level can now be set with the environment variable `QUARK_LOG_LEVEL`, e.g. `QUARK_LOG_LEVEL=debug` or `QUARK_LOG_LEVEL=warning` or `QUARK_LOG_LEVEL=error` or `QUARK_LOG_LEVEL=critical`.

#### Support for online rotations (online hadamard transform)

The rotation algorithm supports online rotations, such that:

$$y = xRR^TW$$

where $x$ is the input activation, $W$ the weight, and $R$ an orthogonal matrix (e.g. hadamard transform). With the quantization operator $\mathcal{Q}$ added, this becomes $\mathcal{Q}(xR) \times \mathcal{Q}(WR)^T$. The activation quantization $\mathcal{Q}(xR)$ is done **online**, that is the rotation is applied during inference and is not fused in a preceding layer.

Online rotations can be enabled using `online_r1_rotation=True` in `RotationConfig`. Please refer to its documentation and to [the user guide](https://quark.docs.amd.com/latest/pytorch/tutorial_rotation.html) for more details.

#### Support for rotation / SmoothQuant scales fine-tuning (SpinQuant/OSTQuant)

We support fine-tuning joint rotations and smoothing scales as a non-destructive transformation $O = DR$, where $R$ is an orthogonal matrix and $D$ is a diagonal matrix (SmoothQuant scales), such that:

$$y = xOO^{-1}W$$
$$= xDRR^TD^{-1}W^T$$
$$= xDR \times (WD^{-1}R)^T$$
$$= ... x'R \times (WD^{-1}R)^T$$

The support is well tested for `llama`, `qwen3`, `qwen3_moe` and `gpt_oss` architectures.

Rotation fine-tuning and online rotations are compatible with other algorithms as GPTQ or Qronos.

Please refer to the documentation of `RotationConfig`, [the example](https://github.com/amd/Quark/tree/release/0.11/examples/torch/language_modeling/rotation) and [the user guide](https://quark.docs.amd.com/latest/pytorch/tutorial_rotation.html) for more details.

#### Minor changes and bug fixes

- Fix memory duplication and OOM issues when loading `gpt_oss` models for quantization.
- `ModelQuantizer.freeze` behavior is changed to permanently quantize weights. Weights are still in high precision, but QDQ (quantize + dequantize) is run on them. This allows to avoid to rerun QDQ on static weights at each subsequent call.
- `scaled_fake_quantize` operator, which is used for QDQ, is now by default compiled with `torch.compile`, allowing significant speedups depending on the quantization scheme (1x - 8x).
- An efficient MXFP4 dynamic quantization kernel is used for activations when quantizing models, fusing scale computation and QDQ operations.
- Batching support is fixed in `lm-evaluation-harness` integration in the examples, correctly passing the user-provided `--eval_batch_size`.
- CPU/GPU communication is removed in quantization observers, allowing for faster quantization and runtime during e.g. the evaluation of models.

#### Deprecations and breaking changes

- Quantization scheme names in `examples/torch/language_modeling/llm_ptq/quantize_quark.py` and `quark torch-llm-ptq` CLI have been simplified and renamed:

  - `w_int4_per_group_sym` is deprecated in favor of `int4_wo_32`, `int4_wo_64`, `int4_wo_128` (depending on group size).
  - `w_uint4_per_group_asym` is deprecated in favor of `uint4_wo_32`, `uint4_wo_64`, `uint4_wo_128` (depending on group size).
  - `w_int8_a_int8_per_tensor_sym` is deprecated in favor of `int8`.
  - `w_fp8_a_fp8` is deprecated in favor of `fp8`.
  - `w_mxfp4_a_mxfp4` is deprecated in favor of `mxfp4`.
  - `w_mxfp4_a_fp8` is deprecated in favor of `mxfp4_fp8`.
  - `w_mxfp6_e3m2_a_mxfp6_e3m2` is deprecated in favor of `mxfp6_e3m2`.
  - `w_mxfp6_e2m3_a_mxfp6_e2m3` is deprecated in favor of `mxfp6_e2m3`.
  - `w_bfp16_a_bfp16` is deprecated in favor of `bfp16`.
  - `w_mx6_a_mx6` is deprecated in favor of `mx6`.

- The `--group_size` and `--group_size_per_layer` arguments in `examples/torch/language_modeling/llm_ptq/quantize_quark.py` and `quark torch-llm-ptq` CLI have been removed. Group size is now embedded in the scheme name (e.g., `int4_wo_32`, `int4_wo_64`, `int4_wo_128`).

- The `--layer_quant_scheme` argument format in `examples/torch/language_modeling/llm_ptq/quantize_quark.py` and `quark torch-llm-ptq` CLI has changed to repeated arguments with pattern and scheme pairs (e.g., `--layer_quant_scheme lm_head int8 --layer_quant_scheme '*down_proj' fp8`).

- The token counter used count the number of tokens seen by each expert during calibration is now disabled by default, and requires the environment variable `QUARK_COUNT_OBSERVED_SAMPLES=1`.

- The export format `"quark_format"` is removed, following deprecation in AMD Quark 0.10. Additionally, `quark.torch.export.api.ModelExporter` and `quark.torch.export.api.ModelImporter` are removed, please refer to the [0.10 release notes](https://quark.docs.amd.com/latest/release_notes.html#release-0-10) and to [the documentation](https://quark.docs.amd.com/latest/pytorch/export/quark_export.html) for the current API.

### AMD Quark for ONNX

#### New Features

- Auto Search Pro

  - Hierarchical Search: Support for conditional and nested hyperparameter trees for advanced search strategies.
  - Custom Objectives: Support custom evaluation logic that perfectly aligns with specific needs.
  - Sampler Flexibility: Various samplers ('TPE', 'Grid Search', etc) are available .
  - Parallel search: Take advantage of parallelization to run multiple searches simultaneously, reducing time to solution.
  - Checkpoint: Resume interrupted hyperparameter optimization from the last checkpoint.
  - Visualization: View real-time visualizations that show your optimization performance and feature importance, making it easier to interpret results.
  - Output Saving: Automatically save the best configuration, study database, and generated plots for your analysis.

- Latency and memory usage profiling

  - Latency Profiling: Each quantization stage performs specific operations that contribute to the overall quantization pipeline, and their individual latency are reported in the profiling results.
  - Memory profiling

    - CPU Memory Profiling: By wrapping the Python script with mprof, we can record detailed memory traces during execution.
    - ROCM GPU Memory Profiling: For workflows involving ROCMExecutionProvider or any GPU-based quantization step, Quark ONNX offers a lightweight tool to monitor ROCm GPU memory usage in real time.

- ONNX Adapter: It is a graph transformation tool that can perform graph transformation of preprocessing like constant folding, operator fusion, removal of redundant nodes, streamlining input and output nodes, and optimizing the graph structure.

  - Support 20 preprocessing features

    - Convert BatchNormalization operations to Conv operations.
    - Convert Clip operations to Relu operations.
    - Convert models from FP16 to FP32.
    - Convert models from NCHW to NHWC.
    - Convert opset version of models.
    - Convert ReduceMean operations to GlobalAveragePool operations.
    - Convert Split operations to Slice operations.
    - Duplicate initializers for shared Bias.
    - Duplicate initializers for shared ones.
    - Fix shapes for models with dynamic shapes.
    - Fold BatchNormalization operations.
    - Fold BatchNormalization operations after Concat operations.
    - Fuse Gelu operations.
    - Fuse InstanceNormalization operations.
    - Fuse LpNormalization operations.
    - Fuse LayerNormalization operations.
    - Optimize models with ONNXRuntime.
    - Remove initializers from model inputs.
    - Simplify models with OnnxSlim.
    - Split GlobalAveragePool operations.

#### Enhancements

- Support Python 3.12 for Quark ONNX and remove dependency on CMake < 4.0.

- Enhance tensor-wise mixed precision for integer quantization data types

  - Enable the option `TensorQuantOverrides` to replace original `MixedPrecisionTensor`.
  - Add support for setting per-tensor or per-channel quantization.
  - Add support for setting symmetric or asymmetric quantization.
  - Add support for setting more parameters, such as scale, zero_point and etc.
  - Prioritize the mixed precision setting when there are multiple settings on the same tensor.

- Refactor the codebase to make the quantizer easier to maintain and more reliable in operation

- Replace ONNX Simplifier with OnnxSlim in preprocessing process before quantization.

- Allow specific inputs or outputs to be converted from NCHW to NHWC.

- Refactor the import paths

  - Before refactor:

    ```python
    from quark.onnx import ModelQuantizer
    from quark.onnx.quantization import QConfig
    from quark.onnx.quantization.config.spec import QLayerConfig, Int8Spec
    from quark.onnx.quantization.config.algorithm import CLEConfig, AdaRoundConfig

    quantization_config = QConfig(
        # This is a global quantization configuration example using Int8 for activation, weight and bias. If the quantization for the bias is not specified, it will automatically follow the same quantization as the weights.
        global_config=QLayerConfig(activation=Int8Spec(), weight=Int8Spec()),
        # For example, quantize the activation, weight, and bias of the two specified nodes using Int16.
        specific_layer_config={Int16: ["/layer.0/Conv_0", "/layer.11/Conv_2"]},
        # For example, quantize the activation, weight, and bias of the all MatMul nodes using Int16 and exclude all Gemm nodes to quantize.
        layer_type_config={Int16: ["MatMul"], None: ["Gemm"]},
    )
    ```

  - After refactor:

    ```python
    # All configurations are now imported uniformly from quark.onnx
    from quark.onnx import ModelQuantizer, QConfig, QLayerConfig, Int8Spec, CLEConfig, AdaRoundConfig

    quantization_config = QConfig(
        # Rename activation to input_tensors in QLayerConfig
        global_config=QLayerConfig(input_tensors=Int8Spec(), weight=Int8Spec()),
        # Compared to before, it is now to specify the quantization for each tensor of a node.
        # For example, keep the input_tensors as Int8, quantize the weight and bias using Int16 for two specified nodes.
        specific_layer_config={QLayerConfig(weight=Int16Spec(), bias=Int16Spec()): ["/layer.0/Conv_0", "/layer.11/Conv_2"]},
        # Compared to before, it is now to specify the quantization for each tensor of all nodes of specific operation types.
        # For example, keep the input_tensors and bias as Int8, only quantize the weight using Int16 for all MatMul nodes and exclude all Gemm nodes to quantize.
        layer_type_config={QLayerConfig(weight=Int16Spec()): ["MatMul"], None: ["Gemm"]},
    )
    ```

- Reduce the memory consumption of the default mode of MinMSE to prevent OOM

- Significantly speedup the calibration process using parallel computation

- Fixed seed for Fast Finetune

#### Documentation

- Removed ONNXRuntime dependency from Quark for simplified environment setup.

#### Bug fixes and minor improvements

- Fixed percentile value selection for LayerwisePercentile

- Fixed the out-of-bounds axis issue when weight or bias is a scalar in BFP and MX quantization

- Fixed bug for replacing clip with ReLU operator.

## Release 0.10

- **AMD Quark for PyTorch**

  - New Features

    - Support PyTorch 2.7.1.
    - Support for int3 quantization and exporting of models.
    - Support the AWQ algorithm with Gemma3 and Phi4.
    - Support Qronos advanced quantization algorithm.
    - Applying the [GPTQ algorithm](https://quark.docs.amd.com/latest/pytorch/quark_torch_best_practices.html#apply-quantization-algorithms) runs x3-x4 faster compared to AMD Quark 0.9, using [CUDA/HIP Graph](https://docs.pytorch.org/docs/stable/notes/cuda.html#cuda-graph-semantics) by default. If requirement, CUDA Graph for GPTQ can be disabled using the environment variable `QUARK_GRAPH_DEBUG=0`.
    - [Quarot](https://quark.docs.amd.com/latest/pytorch/tutorial_quarot.html) algorithm supports a new configuration parameter `rotation_size` to define custom hadamard rotation sizes. Please refer to [QuaRotConfig documentation](https://quark.docs.amd.com/latest/autoapi/quark/torch/quantization/config/config/index.html#quark.torch.quantization.config.config.QuaRotConfig).
    - Support the Qronos post-training quantization algorithm. Please refer to the [arXiv paper](https://arxiv.org/abs/2505.11695) and [Quark documentation](https://quark.docs.amd.com/latest/autoapi/quark/torch/quantization/config/config/index.html#quark.torch.quantization.config.config.QronosConfig).

  - QuantizationSpec check:

    - Every time user finishes init `QuantizationSpec` will automatically perform config check. If any invalid config is supplied, a warning or error message will be given to user for better correction. In this way, find potential error as early as possible rather than cause a runtime error during quantization process.

  - LLM Depth-Wise Pruning tool:

    - Depth-wise pruning tool that can decrease the LLM model size. This tool deletes the consecutive decode layers in LLM under a certain supplied pruning ratio.
    - Based on PPL influence, the consecutive layers that have less influence on PPL will be regarded as having less influence on LLM and can be deleted.

  - Model Support:

    - Support OCP MXFP4, MXFP6, MXFP8 quantization of new models: DeepSeek-R1, Llama4-Scout, Llama4-Maverick, gpt-oss-20b, gpt-oss-120b.

  - Deprecations and breaking changes

    - OCP MXFP6 weight packing layout is modified to fit the expected layout by [CDNA4](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) `mfma_scale` instruction.

    - In the `examples/language_modeling/llm_ptq/quantize_quark.py` example, the quantization scheme `"w_mxfp4_a_mxfp6"` is removed and replaced by `"w_mxfp4_a_mxfp6_e2m3"` and `"w_mxfp4_a_mxfp6_e3m2"`.

  - Important bug fixes

    - A bug in [Quarot](https://quark.docs.amd.com/latest/pytorch/tutorial_quarot.html) and [Rotation](https://quark.docs.amd.com/latest/pytorch/tutorial_rotation.html) algorithms where fused rotations were wrongly applied twice on input embeddings / LM head weights is fixed.

    - Reduce the slowness of the reloading of large quantized models as DeepSeek-R1 using Transformers + Quark.

- **AMD Quark for ONNX**

  - New Features:

    - API Refactor (Introduced the new API design with improved consistency and usability)

      - Supported class-based algorithm usage.
      - Aligned data type both for Quark Torch and Quark ONNX.
      - Refactored quantization configs.

    - Auto Search Enhancements

      - Two-Stage Search: First identifies the best calibration config, then searches for the optimal FastFinetune config based on it. Expands the search space for higher efficiency.
      - Advanced-Fastft Search: Supports continuous search spaces, advanced algorithms (e.g., TPE), and parallel execution for faster, smarter searching.
      - Joint-Parameter Search: Combines coupled parameters into a unified space to avoid ineffective configurations and improve search quality.

    - Added support for ONNX 1.19
    - Added support for ONNXRuntime 1.22.2
    - Added optimized weight-scale calculation with the MinMSE method to improve quantization accuracy.
    - Accelerated calibration with multi-process support, covering algorithms such as MinMSE, Percentile, Entropy, Distribution, and LayerwisePercentile.
    - Added progress bars for Percentile, Entropy, Distribution, and LayerwisePercentile algorithms.
    - Supported users to specify a directory for saving cache files.

  - Enhancements:

    - Significantly reduced memory usage across various configurations, including calibration and FastFinetune stages, with optimizations for both CPU and GPU memory.
    - Improved clarity of error and warning outputs, helping users select better parameters based on memory and disk conditions.

  - Bug fixes and minor improvements:

    - Provided actionable hints when OOM or insufficient disk space issues occur in calibration and fast fine-tuning.
    - Fixed multi-GPU issues during FastFinetune.
    - Fixed a bug related to converting BatchNorm to Conv.
    - Fixed a bug in BF16 conversion on models larger than 2GB.

- **Quark Torch API Refactor**

  - LLMTemplate for simplified quantization configuration:

    - Introduced `LLMTemplate` class for convenient LLM quantization configuration
    - Built-in templates for popular LLM architectures (Llama4, Qwen, Mistral, Phi, DeepSeek, GPT-OSS, etc.)
    - Support for multiple quantization schemes: int4/uint4 (group sizes 32, 64, 128), int8, fp8, mxfp4, mxfp6e2m3, mxfp6e3m2, bfp16, mx6
    - Advanced features: layer-wise quantization, KV cache quantization, attention quantization
    - Algorithm support: AWQ, GPTQ, SmoothQuant, AutoSmoothQuant, Rotation
    - Custom template and scheme registration capabilities for users to define their own template and quantization schemes

      ```python
      from quark.torch import LLMTemplate

      # List available templates
      templates = LLMTemplate.list_available()
      print(templates)  # ['llama', 'opt', 'qwen', 'mistral', ...]

      # Get a specific template
      llama_template = LLMTemplate.get("llama")

      # Create a basic configuration
      config = llama_template.get_config(scheme="fp8", kv_cache_scheme="fp8")
      ```

  - Export and import APIs are deprecated in favor of new ones:

    - `ModelExporter.export_safetensors_model` is deprecated in favor of `export_safetensors`:

      Before:

      ```python
      from quark.torch import ModelExporter
      from quark.torch.export.config.config import ExporterConfig, JsonExporterConfig

      export_config = ExporterConfig(json_export_config=JsonExporterConfig())
      exporter = ModelExporter(config=export_config, export_dir=export_dir)
      exporter.export_safetensors_model(model, quant_config)
      ```

      After:

      ```python
      from quark.torch import export_safetensors
      export_safetensors(model, output_dir=export_dir)
      ```

    - `ModelImporter.import_model_info` is deprecated in favor of `import_model_from_safetensors`:

      Before:

      ```python
      from quark.torch.export.api import ModelImporter

      model_importer = ModelImporter(
         model_info_dir=export_dir,
         saved_format="safetensors"
      )
      quantized_model = model_importer.import_model_info(original_model)
      ```

      After:

      ```python
      from quark.torch import import_model_from_safetensors
      quantized_model = import_model_from_safetensors(
         original_model,
         model_dir=export_dir
      )
      ```

- **Quark ONNX API Refactor**

  - Before:

    - Basic Usage:

      ```python
      from quark.onnx import ModelQuantizer
      from quark.onnx.quantization.config.config import Config
      from quark.onnx.quantization.config.custom_config import get_default_config

      input_model_path = "demo.onnx"
      quantized_model_path = "demo_quantized.onnx"
      calib_data_path = "calib_data"
      calib_data_reader = ImageDataReader(calib_data_path)

      a8w8_config = get_default_config("A8W8")
      quantization_config = Config(global_quant_config=a8w8_config )
      quantizer = ModelQuantizer(quantization_config)
      quantizer.quantize_model(input_model_path, quantized_model_path, calib_data_reader)
      ```

    - Advanced Usage:

      ```python
      from quark.onnx import ModelQuantizer
      from quark.onnx.quantization.config.config import Config, QuantizationConfig
      from onnxruntime.quantization.calibrate import CalibrationMethod
      from onnxruntime.quantization.quant_utils import QuantFormat, QuantType, ExtendedQuantType

      input_model_path = "demo.onnx"
      quantized_model_path = "demo_quantized.onnx"
      calib_data_path = "calib_data"
      calib_data_reader = ImageDataReader(calib_data_path)

      DEFAULT_ADAROUND_PARAMS = {
          "DataSize": 1000,
          "FixedSeed": 1705472343,
          "BatchSize": 2,
          "NumIterations": 1000,
          "LearningRate": 0.1,
          "OptimAlgorithm": "adaround",
          "OptimDevice": "cpu",
          "InferDevice": "cpu",
          "EarlyStop": True,
      }

      quant_config = QuantizationConfig(
          calibrate_method=CalibrationMethod.Percentile,
          quant_format=QuantFormat.QDQ,
          activation_type=QuantType.QInt8,
          weight_type=QuantType.QInt8,
          nodes_to_exclude=["/layer.2/Conv_1", "^/Conv/.*"],
          subgraphs_to_exclude=[(["start_node_1", "start_node_2"], ["end_node_1", "end_node_2"])],
          include_cle=True,
          include_fast_ft=True,
          specific_tensor_precision=True,
          use_external_data_format=False,
          extra_options={
              "MixedPrecisionTensor": {ExtendedQuantType.QInt16: ["/layer.0/Conv_0", "/layer.11/Conv_2"]},
              "CLESteps": 2,
              "FastFinetune": DEFAULT_ADAROUND_PARAMS
          }
      )

      quantization_config = Config(global_quant_config=quant_config)
      quantizer = ModelQuantizer(quantization_config)
      quantizer.quantize_model(input_model_path, quantized_model_path, calib_data_reader)
      ```

  - After:

    - Basic Usage:

      ```python
      from quark.onnx import ModelQuantizer
      from quark.onnx.quantization import QConfig

      input_model_path = "demo.onnx"
      quantized_model_path = "demo_quantized.onnx"
      calib_data_path = "calib_data"
      calib_data_reader = ImageDataReader(calib_data_path)

      quantization_config = QConfig.get_default_config("A8W8")
      quantizer = ModelQuantizer(quantization_config)
      quantizer.quantize_model(input_model_path, quantized_model_path, calib_data_reader)
      ```

    - Advanced Usage:

      ```python
      from quark.onnx import ModelQuantizer
      from quark.onnx.quantization import QConfig
      from quark.onnx.quantization.config.spec import QLayerConfig, Int8Spec
      from quark.onnx.quantization.config.data_type import Int16
      from quark.onnx.quantization.config.algorithm import CLEConfig, AdaRoundConfig

      input_model_path = "demo.onnx"
      quantized_model_path = "demo_quantized.onnx"
      calib_data_path = "calib_data"
      calib_data_reader = ImageDataReader(calib_data_path)

      int8_config = QLayerConfig(activation=Int8Spec, weight=Int8Spec)
      cle_algo = CLEConfig(cle_steps=2)
      adaround_algo = AdaRoundConfig(learning_rate=0.1, num_iterations=1000)

      quantization_config = QConfig(
          global_config=int8_config,
          specific_layer_config={Int16: ["/layer.0/Conv_0", "/layer.11/Conv_2"]},
          layer_type_config={Int16: ["MatMul"], None: ["Gemm"]},
          exclude=["/layer.2/Conv_1", "^/Conv/.*", (["start_node_1", "start_node_2"], ["end_node_1", "end_node_2"])],
          algo_config=[cle_algo, adaround_algo],
          use_external_data_format=False,
          **kwargs
      )
      quantizer = ModelQuantizer(quantization_config)
      quantizer.quantize_model(input_model_path, quantized_model_path, calib_data_reader)
      ```

## Release 0.9

- **AMD Quark for PyTorch**

  - New Features

    - OCP MXFP4 fake quantization and dequantization kernels

      - Efficient kernels are added to Quark's `torch/kernel/hw_emulation/csrc` for OCP MXFP4 quantization and dequantization. They are useful to simulate OCP MXFP4 workload on hardware that does not support natively this data type (e.g. MI300X GPUs).

  - Quantized models can be reloaded with no memory overhead

    - The method `ModelImporter.import_model_info` used to reload a quantized model checkpoint now supports using a non-quantized backbone placed on `torch.device("meta")` ([see PyTorch reference](https://docs.pytorch.org/docs/stable/meta.html)) device, avoiding the memory overhead of instantiating the non-quantized model on device. More details are available in the [Loading Quantized Models documentation](https://quark.docs.amd.com/latest/pytorch/export/quark_export_hf.html#loading-quantized-models-saved-in-hugging-face-format-safetensors-format).

      ```python
      from quark.torch.export.api import ModelImporter
      from transformers import AutoConfig, AutoModelForCausalLM
      import torch

      model_importer = ModelImporter(
         model_info_dir="./opt-125m-quantized",
         saved_format="safetensors"
      )

      # We only need the backbone/architecture of the original model,
      # not its weights, as weights are loaded from the quantized checkpoint.
      config = AutoConfig.from_pretrained("facebook/opt-125m")
      with torch.device("meta"):
         original_model = AutoModelForCausalLM.from_config(config)

      quantized_model = model_importer.import_model_info(original_model)
      ```

  - Deprecations and breaking changes

    - Some quantization schemes in AMD Quark LLM PTQ example are deprecated ([see torch LLM PTQ reference](https://quark.docs.amd.com/latest/pytorch/example_quark_torch_llm_ptq.html)):

      - `w_mx_fp4_a_mx_fp4_sym` is deprecated in favor of: `w_mxfp4_a_mxfp4`,
      - `w_mx_fp6_e3m2_sym` in favor of `w_mxfp6_e3m2`,
      - `w_mx_fp6_e2m3_sym` in favor of `w_mxfp6_e2m3`,
      - `w_mx_int8_per_group_sym` in favor of `w_mxint8`,
      - `w_mxfp4_a_mxfp4_sym` in favor of `w_mxfp4_a_mxfp4`,
      - `w_mx_fp6_e2m3_a_mx_fp6_e2m3` in favor of `w_mxfp6_e2m3_a_mxfp6_e2m3`,
      - `w_mx_fp6_e3m2_a_mx_fp6_e3m2` in favor of `w_mxfp6_e3m2_a_mxfp6_e3m2`,
      - `w_mx_fp4_a_mx_fp6_sym` in favor of `w_mxfp4_a_mxfp6`,
      - `w_mx_fp8_a_mx_fp8` in favor of `w_mxfp8_a_mxfp8`.

  - Bug fixes and minor improvements

    - Fake quantization methods for FP4 and FP6 are made compatible with CUDA Graph.
    - A summary of replaced modules for quantization is displayed when calling `ModelQuantizer.quantize_model` for easier inspection.

  - Model Support:

    - Support Gemma2 in OGA flow.

  - Quantization and Export:

    - Support quantization and export of models in MXFP settings, e.g. MXFP4, MXFP6.
    - Support sequential quantization, e.g. W-A-MXFP4+Scale-FP8e4m3.
    - Support more models with FP8 attention: OPT, LLaMA, Phi, Mixtral.

  - Algorithms:

    - Support GPTQ for MXFP4 Quantization.
    - QAT Enhancements using huggingface Trainer.
    - Fix AWQ implementation for qkv-packed MHA model (e.g., microsoft/Phi-3-mini-4k-instruct) and raise warning to users if using incorrect or unknown AWQ configurations.

  - Performance:

    - Speedup model export.
    - Accelerate FP8 inference acceleration.
    - Tensor parallelism for evaluation of quantized model.
    - Multi-device quantization as well as export.

  - FX Graph quantization:

    - Improve efficiency of power-of-2 scale quantization for less memory and faster computation.
    - Support channel-wise power-of-2 quantization by using per-channel MSE/NON-overflow observer.
    - Support Conv's Bias for int32 power-of-2 quantization, where bias's scale = weight's scale * activation's scale.
    - Support export of INT16/INT32 quantization model to ONNX format and the corresponding ONNXRuntime.

- **AMD Quark for ONNX**

  - New Features:

    - Introduced an encrypted mode for scenarios demanding high model confidentiality.
    - Supported fixing the shape of all tensors.
    - Supported quantization with int16 bias.

  - Enhancements:

    - Supported compatibility with ONNX Runtime version 1.21.x and 1.22.0.
    - Reduced CPU/GPU memory usage to prevent OOM.
    - Improved auto search efficiency by utilizing a cached datareader.
    - Enhanced multi-platform support: now supports Windows (CPU/CUDA) and Linux (CPU/CUDA/ROCm).

  - Examples:

    - Provided quantization examples of TIMM models.

  - Documentation:

    - Added specifications for all custom operators.
    - Improved FAQ documentation.

  - Custom Operations:

    - Renamed custom operation types and updated their domain to the com.amd.quark:

      - BFPFixNeuron → BFPQuantizeDequantize.
      - MXFixNeuron → MXQuantizeDequantize.
      - VitisQuantFormat and VitisQuantType → ExtendedQuantFormat and ExtendedQuantType.

  - Bug fixes and minor improvements

    - Fixed the issue where extremely large or small values caused -inf/inf during scale calculation.

## Release 0.8.2

### New Features

#### AMD Quark for PyTorch

- Added support for ONNX Runtime 1.22.0

## Release 0.8.1

### Bug Fixes and Enhancements

#### AMD Quark for ONNX

- Fixed BFP Kernel compilation issue for GCC 13

## Release 0.8

- **AMD Quark for PyTorch**

  - Model Support:

    - Supported SD3.0 quantization with W-INT4, W-INT8-A-INT8, and W-FP8-A-FP8.
    - Supported FLUX.1 quantization with W-INT4, W-INT8-A-INT8, and W-FP8-A-FP8.
    - Supported DLRM embedding-bag UINT4 weight quantization.

  - Quantization Enhancement:

    - Supported fp8 attention quantization of Llama Family.
    - Integrated SmoothQuant algorithm for SDXL.
    - Enabled quantization for all SDXL components (UNet, VAE, text_encoder, text_encoder_2), supporting both W-INT8-A-INT8 and W-FP8-A-FP8 formats.

  - Model Export:

    - Exported diffusion models (SDXL, SDXL-Turbo and SD1.5) to ONNX format via optimum.

  - Model Evaluation:

    - Added Rouge and Meteor evaluation metrics for LLMs.
    - Supported evaluating ONNX models exported using torch.onnx.export for LLMs.
    - Supported offline evaluation mode (evaluation without generation) for LLMs.

- **AMD Quark for ONNX**

  - Model Support:

    - Provided more ONNX quantization examples of detection models such as yolov7/yolov8.

  - Data Types:

    - Supported Microexponents (MX) data types, including MX4, MX6 and MX9.
    - Enhanced BFloat16 with more implementation formats suitable for deployment.

  - ONNX Quantizer Enhancements:

    - Supported compatibility with ONNX Runtime version 1.20.0 and 1.20.1.
    - Supported quantization with excluding subgraphs.
    - Enhanced mixed precision to support quantizing a model with any two data types.

  - Documentation Enhancements:

    - Supported Best Practice for Quark ONNX.
    - Supported documentation of converting from FP32/FP16 to BF16.
    - Supported documentation of XINT8, A8W8 and A16W8 quantization.

  - Custom Operations:

    - Optimized the customized "QuantizeLinear" and "DequantizeLinear" to support running on GPU.

  - Advanced Quantization Algorithms:

    - Supported Quarot Rotation R1 algorithm.
    - Improved AdaQuant algorithm to support Microexponents and Microscaling data types.
    - Added auto-search algorithm to automatically find the optimal quantized model with the best accuracy within the search space.
    - Enhanced the LLM quantization by using EMA algorithm.

  - Model Evaluation:

    - Supported evaluation of L2/PSNR/VMAF/COS.

## Release 0.7

### New Features

#### PyTorch

- Added quantization error statistics collection tool.
- Added support for reloading quantized models using `load_state_dict`.
- Added support for W8A8 quantization for the Llama-3.1-8B-Instruct example.
- Added option of saving metrics to CSV in examples.
- Added support for HuggingFace integration.
- Added support for more models

  - Added support for Gemma2 quantization using the OGA flow.
  - Added support for Llama-3.2 with FP8 quantization (weight, activation and KV-Cache) for the vision and language components.
  - Added support for Stable Diffusion v1-5 and Stable Diffusion XL Base 1.0

#### ONNX

- Added a tool to replace BFloat16 QDQ with Cast op.
- Added support for rouge and meteor evaluation metrics.
- Added a feature to fuse Gelu ops into a single Gelu op.
- Added the HQQ algorithm for MatMulNBits.
- Added a tool to convert opset version.
- Added support for fast fine-tuning BF16 quantized models.
- Added U8U8_AAWA and some other built-in configurations.

### Bug Fixes and Enhancements

#### PyTorch

- Enhanced LLM examples to support layer group size customization.
- Decoupled model inference from LLM evaluation harness.
- Fixed OOM issues when quantizing the entire SDXL pipeline.
- Fixed LLM eval bugs caused by export and multi-gpu usage.
- Fixed QAT functionality.
- Addressed AWQ preparation issues.
- Fixed mismatching QDQ implementation compared to Torch.
- Enhanced readability and added docstring for graph quantization.
- Fixed config retrieval by name pattern.
- Supported more Torch versions for auto config rotation.
- Refactored dataloader of algorithms.
- Fixed accuracy issues with Qwen2-MOE.
- Fixed upscaling of scales during the export of quantized models.
- Added support for reloading per-layer quantization config.
- Fixed misleading code in `ModelQuantizer._do_calibration` for weight-only quantization.
- Implemented transpose scales for per-group quantization for int8/uint8.
- Implemented export and load for compressed models.
- Fixed auto config rotation compatibility for more PyTorch versions.
- Fixed bug in input of get_config in exporter.
- Fixed bug in input of the eval_model function.
- Refactored LLM PTQ examples.
- Fixed infer_pack_shape function.
- Documented smoothquant alpha and warned users about possible undesired values.
- Fixed slightly misleading code in `ModelQuantizer._do_calibration`.
- Aligned ONNX mean 2 GAP.

#### ONNX

- Refactored documentation for LLM evaluations.
- Fixed NaN issues caused by overflow for BF16 quantization.
- Fixed an issue where trying to fast fine-tune the MatMul layers without weights.
- Updated ONNX unit tests to use temporary paths.
- Removed generated model "sym_shape_infer_temp.onnx" on infer_shape failure.
- Fixed error in mixed-precision weights calculation.
- Fixed a bug when simplifying Llama2-7b without kv_cache.
- Fixed import path and add parent directory to system path in BFP quantize_model.py example.

## Release 0.6

- **AMD Quark for PyTorch**

  - Model Support:

    - Provided more examples of LLM PTQ, such as Llama3.2 and Llama3.2-Vision models (only quantizing the language part).
    - Provided examples of Phi and ChatGLM for LLM QAT.
    - Provided examples of LLM pruning for Qwen2.5, Llama, OPT, CohereForAI/c4ai-command models.
    - Provided an example of YOLO-NAS, a detection model PTQ/QAT, which can partially quantize the model using your configuration under FX mode.
    - Provided an example of SDXL v1.0 with weight INT8 activation INT8 under Eager Mode.
    - Supported more models for rotation, such as Qwen models under Eager Mode.

  - PyTorch Quantizer Enhancements:

    - Supported partially quantizing the model by your config under FX mode.
    - Supported quantization of `ConvTranspose2d` in Eager Mode and FX mode.
    - Advanced Quantization Algorithms: Improved rotation by auto-generating configurations.
    - Optimized Configuration with DataTypeSpec for ease of use.
    - Accelerated in-place replacement under Eager Mode.
    - Supported loading configuration from a file of algorithms and pre-optimizations under Eager Mode.

  - Evaluation:

    - Provided LLM evaluation method of quantized models on benchmark tasks: Open LLM Leaderboard and more such.

  - Export Capabilities:

    - Integrated the export configurations into the Quark format export content, standardizing the pack method for per-group quantization.

  - PyTorch Pruning:

    - Supported LLM pruning algorithm.

- **AMD Quark for ONNX**

  - Model Support:

    - Provided more ONNX quantization examples of LLM models such as Llama2.

  - Data Types:

    - Supported int4 and uint4 data types.
    - Supported Microscaling (MX) data types with `int8`, `fp8_e4m3fn`, `fp8_e5m2`, `fp6_e3m2`, `fp6_e2m3`, and `fp4 elements`.

  - ONNX Quantizer Enhancements:

    - Supported compatibility with ONNX Runtime version 1.19.
    - Supported MatMulNBits quantization for LLM models.
    - Supported fast fine-tuning on the MatMul operator.
    - Supported quantizing specified operators.
    - Supported quantization type alignment of element-wise operators.
    - Supported ONNX graph cleaning for Ryzen AI workflow.
    - Supported int32 bias quantization for Ryzen AI workflow.
    - Enhanced support for Windows systems and ROCm GPU.
    - Optimized the quantization of FP16 models to save memory.
    - Optimized the custom operator compilation process.
    - Optimized the default parameters for auto mixed precision.

  - Advanced Quantization Algorithms:

    - Supported GPTQ for both QDQ format and MatMulNBits format.

## Release 0.5.1

- **AMD Quark for PyTorch**

  - Export Modifications:

    - Ignore the configuration of preprocessing algorithms when exporting JSON-safetensors format
    - Remove sub-directory in the exporting path.

- **AMD Quark for ONNX**

  - ONNX Quantizer Enhancements:

    - Supported compatibility with onnxruntime version 1.19.

## Release 0.5.0

- **AMD Quark for PyTorch**

  - Model Support:

    - Provided more examples of LLM models quantization:

      - INT/OCP_FP8E4M3: Llama-3.1, gpt-j-6b, Qwen1.5-MoE-A2.7B, phi-2, Phi-3-mini, Phi-3.5-mini-instruct, Mistral-7B-v0.1
      - OCP_FP8E4M3: mistralai/Mixtral-8x7B-v0.1, hpcai-tech/grok-1, CohereForAI/c4ai-command-r-plus-08-2024, CohereForAI/c4ai-command-r-08-2024, CohereForAI/c4ai-command-r-plus, CohereForAI/c4ai-command-r-v01, databricks/dbrx-instruct, deepseek-ai/deepseek-moe-16b-chat

    - Provided more examples of diffusion model quantization:

      - Supported models: SDXL, SDXL-Turbo, SD1.5, Controlnet-Canny-SDXL, Controlnet-Depth-SDXL, Controlnet-Canny-SD1.5
      - Supported schemes: FP8, W8, W8A8 with and without SmoothQuant

  - PyTorch Quantizer Enhancements:

    - Supported more CNN models for graph mode quantization.

  - Data Types:

    - Supported BFP16, MXFP8_E5M2.
    - Supported MX6 and MX9. (experimental)

  - Advanced Quantization Algorithms:

    - Supported Rotation for Llama models.
    - Supported SmoothQuant and AWQ for models with GQA and MQA (for example, Llama-3-8B, QWen2-7B).
    - Provided scripts for generating AWQ configuration automatically.(experimental)
    - Supported trained quantization thresholds (TQT) and learned step size quantization (LSQ) for better QAT results. (experimental)

  - Export Capabilities:

    - Supported reloading function of JSON-safetensors export format.
    - Enhanced quantization configuration in JSON-safetensors export format.

- **AMD Quark for ONNX**

  - ONNX Quantizer Enhancements:

    - Supported compatibility with onnxruntime version 1.18.
    - Enhanced quantization support for LLM models.

  - Quantization Strategy:

    - Supported dynamic quantization.

  - Custom operations:

    - Optimized "BFPFixNeuron" to support running on GPU.

  - Advanced Quantization Algorithms:

    - Improved AdaQuant to support BFP data types.

## Release 0.2.0

- **AMD Quark for PyTorch**

  - **PyTorch Quantizer Enhancements**:

    - Post Training Quantization (PTQ) and Quantization-Aware Training (QAT) are now supported in FX graph mode.
    - Introduced quantization support of the following modules: torch.nn.Conv2d.

  - **Data Types**:

    - OCP Microscaling (MX) is supported. Valid element data types include INT8, FP8_E4M3, FP4, FP6_E3M2, and FP6_E2M3.

  - **Export Capabilities**:

    - Quantized models can now be exported in GGUF format. The exported GGUF model is runnable with llama.cpp. Only Llama2 is supported for now.
    - Introduced Quark's native JSON-safetensors export format, which is identical to AutoFP8 and AutoAWQ when used for FP8 and AWQ quantization.

  - **Model Support**:

    - Added support for SDXL model quantization in eager mode, including fp8 per-channel and per-tensor quantization.
    - Added support for PTQ and QAT of CNN models in graph mode, including architectures like ResNet.

  - **Integration with other toolkits**:

    - Provided the integrated example with APL (AMD Pytorch-light, internal project name), supporting the invocation of APL's INT-K, BFP16, and BRECQ.
    - Introduced the experimental Quark extension interface, enabling seamless integration of Brevitas for Stable Diffusion and Imagenet classification model quantization.

- **AMD Quark for ONNX**

  - **ONNX Quantizer Enhancements**:

    - Multiple optimization and refinement strategies for different deployment backends.
    - Supported automatic mixing precision to balance accuracy and performance.

  - **Quantization Strategy**:

    - Supported symmetric and asymmetric quantization.
    - Supported float scale, INT16 scale and power-of-two scale.
    - Supported static quantization and weight-only quantization.

  - **Quantization Granularity**:

    - Supported for per-tensor and per-channel granularity.

  - **Data Types**:

    - Multiple data types are supported, including INT32/UINT32, Float16, Bfloat16, INT16/UINT16, INT8/UINT8 and BFP.

  - **Calibration Methods**:

    - MinMax, Entropy and Percentile for float scale.
    - MinMax for INT16 scale.
    - NonOverflow and MinMSE for power-of-two scale.

  - **Custom operations**:

    - "BFPFixNeuron" which supports block floating-point data type. It can run on the CPU on Windows, and on both the CPU and GPU on Linux.
    - "VitisQuantizeLinear" and "VitisDequantizeLinear" which support INT32/UINT32, Float16, Bfloat16, INT16/UINT16 quantization.
    - "VitisInstanceNormalization" and "VitisLSTM" which have customized Bfloat16 kernels.
    - All custom operations support running on the CPU on both Linux and Windows.

  - **Advanced Quantization Algorithms**:

    - Supported CLE, BiasCorrection, AdaQuant, AdaRound and SmoothQuant.

  - **Operating System Support**:

    - Linux and Windows.

## Release 0.1.0

- **AMD Quark for PyTorch**

  - **Pytorch Quantizer Enhancements**:

    - Eager mode is supported.
    - Post Training Quantization (PTQ) is now available.
    - Automatic in-place replacement of nn.module operations.
    - Quantization of the following modules is supported: torch.nn.linear.
    - The customizable calibration process is introduced.

  - **Quantization Strategy**:

    - Symmetric and asymmetric quantization are supported.
    - Weight-only, dynamic, and static quantization modes are available.

  - **Quantization Granularity**:

    - Support for per-tensor, per-channel, and per-group granularity.

  - **Data Types**:

    - Multiple data types are supported, including float16, bfloat16, int4, uint4, int8, and fp8 (e4m3fn).

  - **Calibration Methods**:

    - MinMax, Percentile, and MSE calibration methods are now supported.

  - **Large Language Model Support**:

    - FP8 KV-cache quantization for large language models (LLMs).

  - **Advanced Quantization Algorithms**:

    - Support SmoothQuant, AWQ (uint4), and GPTQ (uint4) for LLMs. (Note: AWQ/GPTQ/SmoothQuant algorithms are currently limited to single GPU usage.)

  - **Export Capabilities**:

    - Export of Q/DQ quantized models to ONNX and vLLM-adopted JSON-safetensors format now supported.

  - **Operating System Support**:

    - Linux (supports ROCM and CUDA)
    - Windows (supports CPU only).
