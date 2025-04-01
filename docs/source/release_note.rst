Release Notes
==============

New Features (Version 0.8)
--------------------------

-  **AMD Quark for PyTorch**

   -  Model Support:

      -  Supported SD3.0 quantization with W-INT4, W-INT8-A-INT8, and W-FP8-A-FP8.
      -  Supported FLUX.1 quantization with W-INT4, W-INT8-A-INT8, and W-FP8-A-FP8.
      -  Supported DLRM embedding-bag UINT4 weight quantization.

   -  Quantization Enhancement:

      -  Supported fp8 attention quantization of Llama Family.
      -  Integrated SmoothQuant algorithm for SDXL.
      -  Enabled quantization for all SDXL components (UNet, VAE, text_encoder, text_encoder_2), supporting both W-INT8-A-INT8 and W-FP8-A-FP8 formats.

   -  Model Export:

      -  Exported diffusion models (SDXL, SDXL-Turbo and SD1.5) to ONNX format via optimum.

   -  Model Evaluation:

      -  Added Rouge and Meteor evaluation metrics for LLMs.
      -  Supported evaluating ONNX models exported using torch.onnx.export for LLMs.
      -  Supported offline evaluation mode (evaluation without generation) for LLMs.

-  **AMD Quark for ONNX**

   -  Model Support:

      -  Provided more ONNX quantization examples of detection models such as yolov7/yolov8.

   -  Data Types:

      -  Supported Microexponents (MX) data types, including MX4, MX6 and MX9.
      -  Enhanced BFloat16 with more implementation formats suitable for deployment.

   -  ONNX Quantizer Enhancements:

      -  Supported compatibility with ONNX Runtime version 1.20.0 and 1.20.1.
      -  Supported quantization with excluding subgraphs.
      -  Enhanced mixed precision to support quantizing a model with any two data types.

   -  Documentation Enhancements:

      -  Supported Best Practice for Quark ONNX.
      -  Supported documentation of converting from FP32/FP16 to BF16.
      -  Supported documentation of XINT8, A8W8 and A16W8 quantization.

   -  Custom Operations:

      -  Optimized the customized "QuantizeLinear" and “DequantizeLinear” to support running on GPU.

   -  Advanced Quantization Algorithms:

      -  Supported Quarot Rotation R1 algorithm.
      -  Improved AdaQuant algorithm to support Microexponents and Microscaling data types.
      -  Added auto-search algorithm to automatically find the optimal quantized model with the best accuracy within the search space.
      -  Enhanced the LLM quantization by using EMA algorithm.

   -  Model Evaluation:

      -  Supported evaluation of L2/PSNR/VMAF/COS.

Release 0.7
-----------

New Features
^^^^^^^^^^^^

**PyTorch**

* Added quantization error statistics collection tool.
* Added support for reloading quantized models using `load_state_dict`.
* Added support for W8A8 quantization for the Llama-3.1-8B-Instruct example.
* Added option of saving metrics to CSV in examples.
* Added support for HuggingFace integration.
* Added support for more models

    * Added support for Gemma2 quantization using the OGA flow.
    * Added support for Llama-3.2 with FP8 quantization (weight, activation and KV-Cache) for the vision and language components.
    * Added support for Stable Diffusion v1-5 and Stable Diffusion XL Base 1.0

**ONNX**

* Added a tool to replace BFloat16 QDQ with Cast op.
* Added support for rouge and meteor evaluation metrics.
* Added a feature to fuse Gelu ops into a single Gelu op.
* Added the HQQ algorithm for MatMulNBits.
* Added a tool to convert opset version.
* Added support for fast fine-tuning BF16 quantized models.
* Added U8U8_AAWA and some other built-in configurations.

Bug Fixes and Enhancements
^^^^^^^^^^^^^^^^^^^^^^^^^^

**PyTorch**

* Enhanced LLM examples to support layer group size customization.
* Decoupled model inference from LLM evaluation harness.
* Fixed OOM issues when quantizing the entire SDXL pipeline.
* Fixed LLM eval bugs caused by export and multi-gpu usage.
* Fixed QAT functionality.
* Addressed AWQ preparation issues.
* Fixed mismatching QDQ implementation compared to Torch.
* Enhanced readability and added docstring for graph quantization.
* Fixed config retrieval by name pattern.
* Supported more Torch versions for auto config rotation.
* Refactored dataloader of algorithms.
* Fixed accuracy issues with Qwen2-MOE.
* Fixed upscaling of scales during the export of quantized models.
* Added support for reloading per-layer quantization config.
* Fixed misleading code in `ModelQuantizer._do_calibration` for weight-only quantization.
* Implemented transpose scales for per-group quantization for int8/uint8.
* Implemented export and load for compressed models.
* Fixed auto config rotation compatibility for more PyTorch versions.
* Fixed bug in input of get_config in exporter.
* Fixed bug in input of the eval_model function.
* Refactored LLM PTQ examples.
* Fixed infer_pack_shape function.
* Documented smoothquant alpha and warned users about possible undesired values.
* Fixed slightly misleading code in `ModelQuantizer._do_calibration`.
* Aligned ONNX mean 2 GAP.

**ONNX**

* Refactored documentation for LLM evaluations.
* Fixed NaN issues caused by overflow for BF16 quantization.
* Fixed an issue where trying to fast fine-tune the MatMul layers without weights.
* Updated ONNX unit tests to use temporary paths.
* Removed generated model "sym_shape_infer_temp.onnx" on infer_shape failure.
* Fixed error in mixed-precision weights calculation.
* Fixed a bug when simplifying llama2-7b without kv_cache.
* Fixed import path and add parent directory to system path in BFP quantize_model.py example.

New Features (Version 0.6.0)
----------------------------

-  **AMD Quark for PyTorch**

   -  Model Support:

      -  Provided more examples of LLM PTQ, such as LLaMA3.2 and LLaMA3.2-Vision models (only quantizing the language part).
      -  Provided examples of Phi and ChatGLM for LLM QAT.
      -  Provided examples of LLM pruning for Qwen2.5, LLaMA, OPT, CohereForAI/c4ai-command models.
      -  Provided an example of YOLO-NAS, a detection model PTQ/QAT, which can partially quantize the model using your configuration under FX mode.
      -  Provided an example of SDXL v1.0 with weight INT8 activation INT8 under Eager Mode.
      -  Supported more models for rotation, such as Qwen models under Eager Mode.

   -  PyTorch Quantizer Enhancements:

      -  Supported partially quantizing the model by your config under FX mode.
      -  Supported quantization of ``ConvTranspose2d`` in Eager Mode and FX mode.
      -  Advanced Quantization Algorithms: Improved rotation by auto-generating configurations.
      -  Optimized Configuration with DataTypeSpec for ease of use.
      -  Accelerated in-place replacement under Eager Mode.
      -  Supported loading configuration from a file of algorithms and pre-optimizations under Eager Mode.

   -  Evaluation:

      -  Provided LLM evaluation method of quantized models on benchmark tasks: Open LLM Leaderboard and more such.

   -  Export Capabilities:

      -  Integrated the export configurations into the Quark format export content, standardizing the pack method for per-group quantization.

   -  PyTorch Pruning:

      -  Supported LLM pruning algorithm.

-  **AMD Quark for ONNX**

   -  Model Support:

      -  Provided more ONNX quantization examples of LLM models such as LLaMA2.

   -  Data Types:

      -  Supported int4 and uint4 data types.
      -  Supported Microscaling (MX) data types with ``int8``, ``fp8_e4m3fn``, ``fp8_e5m2``, ``fp6_e3m2``, ``fp6_e2m3``, and ``fp4 elements``.

   -  ONNX Quantizer Enhancements:

      -  Supported compatibility with ONNX Runtime version 1.19.
      -  Supported MatMulNBits quantization for LLM models.
      -  Supported fast fine-tuning on the MatMul operator.
      -  Supported quantizing specified operators.
      -  Supported quantization type alignment of element-wise operators.
      -  Supported ONNX graph cleaning for Ryzen AI workflow.
      -  Supported int32 bias quantization for Ryzen AI workflow.
      -  Enhanced support for Windows systems and ROCm GPU.
      -  Optimized the quantization of FP16 models to save memory.
      -  Optimized the custom operator compilation process.
      -  Optimized the default parameters for auto mixed precision.

   -  Advanced Quantization Algorithms:

      -  Supported GPTQ for both QDQ format and MatMulNBits format.

New Features (Version 0.5.1)
----------------------------

-  **AMD Quark for PyTorch**

   -  Export Modifications:

      -  Ignore the configuration of preprocessing algorithms when exporting JSON-safetensors format
      -  Remove sub-directory in the exporting path.

-  **AMD Quark for ONNX**

   -  ONNX Quantizer Enhancements:

      -  Supported compatibility with onnxruntime version 1.19.

New Features (Version 0.5.0)
----------------------------

-  **AMD Quark for PyTorch**

   -  Model Support:

      -  Provided more examples of LLM models quantization:

         -  INT/OCP_FP8E4M3: Llama-3.1, gpt-j-6b, Qwen1.5-MoE-A2.7B, phi-2, Phi-3-mini, Phi-3.5-mini-instruct, Mistral-7B-v0.1
         -  OCP_FP8E4M3: mistralai/Mixtral-8x7B-v0.1, hpcai-tech/grok-1, CohereForAI/c4ai-command-r-plus-08-2024, CohereForAI/c4ai-command-r-08-2024, CohereForAI/c4ai-command-r-plus, CohereForAI/c4ai-command-r-v01, databricks/dbrx-instruct, deepseek-ai/deepseek-moe-16b-chat

      -  Provided more examples of diffusion model quantization:

         -  Supported models: SDXL, SDXL-Turbo, SD1.5, Controlnet-Canny-SDXL, Controlnet-Depth-SDXL, Controlnet-Canny-SD1.5
         -  Supported schemes: FP8, W8, W8A8 with and without SmoothQuant

   -  PyTorch Quantizer Enhancements:

      -  Supported more CNN models for graph mode quantization.

   -  Data Types:

      -  Supported BFP16, MXFP8_E5M2.
      -  Supported MX6 and MX9. (experimental)

   -  Advanced Quantization Algorithms:

      -  Supported Rotation for Llama models.
      -  Supported SmoothQuant and AWQ for models with GQA and MQA (for example, LLaMA-3-8B, QWen2-7B).
      -  Provided scripts for generating AWQ configuration automatically.(experimental)
      -  Supported trained quantization thresholds (TQT) and learned step size quantization (LSQ) for better QAT results. (experimental)

   -  Export Capabilities:

      -  Supported reloading function of JSON-safetensors export format.
      -  Enhanced quantization configuration in JSON-safetensors export format.

-  **AMD Quark for ONNX**

   -  ONNX Quantizer Enhancements:

      -  Supported compatibility with onnxruntime version 1.18.
      -  Enhanced quantization support for LLM models.

   -  Quantization Strategy:

      -  Supported dynamic quantization.

   -  Custom operations:

      -  Optimized "BFPFixNeuron" to support running on GPU.

   -  Advanced Quantization Algorithms:

      -  Improved AdaQuant to support BFP data types.

New Features (Version 0.2.0)
----------------------------

-  **AMD Quark for PyTorch**

   -  **PyTorch Quantizer Enhancements**:

      -  Post Training Quantization (PTQ) and Quantization-Aware Training (QAT) are now supported in FX graph mode.
      -  Introduced quantization support of the following modules: torch.nn.Conv2d.

   -  **Data Types**:

      -  :doc:`OCP Microscaling (MX) is supported. Valid element data types include INT8, FP8_E4M3, FP4, FP6_E3M2, and FP6_E2M3. <./pytorch/adv_mx>`

   -  **Export Capabilities**:

      -  :doc:`Quantized models can now be exported in GGUF format. The exported GGUF model is runnable with llama.cpp. Only Llama2 is supported for now. <./pytorch/export/gguf_llamacpp>`
      -  Introduced Quark's native JSON-safetensors export format, which is identical to AutoFP8 and AutoAWQ when used for FP8 and AWQ quantization.

   -  **Model Support**:

      -  Added support for SDXL model quantization in eager mode, including fp8 per-channel and per-tensor quantization.
      -  Added support for PTQ and QAT of CNN models in graph mode, including architectures like ResNet.

   -  **Integration with other toolkits**:

      -  Provided the integrated example with APL (AMD Pytorch-light, internal project name), supporting the invocation of APL's INT-K, BFP16, and BRECQ.
      -  Introduced the experimental Quark extension interface, enabling seamless integration of Brevitas for Stable Diffusion and Imagenet classification model quantization.

-  **AMD Quark for ONNX**

   -  **ONNX Quantizer Enhancements**:

      -  Multiple optimization and refinement strategies for different deployment backends.
      -  Supported automatic mixing precision to balance accuracy and performance.

   -  **Quantization Strategy**:

      -  Supported symmetric and asymmetric quantization.
      -  Supported float scale, INT16 scale and power-of-two scale.
      -  Supported static quantization and weight-only quantization.

   -  **Quantization Granularity**:

      -  Supported for per-tensor and per-channel granularity.

   -  **Data Types**:

      -  Multiple data types are supported, including INT32/UINT32,
         Float16, Bfloat16, INT16/UINT16, INT8/UINT8 and BFP.

   -  **Calibration Methods**:

      -  MinMax, Entropy and Percentile for float scale.
      -  MinMax for INT16 scale.
      -  NonOverflow and MinMSE for power-of-two scale.

   -  **Custom operations**:

      -  "BFPFixNeuron" which supports block floating-point data type. It can run on the CPU on Windows, and on both the CPU and GPU on Linux.
      -  "VitisQuantizeLinear" and "VitisDequantizeLinear" which support INT32/UINT32, Float16, Bfloat16, INT16/UINT16 quantization.
      -  "VitisInstanceNormalization" and "VitisLSTM" which have customized Bfloat16 kernels.
      -  All custom operations support running on the CPU on both Linux and Windows.

   -  **Advanced Quantization Algorithms**:

      -  Supported CLE, BiasCorrection, AdaQuant, AdaRound and SmoothQuant.

   -  **Operating System Support**:

      -  Linux and Windows.

New Features (Version 0.1.0)
----------------------------

-  **AMD Quark for PyTorch**

   -  **Pytorch Quantizer Enhancements**:

      -  Eager mode is supported.
      -  Post Training Quantization (PTQ) is now available.
      -  Automatic in-place replacement of nn.module operations.
      -  Quantization of the following modules is supported: torch.nn.linear.
      -  The customizable calibration process is introduced.

   -  **Quantization Strategy**:

      -  Symmetric and asymmetric quantization are supported.
      -  Weight-only, dynamic, and static quantization modes are available.

   -  **Quantization Granularity**:

      -  Support for per-tensor, per-channel, and per-group granularity.

   -  **Data Types**:

      -  Multiple data types are supported, including float16, bfloat16, int4, uint4, int8, and fp8 (e4m3fn).

   -  **Calibration Methods**:

      -  MinMax, Percentile, and MSE calibration methods are now supported.

   -  **Large Language Model Support**:

      -  FP8 KV-cache quantization for large language models (LLMs).

   -  **Advanced Quantization Algorithms**:

      -  Support SmoothQuant, AWQ (uint4), and GPTQ (uint4) for LLMs. (Note: AWQ/GPTQ/SmoothQuant algorithms are currently limited to single GPU usage.)

   -  **Export Capabilities**:

      -  Export of Q/DQ quantized models to ONNX and vLLM-adopted JSON-safetensors format now supported.

   -  **Operating System Support**:

      -  Linux (supports ROCM and CUDA)
      -  Windows (supports CPU only).
