.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Language Model Post Training Quantization (PTQ) Using Quark
===========================================================

.. note::

   For information on accessing Quark PyTorch examples, refer to :doc:`Accessing PyTorch Examples <pytorch_examples>`.
   This example and the relevant files are available at ``/torch/language_modeling/llm_ptq``.

This document provides examples of post training quantizing (PTQ) and exporting the language models (such as OPT and Llama) using Quark. For evaluation of quantized model, refer to :doc:`Model Evaluation <example_quark_torch_llm_eval>`.

Supported Models
----------------

.. list-table:: Supported Models
   :widths: 35 8 8 8 8 8 10 10 8
   :header-rows: 1

   * - Model Name
     - FP8①
     - INT②
     - MX③
     - AWQ④
     - GPTQ⑤
     - SmoothQuant
     - AutoSmoothQuant
     - Rotation
   * - meta-llama/Llama-2-\*-hf ⑥
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - meta-llama/Llama-3-\*B(-Instruct)
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - meta-llama/Llama-3.1-\*B(-Instruct)
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - meta-llama/Llama-3.2-\*B(-Instruct)
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
   * - meta-llama/Llama-3.2-\*B-Vision(-Instruct) ⑦
     - ✓
     - ✓
     -
     -
     -
     -
     -
     -
   * - meta-llama/Llama-4-\*
     - ✓
     - ✓
     -
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - facebook/opt-\*
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
     -
   * - EleutherAI/gpt-j-6b
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
     -
   * - THUDM/chatglm3-6b
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
     -
   * - Qwen/Qwen-\*
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
     -
   * - Qwen/Qwen1.5-\*
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
     -
   * - Qwen/Qwen1.5-MoE-A2.7B
     - ✓
     - ✓
     - ✓
     - ✓
     -
     -
     -
     -
   * - Qwen/Qwen2-\*
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
     -
   * - Qwen/Qwen3-\*
     - ✓
     - ✓
     -
     -
     -
     -
     -
     -
   * - Qwen/Qwen3-MoE-\*
     - ✓
     - ✓
     -
     -
     -
     -
     - ✓
     -
   * - microsoft/phi-2
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
     -
   * - microsoft/Phi-3-mini-\*k-instruct
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
     -
   * - microsoft/Phi-3.5-mini-instruct
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
     -
   * - mistralai/Mistral-7B-v0.1
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
     -
   * - mistralai/Mixtral-8x7B-v0.1
     - ✓
     - ✓
     -
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - hpcai-tech/grok-1
     - ✓
     - ✓
     -
     - ✓
     - ✓
     - ✓
     -
     -
   * - google/gemma-2-\*
     - ✓
     - ✓
     -
     - ✓
     - ✓
     - ✓
     -
     -
   * - google/gemma-3-\*
     - ✓
     - ✓
     -
     - ✓
     - ✓
     - ✓
     -
     -
   * - allenai/OLMo-\*
     - ✓
     - ✓
     -
     - ✓
     - ✓
     - ✓
     -
     -
   * - deepseek-ai/deepseek-moe-16b-chat
     - ✓
     -
     -
     -
     - ✓
     -
     -
     -
   * - deepseek-ai/DeepSeek-V2-\*
     - ✓
     - ✓
     -
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - deepseek-ai/DeepSeek-V3
     - ✓
     - ✓
     -
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - CohereForAI/c4ai-command-r-\*
     - ✓
     -
     -
     -
     -
     - ✓
     -
     -
   * - databricks/dbrx-instruct
     - ✓
     -
     -
     -
     -
     -
     -
     -
   * - ibm-granite/granite-\*
     - ✓
     - ✓
     -
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - openai/gpt-oss-\*
     - ✓
     - ✓
     -
     - ✓
     - ✓
     - ✓
     -
     -
   * - AMD/Instella-\*
     - ✓
     - ✓
     -
     - ✓
     -
     -
     -
     -
   * - Qwen/Qwen3-VL-MoE-\*
     - ✓
     - ✓
     -
     -
     -
     -
     -
     -
   * - moonshotai/Kimi-K2.5-\*
     -
     -
     - ✓
     -
     -
     -
     -
     -
   * - moonshotai/Kimi-K2-Instruct-\*
     -
     -
     - ✓
     -
     -
     -
     -
     -
   * - moonshotai/Kimi-K2-Thinking-\*
     -
     -
     - ✓
     -
     -
     -
     -
     -
   * - deepseek-ai/DeepSeek-V3.2-\*
     -
     -
     - ✓
     -
     -
     -
     -
     -

.. note::

   - FP8 means ``OCP fp8_e4m3`` data type quantization.
   - INT includes INT8, INT4, UINT4 data type quantization.
   - MX includes OCP data type MXFP4, MXFP6E3M2, MXFP6E2M3.
   - AWQ supports INT4 and UINT4 weight-only quantization.
   - GPTQ only supports QuantScheme as 'PerGroup' and 'PerChannel'.
   - ``\*`` represents different model sizes, such as ``7b``.
   - meta-llama/Llama-3.2-\*B-Vision models only quantize language parts.
   - moonshotai/Kimi-K2.5-\*, moonshotai/Kimi-K2-Instruct-\*, moonshotai/Kimi-K2-Thinking-\*, and deepseek-ai/DeepSeek-V3.2-\* are supported via file-to-file quantization mode.

Preparation
-----------

For Llama2 models, download the HF Llama2 checkpoint. The Llama2 models checkpoint can be accessed by submitting a permission request to Meta. For additional details, see the Llama2 page on Huggingface. Upon obtaining permission, download the checkpoint to the `[llama checkpoint folder]`.

Quantization & Export Scripts & Import Scripts
----------------------------------------------

You can run the following Python scripts in the current path. Here we use Llama as an example.

.. note::

   - To avoid memory limitations, GPU users can add the `--multi_gpu` argument when running the model on multiple GPUs.
   - CPU users should add the `--device cpu` argument.

Multi-GPU Loading of Very Large MoE Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For very large MoE models (e.g., GLM-5, Qwen3.5-MoE, Kimi-K2.5, DeepSeek-V3) placed across multiple GPUs, Quark automatically throttles the Hugging Face parallel safetensors loader to a single worker. This avoids contention on the HIP/CUDA caching allocator on GPU 0, which would otherwise surface as ``HSA_STATUS_ERROR_OUT_OF_RESOURCES`` (ROCm) or a hang during ``caching_allocator_warmup``.

The throttle engages only when the model is placed on multiple GPUs (``--multi_gpu`` or ``--multi_device``) **and** exceeds an internal size threshold (~30 000 quantized ``nn.Linear`` modules). Single-GPU runs and smaller models are unaffected. To override the worker count, set the ``QUARK_HF_LOADER_WORKERS`` environment variable:

.. code-block:: bash

   export QUARK_HF_LOADER_WORKERS=4   # any positive integer

On Linux, also ensure ``vm.max_map_count`` is at least ``1048576`` (``4194304`` recommended). Quark prints a warning when the value is too low. Raise it with:

.. code-block:: bash

   sudo sysctl -w vm.max_map_count=4194304

Recipe 1: Evaluation of Llama Float16 Model without Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --skip_quantization

Recipe 2: FP8 (OCP fp8_e4m3) Quantization & Json_SafeTensors_Export with KV Cache
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme fp8 \
                             --kv_cache_dtype fp8 \
                             --num_calib_data 128 \
                             --model_export hf_format

Recipe 3: INT4 Weight-Only Quantization & Json_SafeTensors_Export with AWQ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme int4_wo_128 \
                             --num_calib_data 128 \
                             --quant_algo awq \
                             --dataset pileval_for_awq_benchmark \
                             --seq_len 512 \
                             --model_export hf_format

Recipe 4: INT8 Static Quantization & Json_SafeTensors_Export (on CPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme int8 \
                             --num_calib_data 128 \
                             --device cpu \
                             --model_export hf_format

Recipe 5: UINT4 Weight-Only Quantization & GGUF_Export with AWQ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme uint4_wo_32 \
                             --quant_algo awq \
                             --num_calib_data 128 \
                             --dataset pileval_for_awq_benchmark \
                             --model_export gguf

Recipe 6: OCP MX Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark now supports the datatype OCP MXFP4, MXFP6E3M2, MXFP6E2M3. Take ``mxfp4`` scheme as an example to quantize the model to datatype OCP MX:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme mxfp4 \
                             --num_calib_data 32

Other available MX schemes include ``mxfp6_e3m2``, ``mxfp6_e2m3`` and ``mxfp4_fp8``.

Recipe 7: PTPC_FP8 (activation fp8_e4m3 dynamic per-token, weight fp8_e4m3 per-channel) Quantization & Json_SafeTensors_Export
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme ptpc_fp8 \
                             --num_calib_data 128 \
                             --model_export hf_format

Recipe 8: BFP16 Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark now supports the datatype BFP16 (Block Floating Point 16 bits). Use the following command to quantize the model to datatype BFP16:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme bfp16 \
                             --num_calib_data 16

Recipe 9: MX6 Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark now supports the datatype MX6. Use the following command to quantize the model to datatype MX6:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme mx6 \
                             --num_calib_data 16

Recipe 10: Import Quantized Model & Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quantized model can be imported and evaluated:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --import_model_dir [path to quantized model] \
                             --model_reload

.. note::

   Exporting quantized BFP16 and MX6 models is not supported yet.

Recipe 11: File-to-File Quantization (No Full Model Loading)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For ultra-large models (e.g., 600B+) that cause OOM when loaded into memory, Quark provides a file-to-file quantization workflow via the ``--file2file_quantization`` mode.
It quantizes safetensors files **one-by-one** without loading the full model, so peak memory is proportional to a single file (~5-10 GB) rather than the entire model.

This mode supports **weight-only quantization** and **dynamic activation quantization + weight quantization**, exports **hf_format** only, and can also accept pre-quantized inputs (FP8, compressed-tensors) and re-quantize them to a different format. For example, the command below runs file-to-file quantization to MXFP4 and shows common layer exclusions:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [model checkpoint folder] \
                             --output_dir [output folder] \
                             --quant_scheme mxfp4 \
                             --exclude_layers "*self_attn*" "*mlp.gate" "*mlp.gate.linear" "*lm_head" \
                             --file2file_quantization \
                             --skip_evaluation

Tutorial: Running a Model Not on the Supported List
---------------------------------------------------

For a new model that is not listed in Quark, you need to register a custom LLMTemplate in `quantize_quark.py`. The script automatically obtains the model type from `model.config.model_type` and checks if a corresponding template is available.

Follow these steps:

1. Check the model type.

   When you run the script with an unsupported model, you will see an error message like:

   .. code-block:: text

      [ERROR]: Model type 'internlm2' is not supported.

      Available templates: ['chatglm', 'cohere', 'dbrx', 'deepseek', ...]

      To add support for this model, uncomment and modify the 'Custom Model Templates'
      section at the top of this file to register a template for 'internlm2'.

   The model type is obtained from `model.config.model_type`. You need to create a template that matches this model type.

2. Register a custom LLM template in `quantize_quark.py`.

   Open `quantize_quark.py` and find the commented-out "Custom Model Templates" section at the top of the file. Uncomment and modify it to match your model's architecture.

   .. code-block:: python

      from quark.torch import LLMTemplate

      # --- Custom Model Templates ---
      # Define templates for model architectures not in the built-in list.
      # Model: internlm/internlm2-chat-7b
      internlm2_template = LLMTemplate(
          model_type="internlm2",            # Must match model.config.model_type
          kv_layers_name=["*wqkv"],          # KV projection layer patterns
          q_layer_name="*wqkv",              # Q projection layer pattern
          exclude_layers_name=["lm_head"],   # Layers to exclude from quantization
      )
      LLMTemplate.register_template(internlm2_template)
      print(f"[INFO]: Registered template '{internlm2_template.model_type}'")

   To determine the correct layer name patterns for your model, you can print the model structure:

   .. code-block:: python

      print(model)

   Or use ``model.named_modules()`` to list all layer names.

3. [Optional] Register a custom quantization scheme.

   If you need a quantization scheme that is not built-in, you can register it in the "Custom Quantization Schemes" section:

   .. code-block:: python

      from quark.torch.quantization.config.config import (
          Int8PerTensorSpec,
          QLayerConfig,
      )

      # --- Custom Quantization Schemes ---
      # INT8 weight-only quantization
      int8_wo_scheme = QLayerConfig(weight=Int8PerTensorSpec().to_quantization_spec())
      LLMTemplate.register_scheme("int8_wo", config=int8_wo_scheme)
      print(f"[INFO]: Registered quantization scheme 'int8_wo'")

   Then you can use ``--quant_scheme int8_wo`` when running the script.

4. [Optional] If using AWQ, GPTQ, SmoothQuant, or Rotation algorithms.

   For GPTQ:

   In the config json file, you should collate all linear layers in decoder layers and put them in the `inside_layer_modules` list and put the decoder layers name in the `model_decoder_layers` list.

   For AWQ:

   You could refer to the :doc:`AWQ documentation <../pytorch/awq_document>` for guidance on writing the configuration file.

   For SmoothQuant:

   You could refer to the :doc:`SmoothQuant documentation <../pytorch/smoothquant>` for guidance on writing the configuration file.

   After creating the config json file, pass it via command line argument:

   .. code-block:: bash

      python quantize_quark.py \
          --model_dir <model_path> \
          --quant_scheme int4_wo_128 \
          --quant_algo awq \
          --quant_algo_config_file awq ./awq_config.json

   You can specify multiple algorithm config files:

   .. code-block:: bash

      python quantize_quark.py \
          --model_dir <model_path> \
          --quant_scheme int4_wo_128 \
          --quant_algo awq,smoothquant \
          --quant_algo_config_file awq ./awq_config.json \
          --quant_algo_config_file smoothquant ./smoothquant_config.json

End to end tutorials
--------------------

In addition to the snippets above, you can refer to end-to-end tutorials:

.. toctree::
   :caption: More examples
   :maxdepth: 1

   FP4 Post Training Quantization (PTQ) for LLM models <../tutorials/torch/example_fp4>
   FP8 Post Training Quantization (PTQ) for LLM models <../tutorials/torch/example_fp8>

Tutorial: Generating AWQ Configuration Automatically (Experimental)
-------------------------------------------------------------------

We provide a script `awq_auto_config_helper.py` to simplify user operations by quickly identifying modules compatible with the "AWQ" and "SmoothQuant" algorithms within the model through `torch.compile`.

Installation
------------

This script requires PyTorch version 2.4 or higher.

Usage
-----

The `MODEL_DIR` variable should be set to the model name from Hugging Face, such as `facebook/opt-125m`, `Qwen/Qwen2-0.5B`, or `EleutherAI/gpt-j-6b`.

To run the script, use the following command:

.. code-block:: bash

   MODEL_DIR="your_model"
   python awq_auto_config_helper.py --model_dir "${MODEL_DIR}"
