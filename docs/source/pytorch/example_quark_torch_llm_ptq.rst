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
   :widths: 40 10 10 10 10 10 10
   :header-rows: 1

   * - Model Name
     - FP8①
     - INT②
     - MX③
     - AWQ/GPTQ(INT)④
     - SmoothQuant
     - Rotation
   * - meta-llama/Llama-2-\*-hf ⑤
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
   * - meta-llama/Llama-3.1-\*B(-Instruct)
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
   * - meta-llama/Llama-3.2-\*B-Vision(-Instruct) ⑥
     - ✓
     - ✓
     -
     -
     -
     -
   * - facebook/opt-\*
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - EleutherAI/gpt-j-6b
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - THUDM/chatglm3-6b
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - Qwen/Qwen-\*
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - Qwen/Qwen1.5-\*
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - Qwen/Qwen1.5-MoE-A2.7B
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - Qwen/Qwen2-\*
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - microsoft/phi-2
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - microsoft/Phi-3-mini-\*k-instruct
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - microsoft/Phi-3.5-mini-instruct
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - mistralai/Mistral-7B-v0.1
     - ✓
     - ✓
     - ✓
     - ✓
     - ✓
     -
   * - mistralai/Mixtral-8x7B-v0.1
     - ✓
     - ✓
     -
     -
     -
     -
   * - hpcai-tech/grok-1
     - ✓
     - ✓
     -
     - ✓
     -
     -
   * - CohereForAI/c4ai-command-r-plus-08-2024
     - ✓
     -
     -
     -
     -
     -
   * - CohereForAI/c4ai-command-r-08-2024
     - ✓
     -
     -
     -
     -
     -
   * - CohereForAI/c4ai-command-r-plus
     - ✓
     -
     -
     -
     -
     -
   * - CohereForAI/c4ai-command-r-v01
     - ✓
     -
     -
     -
     -
     -
   * - databricks/dbrx-instruct
     - ✓
     -
     -
     -
     -
     -
   * - deepseek-ai/deepseek-moe-16b-chat
     - ✓
     -
     -
     -
     -
     -

.. note::

   - FP8 means ``OCP fp8_e4m3`` data type quantization.
   - INT includes INT8, UINT8, INT4, UINT4 data type quantization
   - MX includes OCP data type MXINT8, MXFP8E4M3, MXFP8E5M2, MXFP4, MXFP6E3M2, MXFP6E2M3.
   - GPTQ only supports QuantScheme as 'PerGroup' and 'PerChannel'.
   - ``\*`` represents different model sizes, such as ``7b``.
   - meta-llama/Llama-3.2-\*B-Vision models only quantize language parts.

Preparation
-----------

For Llama2 models, download the HF Llama2 checkpoint. The Llama2 models checkpoint can be accessed by submitting a permission request to Meta. For additional details, see the Llama2 page on Huggingface. Upon obtaining permission, download the checkpoint to the `[llama checkpoint folder]`.

Quantization & Export Scripts & Import Scripts
----------------------------------------------

You can run the following Python scripts in the current path. Here we use Llama as an example.

.. note::

   - To avoid memory limitations, GPU users can add the `--multi_gpu` argument when running the model on multiple GPUs.
   - CPU users should add the `--device cpu` argument.

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
                             --quant_scheme w_fp8_a_fp8 \
                             --kv_cache_dtype fp8 \
                             --num_calib_data 128 \
                             --model_export hf_format

Recipe 3: INT Weight-Only Quantization & Json_SafeTensors_Export with AWQ
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_int4_per_group_sym \
                             --num_calib_data 128 \
                             --quant_algo awq \
                             --dataset pileval_for_awq_benchmark \
                             --seq_len 512 \
                             --model_export hf_format

Recipe 4: INT Static Quantization & Json_SafeTensors_Export (on CPU)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_int8_a_int8_per_tensor_sym \
                             --num_calib_data 128 \
                             --device cpu \
                             --model_export hf_format

Recipe 5: Quantization & GGUF_Export with AWQ (W_uint4 A_float16 per_group asymmetric)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_uint4_per_group_asym \
                             --quant_algo awq \
                             --num_calib_data 128 \
                             --dataset pileval_for_awq_benchmark \
                             --group_size 32 \
                             --model_export gguf

Recipe 6: OCP MX Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Quark now supports the datatype OCP MXINT8, MXFP8E4M3, MXFP8E5M2, MXFP4, MXFP6E3M2, MXFP6E2M3. Take ``w_mxfp4_a_mxfp4`` scheme as an example to quantize the model to datatype OCP MX:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_mxfp4_a_mxfp4 \
                             --num_calib_data 32

Recipe 7: BFP16 Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark now supports the datatype BFP16 (Block Floating Point 16 bits). Use the following command to quantize the model to datatype BFP16:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_bfp16_a_bfp16 \
                             --num_calib_data 16

Recipe 8: MX6 Quantization
~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark now supports the datatype MX6. Use the following command to quantize the model to datatype MX6:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --output_dir output_dir \
                             --quant_scheme w_mx6_a_mx6 \
                             --num_calib_data 16

Recipe 9: Import Quantized Model & Evaluation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The quantized model can be imported and evaluated:

.. code-block:: bash

   python3 quantize_quark.py --model_dir [llama checkpoint folder] \
                             --import_model_dir [path to quantized model] \
                             --model_reload \
                             --import_file_format hf_format

.. note::

   Exporting quantized BFP16 and MX6 models is not supported yet.

Tutorial: Running a Model Not on the Supported List
---------------------------------------------------

For a new model that is not listed in Quark, you need to modify some relevant files. Follow these steps:

1. Add the model type to `MODEL_NAME_PATTERN_MAP` in `get_model_type` function in `quantize_quark.py`.

   `MODEL_NAME_PATTERN_MAP` describes model type, which is used to configure the `quant_config` for the models. You can use part of the model's HF-ID as the key of the dictionary, and the lowercase version of this key as the value.

   .. code-block:: python

      def get_model_type(model: nn.Module) -> str:
          MODEL_NAME_PATTERN_MAP = {
              "Llama": "llama",
              "OPT": "opt",
              ...
              "Cohere": "cohere",  # <---- Add code HERE
          }
          for k, v in MODEL_NAME_PATTERN_MAP.items():
              if k.lower() in type(model).__name__.lower():
                  return v

2. Customize tokenizer for your model in `get_tokenizer` function in `quantize_quark.py`.

   For the most part, the `get_tokenizer` function is applicable. But for some models, such as `CohereForAI/c4ai-command-r-v01`, `use_fast` can only be set to `True` (as of transformers-4.44.2). You can customize the tokenizer by referring to your model's Model card on Hugging Face and `tokenization_auto.py` in transformers.

   .. code-block:: python

      def get_tokenizer(ckpt_path: str, max_seq_len: int = 2048, model_type: Optional[str] = None) -> AutoTokenizer:
          print(f"Initializing tokenizer from {ckpt_path}")
          use_fast = True if model_type == "grok" or model_type == "cohere" else False
          tokenizer = AutoTokenizer.from_pretrained(ckpt_path,
                                                    model_max_length=max_seq_len,
                                                    padding_side="left",
                                                    trust_remote_code=True,
                                                    use_fast=use_fast)

3. Create a new LLM template for your model.

   For new models not supported by the built-in templates, you need to create a custom template using :py:class:`.LLMTemplate`.

   .. code-block:: python

      from quark.torch import LLMTemplate

      # Create a new template for your model
      new_template = LLMTemplate(
          model_type="cohere",
          kv_layers_name=["*k_proj", "*v_proj"],  # KV projection layer patterns
          q_layer_name="*q_proj",                 # Q projection layer pattern
          exclude_layers_name=["lm_head"]         # Layers to exclude from quantization
      )

      # Register the template if you want to use the template in other places
      LLMTemplate.register_template(new_template)

   Now you can use the template for quantization configuration:

   .. code-block:: python

      # Create quantization configuration
      quant_config = new_template.get_config(
          scheme="fp8",
          kv_cache_scheme="fp8"
      )

5. [Optional] If using AWQ, GPTQ, SmoothQuant for the new model, create the algorithm config json file for the model

   For GPTQ:

   In the config json file, you should collate all linear layers in decoder layers and put them in the `inside_layer_modules` list and put the decoder layers name in the `model_decoder_layers` list.

   For AWQ:

   You could refer to the :doc:`AWQ documentation <../pytorch/awq_document>` for guidance on writing the configuration file.

   For SmoothQuant:

   You could refer to the :doc:`SmoothQuant documentation <../pytorch/smoothquant>` for guidance on writing the configuration file.

   After creating the config json file, you can pass the config json file to template of the model:

   .. code-block:: python

      from quark.torch import LLMTemplate
      from quark.torch.quantization.config.config import load_quant_algo_config_from_file

      # Load the config json file
      awq_config = load_quant_algo_config_from_file("awq_config.json")

      # Create a new template for your model
      new_template = LLMTemplate(
          model_type="cohere",
          exclude_layers_name=["lm_head"]         # Layers to exclude from quantization
          awq_config=awq_config
      )

      # Register the template if you want to use the template in other places
      LLMTemplate.register_template(new_template)

   Now you can use the template for AWQ quantization:

   .. code-block:: python

      quant_config = new_template.get_config(
          scheme="int4_wo_128",
          quant_algo="awq"
      )

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
