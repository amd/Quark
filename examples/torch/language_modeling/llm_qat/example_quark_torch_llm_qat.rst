Language Model QAT Using Quark
===========================================================

This document provides examples of Quantization-Aware Training (QAT) for language models using Quark.

.. note::

   For information on accessing Quark PyTorch examples, refer to :doc:`Accessing PyTorch Examples <pytorch_examples>`.
   This example and the relevant files are available at ``/torch/language_modeling/llm_qat``.

Supported Models
----------------

+-----------------------------------------+-------------------------------+
| Model Name                              | WEIGHT-ONLY (INT4.g128)       |
+=========================================+===============================+
| microsoft/Phi-3-mini-4k-instruct        | ✓                             |
+-----------------------------------------+-------------------------------+
| THUDM/chatglm3-6b                       | ✓                             |
+-----------------------------------------+-------------------------------+

Preparation
-----------

Please install the required packages before running QAT by executing ``pip install -r requirements.txt``. To evaluate the model, install the necessary dependencies by running ``pip install -r ../llm_eval/requirements.txt``.

(Optional) For LLM models, download the Hugging Face checkpoint.

QAT Scripts
-----------

You can run the following Python scripts in the ``examples/torch/language_modeling/llm_qat`` path. Here, ChatGLM3-6B is used as an example.

.. note::

   1. The ChatGLM3-6B model may encounter some `tokenizer-related issues <https://github.com/THUDM/ChatGLM3/issues/1324>`__. To resolve this, please install ``transformers==4.44.0``. Additionally, ensure that torch is updated to version ``2.4.0`` or higher.

   2. When performing full fine-tuning on large language models, it is crucial to carefully select an appropriate fine-tuning dataset for optimal results.

Recipe 1: Evaluation of Original LLM
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   finetune_checkpoint="./finetune_checkpoint/Chatglm3-6B"
   mkdir -p $finetune_checkpoint
   CUDA_VISIBLE_DEVICES=0 python main.py \
                        --model THUDM/chatglm3-6b \
                        --model_trust_remote_code \
                        --skip_quantization \
                        --skip_finetune \
                        --eval_task openllm | tee $finetune_checkpoint/test_bf16.log

Recipe 2: QAT Finetuning and Export to safetensors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   output_dir="./quantized_model/Chatglm3-6B-u4w-ft"
   CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
                        --model THUDM/chatglm3-6b \
                        --model_trust_remote_code \
                        --quant_scheme w_uint4_asym \
                        --group_size 128 \
                        --finetune_dataset wikitext \
                        --finetune_datasubset wikitext-2-raw-v1 \
                        --finetune_epoch 10 \
                        --finetune_lr 2e-5 \
                        --finetune_iter 500 \
                        --finetune_seqlen 512 \
                        --finetune_batchsize 8 \
                        --finetune_checkpoint $finetune_checkpoint \
                        --model_export hf_format \
                        --output_dir $output_dir \
                        --skip_evaluation | tee $finetune_checkpoint/finetune_w_uint4_asym.log

Recipe 3: Reload and Evaluate Finetuned Model
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 python main.py \
                        --model THUDM/chatglm3-6b \
                        --model_trust_remote_code \
                        --skip_finetune \
                        --model_reload \
                        --import_model_dir $output_dir \
                        --eval_task openllm | tee $finetune_checkpoint/test_w_uint4_asym_finetuned.log

.. raw:: html

   <!--
   ## License
   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
