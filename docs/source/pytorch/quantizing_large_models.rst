.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Hands-on Quantizing and Serving of Large Models
===============================================

This page provides hands-on examples to quantize popular large models using the ``examples/torch/language_modeling/llm_ptq/quantize_quark.py`` example script.

moonshotai/Kimi-K2.6
---------------------

`Kimi-K2.6 <https://huggingface.co/moonshotai/Kimi-K2.6>`_ is a Mixture-of-Experts model whose expert weights are compressed to INT4 (group-wise, ``group_size=32``, symmetric), while self-attention, shared experts, dense MLP projections, ``lm_head``, vision tower, and ``mm_projector`` are kept in bfloat16. Because it does not fit on a single GPU, all examples below use ``--multi_gpu balanced`` to distribute layers evenly across visible GPUs.


MXFP4
^^^^^

MXFP4 (Microscaling FP4) is a weight-only format that does not require calibration data. See the `Quark MX quantization guide <https://quark.docs.amd.com/latest/pytorch/adv_mx.html>`_ and the `OCP Microscaling Formats (MX) v1.0 specification <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>`_.

Quantization to MXFP4
"""""""""""""""""""""

.. code-block:: bash

   CUDA_VISIBLE_DEVICES="0,1,2,3" python quantize_quark.py \
       --quant_scheme mxfp4 \
       --model_dir moonshotai/Kimi-K2.6 \
       --model_export hf_format \
       --output_dir Kimi-K2.6-MXFP4 \
       --skip_evaluation \
       --multi_gpu balanced \
       --revision "refs/pr/33" \
       --exclude_layers "*lm_head*" "*vision_tower*" "*.mlp.gate" "*mm_projector*" "*shared_experts*" "*self_attn*" "*mlp.gate_proj*" "*mlp.up_proj*" "*mlp.gate_up_proj*" "*mlp.down_proj*"

The ``--exclude_layers`` patterns match the modules that are already in bfloat16 in the original checkpoint, to avoid quantizing weights that were intentionally kept in higher precision.

``--revision "refs/pr/33"`` pins `Kimi-K2.6 PR #33 <https://huggingface.co/moonshotai/Kimi-K2.6/discussions/33>`_ which fixes the ``mlp.gate.e_score_correction_bias`` dtype (float32 in the checkpoint, but initialized without explicit dtype by the modeling code). Once the PR is merged this flag can be dropped.

Model serving through vLLM
""""""""""""""""""""""""""

MXFP4 models run natively on AMD Instinct MI350 and MI355X. MXFP4 dense and MoE models can also run on MI300 and MI325 through an emulation code path, but this is intended for research purposes only. A ready-to-use quantized checkpoint is available at `amd/Kimi-K2.6-MXFP4 <https://huggingface.co/amd/Kimi-K2.6-MXFP4>`_. See vLLM's `quark_ocp_mx.py <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/quark/schemes/quark_ocp_mx.py>`_ (dense layers) and `quark_moe.py <https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/quark/quark_moe.py>`_ (MoE layers) for the vLLM integration.

.. code-block:: bash

   # Use the pre-quantized checkpoint from Hugging Face, or your own output_dir from the quantization step above.
   vllm serve amd/Kimi-K2.6-MXFP4 --tensor-parallel-size 8 --trust-remote-code

.. code-block:: bash

   curl http://localhost:8000/v1/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "amd/Kimi-K2.6-MXFP4",
           "prompt": "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|><|im_user|>user<|im_middle|>Question: What would you suggest me to do in Paris today? Give 3 suggestions. Answer:<|im_end|><|im_assistant|>assistant<|im_middle|><think>",
           "max_tokens": 50
       }'


NVFP4
^^^^^

NVFP4 (``fp4_block16_scale_e4m3`` in AMD Quark) uses block-wise FP4 with E4M3 group scales. Unlike MXFP4, it requires calibration data in order to pre-compute the NVFP4 global scales for activations.

Quantization to NVFP4
"""""""""""""""""""""

.. code-block:: bash

    # Required by Kimi-K2.6 remote code, not native to Transformers library.
    pip install "transformers<5"

    CUDA_VISIBLE_DEVICES="0,1,2,3,4,5" python quantize_quark.py \
       --quant_scheme nvfp4 \
       --model_dir moonshotai/Kimi-K2.6 \
       --model_export hf_format \
       --output_dir Kimi-K2.6-NVFP4 \
       --skip_evaluation \
       --batch_size 24 \
       --num_calib_data 288 \
       --revision "refs/pr/33" \
       --multi_gpu balanced \
       --exclude_layers "*lm_head*" "*vision_tower*" "*.mlp.gate" "*mm_projector*" "*shared_experts*" "*self_attn*" "*mlp.gate_proj*" "*mlp.up_proj*" "*mlp.gate_up_proj*" "*mlp.down_proj*"

``--batch_size 24`` makes calibration significantly faster than the default of 1, when GPU memory allows it.

``--num_calib_data 288`` is faster than the default of 512 and is usually sufficient. Adjust up if you observe accuracy degradation.

Model serving through vLLM
""""""""""""""""""""""""""

NVFP4 models are compatible with vLLM on AMD Instinct MI3xx through dense-layer and MoE-layer NVFP4 emulation kernels, intended for research purposes only. See `vllm#35859 <https://github.com/vllm-project/vllm/pull/35859>`_ (Quark NVFP4 checkpoint loading) and `vllm#40033 <https://github.com/vllm-project/vllm/pull/40033>`_ (NVFP4 dequantization and QDQ emulation Triton kernels).

.. code-block:: bash

    vllm serve /path/to/Kimi-K2.6-NVFP4 --tensor-parallel-size 8 --trust-remote-code

.. code-block:: bash

    curl http://localhost:8000/v1/completions \
        -H "Content-Type: application/json" \
        -d '{
            "model": "/path/to/Kimi-K2.6-NVFP4",
            "prompt": "<|im_system|>system<|im_middle|>You are Kimi, an AI assistant created by Moonshot AI.<|im_end|><|im_user|>user<|im_middle|>Question: What would you suggest me to do in Paris today? Give 3 suggestions. Answer:<|im_end|><|im_assistant|>assistant<|im_middle|><think>",
            "max_tokens": 50
        }'

Basic model evaluation using GSM8K mathematical reasoning task
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

To validate the correctness of the quantized models, we must evaluate them prior to deployment. Evaluating reasoning models and tool calling models is a complex topic, however, evaluating a quantized model on a mathematical reasoning task as GSM8K is a first good proxy to be confident that the quantization has not been overly aggressive and destructive.

One can for example use, [following lm-evaluation-harness recommendations](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/docs/interface.md#evaluating-thinkingreasoning-models):

.. code-block:: bash

    pip install lm-eval

    export TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
    export PRETRAINED_PATH="/path/to/Kimi-K2.6-NVFP4"

    lm_eval \
        --model vllm \
        --model_args '{"pretrained":"'"${PRETRAINED_PATH}"'","dtype":"auto","tensor_parallel_size":4,"enable_thinking": true,"think_end_token":"</think>","trust_remote_code":true,"mm_encoder_tp_mode":"data","reasoning_parser":"kimi_k2"}' \
        --device "cuda" \
        --gen_kwargs "do_sample=True,temperature=0.6,top_p=0.95,max_gen_toks=4096" \
        --tasks gsm8k_platinum \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --batch_size "auto" \
        --log_samples \
        --output_path eval_results/${TIMESTAMP}_kimi_k26_nvfp4

Giving:

.. code-block::

    vllm ({'pretrained': 'Kimi-K2.6-NVFP4', 'dtype': 'auto', 'tensor_parallel_size': 4, 'enable_thinking': True, 'think_end_token': '</think>', 'mm_encoder_tp_mode': 'data', 'reasoning_parser': 'kimi_k2'}), gen_kwargs: ({'do_sample': True, 'temperature': 0.6, 'top_p': 0.95, 'max_gen_toks': 4096}), limit: None, num_fewshot: None, batch_size: auto
    |    Tasks     |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
    |--------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
    |gsm8k_platinum|      3|flexible-extract|     5|exact_match|↑  |0.9917|±  |0.0026|
    |              |       |strict-match    |     5|exact_match|↑  |0.9909|±  |0.0027|
