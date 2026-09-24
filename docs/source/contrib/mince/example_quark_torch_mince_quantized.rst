Evaluating a Quark-Quantized Model with MINCE
=============================================

The key idea is that you do **not** re-run MINCE on the quantized model. You
calibrate ``n*`` once on the bf16 model, freeze it, and then reuse that frozen
subset for every downstream quantized variant of that model for efficient
evaluation. Below is an example of how to use MINCE to evaluate Quark-quantized
models.

Prerequisites
-------------

Steps 1-3 of :doc:`example_quark_torch_mince` completed for your benchmark, so
you already have:

- the bf16 full-benchmark ``results.json`` (the reference score),
- the bf16 subset score and its validated drift, and
- ``mince_frozen/<benchmark>/<model-label>/subset_samples.json``.

Workflow
--------

.. code-block:: text

   (from example_quark_torch_mince: BF16 EVAL -> SIZING -> FREEZE -> validated n*)
           |
   QUANTIZE         (quantize_quark.py)    -> Quark INT4/AWQ checkpoint (hf_format)
           |
   SUBSET SCORING   (lm_eval --samples)    -> quantized model scored on the frozen subset
           |
   COMPARE          (validate.py)          -> quantization impact on the same items


The steps below span two directories — ``quantize_quark.py`` lives under
``examples/torch/language_modeling/llm_ptq`` and the MINCE CLIs under
``examples/contrib/mince`` — so set the repo root once and use it throughout:

.. code-block:: bash

   export QUARK_REPO=/path/to/Quark

**1. Quantize the model** — Example of INT4 weight-only, group size 128, with AWQ. Export in
``hf_format`` so the result is a Hugging Face-loadable checkpoint:

.. code-block:: bash

   cd $QUARK_REPO/examples/torch/language_modeling/llm_ptq

   python3 quantize_quark.py \
     --model_dir meta-llama/Llama-3.1-8B-Instruct \
     --output_dir $QUARK_REPO/quark_models/Llama-3.1-8B-Instruct-awq-int4-g128 \
     --quant_scheme int4_wo_128 \
     --quant_algo awq \
     --dataset pileval_for_awq_benchmark \
     --num_calib_data 128 \
     --seq_len 512 \
     --model_export hf_format

The ``--model_export hf_format`` flag is what matters for MINCE: it writes a
checkpoint that lm-eval-harness can load directly with ``pretrained=``, so the
subset-scoring step below is identical to the bf16 flow apart from the model
path.

**2. Score the quantized model on the frozen subset** — the *same* lm-eval
command you used in step 4 of the bf16 walkthrough, pointing ``pretrained=`` at
the quantized checkpoint. The
``--samples`` map is unchanged, because it is the frozen bf16-calibrated subset:

.. code-block:: bash

   cd $QUARK_REPO/examples/contrib/mince

   lm_eval --model hf \
     --model_args pretrained=$QUARK_REPO/quark_models/Llama-3.1-8B-Instruct-awq-int4-g128 \
     --tasks gsm8k --num_fewshot 0 --apply_chat_template \
     --samples "$(cat mince_frozen/gsm8k/meta-llama__Llama-3.1-8B-Instruct/subset_samples.json)" \
     --log_samples \
     --output_path eval_test_logs/Llama-3.1-8B-Instruct-awq-int4-g128-GSM8K \
     --batch_size 1

**Reuse the same generation settings as the bf16 runs** (``--num_fewshot``,
``--apply_chat_template``, and any other options). Otherwise the difference you
measure will include unwanted variables, not just quantization effects.

**3. Compare** — run ``validate.py`` on the quantized subset run against the
**bf16 subset run**. Both scored the identical frozen items, so the difference
is attributable to quantization:

.. code-block:: bash

   python validate.py --benchmark gsm8k \
     --subset-results eval_test_logs/Llama-3.1-8B-Instruct-awq-int4-g128-GSM8K/*/results_*.json \
     --baseline-results eval_test_logs/Llama-3.1-8B-Instruct-GSM8K/*/results_*.json

.. note::

   Because the frozen subset was already validated as faithful to the full
   benchmark within the P95 drift budget from sizing, the quantized model's
   score on ``n*`` estimates its full-benchmark score to within roughly that
   same budget — without ever running the full benchmark on the quantized
   model. This cuts the evaluation time across multiple different quantized models that need evaluation
   and serves as one use case of MINCE for efficient evaluation. See the MINCE
   paper: `arxiv.org/abs/2606.22826 <https://arxiv.org/pdf/2606.22826>`_.
