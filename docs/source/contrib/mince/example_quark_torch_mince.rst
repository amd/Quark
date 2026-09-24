MINCE: Compact Evaluation via Monte-Carlo Subset Sizing
=======================================================

MINCE (Monte-Carlo Informed N-sizing for Compact Evaluation) cuts LLM
benchmark evaluation time by evaluating a small,
frozen subset of items instead of the full benchmark, while keeping the subset
score close to the full-benchmark score.

For the method and experiments, see the MINCE paper:
`arxiv.org/abs/2606.22826 <https://arxiv.org/pdf/2606.22826>`_.


Getting Started
---------------

1. Ensure Quark is installed.
2. Install the example requirements:

   .. code-block:: bash

      cd examples/contrib/mince
      pip install -r requirements.txt

Currently supported benchmarks:

- CommonsenseQA
- GSM8K
- IFEval
- MMLU
- MMLU-Pro
- PIQA

Workflow
--------

The workflow is: calibrate once on the bf16 model's per-item logs to find
``n*``, freeze that subset, then reuse it to evaluate other models on the same
subset — cutting per-run eval time.

.. code-block:: text

   FULL BF16 EVALUATION   (lm-eval --log_samples)   -> per-item logs + full results.json
           |
   MINCE SIZING           (size.py)                 -> P95 drift by candidate N -> n*
           |
   FREEZE                 (freeze.py)               -> ID artifact + subset_samples.json
           |
   SUBSET SCORING         (lm-eval --samples)       -> a model scored on the frozen subset
           |
   VALIDATE               (validate.py)             -> drift = subset results vs full baseline

**1. Generate bf16 per-item logs** (full benchmark) with lm-eval-harness:

.. code-block:: bash

   lm_eval --model hf \
     --model_args pretrained=<bf16_model> \
     --tasks gsm8k --num_fewshot 0 --log_samples \
     --output_path bf16_logs/<bf16_model>-GSM8K --batch_size auto

   example:
   lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct --tasks gsm8k --num_fewshot 0 --log_samples --output_path bf16_test_logs/Llama-3.1-8B-Instruct-GSM8K --batch_size 1 --apply_chat_template
   lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct --tasks mmlu --num_fewshot 5 --log_samples --output_path bf16_test_logs/Llama-3.1-8B-Instruct-MMLU --batch_size 1 --apply_chat_template

You can set any other lm-eval-harness generation options you would normally use
for your scenario here — e.g. disabling/enabling thinking for reasoning models,
changing the batch size, or adding chat/system templates. MINCE does not impose
any particular generation configuration; beyond requiring per-item logs
(``--log_samples``), it simply consumes whatever lm-eval produces. The only other
requirement is that you reuse the *same* generation settings in the subset
scoring step (step 4) so the subset run is comparable to the full run.

**2. Size** — find ``n*`` from the Monte-Carlo drift curve:

.. code-block:: bash

   python size.py --benchmark gsm8k --model-dirs logs/<bf16_model>/...

   example:
   python size.py --benchmark gsm8k --model-dirs bf16_test_logs/Llama-3.1-8B-Instruct-GSM8K/meta-llama__Llama-3.1-8B-Instruct
   python size.py --benchmark mmlu --model-dirs bf16_test_logs/Llama-3.1-8B-Instruct-MMLU/meta-llama__Llama-3.1-8B-Instruct


Key sizing parameters:

- ``--B`` (default ``10000``): Monte-Carlo draws per candidate N
- ``--candidate-ns`` (default: the benchmark's configured grid): comma-separated
  subset sizes to sweep, e.g. ``100,200,400``.
- ``--tau`` (default ``0.01`` = 1 pp): marginal-gain threshold for ``n*`` — the
  per-step drop in worst-case P95 drift below which returns are diminishing.
- ``--seed`` (default ``42``).

.. tip::

   **Wanting to experiment with a different tau?** Add ``--plot`` to save ``sizing_plot.png``
   — the per-step marginal gain (pp per step). Read the marginal-gain bars, pick the
   level below which you consider returns diminishing for your use case, and that level is your new
   ``tau``: ``n*`` is the first N whose bar sits below it.
   Based on our experiments and MINCE paper, we have set a default value of ``tau`` of 0.01.

   .. code-block:: bash

      python size.py --benchmark mmlu \
        --model-dirs bf16_test_logs/Llama-3.1-8B-Instruct-MMLU/meta-llama__Llama-3.1-8B-Instruct \
        --plot

**3. Freeze** — draw the subset at a fixed seed and save the ID artifact:

.. code-block:: bash

   python freeze.py --benchmark gsm8k --n <n*> \
     --model-dirs logs/<bf16_model>/...

   example:
   python freeze.py --benchmark gsm8k --n 400 --model-dirs bf16_test_logs/Llama-3.1-8B-Instruct-GSM8K/meta-llama__Llama-3.1-8B-Instruct
   python freeze.py --benchmark mmlu --n 1500 --model-dirs bf16_test_logs/Llama-3.1-8B-Instruct-MMLU/meta-llama__Llama-3.1-8B-Instruct

.. tip::

   **Want to inspect the actual samples in the subset?** ``extract_inputs.py``
   reads ``subset_samples.json`` and renders the real lm-eval prompts (and gold
   targets) for exactly those frozen items into a JSON. Match ``--num-fewshot`` to
   your bf16 run.

   .. code-block:: bash

      python extract_inputs.py \
        --samples mince_frozen/mmlu/<model>/subset_samples.json \
        --out mince_frozen/mmlu/<model>/subset_inputs.json --num-fewshot 5

**4. Score the subset** — run the *same* lm-eval command used in step 1, just
add ``--samples`` with the subset_samples.json map. Point ``--output_path`` at the
model you want to evaluate on the smaller subset. ``--samples`` takes the JSON map
inline, so pass the file contents with ``"$(cat ...)"``:

.. code-block:: bash

   lm_eval --model hf \
     --model_args pretrained=<model> \
     --tasks <benchmark> --num_fewshot <same as step 1> --apply_chat_template \
     --samples "$(cat mince_frozen/<benchmark>/<model-label>/subset_samples.json)" \
     --log_samples --output_path eval_test_logs/<model>-<BENCH> --batch_size auto

   example (GSM8K):
   lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
     --tasks gsm8k --num_fewshot 0 --apply_chat_template \
     --samples "$(cat mince_frozen/gsm8k/meta-llama__Llama-3.1-8B-Instruct/subset_samples.json)" \
     --log_samples --output_path eval_test_logs/Llama-3.1-8B-Instruct-GSM8K --batch_size 1

   example (MMLU):
   lm_eval --model hf --model_args pretrained=meta-llama/Llama-3.1-8B-Instruct \
     --tasks mmlu --num_fewshot 5 --apply_chat_template \
     --samples "$(cat mince_frozen/mmlu/meta-llama__Llama-3.1-8B-Instruct/subset_samples.json)" \
     --log_samples --output_path eval_test_logs/Llama-3.1-8B-Instruct-MMLU --batch_size 1

**It is important to reuse the same generation settings as the full benchmark run** to ensure the subset run is equivalent to the full benchmark run.
The only additions are ``--samples`` (restrict to the
frozen subset) and a fresh ``--output_path``.

**5. Validate** — compute drift between the subset run and the full baseline:

.. code-block:: bash

   python validate.py --benchmark gsm8k \
     --subset-results eval_test_logs/<model>-GSM8K/.../results_*.json \
     --baseline-results bf16_test_logs/Llama-3.1-8B-Instruct-GSM8K/meta-llama__Llama-3.1-8B-Instruct/results_*.json

   example (GSM8K):
   python validate.py --benchmark gsm8k \
     --subset-results eval_test_logs/Llama-3.1-8B-Instruct-GSM8K/*/results_*.json \
     --baseline-results bf16_test_logs/Llama-3.1-8B-Instruct-GSM8K/meta-llama__Llama-3.1-8B-Instruct/results_*.json

   example (MMLU):
   python validate.py --benchmark mmlu \
     --subset-results eval_test_logs/Llama-3.1-8B-Instruct-MMLU/*/results_*.json \
     --baseline-results bf16_test_logs/Llama-3.1-8B-Instruct-MMLU/meta-llama__Llama-3.1-8B-Instruct/results_*.json

``validate.py`` is a pure JSON diff and reports per-metric absolute drift =
``|subset - full|``.

Reusing ``n*`` on other models
------------------------------

Steps 1-3 are a **one-time** calibration. Once the subset is frozen and you have
confirmed its drift is within budget, ``subset_samples.json`` is just a fixed
list of item IDs — so other models of the same family can be scored on those same items by
repeating steps 4-5 with a different ``pretrained=``. That is where the eval-time
saving is realized: every downstream variant costs ``n*`` items instead of the
full benchmark.

For a worked example that reuses ``n*`` to evaluate a Quark-quantized model, see
:doc:`example_quark_torch_mince_quantized`.


Tests
-----

The MINCE package and its unit tests live under ``quark/contrib/mince``. Run the tests
from the repo root with:

.. code-block:: bash

   python -m pytest quark/contrib/mince/test -q
