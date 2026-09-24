Multi-Model Calibration
=======================

The default path is the one in
:doc:`example_quark_torch_mince`: calibrate ``n*`` on a single bf16 model, freeze
the subset, then reuse it on that model's variants — as
:doc:`example_quark_torch_mince_quantized` does for a Quark-quantized checkpoint.
That is the recommended starting point, and it is what the shipped CLIs are
built for.

This page is for teams who want to go further and calibrate on **several** bf16
models at once, so the frozen subset is not tied to the behavior of one model.
Multi-model calibration is validated in the MINCE paper:
`arxiv.org/abs/2606.22826 <https://arxiv.org/pdf/2606.22826>`_.

Why calibrate on more than one model
------------------------------------

``n*`` is chosen from a drift curve, and that curve is a property of the
*calibration model's* per-item scores, not of the benchmark alone. A subset size
that is tight for one model can be loose for another, so a single-model ``n*``
can be optimistic.

That is measurable. The numbers on this page come from two small bf16 models,
``facebook/opt-125m`` and ``EleutherAI/pythia-160m``, each evaluated on the full
CommonsenseQA split (1221 items) zero-shot, with no chat template.
Sizing each model separately, then both together, gives:

.. code-block:: text

   P95 |drift| at each candidate subset size, in percentage points

            N    opt-125m   pythia-160m   worst case
          100     7.02 pp       7.18 pp      7.18 pp
          200     5.02 pp       4.82 pp      5.02 pp
          300     3.98 pp       3.85 pp      3.98 pp
          400     3.23 pp       3.18 pp      3.23 pp
          ...
   n* (items)         400           300          400

Calibrating on ``pythia-160m`` alone selects ``n* = 300``. That subset size does
not hold up for ``opt-125m``, whose marginal gain has not yet flattened at 300.
Calibrating on both selects 400 — the size that satisfies *both* models.

MINCE does this by taking the worst case: at each candidate N, the reported P95
``|drift|`` is the maximum across every calibration model **and** every metric.
Adding models can only hold ``n*`` the same or push it up, never down, so a
multi-model ``n*`` is a conservative envelope rather than an average.

How multi-model calibration works
---------------------------------

Three properties matter:

- **The subset draws are identical for every model.** ``run_sizing`` resets the
  RNG to ``seed`` before each model, so all models are compared on the same
  Monte-Carlo subsets. Differences in the drift curve come from the models, not
  from sampling noise between them.
- **Every calibration model must have logs for every item.** The loader builds
  one item list and attaches each model's scores to it, so all models must have
  been evaluated on the same full benchmark. The ``total_items`` guard enforces
  this.
- **Freezing is unchanged.** The frozen indices depend only on
  ``(total_items, seed)``, so a subset frozen from a multi-model calibration is
  byte-identical to one frozen from a single-model calibration at the same
  ``n`` and ``seed``. What changes is the drift table recorded alongside it,
  which now reports every calibration model.

Step 1 — Generate bf16 logs for each calibration model
------------------------------------------------------

Run step 1 of :doc:`example_quark_torch_mince` once per
calibration model, into a separate ``--output_path``. Keep the generation
settings identical across models so the curves are comparable:

.. code-block:: bash

   for MODEL in <bf16_model_a> <bf16_model_b> <bf16_model_c>; do
     lm_eval --model hf --model_args pretrained=$MODEL \
       --tasks gsm8k --num_fewshot 0 --log_samples \
       --output_path bf16_logs/calib/$(basename $MODEL)-GSM8K --batch_size auto
   done

Every model must be evaluated on the **full** benchmark — this is the
calibration step.

Step 2 — Size across all calibration models
-------------------------------------------

The MINCE package supports multi-model sizing today: ``run_sizing`` takes a
mapping of label to logs directory and accepts any number of entries. Please note
that ``size.py`` and ``freeze.py`` currently accept a **single** logs directory
through the CLI, so multi-model sizing goes through the package API below.

Below are snippets showing the API calls that are relevant. Please combine these
with the code in ``size.py`` and ``freeze.py`` to create the full multi-model
sizing and freezing flow.

.. code-block:: python

   from quark.contrib.mince.config import BENCHMARKS
   from quark.contrib.mince.montecarlo import run_sizing
   from quark.contrib.mince.selection import DEFAULT_TAU, select_n_star

   benchmark = BENCHMARKS["gsm8k"]
   model_paths = {
       "model_a": "bf16_logs/calib/model_a-GSM8K/<hf_dir>",
       "model_b": "bf16_logs/calib/model_b-GSM8K/<hf_dir>",
       "model_c": "bf16_logs/calib/model_c-GSM8K/<hf_dir>",
   }

   bundle = run_sizing(benchmark, model_paths, B=10000, seed=42)
   worst = bundle["worst_p95_by_n"]
   for n in sorted(worst):
       print(f"n={n:5d}   worst-case P95 |drift| = {worst[n] * 100:5.2f} pp")

   n_star = select_n_star(worst, tau=DEFAULT_TAU)
   print(f"n* = {n_star}  ({n_star / benchmark.total_items * 100:.1f}% of full)")

Step 3 — Freeze at the chosen ``n*``
------------------------------------

Freezing does not change. ``build_frozen_subset`` takes the same
``model_paths`` mapping, and the artifact records every calibration model:

.. code-block:: python

   import json, os
   from quark.contrib.mince.subset import build_frozen_subset, build_samples

   SEED = 42
   artifact, items = build_frozen_subset(benchmark, model_paths, n_star, seed=SEED)
   samples = build_samples(artifact)   # the lm-eval --samples map

   print(artifact["models"])           # ['model_a', 'model_b', 'model_c']
   print(artifact["drift"])            # per model, per metric: full / subset / |drift|

   # Same output layout freeze.py uses, so the rest of the flow is unchanged.
   out_dir = os.path.join("mince_frozen", benchmark.name, f"{len(model_paths)}models")
   os.makedirs(out_dir, exist_ok=True)

   with open(os.path.join(out_dir, f"frozen_subset_seed{SEED}.json"), "w") as f:
       json.dump(artifact, f, indent=2)
   with open(os.path.join(out_dir, "subset_samples.json"), "w") as f:
       json.dump(samples, f, indent=2)

``freeze.py`` labels its output directory with the model name for a single model
and ``<k>models`` for several, which is what the path above reproduces.

Read the per-model drift table rather than a single number. On the two-model
CommonsenseQA calibration above, freezing at ``n = 400`` gives:

.. code-block:: text

   opt-125m      acc  full= 19.98  subset= 18.75  |drift|= 1.23 pp
   pythia-160m   acc  full= 19.82  subset= 17.00  |drift|= 2.82 pp

Both are inside the 3.23 pp worst-case P95 budget at that N.

Step 4 — Score downstream models
--------------------------------

Scoring is identical to steps 4 and 5 of :doc:`example_quark_torch_mince`: pass
``subset_samples.json`` to ``lm_eval --samples`` with the same generation
settings as the calibration runs.

The difference is in how far you can reasonably carry the subset. A subset
calibrated on one model is safest on that model's own variants — quantized,
pruned, or distilled versions of it — which is the case
:doc:`example_quark_torch_mince_quantized` walks through. A subset whose ``n*``
had to satisfy several diverse calibration models has been held to a drift
budget across all of them, which is what makes it reasonable to reuse on
downstream models beyond a single family. See the paper for the experiments
supporting this.

Choosing the calibration set
----------------------------

- **Spread, not count.** Models that behave alike produce nearly identical drift
  curves and a worst case that barely moves. Vary what you expect to vary
  downstream: family, scale, and instruction tuning.
- **Cost is per model, once.** Each calibration model needs one full-benchmark
  run. That is the price of the sizing step, paid once, against every subsequent
  subset run costing ``n*`` items.
- **Expect ``n*`` to grow.** A larger, more diverse calibration set means a
  larger subset and a smaller saving. That is the trade being made: breadth of
  reuse in exchange for eval-time reduction.
