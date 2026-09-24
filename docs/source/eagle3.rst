.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

EAGLE-3
=======

AMD Quark provides an AMD/ROCm-native, Quark-integrated pipeline to **train
EAGLE-3 draft models from scratch**, synthesize on-policy training data, export a
**vLLM-loadable** draft, deploy it on vLLM, and evaluate acceptance length and
throughput. It trains the draft, aligned to a Quark (optionally quantized)
target, using EAGLE-3's low/mid/high-level hidden-state features and a
training-time-test (TTT) objective.

.. note::

   **Choose an EAGLE-3 workflow:**

   * **Validated Qwen3-8B path.** The
     :doc:`end-to-end example on AMD Instinct <eagle3_quick_start>` is the fully
     GPU-validated, reproducible recipe (data, training, serving, and a measured
     speedup). The default YAML/CLI on this page packages and drives that same
     `TorchSpec <https://github.com/lightseekorg/TorchSpec>`_ streaming path:
     about 150k target-generated samples, 8-GPU training, vLLM export, and a
     matched baseline/speculative benchmark.

   * **Portable large-model baseline.** The
     :doc:`EAGLE-3 Large-Model Best Recipe <eagle3_best_recipe>` provides a
     model-adapter configuration and domain-manifest workflow for adapting the
     pipeline to a large, quantized target. It is a general starting point:
     validate its smoke profile first, then provide a full manifest before
     starting a full run.

   TorchSpec is an MIT-licensed, open-source distributed training framework
   maintained by the LightSeek Foundation; Quark uses its streaming EAGLE-3
   trainer for the validated multi-GPU execution path.

   The Python API (``qsd.convert`` / ``qsd.train``) remains the Quark-native
   single-GPU reference implementation. Its export path (``export_hf`` /
   ``convert_to_vllm``) emits a
   draft in vLLM's ``LlamaForCausalLMEagle3`` format -- architecture, weight key
   names, and shapes are validated byte-for-byte against a known-good draft -- so
   exports load directly into a vLLM speculative serve. Its native training path
   is **format-validated only**. Select it explicitly with
   ``execution.backend=native``.
   See :ref:`eagle3-extraction-regimes` and :ref:`eagle3-status-limitations`.

Background: EAGLE-3 and EAGLE-3.1
---------------------------------

EAGLE is a family of *feature-level* draft models. Instead of bolting on an
unrelated small language model, an EAGLE draft is trained against the target
model's own internal representations, which is what makes its proposals so
acceptable to the verifier. Across generations it has improved draft quality and
acceptance rate (EAGLE, EAGLE2, EAGLE3).

**EAGLE-3** combines **low-, mid-, and high-level hidden states** from the target
with a **training-time-test (TTT)** objective: the draft is unrolled several steps
during training so it learns to propose multiple tokens in a row, matching how it
is used at inference.

This package trains an **EAGLE-3.1** draft, a small refinement of EAGLE-3 that
adds extra normalization inside the draft head to stabilize cold-start
(from-scratch) training. The draft is a single decoder layer and exports to vLLM's
``LlamaForCausalLMEagle3`` serving format.

Pipeline
--------

.. code-block:: text

              quant/       extraction/
                 │             │
                 ▼             ▼
   data/ ──▶ convert() ──▶ training/ ──▶ export/ ──▶ eval/

``quant/`` (a Quark MXFP4/FP8 target used as the verifier and hidden-state source)
feeds ``convert()``; ``extraction/`` (target hidden states, online / offline /
streaming) feeds ``training/``. Each stage is described in
:ref:`eagle3-module-layout` below.

Requirements
------------

- Python 3.11+, PyTorch (ROCm build), ``transformers``, ``safetensors``,
  ``pyyaml``, ``tqdm``.
- The default CLI requires one node with 8 AMD Instinct GPUs, Docker with access
  to ``/dev/kfd`` and ``/dev/dri``, internet access, and about 45 GB of free
  disk. Heavy dependencies run in the prepared ROCm image.
- For ``online`` / ``streaming`` extraction: a vLLM-ROCm build whose worker
  exposes the hidden-state extraction hook.
- For quantized targets: ``quark.torch`` (MXFP4/FP8 verifier loading).
- Data synthesis (``data/synth.py``) depends only on the standard library and
  ``tqdm``.

Quickstart: Python API
----------------------

The API surface is intentionally small:

.. code-block:: python

   import quark.experimental.speculative_decoding as qsd

   # 1) target -> trainable EAGLE-3 spec model (frozen target verifier + fresh draft)
   spec = qsd.convert(
       "Qwen/Qwen3-8B",
       spec_cfg={
           "method": "eagle3",
           "aux_hidden_layers": [2, -3, -1],   # low / mid / high target layers
           "ttt_length": 7,
           "eagle_architecture_config": {"num_hidden_layers": 1, "fc_norm": True, "norm_output": True},
       },
   )

   # 2) cold-start training (online extraction by default)
   qsd.train(spec, data_cfg=qsd.DataConfig(train="data/onpolicy.jsonl", eval="data/eval.jsonl"),
             train_cfg=qsd.TrainConfig(num_epochs=1, output_dir="ckpts/eagle3"))

   # 3) export a vLLM-loadable EAGLE-3 draft
   qsd.export_hf(spec, "release/draft_hf")

Data utilities and evaluation:

.. code-block:: python

   qsd.data.synthesize(prompts, target_endpoint, served_model_name, out)   # on-policy data
   qsd.data.calibrate_draft_vocab(tokenizer, data, draft_vocab_size=32000) # optional lm_head compression
   qsd.eval.acceptance(draft_hf, target, dataset, target_endpoint=...)     # served AL@NST
   qsd.eval.throughput_sweep(endpoint, served_model_name, conc=[1,8,32], isl_osl=[(1024,1024)])

Serve the exported draft with vLLM:

.. code-block:: bash

   vllm serve <target> --speculative-config \
       '{"method":"eagle3","model":"release/draft_hf","num_speculative_tokens":3}'

Quickstart: YAML recipe and CLI
-------------------------------

A working `Docker environment <https://docs.docker.com/engine/install/>`_ is
required before running the CLI; verify that ``docker info`` succeeds.

Prepare the validated environment once, then launch the packaged Qwen3-8B
recipe:

.. code-block:: bash

   python3 -m quark.experimental.speculative_decoding.setup \
       --base_model Qwen/Qwen3-8B

.. code-block:: bash

   python3 -m quark.experimental.speculative_decoding.run --base_model Qwen/Qwen3-8B

The CLI performs on-policy data generation, 8-GPU training, export, and measured
serving validation. See the :doc:`complete EAGLE-3 Quick Start
<eagle3_quick_start>` for the explicit YAML form, quick/full profiles, hardware
requirements, runtime cost, output layout, benchmark interpretation, and
troubleshooting.

.. _eagle3-module-layout:

Module layout
-------------

.. list-table:: Modules of ``quark.experimental.speculative_decoding``
   :header-rows: 1
   :widths: 20 80

   * - Path
     - Purpose
   * - ``config.py``
     - Typed configs (``SpecConfig``, ``DataConfig``, ``TrainConfig``,
       ``InferenceConfig``, ``QuantConfig``, and so on) -- the single source of
       truth for the API.
   * - ``convert.py``
     - ``convert(target, spec_cfg) -> SpecModel`` (frozen target verifier plus
       EAGLE-3 draft).
   * - ``run.py``
     - Config-driven entry point. Defaults to the validated TorchSpec runner;
       ``execution.backend=native`` selects the reference trainer.
   * - ``data/``
     - On-policy ``synthesize``, dataset and tokenization helpers, draft-vocab
       calibration (``d2t`` / ``t2d``).
   * - ``extraction/``
     - Target hidden-state extraction: ``online`` / ``offline`` / ``streaming``.
   * - ``eagle/``
     - EAGLE-3 draft ``modeling_eagle3.py`` (vLLM-compatible), ``config.py``, and
       TTT ``losses.py``.
   * - ``training/``
     - Cold-start TTT ``trainer.py`` and learning-rate ``schedule.py``.
   * - ``export/``
     - ``export_hf`` (vLLM ``LlamaForCausalLMEagle3``) and ``convert_to_vllm``
       (folds in verifier metadata).
   * - ``eval/``
     - ``acceptance`` (served AL from the vLLM ``/metrics`` endpoint) and
       ``throughput_sweep`` (per-GPU tokens/s).
   * - ``quant/``
     - Quark quantization integration (quantized target as verifier; optional
       draft PTQ/QAT hook).
   * - ``recipes/``
     - Ready-to-run YAML recipes.
   * - ``utils/``
     - Logging, checkpointing, ROCm environment, and extraction watchdog.

.. _eagle3-extraction-regimes:

Extraction regimes
------------------

All three regimes deliver the same training signal (the target's auxiliary hidden
states per position). They differ in *where* the target runs and *how* features
reach the trainer.

.. list-table:: Hidden-state extraction regimes
   :header-rows: 1
   :widths: 15 35 50

   * - Regime
     - When to use it
     - Status
   * - ``online``
     - Target co-located with the trainer (for example Qwen3-8B).
     - Implemented; the recommended path for the library.
   * - ``offline``
     - Dump hidden states to disk, then train the draft alone.
     - Provides ``dump`` / ``load_shard`` primitives; not yet wired into the
       training loop.
   * - ``streaming``
     - A live vLLM serve streams hidden states (large MoE targets).
     - Interface and environment checks only; no concrete transport is
       implemented. Use the TorchSpec-based example for scale.

Quark quantization integration
------------------------------

A supported setup is a **quantized target (MXFP4/FP8) plus a
speculative draft**. ``quant/integrate.py`` loads a Quark-quantized target as the
verifier and hidden-state source (via
``quark.torch.import_model_from_safetensors``, falling back to plain Hugging Face
loading), and exposes a hook to optionally quantize the draft itself. This is the
Quark-specific angle of the package. See ``recipes/eagle3_mxfp4_moe.yaml`` and
the :doc:`large-model recipe template <eagle3_best_recipe>`.

Evaluation
----------

- **Acceptance length (AL)** is the quality and diagnostic metric.
  ``eval/acceptance.py`` drives a running vLLM speculative serve and reads the
  real AL from the engine's spec-decode Prometheus ``/metrics`` endpoint. Served
  AL is the metric to trust; in-training proxies overestimate it.
- **Throughput** is the deployment and SLA metric. ``eval/throughput_sweep.py``
  sweeps concurrency against (ISL, OSL) pairs and reports tokens/s per GPU.

Recipes
-------

.. list-table:: Packaged YAML recipes
   :header-rows: 1
   :widths: 35 40 25

   * - Recipe
     - Target
     - Extraction
   * - ``recipes/eagle3_default.yaml``
     - Validated Qwen3-8B full-vocabulary recipe.
     - ``streaming`` (TorchSpec; 4 extraction + 4 training GPUs)
   * - ``recipes/eagle3_mxfp4_moe.yaml``
     - Portable large-model baseline for ``amd/MiniMax-M3-MXFP4``.
     - ``streaming`` (TorchSpec TP profile)

Relationship to the example and TorchSpec
-----------------------------------------

- The :doc:`end-to-end example <eagle3_quick_start>` contains the canonical
  runner assets used by the default Python CLI. Those assets are included in
  wheels, so the CLI works from any directory after installation.
- This package is the **Quark-native reference implementation** of the same
  pipeline. Its export format matches vLLM, and its distinguishing feature is
  Quark quantization integration.
- The :doc:`large-model recipe template <eagle3_best_recipe>` documents the
  model-adapter configuration and large-model workflow.

.. _eagle3-status-limitations:

Status and limitations
----------------------

- **Experimental.** APIs may change.
- The draft is a **single EAGLE-3 layer**: ``num_hidden_layers`` must be ``1``,
  because the vLLM ``LlamaForCausalLMEagle3`` serving format is single-layer.
  Other values raise an error.
- ``online`` extraction is the implemented training path. ``offline`` is partial
  and ``streaming`` is an interface stub, as described in
  :ref:`eagle3-extraction-regimes`.
- Export is byte-validated against vLLM, but the native trainer is not yet
  scale-validated here. Use the example for a reproduced speedup.

.. toctree::
   :hidden:
   :maxdepth: 1

   Quick Start <eagle3_quick_start>
   Large-Model Best Recipe <eagle3_best_recipe>
