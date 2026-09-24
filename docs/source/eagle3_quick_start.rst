.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Speculative Decoding (EAGLE-3) on AMD Instinct
==============================================

.. note::

   You can get the example code after downloading and unzipping ``amd_quark.zip``
   (refer to :doc:`Installation Guide <install>`). This example and the relevant
   files are available at ``/experimental/speculative_decoding``.

   For the Quark-native library API behind this feature, see
   :doc:`EAGLE-3 <eagle3>`.

.. seealso::

   This page remains the validated Qwen3-8B quick start. For the portable
   large-model baseline and generic domain-manifest workflow, see
   :doc:`EAGLE-3 Large-Model Best Recipe <eagle3_best_recipe>`.

.. warning::

   Speculative decoding is an **experimental** feature. The scripts, flags, and
   measured numbers described here may change in a future release.

Train an **EAGLE-3 draft** for any Hugging Face causal LM from scratch, export it
to a **vLLM-loadable** draft, deploy it with vLLM speculative decoding on ROCm,
and measure the speedup -- end to end, on a single 8x AMD Instinct node. You pick
the target with a single ``--base_model`` flag; the default is ``Qwen/Qwen3-8B``.

This example uses `TorchSpec <https://github.com/lightseekorg/TorchSpec>`_, an
MIT-licensed, open-source distributed training framework maintained by the
LightSeek Foundation. Its proven, vLLM-compatible disaggregated EAGLE-3 trainer
streams target hidden states to the training ranks, and the resulting draft
loads directly into vLLM as ``LlamaForCausalLMEagle3``. Everything runs inside
one ROCm container image you build in step 0.

.. note::

   This is the validated path. The default
   ``quark.experimental.speculative_decoding.run`` CLI drives these same assets
   from an installed wheel. The native
   :doc:`quark.experimental.speculative_decoding <eagle3>`
   Python trainer is a single-GPU reference implementation and is selected only
   with ``execution.backend=native``.

Quick start
-----------

A working `Docker environment <https://docs.docker.com/engine/install/>`_ is
required before running the commands below; verify that ``docker info``
succeeds.

.. code-block:: bash

   # One time: pinned TorchSpec + ROCm image + target model.
   python3 -m quark.experimental.speculative_decoding.setup \
       --base_model Qwen/Qwen3-8B

   # ~5-6 h: full run -> real speedup (Qwen3-8B: AL 2.43, 1.34x).
   python3 -m quark.experimental.speculative_decoding.run \
       --base_model Qwen/Qwen3-8B

   # Equivalent explicit recipe form with output/epoch overrides.
   python3 -m quark.experimental.speculative_decoding.run \
       --config recipes/eagle3_default.yaml \
       --base_model Qwen/Qwen3-8B \
       training.output_dir=ckpts/qwen3-8b-eagle3 training.num_epochs=2

``--base_model`` takes any Hugging Face repository ID (or a local directory) and
defaults to ``Qwen/Qwen3-8B``. The Qwen3-8B recipe is the validated target for
the advisory speedup and served-AL gates. It is an alias for
``model.target_model_path=``;
explicit OmegaConf-style dotlist overrides win over the alias. Packaged
``recipes/...`` paths resolve from the installed wheel and do not depend on the
current directory.

The setup module checks prerequisites first and tells you exactly what is
missing. The runner prints a live heartbeat for the long phases and a final
baseline-versus-speculative speedup table, then leaves the vLLM draft in
``<training.output_dir>/release/draft_hf`` plus a machine-readable
``report.json``. If you press Ctrl-C, it cleans up its GPU containers.

Resuming an interrupted run
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Training is the long phase, so it resumes by default. Re-issue the same command:

.. code-block:: bash

   # Picks up at the newest checkpoint; AUTO_RESUME=0 trains from scratch.
   python3 -m quark.experimental.speculative_decoding.run \
       --base_model Qwen/Qwen3-8B \
       training.output_dir=ckpts/qwen3-8b-eagle3

Model, optimizer, LR schedule and RNG are restored. Only checkpoints from the
same resolved config qualify, so changing the target, topology or data retrains
instead of resuming onto mismatched weights.

Choosing the target model
~~~~~~~~~~~~~~~~~~~~~~~~~

The target is chosen entirely by ``--base_model``. Pass the same Hugging Face
repository ID or local directory to the setup and run commands above.

The draft architecture (width, heads, vocabulary) is **auto-derived** from the
target's ``config.json`` by ``scripts/make_draft_config.py``, so you do not
hand-write a per-model draft config. Two things to know for non-default targets:

- **Chat template.** Set ``CHAT_TEMPLATE=<family>`` to match the target's chat
  format. The default is ``qwen``.
- **Validated target.** The numbers below are for ``Qwen/Qwen3-8B``. Other **dense
  Hugging Face causal LMs** should work with the same flow, but very large, MoE,
  or quantized targets may need tensor-parallelism or memory tuning in
  ``configs/qwen3_8b_eagle3.yaml``.

Pipeline
--------

.. code-block:: text

   prepare prompts -> on-policy data (target generates responses)
      -> cold-start EAGLE-3 training (streaming, 8 GPUs)
      -> convert to vLLM draft -> serve (baseline vs spec) -> measure speedup

Requirements
------------

- A node with **8x AMD Instinct** GPUs (MI300X or MI350X) on ROCm, plus Docker
  with GPU access.
- Internet access, to pull the base vLLM-ROCm image, clone TorchSpec, and
  download the target model. Export ``HF_TOKEN`` first: the model and prompt
  dataset are otherwise fetched unauthenticated and rate-limited.
- **vLLM ROCm v0.28.0** as the container base. Setup builds this for you from
  the pin in ``eagle3/common/docker/Dockerfile.rocm``, so no action is needed;
  ``--base-image`` overrides it if you must move off that release.
- Free disk in two places, on filesystems the container can write (see the
  ``root_squash`` note above):

  - **~120 GB** for the cache (``QUARK_EAGLE3_CACHE``, default
    ``~/.cache/amd-quark/eagle3``) plus the output directory: model (~16 GB for
    Qwen3-8B), on-policy data, and checkpoints (~13 GB each). Setup reports
    available space before downloading.
  - **~36 GB** under the Docker storage root (usually ``/var/lib/docker``) for
    the image. Often a different filesystem, so the check above misses it.

No CUDA or NVIDIA hardware is required. Dependencies (TorchSpec, MIT licensed;
vLLM, Apache-2.0; Mooncake, Apache-2.0) are installed into the image by
``eagle3/common/docker/Dockerfile.rocm``.

Reproduction cost (GPUs and time)
---------------------------------

To reproduce the result below (Qwen3-8B, roughly 150k on-policy samples, 2 epochs,
AL 2.43 and 1.34x speedup) you need **one 8x AMD Instinct node**, measured on 8x
MI350X. All stages use the same 8 GPUs, so a single node is enough end to end.

.. list-table:: Wall-clock cost per stage
   :header-rows: 1
   :widths: 45 20 35

   * - Stage
     - GPUs
     - Wall-clock (8x MI350X)
   * - One-time setup (build image, download Qwen3-8B)
     -
     - ~30-60 min (network-bound)
   * - On-policy data generation (~150k samples)
     - 8 (8 target serves)
     - ~0.5 h
   * - Cold-start EAGLE-3 training (2 epochs, 9,358 steps)
     - 8 (4 train + 4 extract)
     - ~3-3.5 h
   * - Convert checkpoint to vLLM draft
     - 1 (mostly CPU)
     - a few minutes
   * - Serve baseline and speculative, then benchmark
     - 2
     - ~15-30 min
   * - **End to end (after one-time setup)**
     - **8**
     - **~4-5 h**

Numbers are approximate and scale with prompt length, GPU model, and network
bandwidth. More data or more epochs (for a higher AL and speedup) increase the
data-generation and training time roughly linearly.

Step 0: one-time setup
----------------------

Run the setup command from the Quick start section once. It is idempotent:
rerunning it skips completed work. Omit ``--base_model`` to use the default
(``Qwen/Qwen3-8B``).

Step 1: verify the pipeline (fast), then run it for real
--------------------------------------------------------

.. code-block:: bash

   python3 -m quark.experimental.speculative_decoding.run \
       --base_model Qwen/Qwen3-8B execution.profile=quick \
       training.output_dir=ckpts/qwen3-8b-eagle3-quick

The ``quick`` profile uses tiny data on purpose: it validates data generation,
training, conversion, serving, and benchmarking end to end in well under an hour,
so you do not discover a bad config three hours in. Its acceptance length and
speedup are low. If it succeeds, run either full form from the Quick start
section to produce the numbers below.

Both profiles print a live heartbeat during the long phases and a final speedup
and acceptance length (AL) table. The trained vLLM draft lands in
``<training.output_dir>/release/draft_hf``. Full mode runs three matched
benchmark rounds and records ``status=passed`` when median speedup is at least
``1.25x`` and real served AL is at least ``2.35``. Missing either advisory gate
records ``status=failed`` with the measured values and remediation hints, but
does not turn an otherwise completed pipeline into a process error.

Knobs (all optional)
~~~~~~~~~~~~~~~~~~~~

.. list-table:: Python CLI flags and dotlist overrides
   :header-rows: 1
   :widths: 20 55 25

   * - Flag or override
     - Meaning
     - Default (full profile)
   * - ``--base_model``
     - Target Hugging Face repository ID or local directory.
     - ``Qwen/Qwen3-8B``
   * - ``data.chat_template=qwen``
     - TorchSpec chat-template family.
     - ``qwen``
   * - ``execution.profile=full|quick``
     - ``full`` or ``quick``.
     - ``full``
   * - ``data.num_prompts=150000``
     - On-policy samples to generate.
     - ``150000``
   * - ``training.num_epochs=2``
     - Training epochs.
     - ``2``
   * - ``data.eval_size=256``
     - Held-out evaluation prompts (must be a multiple of 8).
     - ``256``
   * - ``benchmark.num_prompts=40``
     - Prompts used in the speedup benchmark.
     - ``40``
   * - ``benchmark.rounds=3``
     - Matched baseline/speculative benchmark repetitions.
     - ``3``
   * - ``training.output_dir=<path>``
     - Run root for checkpoints, draft, logs, and ``report.json``.
     - ``ckpts/eagle3``

For example, append ``training.num_epochs=3 data.num_prompts=300000`` to train
longer on a larger automatically generated dataset. Equivalent low-level
``PROFILE``, ``NUM_PROMPTS``, ``EPOCHS``, ``SKIP_DATA``, ``SKIP_TRAIN``, and
``SKIP_BENCH`` environment variables remain available when invoking
``run_all.sh`` directly from a source checkout.

Step by step: what run_all.sh does
----------------------------------

You do not need this section, because ``run_all.sh`` does it all, but here is each
stage if you want to run or debug them individually.

The commands below spell out the default ``Qwen/Qwen3-8B`` target. Export
``BASE_MODEL`` first if your shell is not inside ``run_all.sh``: the
``scripts/*.sh`` helpers source ``scripts/_common.sh``, which derives ``MODEL``,
``MODEL_NAME``, and ``MODEL_SLUG`` from it, so substitute those into the literal
paths below too.

.. code-block:: bash

   # 0) derive the draft architecture from the target's config.json
   docker run --rm -v "$PWD/models":/models -v "$PWD":/workspace -w /workspace --entrypoint python3 quark-specdec-rocm:latest \
       scripts/make_draft_config.py --target /models/Qwen3-8B --out runtime/draft_config.json

   # 1) prompts + on-policy data: 8 target serves (GPU0-7) regenerate responses
   docker run --rm -v "$PWD":/workspace -w /workspace --entrypoint python3 quark-specdec-rocm:latest \
       scripts/prepare_prompts.py --n 150000 --out data/prompts.jsonl
   for i in 0 1 2 3 4 5 6 7; do GPUS=$i bash scripts/serve_target.sh qsd-gen-$i $((8000+i)) 1; done
   bash scripts/gen_data.sh data/prompts.jsonl data/onpolicy_all.jsonl \
       localhost:8000,localhost:8001,localhost:8002,localhost:8003,localhost:8004,localhost:8005,localhost:8006,localhost:8007
   # (then split into data/onpolicy_{train,eval}.jsonl)

   # 2) cold-start EAGLE-3 training (4 vLLM extraction + 4 FSDP, streaming)
   bash scripts/train.sh /workspace/runtime/active_config.yaml qsd-train

   # 3) convert best checkpoint -> vLLM draft (LlamaForCausalLMEagle3)
   DRAFT_CONFIG=/workspace/runtime/draft_config.json \
       bash scripts/convert.sh training/checkpoints/iter_XXXXXXX release/draft_hf

   # 4) serve baseline + spec, measure speedup + AL
   GPUS=0 bash scripts/serve_target.sh qsd-base 8000 1
   GPUS=1 bash scripts/serve_spec.sh   qsd-spec 8100 release/draft_hf 1 3
   python3 scripts/bench.py --eval data/onpolicy_eval.jsonl --model target \
       --baseline-port 8000 --spec-port 8100 --rounds 3 \
       --min-speedup 1.25 --min-served-al 2.35

Why these choices (lessons baked in)
------------------------------------

- **On-policy data is the number one lever on AL.** ``gen_data.sh`` regenerates
  every response with the target itself. Off-policy "gold" answers lower AL.
- **Full-vocabulary draft** (``draft_vocab_size == vocab_size``) avoids a
  training-time vocabulary-pruning mapping, so ``convert_to_hf --vllm`` produces a
  clean vLLM draft.
- **Cold-start, streaming extraction.** The draft trains from scratch against the
  target's own hidden states, streamed from live vLLM engines, with no
  terabyte-scale disk dump.
- **Trust served AL, not the training metric.** The in-training ``sim_acc_len``
  overestimates; ``scripts/bench.py`` reports the real served AL from vLLM's
  spec-decode ``/metrics`` endpoint.
- **Enough data matters.** On a fast 8B target the draft must be strong enough
  that its acceptance length beats the drafting overhead. Roughly 150k on-policy
  samples and at least 2 epochs give a clear speedup; more data and more steps
  push it higher.

Results
-------

.. list-table:: Measured on 8x MI350X, Qwen/Qwen3-8B
   :header-rows: 1
   :widths: 30 15 25 30

   * - On-policy samples
     - Epochs
     - Served AL
     - Net speedup (concurrency 1)
   * - ~150k
     - 2
     - **2.43**
     - **1.34x**

Measured with roughly 1K-token prompts, ``num_speculative_tokens=3``, and
temperature 0 (baseline 223 tokens/s, EAGLE-3 298 tokens/s), over three matched
benchmark rounds with the packaged ``seed: 43``. Numbers depend on the prompt
mix and hardware; read your own ``bench.py`` output. Speculative decoding helps
most at **low concurrency**.

.. note::

   **More on-policy data and/or more training steps (epochs) raise the acceptance
   length and thus the speedup.** Acceptance length is the ceiling on the speedup,
   so if you want a bigger win, run with a larger ``NUM_PROMPTS`` and/or more
   ``EPOCHS``, for example by appending
   ``data.num_prompts=300000 training.num_epochs=3`` to the Python run command.

Files
-----

Canonical orchestration lives under ``eagle3/common`` and the validated Qwen
assets under ``eagle3/qwen3_8b_quick_start``. Each profile directory holds a
thin ``run.sh``; the Python CLI resolves the profile and calls that wrapper
directly.

.. list-table:: Files in this example
   :header-rows: 1
   :widths: 35 65

   * - Path
     - Purpose
   * - ``eagle3/qwen3_8b_quick_start/run.sh``
     - Selects the validated Qwen profile and delegates to the shared runner.
   * - ``eagle3/common/run_all.sh``
     - Canonical end-to-end orchestration (profiles, live progress, cleanup,
       data, training, export, and benchmark).
   * - ``eagle3/common/scripts/00_setup.sh``
     - Prerequisite checks, clone TorchSpec, build the image, download
       ``--base_model``.
   * - ``eagle3/common/scripts/_common.sh``
     - Shared helpers: derives model paths and slug, preflight, GPU and disk
       checks, cleanup, input identity, and best-checkpoint selection.
   * - ``eagle3/common/docker/Dockerfile.rocm``
     - Builds the ROCm image (vLLM-ROCm, TorchSpec dependencies, Mooncake).
   * - ``eagle3/common/scripts/prepare_prompts.py``
     - Builds a prompt pool from a public dataset.
   * - ``eagle3/common/scripts/make_draft_config.py``
     - Derives the EAGLE-3 draft architecture from the target's ``config.json``.
   * - ``eagle3/common/scripts/serve_target.sh``
     - Serves the target model (data generation and baseline).
   * - ``eagle3/common/scripts/gen_data.sh``
     - On-policy data generation via TorchSpec's ``generate_data``.
   * - ``eagle3/common/scripts/train.sh``
     - Cold-start EAGLE-3 training (TorchSpec, 8 GPUs).
   * - ``eagle3/common/scripts/convert.sh``
     - FSDP checkpoint to vLLM ``LlamaForCausalLMEagle3`` draft.
   * - ``eagle3/common/scripts/serve_spec.sh``
     - Serves target plus draft (speculative decoding).
   * - ``eagle3/common/scripts/bench.py``
     - Baseline versus speculative throughput and served AL.
   * - ``eagle3/qwen3_8b_quick_start/configs/qwen3_8b_eagle3.yaml``
     - TorchSpec training recipe (base recipe; ``run_all.sh`` templates it per
       target).
   * - ``eagle3/qwen3_8b_quick_start/configs/qwen3_8b_eagle3_draft.json``
     - Validated Qwen3-8B draft architecture (other targets are auto-derived).

Generated at run time and Git-ignored in a source-tree run: ``TorchSpec/``,
``models/``, ``data/``, ``training/``, ``runtime/``, ``release/``, and
``cache/``. The installed CLI keeps reusable dependencies and on-policy data
under ``~/.cache/amd-quark/eagle3`` and per-run artifacts under
``training.output_dir``.

Troubleshooting
---------------

- ``image '...' not found`` or ``<model> ... not found`` -- run
  ``python3 -m quark.experimental.speculative_decoding.setup --base_model
  <hf-id>`` first, and use the same ``--base_model`` on the run command.
- ``cannot reach the docker daemon`` -- start Docker and make sure your user is in
  the ``docker`` group.
- ``the container cannot write to ...``, ``Permission denied: '/models/<model>'``,
  or ``error while creating mount source path`` -- the cache or output directory
  is on a share the container's root cannot write (typically NFS with
  ``root_squash``). Move both to local disk; see the note under `Quick start`_.
- ``manifest for vllm/vllm-openai-rocm:... not found`` during the image build --
  the pinned base tag was removed from the registry. Build against a current
  release tag, keeping Python 3.12 and gfx942/gfx950 support::

     python3 -m quark.experimental.speculative_decoding.setup \
         --base_model Qwen/Qwen3-8B \
         --base-image vllm/vllm-openai-rocm:<release-tag>

  ``--base-image`` (or ``VLLM_ROCM_IMAGE``) becomes ``--build-arg
  VLLM_ROCM_IMAGE``, so you do not edit the Dockerfile. Prefer a release tag:
  vLLM prunes ``nightly-*`` tags. An existing local
  ``quark-specdec-rocm:latest`` is reused as-is, so a pruned tag does not
  affect machines that already built the image.
- ``/dev/kfd or /dev/dri is missing`` -- the host has no ROCm GPU access, so
  containers will not see GPUs.
- A serve "did not become ready" -- check ``docker logs <name>`` (for example
  ``qsd-gen-0`` or ``qsd-base``); this is usually an out-of-memory condition or a
  bad ``--max-model-len``.
- ``Memory access fault by GPU node-N ... on address (nil)`` during training,
  always at the same step -- a particular batch in the sampling order trips a
  null GPU pointer in the ROCm training stack. It is not caused by your data
  being malformed, and retrying the same configuration usually reproduces it,
  because the sampling order is fixed by ``training.seed``. Change the seed in
  the training config to reshuffle past it. The packaged Qwen recipe ships
  ``seed: 43`` for this reason: on this dataset ``seed: 42`` faults at step 272
  in roughly four of five runs, while ``seed: 43`` completed cleanly on the
  first attempt. Changing the seed alters the sampling order, so expect
  acceptance length to move by a small amount.
- ``GATE: FAIL`` after benchmarking -- the pipeline and draft completed, but the
  measured speedup or served AL missed its advisory threshold. Inspect
  ``report.json`` and train with more on-policy data or epochs if higher quality
  is required; this condition does not return a process error.
- Stuck, or want to stop -- press Ctrl-C; ``run_all.sh`` removes its own ``qsd-*``
  containers on exit. To clear them manually, run
  ``docker rm -f $(docker ps -aq --filter name=qsd-)``.
- Rerun cheaply -- ``SKIP_DATA=1`` and ``SKIP_TRAIN=1`` reuse prior phases so you
  do not repeat the expensive ones. An interrupted training phase resumes from
  its newest checkpoint on its own; ``AUTO_RESUME=0`` forces a cold start.
