Qwen3-8B EAGLE-3 Quick Start
============================

This directory owns the validated Qwen3-8B TorchSpec configuration. The
``run.sh`` selector delegates all orchestration to ``../common/run_all.sh``.

The public CLI contract remains:

.. code-block:: bash

   python3 -m quark.experimental.speculative_decoding.setup \
       --base_model Qwen/Qwen3-8B

   python3 -m quark.experimental.speculative_decoding.run \
       --base_model Qwen/Qwen3-8B

   python3 -m quark.experimental.speculative_decoding.run \
       --config recipes/eagle3_default.yaml \
       --base_model Qwen/Qwen3-8B \
       training.output_dir=ckpts/qwen3-8b-eagle3 training.num_epochs=2

For direct example use:

.. code-block:: bash

   bash run.sh --base_model Qwen/Qwen3-8B

``PROFILE=quick`` remains a plumbing check; the default ``full`` workload and
all Python CLI override semantics are unchanged.

Resuming an interrupted run
---------------------------

Training is the long phase — roughly 4-5 hours for the two validated epochs —
so it resumes by default. Re-issue the same command and the run picks up from
the newest checkpoint under ``<training.output_dir>/training/checkpoints``,
restoring model, optimizer, LR schedule and RNG:

.. code-block:: bash

   python3 -m quark.experimental.speculative_decoding.run \
       --base_model Qwen/Qwen3-8B \
       training.output_dir=ckpts/qwen3-8b-eagle3

Only checkpoints written by the same resolved config are eligible. The config
covers the target, draft architecture, topology and the train/eval data bytes,
so changing any of them retrains from scratch rather than resuming onto
mismatched weights. Checkpoints left by a *different* config are reported and
ignored. Set ``AUTO_RESUME=0`` to force a cold start.

This also covers the automatic retries: if the training actor dies mid-run,
each attempt restarts from the newest checkpoint instead of step 0.

Data generation is cached separately and is already reused whenever finalized
train/eval files exist, so a resumed run does not regenerate it.
