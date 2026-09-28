MiniMax-M3 Large-Model Best Recipe
==================================

This directory is a sanitized, portable public baseline for
``amd/MiniMax-M3-MXFP4``. It is not a publication of an internal tuned recipe
and makes no quality or throughput claim.

Files
-----

* ``minimax_m3_mxfp4_adapter.yaml`` records only public model structure,
  MXFP4/AITER requirements, chat template, and target tensor parallelism.
  Both this file and ``large_model_profile.yaml`` are checked against a fixed
  key set, so a field the runner does not read is rejected rather than sitting
  there looking like a setting. Draft architecture and auxiliary layers are
  derived from the target's published config; the recipe holds what the
  pipeline acts on.
* ``large_model_profile.yaml`` maps the adapter, generic TorchSpec config, and
  replaceable domain manifests.
* ``domain_manifest.smoke.yaml`` uses tiny repository-authored MIT-licensed
  inputs for plumbing validation.
* ``domain_manifest.full.example.yaml`` contains generic domain names and
  placeholders only; replace every source and record its license/provenance.
* ``run.sh`` selects this profile and delegates to the common runner.

Before a real run, review the public model's current vLLM compatibility, replace
the full manifest sources, and choose workload settings for your hardware. The
runner materializes prompts deterministically and regenerates assistant
responses from the target model on-policy. No private data mix, searched layer
IDs, vocabulary map, checkpoint cadence, benchmark threshold, or internal
result is included here.

Run the small smoke profile with:

.. code-block:: bash

   python3 -m quark.experimental.speculative_decoding.run \
       --config recipes/eagle3_mxfp4_moe.yaml \
       --base_model amd/MiniMax-M3-MXFP4

The Qwen3-8B CLI remains the validated default. This large-model baseline has
been run end to end on one eight-GPU node; see the ``Results`` section of the
EAGLE-3 large-model best recipe documentation for what it measured and cost.
