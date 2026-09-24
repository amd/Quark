.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Speculative Decoding
====================

.. warning::

   Speculative decoding is an **experimental** feature. It lives in
   ``quark.experimental.speculative_decoding`` and its APIs may change or be
   replaced in a future release.

Autoregressive decoding emits one token per target-model forward pass, which is
the main latency and throughput bottleneck for large models. Speculative decoding
accelerates generation **without changing the output**: a small **draft** model
proposes several candidate next tokens, and the **target** model verifies them all
in a single forward pass. Tokens that match what the target would have produced
are accepted together; generation continues from the first mismatch. Because every
emitted token is still verified by the target, the output distribution is
preserved (the method is lossless).

The key quality metric is the **acceptance length (AL)** -- the average number of
tokens accepted per target verification step. ``AL = 1`` means no acceleration; a
strong draft on real workloads typically lands around 2.5 to 3.5.

AMD Quark currently implements :doc:`EAGLE-3 <eagle3>`, covering the whole flow
end to end: on-policy data synthesis, cold-start draft training, export to a
vLLM-loadable draft, deployment on vLLM, and acceptance-length and throughput
evaluation.

Use the :doc:`Qwen3-8B Quick Start <eagle3_quick_start>` for the validated
example. Use the :doc:`Large-Model Best Recipe <eagle3_best_recipe>` for the
public adapter and generic domain-manifest workflow.

.. toctree::
   :hidden:
   :maxdepth: 1

   EAGLE-3 <eagle3>
   Qwen3-8B Quick Start <eagle3_quick_start>
   Large-Model Best Recipe <eagle3_best_recipe>
