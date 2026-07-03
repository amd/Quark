.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Memory and Disk Friendly Settings
=================================

This page summarizes the settings and recent pipeline improvements that help
large ONNX quantization jobs fit within limited memory or temporary-disk
budgets. Use these settings when the default workflow is correct but too
expensive for your current machine.

Recent pipeline improvements
----------------------------

Recent Quark ONNX updates significantly reduced peak RSS for both ``MinMax``
and, especially, ``LayerwisePercentile`` calibration. In internal benchmarking,
some activation-heavy workloads previously pushed ``LayerwisePercentile`` into
very large peak RSS territory. After the optimization work, those same
workloads fit within typical workstation memory budgets.

The main changes are:

- Chunked processing for layerwise percentile calibration, replacing bulk
  loading with small-chunk iteration and buffer reuse.
- Selective Calibration Propagation (SCP), where passthrough ops inherit
  calibration ranges instead of recomputing them.
- Improved disk caching that preserves original tensor precision and avoids
  hidden memory inflation after reload.
- Earlier release of intermediate resources so the high-water mark drops as
  soon as large temporary objects are no longer needed.

In practice, this means calibration method choice is much more accuracy-driven
than memory-driven for most models.

Calibration-focused settings
----------------------------

- ``CalibOptimizeMem`` reduces calibration memory pressure and is a strong
  default for large models or large calibration datasets.
- ``CalibOptimizeDisk`` is particularly useful with
  ``CalibMethod.LayerwisePercentile`` because it avoids caching intermediate
  activation tensors in memory or on disk, recomputing them only when needed.
- ``CalibPassthroughOpTypes`` enables Selective Calibration Propagation for
  distribution-preserving ops such as ``Reshape``, ``Transpose``, ``MaxPool``,
  ``Split``, ``Slice``, ``Squeeze``, ``Unsqueeze``, and ``Gather``.
- ``CalibWorkerNum`` can reduce calibration runtime, but higher values also
  raise memory pressure because each worker needs its own resources.
- ``TmpDir`` moves temporary calibration files to a directory with more
  available space when the default temp filesystem is too small.

Fast finetuning memory settings
-------------------------------

- ``mem_opt_level`` in AdaRound and AdaQuant controls how aggressively
  intermediate data is cached. ``0`` is fastest and uses the most memory,
  ``1`` is the default balance, and ``2`` minimizes memory at the cost of more
  disk activity and longer runtimes.
- If ``mem_opt_level = 2`` is too slow, combine it with data-loader tuning such
  as ``num_workers`` and ``pin_memory`` to recover some throughput.
- On supported NVIDIA environments, ``use_gds = True`` can improve the
  disk-backed ``mem_opt_level = 2`` finetuning path by reducing storage-to-GPU
  transfer overhead. See :doc:`Speed Up Fast Finetuning with GDS <gds_finetuning_speedup>`.

Practical guidance
------------------

- If a model previously failed with ``LayerwisePercentile`` because of memory,
  retry it on the current pipeline before falling back to a less accurate
  calibration method.
- Use ``CalibOptimizeMem`` first when calibration hits memory limits.
- Keep ``CalibOptimizeDisk`` enabled for ``LayerwisePercentile`` unless disk
  activity is the primary bottleneck.
- Set ``CalibPassthroughOpTypes`` to the recommended passthrough-op list to
  avoid redundant calibration work.
- Keep ``CalibWorkerNum`` moderate on memory-constrained machines.
- Use ``TmpDir`` when the default ``/tmp`` location is too small for cached
  files.
- Use ``mem_opt_level = 2`` only when the lower memory footprint matters more
  than raw speed.

Takeaway
--------

With the recent ONNX calibration optimizations, peak RSS for ``MinMax`` and
``LayerwisePercentile`` is now much closer on most models. In practice, choose
the calibration method for accuracy first, then use the settings above if you
still need to reduce memory or disk pressure further.

Related references
------------------

- :doc:`Full List of Quantization Config Features <appendix_full_quant_config_features>`
- :doc:`LayerwisePercentile <LayerwisePercentile>`
- :doc:`AdaQuant and AdaRound <accuracy_algorithms/ada>`
- :doc:`Speed Up Fast Finetuning with GDS <gds_finetuning_speedup>`
- :doc:`ONNX Troubleshooting <onnx_troubleshooting>`
- :doc:`Profile Quantization Latency and Memory <tutorial_profiling>`
