.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

LayerwisePercentile
===================

``CalibMethod.LayerwisePercentile`` is an accuracy-oriented calibration method
that searches percentile candidates per layer instead of applying one fixed
percentile everywhere. It is useful when a model is sensitive to activation
range selection and default calibration methods do not produce stable results.

Recent memory improvements
--------------------------

Recent Quark ONNX pipeline updates significantly reduced the peak RSS memory
footprint of ``LayerwisePercentile``. In internal benchmarking, some
activation-heavy workloads saw roughly 20x to 66x lower peak RSS, turning
``LayerwisePercentile`` from a method that could previously require extremely
large memory budgets into one that fits standard workstation-class hardware.

The main changes are:

- Chunked processing during layerwise percentile calibration, replacing bulk
  loading of all calibration data with small-chunk iteration and buffer reuse.
- Selective Calibration Propagation (SCP), where distribution-preserving ops
  such as ``Reshape`` and ``Transpose`` inherit calibration ranges instead of
  recomputing them.
- Improved disk caching that preserves original tensor precision and avoids
  hidden memory growth caused by dtype promotion on reload.
- Earlier release of intermediate resources, including inference sessions and
  other large temporary objects.

When to use it
--------------

- Try it when ``MinMax`` or a single ``Percentile`` setting produces noticeable
  accuracy degradation.
- It is especially useful for models where different layers need different
  clipping behavior.
- With the recent memory reductions, it is now practical to evaluate
  ``LayerwisePercentile`` on standard hardware instead of ruling it out only
  because of historical memory cost.
- It can require more calibration work than simpler methods, so expect a
  runtime trade-off in exchange for better calibration quality.

Key settings
------------

- Set ``calibration_method`` to ``CalibMethod.LayerwisePercentile`` in the
  tensor config you want to calibrate.
- Use ``LWPMetric`` to choose the metric used to compare percentile candidates.
- Use ``PercentileCandidates`` to control which percentile values are explored
  during the layerwise search.
- Use ``CalibWorkerNum`` to parallelize calibration when you have enough CPU
  and memory headroom.
- Use ``CalibPassthroughOpTypes`` to enable Selective Calibration Propagation
  on distribution-preserving ops. The recommended list is ``["Reshape",
  "Transpose", "MaxPool", "Split", "Slice", "Squeeze", "Unsqueeze",
  "Gather"]``.

Memory and disk behavior
------------------------

- ``CalibOptimizeMem`` reduces calibration memory pressure and is a good
  default when activation caching becomes expensive.
- ``CalibOptimizeDisk`` is specific to ``LayerwisePercentile`` and avoids
  retaining intermediate activation tensors in memory or on disk, recomputing
  them instead when needed.
- ``CalibPassthroughOpTypes`` reduces redundant calibration work on
  distribution-preserving ops and can further lower peak RSS.
- ``TmpDir`` lets you move temporary calibration files away from a small
  default system temp directory when disk space is limited.

Takeaway
--------

For most models, you can now choose ``LayerwisePercentile`` based on accuracy
needs rather than assuming it will exceed available memory. Start with the
default memory-saving settings, then tune percentile candidates and worker
count as needed.

Related references
------------------

- :doc:`Full List of Quantization Config Features <appendix_full_quant_config_features>`
- :doc:`ONNX Troubleshooting <onnx_troubleshooting>`
- :doc:`Profile Quantization Latency and Memory <tutorial_profiling>`
