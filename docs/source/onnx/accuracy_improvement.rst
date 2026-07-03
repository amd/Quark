.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Accuracy Improvement
====================

This page groups the main ONNX accuracy-improvement techniques in AMD Quark.
Use the calibration and algorithm guides below when a default PTQ run does not
meet your accuracy target.

Contents
--------

.. toctree::
   :maxdepth: 1

   LayerwisePercentile <LayerwisePercentile>
   CrossLayerEqualization (CLE) <accuracy_algorithms/cle>
   AdaQuant and AdaRound <accuracy_algorithms/ada>
   AutoSearch <auto_search>
   Mixed Precision <tutorial_mix_precision>
   SmoothQuant <accuracy_algorithms/sq>
   QuaRot (experimental) <accuracy_algorithms/quarot>
