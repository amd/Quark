.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Accelerate with Settings
========================

This page summarizes the settings that most directly affect ONNX quantization
throughput. Use it together with :doc:`Accelerate with GPUs <gpu_usage_guide>`
when you want faster calibration or fast finetuning without changing the model
itself.

Calibration throughput settings
-------------------------------

- ``ExecutionProviders`` controls where calibration runs. Use GPU providers
  such as ``ROCMExecutionProvider`` or ``CUDAExecutionProvider`` when available
  to accelerate ONNX Runtime execution.
- ``CalibWorkerNum`` controls how many workers collect calibration data. Higher
  values can reduce runtime, but they also increase memory usage.

Fast finetuning throughput settings
-----------------------------------

- ``mem_opt_level`` controls the memory-versus-speed trade-off in AdaRound and
  AdaQuant workflows. Lower values are faster; higher values reduce memory
  consumption.
- ``num_workers`` increases host-side data loading parallelism when fast
  finetuning.
- ``pin_memory`` can improve host-to-device transfer throughput when the data
  pipeline is feeding a GPU.
- ``use_gds`` enables GPU Direct Storage for supported NVIDIA-based setups when
  ``optim_device`` is CUDA and ``mem_opt_level`` is 2.

GDS for fast finetuning
-----------------------

GPU Direct Storage (GDS) can reduce fast finetuning latency when the workflow
becomes limited by disk-to-GPU data movement instead of GPU compute. This is
most relevant for ``AdaRound`` and ``AdaQuant`` runs that use
``mem_opt_level = 2``, where layer data is stored on disk to lower memory
usage.

GDS changes the data path rather than the algorithm itself, so it can improve
throughput without changing quantization behavior or accuracy targets.

- Use GDS only on supported NVIDIA GPU environments.
- Set ``use_gds = True`` in the finetuning algorithm config.
- GDS requires ``optim_device`` to use CUDA and works with
  ``mem_opt_level = 2``.
- For setup requirements, bottleneck analysis, and an example configuration,
  see :doc:`Speed Up Fast Finetuning with GDS <gds_finetuning_speedup>`.

Practical guidance
------------------

- Start with a moderate ``CalibWorkerNum`` and increase it only while memory
  usage remains healthy.
- If fast finetuning becomes input-pipeline bound, raise ``num_workers`` and
  enable ``pin_memory`` before increasing algorithm iterations.
- Use ``mem_opt_level = 1`` when you want a balanced default. Switch to
  ``mem_opt_level = 2`` when memory pressure is the limiting factor.

Related references
------------------

- :doc:`Accelerate with GPUs <gpu_usage_guide>`
- :doc:`AdaQuant and AdaRound <accuracy_algorithms/ada>`
- :doc:`Full List of Quantization Config Features <appendix_full_quant_config_features>`
- :doc:`Speed Up Fast Finetuning with GDS <gds_finetuning_speedup>`
- :doc:`ONNX Troubleshooting <onnx_troubleshooting>`

.. toctree::
   :hidden:

   gds_finetuning_speedup
