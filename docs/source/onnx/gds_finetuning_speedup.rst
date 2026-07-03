.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Speed Up Fast Finetuning with GDS
=================================

GPU Direct Storage (GDS) can reduce Quark ONNX fast finetuning latency by
moving data directly between storage and GPU memory. This is most useful for
``AdaRound`` and ``AdaQuant`` runs where data movement becomes the bottleneck.

Because of current software support constraints, this workflow is available
only on supported NVIDIA GPU environments.

Background
----------

Fast finetuning algorithms such as ``AdaRound`` and ``AdaQuant`` can recover
quantization accuracy, but they are often one of the most time-consuming parts
of the ONNX quantization pipeline.

When ``mem_opt_level = 2`` is used, fast finetuning reduces memory usage by
storing intermediate layer data on disk and loading it on demand. This lowers
peak memory, but it can also make storage-to-GPU data transfer the limiting
factor for end-to-end throughput.

Identify the bottleneck
-----------------------

Two simple ways to reason about the bottleneck are:

- **Roofline thinking**: If the workflow is limited by memory bandwidth or data
  movement rather than arithmetic throughput, improving the input pipeline can
  matter more than adding compute.
- **GPU utilization**: If GPU compute utilization stays low while the job is
  actively reading data, the workflow is likely input-pipeline-bound rather
  than compute-bound.

In these cases, GDS can be more effective than purely compute-side tuning.

How GDS helps
-------------

Without GDS, data typically moves through a CPU-mediated path:

``storage -> host memory -> CPU-managed pipeline -> GPU memory``

With GDS, the transfer path becomes more direct:

``storage -> GPU memory``

This reduces extra memory copies and CPU overhead, which can:

- lower input latency,
- improve GPU utilization,
- and reduce the performance penalty of ``mem_opt_level = 2``.

Requirements
------------

To enable GDS in Quark ONNX fast finetuning:

- use a supported NVIDIA GPU environment,
- set ``optim_device`` to CUDA,
- set ``mem_opt_level = 2``,
- and enable ``use_gds = True`` in the finetuning algorithm config.

For software installation and environment requirements, refer to the NVIDIA
documentation:
`NVIDIA DALI installation guide <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html>`__.

Example
-------

.. code-block:: python

    from quark.onnx import AdaRoundConfig, Int8Spec, QConfig, QLayerConfig

    adaround_algo = AdaRoundConfig(
        learning_rate=0.1,
        num_iterations=100,
        mem_opt_level=2,
        optim_device="cuda:0",
        use_gds=True,
    )

    quant_config = QConfig(
        global_config=QLayerConfig(
            input_tensors=Int8Spec(),
            weight=Int8Spec(),
        ),
        algo_config=[adaround_algo],
    )

Practical guidance
------------------

- Try GDS when fast finetuning is limited by data movement rather than GPU
  compute.
- GDS is most relevant when ``mem_opt_level = 2`` is already needed for memory
  reasons.
- If the workflow is already compute-bound, GDS may provide little benefit.
- GDS changes the data path only; it does not change the quantization algorithm
  or introduce an accuracy trade-off by itself.
- On unsupported hardware, keep ``use_gds = False`` and tune settings such as
  ``num_workers`` and ``pin_memory`` instead.
- For a summary of all throughput-related settings including ``num_workers``,
  ``pin_memory``, and ``use_gds``, see
  :doc:`Accelerate with Settings <accelerate_with_settings>`.

Related references
------------------

- :doc:`Accelerate with Settings <accelerate_with_settings>`
- :doc:`Memory and Disk Friendly Settings <memory_disk_friendly_settings>`
- :doc:`AdaQuant and AdaRound <accuracy_algorithms/ada>`
