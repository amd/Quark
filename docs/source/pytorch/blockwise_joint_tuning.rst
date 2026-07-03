.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Blockwise Joint Tuning (Experimental)
=====================================

.. warning::

    Blockwise joint tuning is an **experimental** feature. It depends on
    ``quark.experimental`` (in particular
    ``ExperimentalLearnableQuantizedLinear`` in
    ``quark.experimental.torch.algorithm.blockwise_joint_tuning``),
    a temporary learnable-quantizer implementation that may change or be
    replaced by the official Quark ``QuantLinear`` in a future release. APIs and
    defaults described here are not yet stable.

AMD Quark supports **blockwise joint tuning**, a post-training optimization that
recovers accuracy lost during weight quantization by fine-tuning a quantized
model one decoder block at a time. For each block, the original (full-precision)
model provides a teacher signal, and the quantized block is trained to reproduce
that block's output. "Joint" means the optimization updates **both** the block
weights **and** the quantization parameters (``scale`` / ``zero_point``)
together, each with its own learning rate and schedule.

How blockwise joint tuning works
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quantization error accumulates layer by layer. Rather than retraining the whole
network end to end, blockwise joint tuning performs a localized
knowledge-distillation per decoder block, which keeps the memory footprint small
and the optimization well conditioned:

1. **Teacher forward.** The corresponding full-precision block runs on the
   block's input batch to produce reference (teacher) outputs.
2. **Make the block learnable.** Every ``nn.Linear`` in the quantized block is
   wrapped in an ``ExperimentalLearnableQuantizedLinear`` so that its
   ``scale`` / ``zero_point`` become trainable parameters.
3. **Joint training.** The quantized block is trained to minimize the MSE
   between its output and the teacher output, optimizing two independent
   parameter groups -- block weights and quantization parameters.
4. **Propagate.** The tuned block runs forward to produce the inputs for the
   next block, and the loop continues.

Two parameter groups are optimized with separate ``AdamW`` settings and separate
``CosineAnnealingLR`` schedules:

- **Weights** -- modules selected by ``trainable_modules``, learning rate
  ``weight_lr``, weight decay ``weight_decay``.
- **Quantization parameters** (``scale`` / ``zero_point``) -- modules selected
  by ``quant_trainable_modules``, learning rate ``qparam_lr``, weight decay
  ``qparam_weight_decay``.

To bound memory, only the block currently being tuned is moved to the GPU; all
other blocks stay on CPU. During training the active block is cast to FP32 and
trained under automatic mixed precision (AMP), then restored to its original
dtype so downstream evaluation stays dtype-consistent.

The algorithm is implemented by ``BlockwiseJointTuningProcessor``
(in ``quark.torch.algorithm.blockwise_joint_tuning``).

Relation to ``blockwise_tuning``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Quark also ships a simpler ``blockwise_tuning`` algorithm. The difference is what
gets optimized:

- ``blockwise_tuning`` tunes **only module weights** selected by
  ``trainable_modules`` (plus ``layernorm`` modules).
- ``blockwise_joint_tuning`` **jointly** tunes module weights **and**
  quantization parameters (``scale`` / ``zero_point``), with independent
  controls for each group.

Choose joint tuning when learnable quantization parameters are expected to help
(for example aggressive weight-only INT4 with group quantization); otherwise
plain ``blockwise_tuning`` is lighter weight.

Configuring blockwise joint tuning
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Blockwise joint tuning is configured with ``BlockwiseJointTuningConfig`` (from
``quark.torch.algorithm.config``) and applied with the ``blockwise_tuning_algo``
entry point, which -- unlike the regular ``algo_config`` algorithms -- takes
**both** the full-precision reference model and the quantized model.

.. list-table:: ``BlockwiseJointTuningConfig`` parameters
   :header-rows: 1
   :widths: 30 14 56

   * - Parameter
     - Default
     - Description
   * - ``name``
     - ``"blockwise_joint_tuning"``
     - Algorithm name; selects ``BlockwiseJointTuningProcessor`` in the
       processor registry. Leave at the default.
   * - ``epochs``
     - 5
     - Number of training epochs run per decoder block.
   * - ``weight_lr``
     - 1e-4
     - Learning rate for the block-weight parameter group. Set to ``0`` to skip
       weight tuning.
   * - ``qparam_lr``
     - 1e-4
     - Learning rate for the quantization-parameter (``scale`` / ``zero_point``)
       group. Set to ``0`` to skip quant-parameter tuning.
   * - ``weight_decay``
     - 0.0
     - Weight decay for the block-weight group.
   * - ``qparam_weight_decay``
     - 0.0
     - Weight decay for the quantization-parameter group.
   * - ``min_lr_factor``
     - 20.0
     - Cosine schedule floor: each group anneals to ``lr / min_lr_factor``.
   * - ``max_grad_norm``
     - 0.3
     - Gradient-norm clipping threshold.
   * - ``model_decoder_layers``
     - ``""``
     - Attribute path to the list of decoder blocks to tune
       (for example ``"model.layers"``).
   * - ``trainable_modules``
     - ``[]``
     - Substring filters selecting which modules' **weights** are trainable.
       Empty means match all modules in the block.
   * - ``quant_trainable_modules``
     - ``[]``
     - Substring filters selecting which modules' **quantization parameters**
       are trainable. Empty means match all.

Usage
~~~~~

Blockwise joint tuning is typically applied as a post-PTQ step: quantize the
model (for example weight-only INT4), then tune it against a fresh
full-precision reference model on a calibration/fine-tuning dataloader.

.. code-block:: python

    from quark.torch.algorithm.api import blockwise_tuning_algo
    from quark.torch.algorithm.config import BlockwiseJointTuningConfig

    # `model` is the already-quantized model.
    # `ref_model` is a freshly loaded full-precision copy (the teacher).
    ref_model.eval()

    blockwise_cfg = BlockwiseJointTuningConfig.from_dict({
        "name": "blockwise_joint_tuning",
        "epochs": 5,
        "weight_lr": 1e-4,
        "qparam_lr": 1e-4,
        "weight_decay": 0.0,
        "qparam_weight_decay": 0.0,
        "min_lr_factor": 20.0,
        "max_grad_norm": 0.3,
        "model_decoder_layers": "model.layers",
        "trainable_modules": [],
        "quant_trainable_modules": [],
    })

    is_accelerate = hasattr(model, "hf_device_map")
    model = blockwise_tuning_algo(
        ref_model,
        model,
        blockwise_cfg,
        is_accelerate,
        dataloader,
    )

For a complete, validated end-to-end flow (load model, PTQ weight-only
quantization, blockwise joint tuning, then export) see the EfficientQAT example
at ``examples/torch/language_modeling/llm_qat/efficientqat/main.py``.

Notes and limitations
~~~~~~~~~~~~~~~~~~~~~~~

- **Memory.** Two models are resident at once (quantized + full-precision
  reference). Per-block GPU offload keeps peak usage low, but the
  full-precision reference must be available throughout tuning.
- **Data.** A calibration/fine-tuning dataloader is required; the cached
  block-level activations scale with the number of calibration samples.
- **Placement.** Joint tuning is intended to run **after** weight quantization
  (for example PTQ weight-only INT4), not before.
