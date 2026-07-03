.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

Using Quark-Quantized Diffusion Models with HuggingFace Diffusers
=================================================================

Quark integrates with `HuggingFace Diffusers
<https://huggingface.co/docs/diffusers>`__ so that diffusion models
quantized and exported with Quark can be reloaded through the standard
``from_pretrained`` API, with no per-layer setup code on the user's
side.

This page covers the **reload** path that ships in the Quark wheel
today: take a Quark-quantized checkpoint (produced with
``quark.torch.export_safetensors``) and load it back into a diffusers
pipeline.  For producing the checkpoint in the first place, see
:doc:`Quantizing Diffusion Models with Quark <example_quark_torch_diffusers>`.

.. note::

   The integration mirrors the `Quark integration in HuggingFace
   Transformers <https://huggingface.co/docs/transformers/quantization/quark>`__:
   the quantizer is registered with diffusers' ``AUTO_QUANTIZER_MAPPING``
   under the ``quark`` method, and dispatch is automatic once the
   plugin is imported.

Prerequisites
-------------

.. code-block:: bash

   pip install amd-quark diffusers transformers accelerate

Importing ``quark.integrations.diffusers`` registers
``QuarkDiffusersQuantizer`` and ``QuarkQuantizationConfig`` with
diffusers' auto-quantizer mappings.  This import is required before
loading a Quark-quantized model:

.. code-block:: python

   import quark.integrations.diffusers  # noqa: F401  -- registers the quark quantizer

End-to-end: quantize, export, reload
------------------------------------

The full lifecycle has three stages.  Stages 1 and 2 are the offline
workflow from :doc:`example_quark_torch_diffusers`; stage 3 is the
reload covered here.

Stage 1 -- quantize a submodule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from diffusers import DiffusionPipeline
   from quark.torch import ModelQuantizer
   from quark.torch.quantization.config.config import Int8PerTensorSpec, QConfig, QLayerConfig
   from quark.torch.utils.diffusers import get_calib_dataloader

   pipe = DiffusionPipeline.from_pretrained(
       "stabilityai/stable-diffusion-xl-base-1.0",
       torch_dtype=torch.float16, variant="fp16",
   )
   pipe.to("cuda")

   prompts = [
       "A serene lake reflecting mountains at sunset",
       "A futuristic city with flying cars at night",
   ]
   dataloader = get_calib_dataloader(pipe, pipe.unet, prompts, n_steps=20, guidance_scale=8.0)

   weight_spec = Int8PerTensorSpec(
       observer_method="min_max", symmetric=True, scale_type="float",
       round_method="half_even", is_dynamic=False,
   ).to_quantization_spec()
   qconfig = QConfig(global_quant_config=QLayerConfig(weight=weight_spec))

   pipe.unet = ModelQuantizer(qconfig).quantize_model(pipe.unet, dataloader)

Stage 2 -- export the quantized submodule
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``export_safetensors`` detects a diffusers ``ModelMixin`` and writes a
``diffusion_pytorch_model.safetensors`` plus a ``config.json`` that
carries a ``quantization_config`` block describing the Quark
``QConfig``.

.. code-block:: python

   from quark.torch import export_safetensors

   export_safetensors(pipe.unet, "sdxl-quark-int8/unet")

Stage 3 -- reload through ``from_pretrained``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Because ``config.json`` carries ``quant_method = "quark"``, the
diffusers loader dispatches to ``QuarkDiffusersQuantizer`` automatically.
It replaces the relevant layers and loads the quantized state dict, then
freezes the model so it is inference-ready.

.. code-block:: python

   import torch
   import quark.integrations.diffusers  # noqa: F401  -- registers the quark quantizer
   from diffusers import UNet2DConditionModel, DiffusionPipeline

   unet = UNet2DConditionModel.from_pretrained("sdxl-quark-int8/unet", torch_dtype=torch.float16)

   pipe = DiffusionPipeline.from_pretrained(
       "stabilityai/stable-diffusion-xl-base-1.0",
       unet=unet,
       torch_dtype=torch.float16, variant="fp16",
   ).to("cuda")

   image = pipe("A cat on a windowsill", num_inference_steps=30, guidance_scale=8.0).images[0]
   image.save("sdxl_int8_reloaded.png")

If a model on the Hub already carries a ``quantization_config`` block in
its ``config.json`` (for example, a pre-quantized pipeline published by
AMD), no extra setup beyond the plugin import is required -- the loader
sees ``quant_method = "quark"`` and instantiates
``QuarkDiffusersQuantizer`` for you.

How dispatch works
------------------

``quark/integrations/diffusers/__init__.py`` registers two entries when
imported:

.. code-block:: python

   from diffusers.quantizers.auto import (
       AUTO_QUANTIZATION_CONFIG_MAPPING,
       AUTO_QUANTIZER_MAPPING,
   )

   AUTO_QUANTIZER_MAPPING["quark"] = QuarkDiffusersQuantizer
   AUTO_QUANTIZATION_CONFIG_MAPPING["quark"] = QuarkQuantizationConfig

At load time, ``from_pretrained`` reads the ``quant_method`` field from
the checkpoint's ``quantization_config``, looks it up in these mappings,
and drives the load through ``QuarkDiffusersQuantizer``:

* ``_process_model_before_weight_loading`` rebuilds the ``QConfig`` from
  the serialized dict and applies Quark's module transformation so the
  state dict loads into the correct quantized modules.
* ``_process_model_after_weight_loading`` calls
  ``ModelQuantizer.freeze(model, quantize=False)`` so the model is
  inference-ready and compatible with ``torch.compile``.

What is supported today
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Capability
     - Status
   * - Reload a Quark-quantized checkpoint via ``from_pretrained``
     - Supported
   * - Weight-only configurations (INT8, INT4, MXFP4, ...)
     - Supported and covered by the integration test suite
   * - Activation-quantized configurations (w8a8, SVDQuant, FP8 with
       calibrated activations)
     - The activation scales are captured during offline calibration and
       written into the checkpoint; reload support depends on the
       configuration.  Validate the specific config end-to-end before
       relying on it in production.

Related topics
--------------

* :doc:`Quantizing Diffusion Models with Quark <example_quark_torch_diffusers>`
  -- the offline calibration, quantization, and export workflow.
* :doc:`Exporting Quantized Models <../pytorch/export/quark_export>`
  -- details of ``export_safetensors`` and the on-disk format.
