.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Hugging Face format (safetensors format)
========================================

Hugging Face format (safetensors format) is an optional exporting format for Quark, and the file list of this exporting format is the same as the file list of the original Hugging Face model, with quantization information added to these files. Taking the llama2-7b model as an example, the exported file list and added information are as below:

+------------------------------+---------------------------------------------------------------------------------------------------------------------+
| File name                    | Additional Quantization Information                                                                                 |
+==============================+=====================================================================================================================+
| config.json                  | Original configuration, with quantization configuration added in a ``"quantization_config"`` key                    |
+------------------------------+---------------------------------------------------------------------------------------------------------------------+
| generation_config.json       | \-                                                                                                                  |
+------------------------------+---------------------------------------------------------------------------------------------------------------------+
| model*.safetensors           | Quantized checkpoint (weights, scaling factors, zero points)                                                        |
+------------------------------+---------------------------------------------------------------------------------------------------------------------+
| model.safetensors.index.json | Mapping of weights names to safetensors files, in case the model weights are sharded into multiple files (optional) |
+------------------------------+---------------------------------------------------------------------------------------------------------------------+
| special_tokens_map.json      | \-                                                                                                                  |
+------------------------------+---------------------------------------------------------------------------------------------------------------------+
| tokenizer_config.json        | \-                                                                                                                  |
+------------------------------+---------------------------------------------------------------------------------------------------------------------+
| tokenizer.json               | \-                                                                                                                  |
+------------------------------+---------------------------------------------------------------------------------------------------------------------+

Exporting to Hugging Face format (safetensors format)
-----------------------------------------------------

Here is an example of how to export to Hugging Face format (safetensors format) a Quark model using :py:func:`~quark.torch.export.api.export_safetensors`:

.. code-block:: python

   from quark.torch import ModelQuantizer, export_safetensors
   from quark.torch.quantization.config.config import Int8PerTensorSpec, QuantizationConfig, Config

   from transformers import AutoModelForCausalLM

   quant_spec = Int8PerTensorSpec(
      observer_method="min_max",
      symmetric=True,
      scale_type="float",
      round_method="half_even",
      is_dynamic=False
   ).to_quantization_spec()

   global_quant_config = QuantizationConfig(weight=quant_spec)
   quant_config = Config(global_quant_config=global_quant_config)

   model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m")

   quantizer = ModelQuantizer(quant_config)
   quantized_model = quantizer.quantize_model(model, dataloader=None)
   quantized_model = quantizer.freeze(quantized_model)

   export_safetensors(
      model=quantized_model,
      output_dir="./opt-125m-quantized"
   )

By default, :py:func:`~quark.torch.export.api.export_safetensors` exports models with |save_pretrained|_ using a Quark-specific format for the checkpoint and ``"quantization_config"`` key in the ``config.json`` file. This format may not directly be usable by some downstream libraries (AutoAWQ, vLLM).

.. |save_pretrained| replace:: ``model.save_pretrained()``
.. _save_pretrained: https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.save_pretrained

Until downstream libraries support Quark quantized models, one may export models so that the weight checkpoint and ``config.json`` file targets a specific downstream libraries, using ``custom_mode="awq"`` or ``custom_mode="fp8"``. Example:

.. code-block:: python

   export_safetensors(
      model=quantized_model,
      output_dir="./opt-125m-quantized",
      custom_mode="awq"
   )

.. code-block:: python

   export_safetensors(
      model=quantized_model,
      output_dir="./opt-125m-quantized",
      custom_mode="fp8"
   )

For example, ``custom_mode="awq"`` would e.g. use ``qzeros`` instead of ``weight_zero_point``, ``qweight`` instead of ``weight`` in the checkpoint. Moreover, the ``quantization_config`` in the ``config.json`` file is custom, and the full Quark :py:class:`.quark.torch.quantization.config.config.Config` is not serialized.


In the ``config.json``, such an export results in using ``"quant_method": "awq"``, that can e.g. be loaded through `AutoAWQ <https://github.com/casper-hansen/AutoAWQ>`__ in `Transformers library <https://huggingface.co/docs/transformers/main/en/quantization/awq#awq>`__.

Loading quantized models saved in Hugging Face format (safetensors format)
--------------------------------------------------------------------------

Quark provides the importing function for HF format export files. In other words, these files can be reloaded into Quark. After reloading, the weights of the quantized operators in the model are stored in the real_quantized format.

Currently, this importing function supports weight-only, static, and dynamic quantization for FP8, INT8/UINT8, FP4, INT4/UINT, AWQ, GPTQ and Qronos.

Here is an example of how to load a serialized quantized model from a folder containing the model (as ``*.safetensors``) and its artifacts (``config.json``, etc.), using :py:func:`~quark.torch.export.api.import_model_from_safetensors`:

.. code-block:: python

   from quark.torch import import_model_from_safetensors
   from transformers import AutoConfig, AutoModelForCausalLM
   import torch

   # We only need the backbone/architecture of the original model,
   # not its weights, as weights are loaded from the quantized checkpoint.
   config = AutoConfig.from_pretrained("facebook/opt-125m")
   with torch.device("meta"):
      original_model = AutoModelForCausalLM.from_config(config)

   quantized_model = import_model_from_safetensors(original_model, model_dir="./opt-125m-quantized")
