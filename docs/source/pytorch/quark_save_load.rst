.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

Save & Load Quantized Models
============================

.. note::

    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of reference. When you  encounter the term "Quark" without the "AMD" prefix, it specifically refers to the AMD Quark quantizer unless otherwise stated. Please do not confuse it with other products or technologies that share the name "Quark."

Saving
------

- Save the network architecture or configurations and parameters of the quantized model.

- Support both eager and fx-graph model quantization.

- For eager mode quantization, the model's configurations are stored in JSON file, and parameters including weight, bias, scale, and zero_point are stored in safetensors file.

- For fx_graph mode quantization, the model's network architecture and parameters are stored in PTH file.


Example of Saving in Eager Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from quark.torch import save_params
   save_params(model, model_type=model_type, export_dir="./save_dir")

Example of Saving in FX-graph Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from quark.torch import save_params
   save_params(model,
               model_type=model_type,
               args=example_inputs,
               export_dir="./save_dir",
               quant_mode=QuantizationMode.fx_graph_mode)

Loading
-------

- Instantiates a quantized model from saved model files, which were generated using the previous saving function.
- Supports both eager and FX-Graph model quantization.
- Only supports weight-only and static quantization for now.

Example of Loading in Eager Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from quark.torch import load_params
   model = load_params(model, json_path=json_path, safetensors_path=safetensors_path)

Example of Loading in FX-graph Mode
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code:: python

   from quark.torch import load_params
   model = load_params(pth_path=model_file_path, quant_mode=QuantizationMode.fx_graph_mode)

Diffusers Models
----------------

Diffusion models use a different save/load path that integrates with
HuggingFace Diffusers.  After quantizing a submodule (for example
``pipe.unet`` or ``pipe.transformer``), export it with
``quark.torch.export_safetensors``, which writes a
``diffusion_pytorch_model.safetensors`` plus a ``config.json`` carrying the
quantization metadata.  The exported directory then reloads through
``DiffusionPipeline.from_pretrained`` / ``ModelMixin.from_pretrained`` with no
per-layer setup.

See :doc:`Using Quark-Quantized Diffusion Models with HuggingFace Diffusers <example_quark_torch_huggingface_diffusers>`
for the end-to-end quantize → export → reload workflow.
