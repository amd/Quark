..  Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.

Brevitas API
========

This is the Quark API for Brevitas which is a PyTorch library for neural network quantization, with support for both post-training quantization (PTQ) and quantization-aware training (QAT).

It is currently experimental and under active development, it should not be considered stable and may be subject to change.

For more information specific to Brevitas itself, please refer to its `documentation <https://xilinx.github.io/brevitas/>`__.

Usage
-----

To use this API, it is as simple as defining some configuration parameters, creating a ModelQuantizer and passing your unquantized model through it by calling quantize_model.

For example:
.. code:: python
    import quark.torch.extensions.brevitas.api as brevitas_api
    import quark.torch.extensions.brevitas.config as brevitas_config

    model = ... # A pytorch model created manually or downloaded from huggingface

    global_config = brevitas_config.QLayerConfig() # This config will be applied to the whole model

    config = brevitas_config.Config(global_quant_config=global_config)
    quantizer = brevitas_api.ModelQuantizer(config)

    quantized_model = quantizer.quantize_model(model)

If you want to export the quantized model, it's even simpler:
.. code:: python
    exporter = brevitas_api.ModelExporter("test.onnx")
    exporter.export_onnx_model(quantized_model, torch.ones(1, 3, 32, 32))

To start, you just need to know that you must specify a global_quant_config which controls the quantization settings that will be applied to the whole model.
This is a QLayerConfig object and it has parameters for controlling input/output, bias and weight quantization. By default these will be set to None.

To define the quantization settings for weights for example, you can do this:
.. code:: python
    global_config = brevitas_config.QLayerConfig(weight=brevitas_config.QTensorConfig())
    config = brevitas_config.Config(global_quant_config=global_config)

The default values for QTensorConfig should be reasonable but please refer to QTensorConfig in config.py to get details on the different parameters you can set.

.. raw:: html

   <!--
   ## License
   Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved. SPDX-License-Identifier: MIT
   -->
