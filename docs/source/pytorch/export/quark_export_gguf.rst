.. Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

GGUF Exporting
==============

Currently, only support asymmetric int4 per_group weight-only
quantization, and the group_size must be 32.The models supported include
Llama2-7b, Llama2-13b, Llama2-70b, and Llama3-8b.

Example of GGUF Exporting (for already quantized model)
-------------------------------------------------------

.. code:: python

   from quark.torch import export_gguf

   model_dir = "meta-llama/Llama-2-7b-chat-hf"
   export_gguf(quantized_model, output_dir="./output_dir", model_type="llama", tokenizer_path=model_dir)

After running the code above successfully, there will be a ``.gguf``
file under output_dir, ``./output_dir/llama.gguf`` for example.

.. toctree::
   :hidden:
   :maxdepth: 1

   gguf_llamacpp.rst
