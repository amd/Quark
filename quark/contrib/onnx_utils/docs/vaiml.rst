..
    Copyright (c) 2025 Advanced Micro Devices, Inc.

VAIML
=====

Use this command to generate operator config file can be used to generate NPU ops by VAIML.

By Default ops config will be generated for all the supported operators, namely

- BMM1
- BMM2
- SILU
- RoPE
- MUL
- ADD
- MATMULNBITS
- SILU
- RMSNORM


Usage
-----

Generating ops config for specific ops
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To generate ops config file for specific ops pass ``--ops <list of ops>``, for example

.. code-block:: shell

    onnx_utils vaiml <OGA model Folder> --plugin_name <plg_name> --ops gemm bmm1 bmm2


Generated ops config json file ``<plg_name>\<plg_name>_<ops>.json``

Generating ops config for higher context length
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default ops will be generated up to 2048 context length, to generate ops config for higher context length (for example 3072 or 4096):


.. code-block:: shell

    onnx_utils vaiml <OGA model Folder> --plugin_name <plg_name> --context_length 3072


Generating ops config for exact prompt length
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To generate ops config for exact prompt length (sometimes required for performance benchmarking):

.. code-block:: shell

    onnx_utils vaiml <OGA model Folder> --plugin_name <plg_name> --exact_length 673

See the :ref:`command-line arguments <cli:vaiml>` for the full list of arguments and options.

Example
-------

.. code-block:: shell

    onnx_utils vaiml <OGA model folder> --plugin_name <plg_name>

As a concrete example:

.. code-block:: shell

    onnx_utils vaiml  Llama-2-7b-hf-awq-g128-int4-asym-fp16-onnx-dml --plugin_name llama2_ops


This will generate ops config JSON file inside `llama2_ops\llama2_ops.json`
