.. Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.

Troubleshooting
===============

.. note::

    In this documentation, **AMD Quark** is sometimes referred to simply as **"Quark"** for ease of reference. When you  encounter the term "Quark" without the "AMD" prefix, it specifically refers to the AMD Quark quantizer unless otherwise stated. Please do not confuse it with other products or technologies that share the name "Quark."

AMD Quark for PyTorch
---------------------

Environment Issues
~~~~~~~~~~~~~~~~~~

**Known Issue**: Windows CPU mode does not support fp16.

Because of an existing PyTorch `issue <https://github.com/pytorch/pytorch/issues/52291>`__\ , Windows CPU mode cannot perfectly support fp16.

C++ Compilation Issues
~~~~~~~~~~~~~~~~~~~~~~

**Known Issue**: Stuck in the compilation phase for a long time (over ten minutes), and terminal shows:

.. code-block:: bash

   [QUARK-INFO]: Configuration checking start.
   [QUARK-INFO]: C++ kernel build directory [cache folder path]/torch_extensions/py39...

**Solution**:

Delete the cache folder ``[cache folder path]/torch_extensions`` and run AMD Quark again.

vLLM Integration Issues
~~~~~~~~~~~~~~~~~~~~~~~

**Known Issue**: vLLM fails with ``AttributeError: 'CustomOp' has no attribute 'op_registry'``.

**Typical Error**:

.. code-block:: text

   AttributeError: 'CustomOp' has no attribute 'op_registry'

**Root Cause**:

- Some vLLM builds check for (or rely on) the ``amd-quark`` Python package for **emulation** MXFP4 kernels.
- If you intend to use **native** MXFP4 kernels, ``amd-quark`` is **not required**.

**Solution**:

- Install ``amd-quark`` if your vLLM runtime requires emulation kernels.
- Otherwise, configure vLLM to use its native MXFP4 kernel path (when available) so that ``amd-quark`` is not needed.

**Known Issue**: vLLM weight loading fails with shape mismatch for some models/checkpoints.

**Typical Error**:

.. code-block:: text

   AssertionError: param_data.shape == loaded_weight.shape

**Root Cause**:

- The checkpoint stores packed weights (e.g., packed QKV), but the corresponding vLLM model implementation does not provide the required mapping.

**Solution**:

- Define ``packed_modules_mapping`` in the corresponding vLLM model executor file.
  For example, to support Qwen3, the following mapping is required in `vllm/vllm/model_executor/models/qwen3.py <https://github.com/vllm-project/vllm/blob/v0.15.1/vllm/model_executor/models/qwen3.py#L254>`__:

  .. code-block:: python

      class Qwen3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP, SupportsEagle3):
          packed_modules_mapping = {
              "qkv_proj": ["q_proj", "k_proj", "v_proj"],
              "gate_up_proj": ["gate_proj", "up_proj"],
          }

Quantization Performance Issues
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Known Issue**: Quantization can be extremely slow when running on CPU for very large LLM checkpoints.

**Solution**:

- Use a shard-by-shard (file-by-file) loading/quantization workflow to reduce peak memory and improve throughput.
- See :doc:`Language Model PTQ <example_quark_torch_llm_ptq>` for the recommended workflow and scripts in this repository.
