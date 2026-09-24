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

FlyDSL / native inference issues (gfx950)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These apply to ``enable_native_inference`` with ``native_linear_mode`` set to
``flydsl_a8w4``, ``flydsl_svdquant``, or ``mxfp4``.

**Environment requirements**

.. list-table::
   :header-rows: 1
   :widths: 22 22 56

   * - Component
     - Requirement
     - Why
   * - GPU
     - gfx950 (MI350)
     - the A8W4 / MXFP4 kernels target this arch only
   * - ``flydsl``
     - **exactly 0.2.4**
     - the pin every recorded accuracy number was produced on; installed by
       ``tools/ci/setup_flydsl.sh``. 0.1.x lacks DSL APIs the vendored kernels use
       (``flydsl.expr.typing.T``, ``CompilationContext``); 0.3.x is a different API era
       and untested.
   * - ``aiter``
     - **>= v0.1.20**, installed from git
     - the A8W4 GEMM imports its MFMA epilogue and pipeline helpers from
       ``aiter.ops.flydsl.kernels``, and older aiter pinned its own FlyDSL kernels to
       the 0.1.x API; see "w4a4 silently missing" below. Not on PyPI under that name:
       ``pip install git+https://github.com/ROCm/aiter.git@v0.1.21.post1``
   * - ``$FLYDSL_REPO``
     - **not used**
     - the GEMM is vendored in ``quark/torch/kernel/flydsl/kernels/`` and its helpers
       come from ``aiter``; Quark never consults a FlyDSL repo checkout

**Known Issue**: ``native_linear_mode="mxfp4"`` (w4a4) silently does nothing, while
``flydsl_a8w4`` works.

``mxfp4`` runs on aiter's ASM ``gemm_a4w4``. That kernel has no FlyDSL dependency, but
``aiter/ops/flydsl/__init__`` sits in aiter's top-level import cascade
(``aiter/__init__`` -> ``gemm_op_a8w8`` -> ``ops/flydsl/__init__``) and older releases
**raised** there when the installed ``flydsl`` did not match their pin. That exception
aborts the entire aiter import and takes ``gemm_a4w4`` with it -- so an unrelated version
check disables w4a4, with no error that names the cause.

**Solution**: upgrade aiter to a release whose own minimum is ``flydsl >= 0.2.4``. That
gate was raised in **v0.1.18** (2026-07-19); **v0.1.20** is the earliest verified
end-to-end here. ROCm's aiter is not on PyPI under that name, so install from git:

.. code-block:: bash

   pip install git+https://github.com/ROCm/aiter.git@v0.1.21.post1
   python3 -c "from aiter import gemm_a4w4; print('gemm_a4w4 OK')"

Do **not** downgrade ``flydsl`` to satisfy an older aiter: that breaks Quark's own A8W4
kernels, which need 0.2.x.

.. warning::

   Do not pick an aiter tag by *date*. Tags are cut from a release branch rather than the
   main tip, so a newer tag can lack a newer commit -- and the tag dates are not even
   monotonic (``v0.1.16.post6`` postdates ``v0.1.17``). Verify by content -- the version
   gate is the only thing that moved, so it is the only check that discriminates:

   .. code-block:: bash

      git show <tag>:aiter/ops/flydsl/__init__.py | grep FLYDSL_VERSION   # want 0.2.4

   The releases still import aiter's FlyDSL kernels eagerly, so a future ``flydsl`` API
   break could take ``gemm_a4w4`` down as collateral again. That is why
   ``setup_flydsl.sh`` reports whether ``gemm_a4w4`` is importable rather than assuming
   it.

**Known Issue**: which A8W4 GEMM is Quark running?

Always the vendored snapshot in ``quark/torch/kernel/flydsl/kernels/``. ROCm/FlyDSL#957
(the fused SVD epilogue) has not landed upstream, so the snapshot is the *only*
implementation that supports ``flydsl_svdquant``, and Quark imports it directly rather
than searching a FlyDSL checkout. Only the GEMM is vendored -- its unmodified MFMA
epilogue and pipeline helpers come from ``aiter.ops.flydsl.kernels``. Do not try to
substitute upstream kernels by putting a checkout on ``PYTHONPATH`` -- mixing the pinned
snapshot with a different helper revision is what previously broke native inference (a
compile-time ``TypeError``, then a GPU memory fault).

**Known Issue**: fewer native linears than expected after ``enable_native_inference``.

The FlyDSL A8W4 GEMM requires ``in_features`` to be ``>= 256`` and a multiple of 256, and
``out_features`` to be ``>= 128`` and a multiple of 128. Layers that violate this are
**silently left on the eager path**, so a "native" model is often a mix.
``enable_native_inference`` returns the number converted -- count it rather than assuming.
(On Wan2.2-A14B this is 400/400 per expert, because the one offending layer, ``proj_out``
with ``out_features=64``, is excluded from quantization.)

**Known Issue**: any Quark import hangs for many minutes with no output.

Usually a stale C++ extension build lock, left behind when a process was killed mid-build
(for example with ``kill -9``). Distinguish it from slow compilation with ``time``: a
blocked process shows tiny ``user`` time against huge ``real`` time (for example 18 s of
CPU across 15 minutes).

**Solution**: with no Python process running, delete
``[cache folder path]/torch_extensions/py3xx_cpu/kernel_ext/lock``. The compiled ``.so``
next to it stays valid, so nothing is rebuilt.

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
