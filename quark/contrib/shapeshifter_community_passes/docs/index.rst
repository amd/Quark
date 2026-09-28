..  Copyright (C) 2025 - 2026 Advanced Micro Devices, Inc. All rights reserved.

Shapeshifter Community Passes
=============================

Community-contributed Shapeshifter passes extend the graph-transformation
capabilities of :doc:`Shapeshifter </quark_shapeshifter>` beyond the core passes.
They live in ``quark/contrib/shapeshifter_community_passes/`` and use the same
registration system, discovery mechanism, and implementation contract as core
passes — the only difference is where they live and who maintains them.

.. note::

   Community passes are a ``contrib`` area contribution. They are **not officially
   supported** by the Quark Core Team and are maintained by their original authors.
   See :doc:`/intro_contrib` for the ``contrib`` policies.

Authoring a Community Pass
--------------------------

Follow the same implementation requirements as core passes (subclass ``ONNXPass``
or ``PytorchPass``, apply the ``@register_pass`` decorator, and implement
``_default_config()`` and ``_run_for_config()``) — see
:doc:`Adding New Passes </quark_shapeshifter>` for the full contract and examples.
The differences for a community pass are:

* **Location:** place the pass file in
  ``quark/contrib/shapeshifter_community_passes/`` instead of
  ``quark/shapeshifter/passes/``. It is automatically discovered and registered to
  the same global registry as core passes.
* **Tests:** add your tests to
  ``quark/contrib/shapeshifter_community_passes/test/``.
* **Documentation:** document your pass on the matching page below, depending on
  whether it transforms ONNX or PyTorch models.

.. toctree::
   :maxdepth: 1

   community_onnx_passes
   community_torch_passes
