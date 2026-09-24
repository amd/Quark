.. Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

How to Add a New ``contrib`` Algorithm
======================================

This page is the algorithm-specific companion to :doc:`intro_contrib`. It describes the mechanism
Quark provides for contributing a **quantization algorithm** for the PyTorch backend:
:py:class:`~quark.torch.algorithm.algorithm.QuarkAlgorithm`. Everything in :doc:`intro_contrib` —
the contribution principles, ownership, testing and legal requirements — still applies; this page
only covers the code you write.

.. note::

    The current scope at the time of writing this is to add a developer friendly and safe way to stage new algorithms to quark that can be optionally used. Below is a detailed general example of how one may add an algorithm to quark using this system. In the future, once algorithms are implemented using this system, a more concrete example will be provided in this document.

.. _why_quark_algorithm:

Why ``QuarkAlgorithm``
----------------------

Historically, adding a quantization algorithm to Quark meant editing three core files:

* ``quark/torch/quantization/config/config.py`` — a branch in ``_load_quant_algo_config_from_dict``
  so the algorithm's JSON is parsed into a config object.
* ``quark/torch/quantization/config/algo_configs.py`` — an ``ALGORITHM_CONFIG_MAPS`` entry holding
  the per-model-architecture default settings.
* ``quark/torch/algorithm/api.py`` — a ``PROCESSOR_MAP`` entry so the algorithm is dispatched.

Three files owned by the Quark Core Team, edited for every algorithm, by every author.
``QuarkAlgorithm`` is a wrapper that bundles exactly those three contributions into a single
object, so your algorithm is a file you own plus one registration call — and no core file changes.

.. _anatomy_of_an_algorithm:

Anatomy of an Algorithm
-----------------------

A ``QuarkAlgorithm`` has four fields:

.. list-table::
    :header-rows: 1
    :widths: 25 75

    * - Field
      - What it is
    * - ``name``
      - The algorithm name as it appears in ``{"name": ...}`` in a config file. Matched
        case-insensitively.
    * - ``algo_config``
      - The config dataclass your algorithm's JSON is deserialized into. Replaces a branch of
        ``_load_quant_algo_config_from_dict``.
    * - ``algo_processor``
      - The processor class that runs your algorithm. Replaces a ``PROCESSOR_MAP`` entry.
    * - ``algo_config_map``
      - Optional per-model-architecture default settings, keyed by model type (``"llama"``,
        ``"qwen2"``, ...). Read in place of an ``ALGORITHM_CONFIG_MAPS`` entry.

.. _algorithm_checklist:

Checklist
---------

* Config class subclasses ``AlgoConfig`` or ``PreQuantOptConfig``, with a default for every field.
* Processor subclasses ``BaseAlgoProcessor``.
* Per-model defaults, if any, build a fresh config object per model type.
* Algorithm passed to ``ALGORITHM_REGISTRY.register``, under a name no other algorithm uses, from a
  module that is imported before the algorithm is looked up.
* Tests, ``README.md``, ``CODEOWNERS`` entry and documentation as required by :doc:`intro_contrib`.

.. _using_your_algorithm:

Using Your Algorithm
--------------------

Once registered, your algorithm behaves exactly like a built-in algorithm, and Quark uses it in
the same way:

* ``{"name": "myalgo", ...}`` in a config file deserializes to ``MyAlgoConfig``.
* ``get_supported_algorithm_types()`` lists ``"myalgo"``, and
  ``get_algo_config("myalgo", "llama")`` returns your per-model default.
* ``get_processor("myalgo")`` dispatches to ``MyAlgoProcessor``.

.. code-block:: python

    from quark.torch.quantization.config.algo_configs import get_algo_config

    algo_config = get_algo_config("myalgo", "llama")

.. _algorithm_tests_and_docs:

Tests and Documentation
-----------------------

The requirements in :ref:`testing_contribution` and :ref:`documenting_contribution` apply
unchanged. For a contributed algorithm specifically, please cover at least:

* Your config class deserializes from a representative config dict, including any migration your
  ``build_config`` performs.
* ``get_processor`` returns your processor, and ``get_algo_config`` your per-model defaults, for
  your algorithm's name.
* Each entry of your ``algo_config_map`` is an independent object.

Quark's own tests live in ``test/test_for_torch/test_algo_registry.py`` and cover the registry
machinery itself; your tests belong in ``quark/contrib/your-component-name/tests/`` and should
cover your algorithm.
